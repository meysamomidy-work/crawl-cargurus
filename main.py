import csv
import logging
import os
import random
import sys
import time
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from bs4 import BeautifulSoup
from shapely.geometry import Point

# --- Logging (stdout → docker logs) ---
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
_fmt = logging.Formatter(
    "%(asctime)s | %(levelname)-5s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
_root = logging.getLogger()
_root.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
_sh = logging.StreamHandler()
_sh.setFormatter(_fmt)
_root.addHandler(_sh)
_log_dir = os.environ.get("CRAWL_LOG_DIR", "logs")
try:
    os.makedirs(_log_dir, exist_ok=True)
    _fh = logging.FileHandler(os.path.join(_log_dir, "crawl.log"), encoding="utf-8")
    _fh.setFormatter(_fmt)
    _root.addHandler(_fh)
except OSError as e:
    logging.getLogger("crawl.bootstrap").warning("File logging disabled: %s", e)
log = logging.getLogger("crawl")

# Natural Earth shapefile (Dockerfile downloads it); local fallback path
SHAPEFILE_PATH = os.environ.get(
    "SHAPEFILE_PATH",
    "./us-state-boundaries/us-state-boundaries.shp",
)

# CarGurus (and similar CDNs) often return 406 with an empty body if Accept / fetch
# metadata look like a bare script (e.g. only User-Agent). Mirror a real navigation.
CHROME_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

DEFAULT_HEADERS = {
    "User-Agent": CHROME_UA,
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,"
        "image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    # omit br/zstd unless brotli/zstd extras are installed; gzip/deflate is always safe
    "Accept-Encoding": "gzip, deflate",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-User": "?1",
    "sec-ch-ua": '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
}

CARGURUS_ORIGIN = "https://www.cargurus.com"

# HTTP 4xx/5xx that often clear after backoff + fresh cookies (WAF / edge on GCP).
RETRY_STATUSES = frozenset({403, 406, 429, 502, 503, 504})


def _env_bool(key: str, default: bool) -> bool:
    v = os.environ.get(key)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def nav_headers(*, same_site: bool, referer: str | None = None) -> dict[str, str]:
    h = {**DEFAULT_HEADERS, "Sec-Fetch-Site": "same-origin" if same_site else "none"}
    if referer:
        h["Referer"] = referer
    return h

# Heuristics: not proof of blocking, but worth watching in docker logs
BLOCK_HINTS = (
    "captcha",
    "access denied",
    "forbidden",
    "unusual traffic",
    "automated access",
    "verify you are human",
    "cloudflare",
    "checking your browser",
    "rate limit",
    "too many requests",
    "please enable javascript",
    "datacenter",
    "bot detection",
    "pardon our interruption",
)

_stats: dict[str, int] = {
    "http_requests": 0,
    "http_2xx": 0,
    "http_403": 0,
    "http_429": 0,
    "http_other_4xx": 0,
    "http_5xx": 0,
    "http_errors": 0,
    "block_hint_hits": 0,
    "tiny_html": 0,
    "parse_exceptions": 0,
    "dealer_pages_ok": 0,
    "dealer_pages_failed": 0,
    "http_retry_rounds": 0,
}


def _html_block_hints(html: str) -> list[str]:
    if not html:
        return ["empty_body"]
    low = html.lower()
    return [h for h in BLOCK_HINTS if h in low]


def _record_http_status(code: int) -> None:
    _stats["http_requests"] += 1
    if 200 <= code < 300:
        _stats["http_2xx"] += 1
    elif code == 403:
        _stats["http_403"] += 1
    elif code == 429:
        _stats["http_429"] += 1
    elif 400 <= code < 500:
        _stats["http_other_4xx"] += 1
    elif code >= 500:
        _stats["http_5xx"] += 1


def _url_for_log(url: str, *, head: int = 72, tail: int = 56) -> str:
    """CarGurus URLs are long; truncating the start hides &page=N — keep both ends."""
    if len(url) <= head + tail + 3:
        return url
    return f"{url[:head]}...{url[-tail:]}"


def _response_body_len(response: Any) -> int:
    text = getattr(response, "text", None)
    if text:
        return len(text)
    content = getattr(response, "content", None)
    return len(content) if content else 0


def _response_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if text:
        return text
    content = getattr(response, "content", None)
    return (content or b"").decode("utf-8", errors="replace")


def inspect_http_response(response: Any, label: str, url_short: str) -> None:
    """Log status, size, and soft signals of blocking (for docker logs)."""
    code = response.status_code
    _record_http_status(code)
    text = _response_text(response)
    n = len(text)
    hints = _html_block_hints(text)
    if hints:
        _stats["block_hint_hits"] += 1
    if n < 1500 and code == 200:
        _stats["tiny_html"] += 1

    if code == 200 and not hints and n >= 1500:
        log.info(
            "%s | OK | status=%s | bytes=%s | url=%s",
            label,
            code,
            n,
            _url_for_log(url_short),
        )
    elif code == 200 and hints:
        log.warning(
            "%s | status=200 but HTML hints: %s | bytes=%s | url=%s",
            label,
            hints[:5],
            n,
            _url_for_log(url_short),
        )
    elif code == 200 and n < 1500:
        log.warning(
            "%s | status=200 but very small body (possible block/challenge) | bytes=%s | url=%s",
            label,
            n,
            _url_for_log(url_short),
        )
    elif code == 406:
        log.error(
            "%s | HTTP 406 Not Acceptable (often bad/missing Accept or bot filter) | bytes=%s | url=%s",
            label,
            n,
            _url_for_log(url_short),
        )
    elif code in (403, 429):
        log.error(
            "%s | likely blocked or throttled | status=%s | bytes=%s | url=%s",
            label,
            code,
            n,
            _url_for_log(url_short),
        )
    else:
        log.warning(
            "%s | status=%s | bytes=%s | hints=%s | url=%s",
            label,
            code,
            n,
            hints[:3] if hints else [],
            _url_for_log(url_short),
        )


def log_stats_snapshot(reason: str) -> None:
    log.info(
        "STATS [%s] requests=%s 2xx=%s 403=%s 429=%s other_4xx=%s 5xx=%s "
        "http_errors=%s retry_rounds=%s block_hint_pages=%s tiny_html=%s parse_errors=%s "
        "dealer_ok=%s dealer_fail=%s",
        reason,
        _stats["http_requests"],
        _stats["http_2xx"],
        _stats["http_403"],
        _stats["http_429"],
        _stats["http_other_4xx"],
        _stats["http_5xx"],
        _stats["http_errors"],
        _stats["http_retry_rounds"],
        _stats["block_hint_hits"],
        _stats["tiny_html"],
        _stats["parse_exceptions"],
        _stats["dealer_pages_ok"],
        _stats["dealer_pages_failed"],
    )


class CargurusClient:
    """
    One session + cookie jar for the whole crawl (Compute Engine / GCP friendly).
    curl_cffi Chrome TLS/JA3 impersonation is the default — required for most GCP egress.
    """

    def __init__(self) -> None:
        self._kind: str
        self.session: Any
        self._impersonate: str | None = None
        self._warmed = False
        self._kind, self.session, self._impersonate = self._open_session()

    def _open_session(self) -> tuple[str, Any, str | None]:
        use_cffi = _env_bool("CRAWL_USE_CURL_CFFI", True)
        require_cffi = _env_bool("CRAWL_REQUIRE_CURL_CFFI", True)
        if use_cffi:
            try:
                from curl_cffi import requests as cfr

                impersonate = os.environ.get("CRAWL_IMPERSONATE", "chrome131")
                try:
                    s = cfr.Session(impersonate=impersonate)
                    used_imp = impersonate
                except Exception as ex2:
                    log.warning(
                        "curl_cffi impersonate=%r failed (%s); trying chrome120",
                        impersonate,
                        ex2,
                    )
                    s = cfr.Session(impersonate="chrome120")
                    used_imp = "chrome120"
                log.info("HTTP client: curl_cffi impersonate=%s (required on GCP)", used_imp)
                return "cffi", s, used_imp
            except ImportError as ex:
                if require_cffi:
                    log.error(
                        "curl_cffi is required for CarGurus on datacenter IPs (HTTP 406 otherwise). "
                        "Install: pip install -r requirements.txt  "
                        "Or set CRAWL_REQUIRE_CURL_CFFI=0 to allow plain requests (usually fails on GCP)."
                    )
                    raise SystemExit(1) from ex
                log.warning("curl_cffi unavailable (%s); using std requests", ex)
            except Exception as ex:
                if require_cffi:
                    log.error("curl_cffi session failed: %s", ex)
                    raise SystemExit(1) from ex
                log.warning("curl_cffi unavailable (%s); using std requests", ex)
        adapter = HTTPAdapter(pool_connections=32, pool_maxsize=32, max_retries=0)
        s = requests.Session()
        s.mount("https://", adapter)
        s.mount("http://", adapter)
        log.warning(
            "HTTP client: requests+urllib3 — likely HTTP 406 on GCP; "
            "pip install curl_cffi and keep CRAWL_USE_CURL_CFFI=1"
        )
        return "requests", s, None

    def _get(
        self, url: str, headers: dict[str, str], timeout: float
    ) -> Any:
        kwargs: dict[str, Any] = {"headers": headers, "timeout": timeout, "verify": True}
        if self._kind == "cffi" and self._impersonate:
            kwargs["impersonate"] = self._impersonate
        return self.session.get(url, **kwargs)

    def throttle(self) -> None:
        lo = float(os.environ.get("CRAWL_MIN_DELAY_SEC", "0.55"))
        hi = float(os.environ.get("CRAWL_MAX_DELAY_SEC", "2.4"))
        if hi < lo:
            hi = lo
        time.sleep(random.uniform(lo, hi))

    def warmup(self, force: bool = False) -> bool:
        if self._warmed and not force:
            return True
        self._warmed = False
        tmo = float(os.environ.get("CRAWL_WARMUP_TIMEOUT_SEC", "28"))
        home = f"{CARGURUS_ORIGIN}/"
        self.throttle()
        r1 = self._get(home, nav_headers(same_site=False, referer=home), tmo)
        inspect_http_response(r1, "warmup_home", home)
        ok_home = r1.status_code == 200 and _response_body_len(r1) > 5_000
        cars = f"{CARGURUS_ORIGIN}/Cars/"
        self.throttle()
        r2 = self._get(cars, nav_headers(same_site=True, referer=home), tmo)
        inspect_http_response(r2, "warmup_cars", cars)
        ok_cars = r2.status_code == 200 and _response_body_len(r2) > 1_000
        self._warmed = ok_home and ok_cars
        if self._warmed:
            log.info("warmup | cookie jar primed for CarGurus paths")
        else:
            log.error(
                "warmup | failed (home ok=%s status=%s bytes=%s | cars ok=%s status=%s bytes=%s). "
                "On GCP: pip install curl_cffi, then python check_cargurus_access.py",
                ok_home,
                r1.status_code,
                _response_body_len(r1),
                ok_cars,
                r2.status_code,
                _response_body_len(r2),
            )
        return self._warmed

    def fetch(
        self,
        url: str,
        *,
        label: str,
        headers_fn: Any,
        timeout: float,
    ) -> Any | None:
        max_r = max(1, int(os.environ.get("CRAWL_HTTP_RETRIES", "6")))
        backoff = float(os.environ.get("CRAWL_RETRY_BACKOFF_SEC", "3.5"))
        last: Any | None = None
        for attempt in range(max_r):
            if attempt > 0:
                _stats["http_retry_rounds"] += 1
                log.warning("%s | retry round %s/%s", label, attempt + 1, max_r)
                self.warmup(force=True)
            elif not self._warmed:
                self.warmup(force=False)
            self.throttle()
            try:
                last = self._get(url, headers_fn(), timeout)
            except Exception as e:
                _stats["http_errors"] += 1
                log.warning("%s | attempt %s transport error: %s", label, attempt + 1, e)
                last = None
                time.sleep(backoff * (1.6**attempt) * random.uniform(0.85, 1.15))
                self._warmed = False
                continue
            inspect_http_response(last, label, url)
            if last.status_code == 200:
                return last
            if last.status_code not in RETRY_STATUSES:
                return last
            sleep_s = backoff * (1.45**attempt) * random.uniform(0.85, 1.15)
            log.warning(
                "%s | status=%s — sleeping %.1fs before cookie refresh + retry",
                label,
                last.status_code,
                sleep_s,
            )
            time.sleep(sleep_s)
            self._warmed = False
        return last


_cc_singleton: CargurusClient | None = None


def _cc() -> CargurusClient:
    global _cc_singleton
    if _cc_singleton is None:
        _cc_singleton = CargurusClient()
    return _cc_singleton


def get_link(link: str) -> str | None:
    try:
        url = f"{CARGURUS_ORIGIN}/" + link.lstrip("/")
        tmo = float(os.environ.get("CRAWL_DEALER_TIMEOUT_SEC", "45"))
        response = _cc().fetch(
            url,
            label="dealer_detail",
            headers_fn=lambda: nav_headers(same_site=True, referer=f"{CARGURUS_ORIGIN}/"),
            timeout=tmo,
        )
        if response is None:
            _stats["dealer_pages_failed"] += 1
            return None
        if response.status_code != 200:
            _stats["dealer_pages_failed"] += 1
            return None
        body = _response_text(response)
        hints = _html_block_hints(body)
        if hints:
            log.warning("dealer_detail | block-like hints on 200: %s", hints[:5])
        _stats["dealer_pages_ok"] += 1
        return body
    except Exception:
        _stats["http_errors"] += 1
        _stats["dealer_pages_failed"] += 1
        log.exception("dealer_detail | exception | link=%s", link[:200])
        return None


def get(url: str) -> Any | Exception:
    try:
        tmo = float(os.environ.get("CRAWL_LISTING_TIMEOUT_SEC", "35"))
        html_doc = _cc().fetch(
            url,
            label="listing_search",
            headers_fn=lambda: nav_headers(same_site=False, referer=f"{CARGURUS_ORIGIN}/"),
            timeout=tmo,
        )
        if html_doc is None:
            return RuntimeError("listing_search: no response after retries")
        return html_doc
    except Exception as e:
        _stats["http_errors"] += 1
        log.exception("listing_search | exception | url=%s", url[:200])
        return e


def page(response: Any | Exception) -> Any:
    if isinstance(response, Exception):
        log.error("listing_parse | no response object: %s", response)
        return response
    if response.status_code != 200:
        log.error(
            "listing_parse | skip (non-200) | status=%s | bytes=%s",
            response.status_code,
            _response_body_len(response),
        )
        return ValueError(f"listing HTTP {response.status_code}")
    text = _response_text(response)
    try:
        soup = BeautifulSoup(text, "html.parser")
        page_el = soup.select(".info > strong:nth-child(2)")
        name_el = soup.select(".header5")
        name = None
        for wraper in name_el:
            name = wraper.text[29:-4]
        page_text = None
        for wraper in page_el:
            page_text = wraper.text
        if name is None or page_text is None:
            hints = _html_block_hints(text)
            log.error(
                "listing_parse | missing expected DOM (.header5 / .info strong) | hints=%s",
                hints[:5],
            )
            return ValueError("listing page structure not found (blocked or layout changed)")
        log.info("listing_parse | OK | dealer_name_snippet=%s | total_pages_raw=%s", name[:80], page_text)
        return (name, page_text)
    except Exception as e:
        log.exception("listing_parse | exception")
        return e


def loader(path: str | tuple) -> Any:
    if isinstance(path, tuple):
        path = path[0]
    data = pd.read_excel(path, engine="openpyxl")
    links = data.iloc[:, -1]
    log.info("loader | rows=%s from %s", len(links), path)
    return links


def luncher(urls: str, thread_name: str, number: int) -> tuple:
    response = get(urls)
    info = page(response)
    path, name = link_crawler(info, urls, thread_name, number)
    return path, name


def runner(url: str, thread_name: str, number: int) -> tuple:
    path_thread, name = luncher(url, thread_name, number)
    links = loader(path_thread)
    return links, name


def link_crawler(info: Any, url: str, thread_name: str, number: int) -> tuple:
    dealer_list: list = []
    failed: list = []
    if isinstance(info, Exception):
        log.error("link_crawler | bad listing info: %s", info)
        raise info
    page_raw = info[1]
    if page_raw is None or (isinstance(page_raw, str) and not str(page_raw).strip()):
        log.error("link_crawler | empty page count")
        raise ValueError("empty listing page info")
    page_clean = str(page_raw).replace(",", "")
    page_count = int(page_clean) // 10 + 1
    name = info[0]
    name = str(name).replace(",", "-").replace(" ", "").split()[0]
    log.info("link_crawler | grid_pages=%s name=%s", page_count, name)

    client = _cc()
    grid_tmo = float(os.environ.get("CRAWL_GRID_TIMEOUT_SEC", "28"))
    for start in range(page_count):
        link = url + str(start)
        referer = f"{CARGURUS_ORIGIN}/" if start == 0 else url + str(start - 1)
        response = client.fetch(
            link,
            label="listing_grid",
            headers_fn=lambda r=referer: nav_headers(same_site=True, referer=r),
            timeout=grid_tmo,
        )
        if response is None or response.status_code != 200:
            log.error(
                "listing_grid | skip page=%s status=%s",
                start,
                getattr(response, "status_code", None),
            )
            continue
        soup = BeautifulSoup(_response_text(response), "html.parser")
        dealer_name = soup.find_all("div", attrs={"class": "details"})
        dealer_address_span = soup.find_all("div", attrs={"class": "address"})
        dealer_address = []
        for address_element in dealer_address_span:
            if dealer_address_span:
                clean_address = address_element.get_text(strip=True)
                dealer_address.append(clean_address)
            else:
                dealer_address.append("does not exist")

        inv_anchors = soup.find_all("a", attrs={"class": "viewInventory"})
        href_list = [a.get("href") for a in inv_anchors if a.get("href")]
        for (inv_href, j, k) in zip(href_list, dealer_address, dealer_name):
            j = " ".join(j.split())
            dealer_list.append([k.strong.text, j, inv_href])
        with open("failed.csv", "w", newline="") as f:
            writer = csv.writer(f)
            for item in failed:
                writer.writerow([item])

    df = pd.DataFrame(dealer_list)
    if thread_name == "one":
        thread_name = f"primary/{name}.xlsx"
        writer = pd.ExcelWriter(thread_name, engine="xlsxwriter")
        df.to_excel(writer, index=False)
        writer.close()
        log.info("link_crawler | wrote %s rows → %s", len(df), thread_name)
        return thread_name, name
    path = f"primary/{name}_{number}.xlsx"
    writer = pd.ExcelWriter(path, engine="xlsxwriter")
    df.to_excel(writer, index=False)
    writer.close()
    log.info("link_crawler | wrote %s rows → %s", len(df), path)
    return path, name


def parser(html_doc: str | None) -> tuple:
    if not html_doc:
        log.warning("parser | empty HTML (dealer page failed or blocked)")
        return (
            "bad url",
            "Not Specified",
            "Not Specified",
            "0",
            0,
            0,
            "Not Specified",
        )
    try:
        with open("file_path.txt", "w", encoding="utf-8") as file:
            file.write(html_doc)
    except Exception as e:
        log.warning("parser | could not save debug HTML: %s", e)

    failed: list = []
    soup = BeautifulSoup(html_doc, "html.parser")

    def dealer_name_fn():
        try:
            el = soup.find("h1", attrs={"class": "dealerName"})
            if not el or not el.text:
                return "bad url"
            dn = str(el.text)
            dn = dn[: -dn[::-1].index("-") - 1]
            return dn
        except Exception as e:
            log.debug("dealer_name_fn: %s", e)
            return "bad url"

    def dealer_web_fn():
        try:
            el = soup.find("a", attrs={"target": "_blank"})
            if el:
                return str(el.text)
            return "Not Specified"
        except Exception as e:
            log.debug("dealer_web_fn: %s", e)
            failed.append("web not found")
            return "Not Specified"

    def dealer_count_fn():
        try:
            el = soup.find("div", attrs={"class": "resultCount"})
            if not el or not el.text:
                failed.append("count not found")
                return "0"
            dealer_count_text = el.text.strip()
            num, _res = dealer_count_text.split(" ", 1)
            return num
        except Exception as e:
            log.debug("dealer_count_fn: %s", e)
            failed.append("count not found")
            return "0"

    def dealer_phone_fn():
        try:
            el = soup.find("span", attrs={"class": "dealerSalesPhone"})
            if el:
                return el.text
            return "Not Specified"
        except Exception as e:
            log.debug("dealer_phone_fn: %s", e)
            failed.append("phone not found")
            return "Not Specified"

    def dealer_rate_fn():
        try:
            el = soup.find("div", attrs={"class": "starRating"})
            if el:
                return el.get("title", 0)
            return 0
        except Exception as e:
            log.debug("dealer_rate_fn: %s", e)
            failed.append("rate not found")
            return "Not Specified"

    def dealer_reviews_fn():
        try:
            el = soup.find("div", attrs={"class": "details"})
            if el:
                dr = str(el.text.split()[-1])
                if dr == "Reviews":
                    return 0
                if "(" in dr and ")" in dr:
                    return int(dr[dr.index("(") + 1 : dr.index(")")])
                return dr
            return 0
        except Exception as e:
            log.debug("dealer_reviews_fn: %s", e)
            failed.append("reviews not found")
            return "Not Specified"

    def dealer_time_fn():
        try:
            el = soup.find("div", attrs={"class": "dealerText"})
            if el:
                if el.text:
                    t = " ".join(el.text.split())
                elif el.strong:
                    t = el.strong.text
                else:
                    t = " ".join(el.span.text.split())
                out_time = ""
                for x in t.split():
                    if x == "-" or x.count(":") > 0:
                        out_time += x + " "
                if out_time.strip() == "":
                    return "Not Specified"
                return out_time.strip()
            return "Not Specified"
        except Exception as e:
            log.debug("dealer_time_fn: %s", e)
            failed.append("dealer_time not found")
            return "Not Specified"

    dname = dealer_name_fn()
    out_put = (
        dname,
        dealer_phone_fn(),
        dealer_web_fn(),
        dealer_count_fn(),
        dealer_rate_fn(),
        dealer_reviews_fn(),
        dealer_time_fn(),
    )
    return out_put


def crawler(links: Any, name: str, thread_name: str, number: int) -> None:
    rows: list = []
    total = len(links) if hasattr(links, "__len__") else 0
    log.info("crawler | starting %s dealer pages for %s_%s", total, name, number)
    for idx, link in enumerate(links):
        try:
            out_put = parser(get_link(link))
            rows.append(out_put)
        except Exception as e:
            _stats["parse_exceptions"] += 1
            log.exception("crawler | row %s failed: %s", idx, e)
            time.sleep(3)
        if (idx + 1) % 10 == 0:
            log_stats_snapshot(f"dealer_progress {idx + 1}/{total}")
    df = pd.DataFrame(
        rows,
        columns=[
            "Name",
            "Phone Number",
            "Website",
            "Number Of Cars",
            "Score",
            "Number Of Rates",
            "Business Hours",
        ],
    )
    out_path = f"primary_result/{name}_{number}.xlsx"
    writer = pd.ExcelWriter(out_path, engine="xlsxwriter")
    df.to_excel(writer)
    writer.close()
    log.info("crawler | wrote %s rows → %s", len(df), out_path)
    log_stats_snapshot(f"finished_grid_{name}_{number}")


def get_state_points(state_name: str) -> list[tuple[float, float]]:
    if not os.path.isfile(SHAPEFILE_PATH):
        log.error("Shapefile missing: %s (set SHAPEFILE_PATH or rebuild image)", SHAPEFILE_PATH)
        raise FileNotFoundError(SHAPEFILE_PATH)
    states = gpd.read_file(SHAPEFILE_PATH)
    if "admin" in states.columns:
        states = states[states["admin"] == "United States of America"]
    state = states[states["name"] == state_name]
    if state.empty:
        log.error("No geometry for state name=%r (check spelling)", state_name)
        raise ValueError(f"unknown state: {state_name}")
    minx, miny, maxx, maxy = state.total_bounds
    log.info(
        "get_state_points | state=%s bounds=(%.4f,%.4f)-(%.4f,%.4f)",
        state_name,
        minx,
        miny,
        maxx,
        maxy,
    )
    spacing_km = 160
    points = []
    x = minx
    while x < maxx:
        y = miny
        while y < maxy:
            point = Point(x, y)
            if state.geometry.contains(point).any():
                points.append((point.y, point.x))
            y += spacing_km / 111
        x += spacing_km / (111 * np.cos(np.deg2rad((miny + maxy) / 2)))
    if len(points) == 0:
        log.warning("get_state_points | no grid points; using centroid")
        return [(float((miny + maxy) / 2), float((minx + maxx) / 2))]
    log.info("get_state_points | generated %s sample points", len(points))
    return points


if __name__ == "__main__":
    os.makedirs("primary", exist_ok=True)
    os.makedirs("primary_result", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    log.info("Starting crawl | LOG_LEVEL=%s | SHAPEFILE_PATH=%s", LOG_LEVEL, SHAPEFILE_PATH)
    log.info(
        "HTTP tuning | CRAWL_USE_CURL_CFFI=%s delays=%s-%ss retries=%s",
        os.environ.get("CRAWL_USE_CURL_CFFI", "1"),
        os.environ.get("CRAWL_MIN_DELAY_SEC", "0.55"),
        os.environ.get("CRAWL_MAX_DELAY_SEC", "2.4"),
        os.environ.get("CRAWL_HTTP_RETRIES", "6"),
    )
    client = _cc()
    if not client.warmup():
        log.error(
            "Aborting: CarGurus warmup failed. Run: python check_cargurus_access.py  "
            "On GCP: pip install -r requirements.txt"
        )
        sys.exit(1)

    states_raw = os.environ.get("CRAWL_STATES", "Oregon")
    state_names = [s.strip() for s in states_raw.split(",") if s.strip()]
    if not state_names:
        log.error("CRAWL_STATES is empty")
        sys.exit(1)
    log.info("States to crawl: %s", state_names)
    for state_name in state_names:
        log.info("=== State: %s ===", state_name)
        points = get_state_points(state_name)
        for number, (latitude, longitude) in enumerate(points):
            log.info(
                "--- Grid cell %s/%s | lat=%.5f lon=%.5f ---",
                number + 1,
                len(points),
                latitude,
                longitude,
            )
            url = (
                f"https://www.cargurus.com/Cars/dl.action?entityId=&address={state_name}"
                f"&latitude={latitude}&longitude={longitude}&distance=100"
            )
            url += "&page="
            try:
                links, name = runner(url, "ali", number)
                crawler(links, name, "ali", number)
            except Exception as e:
                log.exception("Fatal step for grid %s: %s", number, e)
                
                log_stats_snapshot(f"error_grid_{number}")
            log_stats_snapshot(f"after_grid_{number}")

    log.info("All configured states finished.")
    log_stats_snapshot("final")
