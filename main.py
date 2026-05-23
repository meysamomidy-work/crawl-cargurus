import argparse
import json
import logging
import os
import queue
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

import geopandas as gpd
import numpy as np
import requests
from tqdm import tqdm
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
CARGURUS_HOME = f"{CARGURUS_ORIGIN}/"

# CarGurus: www root often returns 403 for script/datacenter clients; 418 = non-US egress.
NON_US_STATUS = 418
RETRY_STATUSES = frozenset({403, 406, 429, 502, 503, 504})


def default_warmup_listing_url() -> str:
    """Dealer-locator URL the crawler actually uses (not the marketing homepage)."""
    override = os.environ.get("CRAWL_WARMUP_LISTING_URL", "").strip()
    if override:
        return override
    # First state assigned to this terminal (set in main when using --worker/--workers).
    state_src = os.environ.get("CRAWL_TERMINAL_STATES", "") or os.environ.get("CRAWL_STATES", "Oregon")
    state = (state_src.split(",")[0] or "Oregon").strip()
    lat = os.environ.get("CRAWL_WARMUP_LAT", "44.0")
    lon = os.environ.get("CRAWL_WARMUP_LON", "-120.5")
    return (
        f"{CARGURUS_ORIGIN}/Cars/dl.action?entityId=&address={state}"
        f"&latitude={lat}&longitude={lon}&distance=100&page=0"
    )


def _response_usable(response: Any, *, min_bytes: int) -> bool:
    return response.status_code == 200 and _response_body_len(response) >= min_bytes

# Trial defaults: one state, one map grid cell, ten dealer detail pages. Override via env.
def _crawl_limit_int(key: str, default: int) -> int:
    v = os.environ.get(key)
    if v is None or not str(v).strip():
        return default
    return max(0, int(v))


def crawl_max_dealers() -> int:
    """Per grid cell during discovery; 0 = unlimited."""
    return _crawl_limit_int("CRAWL_MAX_DEALERS", 0)


def crawl_max_grid_cells() -> int:
    """Per state; 0 = all grid points."""
    return _crawl_limit_int("CRAWL_MAX_GRID_CELLS", 0)


def crawl_max_states() -> int:
    """0 = all configured states."""
    return _crawl_limit_int("CRAWL_MAX_STATES", 0)


# Same jurisdiction order as cars crowler/main8.py (override with CRAWL_STATES).
DEFAULT_STATES: list[str] = [
    'Connecticut',
    'Maine',
    'Massachusetts',
    'New Hampshire',
    'New Jersey',
    'New York',
    'Pennsylvania',
    'Rhode Island',
    'Vermont',
    'Wyoming'
]
# DEFAULT_STATES: list[str] = [
#     "Delaware",
#     "District of Columbia",
#     "Virginia",
#     "Maryland",
#     "West Virginia",
#     "North Carolina",
#     "South Carolina",
#     "Georgia",
#     "Florida",
#     "Alabama",
#     "Tennessee",
#     "Mississippi",
#     "Kentucky",
#     "Ohio",
#     "Indiana",
#     "Michigan",
#     "Iowa",
#     "Wisconsin",
#     "Minnesota",
#     "South Dakota",
#     "North Dakota",
#     "Montana",
#     "Illinois",
#     "Missouri",
#     "Kansas",
#     "Nebraska",
#     "Louisiana",
#     "Arkansas",
#     "Oklahoma",
#     "Texas",
#     "Colorado",
#     "Idaho",
#     "Utah",
#     "Arizona",
#     "New Mexico",
#     "Nevada",
#     "California",
#     "Hawaii",
#     "American Samoa",
#     "Guam",
#     "Northern Mariana Islands",
#     "Oregon",
#     "Washington",
#     "Alaska",
# ]



STATE_CSV_HEADER = (
    "Name|Dealer Page Link|List Address|Phone|Website|Inventory Count|"
    "Score|Review Count|Business Hours|State\n"
)

_WORKER_QUEUE_SENTINEL = object()
META_WRITE_LOCK = threading.Lock()
_PROGRESS_LOCK = threading.Lock()
_STATS_LOCK = threading.Lock()
_STATE_REGISTRY_LOCK = threading.Lock()
_thread_local = threading.local()

_STATE_SEEN: dict[str, set[str]] = {}
_STATE_FILE_LOCKS: dict[str, threading.Lock] = {}
_STATE_DEALER_COUNT: dict[str, int] = {}


@dataclass(frozen=True, slots=True)
class GridTask:
    state: str
    latitude: float
    longitude: float


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


def _stat_inc(key: str, n: int = 1) -> None:
    with _STATS_LOCK:
        _stats[key] = _stats.get(key, 0) + n


def _record_http_status(code: int) -> None:
    _stat_inc("http_requests")
    if 200 <= code < 300:
        _stat_inc("http_2xx")
    elif code == 403:
        _stat_inc("http_403")
    elif code == 429:
        _stat_inc("http_429")
    elif 400 <= code < 500:
        _stat_inc("http_other_4xx")
    elif code >= 500:
        _stat_inc("http_5xx")


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
        _stat_inc("block_hint_hits")
    if n < 1500 and code == 200:
        _stat_inc("tiny_html")

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
    elif code == NON_US_STATUS:
        log.error(
            "%s | HTTP 418 — CarGurus blocks non-US egress; use US VPN/proxy or US hosting | bytes=%s | url=%s",
            label,
            n,
            _url_for_log(url_short),
        )
    elif code == 403 and label.startswith("warmup_home"):
        log.warning(
            "%s | HTTP 403 on homepage (common for scripts; listing URLs may still work) | bytes=%s | url=%s",
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
        listing = default_warmup_listing_url()

        self.throttle()
        r_home = self._get(CARGURUS_HOME, nav_headers(same_site=False, referer=CARGURUS_HOME), tmo)
        inspect_http_response(r_home, "warmup_home", CARGURUS_HOME)

        if r_home.status_code == NON_US_STATUS:
            log.error("warmup | HTTP 418 on homepage — egress is not treated as US; aborting")
            return False

        cars = f"{CARGURUS_ORIGIN}/Cars/"
        self.throttle()
        r_cars = self._get(cars, nav_headers(same_site=True, referer=CARGURUS_HOME), tmo)
        inspect_http_response(r_cars, "warmup_cars", cars)

        self.throttle()
        r_list = self._get(
            listing,
            nav_headers(same_site=False, referer=CARGURUS_HOME),
            tmo,
        )
        inspect_http_response(r_list, "warmup_listing", listing)

        if r_list.status_code == NON_US_STATUS:
            log.error(
                "warmup | HTTP 418 on dealer listing — use US egress (VPN/proxy or US ISP hosting)"
            )
            return False

        ok_listing = _response_usable(r_list, min_bytes=1_500)
        # Homepage 403 is normal for GCP/script TLS; cookies from /Cars/ + listing matter for crawl.
        self._warmed = ok_listing
        if self._warmed:
            log.info(
                "warmup | ready (listing OK; home=%s cars=%s — root 403 is OK if listing works)",
                r_home.status_code,
                r_cars.status_code,
            )
        else:
            log.error(
                "warmup | failed — need listing HTTP 200 (home=%s bytes=%s | cars=%s | listing=%s bytes=%s). "
                "Run: python check_cargurus_access.py — on GCP: pip install curl_cffi",
                r_home.status_code,
                _response_body_len(r_home),
                r_cars.status_code,
                r_list.status_code,
                _response_body_len(r_list),
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
                _stat_inc("http_retry_rounds")
                log.warning("%s | retry round %s/%s", label, attempt + 1, max_r)
                self.warmup(force=True)
            elif not self._warmed:
                self.warmup(force=False)
            self.throttle()
            try:
                last = self._get(url, headers_fn(), timeout)
            except Exception as e:
                _stat_inc("http_errors")
                log.warning("%s | attempt %s transport error: %s", label, attempt + 1, e)
                last = None
                time.sleep(backoff * (1.6**attempt) * random.uniform(0.85, 1.15))
                self._warmed = False
                continue
            inspect_http_response(last, label, url)
            if last.status_code == 200:
                return last
            if last.status_code == NON_US_STATUS:
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


def _cc() -> CargurusClient:
    """One HTTP session + cookie jar per worker thread."""
    client = getattr(_thread_local, "client", None)
    if client is None:
        client = CargurusClient()
        if not client.warmup():
            raise RuntimeError("CarGurus warmup failed (check US egress / curl_cffi)")
        _thread_local.client = client
    return client


def _csv_field(value: Any) -> str:
    return str(value if value is not None else "").replace("|", " ").replace("\n", " ").strip()


def _append_line(path: str, line: str) -> None:
    line = line if line.endswith("\n") else line + "\n"
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    rel = os.path.normpath(path).replace("\\", "/").lower()
    if rel.startswith("crawled/meta/"):
        with META_WRITE_LOCK:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)
    else:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)


def _dealer_page_url(href: str) -> str:
    href = (href or "").strip()
    if href.startswith("http"):
        return href.split("?")[0]
    return f"{CARGURUS_ORIGIN}/{href.lstrip('/')}".split("?")[0]


def _write_state_csv_header(state: str) -> None:
    os.makedirs("crawled", exist_ok=True)
    with open(f"crawled/{state}.csv", "w", encoding="utf-8") as f:
        f.write(STATE_CSV_HEADER)


def _load_seen_dealer_urls(state: str) -> set[str]:
    path = f"crawled/{state}.csv"
    seen: set[str] = set()
    if not os.path.isfile(path):
        return seen
    try:
        with open(path, encoding="utf-8") as f:
            f.readline()
            for line in f:
                parts = line.strip().split("|")
                if len(parts) >= 2 and parts[1]:
                    seen.add(parts[1])
    except OSError as exc:
        log.warning("resume | could not read %s: %s", path, exc)
    return seen


def _format_result_row(
    list_name: str,
    list_address: str,
    href: str,
    parsed: tuple,
    state: str,
) -> str:
    name, phone, website, inv, score, reviews, hours = parsed
    if name == "bad url" and list_name:
        name = list_name
    return "|".join(
        [
            _csv_field(name),
            _csv_field(_dealer_page_url(href)),
            _csv_field(list_address),
            _csv_field(phone),
            _csv_field(website),
            _csv_field(inv),
            _csv_field(score),
            _csv_field(reviews),
            _csv_field(hours),
            _csv_field(state),
        ]
    )


def _num_workers() -> int:
    return max(1, int(os.environ.get("CRAWL_NUM_WORKERS", "3")))


def _configured_states() -> list[str]:
    raw = os.environ.get("CRAWL_STATES", "").strip()
    if raw:
        names = [s.strip() for s in raw.split(",") if s.strip()]
    else:
        names = list(DEFAULT_STATES)
    max_states = crawl_max_states()
    if max_states > 0:
        names = names[:max_states]
    return names


def _parse_terminal_worker_cli(argv: list[str] | None = None) -> tuple[int, int]:
    """
    Terminal/process sharding for multi-window runs (not the in-process thread pool).

    Returns (worker_index 0-based, worker_count).
    CLI: --worker 1 --workers 4  (first of four terminals)
    Env: CRAWL_TERMINAL_WORKER, CRAWL_TERMINAL_WORKERS (same 1-based worker id)
    """
    parser = argparse.ArgumentParser(
        description="CarGurus dealer crawler",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-w",
        "--worker",
        type=int,
        default=None,
        help="This terminal's index (1 .. --workers). Divides states across processes.",
    )
    parser.add_argument(
        "-W",
        "--workers",
        type=int,
        default=None,
        help="How many terminals/processes are running the crawl in parallel.",
    )
    args = parser.parse_args(argv)

    env_worker = os.environ.get("CRAWL_TERMINAL_WORKER", "").strip()
    env_workers = os.environ.get("CRAWL_TERMINAL_WORKERS", "").strip()
    worker_1based = args.worker
    if worker_1based is None and env_worker:
        worker_1based = int(env_worker)
    workers = args.workers
    if workers is None and env_workers:
        workers = int(env_workers)

    if worker_1based is None:
        worker_1based = 1
    if workers is None:
        workers = 1

    if workers < 1:
        parser.error("--workers must be >= 1")
    if worker_1based < 1 or worker_1based > workers:
        parser.error(f"--worker must be between 1 and {workers} (got {worker_1based})")

    return worker_1based - 1, workers


def _states_for_terminal(
    all_states: list[str], worker_index: int, worker_count: int
) -> list[str]:
    """Round-robin split so each terminal gets a fair share of states."""
    if worker_count <= 1:
        return list(all_states)
    return [s for i, s in enumerate(all_states) if i % worker_count == worker_index]


def _add_terminal_log_handler(worker_index: int, worker_count: int) -> None:
    if worker_count <= 1:
        return
    path = os.path.join(
        _log_dir, f"crawl-terminal-{worker_index + 1}-of-{worker_count}.log"
    )
    for h in _root.handlers:
        if getattr(h, "baseFilename", None) == os.path.abspath(path):
            return
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setFormatter(_fmt)
    _root.addHandler(fh)
    log.info("Terminal log file: %s", path)


def _delay_range() -> tuple[float, float]:
    lo = float(os.environ.get("CRAWL_MIN_DELAY_SEC", "0.55"))
    hi = float(os.environ.get("CRAWL_MAX_DELAY_SEC", "2.4"))
    if hi < lo:
        lo, hi = hi, lo
    return lo, hi


def _detail_workers() -> int:
    """Concurrent dealer-detail HTTP fetches per grid cell (each uses thread-local session)."""
    return max(1, int(os.environ.get("CRAWL_DETAIL_WORKERS", "1")))


def _state_file_lock(state: str) -> threading.Lock:
    with _STATE_REGISTRY_LOCK:
        lock = _STATE_FILE_LOCKS.get(state)
        if lock is None:
            lock = threading.Lock()
            _STATE_FILE_LOCKS[state] = lock
        return lock


def _ensure_state_tracking(state: str) -> None:
    with _STATE_REGISTRY_LOCK:
        if state in _STATE_SEEN:
            return
        path = f"crawled/{state}.csv"
        if _env_bool("CRAWL_RESUME", True) and os.path.isfile(path) and os.path.getsize(path) > 0:
            seen = _load_seen_dealer_urls(state)
            _STATE_SEEN[state] = seen
            _STATE_DEALER_COUNT[state] = len(seen)
            log.info("resume | %s already has %s dealer(s)", state, len(seen))
        else:
            _STATE_SEEN[state] = set()
            _STATE_DEALER_COUNT[state] = 0
            _write_state_csv_header(state)


def _reserve_dealer_slot(state: str, page_url: str) -> bool:
    max_per_state = _crawl_limit_int("CRAWL_MAX_DEALERS_PER_STATE", 0)
    with _STATE_REGISTRY_LOCK:
        seen = _STATE_SEEN[state]
        if page_url in seen:
            return False
        if max_per_state > 0 and _STATE_DEALER_COUNT.get(state, 0) >= max_per_state:
            return False
        seen.add(page_url)
        _STATE_DEALER_COUNT[state] = _STATE_DEALER_COUNT.get(state, 0) + 1
        return True


def _release_dealer_slot(state: str, page_url: str) -> None:
    with _STATE_REGISTRY_LOCK:
        seen = _STATE_SEEN.get(state)
        if not seen or page_url not in seen:
            return
        seen.discard(page_url)
        _STATE_DEALER_COUNT[state] = max(0, _STATE_DEALER_COUNT.get(state, 1) - 1)


def _append_state_row(state: str, line: str) -> None:
    with _state_file_lock(state):
        path = f"crawled/{state}.csv"
        row = line if line.endswith("\n") else line + "\n"
        os.makedirs("crawled", exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(row)


def _build_grid_tasks(states: list[str]) -> list[GridTask]:
    max_cells = crawl_max_grid_cells()
    tasks: list[GridTask] = []
    for state in states:
        points = get_state_points(state)
        if max_cells > 0:
            points = points[:max_cells]
        for latitude, longitude in points:
            tasks.append(GridTask(state, latitude, longitude))
    if _env_bool("CRAWL_SHUFFLE_GRIDS", True):
        random.shuffle(tasks)
    return tasks


def get_link(link: str) -> str | None:
    try:
        url = f"{CARGURUS_ORIGIN}/" + link.lstrip("/")
        tmo = float(os.environ.get("CRAWL_DEALER_TIMEOUT_SEC", "45"))
        response = _cc().fetch(
            url,
            label="dealer_detail",
            headers_fn=lambda: nav_headers(same_site=True, referer=CARGURUS_HOME),
            timeout=tmo,
        )
        if response is None:
            _stat_inc("dealer_pages_failed")
            return None
        if response.status_code != 200:
            _stat_inc("dealer_pages_failed")
            return None
        body = _response_text(response)
        hints = _html_block_hints(body)
        if hints:
            log.warning("dealer_detail | block-like hints on 200: %s", hints[:5])
        _stat_inc("dealer_pages_ok")
        return body
    except Exception:
        _stat_inc("http_errors")
        _stat_inc("dealer_pages_failed")
        log.exception("dealer_detail | exception | link=%s", link[:200])
        return None


def get(url: str) -> Any | Exception:
    try:
        tmo = float(os.environ.get("CRAWL_LISTING_TIMEOUT_SEC", "35"))
        html_doc = _cc().fetch(
            url,
            label="listing_search",
            headers_fn=lambda: nav_headers(same_site=False, referer=CARGURUS_HOME),
            timeout=tmo,
        )
        if html_doc is None:
            return RuntimeError("listing_search: no response after retries")
        return html_doc
    except Exception as e:
        _stat_inc("http_errors")
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


def discover_dealers_from_grid(
    info: tuple,
    url: str,
    client: CargurusClient,
) -> list[tuple[str, str, str]]:
    dealer_list: list[tuple[str, str, str]] = []
    page_raw = info[1]
    if page_raw is None or (isinstance(page_raw, str) and not str(page_raw).strip()):
        raise ValueError("empty listing page info")
    page_clean = str(page_raw).replace(",", "")
    page_count = int(page_clean) // 10 + 1
    max_dealers = crawl_max_dealers()
    if max_dealers > 0:
        page_count = min(page_count, max(1, (max_dealers + 9) // 10))
    grid_tmo = float(os.environ.get("CRAWL_GRID_TIMEOUT_SEC", "28"))
    for start in range(page_count):
        if max_dealers > 0 and len(dealer_list) >= max_dealers:
            break
        link = url + str(start)
        referer = CARGURUS_HOME if start == 0 else url + str(start - 1)
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
        for inv_href, addr, name_el in zip(href_list, dealer_address, dealer_name):
            if max_dealers > 0 and len(dealer_list) >= max_dealers:
                break
            list_name = name_el.strong.text if name_el.strong else name_el.get_text(strip=True)
            dealer_list.append((list_name, " ".join(addr.split()), inv_href))

    if max_dealers > 0:
        dealer_list = dealer_list[:max_dealers]
    return dealer_list


def _extract_remix_route_loader(html: str) -> dict | None:
    """CarGurus dealer inventory pages embed SSR data in window.__remixContext."""
    marker = "window.__remixContext = "
    start = html.find(marker)
    if start < 0:
        return None
    start += len(marker)
    try:
        data, _ = json.JSONDecoder().raw_decode(html, start)
    except json.JSONDecodeError:
        return None
    loader_data = data.get("state", {}).get("loaderData") or {}
    for val in loader_data.values():
        if isinstance(val, dict) and "dealerInfo" in val:
            return val
    return None


def _normalize_time_text(value: str) -> str:
    return value.replace("\u202f", " ").replace("\u00a0", " ").strip()


def _format_business_hours(business_hours: dict) -> str:
    if not business_hours:
        return "Not Specified"
    parts: list[str] = []
    for day, info in business_hours.items():
        if not isinstance(info, dict) or info.get("availability") != "Open":
            continue
        open_t = _normalize_time_text(str(info.get("openTime", "")))
        close_t = _normalize_time_text(str(info.get("closeTime", "")))
        if open_t and close_t:
            parts.append(f"{day[:3]} {open_t}-{close_t}")
    return " ".join(parts) if parts else "Not Specified"


def _parser_from_remix(html_doc: str) -> tuple | None:
    route = _extract_remix_route_loader(html_doc)
    if not route:
        return None
    dealer = route.get("dealerInfo") or {}
    search = route.get("search") or {}
    name = (dealer.get("name") or "").strip()
    if not name:
        return None
    phone = (
        dealer.get("localizedSalesPhone")
        or dealer.get("salesPhone")
        or "Not Specified"
    )
    website = dealer.get("website") or "Not Specified"
    count = search.get("totalListings")
    count_str = str(count) if count is not None else "0"
    score = dealer.get("averageRating", 0)
    reviews = dealer.get("reviewCount", 0)
    hours = _format_business_hours(dealer.get("businessHours") or {})
    return (name, phone, website, count_str, score, reviews, hours)


def _parser_empty() -> tuple:
    return (
        "bad url",
        "Not Specified",
        "Not Specified",
        "0",
        0,
        0,
        "Not Specified",
    )


def parser(html_doc: str | None) -> tuple:
    if not html_doc:
        log.warning("parser | empty HTML (dealer page failed or blocked)")
        return _parser_empty()

    remix_row = _parser_from_remix(html_doc)
    if remix_row is not None:
        return remix_row

    if _env_bool("CRAWL_SAVE_DEALER_HTML", False):
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
    if dname == "bad url":
        hints = _html_block_hints(html_doc)
        log.warning(
            "parser | legacy DOM parse failed (no __remixContext dealerInfo); hints=%s",
            hints[:5],
        )
    return (
        dname,
        dealer_phone_fn(),
        dealer_web_fn(),
        dealer_count_fn(),
        dealer_rate_fn(),
        dealer_reviews_fn(),
        dealer_time_fn(),
    )


def process_grid_cell(task: GridTask, worker_id: int, client: CargurusClient) -> int:
    """One map grid cell: discover listing pages, fetch dealer details, append CSV rows."""
    _ensure_state_tracking(task.state)
    base_url = (
        f"{CARGURUS_ORIGIN}/Cars/dl.action?entityId=&address={task.state}"
        f"&latitude={task.latitude}&longitude={task.longitude}&distance=100&page="
    )
    try:
        response = get(base_url + "0")
        info = page(response)
        if isinstance(info, Exception):
            raise info
        dealers = discover_dealers_from_grid(info, base_url, client)
    except Exception as exc:
        _append_line(
            "crawled/meta/error_grids.csv",
            f"{task.state}|{task.latitude:.5f}|{task.longitude:.5f}|{type(exc).__name__}: {exc}",
        )
        log.warning(
            "W%s %s grid (%.4f, %.4f) failed: %s",
            worker_id,
            task.state,
            task.latitude,
            task.longitude,
            exc,
        )
        return 0

    work: list[tuple[str, str, str, str]] = []
    for list_name, address, href in dealers:
        page_url = _dealer_page_url(href)
        if _reserve_dealer_slot(task.state, page_url):
            work.append((list_name, address, href, page_url))

    written = 0
    detail_n = _detail_workers()

    def _fetch_one(item: tuple[str, str, str, str]) -> bool:
        list_name, address, href, page_url = item
        try:
            parsed = parser(get_link(href))
        except Exception as exc:
            _release_dealer_slot(task.state, page_url)
            _stat_inc("parse_exceptions")
            log.exception("W%s dealer failed %s: %s", worker_id, page_url, exc)
            return False
        row = _format_result_row(list_name, address, href, parsed, task.state)
        _append_state_row(task.state, row)
        print(f"[W{worker_id} {task.state}] {parsed[0]}")
        return True

    if detail_n <= 1:
        for item in work:
            if _fetch_one(item):
                written += 1
    else:
        with ThreadPoolExecutor(max_workers=detail_n) as pool:
            futures = [pool.submit(_fetch_one, item) for item in work]
            for fut in as_completed(futures):
                if fut.result():
                    written += 1

    dmin, dmax = _delay_range()
    time.sleep(random.uniform(dmin, dmax))
    return written


def worker_entry(worker_id: int, task_queue: queue.Queue, progress: tqdm) -> None:
    tag = f"[W{worker_id}]"
    try:
        client = _cc()
        print(f"{tag} HTTP session ready")
        while True:
            item = task_queue.get()
            if item is _WORKER_QUEUE_SENTINEL:
                break
            task = item
            assert isinstance(task, GridTask)
            try:
                n = process_grid_cell(task, worker_id, client)
            except Exception as exc:
                log.exception("%s grid failed %s: %s", tag, task.state, exc)
                _append_line(
                    "crawled/meta/error_grids.csv",
                    f"{task.state}|{task.latitude:.5f}|{task.longitude:.5f}|{type(exc).__name__}: {exc}",
                )
                n = 0
            with _PROGRESS_LOCK:
                progress.update(1)
                progress.set_postfix(
                    worker=worker_id,
                    state=task.state,
                    last=n,
                    refresh=False,
                )
    finally:
        print(f"{tag} exiting")


def main(
    *,
    terminal_index: int = 0,
    terminal_count: int = 1,
) -> None:
    os.makedirs("crawled/meta", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    _add_terminal_log_handler(terminal_index, terminal_count)

    all_states = _configured_states()
    if not all_states:
        log.error("No states configured (set CRAWL_STATES or use DEFAULT_STATES)")
        sys.exit(1)

    states = _states_for_terminal(all_states, terminal_index, terminal_count)
    if not states:
        print(
            f"[cargurus] Terminal {terminal_index + 1}/{terminal_count}: "
            f"no states assigned (only {len(all_states)} state(s) in config). Exiting."
        )
        return

    os.environ["CRAWL_TERMINAL_STATES"] = ",".join(states)
    term_label = (
        f"terminal {terminal_index + 1}/{terminal_count}"
        if terminal_count > 1
        else "single terminal"
    )
    print(
        f"[cargurus] {term_label} | {len(states)} state(s): "
        + ", ".join(states[:12])
        + (" …" if len(states) > 12 else "")
    )
    log.info(
        "Terminal partition | %s | assigned=%s | all_configured=%s",
        term_label,
        states,
        len(all_states),
    )

    n = _num_workers()
    detail_n = _detail_workers()
    print("[cargurus] Building grid task list (shapefile lookup per state)...")
    grid_tasks = _build_grid_tasks(states)
    if not grid_tasks:
        log.error("No grid tasks generated")
        sys.exit(1)

    print(
        f"[cargurus] {len(states)} state(s) | {len(grid_tasks)} grid task(s) | "
        f"{n} in-process thread(s) (CRAWL_NUM_WORKERS) | "
        f"{detail_n} detail fetch(es) per grid | → crawled/<State>.csv"
    )
    log.info(
        "Starting crawl | terminal=%s/%s | threads=%s | detail_workers=%s | "
        "states=%s | grids=%s",
        terminal_index + 1,
        terminal_count,
        n,
        detail_n,
        len(states),
        len(grid_tasks),
    )

    task_queue: queue.Queue = queue.Queue()
    for task in grid_tasks:
        task_queue.put(task)
    for _ in range(n):
        task_queue.put(_WORKER_QUEUE_SENTINEL)

    progress = tqdm(total=len(grid_tasks), desc="grids", unit="grid")
    stagger = float(os.environ.get("CRAWL_STAGGER_SEC", "3"))
    threads = [
        threading.Thread(
            target=worker_entry,
            args=(wid, task_queue, progress),
            name=f"cargurus-worker-{wid}",
            daemon=False,
        )
        for wid in range(n)
    ]
    for t in threads:
        t.start()
        time.sleep(random.uniform(0.5, max(stagger, 0.5)))
    for t in threads:
        t.join()
    progress.close()

    print("[cargurus] All workers finished.")
    log_stats_snapshot("final")


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
    t_index, t_count = _parse_terminal_worker_cli()
    main(terminal_index=t_index, terminal_count=t_count)
