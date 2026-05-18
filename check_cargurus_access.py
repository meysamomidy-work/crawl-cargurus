#!/usr/bin/env python3
"""
Quick diagnostic: public IP + country, then GET(s) to CarGurus.
Run on the same machine (or in the same Docker network) as your crawler.

  python check_cargurus_access.py

CarGurus behavior (typical):
  - https://www.cargurus.com/  → often HTTP 403 for script/datacenter clients (even in US)
  - Non-US egress              → HTTP 418
  - Dealer listing URLs        → what main.py actually crawls; 200 here means crawl can work

Geo lookup uses only the standard library. HTTP is probed with urllib and curl_cffi when installed.

Install on GCP: pip install curl_cffi
"""

from __future__ import annotations

import json
from collections.abc import Callable
import re
import ssl
import sys
import urllib.error
import urllib.request

CARGURUS_ORIGIN = "https://www.cargurus.com"
CARGURUS_HOME = f"{CARGURUS_ORIGIN}/"
CARGURUS_LISTING_PROBE = (
    f"{CARGURUS_ORIGIN}/Cars/dl.action?entityId=&address=Oregon"
    "&latitude=44.0&longitude=-120.5&distance=100&page=0"
)

GEO_PROVIDERS: tuple[tuple[str, str], ...] = (
    ("ip-api.com", "https://ip-api.com/json/?fields=status,message,country,countryCode,city,query,isp"),
    ("ifconfig.co", "https://ifconfig.co/json"),
    ("ipinfo.io", "https://ipinfo.io/json"),
)

STRONG_BLOCK_HINTS = (
    "unusual traffic from your computer",
    "checking your browser before accessing",
    "enable javascript and cookies to continue",
    "just a moment...",
    "cf-browser-verification",
    "challenge-platform",
    "you have been blocked",
    "why have i been blocked",
    "access to this site has been denied",
    "requests from your network are automated",
    "error code 1020",
    "attention required!",
)

CHROME_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

HEADERS = {
    "User-Agent": CHROME_UA,
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,"
        "image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "sec-ch-ua": '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
}

_MIN_UA = {"User-Agent": HEADERS["User-Agent"]}


def _get_urllib(
    url: str, headers: dict[str, str], timeout: float = 20.0
) -> tuple[int, bytes, str | None]:
    req = urllib.request.Request(url, headers=headers)
    ctx = ssl.create_default_context()
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=timeout) as resp:
            body = resp.read()
            ctype = resp.headers.get("Content-Type")
            return resp.status, body, ctype
    except urllib.error.HTTPError as e:
        body = e.read() if e.fp else b""
        return e.code, body, e.headers.get("Content-Type") if e.headers else None


def _get_curl_cffi(
    url: str, headers: dict[str, str], timeout: float = 20.0
) -> tuple[int, bytes, str | None, str]:
    from curl_cffi import requests as cfr

    impersonate = "chrome131"
    try:
        r = cfr.get(url, headers=headers, timeout=timeout, impersonate=impersonate)
    except Exception:
        impersonate = "chrome120"
        r = cfr.get(url, headers=headers, timeout=timeout, impersonate=impersonate)
    body = r.content or b""
    ctype = r.headers.get("Content-Type") if r.headers else None
    return r.status_code, body, ctype, impersonate


def _parse_geo_json(name: str, raw: bytes) -> dict | None:
    try:
        j = json.loads(raw.decode("utf-8", errors="replace"))
    except json.JSONDecodeError:
        return None
    if name == "ip-api.com":
        if j.get("status") == "fail":
            return None
        return {
            "ip": j.get("query"),
            "country": j.get("country"),
            "country_code": (j.get("countryCode") or "").upper() or None,
            "city": j.get("city"),
            "isp": j.get("isp"),
        }
    if name == "ifconfig.co":
        cc = j.get("country_iso") or j.get("country")
        if isinstance(cc, str):
            cc = cc.upper()
        return {
            "ip": j.get("ip"),
            "country": j.get("country") or j.get("country_name"),
            "country_code": cc if cc and len(cc) == 2 else None,
            "city": j.get("city"),
            "isp": j.get("asn_org") or j.get("asn"),
        }
    if name == "ipinfo.io":
        cc = j.get("country")
        if isinstance(cc, str):
            cc = cc.upper()
        return {
            "ip": j.get("ip"),
            "country": None,
            "country_code": cc if cc and len(cc) == 2 else None,
            "city": j.get("city"),
            "isp": j.get("org"),
        }
    return None


def fetch_public_geo() -> tuple[dict | None, str | None]:
    for name, url in GEO_PROVIDERS:
        try:
            code, raw, _ = _get_urllib(url, _MIN_UA, timeout=15.0)
            if code != 200:
                if name == GEO_PROVIDERS[0][0]:
                    print(f"  {name}: HTTP {code} (common from cloud IPs — trying fallbacks…)")
                continue
            parsed = _parse_geo_json(name, raw)
            if parsed and parsed.get("ip"):
                return parsed, name
        except Exception as e:
            print(f"  {name}: error {e}")
            continue
    return None, None


def _html_title(html_lower: str) -> str:
    m = re.search(r"<title[^>]*>([^<]{0,300})", html_lower, flags=re.I)
    return (m.group(1).strip() if m else "")[:200]


def _status_note(code: int, *, is_homepage: bool) -> str:
    if code == 418:
        return "Non-US egress — CarGurus returns 418 outside the USA"
    if code == 403 and is_homepage:
        return "Expected for many script/datacenter clients; check dealer listing probe below"
    if code == 403:
        return "Forbidden — bot/WAF or missing curl_cffi"
    if code == 406:
        return "Not Acceptable — use curl_cffi (Python TLS fingerprint)"
    if code == 200:
        return "OK"
    return f"HTTP {code}"


def _probe_ok(code: int, body: bytes, low: str, title: str, *, min_bytes: int) -> bool:
    strong_hits = [h for h in STRONG_BLOCK_HINTS if h in low]
    suspicious_title = any(
        x in title.lower()
        for x in ("attention", "just a moment", "access denied", "blocked", "verify")
    )
    return (
        code == 200
        and len(body) >= min_bytes
        and "cargurus" in low
        and not strong_hits
        and not suspicious_title
    )


def _print_probe_result(
    label: str, url: str, code: int, body: bytes, ctype: str | None, *, min_bytes: int
) -> bool:
    n = len(body)
    text = body.decode("utf-8", errors="ignore")
    low = text.lower()
    snippet = text[:500].replace("\n", " ")
    title = _html_title(low)
    strong_hits = [h for h in STRONG_BLOCK_HINTS if h in low]
    is_home = url.rstrip("/") == CARGURUS_ORIGIN

    print(f"\n--- {label} ---")
    print(f"  URL:          {url}")
    print(f"  HTTP status:  {code} — {_status_note(code, is_homepage=is_home)}")
    print(f"  Content-Type: {ctype}")
    print(f"  Body bytes:   {n}")
    if title:
        print(f"  <title>:      {title!r}")
    if n > 0:
        print(f"  Body start:   {snippet!r}...")

    ok = _probe_ok(code, body, low, title, min_bytes=min_bytes)
    if ok:
        print("  Probe OK for crawl.")
    elif code == 418:
        print("  Probe FAIL: need US egress (VPN/proxy or US residential IP).")
    elif code == 403 and is_home:
        print("  Homepage 403 alone does NOT mean crawl is broken — see listing probe.")
    elif code in (403, 406, 429, 451):
        print(f"  Probe FAIL: HTTP {code}.")
    elif code == 200 and n < min_bytes:
        print("  Probe suspicious: 200 but small body.")
    else:
        print(f"  Probe unclear: HTTP {code}, {n} bytes.")
    if strong_hits:
        print(f"  WAF phrases: {strong_hits[:6]}{'…' if len(strong_hits) > 6 else ''}")
    return ok


def _probe_url_urllib(url: str) -> tuple[int, bytes, str | None, str]:
    code, body, ctype = _get_urllib(url, HEADERS, 20.0)
    return code, body, ctype, "stdlib urllib"


def _probe_url_cffi(url: str) -> tuple[int, bytes, str | None, str]:
    code, body, ctype, imp = _get_curl_cffi(url, HEADERS, 20.0)
    return code, body, ctype, f"curl_cffi impersonate={imp}"


def _run_client_probe(
    fetch_one: Callable[[str], tuple[int, bytes, str | None, str]],
) -> tuple[bool, int | None]:
    """Returns (listing_ok, 418 if seen)."""
    listing_ok = False
    geo_block: int | None = None

    for desc, url, min_b in (
        ("homepage", CARGURUS_HOME, 10_000),
        ("dealer listing (crawl path)", CARGURUS_LISTING_PROBE, 1_500),
    ):
        try:
            code, body, ctype, client = fetch_one(url)
            label = f"{client} — {desc}"
            ok = _print_probe_result(label, url, code, body, ctype, min_bytes=min_b)
            if code == 418:
                geo_block = 418
            if desc.startswith("dealer"):
                listing_ok = ok
        except Exception as e:
            print(f"\n--- {desc} ---\n  Request failed: {e}")
    return listing_ok, geo_block


def main() -> int:
    print("=== 1) Your public IP / country ===")
    geo, src = fetch_public_geo()
    cc = "?"
    if not geo:
        print("  Could not determine IP/country (all geo endpoints failed or returned empty).")
    else:
        print(f"  Source:  {src}")
        print(f"  IP:      {geo.get('ip')}")
        cc = geo.get("country_code") or "?"
        print(f"  Country: {geo.get('country') or '(see code)'} ({cc})")
        if geo.get("city"):
            print(f"  City:    {geo.get('city')}")
        if geo.get("isp"):
            print(f"  ISP/org: {geo.get('isp')}")

    print(
        "\n=== 2) GET CarGurus (homepage + dealer listing) ===\n"
        "Note: www.cargurus.com/ often returns 403 for scripts; main.py uses listing URLs."
    )

    urllib_listing, urllib_418 = _run_client_probe(_probe_url_urllib)

    cffi_listing = False
    cffi_418: int | None = None
    cffi_note: str | None = None
    try:
        cffi_listing, cffi_418 = _run_client_probe(_probe_url_cffi)
    except ImportError:
        cffi_note = "curl_cffi not installed — pip install curl_cffi"
    except Exception as e:
        cffi_note = f"curl_cffi failed: {e}"

    if cffi_note:
        print(f"\n--- curl_cffi ---\n  Skipped: {cffi_note}")

    isp = (geo or {}).get("isp") or ""
    is_cloud = any(
        x in isp.lower()
        for x in ("google cloud", "google llc", "amazon", "microsoft azure", "digitalocean", "ovh")
    )

    print("\n=== 3) Verdict ===")
    if urllib_418 == 418 or cffi_418 == 418:
        print("  HTTP 418: egress is not US — CarGurus blocks non-US IPs. Use US VPN/proxy.")
        if cc and cc != "US":
            print(f"  Geo says {cc} — matches 418 behavior.")
        return 2

    if cffi_listing or urllib_listing:
        print("  Dealer listing probe OK — main.py crawl path should work.")
        if cffi_listing and not urllib_listing:
            print("  Use curl_cffi in production (pip install curl_cffi; default in main.py).")
        print("  Homepage 403 is normal and ignored by main.py warmup.")
        return 0

    if cc and cc != "US" and cc != "?":
        print(f"  Country is {cc} — expect HTTP 418 from CarGurus; use US egress.")

    if is_cloud:
        print(
            "  US datacenter IP: homepage 403 + listing fail often means install curl_cffi,\n"
            "  not that Chrome-on-VM working proves Python will (different TLS fingerprint)."
        )
    if cffi_note and "not installed" in cffi_note:
        print(f"  Next: {cffi_note}")
    else:
        print("  Listing probe failed — fix curl_cffi or use US residential/non-cloud egress.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
