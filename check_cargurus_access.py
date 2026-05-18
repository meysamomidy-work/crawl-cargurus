#!/usr/bin/env python3
"""
Quick diagnostic: public IP + country, then GET(s) to CarGurus.
Run on the same machine (or in the same Docker network) as your crawler.

  python check_cargurus_access.py

Geo lookup uses only the standard library. CarGurus is probed twice when possible:
  1) urllib (Python OpenSSL TLS) — often HTTP 406 from GCP/AWS even when Chrome works
  2) curl_cffi Chrome impersonation — same stack as main.py when CRAWL_USE_CURL_CFFI=1

Install the crawler client on the VPS if probe (1) fails but Chrome works:
  pip install curl_cffi

Note: ip-api.com often returns HTTP 403 from cloud/datacenter IPs; this script
falls back to other read-only geo endpoints.

If both probes fail with 403/406/451, egress is likely blocked for scripts; try
curl_cffi, a residential/US proxy, or non-datacenter hosting even for US IPs.
"""

from __future__ import annotations

import json
import re
import ssl
import sys
import urllib.error
import urllib.request

CARGURUS_URL = "https://www.cargurus.com/"

# ip-api is first choice but frequently blocks GCP/AWS egress with 403.
GEO_PROVIDERS: tuple[tuple[str, str], ...] = (
    ("ip-api.com", "https://ip-api.com/json/?fields=status,message,country,countryCode,city,query,isp"),
    ("ifconfig.co", "https://ifconfig.co/json"),
    ("ipinfo.io", "https://ipinfo.io/json"),
)

# Phrases that suggest a WAF/challenge page — not bare "captcha"/"403" (normal sites
# embed reCAPTCHA scripts and mention status codes in JS).
STRONG_BLOCK_HINTS = (
    "unusual traffic from your computer",
    "checking your browser before accessing",
    "enable javascript and cookies to continue",
    "just a moment...",  # Cloudflare interstitial title pattern (partial match ok)
    "cf-browser-verification",
    "challenge-platform",
    "you have been blocked",
    "why have i been blocked",
    "access to this site has been denied",
    "requests from your network are automated",
    "error code 1020",
    "attention required!",  # some CDNs
)

CHROME_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

# Match main.py DEFAULT_HEADERS — bare User-Agent + Accept often still gets 406 on GCP.
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
    """Chrome TLS/JA3 impersonation — same approach as main.CargurusClient."""
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
    """Return (geo_dict, source_name_or_error)."""
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


def _probe_ok(code: int, body: bytes, low: str, title: str) -> bool:
    strong_hits = [h for h in STRONG_BLOCK_HINTS if h in low]
    suspicious_title = any(
        x in title.lower()
        for x in ("attention", "just a moment", "access denied", "blocked", "verify")
    )
    return (
        code == 200
        and len(body) > 10_000
        and "cargurus" in low
        and not strong_hits
        and not suspicious_title
    )


def _print_probe_result(label: str, code: int, body: bytes, ctype: str | None) -> bool:
    n = len(body)
    text = body.decode("utf-8", errors="ignore")
    low = text.lower()
    snippet = text[:500].replace("\n", " ")
    title = _html_title(low)
    strong_hits = [h for h in STRONG_BLOCK_HINTS if h in low]

    print(f"\n--- {label} ---")
    print(f"  HTTP status:  {code}")
    print(f"  Content-Type: {ctype}")
    print(f"  Body bytes:   {n}")
    if title:
        print(f"  <title>:      {title!r}")
    if n > 0:
        print(f"  Body start:   {snippet!r}...")

    ok = _probe_ok(code, body, low, title)
    if ok:
        print("  Probe OK: 200, large HTML, normal-looking title.")
    elif code in (403, 406, 429, 451):
        print(f"  Probe FAIL: HTTP {code} (common for Python/urllib on GCP — see verdict).")
    elif code == 200 and n < 3000:
        print("  Probe suspicious: 200 but very small body.")
    else:
        print(f"  Probe unclear: HTTP {code}, {n} bytes.")
    if strong_hits:
        print(f"  WAF phrases: {strong_hits[:6]}{'…' if len(strong_hits) > 6 else ''}")
    return ok


def main() -> int:
    print("=== 1) Your public IP / country ===")
    geo, src = fetch_public_geo()
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
        if cc and cc != "US" and cc != "?":
            print(
                "\n  Note: CarGurus listing/dealer URLs sometimes behave badly for non-US egress.\n"
                "        Homepage 200 does not guarantee search URLs work — compare with main.py runs."
            )

    print("\n=== 2) GET CarGurus homepage ===")

    urllib_ok = False
    try:
        code_u, body_u, ctype_u = _get_urllib(CARGURUS_URL, HEADERS, timeout=20.0)
        urllib_ok = _print_probe_result("stdlib urllib (Python TLS)", code_u, body_u, ctype_u)
    except Exception as e:
        print(f"\n--- stdlib urllib ---\n  Request failed: {e}")

    cffi_ok = False
    cffi_note: str | None = None
    try:
        code_c, body_c, ctype_c, imp = _get_curl_cffi(CARGURUS_URL, HEADERS, timeout=20.0)
        cffi_ok = _print_probe_result(f"curl_cffi impersonate={imp}", code_c, body_c, ctype_c)
    except ImportError:
        cffi_note = "curl_cffi not installed — pip install curl_cffi"
    except Exception as e:
        cffi_note = f"curl_cffi probe failed: {e}"

    if cffi_note:
        print(f"\n--- curl_cffi ---\n  Skipped: {cffi_note}")

    isp = (geo or {}).get("isp") or ""
    is_cloud = any(
        x in isp.lower()
        for x in ("google cloud", "google llc", "amazon", "microsoft azure", "digitalocean", "ovh")
    )

    print("\n=== 3) Verdict ===")
    if urllib_ok or cffi_ok:
        if cffi_ok and not urllib_ok:
            print(
                "  Crawler path: use curl_cffi (main.py default when installed).\n"
                "  urllib/requests alone will likely keep returning 406 on this host."
            )
        else:
            print("  At least one probe succeeded — crawler HTTP should work with that client.")
        print("  (Ignore random 'captcha'/'403' in huge pages — often legal/script text.)")
        return 0

    if is_cloud:
        print(
            "  Datacenter egress (e.g. Google Cloud) + script TLS often gets HTTP 406 while\n"
            "  Chrome on the same VM works — the site allows browsers, not Python's SSL fingerprint."
        )
    if cffi_note and "not installed" in cffi_note:
        print(f"  Next step: {cffi_note} then re-run this script and ensure main.py logs")
        print("  'HTTP client: curl_cffi impersonate=...'.")
    elif not cffi_ok and not cffi_note:
        print("  Both probes failed — try a US residential proxy or non-cloud host.")
    else:
        print("  Probes failed — review HTTP status and body snippets above.")

    return 2


if __name__ == "__main__":
    sys.exit(main())
