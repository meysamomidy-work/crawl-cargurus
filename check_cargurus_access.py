#!/usr/bin/env python3
"""
Quick diagnostic: public IP + country, then a GET to CarGurus.
Run on the same machine (or in the same Docker network) as your crawler.

  python check_cargurus_access.py

Uses only the standard library (no pip install).

Note: ip-api.com often returns HTTP 403 from cloud/datacenter IPs; this script
falls back to other read-only geo endpoints.

If CarGurus returns 403/406/451 or a tiny body from a non-US IP, a US VPN/proxy
may be required for listing/dealer URLs even when the homepage loads.
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

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "identity",
    "Upgrade-Insecure-Requests": "1",
}

_MIN_UA = {"User-Agent": HEADERS["User-Agent"]}


def _get(url: str, headers: dict[str, str], timeout: float = 20.0) -> tuple[int, bytes, str | None]:
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
            code, raw, _ = _get(url, _MIN_UA, timeout=15.0)
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
    try:
        code, body, ctype = _get(CARGURUS_URL, HEADERS, timeout=20.0)
    except Exception as e:
        print(f"  Request failed: {e}")
        return 1

    n = len(body)
    text = body.decode("utf-8", errors="ignore")
    low = text.lower()
    snippet = text[:500].replace("\n", " ")
    title = _html_title(low)

    print(f"  HTTP status:  {code}")
    print(f"  Content-Type: {ctype}")
    print(f"  Body bytes:   {n}")
    if title:
        print(f"  <title>:      {title!r}")
    if n > 0:
        print(f"  Body start:   {snippet!r}...")

    strong_hits = [h for h in STRONG_BLOCK_HINTS if h in low]
    suspicious_title = any(
        x in title.lower()
        for x in ("attention", "just a moment", "access denied", "blocked", "verify")
    )

    print("\n=== 3) Verdict ===")
    if code == 200 and n > 10_000 and "cargurus" in low and not strong_hits and not suspicious_title:
        print("  Looks OK: 200, large HTML, normal-looking title, no strong WAF phrases.")
        print("  (Ignore random 'captcha'/'403' mentions inside huge pages — those are often scripts/legal text.)")
    elif code == 200 and n < 3000:
        print("  Suspicious: 200 but very small body — might be a challenge or stub page.")
    elif code in (403, 406, 429, 451):
        print(f"  Likely blocked or not acceptable from this network (HTTP {code}).")
    else:
        print(f"  Mixed or unclear (HTTP {code}, {n} bytes). Review title and hints below.")

    if strong_hits:
        print(f"  Strong WAF/challenge phrases found: {strong_hits[:8]}{'…' if len(strong_hits) > 8 else ''}")
    if suspicious_title and not (code == 200 and n > 10_000 and not strong_hits):
        print(f"  Suspicious <title> for a real homepage: {title!r}")

    ok = code == 200 and n > 10_000 and "cargurus" in low and not strong_hits and not suspicious_title
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
