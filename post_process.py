#!/usr/bin/env python3
"""
Post-process CarGurus crawl output (crawled/<State>.csv).

  python post_process.py
  python post_process.py --input crawled --output excels

Steps:
  - Load all state CSVs (pipe-delimited)
  - Clean addresses (remove distance suffix like "(12 mi)")
  - Parse address → US state; move rows into the correct state bucket
  - Drop duplicate dealerships (by dealer page URL, then name + address)
  - Drop the State column from output
  - Write excels/<State>.xlsx per state (sheets by inventory: 0-20, 20-40, …, 100+)
  - Write excels/all_dealerships.xlsx (same sheet layout)
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

try:
    import usaddress
except ImportError:
    print("Install dependencies: pip install pandas openpyxl usaddress", file=sys.stderr)
    raise

# Columns written by main.py (State is dropped in output).
INPUT_COLUMNS = [
    "Name",
    "Dealer Page Link",
    "List Address",
    "Phone",
    "Website",
    "Inventory Count",
    "Score",
    "Review Count",
    "Business Hours",
    "State",
]

OUTPUT_COLUMNS = [c for c in INPUT_COLUMNS if c != "State"]

US_STATE_ABBR_TO_NAME: dict[str, str] = {
    "AL": "Alabama",
    "AK": "Alaska",
    "AZ": "Arizona",
    "AR": "Arkansas",
    "CA": "California",
    "CO": "Colorado",
    "CT": "Connecticut",
    "DE": "Delaware",
    "FL": "Florida",
    "GA": "Georgia",
    "HI": "Hawaii",
    "ID": "Idaho",
    "IL": "Illinois",
    "IN": "Indiana",
    "IA": "Iowa",
    "KS": "Kansas",
    "KY": "Kentucky",
    "LA": "Louisiana",
    "ME": "Maine",
    "MD": "Maryland",
    "MA": "Massachusetts",
    "MI": "Michigan",
    "MN": "Minnesota",
    "MS": "Mississippi",
    "MO": "Missouri",
    "MT": "Montana",
    "NE": "Nebraska",
    "NV": "Nevada",
    "NH": "New Hampshire",
    "NJ": "New Jersey",
    "NM": "New Mexico",
    "NY": "New York",
    "NC": "North Carolina",
    "ND": "North Dakota",
    "OH": "Ohio",
    "OK": "Oklahoma",
    "OR": "Oregon",
    "PA": "Pennsylvania",
    "RI": "Rhode Island",
    "SC": "South Carolina",
    "SD": "South Dakota",
    "TN": "Tennessee",
    "TX": "Texas",
    "UT": "Utah",
    "VT": "Vermont",
    "VA": "Virginia",
    "WA": "Washington",
    "WV": "West Virginia",
    "WI": "Wisconsin",
    "WY": "Wyoming",
    "DC": "District of Columbia",
    "AS": "American Samoa",
    "GU": "Guam",
    "MP": "Northern Mariana Islands",
}

# Canonical title-case names → themselves; aliases normalized here.
US_STATE_NAME_CANONICAL: dict[str, str] = {v: v for v in US_STATE_ABBR_TO_NAME.values()}
US_STATE_NAME_CANONICAL.update(
    {
        "newhampshire": "New Hampshire",
        "newmexico": "New Mexico",
        "newyork": "New York",
        "northcarolina": "North Carolina",
        "northdakota": "North Dakota",
        "rhodeisland": "Rhode Island",
        "southcarolina": "South Carolina",
        "southdakota": "South Dakota",
        "westvirginia": "West Virginia",
        "districtofcolumbia": "District of Columbia",
        "northernmarianaislands": "Northern Mariana Islands",
        "americansamoa": "American Samoa",
    }
)

_MI_PATTERN = re.compile(r"\s*\(\s*\d+\s*mi\s*\)", re.IGNORECASE)
_STATE_ABBR_TAIL = re.compile(
    r",\s*([A-Z]{2})(?:\s+\d{5}(?:-\d{4})?)?\s*$",
    re.IGNORECASE,
)
_ZIP_STATE = re.compile(r"\b([A-Z]{2})\s+\d{5}(?:-\d{4})?\b")
_INVENTORY_DIGITS = re.compile(r"(\d+)")

# Sheet names and upper bounds (inclusive). 100+ is everything above 100.
INVENTORY_BUCKET_BOUNDS: list[tuple[str, int | None]] = [
    ("0-20", 20),
    ("20-40", 40),
    ("40-60", 60),
    ("60-80", 80),
    ("80-100", 100),
    ("100+", None),
]


def _normalize_key(name: str) -> str:
    return re.sub(r"[^a-z]", "", name.lower())


def canonical_state_name(raw: str | None) -> str | None:
    if not raw or not str(raw).strip():
        return None
    s = str(raw).strip()
    if len(s) == 2:
        return US_STATE_ABBR_TO_NAME.get(s.upper())
    key = _normalize_key(s)
    if key in US_STATE_NAME_CANONICAL:
        return US_STATE_NAME_CANONICAL[key]
    # Title-case fallback match
    for canon in US_STATE_NAME_CANONICAL.values():
        if _normalize_key(canon) == key:
            return canon
    return s if len(s) > 2 else None


def parse_state_from_address(address: str) -> str | None:
    if not address or not str(address).strip():
        return None
    addr = str(address).strip()
    try:
        tagged, _ = usaddress.tag(addr)
        state_token = tagged.get("StateName")
        if state_token:
            found = canonical_state_name(state_token)
            if found:
                return found
    except usaddress.RepeatedLabelError:
        pass
    except Exception:
        pass

    m = _STATE_ABBR_TAIL.search(addr)
    if m:
        found = canonical_state_name(m.group(1))
        if found:
            return found

    m = _ZIP_STATE.search(addr)
    if m:
        found = canonical_state_name(m.group(1))
        if found:
            return found

    # Last resort: look for full state name substring (longest names first).
    norm_addr = _normalize_key(addr)
    for canon in sorted(US_STATE_NAME_CANONICAL.values(), key=len, reverse=True):
        if _normalize_key(canon) in norm_addr:
            return canon
    return None


def clean_address(address: str) -> str:
    if pd.isna(address):
        return ""
    text = str(address).strip()
    text = _MI_PATTERN.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def clean_name(name: str) -> str:
    if pd.isna(name):
        return ""
    return re.sub(r"[\n\t]+", " ", str(name)).strip()


def load_crawled_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep="|",
        dtype=str,
        keep_default_na=False,
        encoding="utf-8",
        on_bad_lines="warn",
    )
    # Align columns if header row missing or short rows.
    if list(df.columns) != INPUT_COLUMNS:
        if len(df.columns) == len(INPUT_COLUMNS):
            df.columns = INPUT_COLUMNS
        else:
            df = pd.read_csv(
                path,
                sep="|",
                names=INPUT_COLUMNS,
                header=0,
                dtype=str,
                keep_default_na=False,
                encoding="utf-8",
                on_bad_lines="warn",
            )
    for col in INPUT_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[INPUT_COLUMNS].copy()


def parse_inventory_count(value: object) -> int:
    """Parse Inventory Count to a non-negative integer; unparseable → 0."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return 0
    if isinstance(value, (int, float)) and not pd.isna(value):
        return max(0, int(value))
    text = str(value).strip().replace(",", "")
    if not text or text.lower() in ("not specified", "n/a", "na", "-", "none"):
        return 0
    m = _INVENTORY_DIGITS.search(text)
    return max(0, int(m.group(1))) if m else 0


def inventory_bucket_mask(counts: pd.Series, sheet_name: str) -> pd.Series:
    """Non-overlapping buckets: 0-20 includes 20; 20-40 is 21-40; …; 100+ is >100."""
    c = counts.astype(int)
    if sheet_name == "0-20":
        return c <= 20
    if sheet_name == "20-40":
        return (c > 20) & (c <= 40)
    if sheet_name == "40-60":
        return (c > 40) & (c <= 60)
    if sheet_name == "60-80":
        return (c > 60) & (c <= 80)
    if sheet_name == "80-100":
        return (c > 80) & (c <= 100)
    if sheet_name == "100+":
        return c > 100
    raise ValueError(f"Unknown inventory bucket: {sheet_name}")


def write_inventory_workbook(path: Path, df: pd.DataFrame) -> None:
    """One Excel file; one sheet per inventory range."""
    if df.empty:
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            for sheet_name, _ in INVENTORY_BUCKET_BOUNDS:
                pd.DataFrame(columns=OUTPUT_COLUMNS).to_excel(
                    writer, sheet_name=sheet_name, index=False
                )
        return

    counts = df["Inventory Count"].map(parse_inventory_count)
    df = df.copy()
    df["_inv_sort"] = counts

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, _ in INVENTORY_BUCKET_BOUNDS:
            mask = inventory_bucket_mask(counts, sheet_name)
            part = df.loc[mask].sort_values("_inv_sort", ascending=False, kind="stable")
            part = part.drop(columns=["_inv_sort"], errors="ignore")
            part.to_excel(writer, sheet_name=sheet_name, index=False)


def dedupe_dealerships(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    seen_links: set[str] = set()
    seen_fallback: set[str] = set()
    kept: list[dict] = []
    for row in df.to_dict(orient="records"):
        link = str(row.get("Dealer Page Link", "")).strip().lower()
        fallback = (
            f"{str(row.get('Name', '')).strip().lower()}|"
            f"{str(row.get('List Address', '')).strip().lower()}"
        )
        if link:
            if link in seen_links:
                continue
            seen_links.add(link)
            kept.append(row)
        elif fallback and fallback != "|":
            if fallback in seen_fallback:
                continue
            seen_fallback.add(fallback)
            kept.append(row)
        else:
            kept.append(row)
    return pd.DataFrame(kept, columns=OUTPUT_COLUMNS)


def process(
    input_dir: Path,
    output_dir: Path,
) -> pd.DataFrame:
    csv_files = sorted(
        p for p in input_dir.glob("*.csv") if p.is_file() and p.parent.name != "meta"
    )
    if not csv_files:
        raise FileNotFoundError(f"No CSV files in {input_dir}")

    frames: list[pd.DataFrame] = []
    for path in csv_files:
        file_state = path.stem
        df = load_crawled_csv(path)
        df["_source_file_state"] = file_state
        frames.append(df)

    raw = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(raw):,} rows from {len(csv_files)} file(s)")

    raw["Name"] = raw["Name"].map(clean_name)
    raw["List Address"] = raw["List Address"].map(clean_address)
    raw["_parsed_state"] = raw["List Address"].map(parse_state_from_address)

    # Move to correct state: parsed address wins; else keep CSV State; else source file.
    def resolve_state(row: pd.Series) -> str:
        parsed = row.get("_parsed_state")
        if parsed:
            return parsed
        from_col = canonical_state_name(row.get("State"))
        if from_col:
            return from_col
        from_file = canonical_state_name(row.get("_source_file_state"))
        if from_file:
            return from_file
        return "Unknown"

    raw["_final_state"] = raw.apply(resolve_state, axis=1)

    moved = (
        raw["_final_state"]
        != raw["_source_file_state"].map(lambda s: canonical_state_name(s) or s)
    ).sum()
    unknown = (raw["_final_state"] == "Unknown").sum()
    print(f"Reassigned {moved:,} row(s) to a different state than their CSV file")
    if unknown:
        print(f"Warning: {unknown:,} row(s) could not be assigned a state → Unknown")

    work = raw.drop(columns=["State", "_parsed_state", "_source_file_state"], errors="ignore")
    work = work.rename(columns={"_final_state": "_state_bucket"})

    buckets: dict[str, pd.DataFrame] = {}
    for state_name, group in work.groupby("_state_bucket", sort=True):
        cleaned = group.drop(columns=["_state_bucket"], errors="ignore")
        cleaned = cleaned[OUTPUT_COLUMNS]
        cleaned = dedupe_dealerships(cleaned)
        buckets[state_name] = cleaned
        print(f"  {state_name}: {len(cleaned):,} dealer(s) after dedupe")

    output_dir.mkdir(parents=True, exist_ok=True)

    per_state_frames: list[pd.DataFrame] = []
    for state_name, df in sorted(buckets.items()):
        safe = re.sub(r'[<>:"/\\|?*]', "_", state_name)
        out_path = output_dir / f"{safe}.xlsx"
        write_inventory_workbook(out_path, df)
        inv = df["Inventory Count"].map(parse_inventory_count)
        sheet_counts = {
            name: int(inventory_bucket_mask(inv, name).sum())
            for name, _ in INVENTORY_BUCKET_BOUNDS
        }
        print(f"    → {out_path.name} sheets: {sheet_counts}")
        per_state_frames.append(df)

    combined = pd.concat(per_state_frames, ignore_index=True) if per_state_frames else pd.DataFrame(
        columns=OUTPUT_COLUMNS
    )
    combined = dedupe_dealerships(combined)
    combined_path = output_dir / "all_dealerships.xlsx"
    write_inventory_workbook(combined_path, combined)

    print(
        f"\nWrote {len(buckets)} state workbook(s) + {combined_path.name} "
        f"({len(combined):,} rows, 6 inventory sheets each)"
    )
    return combined


def main() -> int:
    parser = argparse.ArgumentParser(description="Post-process crawled CarGurus CSV data")
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=Path("crawled"),
        help="Directory with <State>.csv crawl output",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("excels"),
        help="Directory for cleaned Excel files",
    )
    args = parser.parse_args()

    if not args.input.is_dir():
        print(f"Input directory not found: {args.input}", file=sys.stderr)
        return 1

    try:
        process(args.input, args.output)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
