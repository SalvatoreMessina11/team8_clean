"""Audit completeness and usability of downloaded GLD market surfaces.

This script does NOT connect to IBKR and does NOT download anything.
It checks the market data already present in the project.

What it verifies
----------------
- expected GLD trading dates from data/processed/gld_daily_history.csv
- presence of adaptive/full/historical surface files
- number of eligible rows
- unique expiries and strikes
- DTE >= 75 compliance
- duplicate (T, K) nodes
- missing/non-finite required fields
- whether each date is:
    COMPLETE_CC      -> >=64 eligible observations, >=3 expiries
    USABLE_SPARSE    -> 8..63 observations, >=3 expiries
    INSUFFICIENT     -> some data exist but not enough for calibration
    EMPTY            -> file exists but contains 0 rows
    MISSING_FILE     -> trading date has no surface file

Important:
The current adaptive IBKR acquisition is NOT the literal full option chain.
So this audit checks whether the data are complete enough for the Team 8
methodology, not whether every listed GLD option contract in the market
was downloaded.

Default folders
---------------
data/processed/gld_daily_history.csv
data/processed/full_surfaces/
data/processed/sparse_historical_surfaces/

Output
------
outputs/market_data_audit.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


SURFACE_RE = re.compile(
    r"GLD_(\d{4}-\d{2}-\d{2})_eligible_"
    r"(adaptive|historical|full|bin_balanced|all_real)_surface\.csv$"
)

PRIORITY = {
    "full": 1,
    "historical": 2,
    "adaptive": 3,
    "bin_balanced": 4,
    "all_real": 5,
}

REQUIRED_COLUMNS = {
    "K",
    "T",
    "price",
    "rate",
    "vega",
    "implied_vol",
}


def load_trading_dates(stock_path: Path) -> pd.DatetimeIndex:
    if not stock_path.exists():
        raise FileNotFoundError(f"GLD history not found: {stock_path}")

    df = pd.read_csv(stock_path)

    date_col = None
    for c in ("timestamp", "date", "Date"):
        if c in df.columns:
            date_col = c
            break

    if date_col is None:
        raise ValueError(
            f"Could not find a date column in {stock_path}. "
            f"Columns: {list(df.columns)}"
        )

    dates = pd.to_datetime(df[date_col], errors="coerce")
    if getattr(dates.dt, "tz", None) is not None:
        dates = dates.dt.tz_localize(None)

    dates = dates.dt.normalize().dropna().drop_duplicates().sort_values()
    return pd.DatetimeIndex(dates)


def discover_surface_files(
    *dirs: Path,
    min_dte: int = 75,
) -> dict[pd.Timestamp, tuple[str, Path]]:
    """Choose the richest valid REAL surface per date after DTE filtering."""
    candidates: dict[pd.Timestamp, list[tuple[str, Path]]] = {}

    for directory in dirs:
        if not directory.exists():
            continue

        for path in directory.glob("GLD_*_eligible_*_surface.csv"):
            m = SURFACE_RE.fullmatch(path.name)
            if not m:
                continue
            date = pd.Timestamp(m.group(1)).normalize()
            kind = m.group(2)
            candidates.setdefault(date, []).append((kind, path))

    best: dict[pd.Timestamp, tuple[str, Path]] = {}

    for date, options in candidates.items():
        scored = []
        for kind, path in options:
            try:
                df = pd.read_csv(path)
            except (pd.errors.EmptyDataError, OSError, ValueError):
                continue
            if df.empty or not {"T", "K"}.issubset(df.columns):
                continue

            T = pd.to_numeric(df["T"], errors="coerce")
            K = pd.to_numeric(df["K"], errors="coerce")
            if "dte" in df.columns:
                dte = pd.to_numeric(df["dte"], errors="coerce")
            else:
                dte = 365.25 * T

            mask = (
                T.gt(0)
                & K.gt(0)
                & dte.ge(float(min_dte))
            )
            eligible = df.loc[mask].copy()
            if eligible.empty:
                continue
            rows = int(
                eligible.drop_duplicates(["T", "K"], keep="last").shape[0]
            )
            if "expiry" in eligible.columns:
                expiries = int(
                    pd.to_datetime(eligible["expiry"], errors="coerce")
                    .dropna().nunique()
                )
            else:
                expiries = int(
                    pd.to_numeric(eligible["T"], errors="coerce")
                    .dropna().round(10).nunique()
                )
            scored.append((rows, expiries, PRIORITY.get(kind, 0), kind, path))

        if scored:
            scored.sort(key=lambda x: x[:3], reverse=True)
            _, _, _, kind, path = scored[0]
            best[date] = (kind, path)

    return best

def raw_checkpoint_path(full_dir: Path, date: pd.Timestamp) -> Path:
    return full_dir / f"GLD_{date.strftime('%Y-%m-%d')}_midpoint_raw.csv"


def safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def inspect_surface(
    date: pd.Timestamp,
    kind: str | None,
    path: Path | None,
    full_dir: Path,
    min_dte: int,
    cc_size: int,
    min_rows: int,
    min_expiries: int,
) -> dict:
    slug = date.strftime("%Y-%m-%d")
    raw_path = raw_checkpoint_path(full_dir, date)

    raw_rows = 0
    if raw_path.exists():
        raw = safe_read_csv(raw_path)
        raw_rows = int(len(raw))

    if path is None:
        return {
            "date": slug,
            "status": "MISSING_FILE",
            "surface_kind": "",
            "surface_file": "",
            "rows": 0,
            "raw_rows": raw_rows,
            "expiries": 0,
            "strikes": 0,
            "min_dte": np.nan,
            "max_dte": np.nan,
            "duplicates_TK": 0,
            "bad_required_values": 0,
            "below_min_dte": 0,
            "enough64": False,
            "usable_for_calibration": False,
        }

    df = safe_read_csv(path)

    if df.empty:
        return {
            "date": slug,
            "status": "EMPTY",
            "surface_kind": kind,
            "surface_file": str(path),
            "rows": 0,
            "raw_rows": raw_rows,
            "expiries": 0,
            "strikes": 0,
            "min_dte": np.nan,
            "max_dte": np.nan,
            "duplicates_TK": 0,
            "bad_required_values": 0,
            "below_min_dte": 0,
            "enough64": False,
            "usable_for_calibration": False,
        }

    missing_cols = sorted(REQUIRED_COLUMNS.difference(df.columns))
    if missing_cols:
        return {
            "date": slug,
            "status": "BAD_SCHEMA",
            "surface_kind": kind,
            "surface_file": str(path),
            "rows": int(len(df)),
            "raw_rows": raw_rows,
            "expiries": 0,
            "strikes": 0,
            "min_dte": np.nan,
            "max_dte": np.nan,
            "duplicates_TK": 0,
            "bad_required_values": np.nan,
            "below_min_dte": np.nan,
            "enough64": False,
            "usable_for_calibration": False,
            "missing_columns": ",".join(missing_cols),
        }

    work = df.copy()

    numeric = ["K", "T", "price", "rate", "vega", "implied_vol"]
    if "dte" in work.columns:
        numeric.append("dte")

    for c in numeric:
        work[c] = pd.to_numeric(work[c], errors="coerce")

    if "dte" in work.columns:
        all_dte = work["dte"]
    else:
        all_dte = 365.25 * work["T"]

    # DTE<min_dte observations are outside the official domain, not a quality
    # failure. Count them for diagnostics and remove them before assessing the
    # surface used by calibration/OOS.
    below_min_dte = int((all_dte < float(min_dte)).fillna(True).sum())
    domain_mask = all_dte.ge(float(min_dte)).fillna(False)
    work = work.loc[domain_mask].copy()
    dte = all_dte.loc[work.index]

    if work.empty:
        return {
            "date": slug,
            "status": "INSUFFICIENT",
            "surface_kind": kind,
            "surface_file": str(path),
            "rows": 0,
            "raw_rows": raw_rows,
            "expiries": 0,
            "strikes": 0,
            "min_dte": np.nan,
            "max_dte": np.nan,
            "duplicates_TK": 0,
            "bad_required_values": 0,
            "below_min_dte": below_min_dte,
            "enough64": False,
            "usable_for_calibration": False,
        }

    required_matrix = work[
        ["K", "T", "price", "rate", "vega", "implied_vol"]
    ].to_numpy(dtype=float)
    bad_required = int((~np.isfinite(required_matrix)).any(axis=1).sum())

    duplicates = int(work.duplicated(["T", "K"], keep=False).sum())

    expiries = 0
    if "expiry" in work.columns:
        expiries = int(
            pd.to_datetime(work["expiry"], errors="coerce")
            .dropna()
            .nunique()
        )
    else:
        expiries = int(work["T"].round(10).nunique())

    strikes = int(work["K"].dropna().nunique())
    rows = int(len(work))

    quality_ok = (
        bad_required == 0
        and duplicates == 0
    )

    enough64 = rows >= int(cc_size)
    usable = (
        rows >= int(min_rows)
        and expiries >= int(min_expiries)
        and quality_ok
    )

    if not quality_ok:
        status = "QUALITY_PROBLEM"
    elif enough64 and expiries >= int(min_expiries):
        status = "COMPLETE_CC"
    elif usable:
        status = "USABLE_SPARSE"
    else:
        status = "INSUFFICIENT"

    return {
        "date": slug,
        "status": status,
        "surface_kind": kind,
        "surface_file": str(path),
        "rows": rows,
        "raw_rows": raw_rows,
        "expiries": expiries,
        "strikes": strikes,
        "min_dte": float(dte.min()) if dte.notna().any() else np.nan,
        "max_dte": float(dte.max()) if dte.notna().any() else np.nan,
        "duplicates_TK": duplicates,
        "bad_required_values": bad_required,
        "below_min_dte": below_min_dte,
        "enough64": enough64,
        "usable_for_calibration": usable,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)

    p.add_argument(
        "--stock",
        default="data/processed/gld_daily_history.csv",
    )
    p.add_argument(
        "--full-dir",
        default="data/processed/full_surfaces",
    )
    p.add_argument(
        "--sparse-dir",
        default="data/processed/sparse_historical_surfaces",
    )
    p.add_argument(
        "--output",
        default="outputs/market_data_audit.csv",
    )

    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)

    p.add_argument("--min-dte", type=int, default=75)
    p.add_argument("--cc-size", type=int, default=64)
    p.add_argument("--min-rows", type=int, default=8)
    p.add_argument("--min-expiries", type=int, default=3)

    args = p.parse_args()

    stock_path = Path(args.stock)
    full_dir = Path(args.full_dir)
    sparse_dir = Path(args.sparse_dir)
    output = Path(args.output)

    trading_dates = load_trading_dates(stock_path)

    if args.start is not None:
        trading_dates = trading_dates[
            trading_dates >= pd.Timestamp(args.start).normalize()
        ]
    if args.end is not None:
        trading_dates = trading_dates[
            trading_dates <= pd.Timestamp(args.end).normalize()
        ]

    surfaces = discover_surface_files(full_dir, sparse_dir, min_dte=args.min_dte)

    rows = []
    for date in trading_dates:
        entry = surfaces.get(date)

        if entry is None:
            kind = None
            path = None
        else:
            kind, path = entry

        rows.append(
            inspect_surface(
                date=date,
                kind=kind,
                path=path,
                full_dir=full_dir,
                min_dte=args.min_dte,
                cc_size=args.cc_size,
                min_rows=args.min_rows,
                min_expiries=args.min_expiries,
            )
        )

    audit = pd.DataFrame(rows).sort_values(
        "date",
        ascending=False,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output, index=False)

    total = len(audit)
    complete = int(audit["status"].eq("COMPLETE_CC").sum())
    sparse = int(audit["status"].eq("USABLE_SPARSE").sum())
    insufficient = int(audit["status"].eq("INSUFFICIENT").sum())
    empty = int(audit["status"].eq("EMPTY").sum())
    missing = int(audit["status"].eq("MISSING_FILE").sum())
    quality = int(
        audit["status"].isin(["QUALITY_PROBLEM", "BAD_SCHEMA"]).sum()
    )

    print("=" * 100)
    print("TEAM 8 MARKET DATA AUDIT")
    print("=" * 100)
    print(f"Trading dates checked     : {total}")
    print(f"COMPLETE_CC (>=64)        : {complete}")
    print(f"USABLE_SPARSE (8..63)     : {sparse}")
    print(f"INSUFFICIENT              : {insufficient}")
    print(f"EMPTY surface files       : {empty}")
    print(f"MISSING surface files     : {missing}")
    print(f"QUALITY / schema problems : {quality}")
    print(f"Audit CSV                 : {output}")
    print("=" * 100)

    problem = audit.loc[
        ~audit["status"].isin(["COMPLETE_CC", "USABLE_SPARSE"]),
        [
            "date",
            "status",
            "rows",
            "raw_rows",
            "expiries",
            "strikes",
        ],
    ]

    if problem.empty:
        print("[OK] Every checked trading date is usable for the project.")
    else:
        print()
        print("DATES TO REVIEW")
        print(problem.to_string(index=False))

    print()
    print(
        "NOTE: COMPLETE_CC means complete enough for the official Team 8 "
        "64-point CC rule. It does NOT mean that every GLD option contract "
        "listed by IBKR was downloaded, because the acquisition is adaptive."
    )


if __name__ == "__main__":
    main()
