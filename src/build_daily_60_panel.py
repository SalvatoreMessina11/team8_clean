"""
Build a recent DAILY GLD option panel from the raw IBKR historical download.

This script does NOT contact IBKR. It reuses:
    data/processed/options_GLD_active_contract_intraday.parquet
    data/processed/gld_daily_history.csv

Default study:
    end date = 2026-09-02
    last 60 available GLD trading sessions
    -> at most 60 daily surfaces and 59 t -> t+1 OOS forecasts.

Why this is separate from ibkr_gld_weekly_data.py
-------------------------------------------------
The current IBKR fetch can keep running untouched. Once it finishes, the same
raw historical bars can be transformed into BOTH:
  * the existing weekly panel, and
  * this recent daily panel.

No re-download is required.

Important:
IBKR cannot reconstruct already-expired option contracts. Therefore the panel
still needs a date-by-date coverage audit. The daily route improves coverage
because it concentrates on the most recent observations, where a larger share
of contracts can still be active today.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_stock_dates(stock_path: Path, end: str, sessions: int):
    stock = pd.read_csv(stock_path)
    required = {"timestamp", "close"}
    missing = required.difference(stock.columns)
    if missing:
        raise ValueError(f"Stock file missing columns: {sorted(missing)}")

    stock["date"] = pd.to_datetime(stock["timestamp"], errors="coerce").dt.normalize()
    stock["spot"] = pd.to_numeric(stock["close"], errors="coerce")
    stock = stock.dropna(subset=["date", "spot"])
    stock = stock.loc[stock["spot"].gt(0.0)].copy()
    stock = stock.loc[stock["date"].le(pd.Timestamp(end))].copy()
    stock = (
        stock.sort_values("date")
        .drop_duplicates("date", keep="last")
        .tail(int(sessions))
        .reset_index(drop=True)
    )
    if stock.empty:
        raise ValueError("No stock observations available for the requested window.")
    return stock[["date", "spot"]]


def build_daily_panel(
    raw_path: Path,
    stock_path: Path,
    end: str = "2026-09-02",
    sessions: int = 60,
    min_moneyness: float = 0.85,
    max_moneyness: float = 1.20,
    min_dte: int = 21,
    max_dte: int = 730,
    min_price: float = 0.10,
):
    raw = pd.read_parquet(raw_path)
    required = {
        "timestamp", "expiry", "opt_type", "strike", "close", "volume", "conId"
    }
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Raw option file missing columns: {sorted(missing)}")

    stock = load_stock_dates(stock_path, end=end, sessions=sessions)
    selected_dates = set(stock["date"])

    # Intraday historical bars are actual instants, so converting UTC timestamps
    # to US/Eastern before extracting the trading date is appropriate here.
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], errors="coerce", utc=True)
    raw["date"] = (
        raw["timestamp"]
        .dt.tz_convert("US/Eastern")
        .dt.tz_localize(None)
        .dt.normalize()
    )
    raw["expiry"] = pd.to_datetime(raw["expiry"], errors="coerce").dt.normalize()
    raw["strike"] = pd.to_numeric(raw["strike"], errors="coerce")
    raw["close"] = pd.to_numeric(raw["close"], errors="coerce")
    raw["volume"] = pd.to_numeric(raw["volume"], errors="coerce").fillna(0.0)

    raw = raw.dropna(subset=["timestamp", "date", "expiry", "strike", "close"])
    raw = raw.loc[raw["date"].isin(selected_dates)].copy()
    raw = raw.loc[raw["opt_type"].astype(str).str.upper().eq("C")].copy()

    # Last available historical bar for each contract on each trading date.
    daily = (
        raw.sort_values("timestamp")
        .groupby(["conId", "date"], sort=True, as_index=False)
        .tail(1)
        .copy()
    )

    daily = daily.merge(stock, on="date", how="inner")
    daily["dte"] = (daily["expiry"] - daily["date"]).dt.days
    daily["T"] = daily["dte"] / 365.25
    daily["moneyness"] = daily["strike"] / daily["spot"]

    # Basic economic validity only. Liquidity thresholds are reported separately
    # in the coverage file so we can decide them after seeing the data.
    daily = daily.loc[
        daily["strike"].gt(0.0)
        & daily["close"].gt(float(min_price))
        & daily["dte"].between(int(min_dte), int(max_dte))
        & daily["moneyness"].between(float(min_moneyness), float(max_moneyness))
    ].copy()

    daily["ts"] = daily["date"]
    daily["expiry"] = daily["expiry"].dt.strftime("%Y-%m-%d")

    preferred = [
        "ts", "date", "expiry", "opt_type", "strike",
        "open", "high", "low", "close", "volume", "bar_count",
        "spot", "dte", "T", "moneyness",
        "conId", "localSymbol", "tradingClass",
    ]
    columns = [c for c in preferred if c in daily.columns]
    daily = daily[columns].sort_values(
        ["date", "expiry", "strike", "conId"]
    ).reset_index(drop=True)

    return daily, stock


def coverage_report(panel: pd.DataFrame, stock: pd.DataFrame):
    base = stock[["date", "spot"]].copy()

    if panel.empty:
        for c in [
            "rows", "expiries", "strikes", "rows_volume_ge_1",
            "rows_volume_ge_25", "rows_core_moneyness",
            "min_dte", "max_dte",
        ]:
            base[c] = 0
        return base

    tmp = panel.copy()
    tmp["expiry_dt"] = pd.to_datetime(tmp["expiry"])
    tmp["date"] = pd.to_datetime(tmp["date"]).dt.normalize()

    grouped = (
        tmp.groupby("date")
        .agg(
            rows=("strike", "size"),
            expiries=("expiry", "nunique"),
            strikes=("strike", "nunique"),
            min_dte=("dte", "min"),
            max_dte=("dte", "max"),
        )
        .reset_index()
    )

    volume1 = (
        tmp.loc[tmp["volume"].ge(1.0)]
        .groupby("date")
        .size()
        .rename("rows_volume_ge_1")
        .reset_index()
    )
    volume25 = (
        tmp.loc[tmp["volume"].ge(25.0)]
        .groupby("date")
        .size()
        .rename("rows_volume_ge_25")
        .reset_index()
    )
    core = (
        tmp.loc[tmp["moneyness"].between(0.90, 1.10)]
        .groupby("date")
        .size()
        .rename("rows_core_moneyness")
        .reset_index()
    )

    report = (
        base.merge(grouped, on="date", how="left")
        .merge(volume1, on="date", how="left")
        .merge(volume25, on="date", how="left")
        .merge(core, on="date", how="left")
    )
    count_cols = [
        "rows", "expiries", "strikes",
        "rows_volume_ge_1", "rows_volume_ge_25",
        "rows_core_moneyness",
    ]
    report[count_cols] = report[count_cols].fillna(0).astype(int)
    return report.sort_values("date").reset_index(drop=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--raw-options",
        default="data/processed/options_GLD_active_contract_intraday.parquet",
    )
    p.add_argument(
        "--stock",
        default="data/processed/gld_daily_history.csv",
    )
    p.add_argument(
        "--output-dir",
        default="data/processed",
    )
    p.add_argument("--end", default="2026-09-02")
    p.add_argument("--sessions", type=int, default=60)
    p.add_argument("--min-moneyness", type=float, default=0.85)
    p.add_argument("--max-moneyness", type=float, default=1.20)
    p.add_argument("--min-dte", type=int, default=21)
    p.add_argument("--max-dte", type=int, default=730)
    p.add_argument("--min-price", type=float, default=0.10)
    args = p.parse_args()

    raw_path = Path(args.raw_options)
    stock_path = Path(args.stock)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not raw_path.exists():
        raise FileNotFoundError(
            f"{raw_path} does not exist yet. Let the current IBKR backfill finish first."
        )

    panel, stock = build_daily_panel(
        raw_path=raw_path,
        stock_path=stock_path,
        end=args.end,
        sessions=args.sessions,
        min_moneyness=args.min_moneyness,
        max_moneyness=args.max_moneyness,
        min_dte=args.min_dte,
        max_dte=args.max_dte,
        min_price=args.min_price,
    )
    report = coverage_report(panel, stock)

    panel_path = out / f"options_GLD_daily_{args.sessions}.parquet"
    coverage_path = out / f"options_GLD_daily_{args.sessions}_coverage.csv"
    dates_path = out / f"gld_daily_{args.sessions}_dates.csv"

    panel.to_parquet(panel_path, index=False)
    report.to_csv(coverage_path, index=False)
    stock.to_csv(dates_path, index=False)

    print(f"[OK] requested sessions: {args.sessions}")
    print(
        f"[OK] study dates: {stock['date'].min().date()} -> "
        f"{stock['date'].max().date()}"
    )
    print(f"[OK] option rows: {len(panel)}")
    print(f"[OK] covered option dates: {(report['rows'] > 0).sum()} / {len(report)}")
    print(f"[OK] daily panel: {panel_path}")
    print(f"[OK] coverage: {coverage_path}")
    print()
    print(report.tail(15).to_string(index=False))


if __name__ == "__main__":
    main()
