"""
Targeted IBKR fetch for recent DAILY GLD option surfaces.

Goal
----
Avoid the very large all-contract backfill. This script focuses on the most
recent N GLD trading sessions (default: 60) and only requests a controlled
subset of currently active call options:

- selected expiries near target DTEs;
- selected strikes spanning the historical spot range;
- historical intraday TRADES bars;
- last available bar per contract per trading day.

This is designed specifically to evaluate whether a recent daily
t -> t+1 rolling OOS exercise is viable.

Typical run
-----------
python src/ibkr_gld_recent_daily_fetch.py \
    --end 2026-09-02 \
    --sessions 60 \
    --output-dir data/processed \
    --port 7497

Expected output
---------------
data/processed/options_GLD_recent_daily_raw.parquet
data/processed/options_GLD_daily_60.parquet
data/processed/options_GLD_daily_60_coverage.csv
data/processed/gld_daily_60_dates.csv
"""

from __future__ import annotations

import argparse
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from ib_insync import IB, Option, Stock, util


TICKER = "GLD"
EXCHANGE = "SMART"
CURRENCY = "USD"


def connect_ib(host="127.0.0.1", port=7497, client_id=91):
    ib = IB()
    ib.connect(
        host,
        port,
        clientId=client_id,
        readonly=True,
        timeout=15,
    )
    if not ib.isConnected():
        raise RuntimeError("IBKR API connection failed.")
    return ib


def load_recent_stock_window(stock_path: Path, end: str, sessions: int):
    stock = pd.read_csv(stock_path)
    stock["date"] = pd.to_datetime(stock["timestamp"], errors="coerce").dt.normalize()
    stock["spot"] = pd.to_numeric(stock["close"], errors="coerce")
    stock = stock.dropna(subset=["date", "spot"])
    stock = stock.loc[stock["spot"].gt(0.0)]
    stock = stock.loc[stock["date"].le(pd.Timestamp(end))]
    stock = (
        stock.sort_values("date")
        .drop_duplicates("date", keep="last")
        .tail(int(sessions))
        .reset_index(drop=True)
    )
    if len(stock) < sessions:
        print(f"[WARN] only {len(stock)} stock sessions available, requested {sessions}.")
    return stock[["date", "spot"]]


def choose_option_definition(ib, stock):
    params = ib.reqSecDefOptParams(stock.symbol, "", stock.secType, stock.conId)
    if not params:
        raise RuntimeError("No GLD option-chain definitions returned by IBKR.")

    preferred = [
        p for p in params
        if str(p.tradingClass).upper() == TICKER
        and p.exchange in {"SMART", ""}
    ]
    if not preferred:
        preferred = [
            p for p in params
            if str(p.tradingClass).upper() == TICKER
        ]
    if not preferred:
        preferred = params

    return max(preferred, key=lambda p: (len(p.expirations), len(p.strikes)))


def nearest_unique(values, targets, max_items):
    values = np.asarray(sorted(set(values)), dtype=float)
    chosen = []
    for target in targets:
        if values.size == 0:
            break
        idx = int(np.argmin(np.abs(values - target)))
        value = float(values[idx])
        if value not in chosen:
            chosen.append(value)
    if len(chosen) < max_items and values.size:
        remaining = [float(v) for v in values if float(v) not in chosen]
        if remaining:
            # Fill by even spacing over the remaining candidate set.
            take = min(max_items - len(chosen), len(remaining))
            idxs = np.linspace(0, len(remaining) - 1, take).round().astype(int)
            for idx in idxs:
                value = remaining[int(idx)]
                if value not in chosen:
                    chosen.append(value)
    return sorted(chosen[:max_items])


def select_expiries(definition, end_date, target_dtes, max_expiries):
    expiries = []
    for raw in definition.expirations:
        expiry = pd.to_datetime(str(raw), format="%Y%m%d", errors="coerce")
        if pd.isna(expiry):
            continue
        # Contract must still be active now to be queryable through IBKR.
        if expiry < pd.Timestamp.now().normalize():
            continue
        dte_at_end = (expiry - end_date).days
        if 21 <= dte_at_end <= 730:
            expiries.append((expiry, dte_at_end))

    if not expiries:
        raise RuntimeError("No active GLD expiries usable for the selected end date.")

    chosen = []
    for target in target_dtes:
        expiry, dte = min(expiries, key=lambda item: abs(item[1] - target))
        if expiry not in chosen:
            chosen.append(expiry)

    if len(chosen) < max_expiries:
        remaining = [e for e, _ in expiries if e not in chosen]
        remaining = sorted(remaining)
        if remaining:
            take = min(max_expiries - len(chosen), len(remaining))
            idxs = np.linspace(0, len(remaining) - 1, take).round().astype(int)
            chosen.extend(remaining[int(i)] for i in idxs if remaining[int(i)] not in chosen)

    return sorted(chosen[:max_expiries])


def select_strikes(definition, stock_window, min_moneyness, max_moneyness, max_strikes):
    min_spot = float(stock_window["spot"].min())
    max_spot = float(stock_window["spot"].max())

    low = min_spot * min_moneyness
    high = max_spot * max_moneyness

    candidates = sorted(
        float(k)
        for k in definition.strikes
        if np.isfinite(k) and low <= float(k) <= high
    )
    if not candidates:
        raise RuntimeError("No strikes in the requested historical moneyness range.")

    # Use targets based on the historical spot range, not just today's spot.
    spot_targets = np.linspace(min_spot, max_spot, 7)
    moneyness_targets = [0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15]
    targets = []
    for spot in spot_targets:
        for m in moneyness_targets:
            target = spot * m
            if low <= target <= high:
                targets.append(target)

    return nearest_unique(candidates, targets, max_items=max_strikes)


def qualify_contracts(ib, contracts: Sequence, batch_size=30):
    qualified = []
    for i in range(0, len(contracts), batch_size):
        batch = contracts[i:i+batch_size]
        try:
            qualified.extend(ib.qualifyContracts(*batch))
        except Exception as exc:
            print(f"[WARN] qualification batch failed: {exc}")
        ib.sleep(0.15)

    unique = {}
    for c in qualified:
        if getattr(c, "conId", 0):
            unique[int(c.conId)] = c
    return list(unique.values())


def bars_to_frame(contract, bars):
    frame = util.df(bars)
    if frame.empty:
        return pd.DataFrame()

    ts = pd.to_datetime(frame["date"], errors="coerce", utc=True)
    out = pd.DataFrame({
        "timestamp": ts,
        "open": pd.to_numeric(frame["open"], errors="coerce"),
        "high": pd.to_numeric(frame["high"], errors="coerce"),
        "low": pd.to_numeric(frame["low"], errors="coerce"),
        "close": pd.to_numeric(frame["close"], errors="coerce"),
        "volume": pd.to_numeric(frame["volume"], errors="coerce"),
        "bar_count": pd.to_numeric(frame.get("barCount"), errors="coerce"),
    })
    out = out.dropna(subset=["timestamp", "close"]).copy()
    out["expiry"] = pd.to_datetime(
        contract.lastTradeDateOrContractMonth[:8],
        format="%Y%m%d",
    )
    out["opt_type"] = str(contract.right).upper()
    out["strike"] = float(contract.strike)
    out["conId"] = int(contract.conId)
    out["localSymbol"] = str(contract.localSymbol)
    out["tradingClass"] = str(contract.tradingClass)
    return out


def fetch_contract_history(ib, contract, end_date, calendar_days, pacing_seconds):
    end_dt = (end_date + pd.Timedelta(days=1)).strftime("%Y%m%d 23:59:59 US/Eastern")
    duration_days = max(int(calendar_days), 30)

    # One compact request per contract.
    try:
        bars = ib.reqHistoricalData(
            contract,
            endDateTime=end_dt,
            durationStr=f"{duration_days} D",
            barSizeSetting="8 hours",
            whatToShow="TRADES",
            useRTH=True,
            formatDate=2,
            keepUpToDate=False,
        )
        frame = bars_to_frame(contract, bars)
    except Exception as exc:
        print(f"[WARN] history failed {contract.localSymbol}: {exc}")
        frame = pd.DataFrame()

    time.sleep(pacing_seconds)
    return frame


def build_daily_panel(raw, stock_window, min_moneyness, max_moneyness, min_dte, max_dte, min_price):
    if raw.empty:
        return pd.DataFrame()

    frame = raw.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["date"] = (
        frame["timestamp"]
        .dt.tz_convert("US/Eastern")
        .dt.tz_localize(None)
        .dt.normalize()
    )

    valid_dates = set(stock_window["date"])
    frame = frame.loc[frame["date"].isin(valid_dates)].copy()

    # Last available bar per option/day.
    frame = (
        frame.sort_values("timestamp")
        .groupby(["conId", "date"], as_index=False, sort=True)
        .tail(1)
        .copy()
    )

    frame = frame.merge(stock_window, on="date", how="inner")
    frame["expiry"] = pd.to_datetime(frame["expiry"]).dt.normalize()
    frame["dte"] = (frame["expiry"] - frame["date"]).dt.days
    frame["T"] = frame["dte"] / 365.25
    frame["moneyness"] = frame["strike"] / frame["spot"]

    frame = frame.loc[
        frame["close"].gt(min_price)
        & frame["dte"].between(min_dte, max_dte)
        & frame["moneyness"].between(min_moneyness, max_moneyness)
    ].copy()

    frame["ts"] = frame["date"]
    frame["expiry"] = frame["expiry"].dt.strftime("%Y-%m-%d")
    return frame.sort_values(["date", "expiry", "strike"]).reset_index(drop=True)


def coverage_report(panel, stock_window):
    report = stock_window[["date", "spot"]].copy()

    if panel.empty:
        for col in [
            "rows", "expiries", "strikes",
            "rows_volume_ge_1", "rows_volume_ge_25",
            "core_rows_90_110",
        ]:
            report[col] = 0
        report["min_dte"] = np.nan
        report["max_dte"] = np.nan
        return report

    grouped = (
        panel.groupby("date")
        .agg(
            rows=("strike", "size"),
            expiries=("expiry", "nunique"),
            strikes=("strike", "nunique"),
            min_dte=("dte", "min"),
            max_dte=("dte", "max"),
        )
        .reset_index()
    )
    vol1 = (
        panel.loc[panel["volume"].fillna(0).ge(1)]
        .groupby("date").size().rename("rows_volume_ge_1").reset_index()
    )
    vol25 = (
        panel.loc[panel["volume"].fillna(0).ge(25)]
        .groupby("date").size().rename("rows_volume_ge_25").reset_index()
    )
    core = (
        panel.loc[panel["moneyness"].between(0.90, 1.10)]
        .groupby("date").size().rename("core_rows_90_110").reset_index()
    )

    report = (
        report.merge(grouped, on="date", how="left")
        .merge(vol1, on="date", how="left")
        .merge(vol25, on="date", how="left")
        .merge(core, on="date", how="left")
    )

    count_cols = [
        "rows", "expiries", "strikes",
        "rows_volume_ge_1", "rows_volume_ge_25",
        "core_rows_90_110",
    ]
    report[count_cols] = report[count_cols].fillna(0).astype(int)
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7497)
    p.add_argument("--client-id", type=int, default=91)
    p.add_argument("--end", default="2026-09-02")
    p.add_argument("--sessions", type=int, default=60)
    p.add_argument("--stock", default="data/processed/gld_daily_history.csv")
    p.add_argument("--output-dir", default="data/processed")
    p.add_argument("--max-expiries", type=int, default=6)
    p.add_argument("--max-strikes", type=int, default=24)
    p.add_argument("--min-moneyness", type=float, default=0.85)
    p.add_argument("--max-moneyness", type=float, default=1.15)
    p.add_argument("--min-dte", type=int, default=21)
    p.add_argument("--max-dte", type=int, default=540)
    p.add_argument("--min-price", type=float, default=0.10)
    p.add_argument("--pacing-seconds", type=float, default=0.20)
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    end_date = pd.Timestamp(args.end)
    stock_window = load_recent_stock_window(
        Path(args.stock), args.end, args.sessions
    )
    first_date = stock_window["date"].min()
    calendar_days = (end_date - first_date).days + 10

    print(
        f"[INFO] stock window: {first_date.date()} -> "
        f"{stock_window['date'].max().date()} ({len(stock_window)} sessions)"
    )

    ib = connect_ib(args.host, args.port, args.client_id)
    try:
        stock = ib.qualifyContracts(Stock(TICKER, EXCHANGE, CURRENCY))[0]
        definition = choose_option_definition(ib, stock)

        target_dtes = [45, 75, 105, 150, 210, 300]
        expiries = select_expiries(
            definition,
            end_date,
            target_dtes=target_dtes,
            max_expiries=args.max_expiries,
        )
        strikes = select_strikes(
            definition,
            stock_window,
            args.min_moneyness,
            args.max_moneyness,
            args.max_strikes,
        )

        print(f"[INFO] selected expiries ({len(expiries)}):")
        for e in expiries:
            print(f"       {e.date()}  DTE@end={(e-end_date).days}")

        print(
            f"[INFO] selected strikes: {len(strikes)} "
            f"({min(strikes):.2f} -> {max(strikes):.2f})"
        )

        contracts = [
            Option(
                TICKER,
                expiry.strftime("%Y%m%d"),
                strike,
                "C",
                EXCHANGE,
                currency=CURRENCY,
                tradingClass=definition.tradingClass,
            )
            for expiry in expiries
            for strike in strikes
        ]

        qualified = qualify_contracts(ib, contracts)
        print(
            f"[INFO] qualified contracts: {len(qualified)} / "
            f"{len(contracts)}"
        )

        pieces = []
        for i, contract in enumerate(qualified, start=1):
            print(
                f"[{i}/{len(qualified)}] "
                f"{contract.localSymbol or contract.conId}"
            )
            piece = fetch_contract_history(
                ib,
                contract,
                end_date,
                calendar_days=calendar_days,
                pacing_seconds=args.pacing_seconds,
            )
            if not piece.empty:
                pieces.append(piece)

        raw = (
            pd.concat(pieces, ignore_index=True)
            if pieces
            else pd.DataFrame()
        )

        raw_path = out / "options_GLD_recent_daily_raw.parquet"
        raw.to_parquet(raw_path, index=False)

        panel = build_daily_panel(
            raw,
            stock_window,
            min_moneyness=args.min_moneyness,
            max_moneyness=args.max_moneyness,
            min_dte=args.min_dte,
            max_dte=args.max_dte,
            min_price=args.min_price,
        )

        panel_path = out / f"options_GLD_daily_{args.sessions}.parquet"
        panel.to_parquet(panel_path, index=False)

        coverage = coverage_report(panel, stock_window)
        coverage_path = out / f"options_GLD_daily_{args.sessions}_coverage.csv"
        coverage.to_csv(coverage_path, index=False)

        dates_path = out / f"gld_daily_{args.sessions}_dates.csv"
        stock_window.to_csv(dates_path, index=False)

        print()
        print(f"[OK] raw: {raw_path}")
        print(f"[OK] daily panel: {panel_path}")
        print(f"[OK] coverage: {coverage_path}")
        print(
            f"[OK] covered dates: "
            f"{int((coverage['rows'] > 0).sum())}/{len(coverage)}"
        )
        print()
        print(coverage.tail(20).to_string(index=False))

    finally:
        ib.disconnect()


if __name__ == "__main__":
    main()
