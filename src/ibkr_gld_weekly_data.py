"""
IBKR utilities retained for reproducibility.

Current project use:
- `rates`: convert the manually supplied 2026 Treasury CSV.
- `stock`: download GLD daily history.
- `backfill-active`: best-effort historical option backfill for contracts that
  are still queryable at IBKR.
- `snapshot`: preserve a current point-in-time option-chain snapshot.

The current OFFLINE surface reconstruction does not invoke this file.
IBKR cannot reconstruct already-expired historical option contracts.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from ib_insync import IB, Option, Stock, util


TICKER = "GLD"
CURRENCY = "USD"
EXCHANGE = "SMART"

RETURN_COLUMNS = (
    "timestamp", "symbol", "open", "high", "low", "close",
    "volume", "log_return", "simple_return",
)

TREASURY_TENORS = {
    "1 Mo": ("US1M", "1M", 1.0 / 12.0),
    "1.5 Month": ("US1_5M", "1.5M", 1.5 / 12.0),
    "2 Mo": ("US2M", "2M", 2.0 / 12.0),
    "3 Mo": ("US3M", "3M", 3.0 / 12.0),
    "4 Mo": ("US4M", "4M", 4.0 / 12.0),
    "6 Mo": ("US6M", "6M", 6.0 / 12.0),
    "1 Yr": ("US1Y", "1Y", 1.0),
    "2 Yr": ("US2Y", "2Y", 2.0),
    "3 Yr": ("US3Y", "3Y", 3.0),
    "5 Yr": ("US5Y", "5Y", 5.0),
    "7 Yr": ("US7Y", "7Y", 7.0),
    "10 Yr": ("US10Y", "10Y", 10.0),
    "20 Yr": ("US20Y", "20Y", 20.0),
    "30 Yr": ("US30Y", "30Y", 30.0),
}


def connect_ib(host="127.0.0.1", port=7497, client_id=81):
    ib = IB()
    ib.connect(host, int(port), clientId=int(client_id), readonly=True, timeout=15)
    if not ib.isConnected():
        raise RuntimeError("IBKR API connection failed.")
    return ib


def chunks(values: Sequence, size: int):
    values = list(values)
    for i in range(0, len(values), int(size)):
        yield values[i:i + int(size)]


def qualify_in_batches(ib, contracts, batch_size=40):
    unique = {}
    for batch in chunks(contracts, batch_size):
        try:
            valid = ib.qualifyContracts(*batch)
        except Exception as exc:
            print(f"[WARN] qualification batch failed: {exc}")
            valid = []
        for contract in valid:
            if getattr(contract, "conId", 0):
                unique[int(contract.conId)] = contract
        ib.sleep(0.15)
    return list(unique.values())


def normalise_treasury_csv(csv_path, output_path):
    source = pd.read_csv(csv_path)
    if "Date" not in source.columns:
        raise ValueError("Treasury CSV must contain a 'Date' column.")
    missing = [c for c in TREASURY_TENORS if c not in source.columns]
    if missing:
        raise ValueError(f"Treasury CSV is missing columns: {missing}")

    source["Date"] = pd.to_datetime(
        source["Date"], format="%m/%d/%Y", errors="coerce"
    )
    source = source.dropna(subset=["Date"]).copy()

    fetched_at = datetime.now(timezone.utc).isoformat()
    rows = []
    for column, (symbol, maturity, years) in TREASURY_TENORS.items():
        yields = pd.to_numeric(source[column], errors="coerce")
        for date, quoted_yield in zip(source["Date"], yields):
            if not np.isfinite(quoted_yield):
                continue
            rows.append({
                "symbol": symbol,
                "date": date.strftime("%Y-%m-%d"),
                "maturity": maturity,
                "maturity_days": years * 365.25,
                "maturity_years": years,
                "par_yield_pct": float(quoted_yield),
                "continuous_rate": float(np.log1p(quoted_yield / 100.0)),
                "source_fetched_at": fetched_at,
            })

    result = pd.DataFrame(rows).sort_values(
        ["date", "maturity_years"]
    ).reset_index(drop=True)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def fetch_stock_history(ib, start, end):
    stock = ib.qualifyContracts(Stock(TICKER, EXCHANGE, CURRENCY))[0]
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    duration_days = max((end_ts - start_ts).days + 10, 30)

    bars = ib.reqHistoricalData(
        stock,
        endDateTime=(end_ts + pd.Timedelta(days=1)).strftime(
            "%Y%m%d 23:59:59 US/Eastern"
        ),
        durationStr=f"{duration_days} D",
        barSizeSetting="1 day",
        whatToShow="TRADES",
        useRTH=True,
        formatDate=2,
        keepUpToDate=False,
    )
    frame = util.df(bars)
    if frame is None or frame.empty:
        raise RuntimeError("IBKR returned no GLD stock history.")

    dates = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    result = pd.DataFrame({
        "timestamp": dates,
        "symbol": TICKER,
        "open": pd.to_numeric(frame["open"], errors="coerce"),
        "high": pd.to_numeric(frame["high"], errors="coerce"),
        "low": pd.to_numeric(frame["low"], errors="coerce"),
        "close": pd.to_numeric(frame["close"], errors="coerce"),
        "volume": pd.to_numeric(frame["volume"], errors="coerce"),
    })
    result = result.dropna(subset=["timestamp", "close"])
    result = result.loc[result["close"].gt(0)]
    result = result.loc[result["timestamp"].between(start_ts, end_ts)]
    result = result.sort_values("timestamp").drop_duplicates(
        "timestamp", keep="last"
    )
    result["log_return"] = np.log(result["close"]).diff()
    result["simple_return"] = result["close"].pct_change(fill_method=None)
    result["timestamp"] = result["timestamp"].dt.strftime("%Y-%m-%d")
    return result.loc[:, RETURN_COLUMNS].reset_index(drop=True)


def weekly_last_stock_dates(stock_history):
    frame = stock_history.copy()
    frame["_date"] = pd.to_datetime(
        frame["timestamp"], errors="coerce"
    ).dt.normalize()
    frame = frame.dropna(subset=["_date"])
    frame["_week"] = frame["_date"].dt.to_period("W-FRI")
    selected = frame.sort_values("_date").groupby("_week", sort=True).tail(1)
    return pd.DataFrame({
        "target_date": selected["_date"].to_numpy(),
        "spot": selected["close"].to_numpy(float),
        "_week": selected["_week"].to_numpy(),
    }).reset_index(drop=True)


def get_option_definition(ib, stock):
    params = ib.reqSecDefOptParams(
        stock.symbol, "", stock.secType, stock.conId
    )
    if not params:
        raise RuntimeError("IBKR returned no GLD option-chain definition.")
    preferred = [
        p for p in params
        if str(p.tradingClass).upper() == "GLD"
        and p.exchange in {"SMART", ""}
    ]
    if not preferred:
        preferred = [p for p in params if str(p.tradingClass).upper() == "GLD"]
    if not preferred:
        preferred = params
    return max(preferred, key=lambda p: (len(p.expirations), len(p.strikes)))


def fetch_active_contract_history(ib, contract, end, pacing_seconds=0.15):
    end_ts = pd.Timestamp(end) + pd.Timedelta(days=1)
    try:
        bars = ib.reqHistoricalData(
            contract,
            endDateTime=end_ts.strftime("%Y%m%d 23:59:59 US/Eastern"),
            durationStr="1 Y",
            barSizeSetting="8 hours",
            whatToShow="TRADES",
            useRTH=True,
            formatDate=2,
            keepUpToDate=False,
        )
    except Exception as exc:
        print(f"[WARN] history failed {contract.localSymbol}: {exc}")
        return pd.DataFrame()

    time.sleep(float(pacing_seconds))
    frame = util.df(bars)
    if frame is None or frame.empty:
        return pd.DataFrame()

    out = pd.DataFrame({
        "timestamp": pd.to_datetime(frame["date"], errors="coerce", utc=True),
        "open": pd.to_numeric(frame["open"], errors="coerce"),
        "high": pd.to_numeric(frame["high"], errors="coerce"),
        "low": pd.to_numeric(frame["low"], errors="coerce"),
        "close": pd.to_numeric(frame["close"], errors="coerce"),
        "volume": pd.to_numeric(frame["volume"], errors="coerce"),
    }).dropna(subset=["timestamp", "close"])
    out["expiry"] = pd.to_datetime(
        contract.lastTradeDateOrContractMonth[:8],
        format="%Y%m%d",
        errors="coerce",
    )
    out["opt_type"] = str(contract.right).upper()
    out["strike"] = float(contract.strike)
    out["conId"] = int(contract.conId)
    out["localSymbol"] = str(contract.localSymbol)
    out["tradingClass"] = str(contract.tradingClass)
    return out


def run_backfill(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    stock_path = out / "gld_daily_history.csv"
    if stock_path.exists():
        stock_history = pd.read_csv(stock_path)
    else:
        ib0 = connect_ib(args.host, args.port, args.client_id)
        try:
            stock_history = fetch_stock_history(ib0, args.start, args.end)
        finally:
            ib0.disconnect()
        stock_history.to_csv(stock_path, index=False)

    stock_dates = pd.to_datetime(stock_history["timestamp"])
    spots = pd.to_numeric(stock_history["close"], errors="coerce")
    min_spot, max_spot = float(spots.min()), float(spots.max())

    ib = connect_ib(args.host, args.port, args.client_id)
    try:
        stock = ib.qualifyContracts(Stock(TICKER, EXCHANGE, CURRENCY))[0]
        definition = get_option_definition(ib, stock)
        today = pd.Timestamp.now().normalize()
        max_expiry = pd.Timestamp(args.end) + pd.Timedelta(days=args.max_dte)

        expiries = []
        for raw in definition.expirations:
            expiry = pd.to_datetime(str(raw), format="%Y%m%d", errors="coerce")
            if pd.notna(expiry) and today <= expiry <= max_expiry:
                expiries.append(expiry)
        expiries = sorted(set(expiries))

        strikes = sorted(
            float(k) for k in definition.strikes
            if np.isfinite(k)
            and min_spot * args.min_moneyness <= float(k)
            <= max_spot * args.max_moneyness
        )

        contracts = [
            Option(
                TICKER, expiry.strftime("%Y%m%d"), strike, "C", EXCHANGE,
                currency=CURRENCY, tradingClass=definition.tradingClass,
            )
            for expiry in expiries for strike in strikes
        ]
        qualified = qualify_in_batches(ib, contracts)
        print(f"[INFO] qualified active option contracts: {len(qualified)}")

        pieces = []
        for i, contract in enumerate(qualified, 1):
            print(f"[{i}/{len(qualified)}] {contract.localSymbol}")
            piece = fetch_active_contract_history(
                ib, contract, args.end, args.pacing_seconds
            )
            if not piece.empty:
                pieces.append(piece)

        raw = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
        raw_path = out / "options_GLD_active_contract_intraday.parquet"
        raw.to_parquet(raw_path, index=False)
        print(f"[OK] raw active-contract history: {raw_path}")
        print("[IMPORTANT] Expired historical contracts remain unrecoverable.")
    finally:
        ib.disconnect()


def run_snapshot(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ib = connect_ib(args.host, args.port, args.client_id)
    try:
        stock = ib.qualifyContracts(Stock(TICKER, EXCHANGE, CURRENCY))[0]
        ticker = ib.reqTickers(stock)[0]
        candidates = [ticker.last, ticker.close, ticker.marketPrice()]
        spot = next(
            float(x) for x in candidates
            if x is not None and np.isfinite(x) and float(x) > 0
        )

        definition = get_option_definition(ib, stock)
        today = pd.Timestamp.now().normalize()
        expiries = []
        for raw in definition.expirations:
            expiry = pd.to_datetime(str(raw), format="%Y%m%d", errors="coerce")
            if pd.isna(expiry):
                continue
            dte = (expiry - today).days
            if args.min_dte <= dte <= args.max_dte:
                expiries.append(expiry)
        expiries = sorted(set(expiries))

        strikes = sorted(
            float(k) for k in definition.strikes
            if np.isfinite(k)
            and spot * args.min_moneyness <= float(k)
            <= spot * args.max_moneyness
        )
        contracts = [
            Option(
                TICKER, expiry.strftime("%Y%m%d"), strike, "C", EXCHANGE,
                currency=CURRENCY, tradingClass=definition.tradingClass,
            )
            for expiry in expiries for strike in strikes
        ]
        qualified = qualify_in_batches(ib, contracts)

        rows = []
        as_of = datetime.now(timezone.utc)
        for batch in chunks(qualified, 40):
            for t in ib.reqTickers(*batch):
                c = t.contract
                bid = float(t.bid) if t.bid is not None and np.isfinite(t.bid) else np.nan
                ask = float(t.ask) if t.ask is not None and np.isfinite(t.ask) else np.nan
                last = float(t.last) if t.last is not None and np.isfinite(t.last) else np.nan
                midpoint = (bid + ask) / 2 if bid > 0 and ask > bid else np.nan
                price = last if last > 0 else midpoint
                if not np.isfinite(price) or price <= 0:
                    continue
                rows.append({
                    "ts": as_of.isoformat(),
                    "expiry": pd.to_datetime(
                        c.lastTradeDateOrContractMonth[:8], format="%Y%m%d"
                    ).strftime("%Y-%m-%d"),
                    "opt_type": str(c.right).upper(),
                    "strike": float(c.strike),
                    "close": price,
                    "bid": bid,
                    "ask": ask,
                    "last": last,
                    "spot": spot,
                    "conId": int(c.conId),
                    "localSymbol": str(c.localSymbol),
                    "tradingClass": str(c.tradingClass),
                })
            ib.sleep(0.15)

        snapshot = pd.DataFrame(rows)
        slug = pd.Timestamp.now().strftime("%Y-%m-%d")
        path = out / "snapshots" / f"GLD_options_{slug}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        snapshot.to_parquet(path, index=False)
        meta = {
            "as_of_utc": as_of.isoformat(),
            "spot": spot,
            "rows": int(len(snapshot)),
            "purpose": "point-in-time surface preservation",
        }
        path.with_suffix(".json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        print(f"[OK] current snapshot: {path}")
    finally:
        ib.disconnect()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "mode", choices=["rates", "stock", "backfill-active", "snapshot"]
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7497)
    p.add_argument("--client-id", type=int, default=81)
    p.add_argument("--start", default="2026-01-02")
    p.add_argument("--end", default="2026-09-02")
    p.add_argument("--output-dir", default="data/processed")
    p.add_argument(
        "--treasury-csv", default="data/raw/daily-treasury-rates2026.csv"
    )
    p.add_argument("--min-moneyness", type=float, default=0.85)
    p.add_argument("--max-moneyness", type=float, default=1.20)
    p.add_argument("--min-dte", type=int, default=21)
    p.add_argument("--max-dte", type=int, default=730)
    p.add_argument("--pacing-seconds", type=float, default=0.15)
    args = p.parse_args()

    if args.mode == "rates":
        path = Path(args.output_dir) / "usd_treasury_history.csv"
        history = normalise_treasury_csv(args.treasury_csv, path)
        dates = pd.to_datetime(history["date"])
        print(f"[OK] Treasury history: {path}")
        print(
            f"[OK] {dates.nunique()} dates, "
            f"{history['symbol'].nunique()} tenors, "
            f"{dates.min().date()} -> {dates.max().date()}"
        )
    elif args.mode == "stock":
        ib = connect_ib(args.host, args.port, args.client_id)
        try:
            history = fetch_stock_history(ib, args.start, args.end)
        finally:
            ib.disconnect()
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        history.to_csv(out / "gld_daily_history.csv", index=False)
        weekly_last_stock_dates(history).to_csv(
            out / "gld_weekly_dates.csv", index=False
        )
        print(f"[OK] stock history: {out / 'gld_daily_history.csv'}")
    elif args.mode == "backfill-active":
        run_backfill(args)
    else:
        run_snapshot(args)


if __name__ == "__main__":
    main()
