"""
IBKR -> Team 8 weekly GLD option-data builder.

Purpose
-------
1. Match the ib_insync workflow already used in the MSTR project.
2. Download GLD stock history from IBKR.
3. Build one Friday-ending observation bucket per week.
4. Best-effort backfill historical option bars for contracts that are STILL
   active/qualifiable at IBKR.
5. Save current point-in-time option-chain snapshots for forward collection.
6. Convert the manually supplied 2026 Treasury CSV into the exact long-form
   rate-history structure expected by the Team 8 no-look-ahead code.

IMPORTANT IBKR LIMITATION
-------------------------
IBKR does not provide historical data for expired options. Therefore, running
this script in September 2026 cannot recreate a complete January 2026 option
surface from IBKR alone. The `backfill-active` mode is intentionally labelled
best-effort and produces a coverage report. Use `snapshot` weekly going forward
to build complete point-in-time surfaces.

Expected Treasury input
-----------------------
daily-treasury-rates2026.csv with columns:
Date, 1 Mo, 1.5 Month, 2 Mo, 3 Mo, 4 Mo, 6 Mo, 1 Yr, 2 Yr, 3 Yr,
5 Yr, 7 Yr, 10 Yr, 20 Yr, 30 Yr

Example
-------
python ibkr_gld_weekly_data.py rates \
    --treasury-csv daily-treasury-rates2026.csv

python ibkr_gld_weekly_data.py stock \
    --start 2026-01-02 --end 2026-09-02

python ibkr_gld_weekly_data.py backfill-active \
    --start 2026-01-02 --end 2026-09-02

python ibkr_gld_weekly_data.py snapshot

TWS / IB Gateway
----------------
Paper TWS default port: 7497
Live  TWS default port: 7496
Paper IB Gateway:       4002
Live  IB Gateway:       4001
Enable API socket clients in TWS / IB Gateway first.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from ib_insync import IB, Option, Stock, util


ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "Data" / "ibkr_local"
DEFAULT_TREASURY = ROOT / "daily-treasury-rates2026.csv"

TICKER = "GLD"
CURRENCY = "USD"
EXCHANGE = "SMART"

# Team 8 historical_validation.py expects these fields at minimum.
OPTION_HISTORY_COLUMNS = (
    "ts",
    "expiry",
    "opt_type",
    "strike",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "bar_count",
    "conId",
    "localSymbol",
    "tradingClass",
    "source_price_method",
)

RETURN_COLUMNS = (
    "timestamp",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "log_return",
    "simple_return",
)

RATE_HISTORY_COLUMNS = (
    "symbol",
    "date",
    "maturity",
    "maturity_days",
    "maturity_years",
    "par_yield_pct",
    "continuous_rate",
    "source_fetched_at",
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


def connect_ib(
    host: str = "127.0.0.1",
    port: int = 7497,
    client_id: int = 81,
    readonly: bool = True,
) -> IB:
    """Connect to TWS / IB Gateway using the same ib_insync style as MSTR."""
    ib = IB()
    ib.connect(
        host,
        port,
        clientId=client_id,
        readonly=readonly,
        timeout=15,
    )
    if not ib.isConnected():
        raise RuntimeError("IBKR API connection failed.")
    return ib


def chunks(values: Sequence, size: int):
    for i in range(0, len(values), size):
        yield values[i : i + size]


def qualify_in_batches(ib: IB, contracts: Sequence, batch_size: int = 40):
    """Avoid sending an excessively large qualification request in one call."""
    result = []
    for batch in chunks(list(contracts), batch_size):
        try:
            qualified = ib.qualifyContracts(*batch)
            result.extend(qualified)
        except Exception as exc:
            print(f"[WARN] qualification batch failed: {exc}")
        ib.sleep(0.20)
    # conId is the safest de-duplication key.
    unique = {}
    for contract in result:
        if getattr(contract, "conId", 0):
            unique[int(contract.conId)] = contract
    return list(unique.values())


def fetch_stock_history(
    ib: IB,
    start: str,
    end: str,
    symbol: str = TICKER,
) -> pd.DataFrame:
    """Download daily underlying bars used for historical spot and returns."""
    stock = ib.qualifyContracts(Stock(symbol, EXCHANGE, CURRENCY))[0]

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    # One request is sufficient for the 2026 window.
    duration_days = max((end_ts - start_ts).days + 10, 30)
    duration = f"{duration_days} D"

    end_dt = (end_ts + pd.Timedelta(days=1)).strftime("%Y%m%d 23:59:59 US/Eastern")
    bars = ib.reqHistoricalData(
        stock,
        endDateTime=end_dt,
        durationStr=duration,
        barSizeSetting="1 day",
        whatToShow="TRADES",
        useRTH=True,
        formatDate=2,
        keepUpToDate=False,
    )
    frame = util.df(bars)
    if frame.empty:
        raise RuntimeError("IBKR returned no GLD stock history.")

    # Daily IBKR bars represent trading CALENDAR DATES, not instants that
    # should be shifted across time zones.  Treat them as naive dates.
    bar_dates = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    result = pd.DataFrame(
        {
            "timestamp": bar_dates,
            "symbol": symbol,
            "open": pd.to_numeric(frame["open"], errors="coerce"),
            "high": pd.to_numeric(frame["high"], errors="coerce"),
            "low": pd.to_numeric(frame["low"], errors="coerce"),
            "close": pd.to_numeric(frame["close"], errors="coerce"),
            "volume": pd.to_numeric(frame["volume"], errors="coerce"),
        }
    )
    result = result.dropna(subset=["timestamp", "close"])
    result = result.loc[result["close"].gt(0.0)].copy()
    result = result.loc[result["timestamp"].dt.date >= start_ts.date()]
    result = result.loc[result["timestamp"].dt.date <= end_ts.date()]
    result = result.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    result["log_return"] = np.log(result["close"]).diff()
    result["simple_return"] = result["close"].pct_change(fill_method=None)
    result["timestamp"] = result["timestamp"].dt.strftime("%Y-%m-%d")
    return result.loc[:, RETURN_COLUMNS].reset_index(drop=True)


def weekly_last_stock_dates(stock_history: pd.DataFrame) -> pd.DataFrame:
    """
    Select the last available trading observation in each Friday-ending week.

    The result also provides the historical spot used for moneyness filters.
    """
    frame = stock_history.copy()
    # Stock history is stored as trading calendar dates. Do not apply timezone
    # conversion here, otherwise midnight UTC can become the previous US date.
    frame["_date"] = pd.to_datetime(
        frame["timestamp"], errors="coerce"
    ).dt.normalize()
    frame = frame.dropna(subset=["_date"]).copy()
    frame["_week"] = frame["_date"].dt.to_period("W-FRI")
    selected = frame.sort_values("_date").groupby("_week", sort=True).tail(1).copy()
    selected["target_date"] = selected["_date"].dt.normalize()
    return selected[["target_date", "close", "_week"]].rename(
        columns={"close": "spot"}
    ).reset_index(drop=True)


def normalise_treasury_csv(
    csv_path: str | Path,
    output_path: str | Path,
) -> pd.DataFrame:
    """
    Convert the manually supplied Treasury CSV to Team 8 rate-history format.

    The project currently uses log(1+y) as an explicitly documented continuous
    proxy for the quoted Treasury par yield. We preserve that convention here.
    """
    csv_path = Path(csv_path)
    source = pd.read_csv(csv_path)
    if "Date" not in source.columns:
        raise ValueError("Treasury CSV must contain a 'Date' column.")

    missing = [column for column in TREASURY_TENORS if column not in source.columns]
    if missing:
        raise ValueError(f"Treasury CSV is missing columns: {missing}")

    source["Date"] = pd.to_datetime(
        source["Date"], format="%m/%d/%Y", errors="coerce"
    )
    source = source.dropna(subset=["Date"]).copy()

    rows = []
    fetched_at = datetime.now(timezone.utc).isoformat()
    for column, (symbol, maturity_label, years) in TREASURY_TENORS.items():
        values = pd.to_numeric(source[column], errors="coerce")
        for date, quoted_yield in zip(source["Date"], values):
            if not np.isfinite(quoted_yield):
                continue
            rows.append(
                {
                    "symbol": symbol,
                    "date": date.strftime("%Y-%m-%d"),
                    "maturity": maturity_label,
                    "maturity_days": years * 365.25,
                    "maturity_years": years,
                    "par_yield_pct": float(quoted_yield),
                    "continuous_rate": float(np.log1p(quoted_yield / 100.0)),
                    "source_fetched_at": fetched_at,
                }
            )

    history = pd.DataFrame(rows, columns=RATE_HISTORY_COLUMNS)
    history = history.sort_values(["date", "maturity_years"]).reset_index(drop=True)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(output_path, index=False)
    return history


def get_option_definition(ib: IB, stock):
    """Choose the SMART/GLD option parameter set returned by IBKR."""
    params = ib.reqSecDefOptParams(
        stock.symbol,
        "",
        stock.secType,
        stock.conId,
    )
    if not params:
        raise RuntimeError("IBKR returned no option-chain definitions for GLD.")

    preferred = [
        p for p in params
        if (p.exchange in {"SMART", ""})
        and str(p.tradingClass).upper() == stock.symbol.upper()
    ]
    if not preferred:
        preferred = [
            p for p in params
            if str(p.tradingClass).upper() == stock.symbol.upper()
        ]
    if not preferred:
        preferred = params

    # Prefer the richest definition.
    preferred = sorted(
        preferred,
        key=lambda p: (len(p.expirations), len(p.strikes)),
        reverse=True,
    )
    return preferred[0]


def build_active_candidate_contracts(
    ib: IB,
    stock_history: pd.DataFrame,
    start: str,
    end: str,
    min_moneyness: float = 0.85,
    max_moneyness: float = 1.20,
    max_dte: int = 730,
    right: str = "C",
):
    """
    Build candidate contracts that remain active today.

    This cannot include already expired historical contracts because IBKR does
    not make their historical data available.
    """
    stock = ib.qualifyContracts(Stock(TICKER, EXCHANGE, CURRENCY))[0]
    definition = get_option_definition(ib, stock)
    weekly = weekly_last_stock_dates(stock_history)

    min_spot = float(weekly["spot"].min())
    max_spot = float(weekly["spot"].max())
    strike_low = min_spot * min_moneyness
    strike_high = max_spot * max_moneyness

    strikes = sorted(
        float(k)
        for k in definition.strikes
        if np.isfinite(k) and strike_low <= float(k) <= strike_high
    )

    today = pd.Timestamp.now(tz="US/Eastern").normalize().tz_localize(None)
    max_expiry = pd.Timestamp(end) + pd.Timedelta(days=max_dte)
    expiries = []
    for raw in definition.expirations:
        expiry = pd.to_datetime(str(raw), format="%Y%m%d", errors="coerce")
        if pd.isna(expiry):
            continue
        if expiry >= today and expiry <= max_expiry:
            expiries.append(expiry)
    expiries = sorted(set(expiries))

    contracts = [
        Option(
            TICKER,
            expiry.strftime("%Y%m%d"),
            strike,
            right,
            EXCHANGE,
            currency=CURRENCY,
            tradingClass=definition.tradingClass,
        )
        for expiry in expiries
        for strike in strikes
    ]

    print(
        f"[INFO] candidate active contracts before qualification: "
        f"{len(contracts)} ({len(expiries)} expiries x {len(strikes)} strikes)"
    )
    qualified = qualify_in_batches(ib, contracts)
    print(f"[INFO] qualified active option contracts: {len(qualified)}")
    return qualified, weekly


def _history_frame_from_bars(contract, bars) -> pd.DataFrame:
    frame = util.df(bars)
    if frame.empty:
        return pd.DataFrame()

    timestamp = pd.to_datetime(frame["date"], errors="coerce", utc=True)
    out = pd.DataFrame(
        {
            "timestamp": timestamp,
            "open": pd.to_numeric(frame["open"], errors="coerce"),
            "high": pd.to_numeric(frame["high"], errors="coerce"),
            "low": pd.to_numeric(frame["low"], errors="coerce"),
            "close": pd.to_numeric(frame["close"], errors="coerce"),
            "volume": pd.to_numeric(frame["volume"], errors="coerce"),
            "bar_count": pd.to_numeric(frame.get("barCount"), errors="coerce"),
        }
    )
    out = out.dropna(subset=["timestamp", "close"]).copy()
    out["conId"] = int(contract.conId)
    out["localSymbol"] = str(contract.localSymbol)
    out["tradingClass"] = str(contract.tradingClass)
    out["expiry"] = pd.to_datetime(
        contract.lastTradeDateOrContractMonth[:8],
        format="%Y%m%d",
        errors="coerce",
    )
    out["opt_type"] = str(contract.right).upper()
    out["strike"] = float(contract.strike)
    return out


def fetch_active_contract_history(
    ib: IB,
    contract,
    end: str,
    pacing_seconds: float = 0.15,
) -> pd.DataFrame:
    """
    Efficient best-effort request for one still-active option contract.

    Primary request:
      1 Y duration / 8-hour TRADES bars / RTH only.

    8-hour intraday bars avoid relying on the unavailable option EOD dataset.
    If IBKR rejects/returns nothing, the function falls back to monthly
    1-hour requests.
    """
    end_ts = pd.Timestamp(end) + pd.Timedelta(days=1)
    end_dt = end_ts.strftime("%Y%m%d 23:59:59 US/Eastern")

    try:
        bars = ib.reqHistoricalData(
            contract,
            endDateTime=end_dt,
            durationStr="1 Y",
            barSizeSetting="8 hours",
            whatToShow="TRADES",
            useRTH=True,
            formatDate=2,
            keepUpToDate=False,
        )
        frame = _history_frame_from_bars(contract, bars)
        if not frame.empty:
            time.sleep(pacing_seconds)
            return frame
    except Exception as exc:
        print(f"[WARN] primary history request failed {contract.localSymbol}: {exc}")

    # Fallback: request one month at a time using 1-hour bars.
    pieces = []
    cursor = end_ts
    for _ in range(13):
        try:
            bars = ib.reqHistoricalData(
                contract,
                endDateTime=cursor.strftime("%Y%m%d 23:59:59 US/Eastern"),
                durationStr="1 M",
                barSizeSetting="1 hour",
                whatToShow="TRADES",
                useRTH=True,
                formatDate=2,
                keepUpToDate=False,
            )
            piece = _history_frame_from_bars(contract, bars)
            if not piece.empty:
                pieces.append(piece)
        except Exception as exc:
            print(
                f"[WARN] fallback history request failed "
                f"{contract.localSymbol} @ {cursor.date()}: {exc}"
            )
        cursor -= pd.DateOffset(months=1)
        time.sleep(max(pacing_seconds, 0.20))

    if not pieces:
        return pd.DataFrame()
    return (
        pd.concat(pieces, ignore_index=True)
        .sort_values("timestamp")
        .drop_duplicates(["timestamp", "conId"], keep="last")
        .reset_index(drop=True)
    )


def history_to_weekly_option_panel(
    raw_history: pd.DataFrame,
    weekly_stock: pd.DataFrame,
    start: str,
    end: str,
    min_volume: float = 1.0,
    min_moneyness: float = 0.85,
    max_moneyness: float = 1.20,
    min_dte: int = 21,
    max_dte: int = 730,
) -> pd.DataFrame:
    """
    Reduce intraday active-contract history to one last traded bar per week.

    Historical Team 8 code can then treat each row's `ts` as the observation
    date for a weekly surface.
    """
    if raw_history.empty:
        return pd.DataFrame(columns=OPTION_HISTORY_COLUMNS)

    frame = raw_history.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    eastern = frame["timestamp"].dt.tz_convert("US/Eastern").dt.tz_localize(None)
    frame["_date"] = eastern.dt.normalize()
    frame["_week"] = eastern.dt.to_period("W-FRI")

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    frame = frame.loc[frame["_date"].between(start_ts, end_ts)].copy()

    # Last available bar for each contract in each Friday-ending week.
    frame = (
        frame.sort_values("timestamp")
        .groupby(["conId", "_week"], sort=True, as_index=False)
        .tail(1)
        .copy()
    )

    stock = weekly_stock.copy()
    stock = stock.rename(columns={"_week": "_week_key"})
    stock["_week_key"] = stock["_week_key"].astype(str)
    frame["_week_key"] = frame["_week"].astype(str)
    frame = frame.merge(
        stock[["_week_key", "target_date", "spot"]],
        on="_week_key",
        how="inner",
    )

    frame["expiry"] = pd.to_datetime(frame["expiry"], errors="coerce")
    frame["dte"] = (frame["expiry"] - frame["target_date"]).dt.days
    frame["moneyness"] = frame["strike"] / frame["spot"]
    frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce").fillna(0.0)

    frame = frame.loc[
        frame["dte"].between(min_dte, max_dte)
        & frame["moneyness"].between(min_moneyness, max_moneyness)
        & frame["close"].gt(0.0)
        & frame["volume"].ge(min_volume)
    ].copy()

    frame["ts"] = pd.to_datetime(frame["target_date"], utc=True)
    frame["expiry"] = frame["expiry"].dt.strftime("%Y-%m-%d")
    frame["source_price_method"] = "IBKR historical intraday TRADES last bar of W-FRI"

    for column in ("open", "high", "low", "close"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["bar_count"] = pd.to_numeric(frame["bar_count"], errors="coerce")

    panel = frame.loc[:, OPTION_HISTORY_COLUMNS].copy()
    panel = panel.sort_values(["ts", "expiry", "strike"]).reset_index(drop=True)
    return panel


def coverage_report(panel: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    if panel.empty:
        report = pd.DataFrame(
            columns=[
                "date", "rows", "expiries", "strikes",
                "min_dte", "max_dte",
            ]
        )
    else:
        tmp = panel.copy()
        tmp["date"] = pd.to_datetime(tmp["ts"], utc=True).dt.strftime("%Y-%m-%d")
        tmp["expiry_dt"] = pd.to_datetime(tmp["expiry"])
        tmp["obs_dt"] = pd.to_datetime(tmp["date"])
        tmp["dte"] = (tmp["expiry_dt"] - tmp["obs_dt"]).dt.days
        report = (
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_path, index=False)
    return report


def current_chain_snapshot(
    ib: IB,
    output_dir: Path,
    min_moneyness: float = 0.85,
    max_moneyness: float = 1.20,
    min_dte: int = 21,
    max_dte: int = 730,
    batch_size: int = 40,
) -> Path:
    """
    Save a full current point-in-time GLD call snapshot.

    Run this once per chosen observation week going forward. Unlike a later
    historical backfill, this preserves contracts before they expire.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stock = ib.qualifyContracts(Stock(TICKER, EXCHANGE, CURRENCY))[0]
    ticker = ib.reqTickers(stock)[0]

    spot_candidates = [ticker.last, ticker.close, ticker.marketPrice()]
    spot = next(
        float(x)
        for x in spot_candidates
        if x is not None and np.isfinite(x) and float(x) > 0.0
    )

    definition = get_option_definition(ib, stock)
    today = pd.Timestamp.now(tz="US/Eastern").normalize().tz_localize(None)
    low_k = spot * min_moneyness
    high_k = spot * max_moneyness

    expiries = []
    for raw in definition.expirations:
        expiry = pd.to_datetime(str(raw), format="%Y%m%d", errors="coerce")
        if pd.isna(expiry):
            continue
        dte = (expiry - today).days
        if min_dte <= dte <= max_dte:
            expiries.append(expiry)
    expiries = sorted(set(expiries))

    strikes = sorted(
        float(k)
        for k in definition.strikes
        if np.isfinite(k) and low_k <= float(k) <= high_k
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
    qualified = qualify_in_batches(ib, contracts)

    rows = []
    as_of = datetime.now(timezone.utc)
    for batch in chunks(qualified, batch_size):
        try:
            tickers = ib.reqTickers(*batch)
        except Exception as exc:
            print(f"[WARN] reqTickers batch failed: {exc}")
            continue

        for t in tickers:
            c = t.contract
            bid = float(t.bid) if t.bid is not None and np.isfinite(t.bid) else np.nan
            ask = float(t.ask) if t.ask is not None and np.isfinite(t.ask) else np.nan
            last = float(t.last) if t.last is not None and np.isfinite(t.last) else np.nan
            midpoint = (bid + ask) / 2.0 if bid > 0.0 and ask > bid else np.nan

            price = last if last > 0.0 else midpoint
            if not np.isfinite(price) or price <= 0.0:
                continue

            model_greeks = getattr(t, "modelGreeks", None)
            rows.append(
                {
                    "ts": as_of.isoformat(),
                    "expiry": pd.to_datetime(
                        c.lastTradeDateOrContractMonth[:8],
                        format="%Y%m%d",
                    ).strftime("%Y-%m-%d"),
                    "opt_type": str(c.right).upper(),
                    "strike": float(c.strike),
                    "close": price,
                    "bid": bid,
                    "ask": ask,
                    "last": last,
                    "volume": float(t.volume)
                    if t.volume is not None and np.isfinite(t.volume)
                    else np.nan,
                    "implied_vol_ib": float(model_greeks.impliedVol)
                    if model_greeks is not None
                    and model_greeks.impliedVol is not None
                    and np.isfinite(model_greeks.impliedVol)
                    else np.nan,
                    "delta_ib": float(model_greeks.delta)
                    if model_greeks is not None
                    and model_greeks.delta is not None
                    and np.isfinite(model_greeks.delta)
                    else np.nan,
                    "vega_ib": float(model_greeks.vega)
                    if model_greeks is not None
                    and model_greeks.vega is not None
                    and np.isfinite(model_greeks.vega)
                    else np.nan,
                    "spot": spot,
                    "conId": int(c.conId),
                    "localSymbol": str(c.localSymbol),
                    "tradingClass": str(c.tradingClass),
                    "source_price_method": "IBKR current last else bid-ask midpoint",
                }
            )
        ib.sleep(0.20)

    snapshot = pd.DataFrame(rows)
    date_tag = as_of.astimezone().strftime("%Y-%m-%d")
    path = output_dir / "snapshots" / f"GLD_options_{date_tag}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.to_parquet(path, index=False)

    meta = {
        "as_of_utc": as_of.isoformat(),
        "symbol": TICKER,
        "spot": spot,
        "rows": int(len(snapshot)),
        "expiries": int(snapshot["expiry"].nunique()) if not snapshot.empty else 0,
        "strikes": int(snapshot["strike"].nunique()) if not snapshot.empty else 0,
        "purpose": "point-in-time weekly surface preservation",
    }
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path


def run_stock(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ib = connect_ib(args.host, args.port, args.client_id)
    try:
        history = fetch_stock_history(ib, args.start, args.end)
        path = out / "gld_daily_history.csv"
        history.to_csv(path, index=False)
        weekly = weekly_last_stock_dates(history)
        weekly.to_csv(out / "gld_weekly_dates.csv", index=False)
        print(f"[OK] stock history: {path}")
        print(
            f"[OK] stock dates: {pd.to_datetime(history['timestamp']).min().date()} "
            f"-> {pd.to_datetime(history['timestamp']).max().date()}"
        )
        print(f"[OK] weekly buckets: {len(weekly)}")
        if not weekly.empty:
            print(
                f"[OK] first weekly observation: "
                f"{pd.to_datetime(weekly['target_date']).min().date()}"
            )
            print(
                f"[OK] last weekly observation: "
                f"{pd.to_datetime(weekly['target_date']).max().date()}"
            )
    finally:
        ib.disconnect()


def run_rates(args):
    out = Path(args.output_dir)
    path = out / "usd_treasury_history.csv"
    history = normalise_treasury_csv(args.treasury_csv, path)
    dates = pd.to_datetime(history["date"])
    print(f"[OK] Treasury history: {path}")
    print(
        f"[OK] {dates.nunique()} dates, "
        f"{history['symbol'].nunique()} tenors, "
        f"{dates.min().date()} -> {dates.max().date()}"
    )


def run_backfill(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    stock_path = out / "gld_daily_history.csv"
    if stock_path.exists():
        stock_history = pd.read_csv(stock_path)
    else:
        ib_tmp = connect_ib(args.host, args.port, args.client_id)
        try:
            stock_history = fetch_stock_history(ib_tmp, args.start, args.end)
        finally:
            ib_tmp.disconnect()
        stock_history.to_csv(stock_path, index=False)

    ib = connect_ib(args.host, args.port, args.client_id)
    try:
        contracts, weekly = build_active_candidate_contracts(
            ib,
            stock_history,
            args.start,
            args.end,
            min_moneyness=args.min_moneyness,
            max_moneyness=args.max_moneyness,
            max_dte=args.max_dte,
        )

        raw_parts = []
        for number, contract in enumerate(contracts, start=1):
            print(
                f"[{number}/{len(contracts)}] "
                f"{contract.localSymbol or contract.conId}"
            )
            history = fetch_active_contract_history(
                ib,
                contract,
                args.end,
                pacing_seconds=args.pacing_seconds,
            )
            if not history.empty:
                raw_parts.append(history)

        if raw_parts:
            raw = pd.concat(raw_parts, ignore_index=True)
            raw = raw.sort_values(["timestamp", "conId"]).reset_index(drop=True)
        else:
            raw = pd.DataFrame()

        raw_path = out / "options_GLD_active_contract_intraday.parquet"
        raw.to_parquet(raw_path, index=False)

        panel = history_to_weekly_option_panel(
            raw,
            weekly,
            args.start,
            args.end,
            min_volume=args.min_volume,
            min_moneyness=args.min_moneyness,
            max_moneyness=args.max_moneyness,
            min_dte=args.min_dte,
            max_dte=args.max_dte,
        )
        panel_path = out / "options_GLD_weekly.parquet"
        panel.to_parquet(panel_path, index=False)

        report = coverage_report(
            panel,
            out / "options_GLD_weekly_coverage.csv",
        )

        print(f"[OK] raw active-contract history: {raw_path}")
        print(f"[OK] weekly option panel: {panel_path}")
        print(f"[OK] weekly rows: {len(panel)}")
        if not report.empty:
            print(
                f"[OK] covered weeks: {report['date'].nunique()} / "
                f"{len(weekly)} target weeks"
            )
            print(report.tail(10).to_string(index=False))
        else:
            print("[WARN] No usable weekly option rows were reconstructed.")
        print(
            "\n[IMPORTANT] This is NOT guaranteed to be a complete historical "
            "surface panel because expired options cannot be recovered from IBKR."
        )
    finally:
        ib.disconnect()


def run_snapshot(args):
    ib = connect_ib(args.host, args.port, args.client_id)
    try:
        path = current_chain_snapshot(
            ib,
            Path(args.output_dir),
            min_moneyness=args.min_moneyness,
            max_moneyness=args.max_moneyness,
            min_dte=args.min_dte,
            max_dte=args.max_dte,
        )
        print(f"[OK] current weekly snapshot saved: {path}")
    finally:
        ib.disconnect()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "mode",
        choices=["rates", "stock", "backfill-active", "snapshot"],
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7497)
    p.add_argument("--client-id", type=int, default=81)
    p.add_argument("--start", default="2026-01-02")
    p.add_argument("--end", default="2026-09-02")
    p.add_argument("--output-dir", default=str(DEFAULT_OUT))
    p.add_argument("--treasury-csv", default=str(DEFAULT_TREASURY))
    p.add_argument("--min-moneyness", type=float, default=0.85)
    p.add_argument("--max-moneyness", type=float, default=1.20)
    p.add_argument("--min-dte", type=int, default=21)
    p.add_argument("--max-dte", type=int, default=730)
    p.add_argument("--min-volume", type=float, default=1.0)
    p.add_argument("--pacing-seconds", type=float, default=0.15)
    return p


def main():
    args = parser().parse_args()
    if args.mode == "rates":
        run_rates(args)
    elif args.mode == "stock":
        run_stock(args)
    elif args.mode == "backfill-active":
        run_backfill(args)
    elif args.mode == "snapshot":
        run_snapshot(args)
    else:
        raise ValueError(args.mode)


if __name__ == "__main__":
    main()
