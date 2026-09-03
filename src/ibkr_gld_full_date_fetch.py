"""
Fetch the widest currently-recoverable GLD call surface for ONE historical date.

Unlike the quick daily-60 coverage fetch, there is NO 6-expiry / 24-strike cap.
All currently queryable expiries and strikes inside the requested target-date
domain are attempted.

Historical MIDPOINT is used as the primary cross-sectional price because many
valid strikes may have no trade on a particular day. IBKR's expired-option
limitation still applies: contracts already expired today cannot be recovered.

This file is retained for reproducibility. The current offline reconstruction
from options_GLD_daily_60.parquet does not call this script.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import pandas as pd
from ib_insync import IB, Option, Stock, util

from BnS import BnS
from rates import load_rate_history, rates_for_date


def connect(host, port, client_id):
    ib = IB()
    ib.connect(
        host, int(port), clientId=int(client_id), readonly=True, timeout=15
    )
    if not ib.isConnected():
        raise RuntimeError("IBKR API connection failed.")
    return ib


def get_spot(path, target):
    frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(
        frame["timestamp"], errors="coerce"
    ).dt.normalize()
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    row = frame.loc[frame["date"].eq(target)].dropna(subset=["close"])
    if row.empty:
        raise ValueError(f"No GLD close for {target.date()}.")
    return float(row.iloc[-1]["close"])


def get_chain(ib, underlying):
    chains = ib.reqSecDefOptParams(
        underlying.symbol, "", underlying.secType, underlying.conId
    )
    if not chains:
        raise RuntimeError("No GLD option chain returned.")
    preferred = [
        c for c in chains
        if str(c.tradingClass).upper() == "GLD"
        and c.exchange in {"SMART", ""}
    ]
    if not preferred:
        preferred = [c for c in chains if str(c.tradingClass).upper() == "GLD"]
    if not preferred:
        preferred = chains
    return max(preferred, key=lambda c: (len(c.expirations), len(c.strikes)))


def qualify(ib, contracts, batch=40):
    result = {}
    for start in range(0, len(contracts), int(batch)):
        part = contracts[start:start + int(batch)]
        try:
            valid = ib.qualifyContracts(*part)
        except Exception as exc:
            print(f"[WARN] qualification error: {exc}")
            valid = []
        for contract in valid:
            if getattr(contract, "conId", 0):
                result[int(contract.conId)] = contract
        ib.sleep(0.1)
        print(
            f"[QUALIFY] {min(start + batch, len(contracts))}/{len(contracts)} "
            f"candidates; {len(result)} valid"
        )
    return list(result.values())


def historical_midpoint(ib, contract, target, pacing):
    end = target.strftime("%Y%m%d 23:59:59 US/Eastern")
    try:
        bars = ib.reqHistoricalData(
            contract,
            endDateTime=end,
            durationStr="2 D",
            barSizeSetting="1 hour",
            whatToShow="MIDPOINT",
            useRTH=True,
            formatDate=2,
            keepUpToDate=False,
        )
    except Exception as exc:
        return None, str(exc)

    if pacing:
        time.sleep(float(pacing))

    frame = util.df(bars)
    if frame is None or frame.empty:
        return None, "no bars"

    frame["timestamp"] = pd.to_datetime(
        frame["date"], errors="coerce", utc=True
    )
    frame["market_date"] = (
        frame["timestamp"]
        .dt.tz_convert("US/Eastern")
        .dt.tz_localize(None)
        .dt.normalize()
    )
    frame = frame.loc[frame["market_date"].eq(target)].sort_values("timestamp")
    if frame.empty:
        return None, "no target-date midpoint"

    row = frame.iloc[-1]
    price = float(row["close"])
    if not np.isfinite(price) or price <= 0:
        return None, "invalid midpoint"

    expiry = pd.to_datetime(
        contract.lastTradeDateOrContractMonth[:8],
        format="%Y%m%d",
        errors="coerce",
    )
    if pd.isna(expiry):
        return None, "invalid expiry"

    return {
        "date": target,
        "timestamp": row["timestamp"],
        "expiry": expiry,
        "K": float(contract.strike),
        "price": price,
        "conId": int(contract.conId),
        "localSymbol": str(contract.localSymbol),
    }, None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--date", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7497)
    p.add_argument("--client-id", type=int, default=93)
    p.add_argument("--stock", default="data/processed/gld_daily_history.csv")
    p.add_argument("--rates", default="data/processed/usd_treasury_history.csv")
    p.add_argument("--output-dir", default="data/processed/full_surfaces")
    p.add_argument("--min-moneyness", type=float, default=0.80)
    p.add_argument("--max-moneyness", type=float, default=1.20)
    p.add_argument("--min-dte", type=int, default=21)
    p.add_argument("--max-dte", type=int, default=730)
    p.add_argument("--min-price", type=float, default=0.05)
    p.add_argument("--min-iv", type=float, default=0.03)
    p.add_argument("--max-iv", type=float, default=1.50)
    p.add_argument("--min-vega", type=float, default=0.10)
    p.add_argument("--pacing-seconds", type=float, default=0.15)
    args = p.parse_args()

    target = pd.Timestamp(args.date).normalize()
    spot = get_spot(args.stock, target)
    today = pd.Timestamp.now().normalize()

    ib = connect(args.host, args.port, args.client_id)
    try:
        underlying = ib.qualifyContracts(Stock("GLD", "SMART", "USD"))[0]
        chain = get_chain(ib, underlying)

        expiries = []
        expired_but_relevant = 0
        for raw in sorted(chain.expirations):
            expiry = pd.to_datetime(
                str(raw), format="%Y%m%d", errors="coerce"
            )
            if pd.isna(expiry):
                continue
            dte = int((expiry - target).days)
            if not args.min_dte <= dte <= args.max_dte:
                continue
            if expiry < today:
                expired_but_relevant += 1
                continue
            expiries.append(expiry)

        strikes = sorted(
            float(k)
            for k in chain.strikes
            if np.isfinite(k)
            and spot * args.min_moneyness
            <= float(k)
            <= spot * args.max_moneyness
        )

        print("=" * 78)
        print(f"TARGET DATE                   : {target.date()}")
        print(f"GLD SPOT                      : {spot:.4f}")
        print(f"QUERYABLE EXPIRIES            : {len(expiries)}")
        print(f"EXPIRED RELEVANT EXPIRIES LOST: {expired_but_relevant}")
        print(f"STRIKES IN RANGE              : {len(strikes)}")
        print(
            f"RAW CROSS PRODUCT             : "
            f"{len(expiries) * len(strikes)} candidates"
        )
        print("=" * 78)

        contracts = [
            Option(
                "GLD",
                expiry.strftime("%Y%m%d"),
                strike,
                "C",
                "SMART",
                currency="USD",
                tradingClass=chain.tradingClass,
            )
            for expiry in expiries
            for strike in strikes
        ]
        contracts = qualify(ib, contracts)
        print(f"[INFO] qualified contracts: {len(contracts)}")

        rows = []
        failures = 0
        for i, contract in enumerate(contracts, start=1):
            print(f"[{i}/{len(contracts)}] {contract.localSymbol}")
            row, error = historical_midpoint(
                ib, contract, target, args.pacing_seconds
            )
            if row is not None:
                rows.append(row)
            else:
                failures += 1

        raw = pd.DataFrame(rows)
        if raw.empty:
            raise RuntimeError("No target-date MIDPOINT observations recovered.")

        raw["T"] = (raw["expiry"] - target).dt.days / 365.25
        raw["dte"] = (raw["expiry"] - target).dt.days
        raw["moneyness"] = raw["K"] / spot
        raw = raw.loc[raw["price"].gt(args.min_price)].copy()

        rate_history = load_rate_history(args.rates)
        rates, curve_date = rates_for_date(
            raw["T"].to_numpy(float),
            target,
            rate_history=rate_history,
        )
        raw["rate"] = rates

        ivs, vegas = [], []
        for row in raw.itertuples(index=False):
            iv = BnS.implied_vol_call(
                row.price, spot, row.K, row.T, row.rate
            )
            ivs.append(iv)
            if np.isfinite(iv):
                vegas.append(
                    BnS.calculate_bs_vega(
                        spot, row.K, row.T, row.rate, 0.0, iv
                    )
                )
            else:
                vegas.append(np.nan)

        raw["implied_vol"] = ivs
        raw["vega"] = vegas
        raw["spot"] = spot
        raw["curve_date"] = pd.Timestamp(curve_date)

        eligible = raw.loc[
            np.isfinite(raw["implied_vol"])
            & raw["implied_vol"].between(args.min_iv, args.max_iv)
            & np.isfinite(raw["vega"])
            & raw["vega"].ge(args.min_vega)
        ].copy()
        eligible = (
            eligible.sort_values(["T", "K"])
            .drop_duplicates(["T", "K"], keep="last")
            .reset_index(drop=True)
        )

        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        slug = target.strftime("%Y-%m-%d")

        raw.to_csv(out / f"GLD_{slug}_midpoint_raw.csv", index=False)
        eligible.to_csv(
            out / f"GLD_{slug}_eligible_full_surface.csv", index=False
        )

        print()
        print("=" * 78)
        print(f"[OK] midpoint observations : {len(raw)}")
        print(f"[OK] eligible IV points    : {len(eligible)}")
        print(f"[OK] unique expiries       : {eligible['expiry'].nunique()}")
        print(f"[OK] unique strikes        : {eligible['K'].nunique()}")
        print(f"[OK] failed history calls  : {failures}")
        print(
            f"[OK] full surface          : "
            f"{out / f'GLD_{slug}_eligible_full_surface.csv'}"
        )
        print("=" * 78)

        if len(eligible) < 64:
            print(
                "[WARN] Fewer than 64 eligible points remain. "
                "Do NOT calibrate a 64-node sample on this date."
            )
        else:
            print(
                "[OK] >=64 eligible points: run compare_sampling.py next."
            )

    finally:
        ib.disconnect()


if __name__ == "__main__":
    main()
