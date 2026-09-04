"""
Current-day GLD option surface from Interactive Brokers.

Purpose
-------
Collect a dense CURRENT GLD call surface for today's date. The script:
- uses the live/current GLD option chain;
- keeps expiries in the requested DTE domain;
- keeps strikes in a moneyness band around the current GLD spot;
- targets up to --max-contracts contracts (default 2000), balanced across expiries;
- requests current quotes in batches;
- uses midpoint when valid, otherwise last/close;
- recomputes Black-Scholes implied volatility and vega;
- uses the project's no-look-ahead NSS Treasury curve;
- writes the official current eligible full surface.

This script is intentionally for TODAY only. Historical dates should use
ibkr_gld_historical_surface.py.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from ib_insync import IB, Option, Stock

from BnS import BnS
from rates import load_rate_history, rates_for_date


TICKER = "GLD"
EXCHANGE = "SMART"
CURRENCY = "USD"


def connect(host, port, client_id):
    ib = IB()
    ib.connect(host, int(port), clientId=int(client_id), readonly=True, timeout=15)
    if not ib.isConnected():
        raise RuntimeError("IBKR API connection failed.")
    return ib


def chunks(values, size):
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
        ib.sleep(0.10)
        print(f"[QUALIFY] valid={len(unique)}")
    return list(unique.values())


def get_chain(ib, stock):
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


def finite_positive(x):
    try:
        x = float(x)
    except Exception:
        return None
    return x if np.isfinite(x) and x > 0 else None


def get_current_spot(ib, stock, stock_history_path):
    t = ib.reqTickers(stock)[0]
    for candidate in [t.marketPrice(), t.last, t.close]:
        value = finite_positive(candidate)
        if value is not None:
            return value, "IBKR current quote"

    path = Path(stock_history_path)
    if path.exists():
        frame = pd.read_csv(path)
        if "close" in frame.columns:
            close = pd.to_numeric(frame["close"], errors="coerce").dropna()
            close = close.loc[close.gt(0)]
            if not close.empty:
                return float(close.iloc[-1]), "last saved GLD close"

    raise RuntimeError("Could not determine a positive GLD spot.")


def evenly_spaced_actual_strikes(strikes, n):
    strikes = np.asarray(sorted(set(float(x) for x in strikes)), dtype=float)
    if len(strikes) <= int(n):
        return strikes.tolist()
    idx = np.linspace(0, len(strikes) - 1, int(n))
    idx = np.unique(np.round(idx).astype(int))
    return strikes[idx].tolist()


def choose_contract_grid(expiries, strikes, max_contracts):
    expiries = list(expiries)
    strikes = list(strikes)
    raw_count = len(expiries) * len(strikes)

    if raw_count <= max_contracts:
        return expiries, strikes

    if not expiries:
        return expiries, []

    n_strikes = max(1, int(max_contracts) // len(expiries))
    n_strikes = min(n_strikes, len(strikes))
    selected = evenly_spaced_actual_strikes(strikes, n_strikes)
    return expiries, selected


def quote_price(ticker):
    bid = finite_positive(ticker.bid)
    ask = finite_positive(ticker.ask)
    last = finite_positive(ticker.last)
    close = finite_positive(ticker.close)

    midpoint = None
    if bid is not None and ask is not None and ask >= bid:
        midpoint = 0.5 * (bid + ask)
        if midpoint <= 0:
            midpoint = None

    price = midpoint or last or close
    return bid, ask, last, close, midpoint, price


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7497)
    p.add_argument("--client-id", type=int, default=91)

    p.add_argument("--rates", default="data/processed/usd_treasury_history.csv")
    p.add_argument("--stock-history", default="data/processed/gld_daily_history.csv")
    p.add_argument("--output-dir", default="data/processed/full_surfaces")

    p.add_argument("--min-moneyness", type=float, default=0.80)
    p.add_argument("--max-moneyness", type=float, default=1.20)
    p.add_argument("--min-dte", type=int, default=75)
    p.add_argument("--max-dte", type=int, default=730)

    p.add_argument("--max-contracts", type=int, default=2000)
    p.add_argument("--ticker-batch", type=int, default=40)

    p.add_argument("--min-price", type=float, default=0.05)
    p.add_argument("--min-iv", type=float, default=0.03)
    p.add_argument("--max-iv", type=float, default=1.50)
    p.add_argument("--min-vega", type=float, default=0.10)

    args = p.parse_args()

    if args.max_contracts < 1:
        raise ValueError("--max-contracts must be >= 1")

    target = pd.Timestamp.now().normalize()
    as_of_utc = datetime.now(timezone.utc)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    slug = target.strftime("%Y-%m-%d")

    raw_path = out / f"GLD_{slug}_current_snapshot_raw.csv"
    eligible_path = out / f"GLD_{slug}_eligible_full_surface.csv"

    rate_history = load_rate_history(args.rates)

    ib = connect(args.host, args.port, args.client_id)
    try:
        stock = ib.qualifyContracts(Stock(TICKER, EXCHANGE, CURRENCY))[0]
        spot, spot_source = get_current_spot(
            ib, stock, args.stock_history
        )
        chain = get_chain(ib, stock)

        expiries = []
        for raw_expiry in sorted(chain.expirations):
            expiry = pd.to_datetime(
                str(raw_expiry), format="%Y%m%d", errors="coerce"
            )
            if pd.isna(expiry):
                continue
            dte = int((expiry - target).days)
            if args.min_dte <= dte <= args.max_dte:
                expiries.append(expiry)
        expiries = sorted(set(expiries))

        strikes = sorted({
            float(k)
            for k in chain.strikes
            if np.isfinite(k)
            and spot * args.min_moneyness <= float(k) <= spot * args.max_moneyness
        })

        raw_cross = len(expiries) * len(strikes)
        expiries, strikes = choose_contract_grid(
            expiries, strikes, args.max_contracts
        )
        selected_cross = len(expiries) * len(strikes)

        print("=" * 86)
        print(f"TODAY                       : {target.date()}")
        print(f"GLD SPOT                    : {spot:.4f} ({spot_source})")
        print(f"DTE DOMAIN                  : {args.min_dte} -> {args.max_dte}")
        print(
            f"MONEYNESS DOMAIN            : "
            f"{args.min_moneyness:.2f} -> {args.max_moneyness:.2f}"
        )
        print(f"QUERYABLE EXPIRIES          : {len(expiries)}")
        print(f"STRIKES SELECTED            : {len(strikes)}")
        print(f"FULL CROSS PRODUCT          : {raw_cross}")
        print(f"CONTRACT TARGET/CAP         : {args.max_contracts}")
        print(f"SELECTED CROSS PRODUCT      : {selected_cross}")
        print("=" * 86)

        contracts = [
            Option(
                TICKER,
                expiry.strftime("%Y%m%d"),
                strike,
                "C",
                EXCHANGE,
                currency=CURRENCY,
                tradingClass=chain.tradingClass,
            )
            for expiry in expiries
            for strike in strikes
        ]

        qualified = qualify_in_batches(ib, contracts)
        print(f"[INFO] qualified contracts: {len(qualified)}")

        rows = []
        for batch_no, batch in enumerate(chunks(qualified, args.ticker_batch), start=1):
            try:
                tickers = ib.reqTickers(*batch)
            except Exception as exc:
                print(f"[WARN] ticker batch {batch_no} failed: {exc}")
                tickers = []

            for t in tickers:
                c = t.contract
                bid, ask, last, close, midpoint, price = quote_price(t)
                if price is None:
                    continue

                expiry = pd.to_datetime(
                    c.lastTradeDateOrContractMonth[:8],
                    format="%Y%m%d",
                    errors="coerce",
                )
                if pd.isna(expiry):
                    continue

                dte = int((expiry - target).days)
                if not args.min_dte <= dte <= args.max_dte:
                    continue

                rows.append({
                    "date": target,
                    "as_of_utc": as_of_utc.isoformat(),
                    "expiry": expiry,
                    "K": float(c.strike),
                    "dte": dte,
                    "T": dte / 365.25,
                    "moneyness": float(c.strike) / spot,
                    "spot": spot,
                    "bid": bid,
                    "ask": ask,
                    "last": last,
                    "close": close,
                    "midpoint": midpoint,
                    "price": price,
                    "conId": int(c.conId),
                    "localSymbol": str(c.localSymbol),
                    "tradingClass": str(c.tradingClass),
                })

            ib.sleep(0.15)
            print(
                f"[MARKET] batch={batch_no} "
                f"rows_with_price={len(rows)}"
            )

        raw = pd.DataFrame(rows)
        if raw.empty:
            raise RuntimeError("No current GLD option prices recovered.")

        raw = raw.loc[
            pd.to_numeric(raw["price"], errors="coerce").gt(args.min_price)
        ].copy()

        rates, curve_date = rates_for_date(
            raw["T"].to_numpy(float),
            target,
            rate_history=rate_history,
        )
        raw["rate"] = rates
        raw["curve_date"] = pd.Timestamp(curve_date)

        ivs = []
        vegas = []
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

        eligible = raw.loc[
            np.isfinite(raw["implied_vol"])
            & raw["implied_vol"].between(args.min_iv, args.max_iv)
            & np.isfinite(raw["vega"])
            & raw["vega"].ge(args.min_vega)
        ].copy()

        eligible = (
            eligible
            .sort_values(["T", "K"])
            .drop_duplicates(["T", "K"], keep="last")
            .reset_index(drop=True)
        )

        raw.to_csv(raw_path, index=False)
        eligible.to_csv(eligible_path, index=False)

        print()
        print("=" * 86)
        print(f"[OK] current price observations : {len(raw)}")
        print(f"[OK] eligible IV points         : {len(eligible)}")
        print(f"[OK] unique expiries            : {eligible['expiry'].nunique()}")
        print(f"[OK] unique strikes             : {eligible['K'].nunique()}")
        print(f"[OK] raw snapshot               : {raw_path}")
        print(f"[OK] official full surface      : {eligible_path}")
        print("=" * 86)

    finally:
        if ib.isConnected():
            ib.disconnect()


if __name__ == "__main__":
    main()
