"""
Adaptive historical GLD option-surface fetch from Interactive Brokers.

Goal
----
Recover a sufficiently rich historical GLD call surface while avoiding the
very expensive "all expiries x all strikes" Cartesian product.

Default strategy
----------------
1. Keep ALL currently-queryable expiries in the requested DTE domain.
2. On strikes, select a fixed grid in moneyness K/S:
       0.80 ... 1.20
   using 17 equally-spaced targets by default.
3. Map each target to the nearest real IBKR strike.
4. Fetch historical MIDPOINT bars only for those contracts.
5. If fewer than --target-eligible eligible IV observations are obtained,
   optionally refine the strike grid (33 targets by default) and fetch only
   the new strikes.
6. Save checkpoints during the download. Re-running the same date resumes
   from already recovered conIds instead of starting from zero.

Important
---------
This is NOT a literal full-strike surface. It is an adaptive reduced surface
designed to preserve all maturities while dramatically reducing IBKR calls.

Expired-option limitation still applies: contracts already expired today
cannot generally be recovered from the current option chain.
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
        preferred = [
            c for c in chains
            if str(c.tradingClass).upper() == "GLD"
        ]
    if not preferred:
        preferred = chains

    return max(
        preferred,
        key=lambda c: (len(c.expirations), len(c.strikes))
    )


def select_nearest_strikes(all_strikes, spot, min_m, max_m, n_targets):
    """
    Select real IBKR strikes nearest to an equally-spaced moneyness grid.
    Duplicate mappings are removed.
    """
    strikes = np.asarray(sorted(set(float(x) for x in all_strikes)), dtype=float)
    strikes = strikes[
        np.isfinite(strikes)
        & (strikes >= spot * min_m)
        & (strikes <= spot * max_m)
    ]

    if len(strikes) == 0:
        return []

    targets = np.linspace(float(min_m), float(max_m), int(n_targets)) * float(spot)

    selected = []
    for target in targets:
        idx = int(np.argmin(np.abs(strikes - target)))
        selected.append(float(strikes[idx]))

    return sorted(set(selected))


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

    frame = frame.loc[
        frame["market_date"].eq(target)
    ].sort_values("timestamp")

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


def load_existing_raw(path):
    if not path.exists():
        return pd.DataFrame()

    try:
        frame = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

    if frame.empty:
        return frame

    if "conId" in frame.columns:
        frame["conId"] = pd.to_numeric(
            frame["conId"], errors="coerce"
        ).astype("Int64")

    return frame


def save_checkpoint(rows, path):
    if not rows:
        return

    frame = pd.DataFrame(rows)
    if frame.empty:
        return

    if "conId" in frame.columns:
        frame["conId"] = pd.to_numeric(
            frame["conId"], errors="coerce"
        )
        frame = frame.drop_duplicates("conId", keep="last")

    frame.to_csv(path, index=False)


def enrich_and_filter(
    raw,
    target,
    spot,
    rate_history,
    min_price,
    min_iv,
    max_iv,
    min_vega,
):
    if raw.empty:
        return raw.copy(), raw.copy(), None

    raw = raw.copy()

    raw["date"] = pd.to_datetime(raw["date"], errors="coerce").dt.normalize()
    raw["expiry"] = pd.to_datetime(raw["expiry"], errors="coerce").dt.normalize()
    raw["K"] = pd.to_numeric(raw["K"], errors="coerce")
    raw["price"] = pd.to_numeric(raw["price"], errors="coerce")

    raw = raw.dropna(subset=["expiry", "K", "price"]).copy()
    raw = raw.loc[raw["price"].gt(float(min_price))].copy()

    raw["T"] = (raw["expiry"] - target).dt.days / 365.25
    raw["dte"] = (raw["expiry"] - target).dt.days
    raw["moneyness"] = raw["K"] / float(spot)

    rates, curve_date = rates_for_date(
        raw["T"].to_numpy(float),
        target,
        rate_history=rate_history,
    )
    raw["rate"] = rates

    ivs = []
    vegas = []

    for row in raw.itertuples(index=False):
        iv = BnS.implied_vol_call(
            row.price, float(spot), row.K, row.T, row.rate
        )
        ivs.append(iv)

        if np.isfinite(iv):
            vegas.append(
                BnS.calculate_bs_vega(
                    float(spot),
                    row.K,
                    row.T,
                    row.rate,
                    0.0,
                    iv,
                )
            )
        else:
            vegas.append(np.nan)

    raw["implied_vol"] = ivs
    raw["vega"] = vegas
    raw["spot"] = float(spot)
    raw["curve_date"] = pd.Timestamp(curve_date)

    eligible = raw.loc[
        np.isfinite(raw["implied_vol"])
        & raw["implied_vol"].between(float(min_iv), float(max_iv))
        & np.isfinite(raw["vega"])
        & raw["vega"].ge(float(min_vega))
    ].copy()

    eligible = (
        eligible
        .sort_values(["T", "K"])
        .drop_duplicates(["T", "K"], keep="last")
        .reset_index(drop=True)
    )

    return raw, eligible, curve_date


def build_contracts(expiries, strikes, trading_class):
    return [
        Option(
            "GLD",
            expiry.strftime("%Y%m%d"),
            strike,
            "C",
            "SMART",
            currency="USD",
            tradingClass=trading_class,
        )
        for expiry in expiries
        for strike in strikes
    ]


def fetch_stage(
    ib,
    contracts,
    target,
    pacing,
    recovered_rows,
    checkpoint_path,
    checkpoint_every,
):
    """
    Fetch one set of qualified contracts.
    Existing recovered conIds are skipped.
    """
    existing_ids = {
        int(x)
        for x in pd.to_numeric(
            pd.DataFrame(recovered_rows).get("conId", pd.Series(dtype=float)),
            errors="coerce",
        ).dropna()
    }

    todo = [
        c for c in contracts
        if int(getattr(c, "conId", 0) or 0) not in existing_ids
    ]

    print(
        f"[INFO] already recovered: {len(existing_ids)} | "
        f"new contracts to query: {len(todo)}"
    )

    failures = 0
    attempted = 0

    for i, contract in enumerate(todo, start=1):
        print(f"[{i}/{len(todo)}] {contract.localSymbol}")

        row, error = historical_midpoint(
            ib, contract, target, pacing
        )
        attempted += 1

        if row is not None:
            recovered_rows.append(row)
        else:
            failures += 1

        if int(checkpoint_every) > 0 and attempted % int(checkpoint_every) == 0:
            save_checkpoint(recovered_rows, checkpoint_path)
            print(
                f"[CHECKPOINT] recovered={len(recovered_rows)} "
                f"attempted={attempted} failed={failures}"
            )

    save_checkpoint(recovered_rows, checkpoint_path)

    return failures, attempted


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--date", required=True)

    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7497)
    p.add_argument("--client-id", type=int, default=93)

    p.add_argument(
        "--stock",
        default="data/processed/gld_daily_history.csv",
    )
    p.add_argument(
        "--rates",
        default="data/processed/usd_treasury_history.csv",
    )
    p.add_argument(
        "--output-dir",
        default="data/processed/full_surfaces",
    )

    # Cross-sectional domain
    p.add_argument("--min-moneyness", type=float, default=0.80)
    p.add_argument("--max-moneyness", type=float, default=1.20)

    # 75 days ~= 0.205 years, close to T = 0.21
    p.add_argument("--min-dte", type=int, default=75)
    p.add_argument("--max-dte", type=int, default=730)

    # Fast strike grid
    p.add_argument(
        "--n-strikes",
        type=int,
        default=17,
        help="Initial number of equally-spaced moneyness targets.",
    )
    p.add_argument(
        "--refined-n-strikes",
        type=int,
        default=33,
        help="Strike targets used by the optional refinement stage.",
    )
    p.add_argument(
        "--target-eligible",
        type=int,
        default=100,
        help="Refine only if eligible IV points are below this threshold.",
    )
    p.add_argument(
        "--no-refine",
        action="store_true",
        help="Disable the second, denser strike-grid stage.",
    )

    # Eligibility filters
    p.add_argument("--min-price", type=float, default=0.05)
    p.add_argument("--min-iv", type=float, default=0.03)
    p.add_argument("--max-iv", type=float, default=1.50)
    p.add_argument("--min-vega", type=float, default=0.10)

    # IBKR / checkpoint controls
    p.add_argument("--pacing-seconds", type=float, default=0.15)
    p.add_argument("--checkpoint-every", type=int, default=25)

    p.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore an existing checkpoint/raw file and start from zero.",
    )

    args = p.parse_args()

    if args.n_strikes < 2:
        raise ValueError("--n-strikes must be >= 2.")
    if args.refined_n_strikes < args.n_strikes:
        raise ValueError(
            "--refined-n-strikes must be >= --n-strikes."
        )

    target = pd.Timestamp(args.date).normalize()
    spot = get_spot(args.stock, target)
    today = pd.Timestamp.now().normalize()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    slug = target.strftime("%Y-%m-%d")

    # Keep the old raw name so interrupted runs can resume easily.
    raw_path = out / f"GLD_{slug}_midpoint_raw.csv"

    # New name: this is not literally every strike in the chain.
    eligible_path = out / f"GLD_{slug}_eligible_adaptive_surface.csv"

    if args.fresh and raw_path.exists():
        print(f"[INFO] --fresh: deleting old checkpoint {raw_path}")
        raw_path.unlink()

    existing_raw = load_existing_raw(raw_path)

    recovered_rows = (
        existing_raw.to_dict("records")
        if not existing_raw.empty
        else []
    )

    rate_history = load_rate_history(args.rates)

    ib = connect(args.host, args.port, args.client_id)

    try:
        underlying = ib.qualifyContracts(
            Stock("GLD", "SMART", "USD")
        )[0]

        chain = get_chain(ib, underlying)

        expiries = []
        expired_but_relevant = 0

        for raw_expiry in sorted(chain.expirations):
            expiry = pd.to_datetime(
                str(raw_expiry),
                format="%Y%m%d",
                errors="coerce",
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

        all_strikes_in_range = sorted(
            float(k)
            for k in chain.strikes
            if np.isfinite(k)
            and spot * args.min_moneyness
            <= float(k)
            <= spot * args.max_moneyness
        )

        initial_strikes = select_nearest_strikes(
            all_strikes_in_range,
            spot,
            args.min_moneyness,
            args.max_moneyness,
            args.n_strikes,
        )

        print("=" * 82)
        print(f"TARGET DATE                    : {target.date()}")
        print(f"GLD SPOT                       : {spot:.4f}")
        print(
            f"DTE DOMAIN                     : "
            f"{args.min_dte} -> {args.max_dte} days"
        )
        print(f"QUERYABLE EXPIRIES             : {len(expiries)}")
        print(
            f"EXPIRED RELEVANT EXPIRIES LOST : "
            f"{expired_but_relevant}"
        )
        print(
            f"ALL STRIKES IN RANGE           : "
            f"{len(all_strikes_in_range)}"
        )
        print(
            f"INITIAL STRIKES SELECTED        : "
            f"{len(initial_strikes)}"
        )
        print(
            f"INITIAL RAW CROSS PRODUCT       : "
            f"{len(expiries) * len(initial_strikes)} candidates"
        )
        print(
            f"RESUME RECOVERED ROWS           : "
            f"{len(recovered_rows)}"
        )
        print("=" * 82)

        # -----------------------------
        # Stage 1: sparse strike grid
        # -----------------------------
        stage1_candidates = build_contracts(
            expiries,
            initial_strikes,
            chain.tradingClass,
        )

        stage1_contracts = qualify(
            ib,
            stage1_candidates,
        )

        print(
            f"[INFO] stage-1 qualified contracts: "
            f"{len(stage1_contracts)}"
        )

        failures1, attempted1 = fetch_stage(
            ib=ib,
            contracts=stage1_contracts,
            target=target,
            pacing=args.pacing_seconds,
            recovered_rows=recovered_rows,
            checkpoint_path=raw_path,
            checkpoint_every=args.checkpoint_every,
        )

        stage_raw = pd.DataFrame(recovered_rows)

        enriched, eligible, curve_date = enrich_and_filter(
            stage_raw,
            target,
            spot,
            rate_history,
            args.min_price,
            args.min_iv,
            args.max_iv,
            args.min_vega,
        )

        enriched.to_csv(raw_path, index=False)
        eligible.to_csv(eligible_path, index=False)

        print()
        print("-" * 82)
        print(
            f"[STAGE 1] midpoint observations : {len(enriched)}"
        )
        print(
            f"[STAGE 1] eligible IV points    : {len(eligible)}"
        )
        print(
            f"[STAGE 1] unique expiries       : "
            f"{eligible['expiry'].nunique() if not eligible.empty else 0}"
        )
        print(
            f"[STAGE 1] unique strikes        : "
            f"{eligible['K'].nunique() if not eligible.empty else 0}"
        )
        print(
            f"[STAGE 1] failed history calls  : {failures1}"
        )
        print("-" * 82)

        # --------------------------------------
        # Stage 2: refine only when necessary
        # --------------------------------------
        total_failures = failures1
        total_attempted = attempted1

        if (
            not args.no_refine
            and len(eligible) < int(args.target_eligible)
            and len(all_strikes_in_range) > len(initial_strikes)
        ):
            refined_strikes = select_nearest_strikes(
                all_strikes_in_range,
                spot,
                args.min_moneyness,
                args.max_moneyness,
                args.refined_n_strikes,
            )

            extra_strikes = sorted(
                set(refined_strikes) - set(initial_strikes)
            )

            print()
            print("=" * 82)
            print(
                f"[REFINE] eligible {len(eligible)} "
                f"< target {args.target_eligible}"
            )
            print(
                f"[REFINE] extra strikes          : "
                f"{len(extra_strikes)}"
            )
            print(
                f"[REFINE] extra raw candidates   : "
                f"{len(expiries) * len(extra_strikes)}"
            )
            print("=" * 82)

            stage2_candidates = build_contracts(
                expiries,
                extra_strikes,
                chain.tradingClass,
            )

            stage2_contracts = qualify(
                ib,
                stage2_candidates,
            )

            print(
                f"[INFO] stage-2 qualified contracts: "
                f"{len(stage2_contracts)}"
            )

            failures2, attempted2 = fetch_stage(
                ib=ib,
                contracts=stage2_contracts,
                target=target,
                pacing=args.pacing_seconds,
                recovered_rows=recovered_rows,
                checkpoint_path=raw_path,
                checkpoint_every=args.checkpoint_every,
            )

            total_failures += failures2
            total_attempted += attempted2

            stage_raw = pd.DataFrame(recovered_rows)

            enriched, eligible, curve_date = enrich_and_filter(
                stage_raw,
                target,
                spot,
                rate_history,
                args.min_price,
                args.min_iv,
                args.max_iv,
                args.min_vega,
            )

            enriched.to_csv(raw_path, index=False)
            eligible.to_csv(eligible_path, index=False)

        print()
        print("=" * 82)
        print(f"[OK] midpoint observations : {len(enriched)}")
        print(f"[OK] eligible IV points    : {len(eligible)}")
        print(
            f"[OK] unique expiries       : "
            f"{eligible['expiry'].nunique() if not eligible.empty else 0}"
        )
        print(
            f"[OK] unique strikes        : "
            f"{eligible['K'].nunique() if not eligible.empty else 0}"
        )
        print(f"[OK] history calls this run: {total_attempted}")
        print(f"[OK] failed history calls  : {total_failures}")
        print(f"[OK] raw/checkpoint        : {raw_path}")
        print(f"[OK] adaptive surface      : {eligible_path}")
        print("=" * 82)

        if len(eligible) < 64:
            print(
                "[WARN] Fewer than 64 eligible points remain. "
                "Use all available observations for calibration."
            )
        else:
            print(
                "[OK] >=64 eligible points: the date is usable "
                "for an 8x8 sampling experiment."
            )

    finally:
        ib.disconnect()


if __name__ == "__main__":
    main()
