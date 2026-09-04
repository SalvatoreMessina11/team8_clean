"""
IBKR GLD backward-reuse TEST.

Goal
----
Test the hypothesis that contracts observed on a later date t+1 can be reused
to recover the historical surface at an earlier date t without rebuilding and
querying the whole expiry x strike universe.

The test does this:

1. Load the richest REAL surface available for --later-date.
2. Reuse its (expiry, strike) call contracts for --date.
3. Recompute DTE using the earlier date and keep only the requested domain.
4. Qualify those reused contracts once.
5. Request historical MIDPOINT data only for the earlier date.
6. Measure the reuse hit rate and DTE x moneyness bin coverage.
7. Refresh the short end explicitly (default DTE 75-120).
8. For still-missing bins, qualify/query only candidates belonging to those
   missing bins, not the whole cross-product.

No interpolation or synthetic calibration observations are created.

This is deliberately a TEST script. If the reuse hit rate and bin coverage are
good, the same logic can replace the production historical collector.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import time

import numpy as np
import pandas as pd
from ib_insync import IB, Option, Stock, util

from BnS import BnS
from rates import load_rate_history, rates_for_date


TICKER = "GLD"
EXCHANGE = "SMART"
CURRENCY = "USD"


def connect(host, port, client_id):
    ib = IB()
    ib.connect(
        host,
        int(port),
        clientId=int(client_id),
        readonly=True,
        timeout=15,
    )
    if not ib.isConnected():
        raise RuntimeError("IBKR API connection failed.")
    return ib


def get_spot(path, target):
    df = pd.read_csv(path)
    if "timestamp" not in df.columns or "close" not in df.columns:
        raise ValueError("Stock history must contain timestamp and close.")
    df["date"] = pd.to_datetime(df["timestamp"], errors="coerce").dt.normalize()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    row = df.loc[df["date"].eq(target)].dropna(subset=["close"])
    if row.empty:
        raise ValueError(f"No GLD close for {target.date()}.")
    return float(row.iloc[-1]["close"])


def find_richest_surface(root, date):
    slug = pd.Timestamp(date).strftime("%Y-%m-%d")
    roots = [
        Path(root),
        Path("data/processed/sparse_historical_surfaces"),
    ]

    patterns = [
        f"GLD_{slug}_eligible_all_real_surface.csv",
        f"GLD_{slug}_eligible_bin_balanced_surface.csv",
        f"GLD_{slug}_eligible_adaptive_surface.csv",
        f"GLD_{slug}_eligible_full_surface.csv",
        f"GLD_{slug}_eligible_historical_surface.csv",
    ]

    candidates = []
    for base in roots:
        for name in patterns:
            p = base / name
            if not p.exists():
                continue
            try:
                df = pd.read_csv(p)
            except Exception:
                continue
            if df.empty:
                continue
            if not {"expiry", "K"}.issubset(df.columns):
                continue
            candidates.append((len(df), p, df))

    if not candidates:
        raise FileNotFoundError(
            f"No usable real surface found for {slug} in "
            f"{root} or sparse_historical_surfaces."
        )

    candidates.sort(key=lambda x: x[0], reverse=True)
    n, p, df = candidates[0]
    print(f"[SOURCE] later surface: {p} ({n} rows)")
    return p, df


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
        key=lambda c: (len(c.expirations), len(c.strikes)),
    )


def qualify(ib, contracts, batch=50):
    contracts = list(contracts)
    unique = {}
    for start in range(0, len(contracts), int(batch)):
        part = contracts[start:start + int(batch)]
        try:
            valid = ib.qualifyContracts(*part)
        except Exception as exc:
            print(f"[WARN] qualify batch failed: {exc}")
            valid = []

        for c in valid:
            cid = int(getattr(c, "conId", 0) or 0)
            if cid:
                unique[cid] = c

        ib.sleep(0.05)
        print(
            f"[QUALIFY] {min(start+len(part), len(contracts))}/"
            f"{len(contracts)} candidates | {len(unique)} valid"
        )
    return list(unique.values())


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
        "source": "reused_or_fallback_api",
    }, None


def merge_real(*frames):
    frames = [x for x in frames if x is not None and not x.empty]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True, sort=False)
    if "conId" in df.columns:
        cid = pd.to_numeric(df["conId"], errors="coerce")
        has = cid.notna()
        a = df.loc[has].copy()
        if not a.empty:
            a["conId"] = cid.loc[has].astype("Int64")
            a = a.drop_duplicates("conId", keep="last")
        b = df.loc[~has].copy()
        if not b.empty:
            b = b.drop_duplicates(["expiry", "K"], keep="last")
        return pd.concat([a, b], ignore_index=True, sort=False)
    return df.drop_duplicates(["expiry", "K"], keep="last")


def enrich_filter(raw, target, spot, rate_history, args):
    if raw.empty:
        return raw.copy(), raw.copy()

    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["expiry"] = pd.to_datetime(df["expiry"], errors="coerce").dt.normalize()
    df["K"] = pd.to_numeric(df["K"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["expiry", "K", "price"]).copy()

    df["dte"] = (df["expiry"] - target).dt.days
    df["T"] = df["dte"] / 365.25
    df["moneyness"] = df["K"] / float(spot)

    df = df.loc[
        df["dte"].between(args.min_dte, args.max_dte)
        & df["moneyness"].between(
            args.min_moneyness, args.max_moneyness
        )
        & df["price"].gt(args.min_price)
    ].copy()

    if df.empty:
        return df.copy(), df.copy()

    rates, curve_date = rates_for_date(
        df["T"].to_numpy(float),
        target,
        rate_history=rate_history,
    )
    df["rate"] = rates
    df["spot"] = float(spot)
    df["curve_date"] = pd.Timestamp(curve_date)

    ivs, vegas = [], []
    for r in df.itertuples(index=False):
        iv = BnS.implied_vol_call(
            r.price, float(spot), r.K, r.T, r.rate
        )
        ivs.append(iv)
        if np.isfinite(iv):
            vegas.append(
                BnS.calculate_bs_vega(
                    float(spot), r.K, r.T, r.rate, 0.0, iv
                )
            )
        else:
            vegas.append(np.nan)

    df["implied_vol"] = ivs
    df["vega"] = vegas

    eligible = df.loc[
        np.isfinite(df["implied_vol"])
        & df["implied_vol"].between(args.min_iv, args.max_iv)
        & np.isfinite(df["vega"])
        & df["vega"].ge(args.min_vega)
    ].copy()

    eligible = (
        eligible
        .sort_values(["T", "K"])
        .drop_duplicates(["T", "K"], keep="last")
        .reset_index(drop=True)
    )
    return df, eligible


def make_edges(a, b, n):
    return np.linspace(float(a), float(b), int(n)+1)


def scalar_bin(x, edges):
    i = int(np.searchsorted(edges, float(x), side="right") - 1)
    if i == len(edges)-1:
        i -= 1
    if i < 0 or i >= len(edges)-1:
        return None
    return i


def key_for(dte, m, dte_edges, m_edges):
    i = scalar_bin(dte, dte_edges)
    j = scalar_bin(m, m_edges)
    if i is None or j is None:
        return None
    return i, j


def bin_counts(df, dte_edges, m_edges):
    c = Counter()
    if df.empty:
        return c
    for r in df.itertuples(index=False):
        k = key_for(r.dte, r.moneyness, dte_edges, m_edges)
        if k is not None:
            c[k] += 1
    return c


def reuse_contracts_from_surface(df, target, spot, args, trading_class):
    x = df.copy()
    x["expiry"] = pd.to_datetime(x["expiry"], errors="coerce").dt.normalize()
    x["K"] = pd.to_numeric(x["K"], errors="coerce")
    x = x.dropna(subset=["expiry", "K"]).copy()
    x["dte_at_earlier"] = (x["expiry"] - target).dt.days
    x["m_at_earlier"] = x["K"] / float(spot)
    x = x.loc[
        x["dte_at_earlier"].between(args.min_dte, args.max_dte)
        & x["m_at_earlier"].between(
            args.min_moneyness, args.max_moneyness
        )
    ].copy()

    pairs = (
        x[["expiry", "K"]]
        .drop_duplicates()
        .sort_values(["expiry", "K"])
    )

    contracts = [
        Option(
            TICKER,
            r.expiry.strftime("%Y%m%d"),
            float(r.K),
            "C",
            EXCHANGE,
            currency=CURRENCY,
            tradingClass=trading_class,
        )
        for r in pairs.itertuples(index=False)
    ]
    return contracts, pairs


def current_chain_expiries(chain, target, today, args):
    out = []
    for s in sorted(chain.expirations):
        expiry = pd.to_datetime(str(s), format="%Y%m%d", errors="coerce")
        if pd.isna(expiry):
            continue
        dte = int((expiry-target).days)
        if not args.min_dte <= dte <= args.max_dte:
            continue
        if expiry < today:
            continue
        out.append(expiry)
    return sorted(set(out))


def fallback_candidates(
    chain,
    target,
    today,
    spot,
    args,
    dte_edges,
    m_edges,
    missing_bins,
    exclude_pairs,
):
    expiries = current_chain_expiries(chain, target, today, args)

    low = spot * args.min_moneyness
    high = spot * args.max_moneyness

    actual_strikes = {
        float(k) for k in chain.strikes
        if np.isfinite(k) and low <= float(k) <= high
    }

    # Add 0.50-dollar probes. Invalid strikes will disappear during qualify.
    step = args.half_strike_step
    start = np.ceil(low / step) * step
    end = np.floor(high / step) * step
    ladder = set()
    if end >= start:
        n = int(round((end-start)/step))
        ladder = {
            round(start+i*step, 8)
            for i in range(n+1)
        }

    strikes = sorted(actual_strikes | ladder)

    contracts = []
    seen = set(exclude_pairs)

    for expiry in expiries:
        dte = int((expiry-target).days)

        # Explicitly refresh the short end even if later-date reuse was good.
        short_end = dte <= args.short_refresh_max_dte

        for K in strikes:
            m = K / float(spot)
            key = key_for(dte, m, dte_edges, m_edges)
            if key is None:
                continue
            if not short_end and key not in missing_bins:
                continue

            pair = (expiry.normalize(), round(float(K), 8))
            if pair in seen:
                continue
            seen.add(pair)

            contracts.append(
                Option(
                    TICKER,
                    expiry.strftime("%Y%m%d"),
                    float(K),
                    "C",
                    EXCHANGE,
                    currency=CURRENCY,
                    tradingClass=chain.tradingClass,
                )
            )

    return contracts


def sort_fallback_by_missing_need(
    qualified,
    target,
    spot,
    args,
    dte_edges,
    m_edges,
    current_counts,
):
    rows = []
    for c in qualified:
        expiry = pd.to_datetime(
            c.lastTradeDateOrContractMonth[:8],
            format="%Y%m%d",
            errors="coerce",
        )
        if pd.isna(expiry):
            continue
        dte = int((expiry-target).days)
        K = float(c.strike)
        m = K/float(spot)
        key = key_for(dte, m, dte_edges, m_edges)
        if key is None:
            continue

        dcenter = 0.5*(dte_edges[key[0]]+dte_edges[key[0]+1])
        mcenter = 0.5*(m_edges[key[1]]+m_edges[key[1]+1])
        dist = (
            abs(dte-dcenter)/(dte_edges[key[0]+1]-dte_edges[key[0]])
            + abs(m-mcenter)/(m_edges[key[1]+1]-m_edges[key[1]])
        )
        rows.append({
            "contract": c,
            "expiry": expiry,
            "K": K,
            "dte": dte,
            "moneyness": m,
            "key": key,
            "need": current_counts.get(key, 0),
            "distance": dist,
        })

    rows.sort(
        key=lambda r: (
            r["need"],
            r["dte"] > args.short_refresh_max_dte,
            r["distance"],
            r["dte"],
            r["K"],
        )
    )
    return rows


def coverage_csv(eligible, dte_edges, m_edges, args, path):
    counts = bin_counts(eligible, dte_edges, m_edges)
    rows = []
    for i in range(args.dte_bins):
        for j in range(args.moneyness_bins):
            n = int(counts.get((i,j), 0))
            rows.append({
                "dte_bin": i,
                "moneyness_bin": j,
                "dte_low": dte_edges[i],
                "dte_high": dte_edges[i+1],
                "moneyness_low": m_edges[j],
                "moneyness_high": m_edges[j+1],
                "eligible": n,
                "minimum": args.min_per_bin,
                "covered": n >= args.min_per_bin,
            })
    pd.DataFrame(rows).to_csv(path, index=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--date", required=True, help="Earlier target date t.")
    p.add_argument(
        "--later-date",
        required=True,
        help="Later date t+1 whose real contracts are reused.",
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7497)
    p.add_argument("--client-id", type=int, default=96)

    p.add_argument(
        "--stock",
        default="data/processed/gld_daily_history.csv",
    )
    p.add_argument(
        "--rates",
        default="data/processed/usd_treasury_history.csv",
    )
    p.add_argument(
        "--surface-dir",
        default="data/processed/full_surfaces",
    )
    p.add_argument(
        "--out-dir",
        default="data/processed/backward_reuse_test",
    )

    p.add_argument("--min-dte", type=int, default=75)
    p.add_argument("--max-dte", type=int, default=730)
    p.add_argument("--min-moneyness", type=float, default=0.60)
    p.add_argument("--max-moneyness", type=float, default=1.40)

    p.add_argument("--dte-bins", type=int, default=6)
    p.add_argument("--moneyness-bins", type=int, default=8)
    p.add_argument("--min-per-bin", type=int, default=2)
    p.add_argument(
        "--short-refresh-max-dte",
        type=int,
        default=120,
        help="Always inspect current-chain candidates in this short-end band.",
    )
    p.add_argument("--half-strike-step", type=float, default=0.50)

    p.add_argument("--min-price", type=float, default=0.05)
    p.add_argument("--min-iv", type=float, default=0.03)
    p.add_argument("--max-iv", type=float, default=1.50)
    p.add_argument("--min-vega", type=float, default=0.10)
    p.add_argument("--pacing-seconds", type=float, default=0.15)

    args = p.parse_args()

    target = pd.Timestamp(args.date).normalize()
    later = pd.Timestamp(args.later_date).normalize()
    today = pd.Timestamp.now().normalize()

    if not target < later:
        raise ValueError("--date must be earlier than --later-date.")
    if target >= today:
        raise ValueError("This test is for historical dates.")

    spot = get_spot(args.stock, target)
    rate_history = load_rate_history(args.rates)

    source_path, later_surface = find_richest_surface(
        args.surface_dir, later
    )

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    slug = target.strftime("%Y-%m-%d")
    later_slug = later.strftime("%Y-%m-%d")

    raw_path = out / f"GLD_{slug}_from_{later_slug}_raw.csv"
    eligible_path = out / f"GLD_{slug}_from_{later_slug}_eligible.csv"
    coverage_path = out / f"GLD_{slug}_from_{later_slug}_coverage.csv"
    manifest_path = out / f"GLD_{slug}_from_{later_slug}_summary.txt"

    dte_edges = make_edges(args.min_dte, args.max_dte, args.dte_bins)
    m_edges = make_edges(
        args.min_moneyness, args.max_moneyness, args.moneyness_bins
    )

    ib = connect(args.host, args.port, args.client_id)

    raw = pd.DataFrame()
    failures = Counter()

    try:
        underlying = ib.qualifyContracts(
            Stock(TICKER, EXCHANGE, CURRENCY)
        )[0]
        chain = get_chain(ib, underlying)

        # -------------------------------------------------------------
        # PHASE A: reuse t+1 contracts directly at t.
        # -------------------------------------------------------------
        reuse_candidates, reuse_pairs = reuse_contracts_from_surface(
            later_surface,
            target,
            spot,
            args,
            chain.tradingClass,
        )

        print("="*90)
        print(f"EARLIER DATE t             : {target.date()}")
        print(f"LATER DATE t+1             : {later.date()}")
        print(f"LATER SOURCE               : {source_path}")
        print(f"EARLIER HISTORICAL SPOT    : {spot:.4f}")
        print(f"REUSE PAIRS BEFORE QUALIFY : {len(reuse_candidates)}")
        print("="*90)

        reuse_qualified = qualify(ib, reuse_candidates)
        print(f"[REUSE] qualified: {len(reuse_qualified)}")

        reuse_success = 0
        for idx, contract in enumerate(reuse_qualified, 1):
            row, err = historical_midpoint(
                ib, contract, target, args.pacing_seconds
            )
            if row is not None:
                row["source"] = "reuse_later_date"
                raw = merge_real(raw, pd.DataFrame([row]))
                reuse_success += 1
            else:
                failures[str(err or "unknown")] += 1

            if idx % 10 == 0 or idx == len(reuse_qualified):
                print(
                    f"[REUSE] {idx}/{len(reuse_qualified)} queried | "
                    f"historical hits={reuse_success}"
                )

        enriched, eligible = enrich_filter(
            raw, target, spot, rate_history, args
        )
        counts = bin_counts(eligible, dte_edges, m_edges)

        occupied = [
            (i,j)
            for i in range(args.dte_bins)
            for j in range(args.moneyness_bins)
        ]
        missing = {
            k for k in occupied
            if counts.get(k,0) < args.min_per_bin
        }

        print()
        print(
            f"[REUSE RESULT] historical hits       : "
            f"{reuse_success}/{len(reuse_qualified)}"
        )
        hit_rate = (
            reuse_success / len(reuse_qualified)
            if reuse_qualified else 0.0
        )
        print(f"[REUSE RESULT] hit rate              : {hit_rate:.1%}")
        print(f"[REUSE RESULT] eligible after filters: {len(eligible)}")
        print(
            f"[REUSE RESULT] bins meeting minimum  : "
            f"{len(occupied)-len(missing)}/{len(occupied)}"
        )

        # -------------------------------------------------------------
        # PHASE B: short-end refresh + only missing bins.
        # -------------------------------------------------------------
        exclude_pairs = {
            (
                pd.Timestamp(r.expiry).normalize(),
                round(float(r.K),8),
            )
            for r in reuse_pairs.itertuples(index=False)
        }

        fallback = fallback_candidates(
            chain,
            target,
            today,
            spot,
            args,
            dte_edges,
            m_edges,
            missing,
            exclude_pairs,
        )

        print()
        print(
            f"[FALLBACK] candidates only for missing bins + "
            f"short DTE <= {args.short_refresh_max_dte}: {len(fallback)}"
        )

        fallback_qualified = qualify(ib, fallback)
        ordered = sort_fallback_by_missing_need(
            fallback_qualified,
            target,
            spot,
            args,
            dte_edges,
            m_edges,
            counts,
        )

        fallback_calls = 0
        fallback_hits = 0

        for item in ordered:
            # Recompute coverage after each successful point.
            counts = bin_counts(eligible, dte_edges, m_edges)
            key = item["key"]

            short_end = item["dte"] <= args.short_refresh_max_dte
            needs_bin = counts.get(key,0) < args.min_per_bin

            if not short_end and not needs_bin:
                continue

            # For short end we still stop querying a bin after it reaches min;
            # the point of the refresh is to discover new short maturities,
            # not to exhaust them unnecessarily.
            if short_end and not needs_bin:
                continue

            row, err = historical_midpoint(
                ib,
                item["contract"],
                target,
                args.pacing_seconds,
            )
            fallback_calls += 1

            if row is not None:
                row["source"] = "short_or_missing_bin_fallback"
                raw = merge_real(raw, pd.DataFrame([row]))
                fallback_hits += 1
                enriched, eligible = enrich_filter(
                    raw, target, spot, rate_history, args
                )
            else:
                failures[str(err or "unknown")] += 1

            if fallback_calls % 10 == 0:
                counts_now = bin_counts(
                    eligible, dte_edges, m_edges
                )
                covered_now = sum(
                    counts_now.get(k,0) >= args.min_per_bin
                    for k in occupied
                )
                print(
                    f"[FALLBACK] calls={fallback_calls} | "
                    f"hits={fallback_hits} | "
                    f"eligible={len(eligible)} | "
                    f"covered bins={covered_now}/{len(occupied)}"
                )

            # Stop once every bin is covered.
            counts_now = bin_counts(eligible, dte_edges, m_edges)
            if all(
                counts_now.get(k,0) >= args.min_per_bin
                for k in occupied
            ):
                break

        enriched, eligible = enrich_filter(
            raw, target, spot, rate_history, args
        )

        enriched.to_csv(raw_path, index=False)
        eligible.to_csv(eligible_path, index=False)
        coverage_csv(
            eligible, dte_edges, m_edges, args, coverage_path
        )

        counts = bin_counts(eligible, dte_edges, m_edges)
        covered = sum(
            counts.get(k,0) >= args.min_per_bin
            for k in occupied
        )

        unique_expiries = (
            eligible["expiry"].nunique() if not eligible.empty else 0
        )

        summary = [
            f"earlier_date={target.date()}",
            f"later_date={later.date()}",
            f"later_source={source_path}",
            f"reuse_candidates={len(reuse_candidates)}",
            f"reuse_qualified={len(reuse_qualified)}",
            f"reuse_historical_hits={reuse_success}",
            f"reuse_hit_rate={hit_rate:.6f}",
            f"fallback_qualified={len(fallback_qualified)}",
            f"fallback_calls={fallback_calls}",
            f"fallback_hits={fallback_hits}",
            f"final_eligible={len(eligible)}",
            f"unique_expiries={unique_expiries}",
            f"covered_bins={covered}",
            f"total_bins={len(occupied)}",
        ]
        manifest_path.write_text("\n".join(summary), encoding="utf-8")

        print()
        print("="*90)
        print("[FINAL TEST RESULT]")
        print(
            f"reuse hit rate       : "
            f"{reuse_success}/{len(reuse_qualified)} = {hit_rate:.1%}"
        )
        print(
            f"fallback API calls   : {fallback_calls} "
            f"(hits {fallback_hits})"
        )
        print(f"final eligible       : {len(eligible)}")
        print(f"unique expiries      : {unique_expiries}")
        print(f"covered bins         : {covered}/{len(occupied)}")
        print(f"eligible output      : {eligible_path}")
        print(f"coverage output      : {coverage_path}")
        print(f"summary              : {manifest_path}")
        if failures:
            print("[TOP FAILURES]")
            for reason, n in failures.most_common(8):
                print(f"  {n:5d}  {reason}")
        print("="*90)

    finally:
        if ib.isConnected():
            ib.disconnect()


if __name__ == "__main__":
    main()
