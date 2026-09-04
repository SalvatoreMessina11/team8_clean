"""
IBKR GLD historical collector using DIRECT conId backward reuse.

Purpose
-------
For an earlier historical date t, reuse the exact option identities (conId)
already observed on a later date t+1 (or any later source date). This avoids
rebuilding and querying the full expiry x strike cross-product on every date.

Core idea
---------
1. Load the richest REAL saved surface for --source-date.
2. Extract its exact conId, expiry and strike.
3. Recompute DTE and moneyness for the earlier --date.
4. Query historical MIDPOINT directly by conId.
5. Query in DTE x moneyness bins, so API calls are spent where coverage is
   missing rather than on arbitrary contracts.
6. If a bin cannot reach its minimum, exhaust all carried conId candidates in
   that bin and retain every real observation recovered.
7. If a bin becomes dense, retain at most --max-per-bin in the balanced output;
   all real observations are also preserved in an audit surface.
8. Explicitly refresh short maturities from the current IBKR chain because,
   when moving backward, expiries that were below DTE=75 on the later date can
   enter the eligible domain on the earlier date.
9. Optionally fill other still-missing bins from the current chain, but ONLY
   those missing bins.

No interpolation and no synthetic calibration observations are created.

This is a test/research collector. If the direct-conId hit rate is high, it can
replace the existing historical collector.

Outputs
-------
data/processed/conid_backward_test/
    GLD_TARGET_from_SOURCE_midpoint_raw.csv
    GLD_TARGET_from_SOURCE_eligible_all_real_surface.csv
    GLD_TARGET_from_SOURCE_eligible_bin_balanced_surface.csv
    GLD_TARGET_from_SOURCE_bin_coverage.csv
    GLD_TARGET_from_SOURCE_attempts.csv
    GLD_TARGET_from_SOURCE_summary.txt
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import time

import numpy as np
import pandas as pd
from ib_insync import IB, Contract, Option, Stock, util

from BnS import BnS
from rates import load_rate_history, rates_for_date


TICKER = "GLD"
EXCHANGE = "SMART"
CURRENCY = "USD"


# ---------------------------------------------------------------------------
# Basic data / IBKR helpers
# ---------------------------------------------------------------------------

def connect(host, port, client_id):
    ib = IB()
    ib.connect(
        host,
        int(port),
        clientId=int(client_id),
        readonly=True,
        timeout=20,
    )
    if not ib.isConnected():
        raise RuntimeError("IBKR API connection failed.")
    return ib


def get_spot(path, target):
    df = pd.read_csv(path)
    if "timestamp" not in df.columns or "close" not in df.columns:
        raise ValueError(
            "GLD history must contain columns 'timestamp' and 'close'."
        )

    df["date"] = pd.to_datetime(
        df["timestamp"], errors="coerce"
    ).dt.normalize()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")

    row = df.loc[df["date"].eq(target)].dropna(subset=["close"])
    if row.empty:
        raise ValueError(f"No GLD close for {target.date()}.")

    return float(row.iloc[-1]["close"])


def get_chain(ib, underlying):
    chains = ib.reqSecDefOptParams(
        underlying.symbol,
        "",
        underlying.secType,
        underlying.conId,
    )
    if not chains:
        raise RuntimeError("No GLD option chain returned by IBKR.")

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


def historical_midpoint(ib, contract, target, pacing):
    """
    Request the last target-date RTH MIDPOINT observation.

    contract may be a direct conId Contract or a fully qualified Option.
    """
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
        return None, f"request exception: {exc}"

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

    return {
        "timestamp": row["timestamp"],
        "price": price,
    }, None


def direct_conid_contract(conid):
    """
    Construct the contract from the exact saved IBKR identity.

    This deliberately does NOT rediscover the option from strike+expiry.
    """
    return Contract(
        conId=int(conid),
        symbol=TICKER,
        secType="OPT",
        exchange=EXCHANGE,
        currency=CURRENCY,
        includeExpired=True,
    )


def historical_by_conid(
    ib,
    conid,
    target,
    pacing,
    resolve_on_failure=True,
):
    """
    First try reqHistoricalData directly with the saved conId.
    If that fails and resolve_on_failure=True, ask IBKR for contract details
    using the SAME conId, then retry once with the returned exact contract.
    """
    base = direct_conid_contract(conid)

    result, error = historical_midpoint(
        ib, base, target, pacing
    )
    if result is not None:
        return result, "direct_conid", None

    if not resolve_on_failure:
        return None, "direct_conid", error

    try:
        details = ib.reqContractDetails(base)
    except Exception as exc:
        return None, "contract_details_failed", (
            f"{error}; contractDetails exception: {exc}"
        )

    if not details:
        return None, "contract_details_empty", (
            f"{error}; no contract details for conId"
        )

    resolved = details[0].contract
    try:
        resolved.includeExpired = True
    except Exception:
        pass

    retry, retry_error = historical_midpoint(
        ib, resolved, target, pacing
    )

    if retry is not None:
        return retry, "resolved_same_conid", None

    return None, "resolved_same_conid", (
        f"direct={error}; resolved={retry_error}"
    )


# ---------------------------------------------------------------------------
# Saved surface discovery
# ---------------------------------------------------------------------------

def load_csv_safe(path):
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df is None or df.empty:
        return None
    return df


def find_richest_source_surface(source_date, full_dir, sparse_dir):
    """
    Inspect all known REAL surface variants and select the one with the most
    usable rows containing conId + expiry + K.

    This avoids the old priority bug where an empty adaptive file could hide a
    richer historical file.
    """
    slug = source_date.strftime("%Y-%m-%d")

    paths = [
        Path(full_dir) / f"GLD_{slug}_eligible_all_real_surface.csv",
        Path(full_dir) / f"GLD_{slug}_eligible_full_surface.csv",
        Path(full_dir) / f"GLD_{slug}_eligible_bin_balanced_surface.csv",
        Path(full_dir) / f"GLD_{slug}_eligible_adaptive_surface.csv",
        Path(sparse_dir) / f"GLD_{slug}_eligible_historical_surface.csv",
    ]

    candidates = []

    for path in paths:
        if not path.exists():
            continue

        df = load_csv_safe(path)
        if df is None:
            continue

        if not {"conId", "expiry", "K"}.issubset(df.columns):
            continue

        work = df.copy()
        work["conId"] = pd.to_numeric(
            work["conId"], errors="coerce"
        )
        work["expiry"] = pd.to_datetime(
            work["expiry"], errors="coerce"
        ).dt.normalize()
        work["K"] = pd.to_numeric(
            work["K"], errors="coerce"
        )

        work = work.dropna(
            subset=["conId", "expiry", "K"]
        ).copy()

        if work.empty:
            continue

        work["conId"] = work["conId"].astype("int64")
        work = work.drop_duplicates("conId", keep="last")

        candidates.append(
            (len(work), path, work)
        )

    if not candidates:
        raise FileNotFoundError(
            f"No usable REAL surface with conId found for "
            f"{source_date.date()}."
        )

    candidates.sort(key=lambda x: x[0], reverse=True)
    n, path, df = candidates[0]

    print(
        f"[SOURCE] {path} | "
        f"{n} unique saved conIds"
    )
    return path, df


# ---------------------------------------------------------------------------
# Enrichment / filters
# ---------------------------------------------------------------------------

def merge_real_rows(*frames):
    valid = [
        x.copy()
        for x in frames
        if x is not None and not x.empty
    ]
    if not valid:
        return pd.DataFrame()

    df = pd.concat(valid, ignore_index=True, sort=False)

    if "conId" in df.columns:
        df["conId"] = pd.to_numeric(
            df["conId"], errors="coerce"
        )
        with_id = df["conId"].notna()

        a = df.loc[with_id].copy()
        if not a.empty:
            a["conId"] = a["conId"].astype("int64")
            a = a.drop_duplicates("conId", keep="last")

        b = df.loc[~with_id].copy()
        if not b.empty and {"expiry", "K"}.issubset(b.columns):
            b = b.drop_duplicates(
                ["expiry", "K"], keep="last"
            )

        return pd.concat(
            [a, b], ignore_index=True, sort=False
        )

    if {"expiry", "K"}.issubset(df.columns):
        return df.drop_duplicates(
            ["expiry", "K"], keep="last"
        )

    return df


def enrich_and_filter(
    raw,
    target,
    spot,
    rate_history,
    args,
):
    if raw.empty:
        return raw.copy(), raw.copy()

    df = raw.copy()

    df["date"] = pd.to_datetime(
        df["date"], errors="coerce"
    ).dt.normalize()
    df["expiry"] = pd.to_datetime(
        df["expiry"], errors="coerce"
    ).dt.normalize()
    df["K"] = pd.to_numeric(
        df["K"], errors="coerce"
    )
    df["price"] = pd.to_numeric(
        df["price"], errors="coerce"
    )

    df = df.dropna(
        subset=["date", "expiry", "K", "price"]
    ).copy()

    df["dte"] = (df["expiry"] - target).dt.days
    df["T"] = df["dte"] / 365.25
    df["moneyness"] = df["K"] / float(spot)

    df = df.loc[
        df["dte"].between(
            args.min_dte, args.max_dte
        )
        & df["moneyness"].between(
            args.min_moneyness,
            args.max_moneyness,
        )
        & df["price"].gt(
            args.min_price
        )
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

    ivs = []
    vegas = []

    for row in df.itertuples(index=False):
        iv = BnS.implied_vol_call(
            row.price,
            float(spot),
            row.K,
            row.T,
            row.rate,
        )
        ivs.append(iv)

        if np.isfinite(iv):
            vega = BnS.calculate_bs_vega(
                float(spot),
                row.K,
                row.T,
                row.rate,
                0.0,
                iv,
            )
        else:
            vega = np.nan

        vegas.append(vega)

    df["implied_vol"] = ivs
    df["vega"] = vegas

    eligible = df.loc[
        np.isfinite(df["implied_vol"])
        & df["implied_vol"].between(
            args.min_iv,
            args.max_iv,
        )
        & np.isfinite(df["vega"])
        & df["vega"].ge(
            args.min_vega
        )
    ].copy()

    eligible = (
        eligible
        .sort_values(["T", "K"])
        .drop_duplicates(["T", "K"], keep="last")
        .reset_index(drop=True)
    )

    return df, eligible


# ---------------------------------------------------------------------------
# Bin geometry
# ---------------------------------------------------------------------------

def make_edges(a, b, n):
    return np.linspace(
        float(a),
        float(b),
        int(n) + 1,
    )


def scalar_bin(value, edges):
    idx = int(
        np.searchsorted(
            edges,
            float(value),
            side="right",
        ) - 1
    )

    if idx == len(edges) - 1:
        idx -= 1

    if idx < 0 or idx >= len(edges) - 1:
        return None

    return idx


def bin_key(
    dte,
    moneyness,
    dte_edges,
    m_edges,
):
    i = scalar_bin(dte, dte_edges)
    j = scalar_bin(moneyness, m_edges)

    if i is None or j is None:
        return None

    return int(i), int(j)


def add_candidate_geometry(
    source,
    target,
    spot,
    args,
    dte_edges,
    m_edges,
):
    work = source.copy()

    work["expiry"] = pd.to_datetime(
        work["expiry"], errors="coerce"
    ).dt.normalize()
    work["K"] = pd.to_numeric(
        work["K"], errors="coerce"
    )
    work["conId"] = pd.to_numeric(
        work["conId"], errors="coerce"
    )

    work = work.dropna(
        subset=["expiry", "K", "conId"]
    ).copy()

    work["conId"] = work["conId"].astype("int64")
    work["dte_target"] = (
        work["expiry"] - target
    ).dt.days
    work["moneyness_target"] = (
        work["K"] / float(spot)
    )

    work = work.loc[
        work["dte_target"].between(
            args.min_dte,
            args.max_dte,
        )
        & work["moneyness_target"].between(
            args.min_moneyness,
            args.max_moneyness,
        )
    ].copy()

    if work.empty:
        return work

    keys = [
        bin_key(
            row.dte_target,
            row.moneyness_target,
            dte_edges,
            m_edges,
        )
        for row in work.itertuples(index=False)
    ]

    work["dte_bin"] = [
        key[0] if key is not None else pd.NA
        for key in keys
    ]
    work["m_bin"] = [
        key[1] if key is not None else pd.NA
        for key in keys
    ]

    work = work.dropna(
        subset=["dte_bin", "m_bin"]
    ).copy()

    work["dte_bin"] = work["dte_bin"].astype(int)
    work["m_bin"] = work["m_bin"].astype(int)

    dte_centers = 0.5 * (
        dte_edges[:-1] + dte_edges[1:]
    )
    m_centers = 0.5 * (
        m_edges[:-1] + m_edges[1:]
    )
    dte_widths = np.diff(dte_edges)
    m_widths = np.diff(m_edges)

    work["bin_distance"] = [
        np.sqrt(
            (
                (
                    row.dte_target
                    - dte_centers[row.dte_bin]
                )
                / dte_widths[row.dte_bin]
            ) ** 2
            +
            (
                (
                    row.moneyness_target
                    - m_centers[row.m_bin]
                )
                / m_widths[row.m_bin]
            ) ** 2
        )
        for row in work.itertuples(index=False)
    ]

    return (
        work
        .drop_duplicates("conId", keep="last")
        .sort_values(
            [
                "dte_bin",
                "m_bin",
                "bin_distance",
                "dte_target",
                "K",
            ]
        )
        .reset_index(drop=True)
    )


def eligible_bin_counts(
    eligible,
    dte_edges,
    m_edges,
):
    counts = Counter()

    if eligible.empty:
        return counts

    for row in eligible.itertuples(index=False):
        key = bin_key(
            row.dte,
            row.moneyness,
            dte_edges,
            m_edges,
        )
        if key is not None:
            counts[key] += 1

    return counts


def candidate_queues(candidates, attempted_ids):
    queues = defaultdict(list)

    for row in candidates.itertuples(index=False):
        cid = int(row.conId)

        if cid in attempted_ids:
            continue

        key = (
            int(row.dte_bin),
            int(row.m_bin),
        )
        queues[key].append(row)

    return queues


# ---------------------------------------------------------------------------
# Balanced output
# ---------------------------------------------------------------------------

def select_spread(group, n):
    group = group.copy().reset_index(drop=True)

    if len(group) <= int(n):
        return group

    x = pd.to_numeric(
        group["T"], errors="coerce"
    ).to_numpy(float)
    y = pd.to_numeric(
        group["moneyness"], errors="coerce"
    ).to_numpy(float)

    def normalize(v):
        lo = np.nanmin(v)
        hi = np.nanmax(v)

        if (
            not np.isfinite(lo)
            or not np.isfinite(hi)
            or hi <= lo
        ):
            return np.zeros_like(v)

        return (v - lo) / (hi - lo)

    points = np.column_stack(
        [normalize(x), normalize(y)]
    )

    center = np.array([0.5, 0.5])

    first = int(
        np.argmin(
            np.sum(
                (points - center) ** 2,
                axis=1,
            )
        )
    )

    chosen = [first]

    while len(chosen) < int(n):
        remaining = [
            i
            for i in range(len(group))
            if i not in chosen
        ]

        if not remaining:
            break

        min_distances = []

        for i in remaining:
            ds = [
                np.linalg.norm(
                    points[i] - points[j]
                )
                for j in chosen
            ]
            min_distances.append(min(ds))

        chosen.append(
            remaining[
                int(np.argmax(min_distances))
            ]
        )

    return group.iloc[
        sorted(chosen)
    ].copy()


def balanced_surface(
    eligible,
    dte_edges,
    m_edges,
    max_per_bin,
):
    if eligible.empty:
        return eligible.copy()

    work = eligible.copy()

    keys = [
        bin_key(
            row.dte,
            row.moneyness,
            dte_edges,
            m_edges,
        )
        for row in work.itertuples(index=False)
    ]

    work["dte_bin"] = [
        key[0] if key is not None else pd.NA
        for key in keys
    ]
    work["m_bin"] = [
        key[1] if key is not None else pd.NA
        for key in keys
    ]

    work = work.dropna(
        subset=["dte_bin", "m_bin"]
    ).copy()

    work["dte_bin"] = work["dte_bin"].astype(int)
    work["m_bin"] = work["m_bin"].astype(int)

    pieces = []

    for _, group in work.groupby(
        ["dte_bin", "m_bin"],
        sort=True,
    ):
        if len(group) <= int(max_per_bin):
            pieces.append(group.copy())
        else:
            pieces.append(
                select_spread(
                    group,
                    int(max_per_bin),
                )
            )

    if not pieces:
        return eligible.iloc[0:0].copy()

    return (
        pd.concat(
            pieces,
            ignore_index=True,
        )
        .drop(
            columns=["dte_bin", "m_bin"],
            errors="ignore",
        )
        .sort_values(["T", "K"])
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Checkpoint / attempts
# ---------------------------------------------------------------------------

def empty_attempts():
    return pd.DataFrame(
        columns=[
            "conId",
            "source",
            "status",
            "method",
            "error",
            "expiry",
            "K",
        ]
    )


def load_attempts(path):
    if not path.exists():
        return empty_attempts()

    try:
        df = pd.read_csv(path)
    except Exception:
        return empty_attempts()

    if "conId" in df.columns:
        df["conId"] = pd.to_numeric(
            df["conId"], errors="coerce"
        ).astype("Int64")

    return df


def save_attempt(
    attempts,
    path,
    *,
    conid,
    source,
    status,
    method,
    error,
    expiry,
    strike,
):
    row = {
        "conId": int(conid),
        "source": source,
        "status": status,
        "method": method,
        "error": error or "",
        "expiry": expiry,
        "K": strike,
    }

    attempts = pd.concat(
        [
            attempts,
            pd.DataFrame([row]),
        ],
        ignore_index=True,
        sort=False,
    )

    attempts["conId"] = pd.to_numeric(
        attempts["conId"], errors="coerce"
    ).astype("Int64")

    attempts = attempts.drop_duplicates(
        "conId",
        keep="last",
    )

    attempts.to_csv(
        path,
        index=False,
    )

    return attempts


# ---------------------------------------------------------------------------
# Optional fallback from current chain
# ---------------------------------------------------------------------------

def qualify_symbolic(
    ib,
    contracts,
    batch=50,
):
    contracts = list(contracts)
    unique = {}

    for start in range(
        0,
        len(contracts),
        int(batch),
    ):
        part = contracts[
            start:start + int(batch)
        ]

        try:
            valid = ib.qualifyContracts(*part)
        except Exception as exc:
            print(
                f"[WARN] symbolic qualification "
                f"batch failed: {exc}"
            )
            valid = []

        for contract in valid:
            cid = int(
                getattr(
                    contract,
                    "conId",
                    0,
                )
                or 0
            )

            if cid:
                unique[cid] = contract

        ib.sleep(0.05)

        print(
            f"[FALLBACK QUALIFY] "
            f"{min(start + len(part), len(contracts))}/"
            f"{len(contracts)} | "
            f"{len(unique)} valid"
        )

    return list(unique.values())


def current_chain_fallback_candidates(
    ib,
    target,
    today,
    spot,
    args,
    dte_edges,
    m_edges,
    missing_bins,
    known_conids,
):
    underlying = ib.qualifyContracts(
        Stock(
            TICKER,
            EXCHANGE,
            CURRENCY,
        )
    )[0]

    chain = get_chain(
        ib,
        underlying,
    )

    expiries = []

    for raw_expiry in sorted(
        chain.expirations
    ):
        expiry = pd.to_datetime(
            str(raw_expiry),
            format="%Y%m%d",
            errors="coerce",
        )

        if pd.isna(expiry):
            continue

        dte = int(
            (expiry - target).days
        )

        if not (
            args.min_dte
            <= dte
            <= args.max_dte
        ):
            continue

        # Current IBKR chain usually cannot qualify
        # already expired option contracts.
        if expiry < today:
            continue

        expiries.append(expiry)

    expiries = sorted(set(expiries))

    low = (
        float(spot)
        * float(args.min_moneyness)
    )
    high = (
        float(spot)
        * float(args.max_moneyness)
    )

    actual_strikes = {
        float(k)
        for k in chain.strikes
        if np.isfinite(k)
        and low <= float(k) <= high
    }

    step = float(
        args.half_strike_step
    )

    ladder = set()

    if step > 0:
        start = (
            np.ceil(low / step)
            * step
        )
        end = (
            np.floor(high / step)
            * step
        )

        if end >= start:
            n = int(
                round(
                    (end - start)
                    / step
                )
            )

            ladder = {
                round(
                    start + i * step,
                    8,
                )
                for i in range(n + 1)
            }

    strikes = sorted(
        actual_strikes | ladder
    )

    contracts = []

    for expiry in expiries:
        dte = int(
            (expiry - target).days
        )

        short_end = (
            dte
            <= args.short_refresh_max_dte
        )

        for strike in strikes:
            m = (
                float(strike)
                / float(spot)
            )

            key = bin_key(
                dte,
                m,
                dte_edges,
                m_edges,
            )

            if key is None:
                continue

            use = short_end

            if (
                args.fill_missing_bins
                and key in missing_bins
            ):
                use = True

            if not use:
                continue

            contracts.append(
                Option(
                    TICKER,
                    expiry.strftime("%Y%m%d"),
                    float(strike),
                    "C",
                    EXCHANGE,
                    currency=CURRENCY,
                    tradingClass=chain.tradingClass,
                )
            )

    qualified = qualify_symbolic(
        ib,
        contracts,
    )

    rows = []

    for contract in qualified:
        cid = int(contract.conId)

        if cid in known_conids:
            continue

        expiry = pd.to_datetime(
            contract.lastTradeDateOrContractMonth[:8],
            format="%Y%m%d",
            errors="coerce",
        )

        if pd.isna(expiry):
            continue

        dte = int(
            (expiry - target).days
        )
        strike = float(
            contract.strike
        )
        m = strike / float(spot)

        key = bin_key(
            dte,
            m,
            dte_edges,
            m_edges,
        )

        if key is None:
            continue

        dte_center = 0.5 * (
            dte_edges[key[0]]
            + dte_edges[key[0] + 1]
        )
        m_center = 0.5 * (
            m_edges[key[1]]
            + m_edges[key[1] + 1]
        )

        distance = (
            abs(dte - dte_center)
            / (
                dte_edges[key[0] + 1]
                - dte_edges[key[0]]
            )
            +
            abs(m - m_center)
            / (
                m_edges[key[1] + 1]
                - m_edges[key[1]]
            )
        )

        rows.append(
            {
                "conId": cid,
                "contract": contract,
                "expiry": expiry,
                "K": strike,
                "dte_target": dte,
                "moneyness_target": m,
                "dte_bin": key[0],
                "m_bin": key[1],
                "bin_distance": distance,
            }
        )

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .drop_duplicates(
            "conId",
            keep="last",
        )
        .sort_values(
            [
                "dte_bin",
                "m_bin",
                "bin_distance",
                "dte_target",
                "K",
            ]
        )
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Coverage report
# ---------------------------------------------------------------------------

def write_coverage(
    eligible,
    carried_candidates,
    dte_edges,
    m_edges,
    args,
    path,
):
    elig_counts = eligible_bin_counts(
        eligible,
        dte_edges,
        m_edges,
    )

    candidate_counts = Counter()

    if not carried_candidates.empty:
        candidate_counts = Counter(
            zip(
                carried_candidates[
                    "dte_bin"
                ].astype(int),
                carried_candidates[
                    "m_bin"
                ].astype(int),
            )
        )

    rows = []

    for i in range(args.dte_bins):
        for j in range(
            args.moneyness_bins
        ):
            key = (i, j)

            rows.append(
                {
                    "dte_bin": i,
                    "moneyness_bin": j,
                    "dte_low": dte_edges[i],
                    "dte_high": dte_edges[
                        i + 1
                    ],
                    "moneyness_low": m_edges[
                        j
                    ],
                    "moneyness_high": m_edges[
                        j + 1
                    ],
                    "carried_conid_candidates":
                        int(
                            candidate_counts.get(
                                key,
                                0,
                            )
                        ),
                    "eligible_real":
                        int(
                            elig_counts.get(
                                key,
                                0,
                            )
                        ),
                    "minimum_per_bin":
                        args.min_per_bin,
                    "maximum_per_bin":
                        args.max_per_bin,
                    "covered":
                        elig_counts.get(
                            key,
                            0,
                        )
                        >= args.min_per_bin,
                }
            )

    pd.DataFrame(rows).to_csv(
        path,
        index=False,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--source-date",
        required=True,
        help="Later date whose exact saved conIds are reused.",
    )
    parser.add_argument(
        "--date",
        required=True,
        help="Earlier target date to recover.",
    )

    parser.add_argument(
        "--host",
        default="127.0.0.1",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7497,
    )
    parser.add_argument(
        "--client-id",
        type=int,
        default=97,
    )

    parser.add_argument(
        "--stock",
        default=(
            "data/processed/"
            "gld_daily_history.csv"
        ),
    )
    parser.add_argument(
        "--rates",
        default=(
            "data/processed/"
            "usd_treasury_history.csv"
        ),
    )

    parser.add_argument(
        "--full-dir",
        default=(
            "data/processed/"
            "full_surfaces"
        ),
    )
    parser.add_argument(
        "--sparse-dir",
        default=(
            "data/processed/"
            "sparse_historical_surfaces"
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=(
            "data/processed/"
            "conid_backward_test"
        ),
    )

    parser.add_argument(
        "--min-dte",
        type=int,
        default=75,
    )
    parser.add_argument(
        "--max-dte",
        type=int,
        default=730,
    )
    parser.add_argument(
        "--min-moneyness",
        type=float,
        default=0.60,
    )
    parser.add_argument(
        "--max-moneyness",
        type=float,
        default=1.40,
    )

    parser.add_argument(
        "--dte-bins",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--moneyness-bins",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--min-per-bin",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--max-per-bin",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--target-total",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--short-refresh-max-dte",
        type=int,
        default=120,
    )
    parser.add_argument(
        "--half-strike-step",
        type=float,
        default=0.50,
    )
    parser.add_argument(
        "--no-fill-missing-bins",
        dest="fill_missing_bins",
        action="store_false",
        help=(
            "Disable non-short-end fallback. "
            "Short-end refresh remains active."
        ),
    )
    parser.set_defaults(
        fill_missing_bins=True
    )

    parser.add_argument(
        "--min-price",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--min-iv",
        type=float,
        default=0.03,
    )
    parser.add_argument(
        "--max-iv",
        type=float,
        default=1.50,
    )
    parser.add_argument(
        "--min-vega",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--pacing-seconds",
        type=float,
        default=0.15,
    )

    parser.add_argument(
        "--no-resolve-on-failure",
        dest="resolve_on_failure",
        action="store_false",
        help=(
            "Do not retry a failed direct conId request "
            "through reqContractDetails."
        ),
    )
    parser.set_defaults(
        resolve_on_failure=True
    )

    parser.add_argument(
        "--fresh",
        action="store_true",
    )

    args = parser.parse_args()

    if args.min_per_bin < 1:
        raise ValueError(
            "--min-per-bin must be >= 1"
        )

    if (
        args.max_per_bin
        < args.min_per_bin
    ):
        raise ValueError(
            "--max-per-bin must be "
            ">= --min-per-bin"
        )

    source_date = pd.Timestamp(
        args.source_date
    ).normalize()

    target = pd.Timestamp(
        args.date
    ).normalize()

    today = pd.Timestamp.now().normalize()

    if not target < source_date:
        raise ValueError(
            "--date must be earlier "
            "than --source-date."
        )

    if target >= today:
        raise ValueError(
            "Target must be historical."
        )

    spot = get_spot(
        args.stock,
        target,
    )

    rate_history = load_rate_history(
        args.rates
    )

    source_path, source = (
        find_richest_source_surface(
            source_date,
            args.full_dir,
            args.sparse_dir,
        )
    )

    out = Path(args.out_dir)
    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    target_slug = target.strftime(
        "%Y-%m-%d"
    )
    source_slug = source_date.strftime(
        "%Y-%m-%d"
    )

    prefix = (
        f"GLD_{target_slug}"
        f"_from_{source_slug}"
    )

    raw_path = (
        out
        / f"{prefix}_midpoint_raw.csv"
    )
    all_real_path = (
        out
        / f"{prefix}_eligible_all_real_surface.csv"
    )
    balanced_path = (
        out
        / f"{prefix}_eligible_bin_balanced_surface.csv"
    )
    coverage_path = (
        out
        / f"{prefix}_bin_coverage.csv"
    )
    attempts_path = (
        out
        / f"{prefix}_attempts.csv"
    )
    summary_path = (
        out
        / f"{prefix}_summary.txt"
    )

    if args.fresh:
        for path in [
            raw_path,
            all_real_path,
            balanced_path,
            coverage_path,
            attempts_path,
            summary_path,
        ]:
            if path.exists():
                path.unlink()

    dte_edges = make_edges(
        args.min_dte,
        args.max_dte,
        args.dte_bins,
    )
    m_edges = make_edges(
        args.min_moneyness,
        args.max_moneyness,
        args.moneyness_bins,
    )

    carried = add_candidate_geometry(
        source,
        target,
        spot,
        args,
        dte_edges,
        m_edges,
    )

    if carried.empty:
        raise RuntimeError(
            "No source conIds remain inside "
            "the target DTE/moneyness domain."
        )

    if raw_path.exists():
        existing_raw = (
            load_csv_safe(raw_path)
        )
        if existing_raw is None:
            existing_raw = pd.DataFrame()
    else:
        existing_raw = pd.DataFrame()

    raw = existing_raw.copy()
    attempts = load_attempts(
        attempts_path
    )

    attempted_ids = set(
        pd.to_numeric(
            attempts.get(
                "conId",
                pd.Series(dtype=float),
            ),
            errors="coerce",
        )
        .dropna()
        .astype(int)
    )

    ib = connect(
        args.host,
        args.port,
        args.client_id,
    )

    direct_calls = 0
    direct_hits = 0
    resolved_hits = 0
    fallback_calls = 0
    fallback_hits = 0
    failures = Counter()

    try:
        print("=" * 92)
        print(
            f"SOURCE DATE              : "
            f"{source_date.date()}"
        )
        print(
            f"TARGET DATE              : "
            f"{target.date()}"
        )
        print(
            f"SOURCE FILE              : "
            f"{source_path}"
        )
        print(
            f"TARGET SPOT              : "
            f"{spot:.4f}"
        )
        print(
            f"CARRIED conIds IN DOMAIN : "
            f"{len(carried)}"
        )
        print(
            f"BIN RULE                 : "
            f"{args.dte_bins}x"
            f"{args.moneyness_bins}, "
            f"min={args.min_per_bin}, "
            f"max={args.max_per_bin}"
        )
        print(
            f"TARGET TOTAL             : "
            f"{args.target_total}"
        )
        print("=" * 92)

        enriched, eligible = (
            enrich_and_filter(
                raw,
                target,
                spot,
                rate_history,
                args,
            )
        )

        queues = candidate_queues(
            carried,
            attempted_ids,
        )

        occupied_carried_bins = sorted(
            set(
                zip(
                    carried[
                        "dte_bin"
                    ].astype(int),
                    carried[
                        "m_bin"
                    ].astype(int),
                )
            )
        )

        # -------------------------------------------------------------
        # PHASE 1:
        # Use direct saved conIds, round-robin across deficient bins.
        # If a bin cannot reach its minimum, exhaust all carried conIds.
        # -------------------------------------------------------------
        print()
        print(
            "[PHASE 1] Direct saved conId -> "
            "historical MIDPOINT."
        )

        while True:
            counts = eligible_bin_counts(
                eligible,
                dte_edges,
                m_edges,
            )

            deficient = [
                key
                for key in occupied_carried_bins
                if counts.get(key, 0)
                < args.min_per_bin
                and len(
                    queues.get(key, [])
                ) > 0
            ]

            if not deficient:
                break

            progress = False

            for key in deficient:
                if not queues.get(key):
                    continue

                candidate = queues[key].pop(0)

                result, method, error = (
                    historical_by_conid(
                        ib,
                        int(candidate.conId),
                        target,
                        args.pacing_seconds,
                        args.resolve_on_failure,
                    )
                )

                direct_calls += 1
                progress = True

                if result is not None:
                    direct_hits += 1

                    if (
                        method
                        == "resolved_same_conid"
                    ):
                        resolved_hits += 1

                    row = {
                        "date": target,
                        "timestamp":
                            result["timestamp"],
                        "expiry":
                            candidate.expiry,
                        "K":
                            float(candidate.K),
                        "price":
                            result["price"],
                        "conId":
                            int(candidate.conId),
                        "localSymbol":
                            str(
                                getattr(
                                    candidate,
                                    "localSymbol",
                                    "",
                                )
                            ),
                        "source":
                            "carried_conid",
                        "conid_method":
                            method,
                    }

                    raw = merge_real_rows(
                        raw,
                        pd.DataFrame([row]),
                    )
                    status = "recovered"

                else:
                    status = "failed"
                    failures[
                        str(error or "unknown")
                    ] += 1

                attempts = save_attempt(
                    attempts,
                    attempts_path,
                    conid=int(
                        candidate.conId
                    ),
                    source="carried_conid",
                    status=status,
                    method=method,
                    error=error,
                    expiry=candidate.expiry,
                    strike=float(
                        candidate.K
                    ),
                )

                enriched, eligible = (
                    enrich_and_filter(
                        raw,
                        target,
                        spot,
                        rate_history,
                        args,
                    )
                )

                if (
                    direct_calls % 10 == 0
                ):
                    counts_now = (
                        eligible_bin_counts(
                            eligible,
                            dte_edges,
                            m_edges,
                        )
                    )

                    covered = sum(
                        counts_now.get(
                            key2,
                            0,
                        )
                        >= args.min_per_bin
                        for key2
                        in occupied_carried_bins
                    )

                    print(
                        f"[CONID] calls="
                        f"{direct_calls} | "
                        f"hits={direct_hits} | "
                        f"eligible={len(eligible)} | "
                        f"carried bins covered="
                        f"{covered}/"
                        f"{len(occupied_carried_bins)}"
                    )

            if not progress:
                break

        # -------------------------------------------------------------
        # PHASE 2:
        # Top-up carried conIds toward target_total, but not above max/bin.
        # -------------------------------------------------------------
        print()
        print(
            "[PHASE 2] Top-up with carried conIds "
            "toward target total."
        )

        while len(eligible) < args.target_total:
            counts = eligible_bin_counts(
                eligible,
                dte_edges,
                m_edges,
            )

            available = [
                key
                for key in occupied_carried_bins
                if counts.get(
                    key,
                    0,
                ) < args.max_per_bin
                and len(
                    queues.get(
                        key,
                        [],
                    )
                ) > 0
            ]

            if not available:
                break

            available.sort(
                key=lambda key: (
                    counts.get(
                        key,
                        0,
                    ),
                    key[0],
                    key[1],
                )
            )

            progress = False

            for key in available:
                if (
                    len(eligible)
                    >= args.target_total
                ):
                    break

                if not queues.get(key):
                    continue

                candidate = (
                    queues[key].pop(0)
                )

                result, method, error = (
                    historical_by_conid(
                        ib,
                        int(candidate.conId),
                        target,
                        args.pacing_seconds,
                        args.resolve_on_failure,
                    )
                )

                direct_calls += 1
                progress = True

                if result is not None:
                    direct_hits += 1

                    if (
                        method
                        == "resolved_same_conid"
                    ):
                        resolved_hits += 1

                    row = {
                        "date": target,
                        "timestamp":
                            result["timestamp"],
                        "expiry":
                            candidate.expiry,
                        "K":
                            float(candidate.K),
                        "price":
                            result["price"],
                        "conId":
                            int(candidate.conId),
                        "localSymbol":
                            str(
                                getattr(
                                    candidate,
                                    "localSymbol",
                                    "",
                                )
                            ),
                        "source":
                            "carried_conid",
                        "conid_method":
                            method,
                    }

                    raw = merge_real_rows(
                        raw,
                        pd.DataFrame([row]),
                    )

                    status = "recovered"

                else:
                    status = "failed"

                    failures[
                        str(
                            error
                            or "unknown"
                        )
                    ] += 1

                attempts = save_attempt(
                    attempts,
                    attempts_path,
                    conid=int(
                        candidate.conId
                    ),
                    source="carried_conid",
                    status=status,
                    method=method,
                    error=error,
                    expiry=candidate.expiry,
                    strike=float(
                        candidate.K
                    ),
                )

                enriched, eligible = (
                    enrich_and_filter(
                        raw,
                        target,
                        spot,
                        rate_history,
                        args,
                    )
                )

                if (
                    direct_calls % 10 == 0
                ):
                    print(
                        f"[CONID TOPUP] "
                        f"calls={direct_calls} | "
                        f"hits={direct_hits} | "
                        f"eligible="
                        f"{len(eligible)}/"
                        f"{args.target_total}"
                    )

            if not progress:
                break

        # -------------------------------------------------------------
        # PHASE 3:
        # Short-end refresh + only still-missing bins from current chain.
        # -------------------------------------------------------------
        counts = eligible_bin_counts(
            eligible,
            dte_edges,
            m_edges,
        )

        all_bins = [
            (i, j)
            for i in range(args.dte_bins)
            for j in range(
                args.moneyness_bins
            )
        ]

        missing_bins = {
            key
            for key in all_bins
            if counts.get(
                key,
                0,
            ) < args.min_per_bin
        }

        known_conids = set(
            carried["conId"]
            .astype(int)
            .tolist()
        )

        print()
        print(
            "[PHASE 3] Current-chain refresh only "
            "for short maturities and missing bins."
        )
        print(
            f"[FALLBACK] missing bins before refresh: "
            f"{len(missing_bins)}/"
            f"{len(all_bins)}"
        )

        fallback = (
            current_chain_fallback_candidates(
                ib,
                target,
                today,
                spot,
                args,
                dte_edges,
                m_edges,
                missing_bins,
                known_conids,
            )
        )

        if fallback.empty:
            print(
                "[FALLBACK] no additional "
                "qualified candidates."
            )
        else:
            print(
                f"[FALLBACK] additional qualified "
                f"contracts: {len(fallback)}"
            )

            fallback_queues = (
                defaultdict(list)
            )

            for row in fallback.itertuples(
                index=False
            ):
                key = (
                    int(row.dte_bin),
                    int(row.m_bin),
                )
                fallback_queues[
                    key
                ].append(row)

            while True:
                counts = (
                    eligible_bin_counts(
                        eligible,
                        dte_edges,
                        m_edges,
                    )
                )

                needed = [
                    key
                    for key in all_bins
                    if counts.get(
                        key,
                        0,
                    ) < args.min_per_bin
                    and len(
                        fallback_queues.get(
                            key,
                            [],
                        )
                    ) > 0
                ]

                if not needed:
                    break

                progress = False

                for key in needed:
                    if not fallback_queues.get(
                        key
                    ):
                        continue

                    candidate = (
                        fallback_queues[
                            key
                        ].pop(0)
                    )

                    result, error = (
                        historical_midpoint(
                            ib,
                            candidate.contract,
                            target,
                            args.pacing_seconds,
                        )
                    )

                    fallback_calls += 1
                    progress = True

                    if result is not None:
                        fallback_hits += 1

                        row = {
                            "date": target,
                            "timestamp":
                                result["timestamp"],
                            "expiry":
                                candidate.expiry,
                            "K":
                                float(candidate.K),
                            "price":
                                result["price"],
                            "conId":
                                int(candidate.conId),
                            "localSymbol":
                                str(
                                    candidate.contract.localSymbol
                                ),
                            "source":
                                "current_chain_fallback",
                            "conid_method":
                                "qualified_symbolic",
                        }

                        raw = merge_real_rows(
                            raw,
                            pd.DataFrame([row]),
                        )

                    else:
                        failures[
                            str(
                                error
                                or "unknown"
                            )
                        ] += 1

                    enriched, eligible = (
                        enrich_and_filter(
                            raw,
                            target,
                            spot,
                            rate_history,
                            args,
                        )
                    )

                    if (
                        fallback_calls
                        % 10
                        == 0
                    ):
                        print(
                            f"[FALLBACK] "
                            f"calls="
                            f"{fallback_calls} | "
                            f"hits="
                            f"{fallback_hits} | "
                            f"eligible="
                            f"{len(eligible)}"
                        )

                if not progress:
                    break

        # -------------------------------------------------------------
        # Final outputs
        # -------------------------------------------------------------
        enriched, eligible = (
            enrich_and_filter(
                raw,
                target,
                spot,
                rate_history,
                args,
            )
        )

        balanced = balanced_surface(
            eligible,
            dte_edges,
            m_edges,
            args.max_per_bin,
        )

        enriched.to_csv(
            raw_path,
            index=False,
        )
        eligible.to_csv(
            all_real_path,
            index=False,
        )
        balanced.to_csv(
            balanced_path,
            index=False,
        )

        write_coverage(
            eligible,
            carried,
            dte_edges,
            m_edges,
            args,
            coverage_path,
        )

        counts = eligible_bin_counts(
            eligible,
            dte_edges,
            m_edges,
        )

        covered_all = sum(
            counts.get(
                key,
                0,
            )
            >= args.min_per_bin
            for key in all_bins
        )

        occupied_covered = sum(
            counts.get(
                key,
                0,
            )
            >= args.min_per_bin
            for key
            in occupied_carried_bins
        )

        hit_rate = (
            direct_hits
            / direct_calls
            if direct_calls
            else 0.0
        )

        expiry_count = (
            eligible["expiry"].nunique()
            if not eligible.empty
            else 0
        )

        summary = [
            f"source_date={source_date.date()}",
            f"target_date={target.date()}",
            f"source_file={source_path}",
            f"carried_conids_in_domain={len(carried)}",
            f"direct_conid_calls={direct_calls}",
            f"direct_conid_hits={direct_hits}",
            f"direct_conid_hit_rate={hit_rate:.8f}",
            f"resolved_same_conid_hits={resolved_hits}",
            f"fallback_calls={fallback_calls}",
            f"fallback_hits={fallback_hits}",
            f"final_eligible_all_real={len(eligible)}",
            f"final_balanced={len(balanced)}",
            f"unique_expiries={expiry_count}",
            f"carried_bins={len(occupied_carried_bins)}",
            f"carried_bins_covered={occupied_covered}",
            f"all_bins={len(all_bins)}",
            f"all_bins_covered={covered_all}",
        ]

        summary_path.write_text(
            "\n".join(summary),
            encoding="utf-8",
        )

        print()
        print("=" * 92)
        print("[FINAL CONID BACKWARD TEST]")
        print(
            f"direct conId calls       : "
            f"{direct_calls}"
        )
        print(
            f"direct conId hits        : "
            f"{direct_hits}"
        )
        print(
            f"direct conId hit rate    : "
            f"{hit_rate:.1%}"
        )
        print(
            f"resolved conId hits      : "
            f"{resolved_hits}"
        )
        print(
            f"fallback calls           : "
            f"{fallback_calls}"
        )
        print(
            f"fallback hits            : "
            f"{fallback_hits}"
        )
        print(
            f"eligible all-real        : "
            f"{len(eligible)}"
        )
        print(
            f"balanced surface         : "
            f"{len(balanced)}"
        )
        print(
            f"unique expiries          : "
            f"{expiry_count}"
        )
        print(
            f"carried bins covered     : "
            f"{occupied_covered}/"
            f"{len(occupied_carried_bins)}"
        )
        print(
            f"all 6x8 bins covered     : "
            f"{covered_all}/"
            f"{len(all_bins)}"
        )
        print(
            f"balanced output          : "
            f"{balanced_path}"
        )
        print(
            f"coverage audit           : "
            f"{coverage_path}"
        )
        print(
            f"summary                  : "
            f"{summary_path}"
        )

        if failures:
            print("[TOP FAILURES]")
            for reason, count in (
                failures.most_common(8)
            ):
                print(
                    f"  {count:5d}  "
                    f"{reason}"
                )

        print("=" * 92)

    finally:
        if ib.isConnected():
            ib.disconnect()


if __name__ == "__main__":
    main()
