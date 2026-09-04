"""
IBKR historical GLD surface collector -- bin-balanced replacement.

This file is intended to REPLACE:
    src/ibkr_gld_historical_surface.py

Design
------
The historical-data calls are allocated by a 2D grid in:
    DTE x moneyness (K / historical spot)

Default grid:
    6 DTE bins x 8 moneyness bins = 48 bins

Default collection rule:
    min_per_bin = 2
    max_per_bin = 4
    target_total = 100

For each occupied candidate bin:
1. Try to obtain at least min_per_bin REAL eligible observations.
2. If a bin cannot reach the minimum, exhaust ALL qualified candidates in that
   bin. Whatever real observations exist are retained.
3. Once the minimum-coverage phase is finished, top up under-represented bins
   toward max_per_bin until target_total is reached or no useful candidates
   remain.
4. If a bin contains more than max_per_bin eligible observations (for example
   because archived seed data already contain many points), ALL observations
   are preserved in an audit file, while the balanced surface keeps at most
   max_per_bin real observations from that bin.
5. No interpolation and no synthetic calibration observations are created.

Strike candidates
-----------------
Qualification is comparatively cheap, so the candidate universe contains:
- every strike returned by the current IBKR chain in the requested moneyness
  range;
- PLUS a 0.50-dollar strike ladder by default.

The expiry x strike candidate universe is qualified first. Invalid/nonexistent
half-dollar contracts are discarded automatically by IBKR qualification.

Outputs
-------
GLD_DATE_midpoint_raw.csv
    Every real MIDPOINT recovered/seeded and enriched.

GLD_DATE_eligible_all_real_surface.csv
    Every real observation passing DTE / price / IV / vega filters.

GLD_DATE_eligible_bin_balanced_surface.csv
    Bin-balanced real surface: all points if a bin is sparse; capped at
    max_per_bin if a bin is dense. This is the surface to feed into the
    downstream fixed Chebyshev-Chebyshev sampler.

GLD_DATE_eligible_adaptive_surface.csv
    Compatibility alias of the bin-balanced surface.

GLD_DATE_bin_coverage.csv
    Per-bin candidate / attempted / eligible counts.

GLD_DATE_attempts.csv
    Historical-call status by conId, allowing resume without retrying failed
    contracts unless --fresh is used.

Important IBKR limitation
-------------------------
Already-expired contracts that are no longer exposed by the current option
chain generally cannot be reconstructed. Existing sparse historical files are
therefore used as REAL seed observations when available.
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


# ---------------------------------------------------------------------------
# IBKR
# ---------------------------------------------------------------------------

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
    frame = pd.read_csv(path)
    if "timestamp" not in frame.columns or "close" not in frame.columns:
        raise ValueError("GLD stock history needs timestamp and close columns.")

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
        key=lambda c: (len(c.expirations), len(c.strikes)),
    )


def qualify(ib, contracts, batch=40):
    contracts = list(contracts)
    unique = {}

    for start in range(0, len(contracts), int(batch)):
        part = contracts[start:start + int(batch)]
        try:
            valid = ib.qualifyContracts(*part)
        except Exception as exc:
            print(f"[WARN] qualification batch failed: {exc}")
            valid = []

        for contract in valid:
            cid = int(getattr(contract, "conId", 0) or 0)
            if cid:
                unique[cid] = contract

        ib.sleep(0.08)

        print(
            f"[QUALIFY] {min(start + len(part), len(contracts))}/"
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
    }, None


# ---------------------------------------------------------------------------
# Load / merge / checkpoints
# ---------------------------------------------------------------------------

def load_existing_raw(path):
    if not path.exists():
        return pd.DataFrame()
    try:
        frame = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

    if not frame.empty and "conId" in frame.columns:
        frame["conId"] = pd.to_numeric(
            frame["conId"], errors="coerce"
        ).astype("Int64")
    return frame


def load_seed(path, target):
    """Load REAL archived observations; never interpolate."""
    if not path.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"[WARN] cannot read seed {path}: {exc}")
        return pd.DataFrame()

    if df.empty:
        return df

    if "date" in df.columns:
        d = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
        df = df.loc[d.eq(target)].copy()

    required = {"expiry", "K", "price"}
    if not required.issubset(df.columns):
        print(
            f"[WARN] seed missing columns: "
            f"{sorted(required - set(df.columns))}"
        )
        return pd.DataFrame()

    keep = [
        c for c in [
            "date", "timestamp", "expiry", "K",
            "price", "conId", "localSymbol"
        ]
        if c in df.columns
    ]
    out = df[keep].copy()

    if "date" not in out.columns:
        out["date"] = target
    if "timestamp" not in out.columns:
        out["timestamp"] = pd.NaT
    if "conId" not in out.columns:
        out["conId"] = pd.NA
    if "localSymbol" not in out.columns:
        out["localSymbol"] = ""

    out["date"] = pd.to_datetime(
        out["date"], errors="coerce"
    ).dt.normalize()
    out["expiry"] = pd.to_datetime(
        out["expiry"], errors="coerce"
    ).dt.normalize()
    out["K"] = pd.to_numeric(out["K"], errors="coerce")
    out["price"] = pd.to_numeric(out["price"], errors="coerce")
    out["conId"] = pd.to_numeric(
        out["conId"], errors="coerce"
    ).astype("Int64")

    return out.dropna(subset=["expiry", "K", "price"]).copy()


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
        cid = pd.to_numeric(df["conId"], errors="coerce")
        has_id = cid.notna()

        a = df.loc[has_id].copy()
        a["conId"] = cid.loc[has_id].astype("Int64")
        a = a.drop_duplicates("conId", keep="last")

        b = df.loc[~has_id].copy()
        if not b.empty:
            b = b.drop_duplicates(["expiry", "K"], keep="last")

        return pd.concat([a, b], ignore_index=True, sort=False)

    return df.drop_duplicates(["expiry", "K"], keep="last")


def load_attempts(path):
    if not path.exists():
        return pd.DataFrame(
            columns=["conId", "status", "error", "expiry", "K"]
        )
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame(
            columns=["conId", "status", "error", "expiry", "K"]
        )

    if "conId" in df.columns:
        df["conId"] = pd.to_numeric(
            df["conId"], errors="coerce"
        ).astype("Int64")
    return df


def append_attempt(attempts, item, path):
    attempts = pd.concat(
        [attempts, pd.DataFrame([item])],
        ignore_index=True,
        sort=False,
    )
    attempts["conId"] = pd.to_numeric(
        attempts["conId"], errors="coerce"
    ).astype("Int64")
    attempts = attempts.drop_duplicates("conId", keep="last")
    attempts.to_csv(path, index=False)
    return attempts


# ---------------------------------------------------------------------------
# Eligibility
# ---------------------------------------------------------------------------

def enrich_and_filter(raw, target, spot, rate_history, args):
    if raw.empty:
        return raw.copy(), raw.copy(), None

    raw = raw.copy()
    raw["date"] = pd.to_datetime(
        raw["date"], errors="coerce"
    ).dt.normalize()
    raw["expiry"] = pd.to_datetime(
        raw["expiry"], errors="coerce"
    ).dt.normalize()
    raw["K"] = pd.to_numeric(raw["K"], errors="coerce")
    raw["price"] = pd.to_numeric(raw["price"], errors="coerce")
    raw = raw.dropna(subset=["expiry", "K", "price"]).copy()

    raw["dte"] = (raw["expiry"] - target).dt.days
    raw["T"] = raw["dte"] / 365.25
    raw["moneyness"] = raw["K"] / float(spot)

    raw = raw.loc[
        raw["price"].gt(float(args.min_price))
        & raw["dte"].between(args.min_dte, args.max_dte)
        & raw["moneyness"].between(
            args.min_moneyness, args.max_moneyness
        )
    ].copy()

    if raw.empty:
        return raw.copy(), raw.copy(), None

    rates, curve_date = rates_for_date(
        raw["T"].to_numpy(float),
        target,
        rate_history=rate_history,
    )
    raw["rate"] = rates

    ivs, vegas = [], []

    for row in raw.itertuples(index=False):
        iv = BnS.implied_vol_call(
            row.price,
            float(spot),
            row.K,
            row.T,
            row.rate,
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
        & raw["implied_vol"].between(
            float(args.min_iv), float(args.max_iv)
        )
        & np.isfinite(raw["vega"])
        & raw["vega"].ge(float(args.min_vega))
    ].copy()

    eligible = (
        eligible
        .sort_values(["T", "K"])
        .drop_duplicates(["T", "K"], keep="last")
        .reset_index(drop=True)
    )

    return raw, eligible, curve_date


# ---------------------------------------------------------------------------
# Candidate universe
# ---------------------------------------------------------------------------

def strike_candidates(
    chain_strikes,
    spot,
    min_m,
    max_m,
    half_step,
    probe_half_strikes,
):
    low = float(spot) * float(min_m)
    high = float(spot) * float(max_m)

    actual = {
        float(k)
        for k in chain_strikes
        if np.isfinite(k) and low <= float(k) <= high
    }

    if not probe_half_strikes:
        return sorted(actual), len(actual), 0

    step = float(half_step)
    if step <= 0:
        raise ValueError("--half-strike-step must be > 0")

    start = np.ceil(low / step) * step
    end = np.floor(high / step) * step
    ladder = set()

    if end >= start:
        n = int(round((end - start) / step))
        ladder = {
            round(start + i * step, 8)
            for i in range(n + 1)
        }

    combined = sorted(actual | ladder)
    return combined, len(actual), len(ladder - actual)


def build_contracts(expiries, strikes, trading_class):
    return [
        Option(
            TICKER,
            expiry.strftime("%Y%m%d"),
            float(strike),
            "C",
            EXCHANGE,
            currency=CURRENCY,
            tradingClass=trading_class,
        )
        for expiry in expiries
        for strike in strikes
    ]


def qualified_frame(contracts, target, spot, args):
    rows = []

    for c in contracts:
        expiry = pd.to_datetime(
            c.lastTradeDateOrContractMonth[:8],
            format="%Y%m%d",
            errors="coerce",
        )
        if pd.isna(expiry):
            continue

        dte = int((expiry - target).days)
        K = float(c.strike)
        m = K / float(spot)

        if (
            args.min_dte <= dte <= args.max_dte
            and args.min_moneyness <= m <= args.max_moneyness
        ):
            rows.append({
                "conId": int(c.conId),
                "expiry": expiry,
                "K": K,
                "dte": dte,
                "moneyness": m,
                "contract": c,
            })

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .drop_duplicates("conId")
        .sort_values(["dte", "K"])
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Bins
# ---------------------------------------------------------------------------

def make_edges(a, b, n):
    return np.linspace(float(a), float(b), int(n) + 1)


def scalar_bin(value, edges):
    idx = int(np.searchsorted(edges, float(value), side="right") - 1)
    if idx == len(edges) - 1:
        idx -= 1
    if idx < 0 or idx >= len(edges) - 1:
        return None
    return idx


def bin_key(dte, moneyness, dte_edges, m_edges):
    i = scalar_bin(dte, dte_edges)
    j = scalar_bin(moneyness, m_edges)
    if i is None or j is None:
        return None
    return int(i), int(j)


def add_bins(frame, dte_edges, m_edges):
    if frame.empty:
        return frame.copy()

    out = frame.copy()
    keys = [
        bin_key(r.dte, r.moneyness, dte_edges, m_edges)
        for r in out.itertuples(index=False)
    ]
    out["dte_bin"] = [
        k[0] if k is not None else pd.NA for k in keys
    ]
    out["m_bin"] = [
        k[1] if k is not None else pd.NA for k in keys
    ]
    out = out.dropna(subset=["dte_bin", "m_bin"]).copy()
    out["dte_bin"] = out["dte_bin"].astype(int)
    out["m_bin"] = out["m_bin"].astype(int)
    return out


def rank_candidates(frame, dte_edges, m_edges):
    """
    Query representative points near each bin centre first; if a bin is sparse
    the scheduler eventually exhausts every qualified candidate in the bin.
    """
    frame = add_bins(frame, dte_edges, m_edges)
    if frame.empty:
        return frame

    dc = 0.5 * (dte_edges[:-1] + dte_edges[1:])
    mc = 0.5 * (m_edges[:-1] + m_edges[1:])
    dw = np.diff(dte_edges)
    mw = np.diff(m_edges)

    frame["bin_distance"] = [
        np.sqrt(
            ((r.dte - dc[r.dte_bin]) / dw[r.dte_bin]) ** 2
            + ((r.moneyness - mc[r.m_bin]) / mw[r.m_bin]) ** 2
        )
        for r in frame.itertuples(index=False)
    ]

    return frame.sort_values(
        ["dte_bin", "m_bin", "bin_distance", "dte", "K"]
    ).reset_index(drop=True)


def eligible_counts(eligible, dte_edges, m_edges):
    counts = Counter()

    if eligible.empty:
        return counts

    for r in eligible.itertuples(index=False):
        key = bin_key(r.dte, r.moneyness, dte_edges, m_edges)
        if key is not None:
            counts[key] += 1

    return counts


def build_queues(candidates, attempted_ids):
    queues = defaultdict(list)

    for r in candidates.itertuples(index=False):
        if int(r.conId) in attempted_ids:
            continue
        queues[(int(r.dte_bin), int(r.m_bin))].append(r)

    return queues


def select_spread_within_bin(group, n):
    """
    Deterministic cap for an over-populated bin. It preserves REAL observations
    and spreads them in normalized (T, moneyness) space.

    This is only the density cap. The official fixed CC/Chebyshev sampler is
    still applied downstream to the resulting balanced surface.
    """
    group = group.copy().reset_index(drop=True)

    if len(group) <= int(n):
        return group

    x = pd.to_numeric(group["T"], errors="coerce").to_numpy(float)
    y = pd.to_numeric(group["moneyness"], errors="coerce").to_numpy(float)

    def norm(v):
        lo, hi = np.nanmin(v), np.nanmax(v)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return np.zeros_like(v)
        return (v - lo) / (hi - lo)

    pts = np.column_stack([norm(x), norm(y)])

    # Start with point nearest the bin centre, then farthest-point sampling.
    centre = np.array([0.5, 0.5])
    first = int(np.argmin(np.sum((pts - centre) ** 2, axis=1)))
    chosen = [first]

    while len(chosen) < int(n):
        remaining = [i for i in range(len(group)) if i not in chosen]
        if not remaining:
            break

        dmin = []
        for i in remaining:
            distances = [
                np.linalg.norm(pts[i] - pts[j])
                for j in chosen
            ]
            dmin.append(min(distances))

        chosen.append(remaining[int(np.argmax(dmin))])

    return group.iloc[sorted(chosen)].copy()


def balanced_surface(eligible, dte_edges, m_edges, max_per_bin):
    """
    Sparse bin -> keep ALL real points.
    Dense bin  -> keep at most max_per_bin representative real points.
    """
    if eligible.empty:
        return eligible.copy()

    work = add_bins(eligible, dte_edges, m_edges)
    pieces = []

    for _, group in work.groupby(["dte_bin", "m_bin"], sort=True):
        if len(group) <= int(max_per_bin):
            pieces.append(group.copy())
        else:
            pieces.append(
                select_spread_within_bin(group, int(max_per_bin))
            )

    if not pieces:
        return eligible.iloc[0:0].copy()

    out = pd.concat(pieces, ignore_index=True)
    return (
        out
        .drop(columns=["dte_bin", "m_bin"], errors="ignore")
        .sort_values(["T", "K"])
        .reset_index(drop=True)
    )


def coverage_table(
    candidates,
    attempted_ids,
    eligible,
    dte_edges,
    m_edges,
    args,
):
    candidate_counts = Counter(
        zip(
            candidates["dte_bin"].astype(int),
            candidates["m_bin"].astype(int),
        )
    )

    attempted_counts = Counter()
    if attempted_ids:
        attempted_subset = candidates.loc[
            candidates["conId"].astype(int).isin(attempted_ids)
        ]
        attempted_counts = Counter(
            zip(
                attempted_subset["dte_bin"].astype(int),
                attempted_subset["m_bin"].astype(int),
            )
        )

    elig_counts = eligible_counts(eligible, dte_edges, m_edges)

    rows = []

    for i in range(args.dte_bins):
        for j in range(args.moneyness_bins):
            key = (i, j)
            cand = int(candidate_counts.get(key, 0))
            elig = int(elig_counts.get(key, 0))
            attempted = int(attempted_counts.get(key, 0))

            rows.append({
                "dte_bin": i,
                "moneyness_bin": j,
                "dte_low": dte_edges[i],
                "dte_high": dte_edges[i + 1],
                "moneyness_low": m_edges[j],
                "moneyness_high": m_edges[j + 1],
                "qualified_candidates": cand,
                "attempted_candidates": attempted,
                "eligible_all_real": elig,
                "minimum_per_bin": args.min_per_bin,
                "maximum_per_bin": args.max_per_bin,
                "candidate_bin_exhausted": bool(cand > 0 and attempted >= cand),
                "below_minimum": bool(cand > 0 and elig < args.min_per_bin),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Historical querying
# ---------------------------------------------------------------------------

def query_candidate(
    ib,
    candidate,
    target,
    pacing,
    raw_rows,
    attempts,
    attempts_path,
):
    contract = candidate.contract
    cid = int(candidate.conId)

    row, error = historical_midpoint(
        ib, contract, target, pacing
    )

    attempts = append_attempt(
        attempts,
        {
            "conId": cid,
            "status": "recovered" if row is not None else "failed",
            "error": "" if row is not None else str(error or "unknown"),
            "expiry": candidate.expiry,
            "K": candidate.K,
        },
        attempts_path,
    )

    if row is not None:
        raw_rows = merge_real_rows(
            raw_rows,
            pd.DataFrame([row]),
        )

    return raw_rows, attempts, row is not None, error


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)

    p.add_argument("--date", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7497)
    p.add_argument("--client-id", type=int, default=94)

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
    p.add_argument(
        "--historical-dir",
        default="data/processed/sparse_historical_surfaces",
    )

    p.add_argument("--min-moneyness", type=float, default=0.60)
    p.add_argument("--max-moneyness", type=float, default=1.40)
    p.add_argument("--min-dte", type=int, default=75)
    p.add_argument("--max-dte", type=int, default=730)

    p.add_argument("--dte-bins", type=int, default=6)
    p.add_argument("--moneyness-bins", type=int, default=8)
    p.add_argument("--min-per-bin", type=int, default=2)
    p.add_argument("--max-per-bin", type=int, default=4)
    p.add_argument("--target-total", type=int, default=100)
    p.add_argument("--min-unique-expiries", type=int, default=8)

    p.add_argument(
        "--half-strike-step",
        type=float,
        default=0.50,
        help="Probe additional strike candidates every $0.50 before qualification.",
    )
    p.add_argument(
        "--no-half-strikes",
        action="store_true",
        help="Use only strikes returned directly by reqSecDefOptParams.",
    )

    p.add_argument("--min-price", type=float, default=0.05)
    p.add_argument("--min-iv", type=float, default=0.03)
    p.add_argument("--max-iv", type=float, default=1.50)
    p.add_argument("--min-vega", type=float, default=0.10)

    p.add_argument("--pacing-seconds", type=float, default=0.15)
    p.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help="Rebuild/save eligible surfaces every N historical calls.",
    )

    p.add_argument("--no-historical-seed", action="store_true")
    p.add_argument(
        "--fresh",
        action="store_true",
        help="Delete raw/attempt checkpoints for this date and start again.",
    )

    args = p.parse_args()

    if args.min_per_bin < 1:
        raise ValueError("--min-per-bin must be >= 1")
    if args.max_per_bin < args.min_per_bin:
        raise ValueError("--max-per-bin must be >= --min-per-bin")
    if args.target_total < 1:
        raise ValueError("--target-total must be >= 1")

    target = pd.Timestamp(args.date).normalize()
    today = pd.Timestamp.now().normalize()

    if target >= today:
        raise ValueError(
            "This is the historical collector. "
            "Use ibkr_gld_today_surface.py for today."
        )

    spot = get_spot(args.stock, target)
    rate_history = load_rate_history(args.rates)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    slug = target.strftime("%Y-%m-%d")

    raw_path = out / f"GLD_{slug}_midpoint_raw.csv"
    all_real_path = out / f"GLD_{slug}_eligible_all_real_surface.csv"
    balanced_path = out / f"GLD_{slug}_eligible_bin_balanced_surface.csv"
    compatibility_path = out / f"GLD_{slug}_eligible_adaptive_surface.csv"
    attempts_path = out / f"GLD_{slug}_attempts.csv"
    coverage_path = out / f"GLD_{slug}_bin_coverage.csv"

    seed_path = (
        Path(args.historical_dir)
        / f"GLD_{slug}_eligible_historical_surface.csv"
    )

    if args.fresh:
        for path in [
            raw_path,
            all_real_path,
            balanced_path,
            compatibility_path,
            attempts_path,
            coverage_path,
        ]:
            if path.exists():
                print(f"[FRESH] deleting {path}")
                path.unlink()

    existing_raw = load_existing_raw(raw_path)

    seed = pd.DataFrame()
    if not args.no_historical_seed:
        seed = load_seed(seed_path, target)
        if not seed.empty:
            print(
                f"[SEED] loaded {len(seed)} REAL archived observations "
                f"from {seed_path}"
            )
        else:
            print("[SEED] no usable archived historical seed")

    raw_rows = merge_real_rows(existing_raw, seed)
    attempts = load_attempts(attempts_path)

    dte_edges = make_edges(args.min_dte, args.max_dte, args.dte_bins)
    m_edges = make_edges(
        args.min_moneyness,
        args.max_moneyness,
        args.moneyness_bins,
    )

    ib = connect(args.host, args.port, args.client_id)

    try:
        underlying = ib.qualifyContracts(
            Stock(TICKER, EXCHANGE, CURRENCY)
        )[0]
        chain = get_chain(ib, underlying)

        expiries = []
        expired_relevant = 0

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
                expired_relevant += 1
                continue

            expiries.append(expiry)

        expiries = sorted(set(expiries))

        strikes, actual_strikes, half_extra = strike_candidates(
            chain.strikes,
            spot,
            args.min_moneyness,
            args.max_moneyness,
            args.half_strike_step,
            not args.no_half_strikes,
        )

        print("=" * 90)
        print(f"TARGET DATE                    : {target.date()}")
        print(f"HISTORICAL GLD SPOT            : {spot:.4f}")
        print(f"DTE DOMAIN                     : {args.min_dte} -> {args.max_dte}")
        print(
            f"MONEYNESS DOMAIN               : "
            f"{args.min_moneyness:.2f} -> {args.max_moneyness:.2f}"
        )
        print(f"QUERYABLE EXPIRIES             : {len(expiries)}")
        print(f"EXPIRED RELEVANT EXPIRIES LOST : {expired_relevant}")
        print(f"CHAIN STRIKES IN RANGE         : {actual_strikes}")
        print(f"EXTRA HALF-STRIKE PROBES       : {half_extra}")
        print(f"TOTAL STRIKE CANDIDATES        : {len(strikes)}")
        print(
            f"RAW QUALIFICATION UNIVERSE     : "
            f"{len(expiries) * len(strikes)}"
        )
        print(
            f"BIN RULE                       : "
            f"{args.dte_bins}x{args.moneyness_bins}, "
            f"min={args.min_per_bin}, max={args.max_per_bin}"
        )
        print(f"TARGET TOTAL ELIGIBLE          : {args.target_total}")
        print("=" * 90)

        if not expiries or not strikes:
            raise RuntimeError("No queryable expiry/strike candidates.")

        candidates = build_contracts(
            expiries,
            strikes,
            chain.tradingClass,
        )

        qualified = qualify(ib, candidates)

        qf = qualified_frame(
            qualified,
            target,
            spot,
            args,
        )
        qf = rank_candidates(
            qf,
            dte_edges,
            m_edges,
        )

        if qf.empty:
            raise RuntimeError("No qualified historical candidate contracts.")

        print(f"[INFO] qualified candidate contracts: {len(qf)}")

        occupied_bins = sorted(
            set(
                zip(
                    qf["dte_bin"].astype(int),
                    qf["m_bin"].astype(int),
                )
            )
        )
        print(
            f"[INFO] occupied candidate bins: "
            f"{len(occupied_bins)}/{args.dte_bins * args.moneyness_bins}"
        )

        # Seed/existing data are enriched before deciding which bins need calls.
        enriched, eligible, _ = enrich_and_filter(
            raw_rows,
            target,
            spot,
            rate_history,
            args,
        )

        attempted_ids = set(
            pd.to_numeric(
                attempts.get("conId", pd.Series(dtype=float)),
                errors="coerce",
            ).dropna().astype(int)
        )

        queues = build_queues(qf, attempted_ids)
        failures = Counter()
        calls = 0

        def refresh_outputs():
            nonlocal enriched, eligible

            enriched, eligible, _ = enrich_and_filter(
                raw_rows,
                target,
                spot,
                rate_history,
                args,
            )

            if enriched.empty:
                pd.DataFrame().to_csv(raw_path, index=False)
            else:
                enriched.to_csv(raw_path, index=False)

            if eligible.empty:
                pd.DataFrame().to_csv(all_real_path, index=False)
                pd.DataFrame().to_csv(balanced_path, index=False)
                pd.DataFrame().to_csv(compatibility_path, index=False)
            else:
                eligible.to_csv(all_real_path, index=False)

                balanced = balanced_surface(
                    eligible,
                    dte_edges,
                    m_edges,
                    args.max_per_bin,
                )
                balanced.to_csv(balanced_path, index=False)
                balanced.to_csv(compatibility_path, index=False)

            attempted_now = set(
                pd.to_numeric(
                    attempts.get("conId", pd.Series(dtype=float)),
                    errors="coerce",
                ).dropna().astype(int)
            )

            coverage = coverage_table(
                qf,
                attempted_now,
                eligible,
                dte_edges,
                m_edges,
                args,
            )
            coverage.to_csv(coverage_path, index=False)

            return balanced_surface(
                eligible,
                dte_edges,
                m_edges,
                args.max_per_bin,
            )

        balanced = refresh_outputs()

        # ---------------------------------------------------------------
        # PHASE 1: minimum coverage.
        # If a bin cannot reach min_per_bin, exhaust ALL candidates in it.
        # ---------------------------------------------------------------
        print()
        print("[PHASE 1] Fill every occupied bin to the minimum where possible.")

        while True:
            counts = eligible_counts(eligible, dte_edges, m_edges)

            deficient = [
                key
                for key in occupied_bins
                if counts.get(key, 0) < args.min_per_bin
                and len(queues.get(key, [])) > 0
            ]

            if not deficient:
                break

            progress = False

            # One call per deficient bin per sweep => balanced collection.
            for key in deficient:
                if not queues.get(key):
                    continue

                candidate = queues[key].pop(0)

                raw_rows, attempts, ok, err = query_candidate(
                    ib,
                    candidate,
                    target,
                    args.pacing_seconds,
                    raw_rows,
                    attempts,
                    attempts_path,
                )

                calls += 1
                progress = True

                if not ok:
                    failures[str(err or "unknown")] += 1

                if (
                    args.checkpoint_every > 0
                    and calls % args.checkpoint_every == 0
                ):
                    balanced = refresh_outputs()
                    counts2 = eligible_counts(
                        eligible, dte_edges, m_edges
                    )
                    below = sum(
                        counts2.get(k, 0) < args.min_per_bin
                        for k in occupied_bins
                    )
                    print(
                        f"[PHASE 1] calls={calls} | "
                        f"eligible={len(eligible)} | "
                        f"balanced={len(balanced)} | "
                        f"bins below min={below}"
                    )

            balanced = refresh_outputs()

            if not progress:
                break

        # ---------------------------------------------------------------
        # PHASE 2: top up toward target_total but never intentionally
        # overfill a bin beyond max_per_bin.
        # ---------------------------------------------------------------
        print()
        print(
            "[PHASE 2] Top up under-represented bins toward the total target, "
            "capped by max-per-bin."
        )

        while len(eligible) < args.target_total:
            counts = eligible_counts(eligible, dte_edges, m_edges)

            available = [
                key
                for key in occupied_bins
                if counts.get(key, 0) < args.max_per_bin
                and len(queues.get(key, [])) > 0
            ]

            if not available:
                break

            # Most under-filled bins first.
            available.sort(
                key=lambda key: (
                    counts.get(key, 0),
                    key[0],
                    key[1],
                )
            )

            progress = False

            for key in available:
                if len(eligible) >= args.target_total:
                    break

                if not queues.get(key):
                    continue

                candidate = queues[key].pop(0)

                raw_rows, attempts, ok, err = query_candidate(
                    ib,
                    candidate,
                    target,
                    args.pacing_seconds,
                    raw_rows,
                    attempts,
                    attempts_path,
                )

                calls += 1
                progress = True

                if not ok:
                    failures[str(err or "unknown")] += 1

                if (
                    args.checkpoint_every > 0
                    and calls % args.checkpoint_every == 0
                ):
                    balanced = refresh_outputs()
                    print(
                        f"[PHASE 2] calls={calls} | "
                        f"eligible={len(eligible)}/{args.target_total} | "
                        f"balanced={len(balanced)}"
                    )

            balanced = refresh_outputs()

            if not progress:
                break

        # Final save / report.
        balanced = refresh_outputs()
        counts = eligible_counts(eligible, dte_edges, m_edges)

        bins_below_min = [
            key
            for key in occupied_bins
            if counts.get(key, 0) < args.min_per_bin
        ]

        attempted_ids = set(
            pd.to_numeric(
                attempts.get("conId", pd.Series(dtype=float)),
                errors="coerce",
            ).dropna().astype(int)
        )
        remaining_queues = build_queues(qf, attempted_ids)

        exhausted_below_min = [
            key
            for key in bins_below_min
            if len(remaining_queues.get(key, [])) == 0
        ]

        expiry_count = (
            eligible["expiry"].nunique()
            if not eligible.empty
            else 0
        )

        print()
        print("=" * 90)
        print(f"[DONE] historical calls this run     : {calls}")
        print(f"[DONE] all REAL eligible observations : {len(eligible)}")
        print(f"[DONE] bin-balanced observations      : {len(balanced)}")
        print(f"[DONE] unique expiries                : {expiry_count}")
        print(f"[DONE] occupied candidate bins        : {len(occupied_bins)}")
        print(f"[DONE] bins below minimum             : {len(bins_below_min)}")
        print(
            f"[DONE] below-min bins fully exhausted : "
            f"{len(exhausted_below_min)}"
        )
        print(f"[DONE] all-real surface               : {all_real_path}")
        print(f"[DONE] balanced surface               : {balanced_path}")
        print(f"[DONE] compatibility surface          : {compatibility_path}")
        print(f"[DONE] coverage audit                 : {coverage_path}")
        print(f"[DONE] attempts/checkpoint            : {attempts_path}")

        if failures:
            print("[FAILURES]")
            for reason, n in failures.most_common(10):
                print(f"  {n:5d}  {reason}")

        if bins_below_min:
            print(
                "[NOTE] Some bins remain below the requested minimum. "
                "For exhausted bins every qualified candidate was tried, "
                "so all recoverable real observations in those bins were kept."
            )

        if expiry_count < args.min_unique_expiries:
            print(
                f"[WARN] only {expiry_count} eligible expiries; "
                f"requested minimum was {args.min_unique_expiries}."
            )

        print(
            "[NEXT] Feed GLD_DATE_eligible_bin_balanced_surface.csv "
            "to the fixed CC/Chebyshev 8x8 sampler."
        )
        print("=" * 90)

    finally:
        if ib.isConnected():
            ib.disconnect()


if __name__ == "__main__":
    main()
