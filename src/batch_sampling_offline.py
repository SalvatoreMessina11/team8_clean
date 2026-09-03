"""
OFFLINE-ONLY sampling audit.

This script NEVER connects to IBKR.

It scans already-downloaded files matching:

    data/processed/full_surfaces/GLD_YYYY-MM-DD_eligible_full_surface.csv

For every available date with at least 64 eligible observations it compares:

    UU = Uniform T + Uniform K
    CU = Chebyshev T + Uniform K
    UC = Uniform T + Chebyshev K
    CC = Chebyshev T + Chebyshev K

and aggregates the holdout interpolation errors across dates.

Primary ranking:
    mean holdout L_inf across available dates.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from Sampling import Sampling


STRATEGIES = {
    "UU": ("uniform", "uniform"),
    "CU": ("chebyshev", "uniform"),
    "UC": ("uniform", "chebyshev"),
    "CC": ("chebyshev", "chebyshev"),
}


DATE_RE = re.compile(
    r"GLD_(\d{4}-\d{2}-\d{2})_eligible_full_surface\.csv$"
)


def discover_surfaces(input_dir, start=None, end=None):
    input_dir = Path(input_dir)
    rows = []

    for path in sorted(input_dir.glob("GLD_*_eligible_full_surface.csv")):
        match = DATE_RE.search(path.name)
        if not match:
            continue
        date = pd.Timestamp(match.group(1)).normalize()

        if start is not None and date < pd.Timestamp(start):
            continue
        if end is not None and date > pd.Timestamp(end):
            continue

        rows.append((date, path))

    return rows


def load_surface(path):
    df = pd.read_csv(path)

    required = {"T", "K", "implied_vol"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"{path.name} missing columns: {sorted(missing)}"
        )

    for col in ["T", "K", "implied_vol"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["T", "K", "implied_vol"]).copy()
    df = df.loc[
        df["T"].gt(0)
        & df["K"].gt(0)
        & df["implied_vol"].gt(0)
    ].copy()

    df = (
        df.drop_duplicates(["T", "K"], keep="last")
        .sort_values(["T", "K"])
        .reset_index(drop=True)
    )
    df["_row_id"] = np.arange(len(df), dtype=int)
    return df


def evaluate_date(date, full, output_dir, n_t=8, n_k=8):
    required_n = int(n_t) * int(n_k)
    if len(full) < required_n:
        return [], {
            "date": date.strftime("%Y-%m-%d"),
            "status": "skip_lt_64",
            "eligible_points": int(len(full)),
        }

    date_out = Path(output_dir) / date.strftime("%Y-%m-%d")
    date_out.mkdir(parents=True, exist_ok=True)

    rows = []

    for strategy, (t_scheme, k_scheme) in STRATEGIES.items():
        sample = Sampling.sample_hybrid(
            full,
            t_scheme=t_scheme,
            k_scheme=k_scheme,
            n_T=n_t,
            n_K=n_k,
        )

        diag = Sampling.interpolation_diagnostics(full, sample)

        sample.to_csv(
            date_out / f"sample_{strategy}_{required_n}.csv",
            index=False,
        )

        rows.append({
            "date": date.strftime("%Y-%m-%d"),
            "strategy": strategy,
            "T_sampling": t_scheme,
            "K_sampling": k_scheme,
            "eligible_points": int(len(full)),
            "holdout_n": int(diag["holdout"]["n"]),
            "holdout_mae_bps_iv":
                10000.0 * float(diag["holdout"]["mae"]),
            "holdout_rmse_bps_iv":
                10000.0 * float(diag["holdout"]["rmse"]),
            "holdout_linf_bps_iv":
                float(diag["holdout"]["linf_bps_iv"]),
        })

    daily = pd.DataFrame(rows).sort_values(
        [
            "holdout_linf_bps_iv",
            "holdout_rmse_bps_iv",
            "holdout_mae_bps_iv",
        ]
    ).reset_index(drop=True)
    daily["daily_rank"] = np.arange(1, len(daily) + 1)
    daily.to_csv(date_out / "sampling_comparison.csv", index=False)

    return rows, {
        "date": date.strftime("%Y-%m-%d"),
        "status": "ok",
        "eligible_points": int(len(full)),
    }


def aggregate(rows):
    raw = pd.DataFrame(rows)
    if raw.empty:
        return raw, pd.DataFrame()

    summary = (
        raw.groupby(
            ["strategy", "T_sampling", "K_sampling"],
            as_index=False,
        )
        .agg(
            dates_used=("date", "nunique"),
            mean_holdout_linf_bps_iv=(
                "holdout_linf_bps_iv", "mean"
            ),
            median_holdout_linf_bps_iv=(
                "holdout_linf_bps_iv", "median"
            ),
            std_holdout_linf_bps_iv=(
                "holdout_linf_bps_iv", "std"
            ),
            max_holdout_linf_bps_iv=(
                "holdout_linf_bps_iv", "max"
            ),
            mean_holdout_rmse_bps_iv=(
                "holdout_rmse_bps_iv", "mean"
            ),
            mean_holdout_mae_bps_iv=(
                "holdout_mae_bps_iv", "mean"
            ),
        )
    )

    daily_winners = (
        raw.sort_values(
            [
                "date",
                "holdout_linf_bps_iv",
                "holdout_rmse_bps_iv",
                "holdout_mae_bps_iv",
            ]
        )
        .groupby("date", as_index=False)
        .first()
        .groupby("strategy")
        .size()
        .rename("daily_wins")
        .reset_index()
    )

    summary = summary.merge(
        daily_winners,
        on="strategy",
        how="left",
    )
    summary["daily_wins"] = (
        summary["daily_wins"].fillna(0).astype(int)
    )

    summary = summary.sort_values(
        [
            "mean_holdout_linf_bps_iv",
            "mean_holdout_rmse_bps_iv",
            "mean_holdout_mae_bps_iv",
        ]
    ).reset_index(drop=True)

    summary["overall_rank"] = np.arange(
        1, len(summary) + 1
    )

    return raw, summary


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-dir",
        default="data/processed/full_surfaces",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/sampling_audit_all",
    )
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--n-t", type=int, default=8)
    parser.add_argument("--n-k", type=int, default=8)

    args = parser.parse_args()

    surfaces = discover_surfaces(
        args.input_dir,
        start=args.start,
        end=args.end,
    )

    if not surfaces:
        raise ValueError(
            "No existing full-surface CSV files found. "
            "This script does not download anything from IBKR."
        )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("OFFLINE SAMPLING AUDIT")
    print(f"Existing surfaces found: {len(surfaces)}")
    print("IBKR connection: NONE")
    print("=" * 90)

    rows = []
    statuses = []

    for i, (date, path) in enumerate(surfaces, 1):
        print(
            f"[{i}/{len(surfaces)}] "
            f"{date.date()} <- {path.name}"
        )

        try:
            full = load_surface(path)
            new_rows, status = evaluate_date(
                date,
                full,
                output_dir=out,
                n_t=args.n_t,
                n_k=args.n_k,
            )
        except Exception as exc:
            statuses.append({
                "date": date.strftime("%Y-%m-%d"),
                "status": f"error: {exc}",
                "eligible_points": np.nan,
            })
            print(f"    [ERROR] {exc}")
            continue

        rows.extend(new_rows)
        statuses.append(status)

        if new_rows:
            daily = pd.DataFrame(new_rows).sort_values(
                "holdout_linf_bps_iv"
            )
            winner = daily.iloc[0]
            print(
                f"    eligible={len(full)} | "
                f"winner={winner['strategy']} | "
                f"L_inf={winner['holdout_linf_bps_iv']:.3f} IV bp"
            )
        else:
            print(
                f"    skipped: {len(full)} < "
                f"{args.n_t * args.n_k}"
            )

    raw, summary = aggregate(rows)

    pd.DataFrame(statuses).to_csv(
        out / "date_status.csv",
        index=False,
    )

    if not raw.empty:
        raw.to_csv(
            out / "all_date_sampling_errors.csv",
            index=False,
        )
        summary.to_csv(
            out / "sampling_error_summary.csv",
            index=False,
        )

    print()
    print("=" * 110)
    print("FINAL CROSS-DATE SUMMARY")
    print("=" * 110)

    if summary.empty:
        print("[WARN] No date had >=64 eligible observations.")
        return

    cols = [
        "overall_rank",
        "strategy",
        "T_sampling",
        "K_sampling",
        "dates_used",
        "daily_wins",
        "mean_holdout_linf_bps_iv",
        "median_holdout_linf_bps_iv",
        "mean_holdout_rmse_bps_iv",
        "mean_holdout_mae_bps_iv",
    ]
    print(summary[cols].to_string(index=False))

    winner = summary.iloc[0]
    print("=" * 110)
    print(
        f"[OK] OVERALL WINNER: {winner['strategy']} | "
        f"mean holdout L_inf = "
        f"{winner['mean_holdout_linf_bps_iv']:.3f} IV bp | "
        f"dates = {int(winner['dates_used'])}"
    )
    print(
        f"[OK] summary: "
        f"{out / 'sampling_error_summary.csv'}"
    )


if __name__ == "__main__":
    main()
