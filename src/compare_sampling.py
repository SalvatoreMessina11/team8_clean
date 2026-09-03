"""Compare four 64-node sampling geometries before stochastic-model calibration."""

from __future__ import annotations

import argparse
import json
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


def load_surface(path):
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path)

    required = {"T", "K", "implied_vol"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Surface missing columns: {sorted(missing)}")

    for col in ["T", "K", "implied_vol"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=["T", "K", "implied_vol"]).copy()
    frame = frame.loc[
        frame["T"].gt(0)
        & frame["K"].gt(0)
        & frame["implied_vol"].gt(0)
    ].copy()
    frame = frame.drop_duplicates(["T", "K"], keep="last")
    frame = frame.sort_values(["T", "K"]).reset_index(drop=True)
    frame["_row_id"] = np.arange(len(frame), dtype=int)
    return frame


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--surface", required=True)
    p.add_argument("--output-dir", default="outputs/sampling_audit")
    p.add_argument("--n-t", type=int, default=8)
    p.add_argument("--n-k", type=int, default=8)
    args = p.parse_args()

    full = load_surface(args.surface)
    required_n = args.n_t * args.n_k
    if len(full) < required_n:
        raise ValueError(
            f"64-node comparison impossible: full eligible surface has "
            f"{len(full)} points, requires at least {required_n}."
        )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    full.to_csv(out / "full_eligible_surface.csv", index=False)

    rows = []
    details = {}

    for code, (t_scheme, k_scheme) in STRATEGIES.items():
        sample = Sampling.sample_hybrid(
            full,
            t_scheme=t_scheme,
            k_scheme=k_scheme,
            n_T=args.n_t,
            n_K=args.n_k,
        )
        diag = Sampling.interpolation_diagnostics(full, sample)
        sample.to_csv(out / f"sample_{code}_64.csv", index=False)

        row = {
            "strategy": code,
            "T_sampling": t_scheme,
            "K_sampling": k_scheme,
            "n_full": len(full),
            "n_sample": len(sample),
            "all_mae_bps_iv": 10000.0 * diag["all"]["mae"],
            "all_rmse_bps_iv": 10000.0 * diag["all"]["rmse"],
            "all_linf_bps_iv": diag["all"]["linf_bps_iv"],
            "holdout_n": diag["holdout"]["n"],
            "holdout_mae_bps_iv": 10000.0 * diag["holdout"]["mae"],
            "holdout_rmse_bps_iv": 10000.0 * diag["holdout"]["rmse"],
            "holdout_linf_bps_iv": diag["holdout"]["linf_bps_iv"],
        }
        rows.append(row)
        details[code] = diag

    comparison = pd.DataFrame(rows)

    # Primary criterion requested: infinity norm on unseen market points.
    # Tie-breakers: holdout RMSE then holdout MAE.
    comparison = comparison.sort_values(
        [
            "holdout_linf_bps_iv",
            "holdout_rmse_bps_iv",
            "holdout_mae_bps_iv",
        ],
        ascending=True,
    ).reset_index(drop=True)
    comparison["rank"] = np.arange(1, len(comparison) + 1)
    comparison.to_csv(out / "sampling_comparison.csv", index=False)

    winner = comparison.iloc[0].to_dict()
    payload = {
        "selection_rule": (
            "minimize holdout infinity norm; tie-break by holdout RMSE and MAE"
        ),
        "winner": winner,
        "diagnostics": details,
    }
    (out / "sampling_winner.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    print("=" * 86)
    print(f"FULL ELIGIBLE SURFACE: {len(full)} observations")
    print(f"SAMPLE SIZE          : {required_n} = {args.n_t} x {args.n_k}")
    print("=" * 86)
    print(
        comparison[
            [
                "rank",
                "strategy",
                "T_sampling",
                "K_sampling",
                "holdout_mae_bps_iv",
                "holdout_rmse_bps_iv",
                "holdout_linf_bps_iv",
            ]
        ].to_string(index=False)
    )
    print("=" * 86)
    print(
        f"[OK] Winner: {comparison.iloc[0]['strategy']} "
        f"(holdout L_inf = "
        f"{comparison.iloc[0]['holdout_linf_bps_iv']:.3f} IV bp)"
    )
    print(f"[OK] Results: {out}")


if __name__ == "__main__":
    main()
