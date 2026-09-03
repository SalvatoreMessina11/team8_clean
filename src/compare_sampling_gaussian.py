"""Extended Team 8 sampling comparison with center-weighted Gaussian maturity nodes.

This script compares eight 8x8 geometries on the SAME eligible market surface:

    UU = Uniform T, Uniform K
    CU = Chebyshev T, Uniform K
    UC = Uniform T, Chebyshev K
    CC = Chebyshev T, Chebyshev K

    EU = Exponential T (more nodes near Tmin), Uniform K
    EC = Exponential T (more nodes near Tmin), Chebyshev K

    GU = Gaussian-centered T (more nodes near the middle), Uniform K
    GC = Gaussian-centered T (more nodes near the middle), Chebyshev K

The Gaussian design is useful as a deliberately center-weighted benchmark:
it allocates relatively fewer target maturities near the short and long ends
and more target maturities around the middle of [Tmin, Tmax].

IMPORTANT:
The theoretical target grid is NEVER treated as synthetic option data.
Every grid target is mapped to the nearest still-unused ACTUAL market
observation, exactly as in the existing Team 8 Sampling.py methodology.

Primary criterion:
    holdout L_inf of IV reconstruction error.
Tie-breakers:
    holdout RMSE, then holdout MAE.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from Sampling import Sampling


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


def exponential_nodes(n, a, b, lam=1.0):
    """Asymmetric grid concentrated toward Tmin."""
    n = int(n)
    lam = float(lam)

    if n < 2:
        raise ValueError("n must be at least 2.")
    if not np.isfinite(a) or not np.isfinite(b) or b <= a:
        raise ValueError("Sampling axis must have a non-zero finite range.")
    if not np.isfinite(lam) or lam <= 0:
        raise ValueError("--lambda-t must be strictly positive.")

    u = np.linspace(0.0, 1.0, n)
    scaled = np.expm1(lam * u) / np.expm1(lam)
    return a + (b - a) * scaled


def gaussian_center_nodes(n, a, b, z_clip=2.0):
    """Symmetric maturity grid concentrated around the interval midpoint.

    We take equally spaced cumulative probabilities from a truncated standard
    normal on [-z_clip, +z_clip], transform them through the normal inverse CDF,
    and linearly map the resulting z-values to [a, b].

    Because normal density is highest around z=0, the resulting target
    maturities are closer together near the middle and farther apart toward
    Tmin and Tmax.

    Larger z_clip => stronger concentration around the middle.
    """
    n = int(n)
    z_clip = float(z_clip)

    if n < 2:
        raise ValueError("n must be at least 2.")
    if not np.isfinite(a) or not np.isfinite(b) or b <= a:
        raise ValueError("Sampling axis must have a non-zero finite range.")
    if not np.isfinite(z_clip) or z_clip <= 0:
        raise ValueError("--gaussian-z must be strictly positive.")

    p_lo = norm.cdf(-z_clip)
    p_hi = norm.cdf(z_clip)
    probs = np.linspace(p_lo, p_hi, n)

    z = norm.ppf(probs)

    # Numerical protection at the exact truncated endpoints.
    z = np.clip(z, -z_clip, z_clip)

    # Map [-z_clip, z_clip] -> [a, b].
    scaled = (z + z_clip) / (2.0 * z_clip)
    return a + (b - a) * scaled


def axis_nodes(series, n, scheme, lam=1.0, gaussian_z=2.0):
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        raise ValueError("Cannot build nodes on an empty axis.")

    a, b = float(values.min()), float(values.max())

    if scheme == "uniform":
        return np.linspace(a, b, int(n))
    if scheme == "chebyshev":
        return Sampling.chebyshev_roots(int(n), a, b)
    if scheme == "exponential":
        return exponential_nodes(int(n), a, b, lam=lam)
    if scheme == "gaussian":
        return gaussian_center_nodes(int(n), a, b, z_clip=gaussian_z)

    raise ValueError(f"Unknown scheme: {scheme}")


def sample_strategy(full, t_scheme, k_scheme, n_t, n_k, lam, gaussian_z):
    target_t = axis_nodes(
        full["T"],
        n_t,
        t_scheme,
        lam=lam,
        gaussian_z=gaussian_z,
    )
    target_k = axis_nodes(
        full["K"],
        n_k,
        k_scheme,
        lam=lam,
        gaussian_z=gaussian_z,
    )

    sample = Sampling.get_nearest_market_points(
        full,
        target_T=target_t,
        target_K=target_k,
    )
    return sample, target_t, target_k


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--surface", required=True)
    p.add_argument(
        "--output-dir",
        default="outputs/sampling_gaussian_test",
    )
    p.add_argument("--n-t", type=int, default=8)
    p.add_argument("--n-k", type=int, default=8)

    p.add_argument(
        "--lambda-t",
        type=float,
        default=1.0,
        help="Exponential short-end concentration. Default: 1.0",
    )
    p.add_argument(
        "--gaussian-z",
        type=float,
        default=2.0,
        help=(
            "Strength of Gaussian center concentration. "
            "Larger values create a larger gap toward Tmin/Tmax. Default: 2.0"
        ),
    )

    args = p.parse_args()

    full = load_surface(args.surface)
    required_n = args.n_t * args.n_k

    if len(full) <= required_n:
        raise ValueError(
            f"Need more than {required_n} observations for a non-empty "
            f"holdout; surface contains {len(full)}."
        )

    strategies = {
        "UU": ("uniform", "uniform"),
        "CU": ("chebyshev", "uniform"),
        "UC": ("uniform", "chebyshev"),
        "CC": ("chebyshev", "chebyshev"),
        "EU": ("exponential", "uniform"),
        "EC": ("exponential", "chebyshev"),
        "GU": ("gaussian", "uniform"),
        "GC": ("gaussian", "chebyshev"),
    }

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    full.to_csv(out / "full_eligible_surface.csv", index=False)

    rows = []
    details = {}

    for code, (t_scheme, k_scheme) in strategies.items():
        sample, target_t, target_k = sample_strategy(
            full=full,
            t_scheme=t_scheme,
            k_scheme=k_scheme,
            n_t=args.n_t,
            n_k=args.n_k,
            lam=args.lambda_t,
            gaussian_z=args.gaussian_z,
        )

        diag = Sampling.interpolation_diagnostics(full, sample)

        sample.to_csv(
            out / f"sample_{code}_{required_n}.csv",
            index=False,
        )

        pd.DataFrame({"T_target": target_t}).to_csv(
            out / f"target_T_{code}.csv",
            index=False,
        )
        pd.DataFrame({"K_target": target_k}).to_csv(
            out / f"target_K_{code}.csv",
            index=False,
        )

        rows.append(
            {
                "strategy": code,
                "T_sampling": t_scheme,
                "K_sampling": k_scheme,
                "lambda_T": (
                    args.lambda_t if t_scheme == "exponential" else np.nan
                ),
                "gaussian_z": (
                    args.gaussian_z if t_scheme == "gaussian" else np.nan
                ),
                "n_full": len(full),
                "n_sample": len(sample),
                "holdout_n": diag["holdout"]["n"],
                "holdout_mae_bps_iv": (
                    10000.0 * diag["holdout"]["mae"]
                ),
                "holdout_rmse_bps_iv": (
                    10000.0 * diag["holdout"]["rmse"]
                ),
                "holdout_linf_bps_iv": (
                    diag["holdout"]["linf_bps_iv"]
                ),
            }
        )

        details[code] = diag

    comparison = pd.DataFrame(rows).sort_values(
        [
            "holdout_linf_bps_iv",
            "holdout_rmse_bps_iv",
            "holdout_mae_bps_iv",
        ],
        ascending=True,
    ).reset_index(drop=True)

    comparison["rank"] = np.arange(1, len(comparison) + 1)

    comparison.to_csv(
        out / "sampling_comparison_gaussian.csv",
        index=False,
    )

    winner = comparison.iloc[0].to_dict()

    payload = {
        "selection_rule": (
            "minimize holdout infinity norm; "
            "tie-break by holdout RMSE and MAE"
        ),
        "lambda_T": args.lambda_t,
        "gaussian_z": args.gaussian_z,
        "winner": winner,
        "diagnostics": details,
    }

    (out / "sampling_winner_gaussian.json").write_text(
        json.dumps(payload, indent=2, default=float),
        encoding="utf-8",
    )

    t_min = float(full["T"].min())
    t_max = float(full["T"].max())

    exp_t = exponential_nodes(
        args.n_t,
        t_min,
        t_max,
        args.lambda_t,
    )
    gauss_t = gaussian_center_nodes(
        args.n_t,
        t_min,
        t_max,
        args.gaussian_z,
    )

    print("=" * 116)
    print(f"FULL ELIGIBLE SURFACE : {len(full)} observations")
    print(f"SAMPLE SIZE           : {required_n} = {args.n_t} x {args.n_k}")
    print(f"T RANGE               : {t_min:.6f} -> {t_max:.6f} years")
    print(f"EXP lambda_T          : {args.lambda_t:g}")
    print(
        "EXP T NODES           : "
        + ", ".join(f"{x:.4f}" for x in exp_t)
    )
    print(f"GAUSSIAN z_clip       : {args.gaussian_z:g}")
    print(
        "GAUSSIAN T NODES      : "
        + ", ".join(f"{x:.4f}" for x in gauss_t)
    )
    print("=" * 116)

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

    print("=" * 116)
    print(
        f"[OK] Winner: {comparison.iloc[0]['strategy']} "
        f"(holdout L_inf = "
        f"{comparison.iloc[0]['holdout_linf_bps_iv']:.3f} IV bp)"
    )
    print(f"[OK] Results: {out}")
    print()
    print("Legend:")
    print("  GU = Gaussian-centered maturity + Uniform strike")
    print("  GC = Gaussian-centered maturity + Chebyshev strike")
    print("  EU = Exponential short-end maturity + Uniform strike")
    print("  EC = Exponential short-end maturity + Chebyshev strike")
    print()
    print(
        "Use GU/GC as a center-weighted benchmark. "
        "Do not assume in advance that they must perform worse."
    )


if __name__ == "__main__":
    main()
