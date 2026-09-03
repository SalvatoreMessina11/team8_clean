"""Unified node-selection and interpolation audit for Team 8 GLD option surfaces.

This file consolidates the functionality previously split across:
    - Sampling.py
    - compare_sampling.py
    - compare_sampling_exponential.py
    - compare_sampling_gaussian.py

It is standalone: no project-local sampling module is required.

Default strategies
------------------
UU = Uniform T, Uniform K
CU = Chebyshev T, Uniform K
UC = Uniform T, Chebyshev K
CC = Chebyshev T, Chebyshev K
EU = Exponential T (short-end weighted), Uniform K
EC = Exponential T (short-end weighted), Chebyshev K
GU = Gaussian-centered T, Uniform K
GC = Gaussian-centered T, Chebyshev K

Methodology
-----------
1. Build theoretical target nodes on T and K.
2. Map each target (T, K) to the nearest still-unused ACTUAL market observation.
3. Reconstruct the IV surface with a normalized thin-plate RBF.
4. Evaluate errors on observations not selected as nodes (holdout).
5. Rank by holdout L_inf, then RMSE, then MAE.

No synthetic option observation is ever used as a calibration node.

Recommended fixed robustness parameters:
    lambda_T = 1.0
    gaussian_z = 2.0

Example
-------
python src/compare_sampling_all.py ^
  --surface data/processed/full_surfaces/GLD_2026-09-02_eligible_adaptive_surface.csv ^
  --output-dir outputs/sampling/2026-09-02 ^
  --n-t 8 ^
  --n-k 8 ^
  --lambda-t 1 ^
  --gaussian-z 2
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import Rbf
from scipy.stats import norm


DEFAULT_STRATEGIES = {
    "UU": ("uniform", "uniform"),
    "CU": ("chebyshev", "uniform"),
    "UC": ("uniform", "chebyshev"),
    "CC": ("chebyshev", "chebyshev"),
    "EU": ("exponential", "uniform"),
    "EC": ("exponential", "chebyshev"),
    "GU": ("gaussian", "uniform"),
    "GC": ("gaussian", "chebyshev"),
}


def load_surface(path: str | Path) -> pd.DataFrame:
    """Load and clean an eligible IV surface."""
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

    # A surface point is uniquely identified by maturity and strike.
    frame = frame.drop_duplicates(["T", "K"], keep="last")
    frame = frame.sort_values(["T", "K"]).reset_index(drop=True)
    frame["_row_id"] = np.arange(len(frame), dtype=int)

    return frame


def _validate_axis(a: float, b: float) -> None:
    if not np.isfinite(a) or not np.isfinite(b) or b <= a:
        raise ValueError("Sampling axis must have a non-zero finite range.")


def uniform_nodes(n: int, a: float, b: float) -> np.ndarray:
    n = int(n)
    _validate_axis(a, b)

    if n < 2:
        raise ValueError("n must be at least 2.")

    return np.linspace(a, b, n)


def chebyshev_nodes(n: int, a: float, b: float) -> np.ndarray:
    """Chebyshev roots mapped from [-1, 1] to [a, b].

    These nodes are symmetric and denser near both endpoints.
    """
    n = int(n)
    _validate_axis(a, b)

    if n < 2:
        raise ValueError("n must be at least 2.")

    k = np.arange(1, n + 1)
    roots = np.sort(np.cos((2 * k - 1) * np.pi / (2 * n)))

    return 0.5 * (a + b) + 0.5 * (b - a) * roots


def exponential_nodes(
    n: int,
    a: float,
    b: float,
    lam: float = 1.0,
) -> np.ndarray:
    """Asymmetric grid concentrated toward a = Tmin.

    T(u) = a + (b-a) * [exp(lambda*u)-1] / [exp(lambda)-1].

    lambda > 0:
        more target maturities near Tmin.

    lambda -> 0:
        approaches a uniform grid.
    """
    n = int(n)
    lam = float(lam)

    _validate_axis(a, b)

    if n < 2:
        raise ValueError("n must be at least 2.")
    if not np.isfinite(lam) or lam <= 0:
        raise ValueError("--lambda-t must be strictly positive.")

    u = np.linspace(0.0, 1.0, n)
    scaled = np.expm1(lam * u) / np.expm1(lam)

    return a + (b - a) * scaled


def gaussian_center_nodes(
    n: int,
    a: float,
    b: float,
    z_clip: float = 2.0,
) -> np.ndarray:
    """Symmetric grid concentrated around the middle of [a, b].

    Equally spaced probabilities over a standard normal truncated to
    [-z_clip, z_clip] are mapped through the inverse normal CDF and then
    linearly rescaled to [a, b].

    This implementation, matching the tested version, keeps a and b as the
    first and last theoretical target nodes.
    """
    n = int(n)
    z_clip = float(z_clip)

    _validate_axis(a, b)

    if n < 2:
        raise ValueError("n must be at least 2.")
    if not np.isfinite(z_clip) or z_clip <= 0:
        raise ValueError("--gaussian-z must be strictly positive.")

    p_lo = norm.cdf(-z_clip)
    p_hi = norm.cdf(z_clip)
    probs = np.linspace(p_lo, p_hi, n)

    z = norm.ppf(probs)
    z = np.clip(z, -z_clip, z_clip)

    scaled = (z + z_clip) / (2.0 * z_clip)

    return a + (b - a) * scaled


def axis_nodes(
    series: pd.Series,
    n: int,
    scheme: str,
    *,
    lambda_t: float = 1.0,
    gaussian_z: float = 2.0,
) -> np.ndarray:
    """Construct theoretical nodes for one sampling axis."""
    values = pd.to_numeric(series, errors="coerce").dropna()

    if values.empty:
        raise ValueError("Cannot build nodes on an empty axis.")

    a = float(values.min())
    b = float(values.max())

    scheme = scheme.lower()

    if scheme == "uniform":
        return uniform_nodes(n, a, b)

    if scheme == "chebyshev":
        return chebyshev_nodes(n, a, b)

    if scheme == "exponential":
        return exponential_nodes(n, a, b, lam=lambda_t)

    if scheme == "gaussian":
        return gaussian_center_nodes(n, a, b, z_clip=gaussian_z)

    raise ValueError(f"Unknown sampling scheme: {scheme}")


def get_nearest_market_points(
    df: pd.DataFrame,
    target_T: np.ndarray,
    target_K: np.ndarray,
) -> pd.DataFrame:
    """Map theoretical target nodes to unique actual market observations.

    Distance is Euclidean in normalized T-K coordinates. Once a market point
    has been selected it cannot be reused by another theoretical target.
    """
    frame = pd.DataFrame(df).copy().reset_index(drop=True)

    required_n = len(target_T) * len(target_K)

    if len(frame) < required_n:
        raise ValueError(
            f"Need at least {required_n} eligible market observations; "
            f"only {len(frame)} are available."
        )

    if "_row_id" not in frame.columns:
        frame["_row_id"] = np.arange(len(frame), dtype=int)

    t_min = float(frame["T"].min())
    t_max = float(frame["T"].max())
    k_min = float(frame["K"].min())
    k_max = float(frame["K"].max())

    t_scale = max(t_max - t_min, 1e-12)
    k_scale = max(k_max - k_min, 1e-12)

    selected_indices: set[int] = set()
    selected: list[int] = []

    for t_target in np.asarray(target_T, dtype=float):
        for k_target in np.asarray(target_K, dtype=float):
            distance = (
                ((frame["T"] - t_target) / t_scale) ** 2
                + ((frame["K"] - k_target) / k_scale) ** 2
            )

            for idx in distance.sort_values(kind="stable").index:
                idx_int = int(idx)

                if idx_int not in selected_indices:
                    selected_indices.add(idx_int)
                    selected.append(idx_int)
                    break

    sample = frame.loc[selected].copy()

    if len(sample) != required_n:
        raise RuntimeError(
            f"Expected {required_n} unique market nodes, obtained {len(sample)}."
        )

    return sample.sort_values(["T", "K"]).reset_index(drop=True)


def sample_strategy(
    full: pd.DataFrame,
    t_scheme: str,
    k_scheme: str,
    *,
    n_t: int,
    n_k: int,
    lambda_t: float,
    gaussian_z: float,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Build one structured sample and return sample plus theoretical nodes."""
    target_t = axis_nodes(
        full["T"],
        n_t,
        t_scheme,
        lambda_t=lambda_t,
        gaussian_z=gaussian_z,
    )

    target_k = axis_nodes(
        full["K"],
        n_k,
        k_scheme,
        lambda_t=lambda_t,
        gaussian_z=gaussian_z,
    )

    sample = get_nearest_market_points(
        full,
        target_T=target_t,
        target_K=target_k,
    )

    return sample, target_t, target_k


def _metrics(err: np.ndarray) -> dict[str, float | int]:
    err = np.asarray(err, dtype=float)

    if err.size == 0:
        return {
            "n": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "linf": np.nan,
            "mae_vol_points": np.nan,
            "rmse_vol_points": np.nan,
            "linf_vol_points": np.nan,
            "mae_bps_iv": np.nan,
            "rmse_bps_iv": np.nan,
            "linf_bps_iv": np.nan,
        }

    absolute = np.abs(err)

    mae = float(np.mean(absolute))
    rmse = float(np.sqrt(np.mean(err**2)))
    linf = float(np.max(absolute))

    return {
        "n": int(err.size),
        "mae": mae,
        "rmse": rmse,
        "linf": linf,
        "mae_vol_points": 100.0 * mae,
        "rmse_vol_points": 100.0 * rmse,
        "linf_vol_points": 100.0 * linf,
        "mae_bps_iv": 10000.0 * mae,
        "rmse_bps_iv": 10000.0 * rmse,
        "linf_bps_iv": 10000.0 * linf,
    }


def interpolation_diagnostics(
    df_full: pd.DataFrame,
    df_sampled: pd.DataFrame,
) -> dict[str, dict[str, float | int]]:
    """Reconstruct IV through normalized thin-plate RBF and evaluate errors."""
    full = pd.DataFrame(df_full).copy().reset_index(drop=True)
    sample = pd.DataFrame(df_sampled).copy().reset_index(drop=True)

    if "_row_id" not in full.columns:
        full["_row_id"] = np.arange(len(full), dtype=int)

    if "_row_id" not in sample.columns:
        raise ValueError("Sample must preserve _row_id from the full surface.")

    t_min = float(full["T"].min())
    t_max = float(full["T"].max())
    k_min = float(full["K"].min())
    k_max = float(full["K"].max())

    t_scale = max(t_max - t_min, 1e-12)
    k_scale = max(k_max - k_min, 1e-12)

    t_s = (sample["T"].to_numpy(float) - t_min) / t_scale
    k_s = (sample["K"].to_numpy(float) - k_min) / k_scale
    iv_s = sample["implied_vol"].to_numpy(float)

    rbf = Rbf(
        t_s,
        k_s,
        iv_s,
        function="thin_plate",
    )

    t_f = (full["T"].to_numpy(float) - t_min) / t_scale
    k_f = (full["K"].to_numpy(float) - k_min) / k_scale

    truth = full["implied_vol"].to_numpy(float)
    prediction = np.asarray(rbf(t_f, k_f), dtype=float)

    error = prediction - truth

    sampled_ids = set(sample["_row_id"].astype(int))

    holdout_mask = (
        ~full["_row_id"].astype(int).isin(sampled_ids).to_numpy()
    )

    return {
        "all": _metrics(error),
        "holdout": _metrics(error[holdout_mask]),
    }


def parse_strategies(raw: str) -> dict[str, tuple[str, str]]:
    """Parse e.g. 'UU,CC,EU,GU'; default 'all'."""
    raw = raw.strip()

    if raw.lower() == "all":
        return DEFAULT_STRATEGIES.copy()

    requested = [x.strip().upper() for x in raw.split(",") if x.strip()]

    if not requested:
        raise ValueError("No strategies requested.")

    unknown = [x for x in requested if x not in DEFAULT_STRATEGIES]

    if unknown:
        raise ValueError(
            f"Unknown strategies: {unknown}. "
            f"Allowed: {sorted(DEFAULT_STRATEGIES)}"
        )

    # Keep the user's requested order.
    return {code: DEFAULT_STRATEGIES[code] for code in requested}


def run_comparison(
    full: pd.DataFrame,
    *,
    strategies: dict[str, tuple[str, str]],
    n_t: int,
    n_k: int,
    lambda_t: float,
    gaussian_z: float,
    output_dir: str | Path,
    official_min_dte: int = 75,
    official_strategy: str = "CC",
) -> pd.DataFrame:
    required_n = int(n_t) * int(n_k)

    # Strictly greater than required_n so holdout is non-empty.
    if len(full) <= required_n:
        raise ValueError(
            f"Need more than {required_n} eligible observations for a "
            f"non-empty holdout; surface contains {len(full)}."
        )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    full.to_csv(
        out / "full_eligible_surface.csv",
        index=False,
    )

    rows = []
    details = {}

    for code, (t_scheme, k_scheme) in strategies.items():
        sample, target_t, target_k = sample_strategy(
            full,
            t_scheme,
            k_scheme,
            n_t=n_t,
            n_k=n_k,
            lambda_t=lambda_t,
            gaussian_z=gaussian_z,
        )

        diag = interpolation_diagnostics(full, sample)

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
                    lambda_t if t_scheme == "exponential" else np.nan
                ),
                "gaussian_z": (
                    gaussian_z if t_scheme == "gaussian" else np.nan
                ),
                "n_full": len(full),
                "n_sample": len(sample),
                "all_mae_bps_iv": diag["all"]["mae_bps_iv"],
                "all_rmse_bps_iv": diag["all"]["rmse_bps_iv"],
                "all_linf_bps_iv": diag["all"]["linf_bps_iv"],
                "holdout_n": diag["holdout"]["n"],
                "holdout_mae_bps_iv": diag["holdout"]["mae_bps_iv"],
                "holdout_rmse_bps_iv": diag["holdout"]["rmse_bps_iv"],
                "holdout_linf_bps_iv": diag["holdout"]["linf_bps_iv"],
            }
        )

        details[code] = {
            "T_sampling": t_scheme,
            "K_sampling": k_scheme,
            "target_T": [float(x) for x in target_t],
            "target_K": [float(x) for x in target_k],
            "diagnostics": diag,
        }

    comparison = pd.DataFrame(rows)

    comparison = comparison.sort_values(
        [
            "holdout_linf_bps_iv",
            "holdout_rmse_bps_iv",
            "holdout_mae_bps_iv",
        ],
        ascending=True,
    ).reset_index(drop=True)

    comparison["rank"] = np.arange(
        1,
        len(comparison) + 1,
        dtype=int,
    )

    # Put rank first for easier reading.
    ordered = ["rank"] + [
        col for col in comparison.columns if col != "rank"
    ]
    comparison = comparison[ordered]

    comparison.to_csv(
        out / "sampling_comparison.csv",
        index=False,
    )

    payload = {
        "purpose": "sampling-distribution robustness diagnostics",
        "diagnostic_ranking_rule": (
            "minimize holdout infinity norm; "
            "tie-break by holdout RMSE and holdout MAE"
        ),
        "official_strategy": str(official_strategy).upper(),
        "official_strategy_rule": (
            "CC is fixed ex ante for calibration and is not changed by the "
            "daily diagnostic ranking"
        ),
        "official_min_dte": int(official_min_dte),
        "n_T": int(n_t),
        "n_K": int(n_k),
        "lambda_T": float(lambda_t),
        "gaussian_z": float(gaussian_z),
        "strategies": list(strategies.keys()),
        "diagnostic_rank_1": comparison.iloc[0].to_dict(),
        "details": details,
    }

    encoded = json.dumps(payload, indent=2, default=float)
    (out / "sampling_diagnostics.json").write_text(
        encoded,
        encoding="utf-8",
    )
    # Backward-compatible filename retained for existing notebooks/scripts.
    (out / "sampling_winner.json").write_text(
        encoded,
        encoding="utf-8",
    )

    return comparison


def print_header(
    full: pd.DataFrame,
    *,
    n_t: int,
    n_k: int,
    lambda_t: float,
    gaussian_z: float,
) -> None:
    t_min = float(full["T"].min())
    t_max = float(full["T"].max())
    k_min = float(full["K"].min())
    k_max = float(full["K"].max())

    print("=" * 120)
    print(f"FULL ELIGIBLE SURFACE : {len(full)} observations")
    print(f"SAMPLE SIZE           : {n_t * n_k} = {n_t} x {n_k}")
    print(f"T RANGE               : {t_min:.6f} -> {t_max:.6f} years")
    print(f"K RANGE               : {k_min:.6f} -> {k_max:.6f}")
    print(f"EXP lambda_T          : {lambda_t:g}")
    print(f"GAUSSIAN z_clip       : {gaussian_z:g}")

    print(
        "UNIFORM T NODES       : "
        + ", ".join(
            f"{x:.4f}"
            for x in uniform_nodes(n_t, t_min, t_max)
        )
    )

    print(
        "CHEBYSHEV T NODES     : "
        + ", ".join(
            f"{x:.4f}"
            for x in chebyshev_nodes(n_t, t_min, t_max)
        )
    )

    print(
        "EXPONENTIAL T NODES   : "
        + ", ".join(
            f"{x:.4f}"
            for x in exponential_nodes(
                n_t,
                t_min,
                t_max,
                lam=lambda_t,
            )
        )
    )

    print(
        "GAUSSIAN T NODES      : "
        + ", ".join(
            f"{x:.4f}"
            for x in gaussian_center_nodes(
                n_t,
                t_min,
                t_max,
                z_clip=gaussian_z,
            )
        )
    )

    print("=" * 120)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Unified structured-node sampling audit for GLD IV surfaces."
        )
    )

    parser.add_argument(
        "--surface",
        required=True,
        help="Eligible market-surface CSV or Parquet file.",
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Output directory. If omitted, GLD_YYYY-MM-DD input filenames are "
            "written to outputs/sampling/YYYY-MM-DD."
        ),
    )

    parser.add_argument(
        "--n-t",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--n-k",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--min-dte",
        type=int,
        default=75,
        help=(
            "Official calibration-domain floor in calendar days. "
            "Observations below this DTE are excluded before node construction. "
            "Default: 75."
        ),
    )

    parser.add_argument(
        "--lambda-t",
        type=float,
        default=1.0,
        help=(
            "Exponential maturity concentration parameter. "
            "Default 1.0."
        ),
    )

    parser.add_argument(
        "--gaussian-z",
        type=float,
        default=2.0,
        help=(
            "Gaussian center-concentration parameter. "
            "Default 2.0."
        ),
    )

    parser.add_argument(
        "--strategies",
        default="all",
        help=(
            "Comma-separated subset, e.g. UU,CC,EU,GU. "
            "Default: all."
        ),
    )

    args = parser.parse_args()

    if args.n_t < 2 or args.n_k < 2:
        raise ValueError("--n-t and --n-k must both be at least 2.")

    strategies = parse_strategies(args.strategies)

    if args.output_dir is None:
        match = re.search(r"GLD_(\d{4}-\d{2}-\d{2})_", Path(args.surface).name)
        if match is None:
            raise ValueError(
                "Could not infer date from --surface filename. Pass --output-dir "
                "explicitly, e.g. outputs/sampling/2026-09-02."
            )
        output_dir = str(Path("outputs") / "sampling" / match.group(1))
    else:
        output_dir = args.output_dir

    full = load_surface(args.surface)

    if args.min_dte < 1:
        raise ValueError("--min-dte must be at least 1 day.")

    if "dte" in full.columns:
        dte = pd.to_numeric(full["dte"], errors="coerce")
    else:
        dte = 365.25 * pd.to_numeric(full["T"], errors="coerce")

    before_dte = len(full)
    full = full.loc[dte.ge(float(args.min_dte))].copy()
    full = full.sort_values(["T", "K"]).reset_index(drop=True)
    full["_row_id"] = np.arange(len(full), dtype=int)

    required_n = int(args.n_t) * int(args.n_k)
    if len(full) <= required_n:
        raise ValueError(
            f"After the DTE >= {args.min_dte} filter only {len(full)} market "
            f"observations remain; more than {required_n} are required so that "
            "the holdout set is non-empty."
        )

    print(
        f"[DOMAIN] DTE >= {args.min_dte} days: "
        f"{before_dte} -> {len(full)} eligible observations"
    )

    print_header(
        full,
        n_t=args.n_t,
        n_k=args.n_k,
        lambda_t=args.lambda_t,
        gaussian_z=args.gaussian_z,
    )

    comparison = run_comparison(
        full,
        strategies=strategies,
        n_t=args.n_t,
        n_k=args.n_k,
        lambda_t=args.lambda_t,
        gaussian_z=args.gaussian_z,
        output_dir=output_dir,
        official_min_dte=args.min_dte,
        official_strategy="CC",
    )

    display_cols = [
        "rank",
        "strategy",
        "T_sampling",
        "K_sampling",
        "holdout_mae_bps_iv",
        "holdout_rmse_bps_iv",
        "holdout_linf_bps_iv",
    ]

    print(
        comparison[display_cols].to_string(index=False)
    )

    print("=" * 120)

    winner = comparison.iloc[0]

    print(
        f"[DIAGNOSTIC] Rank #1: {winner['strategy']} "
        f"(holdout L_inf = "
        f"{winner['holdout_linf_bps_iv']:.3f} IV bp)"
    )
    print("[OFFICIAL] Calibration strategy remains fixed at CC.")
    print(f"[OFFICIAL] Minimum DTE domain: {args.min_dte} days.")

    print(f"[OK] Results: {output_dir}")
    print()
    print("Legend:")
    print("  UU = Uniform T + Uniform K")
    print("  CU = Chebyshev T + Uniform K")
    print("  UC = Uniform T + Chebyshev K")
    print("  CC = Chebyshev T + Chebyshev K")
    print("  EU = Exponential T + Uniform K")
    print("  EC = Exponential T + Chebyshev K")
    print("  GU = Gaussian-centered T + Uniform K")
    print("  GC = Gaussian-centered T + Chebyshev K")
    print()
    print(
        "Diagnostic ranking: holdout L_inf; "
        "tie-breakers: RMSE, then MAE. Official calibration remains CC."
    )


if __name__ == "__main__":
    main()
