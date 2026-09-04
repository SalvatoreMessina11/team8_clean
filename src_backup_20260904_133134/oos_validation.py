"""Rolling historical / OOS validation for the Team 8 GLD project.

This file is intentionally ready BEFORE the historical panel is complete.
Once the missing IBKR historical data have been reconstructed, it can run the
entire chronological validation without changing the folder architecture.

Official Team 8 rules implemented here
--------------------------------------
1. No calibration uses observations dated after the forecast origin.
2. Official maturity domain: DTE >= 75 calendar days.
3. DENSE-ONLY OOS SAMPLE: a date is used only if, after DTE>=75 filtering,
   it has at least 64 unique actual observations and at least 3 expiries.
4. Every admitted origin is calibrated on the fixed 8x8
   Chebyshev-Chebyshev (CC) sample of 64 ACTUAL observations. Sparse dates are
   excluded from this OOS exercise; no interpolation creates calibration data.
5. Four models are always compared on common target support:
      Black-Scholes, Heston, Bates-Poisson, Full Exact Bates-Hawkes.
6. Heston/Bates variance state and Bates-Hawkes variance/intensity state are
   projected from origin t to the next available target date.
7. Scoring is a CONDITIONAL IV-surface validation:
      - model parameters/states come only from the origin;
      - the realized target spot and target no-look-ahead rate curve are used as
        conditioning variables when evaluating the target option cross-section;
      - target option prices / IVs are used only for scoring.
   This isolates temporal stability of the calibrated risk-neutral surface from
   the separate problem of forecasting the underlying asset price.
8. Persistence benchmark: previous observed IV surface, linearly interpolated
   in (T, log-moneyness) only INSIDE its convex hull. No extrapolation.
9. Mean benchmark: origin-date cross-sectional mean implied volatility.
10. Date-level losses are retained so dense dates do not mechanically dominate.

Expected inputs when the panel is ready
---------------------------------------
Sparse historical surfaces:
    data/processed/sparse_historical_surfaces/
        GLD_YYYY-MM-DD_eligible_historical_surface.csv

Dense/recent surfaces:
    data/processed/full_surfaces/
        GLD_YYYY-MM-DD_eligible_adaptive_surface.csv
        GLD_YYYY-MM-DD_eligible_full_surface.csv  (legacy fallback)

For duplicate dates the script loads every candidate file, applies the
official DTE filter, and keeps the file with the largest number of valid unique
ACTUAL observations. This prevents an empty/small adaptive file from hiding a
richer historical/full file.

Outputs
-------
outputs/oos/
    calibration_sets/YYYY-MM-DD/calibration_surface.csv
    calibrations/ROLLING/YYYY-MM-DD/*.json
    rolling/
        option_forecasts.csv
        date_metrics.csv
        model_summary.csv
        origin_target_coverage.csv
        run_manifest.json

img/diagnostics_ibkr/
    online_welch_goyal_cumulative.png
    oos_table_rows.tex
    oos_article_numbers.tex

Optional fixed-parameter robustness:
    add --fixed-origin YYYY-MM-DD
which also creates:
    outputs/oos/fixed_<YYYY-MM-DD>/
    img/diagnostics_ibkr/oos_welch_goyal_cumulative.png

Examples
--------
Inspect what is available, without calibrating:
    python src/oos_validation.py --dry-run

Calibrate all dense rolling origins only:
    python src/oos_validation.py --profile full --calibrate-only

Final rolling run (reuses successful calibrations):
    python src/oos_validation.py --profile full

Resume is automatic because calibrate_surface.py already skips successful JSONs.

Force all origin calibrations again:
    python src/oos_validation.py --profile full --force-calibration

Optional frozen-parameter robustness:
    python src/oos_validation.py --profile full --fixed-origin 2026-08-17
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import QhullError

from Sampling import Sampling
from BnS import BnS
from Heston import Heston
from Bates import Bates
from BatesHawkesExact import BatesHawkesExact


MODEL_KEYS = ("bs", "heston", "bates", "hawkes")
MODEL_LABELS = {
    "bs": "Black-Scholes",
    "heston": "Heston",
    "bates": "Bates",
    "hawkes": "Full Bates-Hawkes",
}
MODEL_FILES = {
    "bs": "black_scholes.json",
    "heston": "heston.json",
    "bates": "bates.json",
    "hawkes": "full_bates_hawkes.json",
}

SURFACE_RE = re.compile(
    r"GLD_(\d{4}-\d{2}-\d{2})_eligible_"
    r"(historical|adaptive|full|bin_balanced|all_real)_surface\.csv$"
)
SURFACE_PRIORITY = {
    "full": 1,
    "historical": 2,
    "adaptive": 3,
    "bin_balanced": 4,
    "all_real": 5,
}


@dataclass(frozen=True)
class SurfaceRef:
    date: pd.Timestamp
    kind: str
    path: Path


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, default=_json_default),
        encoding="utf-8",
    )


def discover_surfaces(
    sparse_dir: Path,
    dense_dir: Path,
    start: str | None = None,
    end: str | None = None,
    min_dte: int = 75,
) -> list[SurfaceRef]:
    """Discover surfaces and choose the richest valid REAL file per date.

    Every candidate is loaded with the same official DTE filter used by the
    OOS engine. The representative is selected by valid unique row count,
    then expiry count; filename kind is only a deterministic final tie-break.
    """
    candidates: dict[pd.Timestamp, list[SurfaceRef]] = {}

    for directory in (Path(sparse_dir), Path(dense_dir)):
        if not directory.exists():
            continue

        for path in directory.glob("GLD_*_eligible_*_surface.csv"):
            match = SURFACE_RE.match(path.name)
            if not match:
                continue

            date = pd.Timestamp(match.group(1)).normalize()
            kind = match.group(2)

            if start is not None and date < pd.Timestamp(start).normalize():
                continue
            if end is not None and date > pd.Timestamp(end).normalize():
                continue

            candidates.setdefault(date, []).append(
                SurfaceRef(date=date, kind=kind, path=path)
            )

    selected: list[SurfaceRef] = []

    for date in sorted(candidates):
        scored: list[tuple[int, int, int, SurfaceRef]] = []
        for ref in candidates[date]:
            try:
                frame = load_surface(ref, min_dte=int(min_dte))
                st = surface_stats(frame)
            except Exception:
                continue

            scored.append(
                (
                    int(st["rows"]),
                    int(st["expiries"]),
                    int(SURFACE_PRIORITY.get(ref.kind, 0)),
                    ref,
                )
            )

        if scored:
            scored.sort(key=lambda item: item[:3], reverse=True)
            selected.append(scored[0][3])

    return selected

def load_surface(ref: SurfaceRef, min_dte: int = 75) -> pd.DataFrame:
    df = pd.read_csv(ref.path)

    required = {
        "K",
        "T",
        "rate",
        "price",
        "vega",
        "implied_vol",
        "spot",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"{ref.path.name} missing required columns: {sorted(missing)}"
        )

    numeric = [
        "K",
        "T",
        "rate",
        "price",
        "vega",
        "implied_vol",
        "spot",
    ]
    for optional in [
        "dte",
        "moneyness",
        "nss_beta0",
        "nss_beta1",
        "nss_beta2",
        "nss_beta3",
        "nss_tau1",
        "nss_tau2",
        "nss_rmse_bps",
    ]:
        if optional in df.columns:
            numeric.append(optional)

    for c in numeric:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(
        subset=[
            "K",
            "T",
            "rate",
            "price",
            "vega",
            "implied_vol",
            "spot",
        ]
    ).copy()

    if "dte" in df.columns:
        dte = pd.to_numeric(df["dte"], errors="coerce")
    else:
        dte = 365.25 * df["T"]

    df = df.loc[
        df["K"].gt(0.0)
        & df["T"].gt(0.0)
        & df["price"].gt(0.0)
        & df["vega"].gt(0.0)
        & df["implied_vol"].gt(0.0)
        & df["spot"].gt(0.0)
        & np.isfinite(df["rate"])
        & dte.ge(float(min_dte))
    ].copy()

    df["dte"] = dte.loc[df.index].astype(float)
    if "moneyness" not in df.columns:
        df["moneyness"] = df["K"] / df["spot"]

    if "date" not in df.columns:
        df["date"] = ref.date
    else:
        parsed = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
        df["date"] = parsed.fillna(ref.date)

    df = (
        df.sort_values(["T", "K"])
        .drop_duplicates(["T", "K"], keep="last")
        .reset_index(drop=True)
    )

    if df.empty:
        raise ValueError(f"No valid DTE>={min_dte} observations remain.")

    return df


def surface_stats(frame: pd.DataFrame) -> dict[str, Any]:
    expiries = (
        int(pd.Series(frame["expiry"]).nunique())
        if "expiry" in frame.columns
        else int(pd.Series(frame["T"]).round(8).nunique())
    )
    return {
        "rows": int(len(frame)),
        "expiries": expiries,
        "strikes": int(frame["K"].nunique()),
        "spot": float(frame["spot"].median()),
        "min_T": float(frame["T"].min()),
        "max_T": float(frame["T"].max()),
        "min_dte": float(frame["dte"].min()),
        "max_dte": float(frame["dte"].max()),
    }


def build_origin_calibration_set(
    full: pd.DataFrame,
    n_t: int,
    n_k: int,
) -> tuple[pd.DataFrame, str]:
    required_n = int(n_t) * int(n_k)

    if len(full) < required_n:
        raise ValueError(
            f"Dense-only OOS requires at least {required_n} actual points; "
            f"only {len(full)} are available."
        )

    sample = Sampling.sample_hybrid(
        full,
        t_scheme="chebyshev",
        k_scheme="chebyshev",
        n_T=int(n_t),
        n_K=int(n_k),
    )
    mode = f"CC_{required_n}"

    sample = (
        sample.sort_values(["T", "K"])
        .drop_duplicates(["T", "K"], keep="last")
        .reset_index(drop=True)
    )
    if len(sample) != required_n:
        raise RuntimeError(
            f"Expected exactly {required_n} unique CC nodes, got {len(sample)}."
        )
    return sample, mode

def ensure_origin_set(
    ref: SurfaceRef,
    full: pd.DataFrame,
    output_root: Path,
    n_t: int,
    n_k: int,
) -> tuple[Path, str, pd.DataFrame]:
    sample, mode = build_origin_calibration_set(full, n_t=n_t, n_k=n_k)

    date_dir = Path(output_root) / ref.date.strftime("%Y-%m-%d")
    date_dir.mkdir(parents=True, exist_ok=True)
    path = date_dir / "calibration_surface.csv"
    sample.to_csv(path, index=False)

    meta = {
        "date": ref.date.strftime("%Y-%m-%d"),
        "source_kind": ref.kind,
        "source_path": str(ref.path),
        "source_rows": int(len(full)),
        "calibration_rows": int(len(sample)),
        "calibration_mode": mode,
        "n_t": int(n_t),
        "n_k": int(n_k),
    }
    json_write(date_dir / "manifest.json", meta)
    return path, mode, sample


def calibration_dir(
    calibration_root: Path,
    strategy: str,
    date: pd.Timestamp,
) -> Path:
    return (
        Path(calibration_root)
        / strategy
        / date.strftime("%Y-%m-%d")
    )


def calibration_successful(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return bool(payload.get("success", False))


def all_calibrations_successful(date_dir: Path) -> bool:
    return all(
        calibration_successful(date_dir / MODEL_FILES[m])
        for m in MODEL_KEYS
    )


def invoke_calibration(
    repo_root: Path,
    date: pd.Timestamp,
    surface_path: Path,
    calibration_root: Path,
    strategy: str,
    profile: str,
    seed: int,
    force: bool,
) -> None:
    cmd = [
        sys.executable,
        str(repo_root / "src" / "calibrate_surface.py"),
        "--date",
        date.strftime("%Y-%m-%d"),
        "--surface",
        str(surface_path),
        "--strategy",
        strategy,
        "--profile",
        profile,
        "--models",
        "bs,heston,bates,hawkes",
        "--seed",
        str(int(seed)),
        "--output-root",
        str(calibration_root),
    ]

    if force:
        cmd.append("--force")

    print("[CALIBRATE]", " ".join(cmd))
    subprocess.run(cmd, cwd=repo_root, check=True)


def read_model_payloads(date_dir: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}

    for model in MODEL_KEYS:
        path = date_dir / MODEL_FILES[model]
        if not calibration_successful(path):
            raise RuntimeError(f"Missing successful calibration: {path}")
        out[model] = json.loads(path.read_text(encoding="utf-8"))

    return out


def params(payload: dict[str, Any]) -> dict[str, float]:
    return {
        str(k): float(v)
        for k, v in (payload.get("parameters", {}) or {}).items()
    }


def project_model_state(
    model: str,
    p: dict[str, float],
    delta_years: float,
) -> dict[str, float]:
    """Project only state variables; structural parameters remain frozen."""
    out = dict(p)
    dt = max(float(delta_years), 0.0)

    if model in {"heston", "bates", "hawkes"}:
        kappa = float(out["kappa"])
        theta = float(out["theta"])
        v0 = float(out["v0"])
        out["v0"] = theta + (v0 - theta) * math.exp(-kappa * dt)
        out["v0"] = max(out["v0"], 1e-10)

    if model == "hawkes":
        beta = float(out["beta"])
        branching = float(out["branching_ratio"])
        alpha = float(out.get("alpha", branching * beta))
        lambda_bar = float(out["lambda_bar"])
        lambda0 = float(out["lambda0"])

        decay = beta - alpha
        if decay <= 0.0:
            raise ValueError(
                f"Invalid Hawkes projection: beta-alpha={decay}"
            )

        # E[lambda_t] solves
        # dE(lambda)/dt = beta*lambda_bar - (beta-alpha)E(lambda).
        lambda_inf = beta * lambda_bar / decay
        out["lambda0"] = (
            lambda_inf
            + (lambda0 - lambda_inf) * math.exp(-decay * dt)
        )
        out["lambda0"] = max(out["lambda0"], 1e-10)
        out["alpha"] = alpha

    return out


def _group_indices(frame: pd.DataFrame) -> list[np.ndarray]:
    work = frame[["T", "rate"]].copy()
    work["_Tkey"] = work["T"].round(10)
    work["_rkey"] = work["rate"].round(12)

    groups = []
    for _, idx in work.groupby(["_Tkey", "_rkey"]).groups.items():
        groups.append(np.asarray(list(idx), dtype=int))
    return groups


def model_prices(
    model: str,
    target: pd.DataFrame,
    p: dict[str, float],
    cos_n: int = 192,
    hawkes_cos_n: int = 128,
) -> np.ndarray:
    """Price target calls using target spot/rates and origin-projected state."""
    out = np.full(len(target), np.nan, dtype=float)
    spot = float(target["spot"].median())

    if model == "bs":
        sigma = float(p["sigma"])
        for i, row in target.iterrows():
            try:
                out[i] = BnS.bs_call_price(
                    spot,
                    float(row.K),
                    float(row.T),
                    float(row.rate),
                    sigma,
                    0.0,
                )
            except Exception:
                out[i] = np.nan
        return out

    for idx in _group_indices(target):
        rows = target.loc[idx]
        T = float(rows["T"].iloc[0])
        r = float(rows["rate"].iloc[0])
        K = rows["K"].to_numpy(dtype=float)

        try:
            if model == "heston":
                values = Heston.heston_prices_cos(
                    spot,
                    K,
                    T,
                    float(p["v0"]),
                    float(p["kappa"]),
                    float(p["theta"]),
                    float(p["sigma"]),
                    float(p["rho"]),
                    r,
                    0.0,
                    N=int(cos_n),
                )

            elif model == "bates":
                values = Bates.bates_prices_cos(
                    spot,
                    K,
                    T,
                    float(p["v0"]),
                    float(p["kappa"]),
                    float(p["theta"]),
                    float(p["sigma"]),
                    float(p["rho"]),
                    float(p["lambd"]),
                    float(p["mu_J"]),
                    float(p["sigma_J"]),
                    r,
                    0.0,
                    N=int(cos_n),
                )

            elif model == "hawkes":
                alpha = float(
                    p.get(
                        "alpha",
                        float(p["branching_ratio"]) * float(p["beta"]),
                    )
                )
                values = BatesHawkesExact.hawkes_price_cos(
                    spot,
                    K,
                    T,
                    float(p["v0"]),
                    float(p["kappa"]),
                    float(p["theta"]),
                    float(p.get("xi", p.get("sigma"))),
                    float(p["rho"]),
                    float(p["lambda0"]),
                    float(p["lambda_bar"]),
                    alpha,
                    float(p["beta"]),
                    float(p["mu_J"]),
                    float(p["sigma_J"]),
                    r,
                    0.0,
                    N=int(hawkes_cos_n),
                )
            else:
                raise ValueError(model)

            values = np.asarray(values, dtype=float)
            if values.shape != (len(idx),):
                values = values.reshape(-1)
            out[idx] = values

        except Exception:
            out[idx] = np.nan

    return out


def prices_to_iv(
    target: pd.DataFrame,
    model_price: np.ndarray,
) -> np.ndarray:
    spot = float(target["spot"].median())
    iv = np.full(len(target), np.nan, dtype=float)

    for i, row in target.iterrows():
        price = float(model_price[i])
        if not np.isfinite(price) or price <= 0.0:
            continue

        iv[i] = BnS.implied_vol_call(
            price,
            spot,
            float(row.K),
            float(row.T),
            float(row.rate),
            q=0.0,
        )

    return iv


def persistence_forecast(
    origin: pd.DataFrame,
    target: pd.DataFrame,
) -> np.ndarray:
    """Previous observed IV surface interpolated inside its convex hull only."""
    origin_spot = float(origin["spot"].median())
    target_spot = float(target["spot"].median())

    T0 = origin["T"].to_numpy(dtype=float)
    M0 = np.log(origin["K"].to_numpy(dtype=float) / origin_spot)
    Y0 = origin["implied_vol"].to_numpy(dtype=float)

    t_min = float(np.min(T0))
    t_scale = float(np.ptp(T0))
    m_min = float(np.min(M0))
    m_scale = float(np.ptp(M0))

    if t_scale <= 1e-12 or m_scale <= 1e-12:
        return np.full(len(target), np.nan)

    X0 = np.column_stack(
        [
            (T0 - t_min) / t_scale,
            (M0 - m_min) / m_scale,
        ]
    )

    T1 = target["T"].to_numpy(dtype=float)
    M1 = np.log(target["K"].to_numpy(dtype=float) / target_spot)
    X1 = np.column_stack(
        [
            (T1 - t_min) / t_scale,
            (M1 - m_min) / m_scale,
        ]
    )

    try:
        interpolator = LinearNDInterpolator(
            X0,
            Y0,
            fill_value=np.nan,
        )
        pred = np.asarray(interpolator(X1), dtype=float)
    except (QhullError, ValueError, FloatingPointError):
        pred = np.full(len(target), np.nan)

    return pred


def score_pair(
    origin_ref: SurfaceRef,
    target_ref: SurfaceRef,
    origin: pd.DataFrame,
    target: pd.DataFrame,
    payloads: dict[str, dict[str, Any]],
    cos_n: int,
    hawkes_cos_n: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    gap_days = int((target_ref.date - origin_ref.date).days)
    if gap_days <= 0:
        raise ValueError("Target must be after origin.")
    gap_years = gap_days / 365.25

    work = target.copy().reset_index(drop=True)
    work["origin_date"] = origin_ref.date.strftime("%Y-%m-%d")
    work["target_date"] = target_ref.date.strftime("%Y-%m-%d")
    work["gap_days"] = gap_days
    work["market_iv"] = pd.to_numeric(
        work["implied_vol"],
        errors="coerce",
    )
    work["mean_benchmark_iv"] = float(
        origin["implied_vol"].mean()
    )
    work["persistence_iv"] = persistence_forecast(origin, work)

    projected_payloads: dict[str, dict[str, float]] = {}
    for model in MODEL_KEYS:
        projected = project_model_state(
            model,
            params(payloads[model]),
            gap_years,
        )
        projected_payloads[model] = projected
        prices = model_prices(
            model,
            work,
            projected,
            cos_n=cos_n,
            hawkes_cos_n=hawkes_cos_n,
        )
        iv = prices_to_iv(work, prices)

        work[f"{model}_price"] = prices
        work[f"{model}_iv"] = iv
        work[f"{model}_error"] = work["market_iv"] - iv

    model_iv_cols = [f"{m}_iv" for m in MODEL_KEYS]
    common_model = (
        np.isfinite(work["market_iv"].to_numpy(dtype=float))
        & np.all(
            np.isfinite(
                work[model_iv_cols].to_numpy(dtype=float)
            ),
            axis=1,
        )
    )
    work["common_model_support"] = common_model

    common_persistence = (
        common_model
        & np.isfinite(
            work["persistence_iv"].to_numpy(dtype=float)
        )
    )
    work["common_persistence_support"] = common_persistence

    metrics: list[dict[str, Any]] = []

    if common_model.any():
        idx = np.where(common_model)[0]
        market = work.loc[idx, "market_iv"].to_numpy(dtype=float)
        mean_bench = work.loc[
            idx,
            "mean_benchmark_iv",
        ].to_numpy(dtype=float)
        mean_err = market - mean_bench
        mse_mean = float(np.mean(mean_err**2))

        for model in MODEL_KEYS:
            pred = work.loc[idx, f"{model}_iv"].to_numpy(dtype=float)
            err = market - pred
            metrics.append(
                {
                    "origin_date": origin_ref.date.strftime("%Y-%m-%d"),
                    "target_date": target_ref.date.strftime("%Y-%m-%d"),
                    "gap_days": gap_days,
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "common_n": int(len(idx)),
                    "mae_iv": float(np.mean(np.abs(err))),
                    "rmse_iv": float(np.sqrt(np.mean(err**2))),
                    "mse_iv": float(np.mean(err**2)),
                    "mean_benchmark_mse_iv": mse_mean,
                    "persistence_n": 0,
                    "persistence_mse_iv": np.nan,
                    "model_mse_on_persistence_support_iv": np.nan,
                }
            )

    metrics_df = pd.DataFrame(metrics)

    if common_persistence.any() and not metrics_df.empty:
        idx = np.where(common_persistence)[0]
        market = work.loc[idx, "market_iv"].to_numpy(dtype=float)
        persistence = work.loc[
            idx,
            "persistence_iv",
        ].to_numpy(dtype=float)
        mse_persistence = float(
            np.mean((market - persistence) ** 2)
        )

        for model in MODEL_KEYS:
            pred = work.loc[idx, f"{model}_iv"].to_numpy(dtype=float)
            mse_model = float(np.mean((market - pred) ** 2))
            mask = metrics_df["model"].eq(model)
            metrics_df.loc[mask, "persistence_n"] = int(len(idx))
            metrics_df.loc[
                mask,
                "persistence_mse_iv",
            ] = mse_persistence
            metrics_df.loc[
                mask,
                "model_mse_on_persistence_support_iv",
            ] = mse_model

    return work, metrics_df


def build_summary(
    forecasts: pd.DataFrame,
    date_metrics: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for model in MODEL_KEYS:
        dm = date_metrics.loc[
            date_metrics["model"].eq(model)
        ].copy()

        if dm.empty:
            continue

        f = forecasts.loc[
            forecasts["common_model_support"].astype(bool)
        ].copy()
        market = f["market_iv"].to_numpy(dtype=float)
        pred = f[f"{model}_iv"].to_numpy(dtype=float)
        mean_bench = f["mean_benchmark_iv"].to_numpy(dtype=float)

        pooled_mse = float(np.mean((market - pred) ** 2))
        pooled_mean_mse = float(
            np.mean((market - mean_bench) ** 2)
        )

        # Equal-weight-by-date OOS R2.
        denom_mean = float(dm["mean_benchmark_mse_iv"].sum())
        numer_mean = float(dm["mse_iv"].sum())
        r2_mean_date = (
            1.0 - numer_mean / denom_mean
            if denom_mean > 0.0
            else np.nan
        )

        persist_dm = dm.loc[
            dm["persistence_n"].gt(0)
            & np.isfinite(dm["persistence_mse_iv"])
            & np.isfinite(
                dm["model_mse_on_persistence_support_iv"]
            )
        ].copy()

        if persist_dm.empty:
            r2_persist_date = np.nan
        else:
            denom_p = float(
                persist_dm["persistence_mse_iv"].sum()
            )
            numer_p = float(
                persist_dm[
                    "model_mse_on_persistence_support_iv"
                ].sum()
            )
            r2_persist_date = (
                1.0 - numer_p / denom_p
                if denom_p > 0.0
                else np.nan
            )

        pf = forecasts.loc[
            forecasts["common_persistence_support"].astype(bool)
        ].copy()

        if pf.empty:
            pooled_r2_persist = np.nan
            pooled_persist_n = 0
        else:
            p_market = pf["market_iv"].to_numpy(dtype=float)
            p_model = pf[f"{model}_iv"].to_numpy(dtype=float)
            p_bench = pf["persistence_iv"].to_numpy(dtype=float)
            denom = float(
                np.sum((p_market - p_bench) ** 2)
            )
            numer = float(
                np.sum((p_market - p_model) ** 2)
            )
            pooled_r2_persist = (
                1.0 - numer / denom
                if denom > 0.0
                else np.nan
            )
            pooled_persist_n = int(len(pf))

        rows.append(
            {
                "model": model,
                "model_label": MODEL_LABELS[model],
                "forecast_origins": int(
                    dm["origin_date"].nunique()
                ),
                "common_target_observations": int(len(f)),
                "persistence_common_observations": pooled_persist_n,
                "date_mean_mae_bp": 10000.0
                * float(dm["mae_iv"].mean()),
                "date_mean_rmse_bp": 10000.0
                * float(dm["rmse_iv"].mean()),
                "pooled_mae_bp": 10000.0
                * float(np.mean(np.abs(market - pred))),
                "pooled_rmse_bp": 10000.0
                * math.sqrt(pooled_mse),
                "r2_oos_vs_mean_date_equal": r2_mean_date,
                "r2_oos_vs_mean_pooled": (
                    1.0 - pooled_mse / pooled_mean_mse
                    if pooled_mean_mse > 0.0
                    else np.nan
                ),
                "r2_oos_vs_persistence_date_equal":
                    r2_persist_date,
                "r2_oos_vs_persistence_pooled":
                    pooled_r2_persist,
            }
        )

    return pd.DataFrame(rows)


def plot_welch_goyal(
    date_metrics: pd.DataFrame,
    out_path: Path,
    title_prefix: str = "Rolling historical validation",
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

    for model in MODEL_KEYS:
        dm = (
            date_metrics.loc[
                date_metrics["model"].eq(model)
            ]
            .sort_values("target_date")
            .copy()
        )
        if dm.empty:
            continue

        # Welch-Goyal-style cumulative benchmark SSE advantage.
        delta_mean = (
            dm["mean_benchmark_mse_iv"]
            - dm["mse_iv"]
        ) * (10000.0**2)
        axes[0].plot(
            pd.to_datetime(dm["target_date"]),
            delta_mean.cumsum(),
            label=MODEL_LABELS[model],
        )

        pdm = dm.loc[
            dm["persistence_n"].gt(0)
            & np.isfinite(dm["persistence_mse_iv"])
            & np.isfinite(
                dm["model_mse_on_persistence_support_iv"]
            )
        ].copy()
        if not pdm.empty:
            delta_p = (
                pdm["persistence_mse_iv"]
                - pdm[
                    "model_mse_on_persistence_support_iv"
                ]
            ) * (10000.0**2)
            axes[1].plot(
                pd.to_datetime(pdm["target_date"]),
                delta_p.cumsum(),
                label=MODEL_LABELS[model],
            )

    axes[0].axhline(0.0, linewidth=1.0)
    axes[0].set_title("Cumulative MSE difference vs previous-date mean")
    axes[0].set_ylabel(
        r"$\sum_t(\mathrm{MSE}_{bench,t}-\mathrm{MSE}_{model,t})$ "
        r"(IV bp$^2$)"
    )
    axes[0].set_xlabel("Target date")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="best")

    axes[1].axhline(0.0, linewidth=1.0)
    axes[1].set_title("Cumulative MSE difference vs IV-surface persistence")
    axes[1].set_ylabel(
        r"$\sum_t(\mathrm{MSE}_{bench,t}-\mathrm{MSE}_{model,t})$ "
        r"(IV bp$^2$)"
    )
    axes[1].set_xlabel("Target date")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="best")

    fig.suptitle(title_prefix)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_latex_outputs(
    summary: pd.DataFrame,
    img_dir: Path,
    prefix: str = "OOS",
) -> None:
    img_dir.mkdir(parents=True, exist_ok=True)

    labels = {
        "bs": r"Black--Scholes",
        "heston": r"Heston",
        "bates": r"Bates",
        "hawkes": r"Full Bates--Hawkes",
    }

    rows = []
    for _, row in summary.iterrows():
        model = str(row["model"])
        r2m = 100.0 * float(row["r2_oos_vs_mean_date_equal"])
        r2p = 100.0 * float(
            row["r2_oos_vs_persistence_date_equal"]
        )
        rows.append(
            f"{labels[model]} & "
            f"{float(row['date_mean_mae_bp']):.2f} & "
            f"{float(row['date_mean_rmse_bp']):.2f} & "
            f"{r2m:.2f}\\% & "
            f"{r2p:.2f}\\% \\\\"
        )

    (img_dir / "oos_table_rows.tex").write_text(
        "\n".join(rows) + "\n",
        encoding="utf-8",
    )

    origins = (
        int(summary["forecast_origins"].max())
        if not summary.empty
        else 0
    )
    common_n = (
        int(summary["common_target_observations"].min())
        if not summary.empty
        else 0
    )

    macros = (
        f"\\newcommand{{\\OOSForecastOrigins}}{{{origins}}}\n"
        f"\\newcommand{{\\OOSCommonTargetObservations}}{{{common_n}}}\n"
    )
    (img_dir / "oos_article_numbers.tex").write_text(
        macros,
        encoding="utf-8",
    )


def create_coverage_rows(
    refs: list[SurfaceRef],
    loaded: dict[pd.Timestamp, pd.DataFrame],
) -> pd.DataFrame:
    rows = []
    for ref in refs:
        if ref.date not in loaded:
            continue
        st = surface_stats(loaded[ref.date])
        rows.append(
            {
                "date": ref.date.strftime("%Y-%m-%d"),
                "surface_kind": ref.kind,
                "surface_path": str(ref.path),
                **st,
            }
        )
    return pd.DataFrame(rows)


def run_calibrations_only(
    repo_root: Path,
    refs: list[SurfaceRef],
    loaded: dict[pd.Timestamp, pd.DataFrame],
    output_root: Path,
    n_t: int,
    n_k: int,
    profile: str,
    seed: int,
    force_calibration: bool,
) -> None:
    """Calibrate every rolling origin without OOS scoring."""
    if len(refs) < 2:
        raise ValueError("At least two dense dates are required.")

    calibration_sets_root = output_root / "calibration_sets"
    calibration_root = output_root / "calibrations"
    strategy = "ROLLING"
    origins = refs[:-1]

    for i, ref in enumerate(origins, start=1):
        full = loaded[ref.date]
        set_path, mode, sample = ensure_origin_set(
            ref,
            full,
            calibration_sets_root,
            n_t=n_t,
            n_k=n_k,
        )
        cal_dir = calibration_dir(calibration_root, strategy, ref.date)
        print(
            f"[CALIBRATION {i}/{len(origins)}] {ref.date.date()} | "
            f"{mode} | rows={len(sample)}"
        )
        if force_calibration or not all_calibrations_successful(cal_dir):
            invoke_calibration(
                repo_root=repo_root,
                date=ref.date,
                surface_path=set_path,
                calibration_root=calibration_root,
                strategy=strategy,
                profile=profile,
                seed=seed,
                force=force_calibration,
            )
        else:
            print("[SKIP] all four model calibrations already successful")

    print(
        f"[DONE] calibrated {len(origins)} dense rolling origins; "
        "last dense date is target-only."
    )


def run_rolling(
    repo_root: Path,
    refs: list[SurfaceRef],
    loaded: dict[pd.Timestamp, pd.DataFrame],
    output_root: Path,
    img_dir: Path,
    n_t: int,
    n_k: int,
    profile: str,
    seed: int,
    force_calibration: bool,
    min_origin_points: int,
    min_origin_expiries: int,
    cos_n: int,
    hawkes_cos_n: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if len(refs) < 2:
        raise ValueError("At least two usable dates are required.")

    calibration_sets_root = output_root / "calibration_sets"
    calibration_root = output_root / "calibrations"
    strategy = "ROLLING"

    forecast_frames = []
    metric_frames = []

    for pair_i in range(len(refs) - 1):
        origin_ref = refs[pair_i]
        target_ref = refs[pair_i + 1]
        origin = loaded[origin_ref.date]
        target = loaded[target_ref.date]

        st = surface_stats(origin)
        if st["rows"] < int(min_origin_points):
            print(
                f"[SKIP] {origin_ref.date.date()} -> "
                f"{target_ref.date.date()}: "
                f"origin rows={st['rows']} < {min_origin_points}"
            )
            continue
        if st["expiries"] < int(min_origin_expiries):
            print(
                f"[SKIP] {origin_ref.date.date()} -> "
                f"{target_ref.date.date()}: "
                f"origin expiries={st['expiries']} < "
                f"{min_origin_expiries}"
            )
            continue

        set_path, mode, _ = ensure_origin_set(
            origin_ref,
            origin,
            calibration_sets_root,
            n_t=n_t,
            n_k=n_k,
        )

        cal_dir = calibration_dir(
            calibration_root,
            strategy,
            origin_ref.date,
        )

        if force_calibration or not all_calibrations_successful(cal_dir):
            invoke_calibration(
                repo_root=repo_root,
                date=origin_ref.date,
                surface_path=set_path,
                calibration_root=calibration_root,
                strategy=strategy,
                profile=profile,
                seed=seed,
                force=force_calibration,
            )

        if not all_calibrations_successful(cal_dir):
            print(
                f"[SKIP] unsuccessful calibration at "
                f"{origin_ref.date.date()}"
            )
            continue

        payloads = read_model_payloads(cal_dir)

        print(
            f"[SCORE] {origin_ref.date.date()} -> "
            f"{target_ref.date.date()} | "
            f"origin={len(origin)} ({mode}) | "
            f"target={len(target)}"
        )

        forecasts, metrics = score_pair(
            origin_ref,
            target_ref,
            origin,
            target,
            payloads,
            cos_n=cos_n,
            hawkes_cos_n=hawkes_cos_n,
        )

        if metrics.empty:
            print("        [WARN] no common model support")
            continue

        forecast_frames.append(forecasts)
        metric_frames.append(metrics)

    if not forecast_frames or not metric_frames:
        raise RuntimeError(
            "No origin-target pair produced a valid common-support score."
        )

    forecasts = pd.concat(
        forecast_frames,
        ignore_index=True,
        sort=False,
    )
    date_metrics = pd.concat(
        metric_frames,
        ignore_index=True,
        sort=False,
    )
    summary = build_summary(forecasts, date_metrics)

    rolling_dir = output_root / "rolling"
    rolling_dir.mkdir(parents=True, exist_ok=True)

    forecasts.to_csv(
        rolling_dir / "option_forecasts.csv",
        index=False,
    )
    date_metrics.to_csv(
        rolling_dir / "date_metrics.csv",
        index=False,
    )
    summary.to_csv(
        rolling_dir / "model_summary.csv",
        index=False,
    )

    plot_welch_goyal(
        date_metrics,
        img_dir / "online_welch_goyal_cumulative.png",
        title_prefix="Rolling historical validation",
    )
    write_latex_outputs(summary, img_dir)

    return forecasts, date_metrics, summary


def run_fixed_origin(
    repo_root: Path,
    fixed_origin: pd.Timestamp,
    refs: list[SurfaceRef],
    loaded: dict[pd.Timestamp, pd.DataFrame],
    output_root: Path,
    img_dir: Path,
    n_t: int,
    n_k: int,
    profile: str,
    seed: int,
    force_calibration: bool,
    cos_n: int,
    hawkes_cos_n: int,
) -> None:
    """Optional fixed-parameter stress test.

    Structural parameters are frozen at one origin. State variables are
    projected from that origin over each target gap.
    """
    ref_by_date = {r.date: r for r in refs}
    if fixed_origin not in ref_by_date:
        raise ValueError(
            f"Fixed origin {fixed_origin.date()} is not in discovered surfaces."
        )

    origin_ref = ref_by_date[fixed_origin]
    origin = loaded[fixed_origin]

    calibration_sets_root = output_root / "calibration_sets"
    calibration_root = output_root / "calibrations"
    strategy = "ROLLING"

    set_path, _, _ = ensure_origin_set(
        origin_ref,
        origin,
        calibration_sets_root,
        n_t=n_t,
        n_k=n_k,
    )
    cal_dir = calibration_dir(
        calibration_root,
        strategy,
        fixed_origin,
    )

    if force_calibration or not all_calibrations_successful(cal_dir):
        invoke_calibration(
            repo_root=repo_root,
            date=fixed_origin,
            surface_path=set_path,
            calibration_root=calibration_root,
            strategy=strategy,
            profile=profile,
            seed=seed,
            force=force_calibration,
        )

    payloads = read_model_payloads(cal_dir)

    later_refs = [r for r in refs if r.date > fixed_origin]
    if not later_refs:
        raise ValueError("No target date exists after fixed origin.")

    forecast_frames = []
    metric_frames = []

    # For the persistence benchmark, use the immediately previous observed
    # surface, while model parameters remain frozen at fixed_origin.
    previous_ref = origin_ref
    previous_surface = origin

    for target_ref in later_refs:
        target = loaded[target_ref.date]

        forecasts, metrics = score_pair(
            origin_ref,
            target_ref,
            previous_surface,
            target,
            payloads,
            cos_n=cos_n,
            hawkes_cos_n=hawkes_cos_n,
        )

        # Restore fixed-origin mean benchmark: score_pair's persistence origin
        # is the previous surface, but the fixed exercise's mean benchmark is
        # deliberately frozen at the fixed calibration origin.
        fixed_mean = float(origin["implied_vol"].mean())
        forecasts["mean_benchmark_iv"] = fixed_mean

        common = forecasts["common_model_support"].astype(bool)
        if common.any():
            idx = forecasts.index[common]
            market = forecasts.loc[idx, "market_iv"].to_numpy(dtype=float)
            mse_mean = float(np.mean((market - fixed_mean) ** 2))
            metrics["mean_benchmark_mse_iv"] = mse_mean

        forecast_frames.append(forecasts)
        metric_frames.append(metrics)

        previous_ref = target_ref
        previous_surface = target

    forecasts = pd.concat(forecast_frames, ignore_index=True)
    date_metrics = pd.concat(metric_frames, ignore_index=True)
    summary = build_summary(forecasts, date_metrics)

    fixed_dir = (
        output_root
        / f"fixed_{fixed_origin.strftime('%Y-%m-%d')}"
    )
    fixed_dir.mkdir(parents=True, exist_ok=True)

    forecasts.to_csv(
        fixed_dir / "option_forecasts.csv",
        index=False,
    )
    date_metrics.to_csv(
        fixed_dir / "date_metrics.csv",
        index=False,
    )
    summary.to_csv(
        fixed_dir / "model_summary.csv",
        index=False,
    )

    plot_welch_goyal(
        date_metrics,
        img_dir / "oos_welch_goyal_cumulative.png",
        title_prefix=(
            "Fixed-parameter robustness from "
            f"{fixed_origin.strftime('%Y-%m-%d')}"
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--repo-root",
        default=".",
    )
    parser.add_argument(
        "--sparse-dir",
        default="data/processed/sparse_historical_surfaces",
    )
    parser.add_argument(
        "--dense-dir",
        default="data/processed/full_surfaces",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/oos",
    )
    parser.add_argument(
        "--img-dir",
        default="img/diagnostics_ibkr",
    )
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)

    parser.add_argument("--min-dte", type=int, default=75)
    parser.add_argument("--n-t", type=int, default=8)
    parser.add_argument("--n-k", type=int, default=8)
    parser.add_argument(
        "--min-surface-points",
        type=int,
        default=64,
        help=(
            "Minimum DTE-filtered actual observations required for EVERY date "
            "used in dense-only OOS. Default 64."
        ),
    )
    parser.add_argument(
        "--min-origin-points",
        type=int,
        default=None,
        help=(
            "Legacy extra floor. Effective threshold is the maximum of this "
            "value, --min-surface-points and n_t*n_k."
        ),
    )
    parser.add_argument(
        "--min-origin-expiries",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--profile",
        choices=["quick", "full"],
        default="full",
    )
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument(
        "--force-calibration",
        action="store_true",
    )
    parser.add_argument("--cos-n", type=int, default=192)
    parser.add_argument(
        "--hawkes-cos-n",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--fixed-origin",
        default=None,
        help=(
            "Optional YYYY-MM-DD frozen-parameter robustness origin. "
            "The rolling validation is still run first."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only discover and report usable surface files.",
    )
    parser.add_argument(
        "--calibrate-only",
        action="store_true",
        help=(
            "Build fixed CC64 origin sets and calibrate all rolling origins, "
            "then stop before OOS scoring."
        ),
    )

    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    sparse_dir = (repo_root / args.sparse_dir).resolve()
    dense_dir = (repo_root / args.dense_dir).resolve()
    output_root = (repo_root / args.output_root).resolve()
    img_dir = (repo_root / args.img_dir).resolve()

    refs = discover_surfaces(
        sparse_dir=sparse_dir,
        dense_dir=dense_dir,
        start=args.start,
        end=args.end,
        min_dte=int(args.min_dte),
    )

    if not refs:
        raise FileNotFoundError(
            "No historical/adaptive/full eligible surface CSVs found. "
            "When the historical panel is ready, first run "
            "surface_builder.py --build-all or place the recovered surfaces "
            "in the configured directories."
        )

    loaded: dict[pd.Timestamp, pd.DataFrame] = {}
    usable_refs: list[SurfaceRef] = []
    failures: list[dict[str, str]] = []

    legacy_floor = (
        int(args.min_origin_points)
        if args.min_origin_points is not None
        else 0
    )
    dense_min_points = max(
        int(args.min_surface_points),
        int(args.n_t) * int(args.n_k),
        legacy_floor,
    )
    dense_min_expiries = int(args.min_origin_expiries)

    print("=" * 100)
    print("TEAM 8 ROLLING HISTORICAL / OOS VALIDATION")
    print(f"Discovered candidate dates : {len(refs)}")
    print(f"Official min DTE           : {args.min_dte}")
    print(f"Dense rule                 : fixed CC {args.n_t}x{args.n_k}")
    print(f"Minimum rows/date          : {dense_min_points}")
    print(f"Minimum expiries/date      : {dense_min_expiries}")
    print("Sparse dates               : excluded from this OOS exercise")
    print("Target scoring             : conditional target-state IV surface")
    print("=" * 100)

    for ref in refs:
        try:
            frame = load_surface(
                ref,
                min_dte=int(args.min_dte),
            )
            st = surface_stats(frame)
            if st["rows"] < dense_min_points:
                print(
                    f"{ref.date.date()} | [EXCLUDE DENSE] "
                    f"rows={st['rows']} < {dense_min_points} | "
                    f"{ref.path.name}"
                )
                continue
            if st["expiries"] < dense_min_expiries:
                print(
                    f"{ref.date.date()} | [EXCLUDE DENSE] "
                    f"expiries={st['expiries']} < {dense_min_expiries} | "
                    f"{ref.path.name}"
                )
                continue

            loaded[ref.date] = frame
            usable_refs.append(ref)
            print(
                f"{ref.date.date()} | {ref.kind:12s} | "
                f"rows={st['rows']:4d} | "
                f"exp={st['expiries']:2d} | "
                f"K={st['strikes']:3d} | "
                f"{ref.path.name}"
            )
        except Exception as exc:
            failures.append(
                {
                    "date": ref.date.strftime("%Y-%m-%d"),
                    "path": str(ref.path),
                    "error": str(exc),
                }
            )
            print(
                f"{ref.date.date()} | [SKIP LOAD] {exc}"
            )

    if len(usable_refs) < 2:
        raise RuntimeError(
            "Fewer than two usable dates are currently available. "
            "The code is ready; rerun it once the historical panel contains "
            "at least two valid surfaces."
        )

    coverage = create_coverage_rows(usable_refs, loaded)
    output_root.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(
        output_root / "origin_target_coverage.csv",
        index=False,
    )
    if failures:
        pd.DataFrame(failures).to_csv(
            output_root / "surface_load_failures.csv",
            index=False,
        )

    if args.dry_run:
        print("=" * 100)
        print("[DRY RUN] No calibration or scoring performed.")
        print(f"[OK] dense dates retained: {len(usable_refs)}")
        print(
            f"[OK] coverage preview: "
            f"{output_root / 'origin_target_coverage.csv'}"
        )
        return

    if args.calibrate_only:
        run_calibrations_only(
            repo_root=repo_root,
            refs=usable_refs,
            loaded=loaded,
            output_root=output_root,
            n_t=int(args.n_t),
            n_k=int(args.n_k),
            profile=args.profile,
            seed=int(args.seed),
            force_calibration=bool(args.force_calibration),
        )
        return

    forecasts, date_metrics, summary = run_rolling(
        repo_root=repo_root,
        refs=usable_refs,
        loaded=loaded,
        output_root=output_root,
        img_dir=img_dir,
        n_t=int(args.n_t),
        n_k=int(args.n_k),
        profile=args.profile,
        seed=int(args.seed),
        force_calibration=bool(args.force_calibration),
        min_origin_points=dense_min_points,
        min_origin_expiries=dense_min_expiries,
        cos_n=int(args.cos_n),
        hawkes_cos_n=int(args.hawkes_cos_n),
    )

    run_manifest = {
        "surface_dates": [
            r.date.strftime("%Y-%m-%d")
            for r in usable_refs
        ],
        "surface_count": len(usable_refs),
        "official_min_dte": int(args.min_dte),
        "dense_sampling": f"CC_{args.n_t * args.n_k}",
        "dense_only_oos": True,
        "minimum_surface_points": int(dense_min_points),
        "minimum_surface_expiries": int(dense_min_expiries),
        "sparse_sampling": "EXCLUDED_FROM_OOS",
        "profile": args.profile,
        "seed": int(args.seed),
        "forecast_design": (
            "rolling one-next-available-date conditional IV-surface "
            "validation; structural parameters estimated at origin, "
            "Heston variance and Hawkes intensity states projected to target; "
            "target spot/rates used only as scoring conditioning variables"
        ),
        "mean_benchmark": "origin cross-sectional mean IV",
        "persistence_benchmark": (
            "origin observed IV surface interpolated in "
            "(T, log-moneyness) inside convex hull only"
        ),
        "forecast_rows": int(len(forecasts)),
        "date_metric_rows": int(len(date_metrics)),
    }
    json_write(
        output_root / "rolling" / "run_manifest.json",
        run_manifest,
    )

    if args.fixed_origin is not None:
        fixed = pd.Timestamp(args.fixed_origin).normalize()
        run_fixed_origin(
            repo_root=repo_root,
            fixed_origin=fixed,
            refs=usable_refs,
            loaded=loaded,
            output_root=output_root,
            img_dir=img_dir,
            n_t=int(args.n_t),
            n_k=int(args.n_k),
            profile=args.profile,
            seed=int(args.seed),
            force_calibration=bool(args.force_calibration),
            cos_n=int(args.cos_n),
            hawkes_cos_n=int(args.hawkes_cos_n),
        )

    print()
    print("=" * 100)
    print("[DONE] Rolling historical validation complete")
    print(f"[DONE] outputs : {output_root / 'rolling'}")
    print(
        f"[DONE] figure  : "
        f"{img_dir / 'online_welch_goyal_cumulative.png'}"
    )
    if args.fixed_origin is not None:
        print(
            f"[DONE] fixed robustness figure: "
            f"{img_dir / 'oos_welch_goyal_cumulative.png'}"
        )
    print("=" * 100)
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
