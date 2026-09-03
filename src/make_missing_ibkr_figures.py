"""Generate the IBKR figures that are present in the old paper but
required by the IBKR LaTeX version, excluding OOS figures.

Target figures (saved under img/diagnostics_ibkr/):
    usd_treasury_curve.png
    sampling_comparison.png
    volatility_surface_3d.png
    gld_return_normality.png

    black_scholes_residual_heatmap.png
    black_scholes_volatility_smile.png
    heston_residual_heatmap.png
    heston_volatility_smile.png
    bates_residual_heatmap.png
    bates_volatility_smile.png
    hawkes_intensity_comparison.png
    bates_hawkes_residual_heatmap.png
    bates_hawkes_volatility_smile.png
    hawkes_exact_vs_proxy.png
    bates_hawkes_vs_bates.png

    terminal_return_percentiles.png
    paths_black_scholes_5.png
    paths_heston_5.png
    paths_bates_5.png
    paths_bates_hawkes_5.png
    gold_path_stats_by_model.png
    volatility_state_paths.png
    hawkes_jump_paths.png
    bates_poisson_jump_paths.png

OOS-only figures from the legacy article are intentionally NOT generated here
until the final historical-validation panel is available.

Inputs expected from the Team 8 repository:
    data/processed/usd_treasury_history.csv
    data/processed/gld_daily_history.csv
    data/processed/full_surfaces/GLD_<DATE>_eligible_adaptive_surface.csv
    outputs/sampling/<DATE>/sample_CC_64.csv
    outputs/calibrations/CC/<DATE>/
        black_scholes.json
        heston.json
        bates.json
        full_bates_hawkes.json

Example:
    python src/make_missing_ibkr_figures.py --date 2026-09-02
"""

from __future__ import annotations

import argparse
import json
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator, FuncFormatter
from matplotlib.colors import TwoSlopeNorm
from scipy.interpolate import LinearNDInterpolator
from scipy.stats import gaussian_kde, jarque_bera, norm, normaltest, probplot, shapiro, skew, kurtosis

from rates import load_rate_history, curve_without_lookahead, fit_nss_curve, nss_rates
from BnS import BnS
from Heston import Heston
from Bates import Bates
from BatesHawkesExact import BatesHawkesExact


IV_CMAP = "viridis"

TARGET_NON_OOS_PNGS = [
    "usd_treasury_curve.png",
    "sampling_comparison.png",
    "volatility_surface_3d.png",
    "gld_return_normality.png",
    "black_scholes_residual_heatmap.png",
    "black_scholes_volatility_smile.png",
    "heston_residual_heatmap.png",
    "heston_volatility_smile.png",
    "bates_residual_heatmap.png",
    "bates_volatility_smile.png",
    "hawkes_intensity_comparison.png",
    "bates_hawkes_residual_heatmap.png",
    "bates_hawkes_volatility_smile.png",
    "hawkes_exact_vs_proxy.png",
    "bates_hawkes_vs_bates.png",
    "terminal_return_percentiles.png",
    "paths_black_scholes_5.png",
    "paths_heston_5.png",
    "paths_bates_5.png",
    "paths_bates_hawkes_5.png",
    "gold_path_stats_by_model.png",
    "volatility_state_paths.png",
    "hawkes_jump_paths.png",
    "bates_poisson_jump_paths.png",
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_png(fig, out: Path, *, dpi: int = 220, bbox_inches: str = "tight") -> None:
    """Save a Matplotlib figure robustly on Windows.

    Matplotlib/Pillow normally receives the output filename directly. On some
    Windows setups this can raise ``OSError: [Errno 22] Invalid argument`` even
    when the visible path is valid. Rendering to an in-memory PNG first avoids
    that filename handoff entirely; pathlib then performs the final byte write.
    """
    out = Path(str(out).strip().strip('"')).expanduser()

    if not out.is_absolute():
        out = (Path.cwd() / out).resolve()

    out.parent.mkdir(parents=True, exist_ok=True)

    buffer = BytesIO()
    fig.savefig(
        buffer,
        format="png",
        dpi=int(dpi),
        bbox_inches=bbox_inches,
    )
    out.write_bytes(buffer.getvalue())


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_close_column(df: pd.DataFrame) -> str:
    candidates = ["close", "Close", "adj_close", "Adj Close", "price", "Price"]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Could not infer close column from {list(df.columns)}")


def safe_date_column(df: pd.DataFrame) -> str:
    candidates = ["date", "Date", "timestamp", "Timestamp"]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Could not infer date column from {list(df.columns)}")


def load_surface(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    for c in ["T", "K", "implied_vol"]:
        if c not in df.columns:
            raise ValueError(f"{path} missing column {c}")
        df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in ["price", "rate", "vega", "spot"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["T", "K", "implied_vol"]).copy()
    df = df.loc[(df["T"] > 0) & (df["K"] > 0) & (df["implied_vol"] > 0)].copy()
    df = df.sort_values(["T", "K"]).reset_index(drop=True)
    return df


def read_model_params(calib_dir: Path) -> dict[str, dict[str, Any]]:
    out = {}
    files = {
        "black_scholes": calib_dir / "black_scholes.json",
        "heston": calib_dir / "heston.json",
        "bates": calib_dir / "bates.json",
        "bates_hawkes": calib_dir / "full_bates_hawkes.json",
    }
    for k, p in files.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing calibration file: {p}")
        payload = load_json(p)
        if not payload.get("success", False):
            raise RuntimeError(f"Calibration result is not successful: {p}")
        out[k] = payload
    return out


def get_param(payload: dict[str, Any], *names: str, default: float | None = None) -> float:
    params = payload.get("parameters", {}) or {}
    for name in names:
        if name in params:
            return float(params[name])
    if default is not None:
        return float(default)
    raise KeyError(f"None of {names} found in parameters keys {list(params.keys())}")


def _pct_colorbar(cbar) -> None:
    cbar.ax.yaxis.set_major_formatter(
        FuncFormatter(lambda x, pos: f"{100.0 * x:.0f}")
    )
    cbar.set_label("Implied volatility (%)")


def resolve_market_surface_path(repo: Path, date: str) -> tuple[Path, str]:
    """Prefer the current adaptive IBKR surface, retain legacy compatibility."""
    root = repo / "data" / "processed" / "full_surfaces"
    adaptive = root / f"GLD_{date}_eligible_adaptive_surface.csv"
    legacy = root / f"GLD_{date}_eligible_full_surface.csv"

    if adaptive.exists():
        return adaptive, "adaptive"
    if legacy.exists():
        return legacy, "legacy_full"

    raise FileNotFoundError(
        "No eligible market surface found. Expected one of:\n"
        f"  {adaptive}\n"
        f"  {legacy}"
    )



def _sample_signature(frame: pd.DataFrame) -> pd.DataFrame:
    required = ["T", "K", "price", "rate", "vega"]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(
            f"Cannot validate current calibration/sample consistency; "
            f"missing columns: {missing}"
        )
    out = frame[required].copy()
    for c in required:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna().sort_values(["T", "K"]).reset_index(drop=True)
    return out.round(10)


def validate_current_calibration_sample(
    sample: pd.DataFrame,
    calib_dir: Path,
) -> None:
    """Block generation if current sample and frozen calibration differ."""
    frozen_path = calib_dir / "calibration_surface.csv"
    if not frozen_path.exists():
        raise FileNotFoundError(
            f"Missing {frozen_path}. Re-run calibrate_surface.py on the "
            "CURRENT adaptive CC sample before generating article figures."
        )

    current = _sample_signature(sample)
    frozen = _sample_signature(pd.read_csv(frozen_path))

    if current.shape != frozen.shape or not np.allclose(
        current.to_numpy(dtype=float),
        frozen.to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-9,
    ):
        raise RuntimeError(
            "CURRENT CC sample and frozen calibration_surface.csv do not match. "
            "The figures would mix new adaptive data with old parameters. "
            "Re-run BS/Heston/Bates/Bates-Hawkes with --force first."
        )


def clean_generated_diagnostics(out_dir: Path) -> int:
    """Remove stale diagnostics so no old-data PNG survives this run."""
    removed = 0

    # This folder is dedicated to generated diagnostics.
    for path in out_dir.glob("*.png"):
        path.unlink()
        removed += 1

    for pattern in (
        "article_numbers.tex",
        "article_percentiles.tex",
        "model_error_summary.csv",
        "terminal_path_stats.csv",
        "normality_stats.json",
        "figure_manifest.json",
        "diagnostics_*.csv",
    ):
        for path in out_dir.glob(pattern):
            if path.is_file():
                path.unlink()
                removed += 1

    return removed


def assert_complete_non_oos_set(out_dir: Path) -> None:
    missing = [
        name for name in TARGET_NON_OOS_PNGS
        if not (out_dir / name).exists()
    ]
    if missing:
        raise RuntimeError(
            "Generation did not create all required NON-OOS figures:\\n  - "
            + "\\n  - ".join(missing)
        )


def plot_treasury_curve(rates_path: Path, asof: pd.Timestamp, out: Path) -> None:
    """Plot observed Treasury tenors and the no-look-ahead NSS curve."""
    history = load_rate_history(rates_path)
    curve, curve_date = curve_without_lookahead(history, asof)
    fit = fit_nss_curve(curve)

    observed_T = curve["maturity_years"].to_numpy(dtype=float)
    observed_r = curve["continuous_rate"].to_numpy(dtype=float)

    t_min = max(1.0 / 365.25, float(observed_T.min()))
    t_max = float(observed_T.max())
    dense_T = np.linspace(t_min, t_max, 600)
    dense_r = nss_rates(dense_T, fit)

    fig, ax = plt.subplots(figsize=(8.8, 5.6))

    ax.scatter(
        observed_T,
        100.0 * observed_r,
        s=45,
        label="Observed Treasury tenors",
        zorder=3,
    )
    ax.plot(
        dense_T,
        100.0 * dense_r,
        linewidth=2.0,
        label="Nelson-Siegel-Svensson fit",
    )

    ax.set_title(
        f"USD Treasury NSS curve used for option discounting ({curve_date.date()})"
    )
    ax.set_xlabel("Maturity (years)")
    ax.set_ylabel("Continuously compounded rate (%)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    diagnostics = (
        f"NSS RMSE = {fit.rmse_bps:.3f} bp\\n"
        f"$\\tau_1$ = {fit.tau1:.3f} y\\n"
        f"$\\tau_2$ = {fit.tau2:.3f} y"
    )
    ax.text(
        0.98,
        0.04,
        diagnostics,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round", alpha=0.15),
    )

    fig.tight_layout()
    save_png(fig, out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_sampling_comparison(
    full_surface: pd.DataFrame,
    sample_cc: pd.DataFrame,
    out: Path,
) -> None:
    """Plot the eligible universe and CC nodes with one common IV palette."""
    vmin = float(full_surface["implied_vol"].min())
    vmax = float(full_surface["implied_vol"].max())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))

    sc0 = axes[0].scatter(
        full_surface["T"],
        full_surface["K"],
        c=full_surface["implied_vol"],
        cmap=IV_CMAP,
        vmin=vmin,
        vmax=vmax,
        s=16,
    )
    axes[0].set_title("Eligible surface (DTE >= 75 days)")
    axes[0].set_xlabel("Maturity T (years)")
    axes[0].set_ylabel("Strike K")
    axes[0].grid(True, alpha=0.22)
    _pct_colorbar(fig.colorbar(sc0, ax=axes[0]))

    axes[1].scatter(
        full_surface["T"],
        full_surface["K"],
        s=11,
        alpha=0.15,
        color="0.70",
        label="Eligible market points",
    )
    sc1 = axes[1].scatter(
        sample_cc["T"],
        sample_cc["K"],
        c=sample_cc["implied_vol"],
        cmap=IV_CMAP,
        vmin=vmin,
        vmax=vmax,
        s=65,
        edgecolors="black",
        linewidths=0.55,
        label="CC 64-node sample",
        zorder=3,
    )
    axes[1].set_title("Chebyshev-Chebyshev calibration nodes")
    axes[1].set_xlabel("Maturity T (years)")
    axes[1].set_ylabel("Strike K")
    axes[1].grid(True, alpha=0.22)
    axes[1].legend(loc="best")
    _pct_colorbar(fig.colorbar(sc1, ax=axes[1]))

    fig.suptitle("GLD eligible IV surface and 64-point CC calibration sample")
    fig.tight_layout()
    save_png(fig, out)
    plt.close(fig)


def plot_surface_3d(
    full_surface: pd.DataFrame,
    sample_cc: pd.DataFrame,
    out: Path,
) -> None:
    """Stable IV visualization in normalized coordinates using viridis colors.

    Linear interpolation is used only for the visual surface. Calibration
    continues to use actual market observations only.
    """
    frame = full_surface[["T", "K", "implied_vol"]].dropna().copy()
    frame = frame.drop_duplicates(["T", "K"], keep="last")

    t = frame["T"].to_numpy(dtype=float)
    k = frame["K"].to_numpy(dtype=float)
    iv = frame["implied_vol"].to_numpy(dtype=float)

    t_min, t_max = float(t.min()), float(t.max())
    k_min, k_max = float(k.min()), float(k.max())
    t_scale = max(t_max - t_min, 1e-12)
    k_scale = max(k_max - k_min, 1e-12)

    points_norm = np.column_stack(
        ((t - t_min) / t_scale, (k - k_min) / k_scale)
    )
    interp = LinearNDInterpolator(points_norm, iv, fill_value=np.nan)

    t_grid = np.linspace(t_min, t_max, 90)
    k_grid = np.linspace(k_min, k_max, 120)
    tt, kk = np.meshgrid(t_grid, k_grid, indexing="ij")
    query_norm = np.column_stack(
        (
            (tt.ravel() - t_min) / t_scale,
            (kk.ravel() - k_min) / k_scale,
        )
    )
    zz = np.asarray(interp(query_norm), dtype=float).reshape(tt.shape)
    zz = np.ma.masked_invalid(zz)

    if np.asarray(zz.compressed()).size == 0:
        raise RuntimeError(
            "Linear IV interpolation produced no finite grid values."
        )

    vmin = float(full_surface["implied_vol"].min())
    vmax = float(full_surface["implied_vol"].max())

    fig = plt.figure(figsize=(10.5, 7.2))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        kk,
        tt,
        zz,
        cmap=IV_CMAP,
        vmin=vmin,
        vmax=vmax,
        linewidth=0,
        antialiased=True,
        alpha=0.94,
        rcount=90,
        ccount=120,
    )

    ax.scatter(
        sample_cc["K"],
        sample_cc["T"],
        sample_cc["implied_vol"],
        c=sample_cc["implied_vol"],
        cmap=IV_CMAP,
        vmin=vmin,
        vmax=vmax,
        s=42,
        edgecolors="black",
        linewidths=0.65,
        depthshade=False,
        label="CC nodes",
        zorder=5,
    )

    ax.set_xlabel("Strike K")
    ax.set_ylabel("Maturity T (years)")
    ax.set_zlabel("Implied volatility")
    ax.set_title("GLD implied-volatility surface with selected CC nodes")
    ax.view_init(elev=27, azim=-58)
    ax.legend(loc="upper right")

    cbar = fig.colorbar(surf, ax=ax, shrink=0.62, pad=0.10)
    _pct_colorbar(cbar)

    fig.tight_layout()
    save_png(fig, out)
    plt.close(fig)


def plot_return_normality(gld_path: Path, out: Path) -> dict[str, float]:
    df = pd.read_csv(gld_path)
    date_col = safe_date_column(df)
    close_col = safe_close_column(df)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[close_col] = pd.to_numeric(df[close_col], errors="coerce")
    df = df.dropna(subset=[date_col, close_col]).sort_values(date_col).copy()

    ret = 100.0 * np.log(df[close_col] / df[close_col].shift(1))
    ret = pd.Series(ret).dropna()
    mu = float(ret.mean())
    sd = float(ret.std(ddof=1))
    sample_skew = float(skew(ret.to_numpy(), bias=False))
    sample_excess_kurt = float(kurtosis(ret.to_numpy(), fisher=True, bias=False))

    # Tests
    sh_stat, sh_p = shapiro(ret.to_numpy()) if len(ret) <= 5000 else (np.nan, np.nan)
    jb = jarque_bera(ret.to_numpy())
    dag_stat, dag_p = normaltest(ret.to_numpy())

    # Plot
    fig = plt.figure(figsize=(12.5, 5.8))
    gs = GridSpec(1, 2, figure=fig)

    ax0 = fig.add_subplot(gs[0, 0])
    xs = np.linspace(float(ret.min()), float(ret.max()), 400)
    kde = gaussian_kde(ret.to_numpy())
    ax0.hist(ret, bins=24, density=True, alpha=0.35, label="Empirical histogram")
    ax0.plot(xs, kde(xs), label="Kernel density")
    ax0.plot(xs, norm.pdf(xs, loc=mu, scale=sd), label="Matched normal density")
    ax0.set_title("Daily GLD log-return distribution")
    ax0.set_xlabel("Log return (%)")
    ax0.set_ylabel("Density")
    ax0.legend(loc="best")
    ax0.grid(True, alpha=0.25)

    txt = (
        f"Shapiro-Wilk p = {sh_p:.3g}\n"
        f"Jarque-Bera p = {jb.pvalue:.3g}\n"
        f"D'Agostino K² p = {dag_p:.3g}\n"
        f"n = {len(ret)}"
    )
    ax0.text(0.97, 0.97, txt, transform=ax0.transAxes, ha="right", va="top",
             bbox=dict(boxstyle="round", alpha=0.15))

    ax1 = fig.add_subplot(gs[0, 1])
    (osm, osr), (slope, intercept, r) = probplot(ret.to_numpy(), dist="norm")
    ax1.scatter(osm, osr, s=16)
    qline = np.array([np.min(osm), np.max(osm)])
    ax1.plot(qline, slope * qline + intercept)
    ax1.set_title("Normal Q-Q plot of daily GLD log returns")
    ax1.set_xlabel("Theoretical quantiles")
    ax1.set_ylabel("Sample quantiles")
    ax1.grid(True, alpha=0.25)

    fig.tight_layout()
    save_png(fig, out, dpi=220, bbox_inches="tight")
    plt.close(fig)

    return {
        "n": int(len(ret)),
        "mean": mu,
        "std": sd,
        "skewness": sample_skew,
        "excess_kurtosis": sample_excess_kurt,
        "shapiro_stat": float(sh_stat),
        "shapiro_p": float(sh_p),
        "jarque_bera_stat": float(jb.statistic),
        "jarque_bera_p": float(jb.pvalue),
        "dagostino_k2_stat": float(dag_stat),
        "dagostino_k2_p": float(dag_p),
    }


def _vector_full_trunc_cir(v, kappa, theta, xi, dt, z):
    v_pos = np.maximum(v, 0.0)
    vn = v + kappa * (theta - v_pos) * dt + xi * np.sqrt(v_pos) * np.sqrt(dt) * z
    return np.maximum(vn, 0.0)


def simulate_black_scholes(spot, sigma, rate, years, n_steps, n_paths, seed):
    rng = np.random.default_rng(seed)
    dt = years / n_steps
    s = np.empty((n_steps + 1, n_paths), dtype=float)
    s[0] = spot
    for t in range(n_steps):
        z = rng.standard_normal(n_paths)
        s[t + 1] = s[t] * np.exp((rate - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
    return {"S": s}


def simulate_heston(spot, v0, kappa, theta, xi, rho, rate, years, n_steps, n_paths, seed):
    rng = np.random.default_rng(seed)
    dt = years / n_steps
    s = np.empty((n_steps + 1, n_paths), dtype=float)
    v = np.empty((n_steps + 1, n_paths), dtype=float)
    s[0] = spot
    v[0] = max(v0, 1e-8)

    for t in range(n_steps):
        z1 = rng.standard_normal(n_paths)
        z2 = rng.standard_normal(n_paths)
        zv = z1
        zs = rho * z1 + np.sqrt(max(1.0 - rho**2, 0.0)) * z2

        v[t + 1] = _vector_full_trunc_cir(v[t], kappa, theta, xi, dt, zv)
        vt = np.maximum(v[t], 0.0)
        s[t + 1] = s[t] * np.exp((rate - 0.5 * vt) * dt + np.sqrt(vt * dt) * zs)

    return {"S": s, "V": v}


def simulate_bates(spot, v0, kappa, theta, xi, rho, lambd, mu_j, sigma_j, rate, years, n_steps, n_paths, seed):
    rng = np.random.default_rng(seed)
    dt = years / n_steps
    s = np.empty((n_steps + 1, n_paths), dtype=float)
    v = np.empty((n_steps + 1, n_paths), dtype=float)
    n_cum = np.empty((n_steps + 1, n_paths), dtype=float)
    lam_path = np.empty((n_steps + 1, n_paths), dtype=float)

    s[0] = spot
    v[0] = max(v0, 1e-8)
    n_cum[0] = 0.0
    lam_path[:] = lambd

    # Risk-neutral compensator for multiplicative lognormal jumps
    k_jump = np.exp(mu_j + 0.5 * sigma_j**2) - 1.0

    for t in range(n_steps):
        z1 = rng.standard_normal(n_paths)
        z2 = rng.standard_normal(n_paths)
        zv = z1
        zs = rho * z1 + np.sqrt(max(1.0 - rho**2, 0.0)) * z2

        v[t + 1] = _vector_full_trunc_cir(v[t], kappa, theta, xi, dt, zv)
        vt = np.maximum(v[t], 0.0)

        n = rng.poisson(lam=lambd * dt, size=n_paths)
        jump_log = np.where(
            n > 0,
            n * mu_j + np.sqrt(np.maximum(n, 0.0)) * sigma_j * rng.standard_normal(n_paths),
            0.0,
        )

        drift = (rate - lambd * k_jump - 0.5 * vt) * dt
        diff = np.sqrt(vt * dt) * zs
        s[t + 1] = s[t] * np.exp(drift + diff + jump_log)

        n_cum[t + 1] = n_cum[t] + n

    return {"S": s, "V": v, "N": n_cum, "Lambda": lam_path}


def simulate_bates_hawkes(spot, v0, kappa, theta, xi, rho,
                          lambda0, lambda_bar, branching_ratio, beta,
                          mu_j, sigma_j, rate, years, n_steps, n_paths, seed):
    rng = np.random.default_rng(seed)
    dt = years / n_steps
    alpha = branching_ratio * beta

    s = np.empty((n_steps + 1, n_paths), dtype=float)
    v = np.empty((n_steps + 1, n_paths), dtype=float)
    lam = np.empty((n_steps + 1, n_paths), dtype=float)
    n_cum = np.empty((n_steps + 1, n_paths), dtype=float)

    s[0] = spot
    v[0] = max(v0, 1e-8)
    lam[0] = max(lambda0, 1e-10)
    n_cum[0] = 0.0

    base_comp = np.exp(mu_j + 0.5 * sigma_j**2) - 1.0

    for t in range(n_steps):
        z1 = rng.standard_normal(n_paths)
        z2 = rng.standard_normal(n_paths)
        zv = z1
        zs = rho * z1 + np.sqrt(max(1.0 - rho**2, 0.0)) * z2

        v[t + 1] = _vector_full_trunc_cir(v[t], kappa, theta, xi, dt, zv)
        vt = np.maximum(v[t], 0.0)

        lam_curr = np.maximum(lam[t], 1e-10)
        n = rng.poisson(lam=lam_curr * dt, size=n_paths)
        jump_log = np.where(
            n > 0,
            n * mu_j + np.sqrt(np.maximum(n, 0.0)) * sigma_j * rng.standard_normal(n_paths),
            0.0,
        )

        drift = (rate - lam_curr * base_comp - 0.5 * vt) * dt
        diff = np.sqrt(vt * dt) * zs
        s[t + 1] = s[t] * np.exp(drift + diff + jump_log)

        lam[t + 1] = np.maximum(lam[t] + beta * (lambda_bar - lam[t]) * dt + alpha * n, 1e-10)
        n_cum[t + 1] = n_cum[t] + n

    return {"S": s, "V": v, "N": n_cum, "Lambda": lam}



def _successful_payload(payload: dict[str, Any], label: str) -> None:
    if not bool(payload.get("success", False)):
        raise RuntimeError(
            f"Calibration result for {label} is not successful: "
            f"{payload.get('message', 'no message')}"
        )


def _market_iv_column(sample: pd.DataFrame, spot: float) -> np.ndarray:
    if "implied_vol" in sample.columns:
        values = pd.to_numeric(sample["implied_vol"], errors="coerce").to_numpy(float)
    else:
        values = np.full(len(sample), np.nan, dtype=float)

    missing = ~np.isfinite(values)
    if missing.any():
        for i in np.where(missing)[0]:
            row = sample.iloc[int(i)]
            values[i] = BnS.implied_vol_call(
                float(row["price"]),
                float(spot),
                float(row["K"]),
                float(row["T"]),
                float(row["rate"]),
            )
    return values


def _model_prices_for_slice(
    model_name: str,
    strikes: np.ndarray,
    maturity: float,
    rate: float,
    spot: float,
    payloads: dict[str, dict[str, Any]],
    *,
    cos_N: int = 192,
) -> np.ndarray:
    strikes = np.atleast_1d(np.asarray(strikes, dtype=float))
    T = float(maturity)
    r = float(rate)
    S0 = float(spot)

    if model_name == "Black-Scholes":
        p = payloads["black_scholes"]
        sigma = get_param(p, "sigma")
        return np.asarray(
            [BnS.bs_call_price(S0, float(K), T, r, sigma) for K in strikes],
            dtype=float,
        )

    if model_name == "Heston":
        p = payloads["heston"]
        values = (
            get_param(p, "v0"),
            get_param(p, "kappa"),
            get_param(p, "theta"),
            get_param(p, "xi", "sigma"),
            get_param(p, "rho"),
        )
        return np.asarray(
            Heston.heston_prices_cos(
                S0, strikes, T, *values, r, 0.0, N=int(cos_N)
            ),
            dtype=float,
        )

    if model_name == "Bates":
        p = payloads["bates"]
        values = (
            get_param(p, "v0"),
            get_param(p, "kappa"),
            get_param(p, "theta"),
            get_param(p, "xi", "sigma"),
            get_param(p, "rho"),
            get_param(p, "lambd", "lambda", "lambda_J"),
            get_param(p, "mu_J"),
            get_param(p, "sigma_J"),
        )
        return np.asarray(
            Bates.bates_prices_cos(
                S0, strikes, T, *values, r, 0.0, N=int(cos_N)
            ),
            dtype=float,
        )

    if model_name == "Bates-Hawkes":
        p = payloads["bates_hawkes"]
        beta = get_param(p, "beta")
        branching = get_param(p, "branching_ratio")
        alpha = get_param(p, "alpha", default=branching * beta)
        values = (
            get_param(p, "v0"),
            get_param(p, "kappa"),
            get_param(p, "theta"),
            get_param(p, "xi", "sigma"),
            get_param(p, "rho"),
            get_param(p, "lambda0"),
            get_param(p, "lambda_bar"),
            alpha,
            beta,
            get_param(p, "mu_J"),
            get_param(p, "sigma_J"),
        )
        return np.asarray(
            BatesHawkesExact.hawkes_price_cos(
                S0, strikes, T, *values, r, 0.0, N=int(cos_N)
            ),
            dtype=float,
        )

    raise ValueError(f"Unknown model_name: {model_name}")


def compute_model_diagnostics(
    sample: pd.DataFrame,
    spot: float,
    payloads: dict[str, dict[str, Any]],
    *,
    cos_N: int = 192,
) -> dict[str, pd.DataFrame]:
    """Price the common CC sample and convert every model price back to IV."""
    for key, label in [
        ("black_scholes", "Black-Scholes"),
        ("heston", "Heston"),
        ("bates", "Bates"),
        ("bates_hawkes", "Bates-Hawkes"),
    ]:
        _successful_payload(payloads[key], label)

    base = sample.copy().reset_index(drop=True)
    base["market_iv"] = _market_iv_column(base, spot)
    if "moneyness" not in base.columns:
        base["moneyness"] = pd.to_numeric(base["K"], errors="coerce") / float(spot)

    results: dict[str, pd.DataFrame] = {}

    for model_name in ["Black-Scholes", "Heston", "Bates", "Bates-Hawkes"]:
        model_price = np.full(len(base), np.nan, dtype=float)

        # Exact same maturity/rate bucket -> one vectorized COS call.
        grouped = base.groupby(["T", "rate"], sort=True, dropna=False)
        for (T, r), group in grouped:
            idx = group.index.to_numpy(dtype=int)
            try:
                model_price[idx] = _model_prices_for_slice(
                    model_name,
                    group["K"].to_numpy(dtype=float),
                    float(T),
                    float(r),
                    float(spot),
                    payloads,
                    cos_N=cos_N,
                )
            except Exception as exc:
                print(
                    f"[WARN] {model_name} pricing failed at T={float(T):.6f}: {exc}"
                )

        model_iv = np.full(len(base), np.nan, dtype=float)
        for i, row in base.iterrows():
            price_i = model_price[i]
            if not np.isfinite(price_i) or price_i <= 0.0:
                continue
            model_iv[i] = BnS.implied_vol_call(
                float(price_i),
                float(spot),
                float(row["K"]),
                float(row["T"]),
                float(row["rate"]),
            )

        frame = base.copy()
        frame["model_price"] = model_price
        frame["model_iv"] = model_iv
        # Paper convention: market IV minus model IV.
        frame["iv_residual"] = frame["market_iv"] - frame["model_iv"]
        frame["price_residual"] = (
            pd.to_numeric(frame["price"], errors="coerce") - frame["model_price"]
        )
        results[model_name] = frame

    return results


def model_error_summary(
    diagnostics: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []
    for model_name, frame in diagnostics.items():
        market_price = pd.to_numeric(frame["price"], errors="coerce").to_numpy(float)
        model_price = pd.to_numeric(frame["model_price"], errors="coerce").to_numpy(float)
        market_iv = pd.to_numeric(frame["market_iv"], errors="coerce").to_numpy(float)
        model_iv = pd.to_numeric(frame["model_iv"], errors="coerce").to_numpy(float)

        mask_p = np.isfinite(market_price) & np.isfinite(model_price)
        mask_iv = np.isfinite(market_iv) & np.isfinite(model_iv)

        pe = model_price[mask_p] - market_price[mask_p]
        ive = model_iv[mask_iv] - market_iv[mask_iv]

        # Residual normality is a diagnostic only; it is not a model-selection
        # criterion. Use the paper convention residual = market IV - model IV.
        resid = market_iv[mask_iv] - model_iv[mask_iv]
        if resid.size >= 8:
            sh = shapiro(resid) if resid.size <= 5000 else None
            jb_r = jarque_bera(resid)
            dag_r = normaltest(resid)
            sh_p = float(sh.pvalue) if sh is not None else np.nan
            jb_p = float(jb_r.pvalue)
            dag_p = float(dag_r.pvalue)
        else:
            sh_p = jb_p = dag_p = np.nan

        rows.append(
            {
                "model": model_name,
                "n_price": int(mask_p.sum()),
                "n_iv": int(mask_iv.sum()),
                "price_mae": float(np.mean(np.abs(pe))) if pe.size else np.nan,
                "price_rmse": float(np.sqrt(np.mean(pe**2))) if pe.size else np.nan,
                "iv_mae_bps": float(10000.0 * np.mean(np.abs(ive))) if ive.size else np.nan,
                "iv_rmse_bps": float(10000.0 * np.sqrt(np.mean(ive**2))) if ive.size else np.nan,
                "iv_linf_bps": float(10000.0 * np.max(np.abs(ive))) if ive.size else np.nan,
                "residual_shapiro_p": sh_p,
                "residual_jarque_bera_p": jb_p,
                "residual_dagostino_p": dag_p,
            }
        )
    return pd.DataFrame(rows)


def plot_residual_heatmap(
    frame: pd.DataFrame,
    model_name: str,
    out: Path,
) -> None:
    data = frame[
        ["T", "moneyness", "iv_residual"]
    ].replace([np.inf, -np.inf], np.nan).dropna().copy()

    if len(data) < 4:
        raise RuntimeError(f"Not enough residual points for {model_name} heatmap.")

    t = data["T"].to_numpy(float)
    m = data["moneyness"].to_numpy(float)
    residual_bp = 10000.0 * data["iv_residual"].to_numpy(float)

    t_min, t_max = float(t.min()), float(t.max())
    m_min, m_max = float(m.min()), float(m.max())
    t_scale = max(t_max - t_min, 1e-12)
    m_scale = max(m_max - m_min, 1e-12)

    pts = np.column_stack(
        [
            (t - t_min) / t_scale,
            (m - m_min) / m_scale,
        ]
    )
    interp = LinearNDInterpolator(pts, residual_bp, fill_value=np.nan)

    tg = np.linspace(t_min, t_max, 100)
    mg = np.linspace(m_min, m_max, 120)
    TT, MM = np.meshgrid(tg, mg, indexing="ij")
    q = np.column_stack(
        [
            (TT.ravel() - t_min) / t_scale,
            (MM.ravel() - m_min) / m_scale,
        ]
    )
    ZZ = np.asarray(interp(q), dtype=float).reshape(TT.shape)
    ZZ = np.ma.masked_invalid(ZZ)

    finite = ZZ.compressed()
    if finite.size == 0:
        raise RuntimeError(f"No finite interpolated residuals for {model_name}.")

    vmax = max(float(np.nanmax(np.abs(finite))), 1e-8)
    norm_obj = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    mesh = ax.pcolormesh(
        MM,
        TT,
        ZZ,
        shading="auto",
        cmap="coolwarm",
        norm=norm_obj,
    )
    ax.scatter(
        m,
        t,
        facecolors="none",
        edgecolors="black",
        s=32,
        linewidths=0.65,
        label="CC market nodes",
    )
    ax.axvline(1.0, linewidth=0.8, linestyle="--", color="black", alpha=0.55)
    ax.set_xlabel("Moneyness K / S0")
    ax.set_ylabel("Maturity T (years)")
    ax.set_title(f"{model_name} implied-volatility residuals")
    ax.grid(False)
    ax.legend(loc="best")
    fig.colorbar(mesh, ax=ax, label="Market IV - model IV (bp)")
    fig.tight_layout()
    save_png(fig, out)
    plt.close(fig)


def _representative_maturities(frame: pd.DataFrame, max_slices: int = 8) -> list[float]:
    unique = np.sort(pd.to_numeric(frame["T"], errors="coerce").dropna().unique())
    if len(unique) <= max_slices:
        return [float(x) for x in unique]
    positions = np.linspace(0, len(unique) - 1, max_slices)
    idx = np.unique(np.round(positions).astype(int))
    return [float(unique[i]) for i in idx[:max_slices]]


def plot_smile_slices(
    sample: pd.DataFrame,
    model_name: str,
    payloads: dict[str, dict[str, Any]],
    spot: float,
    out: Path,
    *,
    max_slices: int = 8,
    cos_N: int = 192,
) -> None:
    maturities = _representative_maturities(sample, max_slices=max_slices)
    n = len(maturities)
    if n == 0:
        raise RuntimeError(f"No maturity slices available for {model_name}.")

    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(14.0, 3.4 * nrows),
        squeeze=False,
        sharey=True,
    )

    for ax, T in zip(axes.ravel(), maturities):
        group = sample.loc[np.isclose(sample["T"].to_numpy(float), T)].copy()
        group = group.sort_values("K")
        if group.empty:
            ax.set_visible(False)
            continue

        r = float(pd.to_numeric(group["rate"], errors="coerce").median())
        k_min = float(group["K"].min())
        k_max = float(group["K"].max())
        if k_max <= k_min:
            k_grid = np.asarray([k_min], dtype=float)
        else:
            k_grid = np.linspace(k_min, k_max, 60)

        prices = _model_prices_for_slice(
            model_name,
            k_grid,
            T,
            r,
            spot,
            payloads,
            cos_N=cos_N,
        )
        model_iv = np.asarray(
            [
                BnS.implied_vol_call(
                    float(p), float(spot), float(k), T, r
                )
                if np.isfinite(p) and p > 0.0
                else np.nan
                for p, k in zip(prices, k_grid)
            ],
            dtype=float,
        )

        market_iv = _market_iv_column(group, spot)
        x_market = group["K"].to_numpy(float) / float(spot)
        x_grid = k_grid / float(spot)

        ax.scatter(
            x_market,
            100.0 * market_iv,
            s=32,
            label="Market",
            zorder=3,
        )
        good = np.isfinite(model_iv)
        if good.any():
            ax.plot(
                x_grid[good],
                100.0 * model_iv[good],
                linewidth=1.8,
                label=model_name,
            )

        dte = int(round(T * 365.25))
        ax.set_title(f"DTE ~ {dte} days")
        ax.set_xlabel("Moneyness K / S0")
        ax.grid(True, alpha=0.25)

    for ax in axes[:, 0]:
        if ax.get_visible():
            ax.set_ylabel("Implied volatility (%)")

    for ax in axes.ravel()[n:]:
        ax.set_visible(False)

    # Use one compact common legend.
    visible_axes = [ax for ax in axes.ravel() if ax.get_visible()]
    if visible_axes:
        handles, labels = visible_axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)

    fig.suptitle(f"{model_name}: market and model smile slices", y=1.01)
    fig.tight_layout()
    save_png(fig, out)
    plt.close(fig)


def plot_hawkes_intensity_comparison(
    time_grid: np.ndarray,
    bates_sim: dict[str, np.ndarray],
    hawkes_sim: dict[str, np.ndarray],
    out: Path,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.8), sharex=True)

    axes[0].plot(
        time_grid,
        bates_sim["Lambda"][:, 0],
        linewidth=1.8,
        label="Bates constant intensity",
    )
    axes[0].plot(
        time_grid,
        hawkes_sim["Lambda"][:, 0],
        linewidth=1.6,
        label="Bates-Hawkes intensity",
    )
    axes[0].set_ylabel("Jump intensity")
    axes[0].set_title("Constant Poisson intensity versus self-exciting intensity")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.25)

    axes[1].step(
        time_grid,
        bates_sim["N"][:, 0],
        where="post",
        linewidth=1.5,
        label="Bates jumps",
    )
    axes[1].step(
        time_grid,
        hawkes_sim["N"][:, 0],
        where="post",
        linewidth=1.5,
        label="Bates-Hawkes jumps",
    )
    axes[1].set_xlabel("Years")
    axes[1].set_ylabel("Cumulative jump count")
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.25)

    fig.tight_layout()
    save_png(fig, out)
    plt.close(fig)


def plot_hawkes_exact_vs_proxy(
    sample: pd.DataFrame,
    payloads: dict[str, dict[str, Any]],
    spot: float,
    out: Path,
) -> None:
    """Constant-volatility transform diagnostic; not the final Heston-Hawkes fit."""
    bs = payloads["black_scholes"]
    hw = payloads["bates_hawkes"]

    sigma = get_param(bs, "sigma")
    lambda0 = get_param(hw, "lambda0")
    lambda_bar = get_param(hw, "lambda_bar")
    branching = get_param(hw, "branching_ratio")
    beta = get_param(hw, "beta")
    alpha = get_param(hw, "alpha", default=branching * beta)
    mu_j = get_param(hw, "mu_J")
    sigma_j = get_param(hw, "sigma_J")

    unique_t = np.sort(sample["T"].astype(float).unique())
    T = float(unique_t[len(unique_t) // 2])
    near = sample.iloc[np.argmin(np.abs(sample["T"].to_numpy(float) - T))]
    r = float(near["rate"])

    strikes = np.linspace(0.85 * spot, 1.15 * spot, 70)
    exact = BatesHawkesExact.hawkes_price_constvol_cos(
        spot,
        strikes,
        T,
        sigma,
        lambda0,
        lambda_bar,
        alpha,
        beta,
        mu_j,
        sigma_j,
        r,
        0.0,
        N=192,
    )

    stationary_intensity = lambda_bar / max(1.0 - branching, 1e-6)
    proxy = BatesHawkesExact.hawkes_price_constvol_cos(
        spot,
        strikes,
        T,
        sigma,
        stationary_intensity,
        stationary_intensity,
        0.0,
        beta,
        mu_j,
        sigma_j,
        r,
        0.0,
        N=192,
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.9))

    axes[0].plot(strikes / spot, exact, label="Exact Hawkes transform")
    axes[0].plot(
        strikes / spot,
        proxy,
        linestyle="--",
        label="Stationary-intensity Poisson proxy",
    )
    axes[0].set_xlabel("Moneyness K / S0")
    axes[0].set_ylabel("Call price")
    axes[0].set_title(f"Constant-volatility benchmark, DTE ~ {int(round(T*365.25))} days")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="best")

    sensitivity = np.linspace(0.0, min(0.90, max(0.60, branching + 0.25)), 13)
    atm_prices = []
    for n in sensitivity:
        a = float(n * beta)
        p = BatesHawkesExact.hawkes_price_constvol_cos(
            spot,
            np.asarray([spot]),
            T,
            sigma,
            lambda0,
            lambda_bar,
            a,
            beta,
            mu_j,
            sigma_j,
            r,
            0.0,
            N=160,
        )
        atm_prices.append(float(np.asarray(p).ravel()[0]))

    axes[1].plot(sensitivity, atm_prices, marker="o", markersize=3)
    axes[1].axvline(branching, linestyle="--", linewidth=1.0, label="Calibrated n")
    axes[1].set_xlabel("Branching ratio n = alpha / beta")
    axes[1].set_ylabel("ATM call price")
    axes[1].set_title("Exact-Hawkes branching-ratio sensitivity")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="best")

    fig.tight_layout()
    save_png(fig, out)
    plt.close(fig)


def plot_bates_hawkes_vs_bates(
    diagnostics: dict[str, pd.DataFrame],
    out: Path,
) -> None:
    bates = diagnostics["Bates"].copy()
    hawkes = diagnostics["Bates-Hawkes"].copy()

    market = pd.to_numeric(bates["price"], errors="coerce").to_numpy(float)
    b_price = pd.to_numeric(bates["model_price"], errors="coerce").to_numpy(float)
    h_price = pd.to_numeric(hawkes["model_price"], errors="coerce").to_numpy(float)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0))

    mask_b = np.isfinite(market) & np.isfinite(b_price)
    mask_h = np.isfinite(market) & np.isfinite(h_price)

    axes[0].scatter(market[mask_b], b_price[mask_b], s=25, alpha=0.70, label="Bates")
    axes[0].scatter(market[mask_h], h_price[mask_h], s=25, alpha=0.70, label="Bates-Hawkes")
    all_vals = np.concatenate(
        [
            market[np.isfinite(market)],
            b_price[np.isfinite(b_price)],
            h_price[np.isfinite(h_price)],
        ]
    )
    lo, hi = float(np.min(all_vals)), float(np.max(all_vals))
    axes[0].plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.0, color="black")
    axes[0].set_xlabel("Market call price")
    axes[0].set_ylabel("Model call price")
    axes[0].set_title("Call-price fit on the common CC sample")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="best")

    m = pd.to_numeric(bates["moneyness"], errors="coerce").to_numpy(float)
    b_res = 10000.0 * pd.to_numeric(
        bates["iv_residual"], errors="coerce"
    ).to_numpy(float)
    h_res = 10000.0 * pd.to_numeric(
        hawkes["iv_residual"], errors="coerce"
    ).to_numpy(float)
    good_b = np.isfinite(m) & np.isfinite(b_res)
    good_h = np.isfinite(m) & np.isfinite(h_res)

    axes[1].scatter(m[good_b], b_res[good_b], s=25, alpha=0.70, label="Bates")
    axes[1].scatter(m[good_h], h_res[good_h], s=25, alpha=0.70, label="Bates-Hawkes")
    axes[1].axhline(0.0, linewidth=1.0, color="black")
    axes[1].set_xlabel("Moneyness K / S0")
    axes[1].set_ylabel("Market IV - model IV (bp)")
    axes[1].set_title("Implied-volatility residual comparison")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="best")

    fig.tight_layout()
    save_png(fig, out)
    plt.close(fig)



def write_article_percentiles_tex(
    out: Path,
    sims: dict[str, dict[str, np.ndarray]],
) -> None:
    """Write the 0--100 simple-return percentile rows used by the appendix."""
    models = ["Black-Scholes", "Heston", "Bates", "Bates-Hawkes"]
    values: dict[str, np.ndarray] = {}
    for model in models:
        s = sims[model]["S"]
        ret = 100.0 * (s[-1] / s[0, 0] - 1.0)
        values[model] = np.percentile(ret, np.arange(101))

    rows = []
    for p in range(51):
        left = [p] + [values[m][p] for m in models]
        q = p + 51
        if q <= 100:
            right = [q] + [values[m][q] for m in models]
            rows.append(
                f"{left[0]} & "
                + " & ".join(f"{x:.3f}" for x in left[1:])
                + " & "
                + f"{right[0]} & "
                + " & ".join(f"{x:.3f}" for x in right[1:])
                + r" \\"
            )
        else:
            rows.append(
                f"{left[0]} & "
                + " & ".join(f"{x:.3f}" for x in left[1:])
                + r" & & & & & \\"
            )

    out.write_text(
        "% Auto-generated five-year simple-return percentile rows.\n"
        + "\n".join(rows)
        + "\n",
        encoding="utf-8",
    )


def _fmt(value: Any, digits: int = 5) -> str:
    try:
        x = float(value)
    except Exception:
        return "***"
    if not np.isfinite(x):
        return "***"
    return f"{x:.{digits}f}"


def _decision(p: float) -> str:
    if not np.isfinite(p):
        return "***"
    return "Reject normality" if p < 0.05 else "Do not reject"


def write_article_numbers_tex(
    out: Path,
    *,
    normality_stats: dict[str, float],
    payloads: dict[str, dict[str, Any]],
    error_summary: pd.DataFrame,
    path_summary_df: pd.DataFrame,
) -> None:
    """Write LaTeX macros so the article can ingest frozen numerical outputs."""
    errors = error_summary.set_index("model").to_dict(orient="index")
    paths = path_summary_df.set_index("model").to_dict(orient="index")

    h = payloads["heston"]["parameters"]
    b = payloads["bates"]["parameters"]
    w = payloads["bates_hawkes"]["parameters"]
    bs = payloads["black_scholes"]["parameters"]

    h_sigma = h.get("sigma", h.get("xi", np.nan))
    b_sigma = b.get("sigma", b.get("xi", np.nan))
    w_xi = w.get("xi", w.get("sigma", np.nan))

    h_feller = 2.0 * float(h["kappa"]) * float(h["theta"]) - float(h_sigma) ** 2
    b_feller = 2.0 * float(b["kappa"]) * float(b["theta"]) - float(b_sigma) ** 2
    w_feller = 2.0 * float(w["kappa"]) * float(w["theta"]) - float(w_xi) ** 2

    def err(model, field, digits=4):
        return _fmt(errors.get(model, {}).get(field, np.nan), digits)

    def path(model, field, digits=4):
        return _fmt(paths.get(model, {}).get(field, np.nan), digits)

    macros = {
        "ReturnN": str(int(normality_stats.get("n", 0))),
        "ReturnSkew": _fmt(normality_stats.get("skewness"), 4),
        "ReturnExKurt": _fmt(normality_stats.get("excess_kurtosis"), 4),
        "ShapiroStat": _fmt(normality_stats.get("shapiro_stat"), 5),
        "ShapiroP": _fmt(normality_stats.get("shapiro_p"), 6),
        "ShapiroDecision": _decision(float(normality_stats.get("shapiro_p", np.nan))),
        "JBStat": _fmt(normality_stats.get("jarque_bera_stat"), 4),
        "JBP": _fmt(normality_stats.get("jarque_bera_p"), 6),
        "JBDecision": _decision(float(normality_stats.get("jarque_bera_p", np.nan))),
        "DagoStat": _fmt(normality_stats.get("dagostino_k2_stat"), 4),
        "DagoP": _fmt(normality_stats.get("dagostino_k2_p"), 6),
        "DagoDecision": _decision(float(normality_stats.get("dagostino_k2_p", np.nan))),
        "BSsigma": _fmt(bs.get("sigma"), 6),
        "Hvzero": _fmt(h.get("v0"), 6),
        "Hkappa": _fmt(h.get("kappa"), 6),
        "Htheta": _fmt(h.get("theta"), 6),
        "Hxi": _fmt(h_sigma, 6),
        "Hrho": _fmt(h.get("rho"), 6),
        "HFeller": _fmt(h_feller, 6),
        "Bvzero": _fmt(b.get("v0"), 6),
        "Bkappa": _fmt(b.get("kappa"), 6),
        "Btheta": _fmt(b.get("theta"), 6),
        "Bxi": _fmt(b_sigma, 6),
        "Brho": _fmt(b.get("rho"), 6),
        "Blambda": _fmt(b.get("lambd", b.get("lambda")), 6),
        "BmuJ": _fmt(b.get("mu_J"), 6),
        "BsigmaJ": _fmt(b.get("sigma_J"), 6),
        "BFeller": _fmt(b_feller, 6),
        "Wvzero": _fmt(w.get("v0"), 6),
        "Wkappa": _fmt(w.get("kappa"), 6),
        "Wtheta": _fmt(w.get("theta"), 6),
        "Wxi": _fmt(w_xi, 6),
        "Wrho": _fmt(w.get("rho"), 6),
        "WlambdaZero": _fmt(w.get("lambda0"), 6),
        "WlambdaBar": _fmt(w.get("lambda_bar"), 6),
        "Wbranch": _fmt(w.get("branching_ratio"), 6),
        "Wbeta": _fmt(w.get("beta"), 6),
        "WmuJ": _fmt(w.get("mu_J"), 6),
        "WsigmaJ": _fmt(w.get("sigma_J"), 6),
        "WFeller": _fmt(w_feller, 6),
        "BSPriceMAE": err("Black-Scholes", "price_mae"),
        "BSPriceRMSE": err("Black-Scholes", "price_rmse"),
        "BSIVMAE": err("Black-Scholes", "iv_mae_bps"),
        "BSIVRMSE": err("Black-Scholes", "iv_rmse_bps"),
        "HPriceMAE": err("Heston", "price_mae"),
        "HPriceRMSE": err("Heston", "price_rmse"),
        "HIVMAE": err("Heston", "iv_mae_bps"),
        "HIVRMSE": err("Heston", "iv_rmse_bps"),
        "BPriceMAE": err("Bates", "price_mae"),
        "BPriceRMSE": err("Bates", "price_rmse"),
        "BIVMAE": err("Bates", "iv_mae_bps"),
        "BIVRMSE": err("Bates", "iv_rmse_bps"),
        "WPriceMAE": err("Bates-Hawkes", "price_mae"),
        "WPriceRMSE": err("Bates-Hawkes", "price_rmse"),
        "WIVMAE": err("Bates-Hawkes", "iv_mae_bps"),
        "WIVRMSE": err("Bates-Hawkes", "iv_rmse_bps"),
        "BSResidShapiroP": err("Black-Scholes", "residual_shapiro_p", 6),
        "BSResidJBP": err("Black-Scholes", "residual_jarque_bera_p", 6),
        "BSResidDagoP": err("Black-Scholes", "residual_dagostino_p", 6),
        "HResidShapiroP": err("Heston", "residual_shapiro_p", 6),
        "HResidJBP": err("Heston", "residual_jarque_bera_p", 6),
        "HResidDagoP": err("Heston", "residual_dagostino_p", 6),
        "BResidShapiroP": err("Bates", "residual_shapiro_p", 6),
        "BResidJBP": err("Bates", "residual_jarque_bera_p", 6),
        "BResidDagoP": err("Bates", "residual_dagostino_p", 6),
        "WResidShapiroP": err("Bates-Hawkes", "residual_shapiro_p", 6),
        "WResidJBP": err("Bates-Hawkes", "residual_jarque_bera_p", 6),
        "WResidDagoP": err("Bates-Hawkes", "residual_dagostino_p", 6),
    }

    model_key = {
        "Black-Scholes": "BS",
        "Heston": "H",
        "Bates": "B",
        "Bates-Hawkes": "W",
    }
    for model, key in model_key.items():
        macros[f"{key}PathMean"] = path(model, "terminal_mean", 3)
        macros[f"{key}PathStd"] = path(model, "terminal_std", 3)
        macros[f"{key}PathPZeroFive"] = path(model, "terminal_p05", 3)
        macros[f"{key}PathPFifty"] = path(model, "terminal_p50", 3)
        macros[f"{key}PathPNinetyFive"] = path(model, "terminal_p95", 3)
        macros[f"{key}ReturnSkew"] = path(model, "return_skewness", 3)
        macros[f"{key}ReturnKurt"] = path(model, "return_excess_kurtosis", 3)
        macros[f"{key}ReturnMeanPct"] = path(model, "return_mean_pct", 3)
        macros[f"{key}ReturnStdPct"] = path(model, "return_std_pct", 3)
        macros[f"{key}ReturnPZero"] = path(model, "return_p00", 3)
        macros[f"{key}ReturnPFiftyPct"] = path(model, "return_p50", 3)
        macros[f"{key}ReturnPOneHundred"] = path(model, "return_p100", 3)
        macros[f"{key}MeanJumps"] = path(model, "mean_jumps", 3)

    lines = [
        "% Auto-generated by src/make_missing_ibkr_figures.py.",
        "% Do not edit numerical macros manually; regenerate from frozen outputs.",
    ]
    for name, value in macros.items():
        lines.append(rf"\def\{name}{{{value}}}")

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def path_summary(sim: dict[str, np.ndarray]) -> dict[str, Any]:
    s = sim["S"]
    terminal = s[-1]
    simple_ret = 100.0 * (terminal / s[0, 0] - 1.0)
    out = {
        "terminal_mean": float(np.mean(terminal)),
        "terminal_median": float(np.median(terminal)),
        "terminal_std": float(np.std(terminal, ddof=1)),
        "return_mean_pct": float(np.mean(simple_ret)),
        "return_median_pct": float(np.median(simple_ret)),
        "return_std_pct": float(np.std(simple_ret, ddof=1)),
        "return_skewness": float(skew(simple_ret, bias=False)),
        "return_excess_kurtosis": float(kurtosis(simple_ret, fisher=True, bias=False)),
        "return_p00": float(np.percentile(simple_ret, 0)),
        "return_p50": float(np.percentile(simple_ret, 50)),
        "return_p100": float(np.percentile(simple_ret, 100)),
        "terminal_p01": float(np.percentile(terminal, 1)),
        "terminal_p05": float(np.percentile(terminal, 5)),
        "terminal_p50": float(np.percentile(terminal, 50)),
        "terminal_p95": float(np.percentile(terminal, 95)),
        "terminal_p99": float(np.percentile(terminal, 99)),
        "mean_jumps": 0.0,
    }
    if "N" in sim:
        out["mean_jumps"] = float(np.mean(sim["N"][-1]))
    return out


def plot_single_path_panel(time_grid, values, title, ylabel, out, n_show=5):
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for i in range(min(n_show, values.shape[1])):
        ax.plot(time_grid, values[:, i], alpha=0.95)
    ax.set_title(title)
    ax.set_xlabel("Years")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    save_png(fig, out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_path_bands(time_grid, sims: dict[str, dict[str, np.ndarray]], out: Path):
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.8))
    axes = axes.ravel()

    titles = {
        "Black-Scholes": "Black-Scholes",
        "Heston": "Heston",
        "Bates": "Bates",
        "Bates-Hawkes": "Bates-Hawkes",
    }

    for ax, (name, sim) in zip(axes, sims.items()):
        s = sim["S"]
        p10 = np.percentile(s, 10, axis=1)
        p25 = np.percentile(s, 25, axis=1)
        p50 = np.percentile(s, 50, axis=1)
        p75 = np.percentile(s, 75, axis=1)
        p90 = np.percentile(s, 90, axis=1)
        ax.fill_between(time_grid, p10, p90, alpha=0.18, label="10-90 band")
        ax.fill_between(time_grid, p25, p75, alpha=0.30, label="25-75 band")
        ax.plot(time_grid, p50, linewidth=2.0, label="Median")
        ax.set_title(titles[name])
        ax.set_xlabel("Years")
        ax.set_ylabel("Simulated GLD price")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")

    fig.suptitle("Five-year GLD path bands by model")
    fig.tight_layout()
    save_png(fig, out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_terminal_return_percentiles(sims: dict[str, dict[str, np.ndarray]], out: Path):
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    q = np.arange(0, 101, 1)

    for name, sim in sims.items():
        terminal = sim["S"][-1]
        ret = 100.0 * (terminal / sim["S"][0, 0] - 1.0)
        pct = np.percentile(ret, q)
        ax.plot(q, pct, label=name)

    ax.set_title("Terminal simple-return percentiles over five-year simulations")
    ax.set_xlabel("Percentile")
    ax.set_ylabel("Terminal simple return (%)")
    ax.set_yscale("symlog", linthresh=5.0)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    save_png(fig, out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_volatility_states(time_grid, sims: dict[str, dict[str, np.ndarray]], out: Path, n_show=5):
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.6), sharey=True)
    keys = [("Heston", "Heston"), ("Bates", "Bates"), ("Bates-Hawkes", "Bates-Hawkes")]
    for ax, (k, title) in zip(axes, keys):
        v = sims[k]["V"]
        for i in range(min(n_show, v.shape[1])):
            ax.plot(time_grid, v[:, i], alpha=0.95)
        ax.set_title(title)
        ax.set_xlabel("Years")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Variance state")
    fig.suptitle("Stochastic-volatility paths on a common scale")
    fig.tight_layout()
    save_png(fig, out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_hawkes_jump_states(time_grid, sim: dict[str, np.ndarray], out: Path, n_show=5):
    fig, axes = plt.subplots(2, 1, figsize=(8.6, 6.6), sharex=True)
    lam = sim["Lambda"]
    n = sim["N"]
    for i in range(min(n_show, lam.shape[1])):
        axes[0].plot(time_grid, lam[:, i], alpha=0.95)
        axes[1].plot(time_grid, n[:, i], alpha=0.95)
    axes[0].set_title("Bates-Hawkes intensity paths")
    axes[0].set_ylabel("Intensity")
    axes[0].grid(True, alpha=0.25)
    axes[1].set_title("Bates-Hawkes cumulative jump counts")
    axes[1].set_xlabel("Years")
    axes[1].set_ylabel("Cumulative count")
    axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    save_png(fig, out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_bates_jump_states(time_grid, sim: dict[str, np.ndarray], out: Path, n_show=5):
    fig, axes = plt.subplots(2, 1, figsize=(8.6, 6.6), sharex=True)
    lam = sim["Lambda"]
    n = sim["N"]
    for i in range(min(n_show, lam.shape[1])):
        axes[0].plot(time_grid, lam[:, i], alpha=0.95)
        axes[1].plot(time_grid, n[:, i], alpha=0.95)
    axes[0].set_title("Bates constant Poisson intensity")
    axes[0].set_ylabel("Intensity")
    axes[0].grid(True, alpha=0.25)
    axes[1].set_title("Bates cumulative jump counts")
    axes[1].set_xlabel("Years")
    axes[1].set_ylabel("Cumulative count")
    axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    save_png(fig, out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def infer_constant_rate(surface: pd.DataFrame) -> float:
    if "rate" in surface.columns and surface["rate"].notna().any():
        return float(pd.to_numeric(surface["rate"], errors="coerce").dropna().median())
    return 0.04


def infer_spot(surface: pd.DataFrame) -> float:
    if "spot" in surface.columns and surface["spot"].notna().any():
        return float(pd.to_numeric(surface["spot"], errors="coerce").dropna().median())
    raise ValueError("Surface has no valid spot column.")


def build_simulations(sample_cc: pd.DataFrame, model_payloads: dict[str, dict[str, Any]],
                      years: float, n_steps: int, n_paths: int, seed: int) -> tuple[np.ndarray, dict[str, dict[str, np.ndarray]], pd.DataFrame]:
    spot = infer_spot(sample_cc)
    rate = infer_constant_rate(sample_cc)
    time_grid = np.linspace(0.0, years, n_steps + 1)

    bs = model_payloads["black_scholes"]
    heston = model_payloads["heston"]
    bates = model_payloads["bates"]
    hawkes = model_payloads["bates_hawkes"]

    sigma_bs = get_param(bs, "sigma")

    h_v0 = get_param(heston, "v0")
    h_kappa = get_param(heston, "kappa")
    h_theta = get_param(heston, "theta")
    h_xi = get_param(heston, "xi", "sigma")
    h_rho = get_param(heston, "rho")

    b_v0 = get_param(bates, "v0")
    b_kappa = get_param(bates, "kappa")
    b_theta = get_param(bates, "theta")
    b_xi = get_param(bates, "xi", "sigma")
    b_rho = get_param(bates, "rho")
    b_lambd = get_param(bates, "lambd", "lambda", "lambda_J")
    b_mu_j = get_param(bates, "mu_J")
    b_sigma_j = get_param(bates, "sigma_J")

    hw_v0 = get_param(hawkes, "v0")
    hw_kappa = get_param(hawkes, "kappa")
    hw_theta = get_param(hawkes, "theta")
    hw_xi = get_param(hawkes, "xi", "sigma")
    hw_rho = get_param(hawkes, "rho")
    hw_lambda0 = get_param(hawkes, "lambda0")
    hw_lambda_bar = get_param(hawkes, "lambda_bar")
    hw_branch = get_param(hawkes, "branching_ratio")
    hw_beta = get_param(hawkes, "beta")
    hw_mu_j = get_param(hawkes, "mu_J")
    hw_sigma_j = get_param(hawkes, "sigma_J")

    sims = {
        "Black-Scholes": simulate_black_scholes(spot, sigma_bs, rate, years, n_steps, n_paths, seed + 10),
        "Heston": simulate_heston(spot, h_v0, h_kappa, h_theta, h_xi, h_rho, rate, years, n_steps, n_paths, seed + 20),
        "Bates": simulate_bates(spot, b_v0, b_kappa, b_theta, b_xi, b_rho, b_lambd, b_mu_j, b_sigma_j, rate, years, n_steps, n_paths, seed + 30),
        "Bates-Hawkes": simulate_bates_hawkes(spot, hw_v0, hw_kappa, hw_theta, hw_xi, hw_rho,
                                              hw_lambda0, hw_lambda_bar, hw_branch, hw_beta,
                                              hw_mu_j, hw_sigma_j, rate, years, n_steps, n_paths, seed + 40),
    }

    rows = []
    for name, sim in sims.items():
        row = {"model": name, "spot0": spot, "rate": rate, "years": years, "n_steps": n_steps, "n_paths": n_paths}
        row.update(path_summary(sim))
        rows.append(row)
    summary_df = pd.DataFrame(rows)

    return time_grid, sims, summary_df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, help="Calibration / latest surface date, e.g. 2026-09-02")
    parser.add_argument("--strategy", default="CC")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--years", type=float, default=5.0)
    parser.add_argument("--n-steps", type=int, default=260)
    parser.add_argument("--n-paths", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--min-dte", type=int, default=75)
    parser.add_argument("--out-dir", default="img/diagnostics_ibkr")
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    date = pd.Timestamp(args.date).strftime("%Y-%m-%d")
    strategy = args.strategy.upper()

    out_dir = (repo / args.out_dir).resolve()
    ensure_dir(out_dir)

    rates_path = repo / "data" / "processed" / "usd_treasury_history.csv"
    gld_path = repo / "data" / "processed" / "gld_daily_history.csv"
    full_surface_path, surface_kind = resolve_market_surface_path(repo, date)
    sample_path = repo / "outputs" / "sampling" / date / f"sample_{strategy}_64.csv"
    calib_dir = repo / "outputs" / "calibrations" / strategy / date

    if not rates_path.exists():
        raise FileNotFoundError(rates_path)
    if not gld_path.exists():
        raise FileNotFoundError(gld_path)
    if not full_surface_path.exists():
        raise FileNotFoundError(full_surface_path)
    if not sample_path.exists():
        raise FileNotFoundError(sample_path)
    if not calib_dir.exists():
        raise FileNotFoundError(calib_dir)

    full_surface = load_surface(full_surface_path)
    sample_cc = load_surface(sample_path)

    if args.min_dte < 1:
        raise ValueError("--min-dte must be at least 1 day.")

    if "dte" in full_surface.columns:
        full_dte = pd.to_numeric(full_surface["dte"], errors="coerce")
    else:
        full_dte = 365.25 * pd.to_numeric(full_surface["T"], errors="coerce")
    full_surface = full_surface.loc[full_dte.ge(float(args.min_dte))].copy()
    full_surface = full_surface.sort_values(["T", "K"]).reset_index(drop=True)

    if "dte" in sample_cc.columns:
        sample_dte = pd.to_numeric(sample_cc["dte"], errors="coerce")
    else:
        sample_dte = 365.25 * pd.to_numeric(sample_cc["T"], errors="coerce")
    if sample_dte.lt(float(args.min_dte)).any() or sample_dte.isna().any():
        raise ValueError(
            f"The selected {strategy} sample contains observations below the "
            f"official DTE >= {args.min_dte} day domain. Regenerate sampling "
            "before producing figures."
        )

    model_payloads = read_model_params(calib_dir)

    # Never mix current adaptive data with stale calibration parameters.
    validate_current_calibration_sample(sample_cc, calib_dir)

    # Remove every existing diagnostic PNG/numerical companion before this run.
    # This guarantees that old LSE / pre-NSS / pre-DTE75 / pre-adaptive images
    # cannot survive beside the current results.
    removed = clean_generated_diagnostics(out_dir)
    if removed:
        print(f"[CLEAN] removed {removed} old diagnostic files")

    # Price the common CC sample under every calibrated model. These outputs
    # drive residual heatmaps, maturity-by-maturity smile slices, model tables,
    # and the direct Bates-vs-Hawkes diagnostic.
    spot = infer_spot(sample_cc)
    model_diagnostics = compute_model_diagnostics(
        sample_cc,
        spot,
        model_payloads,
        cos_N=192,
    )
    error_summary = model_error_summary(model_diagnostics)
    error_summary.to_csv(out_dir / "model_error_summary.csv", index=False)
    for model_name, frame in model_diagnostics.items():
        slug = model_name.lower().replace("-", "_").replace(" ", "_")
        frame.to_csv(out_dir / f"diagnostics_{slug}.csv", index=False)

    # Fail early if Windows cannot create a normal file in the requested folder.
    probe = out_dir / "_write_test.tmp"
    try:
        probe.write_bytes(b"ok")
        probe.unlink()
    except OSError as exc:
        raise OSError(
            f"Output directory is not writable: {out_dir!s}. "
            f"Original error: {exc}"
        ) from exc

    manifest = {
        "date": date,
        "strategy": strategy,
        "repo_root": str(repo),
        "market_surface_path": str(full_surface_path),
        "market_surface_kind": surface_kind,
        "ibkr_acquisition": "adaptive_all_expiries_moneyness_strike_grid",
        "sample_path": str(sample_path),
        "calibration_dir": str(calib_dir),
        "out_dir": str(out_dir),
        "rate_curve_model": "Nelson-Siegel-Svensson",
        "rate_fit_target": "continuous_rate",
        "official_min_dte": int(args.min_dte),
        "iv_colormap": IV_CMAP,
        "cc_nodes_share_surface_colormap": True,
    }

    # Static / descriptive figures
    plot_treasury_curve(rates_path, pd.Timestamp(date), out_dir / "usd_treasury_curve.png")
    plot_sampling_comparison(full_surface, sample_cc, out_dir / "sampling_comparison.png")
    plot_surface_3d(full_surface, sample_cc, out_dir / "volatility_surface_3d.png")
    normality_stats = plot_return_normality(gld_path, out_dir / "gld_return_normality.png")

    # Calibration/slicing diagnostics restored from the previous article.
    plot_residual_heatmap(
        model_diagnostics["Black-Scholes"],
        "Black-Scholes",
        out_dir / "black_scholes_residual_heatmap.png",
    )
    plot_smile_slices(
        sample_cc,
        "Black-Scholes",
        model_payloads,
        spot,
        out_dir / "black_scholes_volatility_smile.png",
    )

    plot_residual_heatmap(
        model_diagnostics["Heston"],
        "Heston",
        out_dir / "heston_residual_heatmap.png",
    )
    plot_smile_slices(
        sample_cc,
        "Heston",
        model_payloads,
        spot,
        out_dir / "heston_volatility_smile.png",
    )

    plot_residual_heatmap(
        model_diagnostics["Bates"],
        "Bates",
        out_dir / "bates_residual_heatmap.png",
    )
    plot_smile_slices(
        sample_cc,
        "Bates",
        model_payloads,
        spot,
        out_dir / "bates_volatility_smile.png",
    )

    plot_residual_heatmap(
        model_diagnostics["Bates-Hawkes"],
        "Full Bates-Hawkes",
        out_dir / "bates_hawkes_residual_heatmap.png",
    )
    plot_smile_slices(
        sample_cc,
        "Bates-Hawkes",
        model_payloads,
        spot,
        out_dir / "bates_hawkes_volatility_smile.png",
    )

    plot_hawkes_exact_vs_proxy(
        sample_cc,
        model_payloads,
        spot,
        out_dir / "hawkes_exact_vs_proxy.png",
    )
    plot_bates_hawkes_vs_bates(
        model_diagnostics,
        out_dir / "bates_hawkes_vs_bates.png",
    )

    # Simulations and simulation-based figures
    time_grid, sims, summary_df = build_simulations(
        sample_cc=sample_cc,
        model_payloads=model_payloads,
        years=float(args.years),
        n_steps=int(args.n_steps),
        n_paths=int(args.n_paths),
        seed=int(args.seed),
    )

    plot_single_path_panel(time_grid, sims["Black-Scholes"]["S"], "Five sample Black-Scholes paths", "GLD price", out_dir / "paths_black_scholes_5.png")
    plot_single_path_panel(time_grid, sims["Heston"]["S"], "Five sample Heston paths", "GLD price", out_dir / "paths_heston_5.png")
    plot_single_path_panel(time_grid, sims["Bates"]["S"], "Five sample Bates paths", "GLD price", out_dir / "paths_bates_5.png")
    plot_single_path_panel(time_grid, sims["Bates-Hawkes"]["S"], "Five sample Bates-Hawkes paths", "GLD price", out_dir / "paths_bates_hawkes_5.png")

    plot_hawkes_intensity_comparison(
        time_grid,
        sims["Bates"],
        sims["Bates-Hawkes"],
        out_dir / "hawkes_intensity_comparison.png",
    )

    plot_path_bands(time_grid, sims, out_dir / "gold_path_stats_by_model.png")
    plot_terminal_return_percentiles(sims, out_dir / "terminal_return_percentiles.png")
    plot_volatility_states(time_grid, sims, out_dir / "volatility_state_paths.png")
    plot_hawkes_jump_states(time_grid, sims["Bates-Hawkes"], out_dir / "hawkes_jump_paths.png")
    plot_bates_jump_states(time_grid, sims["Bates"], out_dir / "bates_poisson_jump_paths.png")

    # Useful metadata for the later LaTeX update
    summary_df.to_csv(out_dir / "terminal_path_stats.csv", index=False)
    write_article_percentiles_tex(
        out_dir / "article_percentiles.tex",
        sims,
    )
    write_article_numbers_tex(
        out_dir / "article_numbers.tex",
        normality_stats=normality_stats,
        payloads=model_payloads,
        error_summary=error_summary,
        path_summary_df=summary_df,
    )
    (out_dir / "normality_stats.json").write_text(json.dumps(normality_stats, indent=2), encoding="utf-8")
    (out_dir / "figure_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    assert_complete_non_oos_set(out_dir)

    print("=" * 92)
    print("[OK] ALL current NON-OOS IBKR figures generated")
    print(f"[OK] Output directory : {out_dir}")
    print(f"[OK] Market surface   : {full_surface_path.name} | rows = {len(full_surface)}")
    print(f"[OK] CC sample        : {sample_path.name} | rows = {len(sample_cc)}")
    print(f"[OK] Calibration dir  : {calib_dir}")
    print(f"[OK] IV palette       : {IV_CMAP} (same normalization for surface and CC nodes)")
    print("[OK] Figures:")
    for name in [
        "usd_treasury_curve.png",
        "sampling_comparison.png",
        "volatility_surface_3d.png",
        "gld_return_normality.png",
        "black_scholes_residual_heatmap.png",
        "black_scholes_volatility_smile.png",
        "heston_residual_heatmap.png",
        "heston_volatility_smile.png",
        "bates_residual_heatmap.png",
        "bates_volatility_smile.png",
        "hawkes_intensity_comparison.png",
        "bates_hawkes_residual_heatmap.png",
        "bates_hawkes_volatility_smile.png",
        "hawkes_exact_vs_proxy.png",
        "bates_hawkes_vs_bates.png",
        "terminal_return_percentiles.png",
        "paths_black_scholes_5.png",
        "paths_heston_5.png",
        "paths_bates_5.png",
        "paths_bates_hawkes_5.png",
        "gold_path_stats_by_model.png",
        "volatility_state_paths.png",
        "hawkes_jump_paths.png",
        "bates_poisson_jump_paths.png",
    ]:
        print(f"   - {name}")
    print("=" * 92)


if __name__ == "__main__":
    main()
