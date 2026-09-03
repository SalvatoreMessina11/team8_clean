"""Generate non-OOS IBKR diagnostic figures for the Team 8 GLD project.

This replacement is designed for the current repository structure:

    data/processed/full_surfaces/GLD_<DATE>_eligible_full_surface.csv
    outputs/sampling/<DATE>/sample_<STRATEGY>_64.csv
    outputs/calibrations/<STRATEGY>/<DATE>/

Methodological conventions preserved:
- no-look-ahead Nelson-Siegel-Svensson Treasury curve;
- official calibration domain DTE >= 75 days by default;
- fixed CC = Chebyshev T x Chebyshev K calibration geometry;
- interpolation is used for visualization only, never to manufacture
  calibration observations.

Publication style update:
- the IV surface uses the ``viridis`` palette (dark purple -> green -> yellow);
- Chebyshev/CC nodes use the SAME IV palette and normalization;
- PNGs are rendered to memory before being written, avoiding a Windows/Pillow
  ``OSError: [Errno 22] Invalid argument`` seen on some systems.
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
from matplotlib.ticker import FuncFormatter
from scipy.interpolate import LinearNDInterpolator
from scipy.stats import gaussian_kde, jarque_bera, norm, normaltest, probplot, shapiro

from rates import curve_without_lookahead, fit_nss_curve, load_rate_history, nss_rates


IV_CMAP = "viridis"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_png(fig, out: Path, *, dpi: int = 220, bbox_inches: str = "tight") -> None:
    """Save a Matplotlib figure robustly on Windows via an in-memory buffer."""
    out = Path(str(out).strip().strip('"')).expanduser()
    if not out.is_absolute():
        out = (Path.cwd() / out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=int(dpi), bbox_inches=bbox_inches)
    out.write_bytes(buffer.getvalue())


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_close_column(df: pd.DataFrame) -> str:
    for c in ["close", "Close", "adj_close", "Adj Close", "price", "Price"]:
        if c in df.columns:
            return c
    raise ValueError(f"Could not infer close column from {list(df.columns)}")


def safe_date_column(df: pd.DataFrame) -> str:
    for c in ["date", "Date", "timestamp", "Timestamp"]:
        if c in df.columns:
            return c
    raise ValueError(f"Could not infer date column from {list(df.columns)}")


def load_surface(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)

    for c in ["T", "K", "implied_vol"]:
        if c not in df.columns:
            raise ValueError(f"{path} missing column {c}")
        df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in ["price", "rate", "vega", "spot", "dte"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["T", "K", "implied_vol"]).copy()
    df = df.loc[(df["T"] > 0) & (df["K"] > 0) & (df["implied_vol"] > 0)].copy()
    return df.sort_values(["T", "K"]).reset_index(drop=True)


def read_model_params(calib_dir: Path) -> dict[str, dict[str, Any]]:
    files = {
        "black_scholes": calib_dir / "black_scholes.json",
        "heston": calib_dir / "heston.json",
        "bates": calib_dir / "bates.json",
        "bates_hawkes": calib_dir / "full_bates_hawkes.json",
    }
    out: dict[str, dict[str, Any]] = {}
    for name, path in files.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing calibration file: {path}")
        payload = load_json(path)
        if not payload.get("success", False):
            raise RuntimeError(f"Calibration result is not successful: {path}")
        out[name] = payload
    return out


def get_param(payload: dict[str, Any], *names: str, default: float | None = None) -> float:
    params = payload.get("parameters", {}) or {}
    for name in names:
        if name in params:
            return float(params[name])
    if default is not None:
        return float(default)
    raise KeyError(f"None of {names} found in parameter keys {list(params.keys())}")


def _pct_colorbar(cbar) -> None:
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{100.0 * x:.0f}"))
    cbar.set_label("Implied volatility (%)")


def plot_treasury_curve(rates_path: Path, asof: pd.Timestamp, out: Path) -> None:
    history = load_rate_history(rates_path)
    curve, curve_date = curve_without_lookahead(history, asof)
    fit = fit_nss_curve(curve)

    observed_t = curve["maturity_years"].to_numpy(dtype=float)
    observed_r = curve["continuous_rate"].to_numpy(dtype=float)
    dense_t = np.linspace(max(1.0 / 365.25, observed_t.min()), observed_t.max(), 600)
    dense_r = nss_rates(dense_t, fit)

    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    ax.scatter(observed_t, 100.0 * observed_r, s=45, label="Observed Treasury tenors", zorder=3)
    ax.plot(dense_t, 100.0 * dense_r, linewidth=2.0, label="Nelson-Siegel-Svensson fit")
    ax.set_title(f"USD Treasury NSS curve used for option discounting ({curve_date.date()})")
    ax.set_xlabel("Maturity (years)")
    ax.set_ylabel("Continuously compounded rate (%)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    ax.text(
        0.98,
        0.04,
        f"NSS RMSE = {fit.rmse_bps:.3f} bp\n$\\tau_1$ = {fit.tau1:.3f} y\n$\\tau_2$ = {fit.tau2:.3f} y",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round", alpha=0.15),
    )
    fig.tight_layout()
    save_png(fig, out)
    plt.close(fig)


def plot_sampling_comparison(full_surface: pd.DataFrame, sample_cc: pd.DataFrame, out: Path) -> None:
    """Plot the eligible universe and the CC nodes with one common IV palette."""
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


def plot_surface_3d(full_surface: pd.DataFrame, sample_cc: pd.DataFrame, out: Path) -> None:
    """Stable IV visualization using normalized coordinates and common viridis colors."""
    frame = full_surface[["T", "K", "implied_vol"]].dropna().copy()
    frame = frame.drop_duplicates(["T", "K"], keep="last")

    t = frame["T"].to_numpy(dtype=float)
    k = frame["K"].to_numpy(dtype=float)
    iv = frame["implied_vol"].to_numpy(dtype=float)

    t_min, t_max = float(t.min()), float(t.max())
    k_min, k_max = float(k.min()), float(k.max())
    t_scale = max(t_max - t_min, 1e-12)
    k_scale = max(k_max - k_min, 1e-12)

    points_norm = np.column_stack(((t - t_min) / t_scale, (k - k_min) / k_scale))
    interp = LinearNDInterpolator(points_norm, iv, fill_value=np.nan)

    t_grid = np.linspace(t_min, t_max, 90)
    k_grid = np.linspace(k_min, k_max, 120)
    tt, kk = np.meshgrid(t_grid, k_grid, indexing="ij")
    query_norm = np.column_stack(
        ((tt.ravel() - t_min) / t_scale, (kk.ravel() - k_min) / k_scale)
    )
    zz = np.asarray(interp(query_norm), dtype=float).reshape(tt.shape)
    zz = np.ma.masked_invalid(zz)

    finite_iv = np.asarray(zz.compressed(), dtype=float)
    if finite_iv.size == 0:
        raise RuntimeError("Linear IV interpolation produced no finite grid values.")

    # Use the market-surface range for BOTH the surface and Chebyshev nodes.
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

    # CC / Chebyshev nodes: same palette and same normalization as the surface.
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

    ret = (100.0 * np.log(df[close_col] / df[close_col].shift(1))).dropna()
    mu, sd = float(ret.mean()), float(ret.std(ddof=1))
    _, sh_p = shapiro(ret.to_numpy()) if len(ret) <= 5000 else (np.nan, np.nan)
    jb = jarque_bera(ret.to_numpy())
    _, dag_p = normaltest(ret.to_numpy())

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
    ax0.text(
        0.97,
        0.97,
        f"Shapiro-Wilk p = {sh_p:.3g}\nJarque-Bera p = {jb.pvalue:.3g}\nD'Agostino K² p = {dag_p:.3g}\nn = {len(ret)}",
        transform=ax0.transAxes,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round", alpha=0.15),
    )

    ax1 = fig.add_subplot(gs[0, 1])
    (osm, osr), (slope, intercept, _) = probplot(ret.to_numpy(), dist="norm")
    ax1.scatter(osm, osr, s=16)
    qline = np.array([np.min(osm), np.max(osm)])
    ax1.plot(qline, slope * qline + intercept)
    ax1.set_title("Normal Q-Q plot of daily GLD log returns")
    ax1.set_xlabel("Theoretical quantiles")
    ax1.set_ylabel("Sample quantiles")
    ax1.grid(True, alpha=0.25)

    fig.tight_layout()
    save_png(fig, out)
    plt.close(fig)

    return {
        "n": int(len(ret)),
        "mean": mu,
        "std": sd,
        "shapiro_p": float(sh_p),
        "jarque_bera_p": float(jb.pvalue),
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
    s[0], v[0] = spot, max(v0, 1e-8)

    for t in range(n_steps):
        z1, z2 = rng.standard_normal(n_paths), rng.standard_normal(n_paths)
        zs = rho * z1 + np.sqrt(max(1.0 - rho**2, 0.0)) * z2
        v[t + 1] = _vector_full_trunc_cir(v[t], kappa, theta, xi, dt, z1)
        vt = np.maximum(v[t], 0.0)
        s[t + 1] = s[t] * np.exp((rate - 0.5 * vt) * dt + np.sqrt(vt * dt) * zs)
    return {"S": s, "V": v}


def simulate_bates(spot, v0, kappa, theta, xi, rho, lambd, mu_j, sigma_j, rate, years, n_steps, n_paths, seed):
    rng = np.random.default_rng(seed)
    dt = years / n_steps
    s = np.empty((n_steps + 1, n_paths), dtype=float)
    v = np.empty((n_steps + 1, n_paths), dtype=float)
    n_cum = np.empty((n_steps + 1, n_paths), dtype=float)
    lam_path = np.full((n_steps + 1, n_paths), float(lambd), dtype=float)
    s[0], v[0], n_cum[0] = spot, max(v0, 1e-8), 0.0
    k_jump = np.exp(mu_j + 0.5 * sigma_j**2) - 1.0

    for t in range(n_steps):
        z1, z2 = rng.standard_normal(n_paths), rng.standard_normal(n_paths)
        zs = rho * z1 + np.sqrt(max(1.0 - rho**2, 0.0)) * z2
        v[t + 1] = _vector_full_trunc_cir(v[t], kappa, theta, xi, dt, z1)
        vt = np.maximum(v[t], 0.0)
        n = rng.poisson(lam=lambd * dt, size=n_paths)
        jump_log = np.where(
            n > 0,
            n * mu_j + np.sqrt(np.maximum(n, 0.0)) * sigma_j * rng.standard_normal(n_paths),
            0.0,
        )
        s[t + 1] = s[t] * np.exp((rate - lambd * k_jump - 0.5 * vt) * dt + np.sqrt(vt * dt) * zs + jump_log)
        n_cum[t + 1] = n_cum[t] + n
    return {"S": s, "V": v, "N": n_cum, "Lambda": lam_path}


def simulate_bates_hawkes(
    spot, v0, kappa, theta, xi, rho, lambda0, lambda_bar, branching_ratio,
    beta, mu_j, sigma_j, rate, years, n_steps, n_paths, seed,
):
    rng = np.random.default_rng(seed)
    dt = years / n_steps
    alpha = branching_ratio * beta
    s = np.empty((n_steps + 1, n_paths), dtype=float)
    v = np.empty((n_steps + 1, n_paths), dtype=float)
    lam = np.empty((n_steps + 1, n_paths), dtype=float)
    n_cum = np.empty((n_steps + 1, n_paths), dtype=float)
    s[0], v[0], lam[0], n_cum[0] = spot, max(v0, 1e-8), max(lambda0, 1e-10), 0.0
    k_jump = np.exp(mu_j + 0.5 * sigma_j**2) - 1.0

    for t in range(n_steps):
        z1, z2 = rng.standard_normal(n_paths), rng.standard_normal(n_paths)
        zs = rho * z1 + np.sqrt(max(1.0 - rho**2, 0.0)) * z2
        v[t + 1] = _vector_full_trunc_cir(v[t], kappa, theta, xi, dt, z1)
        vt = np.maximum(v[t], 0.0)
        lam_curr = np.maximum(lam[t], 1e-10)
        n = rng.poisson(lam=lam_curr * dt, size=n_paths)
        jump_log = np.where(
            n > 0,
            n * mu_j + np.sqrt(np.maximum(n, 0.0)) * sigma_j * rng.standard_normal(n_paths),
            0.0,
        )
        s[t + 1] = s[t] * np.exp((rate - lam_curr * k_jump - 0.5 * vt) * dt + np.sqrt(vt * dt) * zs + jump_log)
        lam[t + 1] = np.maximum(lam[t] + beta * (lambda_bar - lam[t]) * dt + alpha * n, 1e-10)
        n_cum[t + 1] = n_cum[t] + n
    return {"S": s, "V": v, "N": n_cum, "Lambda": lam}


def path_summary(sim: dict[str, np.ndarray]) -> dict[str, Any]:
    s = sim["S"]
    terminal = s[-1]
    simple_ret = 100.0 * (terminal / s[0, 0] - 1.0)
    return {
        "terminal_mean": float(np.mean(terminal)),
        "terminal_median": float(np.median(terminal)),
        "terminal_std": float(np.std(terminal, ddof=1)),
        "return_mean_pct": float(np.mean(simple_ret)),
        "return_median_pct": float(np.median(simple_ret)),
        "return_std_pct": float(np.std(simple_ret, ddof=1)),
        "terminal_p01": float(np.percentile(terminal, 1)),
        "terminal_p05": float(np.percentile(terminal, 5)),
        "terminal_p50": float(np.percentile(terminal, 50)),
        "terminal_p95": float(np.percentile(terminal, 95)),
        "terminal_p99": float(np.percentile(terminal, 99)),
    }


def plot_single_path_panel(time_grid, values, title, ylabel, out, n_show=5):
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for i in range(min(n_show, values.shape[1])):
        ax.plot(time_grid, values[:, i], alpha=0.95)
    ax.set_title(title)
    ax.set_xlabel("Years")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    save_png(fig, out)
    plt.close(fig)


def plot_path_bands(time_grid, sims: dict[str, dict[str, np.ndarray]], out: Path):
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.8))
    for ax, (name, sim) in zip(axes.ravel(), sims.items()):
        s = sim["S"]
        p10, p25, p50, p75, p90 = [np.percentile(s, q, axis=1) for q in (10, 25, 50, 75, 90)]
        ax.fill_between(time_grid, p10, p90, alpha=0.18, label="10-90 band")
        ax.fill_between(time_grid, p25, p75, alpha=0.30, label="25-75 band")
        ax.plot(time_grid, p50, linewidth=2.0, label="Median")
        ax.set_title(name)
        ax.set_xlabel("Years")
        ax.set_ylabel("Simulated GLD price")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
    fig.suptitle("Five-year GLD path bands by model")
    fig.tight_layout()
    save_png(fig, out)
    plt.close(fig)


def plot_terminal_return_percentiles(sims: dict[str, dict[str, np.ndarray]], out: Path):
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    q = np.arange(0, 101, 1)
    for name, sim in sims.items():
        ret = 100.0 * (sim["S"][-1] / sim["S"][0, 0] - 1.0)
        ax.plot(q, np.percentile(ret, q), label=name)
    ax.set_title("Terminal simple-return percentiles over five-year simulations")
    ax.set_xlabel("Percentile")
    ax.set_ylabel("Terminal simple return (%)")
    ax.set_yscale("symlog", linthresh=5.0)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    save_png(fig, out)
    plt.close(fig)


def plot_volatility_states(time_grid, sims: dict[str, dict[str, np.ndarray]], out: Path, n_show=5):
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.6), sharey=True)
    for ax, name in zip(axes, ["Heston", "Bates", "Bates-Hawkes"]):
        v = sims[name]["V"]
        for i in range(min(n_show, v.shape[1])):
            ax.plot(time_grid, v[:, i], alpha=0.95)
        ax.set_title(name)
        ax.set_xlabel("Years")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Variance state")
    fig.suptitle("Stochastic-volatility paths on a common scale")
    fig.tight_layout()
    save_png(fig, out)
    plt.close(fig)


def plot_hawkes_jump_states(time_grid, sim: dict[str, np.ndarray], out: Path, n_show=5):
    fig, axes = plt.subplots(2, 1, figsize=(8.6, 6.6), sharex=True)
    for i in range(min(n_show, sim["Lambda"].shape[1])):
        axes[0].plot(time_grid, sim["Lambda"][:, i], alpha=0.95)
        axes[1].plot(time_grid, sim["N"][:, i], alpha=0.95)
    axes[0].set_title("Bates-Hawkes intensity paths")
    axes[0].set_ylabel("Intensity")
    axes[0].grid(True, alpha=0.25)
    axes[1].set_title("Bates-Hawkes cumulative jump counts")
    axes[1].set_xlabel("Years")
    axes[1].set_ylabel("Cumulative count")
    axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    save_png(fig, out)
    plt.close(fig)


def plot_bates_jump_states(time_grid, sim: dict[str, np.ndarray], out: Path, n_show=5):
    fig, axes = plt.subplots(2, 1, figsize=(8.6, 6.6), sharex=True)
    for i in range(min(n_show, sim["Lambda"].shape[1])):
        axes[0].plot(time_grid, sim["Lambda"][:, i], alpha=0.95)
        axes[1].plot(time_grid, sim["N"][:, i], alpha=0.95)
    axes[0].set_title("Bates constant Poisson intensity")
    axes[0].set_ylabel("Intensity")
    axes[0].grid(True, alpha=0.25)
    axes[1].set_title("Bates cumulative jump counts")
    axes[1].set_xlabel("Years")
    axes[1].set_ylabel("Cumulative count")
    axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    save_png(fig, out)
    plt.close(fig)


def infer_constant_rate(surface: pd.DataFrame) -> float:
    if "rate" in surface.columns and surface["rate"].notna().any():
        return float(pd.to_numeric(surface["rate"], errors="coerce").dropna().median())
    return 0.04


def infer_spot(surface: pd.DataFrame) -> float:
    if "spot" in surface.columns and surface["spot"].notna().any():
        return float(pd.to_numeric(surface["spot"], errors="coerce").dropna().median())
    raise ValueError("Surface has no valid spot column.")


def build_simulations(sample_cc, model_payloads, years, n_steps, n_paths, seed):
    spot = infer_spot(sample_cc)
    rate = infer_constant_rate(sample_cc)
    time_grid = np.linspace(0.0, years, n_steps + 1)

    bs, heston, bates, hawkes = (
        model_payloads["black_scholes"],
        model_payloads["heston"],
        model_payloads["bates"],
        model_payloads["bates_hawkes"],
    )

    sims = {
        "Black-Scholes": simulate_black_scholes(
            spot, get_param(bs, "sigma"), rate, years, n_steps, n_paths, seed + 10
        ),
        "Heston": simulate_heston(
            spot,
            get_param(heston, "v0"), get_param(heston, "kappa"), get_param(heston, "theta"),
            get_param(heston, "xi", "sigma"), get_param(heston, "rho"), rate,
            years, n_steps, n_paths, seed + 20,
        ),
        "Bates": simulate_bates(
            spot,
            get_param(bates, "v0"), get_param(bates, "kappa"), get_param(bates, "theta"),
            get_param(bates, "xi", "sigma"), get_param(bates, "rho"),
            get_param(bates, "lambd", "lambda", "lambda_J"),
            get_param(bates, "mu_J"), get_param(bates, "sigma_J"), rate,
            years, n_steps, n_paths, seed + 30,
        ),
        "Bates-Hawkes": simulate_bates_hawkes(
            spot,
            get_param(hawkes, "v0"), get_param(hawkes, "kappa"), get_param(hawkes, "theta"),
            get_param(hawkes, "xi", "sigma"), get_param(hawkes, "rho"),
            get_param(hawkes, "lambda0"), get_param(hawkes, "lambda_bar"),
            get_param(hawkes, "branching_ratio"), get_param(hawkes, "beta"),
            get_param(hawkes, "mu_J"), get_param(hawkes, "sigma_J"), rate,
            years, n_steps, n_paths, seed + 40,
        ),
    }

    rows = []
    for name, sim in sims.items():
        row = {"model": name, "spot0": spot, "rate": rate, "years": years, "n_steps": n_steps, "n_paths": n_paths}
        row.update(path_summary(sim))
        rows.append(row)
    return time_grid, sims, pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
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

    # Fail early if Windows cannot write to the target directory.
    probe = out_dir / "_write_test.tmp"
    try:
        probe.write_bytes(b"ok")
        probe.unlink()
    except OSError as exc:
        raise OSError(f"Output directory is not writable: {out_dir}. Original error: {exc}") from exc

    rates_path = repo / "data" / "processed" / "usd_treasury_history.csv"
    gld_path = repo / "data" / "processed" / "gld_daily_history.csv"
    full_surface_path = repo / "data" / "processed" / "full_surfaces" / f"GLD_{date}_eligible_full_surface.csv"
    sample_path = repo / "outputs" / "sampling" / date / f"sample_{strategy}_64.csv"
    calib_dir = repo / "outputs" / "calibrations" / strategy / date

    for path in [rates_path, gld_path, full_surface_path, sample_path, calib_dir]:
        if not path.exists():
            raise FileNotFoundError(path)

    full_surface = load_surface(full_surface_path)
    sample_cc = load_surface(sample_path)

    if args.min_dte < 1:
        raise ValueError("--min-dte must be at least 1 day.")

    full_dte = (
        pd.to_numeric(full_surface["dte"], errors="coerce")
        if "dte" in full_surface.columns
        else 365.25 * full_surface["T"]
    )
    full_surface = full_surface.loc[full_dte.ge(float(args.min_dte))].copy()
    full_surface = full_surface.sort_values(["T", "K"]).reset_index(drop=True)

    sample_dte = (
        pd.to_numeric(sample_cc["dte"], errors="coerce")
        if "dte" in sample_cc.columns
        else 365.25 * sample_cc["T"]
    )
    if sample_dte.lt(float(args.min_dte)).any() or sample_dte.isna().any():
        raise ValueError(
            f"The selected {strategy} sample contains observations below the official "
            f"DTE >= {args.min_dte} day domain. Regenerate sampling first."
        )

    model_payloads = read_model_params(calib_dir)

    manifest = {
        "date": date,
        "strategy": strategy,
        "repo_root": str(repo),
        "full_surface_path": str(full_surface_path),
        "sample_path": str(sample_path),
        "calibration_dir": str(calib_dir),
        "out_dir": str(out_dir),
        "rate_curve_model": "Nelson-Siegel-Svensson",
        "rate_fit_target": "continuous_rate",
        "official_min_dte": int(args.min_dte),
        "iv_colormap": IV_CMAP,
        "cc_nodes_share_surface_colormap": True,
    }

    plot_treasury_curve(rates_path, pd.Timestamp(date), out_dir / "usd_treasury_curve.png")
    plot_sampling_comparison(full_surface, sample_cc, out_dir / "sampling_comparison.png")
    plot_surface_3d(full_surface, sample_cc, out_dir / "volatility_surface_3d.png")
    normality_stats = plot_return_normality(gld_path, out_dir / "gld_return_normality.png")

    time_grid, sims, summary_df = build_simulations(
        sample_cc, model_payloads, float(args.years), int(args.n_steps), int(args.n_paths), int(args.seed)
    )

    plot_single_path_panel(time_grid, sims["Black-Scholes"]["S"], "Five sample Black-Scholes paths", "GLD price", out_dir / "paths_black_scholes_5.png")
    plot_single_path_panel(time_grid, sims["Heston"]["S"], "Five sample Heston paths", "GLD price", out_dir / "paths_heston_5.png")
    plot_single_path_panel(time_grid, sims["Bates"]["S"], "Five sample Bates paths", "GLD price", out_dir / "paths_bates_5.png")
    plot_single_path_panel(time_grid, sims["Bates-Hawkes"]["S"], "Five sample Bates-Hawkes paths", "GLD price", out_dir / "paths_bates_hawkes_5.png")
    plot_path_bands(time_grid, sims, out_dir / "gold_path_stats_by_model.png")
    plot_terminal_return_percentiles(sims, out_dir / "terminal_return_percentiles.png")
    plot_volatility_states(time_grid, sims, out_dir / "volatility_state_paths.png")
    plot_hawkes_jump_states(time_grid, sims["Bates-Hawkes"], out_dir / "hawkes_jump_paths.png")
    plot_bates_jump_states(time_grid, sims["Bates"], out_dir / "bates_poisson_jump_paths.png")

    summary_df.to_csv(out_dir / "terminal_path_stats.csv", index=False)
    (out_dir / "normality_stats.json").write_text(json.dumps(normality_stats, indent=2), encoding="utf-8")
    (out_dir / "figure_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("=" * 92)
    print("[OK] IBKR figures generated")
    print(f"[OK] Output directory : {out_dir}")
    print(f"[OK] Full surface     : {full_surface_path.name} | rows = {len(full_surface)}")
    print(f"[OK] CC sample        : {sample_path.name} | rows = {len(sample_cc)}")
    print(f"[OK] IV palette       : {IV_CMAP} (same normalization for surface and CC nodes)")
    print("=" * 92)


if __name__ == "__main__":
    main()
