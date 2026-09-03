"""Generate the IBKR figures that are present in the old paper but
not yet reproduced in the IBKR LaTeX version, excluding OOS figures.

Target figures (saved under img/diagnostics_ibkr/):
    usd_treasury_curve.png
    sampling_comparison.png
    volatility_surface_3d.png
    gld_return_normality.png
    terminal_return_percentiles.png
    paths_black_scholes_5.png
    paths_heston_5.png
    paths_bates_5.png
    paths_bates_hawkes_5.png
    gold_path_stats_by_model.png
    volatility_state_paths.png
    hawkes_jump_paths.png
    bates_poisson_jump_paths.png

Inputs expected from the Team 8 repository:
    data/processed/usd_treasury_history.csv
    data/processed/gld_daily_history.csv
    data/processed/full_surfaces/GLD_<DATE>_eligible_full_surface.csv
    outputs/sampling_all_<DATE>/sample_CC_64.csv
    outputs/calibrations/CC/<DATE>/
        black_scholes.json
        heston.json
        bates.json
        full_bates_hawkes.json

Example:
    python src\make_missing_ibkr_figures.py --date 2026-09-02
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
from scipy.interpolate import LinearNDInterpolator
from scipy.stats import gaussian_kde, jarque_bera, norm, normaltest, probplot, shapiro

from rates import load_rate_history, curve_without_lookahead, fit_nss_curve, nss_rates


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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


def load_rates_long(path: Path) -> pd.DataFrame:
    """Robust loader for Treasury history.

    Supports either:
    - long format: date, tenor_years, rate
    - wide format: date + tenor columns
    """
    df = pd.read_csv(path)
    date_col = safe_date_column(df)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).copy()

    cols_lower = {c.lower(): c for c in df.columns}
    if {"tenor_years", "rate"}.issubset(cols_lower):
        out = df[[cols_lower["tenor_years"], cols_lower["rate"], date_col]].copy()
        out.columns = ["tenor_years", "rate", "date"]
        out["tenor_years"] = pd.to_numeric(out["tenor_years"], errors="coerce")
        out["rate"] = pd.to_numeric(out["rate"], errors="coerce")
        return out.dropna(subset=["tenor_years", "rate"])

    wide = df.copy()
    non_tenor = {date_col, "date"}
    tenor_rows = []
    for c in wide.columns:
        if c in non_tenor:
            continue
        s = pd.to_numeric(wide[c], errors="coerce")
        if s.notna().sum() == 0:
            continue
        tenor = parse_tenor_label_to_years(c)
        if tenor is None:
            continue
        tmp = pd.DataFrame(
            {
                "date": wide[date_col],
                "tenor_years": tenor,
                "rate": s,
            }
        )
        tenor_rows.append(tmp)

    if not tenor_rows:
        raise ValueError(f"Could not infer tenor columns from {path}")

    out = pd.concat(tenor_rows, ignore_index=True)
    out = out.dropna(subset=["date", "tenor_years", "rate"]).copy()
    return out


def parse_tenor_label_to_years(label: str) -> float | None:
    txt = label.strip().lower().replace("_", " ")
    mapping = {
        "1 mo": 1/12,
        "2 mo": 2/12,
        "3 mo": 3/12,
        "4 mo": 4/12,
        "6 mo": 6/12,
        "1 yr": 1.0,
        "2 yr": 2.0,
        "3 yr": 3.0,
        "5 yr": 5.0,
        "7 yr": 7.0,
        "10 yr": 10.0,
        "20 yr": 20.0,
        "30 yr": 30.0,
        "1m": 1/12,
        "2m": 2/12,
        "3m": 3/12,
        "4m": 4/12,
        "6m": 6/12,
        "1y": 1.0,
        "2y": 2.0,
        "3y": 3.0,
        "5y": 5.0,
        "7y": 7.0,
        "10y": 10.0,
        "20y": 20.0,
        "30y": 30.0,
    }
    if txt in mapping:
        return mapping[txt]
    return None


def nearest_date_at_or_before(series: pd.Series, date: pd.Timestamp) -> pd.Timestamp:
    s = pd.to_datetime(series, errors="coerce").dropna().sort_values().unique()
    s = pd.to_datetime(pd.Series(s))
    s = s[s <= date]
    if len(s) == 0:
        raise ValueError(f"No date in series <= {date.date()}")
    return pd.Timestamp(s.iloc[-1])


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
        out[k] = load_json(p)
    return out


def get_param(payload: dict[str, Any], *names: str, default: float | None = None) -> float:
    params = payload.get("parameters", {}) or {}
    for name in names:
        if name in params:
            return float(params[name])
    if default is not None:
        return float(default)
    raise KeyError(f"None of {names} found in parameters keys {list(params.keys())}")


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
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_sampling_comparison(full_surface: pd.DataFrame, sample_cc: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))

    sc0 = axes[0].scatter(full_surface["T"], full_surface["K"], c=full_surface["implied_vol"], s=12)
    axes[0].set_title("Full eligible surface")
    axes[0].set_xlabel("Maturity T (years)")
    axes[0].set_ylabel("Strike K")
    axes[0].grid(True, alpha=0.25)
    fig.colorbar(sc0, ax=axes[0], label="Implied volatility")

    axes[1].scatter(full_surface["T"], full_surface["K"], s=10, alpha=0.20, label="Full surface")
    sc1 = axes[1].scatter(sample_cc["T"], sample_cc["K"], c=sample_cc["implied_vol"], s=55, marker="o", label="CC 64-node sample")
    axes[1].set_title("Selected CC calibration nodes")
    axes[1].set_xlabel("Maturity T (years)")
    axes[1].set_ylabel("Strike K")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="best")
    fig.colorbar(sc1, ax=axes[1], label="Implied volatility")

    fig.suptitle("Full surface and 64-point Chebyshev-Chebyshev calibration sample")
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_surface_3d(full_surface: pd.DataFrame, sample_cc: pd.DataFrame, out: Path) -> None:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    x = full_surface["K"].to_numpy(float)
    y = full_surface["T"].to_numpy(float)
    z = full_surface["implied_vol"].to_numpy(float)

    fig = plt.figure(figsize=(10.5, 7.2))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True, alpha=0.85)
    ax.scatter(sample_cc["K"], sample_cc["T"], sample_cc["implied_vol"], s=28, color="black")
    ax.set_xlabel("Strike K")
    ax.set_ylabel("Maturity T (years)")
    ax.set_zlabel("Implied volatility")
    ax.set_title("GLD implied-volatility surface with selected CC nodes")
    fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1, label="Implied volatility")
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
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
    fig.savefig(out, dpi=220, bbox_inches="tight")
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
    fig.savefig(out, dpi=220, bbox_inches="tight")
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
    fig.savefig(out, dpi=220, bbox_inches="tight")
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
    fig.savefig(out, dpi=220, bbox_inches="tight")
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
    fig.savefig(out, dpi=220, bbox_inches="tight")
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
    fig.savefig(out, dpi=220, bbox_inches="tight")
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
    fig.savefig(out, dpi=220, bbox_inches="tight")
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
    parser.add_argument("--out-dir", default="img/diagnostics_ibkr")
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    date = pd.Timestamp(args.date).strftime("%Y-%m-%d")
    strategy = args.strategy.upper()

    out_dir = repo / args.out_dir
    ensure_dir(out_dir)

    rates_path = repo / "data" / "processed" / "usd_treasury_history.csv"
    gld_path = repo / "data" / "processed" / "gld_daily_history.csv"
    full_surface_path = repo / "data" / "processed" / "full_surfaces" / f"GLD_{date}_eligible_full_surface.csv"
    sample_path = repo / "outputs" / f"sampling_all_{date}" / f"sample_{strategy}_64.csv"
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
    }

    # Static / descriptive figures
    plot_treasury_curve(rates_path, pd.Timestamp(date), out_dir / "usd_treasury_curve.png")
    plot_sampling_comparison(full_surface, sample_cc, out_dir / "sampling_comparison.png")
    plot_surface_3d(full_surface, sample_cc, out_dir / "volatility_surface_3d.png")
    normality_stats = plot_return_normality(gld_path, out_dir / "gld_return_normality.png")

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

    plot_path_bands(time_grid, sims, out_dir / "gold_path_stats_by_model.png")
    plot_terminal_return_percentiles(sims, out_dir / "terminal_return_percentiles.png")
    plot_volatility_states(time_grid, sims, out_dir / "volatility_state_paths.png")
    plot_hawkes_jump_states(time_grid, sims["Bates-Hawkes"], out_dir / "hawkes_jump_paths.png")
    plot_bates_jump_states(time_grid, sims["Bates"], out_dir / "bates_poisson_jump_paths.png")

    # Useful metadata for the later LaTeX update
    summary_df.to_csv(out_dir / "terminal_path_stats.csv", index=False)
    (out_dir / "normality_stats.json").write_text(json.dumps(normality_stats, indent=2), encoding="utf-8")
    (out_dir / "figure_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("=" * 92)
    print("[OK] Missing IBKR figures generated")
    print(f"[OK] Output directory : {out_dir}")
    print(f"[OK] Full surface     : {full_surface_path.name} | rows = {len(full_surface)}")
    print(f"[OK] CC sample        : {sample_path.name} | rows = {len(sample_cc)}")
    print(f"[OK] Calibration dir  : {calib_dir}")
    print("[OK] Figures:")
    for name in [
        "usd_treasury_curve.png",
        "sampling_comparison.png",
        "volatility_surface_3d.png",
        "gld_return_normality.png",
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
