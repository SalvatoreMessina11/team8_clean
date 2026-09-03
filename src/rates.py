"""No-look-ahead Treasury curve utilities using Nelson-Siegel-Svensson (NSS).

Team 8 convention
-----------------
The stored Treasury file contains the U.S. Treasury par-yield observations and
the project column ``continuous_rate`` obtained from them.  NSS is fitted to
that continuously-compounded par-yield proxy; this module does NOT claim to
bootstrap a zero-coupon curve.

For every option observation date t:
    1. select the latest Treasury curve date d <= t (no look-ahead);
    2. fit one NSS curve to ``continuous_rate`` across maturity;
    3. evaluate that smooth NSS curve at the option maturities.

The public ``rates_for_date`` API intentionally remains backward-compatible:
by default it returns ``(rates, curve_date)``.  Existing callers such as
surface_builder.py and ibkr_gld_full_date_fetch.py therefore switch to NSS
simply by replacing this file.

Rates are decimals, continuously compounded.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize


REQUIRED_COLUMNS = {
    "date",
    "maturity_years",
    "continuous_rate",
}

# Broad but economically sensible time-scale bounds (years).
TAU1_MIN = 0.03
TAU1_MAX = 10.0
DELTA_MIN = 0.03
DELTA_MAX = 30.0

# Deterministic multi-start grid.  The fit is only two-dimensional because
# beta0..beta3 are solved by linear least squares conditional on tau1,tau2.
_TAU1_STARTS = (0.15, 0.40, 0.80, 1.50, 3.00, 6.00)
_DELTA_STARTS = (0.20, 0.60, 1.50, 4.00, 10.00, 20.00)


@dataclass(frozen=True)
class NSSFit:
    beta0: float
    beta1: float
    beta2: float
    beta3: float
    tau1: float
    tau2: float
    rmse: float
    mae: float
    max_abs_error: float
    n_tenors: int
    success: bool
    objective: float

    @property
    def rmse_bps(self) -> float:
        return 10000.0 * float(self.rmse)

    @property
    def mae_bps(self) -> float:
        return 10000.0 * float(self.mae)

    @property
    def max_abs_error_bps(self) -> float:
        return 10000.0 * float(self.max_abs_error)

    def to_dict(self) -> dict:
        out = asdict(self)
        out["rmse_bps"] = self.rmse_bps
        out["mae_bps"] = self.mae_bps
        out["max_abs_error_bps"] = self.max_abs_error_bps
        out["model"] = "Nelson-Siegel-Svensson"
        out["fit_target"] = "continuous_rate"
        return out


def load_rate_history(
    path: str | Path = "data/processed/usd_treasury_history.csv",
) -> pd.DataFrame:
    """Load and clean the stored Treasury history."""
    path = Path(path)
    frame = pd.read_csv(path)

    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(
            f"Rate history missing columns: {sorted(missing)}. "
            f"Available columns: {list(frame.columns)}"
        )

    frame["date"] = pd.to_datetime(
        frame["date"], errors="coerce"
    ).dt.normalize()

    frame["maturity_years"] = pd.to_numeric(
        frame["maturity_years"], errors="coerce"
    )
    frame["continuous_rate"] = pd.to_numeric(
        frame["continuous_rate"], errors="coerce"
    )

    # Preserve optional source columns when present.
    for col in ("par_yield_pct", "maturity_days"):
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

    frame = frame.dropna(
        subset=["date", "maturity_years", "continuous_rate"]
    ).copy()

    frame = frame.loc[
        frame["maturity_years"].gt(0.0)
        & np.isfinite(frame["continuous_rate"])
    ].copy()

    return (
        frame.sort_values(["date", "maturity_years"])
        .reset_index(drop=True)
    )


def curve_without_lookahead(
    rate_history: pd.DataFrame,
    observation_date,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Return the latest Treasury cross-section with curve_date <= observation_date."""
    date = pd.Timestamp(observation_date).normalize()

    eligible = rate_history.loc[
        pd.to_datetime(rate_history["date"]).le(date)
    ].copy()

    if eligible.empty:
        raise ValueError(
            f"No Treasury curve available on or before {date.date()}"
        )

    curve_date = pd.Timestamp(eligible["date"].max()).normalize()

    curve = eligible.loc[
        pd.to_datetime(eligible["date"]).eq(curve_date)
    ].copy()

    curve = (
        curve.sort_values("maturity_years")
        .drop_duplicates("maturity_years", keep="last")
        .reset_index(drop=True)
    )

    if len(curve) < 6:
        raise ValueError(
            f"Treasury curve on {curve_date.date()} has only "
            f"{len(curve)} tenors; NSS requires at least 6."
        )

    return curve, curve_date


def _validate_maturities(maturities) -> np.ndarray:
    T = np.asarray(maturities, dtype=float)

    if T.ndim == 0:
        T = T.reshape(1)

    if np.any(~np.isfinite(T)) or np.any(T <= 0.0):
        raise ValueError(
            "All requested maturities must be finite and strictly positive."
        )

    return T


def _phi1(x: np.ndarray) -> np.ndarray:
    """Stable (1-exp(-x))/x."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)

    small = np.abs(x) < 1e-6
    xs = x[small]
    # Taylor expansion around zero.
    out[small] = 1.0 - xs / 2.0 + xs**2 / 6.0 - xs**3 / 24.0

    xb = x[~small]
    out[~small] = -np.expm1(-xb) / xb
    return out


def nss_design_matrix(
    maturities,
    tau1: float,
    tau2: float,
) -> np.ndarray:
    """Return the standard 4-column NSS loading matrix."""
    T = _validate_maturities(maturities)
    tau1 = float(tau1)
    tau2 = float(tau2)

    if not np.isfinite(tau1) or tau1 <= 0.0:
        raise ValueError("tau1 must be finite and positive.")
    if not np.isfinite(tau2) or tau2 <= tau1:
        raise ValueError("tau2 must be finite and strictly greater than tau1.")

    x1 = T / tau1
    x2 = T / tau2

    l1 = _phi1(x1)
    l2 = l1 - np.exp(-x1)
    l3 = _phi1(x2) - np.exp(-x2)

    return np.column_stack(
        [
            np.ones_like(T),
            l1,
            l2,
            l3,
        ]
    )


def nss_rates(
    maturities,
    fit: NSSFit | dict,
) -> np.ndarray:
    """Evaluate a fitted NSS continuously-compounded rate curve."""
    if isinstance(fit, NSSFit):
        beta = np.array(
            [fit.beta0, fit.beta1, fit.beta2, fit.beta3],
            dtype=float,
        )
        tau1 = fit.tau1
        tau2 = fit.tau2
    else:
        beta = np.array(
            [
                float(fit["beta0"]),
                float(fit["beta1"]),
                float(fit["beta2"]),
                float(fit["beta3"]),
            ],
            dtype=float,
        )
        tau1 = float(fit["tau1"])
        tau2 = float(fit["tau2"])

    X = nss_design_matrix(maturities, tau1=tau1, tau2=tau2)
    return X @ beta


def _solve_betas(
    maturities: np.ndarray,
    rates: np.ndarray,
    tau1: float,
    tau2: float,
) -> tuple[np.ndarray, np.ndarray]:
    X = nss_design_matrix(maturities, tau1=tau1, tau2=tau2)
    beta, *_ = np.linalg.lstsq(X, rates, rcond=None)
    residual = X @ beta - rates
    return beta, residual


def _tau_objective(
    transformed: np.ndarray,
    maturities: np.ndarray,
    rates: np.ndarray,
) -> float:
    """Variable-projection objective over log(tau1), log(tau2-tau1)."""
    log_tau1, log_delta = np.asarray(transformed, dtype=float)

    tau1 = float(np.exp(log_tau1))
    delta = float(np.exp(log_delta))
    tau2 = tau1 + delta

    try:
        beta, residual = _solve_betas(
            maturities,
            rates,
            tau1=tau1,
            tau2=tau2,
        )
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        return 1e12

    if not np.all(np.isfinite(beta)):
        return 1e12

    # Guard against numerically degenerate decompositions.
    if np.max(np.abs(beta)) > 2.0:
        return 1e8 + float(np.max(np.abs(beta)))

    return float(np.mean(residual**2))


def fit_nss_curve(curve: pd.DataFrame) -> NSSFit:
    """Fit one deterministic NSS curve to a Treasury cross-section.

    The beta parameters are profiled out analytically with linear least
    squares for each pair (tau1, tau2).  Only two nonlinear time-scale
    variables are optimized, which is substantially more stable than fitting
    all six NSS parameters simultaneously.
    """
    required = {"maturity_years", "continuous_rate"}
    missing = required.difference(curve.columns)
    if missing:
        raise ValueError(
            f"Curve missing required columns: {sorted(missing)}"
        )

    x = pd.to_numeric(
        curve["maturity_years"], errors="coerce"
    ).to_numpy(dtype=float)
    y = pd.to_numeric(
        curve["continuous_rate"], errors="coerce"
    ).to_numpy(dtype=float)

    mask = (
        np.isfinite(x)
        & np.isfinite(y)
        & (x > 0.0)
    )

    x = x[mask]
    y = y[mask]

    if len(x) < 6:
        raise ValueError(
            f"NSS fit requires at least 6 valid tenors; got {len(x)}."
        )

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    # Bounds are applied to log(tau1) and log(delta=tau2-tau1).
    bounds = [
        (np.log(TAU1_MIN), np.log(TAU1_MAX)),
        (np.log(DELTA_MIN), np.log(DELTA_MAX)),
    ]

    best = None

    for tau1_start in _TAU1_STARTS:
        if not TAU1_MIN <= tau1_start <= TAU1_MAX:
            continue

        for delta_start in _DELTA_STARTS:
            if not DELTA_MIN <= delta_start <= DELTA_MAX:
                continue

            x0 = np.array(
                [np.log(tau1_start), np.log(delta_start)],
                dtype=float,
            )

            result = minimize(
                _tau_objective,
                x0=x0,
                args=(x, y),
                method="L-BFGS-B",
                bounds=bounds,
                options={
                    "ftol": 1e-18,
                    "gtol": 1e-12,
                    "maxiter": 1000,
                    "maxls": 50,
                },
            )

            value = float(
                _tau_objective(result.x, x, y)
            )

            if best is None or value < best[0]:
                best = (value, result)

    if best is None:
        raise RuntimeError("NSS optimization produced no candidate.")

    objective, result = best

    tau1 = float(np.exp(result.x[0]))
    tau2 = tau1 + float(np.exp(result.x[1]))

    beta, residual = _solve_betas(
        x,
        y,
        tau1=tau1,
        tau2=tau2,
    )

    fitted = y + residual

    if (
        not np.all(np.isfinite(beta))
        or not np.all(np.isfinite(fitted))
        or tau2 <= tau1
    ):
        raise RuntimeError("NSS fit returned non-finite parameters.")

    rmse = float(np.sqrt(np.mean(residual**2)))
    mae = float(np.mean(np.abs(residual)))
    max_abs = float(np.max(np.abs(residual)))

    # A successful numerical candidate is accepted only if the fit is not
    # pathological.  100 bp RMSE is deliberately loose: normal Treasury
    # curves should be far below it, while corrupt inputs fail loudly.
    success = bool(
        np.isfinite(objective)
        and rmse < 0.01
        and tau1 > 0.0
        and tau2 > tau1
    )

    if not success:
        raise RuntimeError(
            "NSS fit failed quality control: "
            f"RMSE={10000.0 * rmse:.3f} bp, "
            f"tau1={tau1:.6f}, tau2={tau2:.6f}."
        )

    return NSSFit(
        beta0=float(beta[0]),
        beta1=float(beta[1]),
        beta2=float(beta[2]),
        beta3=float(beta[3]),
        tau1=tau1,
        tau2=tau2,
        rmse=rmse,
        mae=mae,
        max_abs_error=max_abs,
        n_tenors=int(len(x)),
        success=True,
        objective=float(objective),
    )


def nss_fit_for_date(
    observation_date,
    rate_history: pd.DataFrame | None = None,
    path: str | Path = "data/processed/usd_treasury_history.csv",
) -> tuple[NSSFit, pd.DataFrame, pd.Timestamp]:
    """Return (NSS fit, selected Treasury cross-section, curve date)."""
    if rate_history is None:
        rate_history = load_rate_history(path)

    curve, curve_date = curve_without_lookahead(
        rate_history,
        observation_date,
    )

    fit = fit_nss_curve(curve)
    return fit, curve, curve_date


def interpolate_rates(
    maturities,
    curve: pd.DataFrame,
) -> np.ndarray:
    """Backward-compatible name; now evaluates an NSS fit, not linear interpolation."""
    fit = fit_nss_curve(curve)
    return nss_rates(maturities, fit)


def rates_for_date(
    maturities,
    observation_date,
    rate_history: pd.DataFrame | None = None,
    path: str | Path = "data/processed/usd_treasury_history.csv",
    *,
    return_fit: bool = False,
):
    """Evaluate no-look-ahead NSS rates at requested maturities.

    Default return value (backward compatible):
        rates, curve_date

    With return_fit=True:
        rates, curve_date, fit
    """
    T = _validate_maturities(maturities)

    fit, _, curve_date = nss_fit_for_date(
        observation_date,
        rate_history=rate_history,
        path=path,
    )

    rates = nss_rates(T, fit)

    if np.any(~np.isfinite(rates)):
        raise RuntimeError(
            f"NSS generated non-finite rates for {pd.Timestamp(observation_date).date()}."
        )

    if return_fit:
        return rates, curve_date, fit

    return rates, curve_date
