"""No-look-ahead Treasury curve utilities for the clean Team 8 build."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "date",
    "maturity_years",
    "continuous_rate",
}


def load_rate_history(path="data/processed/usd_treasury_history.csv"):
    path = Path(path)
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Rate history missing columns: {sorted(missing)}")

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["maturity_years"] = pd.to_numeric(
        frame["maturity_years"], errors="coerce"
    )
    frame["continuous_rate"] = pd.to_numeric(
        frame["continuous_rate"], errors="coerce"
    )
    frame = frame.dropna(
        subset=["date", "maturity_years", "continuous_rate"]
    ).copy()
    frame = frame.loc[frame["maturity_years"].gt(0.0)].copy()
    return frame.sort_values(["date", "maturity_years"]).reset_index(drop=True)


def curve_without_lookahead(rate_history, observation_date):
    """
    Return the latest complete Treasury curve with curve_date <= observation_date.
    """
    date = pd.Timestamp(observation_date).normalize()
    eligible = rate_history.loc[rate_history["date"].le(date)].copy()
    if eligible.empty:
        raise ValueError(f"No Treasury curve available on or before {date.date()}")

    curve_date = eligible["date"].max()
    curve = eligible.loc[eligible["date"].eq(curve_date)].copy()
    curve = (
        curve.sort_values("maturity_years")
        .drop_duplicates("maturity_years", keep="last")
        .reset_index(drop=True)
    )
    if len(curve) < 2:
        raise ValueError(f"Treasury curve on {curve_date.date()} has < 2 tenors")
    return curve, curve_date


def interpolate_rates(maturities, curve):
    """
    Linear interpolation in maturity, flat extrapolation at the two ends.
    Rates are continuously compounded, matching usd_treasury_history.csv.
    """
    T = np.asarray(maturities, dtype=float)
    x = curve["maturity_years"].to_numpy(dtype=float)
    y = curve["continuous_rate"].to_numpy(dtype=float)

    if np.any(~np.isfinite(T)) or np.any(T <= 0.0):
        raise ValueError("All requested maturities must be finite and positive")

    return np.interp(T, x, y, left=y[0], right=y[-1])


def rates_for_date(
    maturities,
    observation_date,
    rate_history=None,
    path="data/processed/usd_treasury_history.csv",
):
    if rate_history is None:
        rate_history = load_rate_history(path)
    curve, curve_date = curve_without_lookahead(
        rate_history, observation_date
    )
    return interpolate_rates(maturities, curve), curve_date
