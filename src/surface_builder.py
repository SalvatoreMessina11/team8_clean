"""Build one historical GLD option calibration surface from the IBKR daily panel."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from BnS import BnS
from rates import load_rate_history, rates_for_date


DEFAULT_OPTIONS_PATH = "data/processed/options_GLD_daily_60.parquet"
DEFAULT_RATES_PATH = "data/processed/usd_treasury_history.csv"


def _normalise_panel(frame):
    required = {
        "date", "expiry", "opt_type", "strike", "close",
        "spot", "dte", "T", "moneyness",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Option panel missing columns: {sorted(missing)}")

    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out["expiry"] = pd.to_datetime(out["expiry"], errors="coerce").dt.normalize()

    numeric = ["strike", "close", "spot", "dte", "T", "moneyness"]
    if "volume" in out.columns:
        numeric.append("volume")
    for col in numeric:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(
        subset=["date", "expiry", "strike", "close", "spot", "T", "moneyness"]
    )
    return out


def build_calibration_surface(
    target_date,
    options_path=DEFAULT_OPTIONS_PATH,
    rates_path=DEFAULT_RATES_PATH,
    min_moneyness=0.90,
    max_moneyness=1.10,
    min_dte=90,
    max_dte=320,
    min_price=0.10,
    min_iv=0.03,
    max_iv=1.50,
    min_vega=0.10,
    q=0.0,
):
    """
    Build the actual traded-close surface used for calibration.

    No interpolation of option prices is performed here. Each retained row is
    an actual IBKR historical option observation. The Treasury curve is the
    latest curve available on or before target_date.
    """
    target_date = pd.Timestamp(target_date).normalize()

    panel = pd.read_parquet(Path(options_path))
    panel = _normalise_panel(panel)

    surface = panel.loc[panel["date"].eq(target_date)].copy()
    surface = surface.loc[
        surface["opt_type"].astype(str).str.upper().eq("C")
    ].copy()

    surface = surface.loc[
        surface["close"].gt(float(min_price))
        & surface["moneyness"].between(
            float(min_moneyness), float(max_moneyness)
        )
        & surface["dte"].between(int(min_dte), int(max_dte))
        & surface["T"].gt(0.0)
    ].copy()

    if surface.empty:
        raise ValueError(
            f"No options survive basic filters on {target_date.date()}"
        )

    # Surface spot must be one market close for the calibration date.
    spot = float(surface["spot"].median())
    if not np.isfinite(spot) or spot <= 0.0:
        raise ValueError("Invalid GLD spot")

    rate_history = load_rate_history(rates_path)
    rates, curve_date = rates_for_date(
        surface["T"].to_numpy(dtype=float),
        target_date,
        rate_history=rate_history,
    )
    surface["rate"] = rates

    market_iv = []
    market_vega = []

    for row in surface.itertuples(index=False):
        iv = BnS.implied_vol_call(
            float(row.close),
            spot,
            float(row.strike),
            float(row.T),
            float(row.rate),
            q=q,
        )
        market_iv.append(iv)

        if np.isfinite(iv):
            vega = BnS.calculate_bs_vega(
                spot,
                float(row.strike),
                float(row.T),
                float(row.rate),
                q,
                float(iv),
            )
        else:
            vega = np.nan
        market_vega.append(vega)

    surface["implied_vol"] = market_iv
    surface["vega"] = market_vega

    surface = surface.loc[
        np.isfinite(surface["implied_vol"])
        & np.isfinite(surface["vega"])
        & surface["implied_vol"].between(float(min_iv), float(max_iv))
        & surface["vega"].ge(float(min_vega))
    ].copy()

    if surface.empty:
        raise ValueError(
            f"No options survive IV/Vega checks on {target_date.date()}"
        )

    # Exact duplicate strike/maturity rows are not useful in calibration.
    surface = (
        surface.sort_values(["T", "strike"])
        .drop_duplicates(["expiry", "strike"], keep="last")
        .reset_index(drop=True)
    )

    calibration = pd.DataFrame(
        {
            "K": surface["strike"].to_numpy(dtype=float),
            "T": surface["T"].to_numpy(dtype=float),
            "rate": surface["rate"].to_numpy(dtype=float),
            "price": surface["close"].to_numpy(dtype=float),
            "vega": surface["vega"].to_numpy(dtype=float),
            "implied_vol": surface["implied_vol"].to_numpy(dtype=float),
            "moneyness": surface["moneyness"].to_numpy(dtype=float),
            "expiry": surface["expiry"].dt.strftime("%Y-%m-%d"),
        }
    )

    diagnostics = {
        "date": target_date.strftime("%Y-%m-%d"),
        "curve_date": pd.Timestamp(curve_date).strftime("%Y-%m-%d"),
        "spot": spot,
        "rows": int(len(calibration)),
        "expiries": int(calibration["expiry"].nunique()),
        "strikes": int(calibration["K"].nunique()),
        "min_moneyness": float(calibration["moneyness"].min()),
        "max_moneyness": float(calibration["moneyness"].max()),
        "min_dte": float(calibration["T"].min() * 365.25),
        "max_dte": float(calibration["T"].max() * 365.25),
        "min_iv": float(calibration["implied_vol"].min()),
        "max_iv": float(calibration["implied_vol"].max()),
    }

    if diagnostics["rows"] < 8:
        raise ValueError(
            f"Only {diagnostics['rows']} valid options on {target_date.date()}; "
            "at least 8 are required for this first calibration test."
        )
    if diagnostics["expiries"] < 3:
        raise ValueError(
            f"Only {diagnostics['expiries']} expiries on {target_date.date()}; "
            "at least 3 are required."
        )

    return calibration, spot, diagnostics
