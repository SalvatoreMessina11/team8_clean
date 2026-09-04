"""Rebuild an existing IBKR full GLD surface with NSS Treasury rates OFFLINE.

This script does NOT call IBKR.  It starts from the previously downloaded
market midpoint file:
    data/processed/full_surfaces/GLD_<DATE>_midpoint_raw.csv

It then:
    1. selects the latest Treasury curve with curve_date <= option date;
    2. fits Nelson-Siegel-Svensson to ``continuous_rate``;
    3. evaluates the NSS curve at every option maturity;
    4. restricts the official calibration domain to DTE >= 75 days by default;
    5. recomputes implied volatility and Black-Scholes Vega;
    6. reapplies the IV/Vega eligibility filters;
    7. overwrites the canonical eligible full-surface CSV, after making a
       one-time backup of the pre-DTE-floor version;
    8. writes an NSS fit JSON for reproducibility.

No option price, strike, expiry, conId, or other market observation is created
or altered.

Example
-------
python src\rebuild_full_surface_nss.py --date 2026-09-02
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from BnS import BnS
from rates import load_rate_history, rates_for_date


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _get_spot(raw: pd.DataFrame, stock_path: Path, target: pd.Timestamp) -> float:
    if "spot" in raw.columns:
        spot_values = pd.to_numeric(raw["spot"], errors="coerce").dropna()
        if not spot_values.empty:
            spot = float(spot_values.median())
            if np.isfinite(spot) and spot > 0.0:
                return spot

    stock = _read_csv(stock_path)

    date_col = "timestamp" if "timestamp" in stock.columns else "date"
    if date_col not in stock.columns or "close" not in stock.columns:
        raise ValueError(
            f"Cannot infer GLD spot from {stock_path}; "
            f"columns={list(stock.columns)}"
        )

    stock["_date"] = pd.to_datetime(
        stock[date_col], errors="coerce"
    ).dt.normalize()
    stock["close"] = pd.to_numeric(stock["close"], errors="coerce")

    row = stock.loc[
        stock["_date"].eq(target)
    ].dropna(subset=["close"])

    if row.empty:
        raise ValueError(f"No GLD close found for {target.date()}")

    spot = float(row.iloc[-1]["close"])
    if not np.isfinite(spot) or spot <= 0.0:
        raise ValueError(f"Invalid GLD spot: {spot}")
    return spot


def _normalise_raw(
    raw: pd.DataFrame,
    target: pd.Timestamp,
    spot: float,
) -> pd.DataFrame:
    out = raw.copy()

    required = {"expiry", "K", "price"}
    missing = required.difference(out.columns)
    if missing:
        raise ValueError(
            f"Raw midpoint file missing columns: {sorted(missing)}"
        )

    out["expiry"] = pd.to_datetime(out["expiry"], errors="coerce").dt.normalize()
    out["K"] = pd.to_numeric(out["K"], errors="coerce")
    out["price"] = pd.to_numeric(out["price"], errors="coerce")

    out = out.dropna(subset=["expiry", "K", "price"]).copy()
    out = out.loc[
        out["K"].gt(0.0)
        & out["price"].gt(0.0)
    ].copy()

    out["date"] = target
    out["dte"] = (out["expiry"] - target).dt.days
    out["T"] = out["dte"] / 365.25
    out["moneyness"] = out["K"] / float(spot)
    out["spot"] = float(spot)

    out = out.loc[out["T"].gt(0.0)].copy()

    return (
        out.sort_values(["T", "K"])
        .drop_duplicates(["T", "K"], keep="last")
        .reset_index(drop=True)
    )


def rebuild(
    *,
    target,
    raw_path: Path,
    stock_path: Path,
    rates_path: Path,
    output_dir: Path,
    min_dte: int,
    max_dte: int,
    min_price: float,
    min_iv: float,
    max_iv: float,
    min_vega: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    target = pd.Timestamp(target).normalize()

    raw_original = _read_csv(raw_path)
    spot = _get_spot(raw_original, stock_path, target)
    raw = _normalise_raw(raw_original, target, spot)

    if int(min_dte) < 1 or int(max_dte) < int(min_dte):
        raise ValueError("Require 1 <= min_dte <= max_dte.")

    raw = raw.loc[
        raw["dte"].between(int(min_dte), int(max_dte))
        & raw["price"].gt(float(min_price))
    ].copy()

    if raw.empty:
        raise RuntimeError(
            f"No midpoint observations remain after DTE {min_dte}-{max_dte} "
            "and price-domain filters."
        )

    rate_history = load_rate_history(rates_path)
    rates, curve_date, fit = rates_for_date(
        raw["T"].to_numpy(dtype=float),
        target,
        rate_history=rate_history,
        return_fit=True,
    )

    raw["rate"] = rates
    raw["curve_date"] = pd.Timestamp(curve_date)
    raw["rate_curve_model"] = "NSS"
    raw["nss_fit_target"] = "continuous_rate"

    fit_dict = fit.to_dict()

    # Keep fit parameters inside the surface for full auditability.
    raw["nss_beta0"] = fit.beta0
    raw["nss_beta1"] = fit.beta1
    raw["nss_beta2"] = fit.beta2
    raw["nss_beta3"] = fit.beta3
    raw["nss_tau1"] = fit.tau1
    raw["nss_tau2"] = fit.tau2
    raw["nss_rmse_bps"] = fit.rmse_bps

    ivs: list[float] = []
    vegas: list[float] = []

    for row in raw.itertuples(index=False):
        iv = BnS.implied_vol_call(
            float(row.price),
            float(spot),
            float(row.K),
            float(row.T),
            float(row.rate),
        )
        ivs.append(iv)

        if np.isfinite(iv):
            vega = BnS.calculate_bs_vega(
                float(spot),
                float(row.K),
                float(row.T),
                float(row.rate),
                0.0,
                float(iv),
            )
        else:
            vega = np.nan

        vegas.append(vega)

    raw["implied_vol"] = ivs
    raw["vega"] = vegas

    eligible = raw.loc[
        np.isfinite(raw["implied_vol"])
        & np.isfinite(raw["vega"])
        & raw["implied_vol"].between(float(min_iv), float(max_iv))
        & raw["vega"].ge(float(min_vega))
    ].copy()

    eligible = (
        eligible.sort_values(["T", "K"])
        .drop_duplicates(["T", "K"], keep="last")
        .reset_index(drop=True)
    )

    if eligible.empty:
        raise RuntimeError("No eligible option observations remain after NSS rebuild.")

    fit_payload = {
        "observation_date": target.strftime("%Y-%m-%d"),
        "curve_date": pd.Timestamp(curve_date).strftime("%Y-%m-%d"),
        "rate_curve_model": "Nelson-Siegel-Svensson",
        "fit_target": "continuous_rate",
        "official_min_dte": int(min_dte),
        "max_dte": int(max_dte),
        "note": (
            "NSS is fitted to the project's continuously compounded "
            "Treasury par-yield proxy; this is not a bootstrapped zero curve."
        ),
        **fit_dict,
    }

    return raw, eligible, fit_payload


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--date", required=True)
    p.add_argument(
        "--output-dir",
        default="data/processed/full_surfaces",
    )
    p.add_argument(
        "--raw",
        default=None,
        help=(
            "Optional raw midpoint CSV. Default: "
            "data/processed/full_surfaces/GLD_<DATE>_midpoint_raw.csv"
        ),
    )
    p.add_argument(
        "--stock",
        default="data/processed/gld_daily_history.csv",
    )
    p.add_argument(
        "--rates",
        default="data/processed/usd_treasury_history.csv",
    )
    p.add_argument("--min-dte", type=int, default=75)
    p.add_argument("--max-dte", type=int, default=730)
    p.add_argument("--min-price", type=float, default=0.05)
    p.add_argument("--min-iv", type=float, default=0.03)
    p.add_argument("--max-iv", type=float, default=1.50)
    p.add_argument("--min-vega", type=float, default=0.10)
    args = p.parse_args()

    target = pd.Timestamp(args.date).normalize()
    slug = target.strftime("%Y-%m-%d")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    raw_path = (
        Path(args.raw)
        if args.raw is not None
        else out / f"GLD_{slug}_midpoint_raw.csv"
    )

    canonical = out / f"GLD_{slug}_eligible_adaptive_surface.csv"
    backup = out / f"GLD_{slug}_eligible_adaptive_surface_pre_dte75_backup.csv"
    raw_nss_path = out / f"GLD_{slug}_midpoint_raw_nss.csv"
    fit_path = out / f"GLD_{slug}_nss_curve.json"

    raw_nss, eligible, fit_payload = rebuild(
        target=target,
        raw_path=raw_path,
        stock_path=Path(args.stock),
        rates_path=Path(args.rates),
        output_dir=out,
        min_dte=args.min_dte,
        max_dte=args.max_dte,
        min_price=args.min_price,
        min_iv=args.min_iv,
        max_iv=args.max_iv,
        min_vega=args.min_vega,
    )

    # One-time backup of the old canonical full surface.
    if canonical.exists() and not backup.exists():
        shutil.copy2(canonical, backup)
        print(f"[OK] backup old surface : {backup}")

    raw_nss.to_csv(raw_nss_path, index=False)
    eligible.to_csv(canonical, index=False)
    fit_path.write_text(
        json.dumps(fit_payload, indent=2),
        encoding="utf-8",
    )

    print("=" * 88)
    print(f"[OK] DATE                 : {slug}")
    print(f"[OK] RATE MODEL           : NSS")
    print(f"[OK] DTE DOMAIN           : {args.min_dte} -> {args.max_dte} days")
    print(f"[OK] TREASURY CURVE DATE  : {fit_payload['curve_date']}")
    print(
        f"[OK] NSS TAU              : "
        f"tau1={fit_payload['tau1']:.6f}, "
        f"tau2={fit_payload['tau2']:.6f}"
    )
    print(
        f"[OK] NSS FIT RMSE         : "
        f"{fit_payload['rmse_bps']:.4f} bp"
    )
    print(f"[OK] RAW NSS ROWS         : {len(raw_nss)}")
    print(f"[OK] ELIGIBLE NSS ROWS    : {len(eligible)}")
    print(
        f"[OK] UNIQUE EXPIRIES      : "
        f"{eligible['expiry'].nunique()}"
    )
    print(
        f"[OK] UNIQUE STRIKES       : "
        f"{eligible['K'].nunique()}"
    )
    print(f"[OK] RAW NSS              : {raw_nss_path}")
    print(f"[OK] FULL SURFACE         : {canonical}")
    print(f"[OK] NSS FIT JSON         : {fit_path}")
    print("=" * 88)


if __name__ == "__main__":
    main()
