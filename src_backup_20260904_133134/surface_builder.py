"""Offline builders for historical GLD option surfaces.

This module never connects to IBKR.

It uses the option panel that has already been downloaded, together with the
stored GLD closes and the stored Treasury history.  The purpose of
``--build-all`` is to reconstruct every usable historical cross-section from
the existing local data without making new API requests.

Important: this processing can clean and enrich observations that are already
in the downloaded panel, but it cannot create contracts that were never
downloaded.  No synthetic option observations are added.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from BnS import BnS
from rates import load_rate_history, rates_for_date


DEFAULT_OPTIONS_PATH = "data/processed/options_GLD_daily_60.parquet"
DEFAULT_RATES_PATH = "data/processed/usd_treasury_history.csv"
DEFAULT_OUTPUT_DIR = "data/processed/sparse_historical_surfaces"


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
    for optional in ["volume", "conId"]:
        if optional in out.columns:
            numeric.append(optional)

    for col in numeric:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(
        subset=[
            "date", "expiry", "strike", "close",
            "spot", "dte", "T", "moneyness",
        ]
    ).copy()

    # Recompute these two quantities from dates/spot to avoid carrying stale
    # values if the panel was edited or filtered after the original download.
    out["dte"] = (out["expiry"] - out["date"]).dt.days
    out["T"] = out["dte"] / 365.25
    out["moneyness"] = out["strike"] / out["spot"]
    return out


def load_option_panel(path=DEFAULT_OPTIONS_PATH):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Option panel not found: {path}")
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path)
    return _normalise_panel(frame)


def _eligible_surface_from_panel(
    target_date,
    panel,
    rate_history,
    min_moneyness,
    max_moneyness,
    min_dte,
    max_dte,
    min_price,
    min_iv,
    max_iv,
    min_vega,
    q=0.0,
):
    target_date = pd.Timestamp(target_date).normalize()

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
            f"No downloaded options survive basic filters on {target_date.date()}"
        )

    # A historical option cross-section must use the contemporaneous GLD close.
    spot = float(surface["spot"].median())
    if not np.isfinite(spot) or spot <= 0.0:
        raise ValueError(f"Invalid GLD spot on {target_date.date()}")

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
            f"No downloaded options survive IV/Vega checks on {target_date.date()}"
        )

    surface["K"] = surface["strike"].astype(float)
    surface["price"] = surface["close"].astype(float)
    surface["curve_date"] = pd.Timestamp(curve_date)
    surface["spot"] = spot

    # Duplicated K/T rows do not provide an additional surface node.
    surface = (
        surface.sort_values(["T", "K"])
        .drop_duplicates(["T", "K"], keep="last")
        .reset_index(drop=True)
    )

    preferred = [
        "date", "expiry", "K", "T", "dte", "moneyness", "spot",
        "price", "rate", "curve_date", "implied_vol", "vega",
        "volume", "conId", "localSymbol", "tradingClass",
    ]
    columns = [col for col in preferred if col in surface.columns]
    eligible = surface.loc[:, columns].copy()

    diagnostics = {
        "date": target_date.strftime("%Y-%m-%d"),
        "curve_date": pd.Timestamp(curve_date).strftime("%Y-%m-%d"),
        "spot": spot,
        "rows": int(len(eligible)),
        "expiries": int(eligible["expiry"].nunique()),
        "strikes": int(eligible["K"].nunique()),
        "min_moneyness": float(eligible["moneyness"].min()),
        "max_moneyness": float(eligible["moneyness"].max()),
        "min_dte": float(eligible["dte"].min()),
        "max_dte": float(eligible["dte"].max()),
        "min_iv": float(eligible["implied_vol"].min()),
        "max_iv": float(eligible["implied_vol"].max()),
    }
    return eligible, spot, diagnostics


def build_calibration_surface(
    target_date,
    options_path=DEFAULT_OPTIONS_PATH,
    rates_path=DEFAULT_RATES_PATH,
    min_moneyness=0.90,
    max_moneyness=1.10,
    min_dte=75,
    max_dte=320,
    min_price=0.10,
    min_iv=0.03,
    max_iv=1.50,
    min_vega=0.10,
    q=0.0,
):
    """Build one calibration surface from already-downloaded market rows."""
    panel = load_option_panel(options_path)
    rate_history = load_rate_history(rates_path)

    eligible, spot, diagnostics = _eligible_surface_from_panel(
        target_date,
        panel,
        rate_history,
        min_moneyness,
        max_moneyness,
        min_dte,
        max_dte,
        min_price,
        min_iv,
        max_iv,
        min_vega,
        q=q,
    )

    calibration = pd.DataFrame(
        {
            "K": eligible["K"].to_numpy(dtype=float),
            "T": eligible["T"].to_numpy(dtype=float),
            "rate": eligible["rate"].to_numpy(dtype=float),
            "price": eligible["price"].to_numpy(dtype=float),
            "vega": eligible["vega"].to_numpy(dtype=float),
            "implied_vol": eligible["implied_vol"].to_numpy(dtype=float),
            "moneyness": eligible["moneyness"].to_numpy(dtype=float),
            "expiry": pd.to_datetime(eligible["expiry"]).dt.strftime("%Y-%m-%d"),
        }
    )

    if diagnostics["rows"] < 8:
        raise ValueError(
            f"Only {diagnostics['rows']} valid options on "
            f"{pd.Timestamp(target_date).date()}; at least 8 are required."
        )
    if diagnostics["expiries"] < 3:
        raise ValueError(
            f"Only {diagnostics['expiries']} expiries on "
            f"{pd.Timestamp(target_date).date()}; at least 3 are required."
        )

    return calibration, spot, diagnostics


def build_all_historical_surfaces(
    options_path=DEFAULT_OPTIONS_PATH,
    rates_path=DEFAULT_RATES_PATH,
    output_dir=DEFAULT_OUTPUT_DIR,
    start=None,
    end=None,
    min_moneyness=0.85,
    max_moneyness=1.15,
    min_dte=75,
    max_dte=540,
    min_price=0.10,
    min_iv=0.03,
    max_iv=1.50,
    min_vega=0.10,
    q=0.0,
):
    """
    Rebuild every available historical surface OFFLINE.

    The historical reconstruction uses the same official maturity floor as
    the dense calibration exercise: DTE >= 75 days.  This removes very
    short-dated observations whose implied volatility can be unstable when
    option Vega is very small.  No synthetic rows are created.
    """
    panel = load_option_panel(options_path)
    rate_history = load_rate_history(rates_path)

    dates = pd.DatetimeIndex(sorted(panel["date"].dropna().unique()))
    if start is not None:
        dates = dates[dates >= pd.Timestamp(start).normalize()]
    if end is not None:
        dates = dates[dates <= pd.Timestamp(end).normalize()]

    if len(dates) == 0:
        raise ValueError("No panel dates remain in the requested interval.")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for i, target in enumerate(dates, start=1):
        slug = pd.Timestamp(target).strftime("%Y-%m-%d")
        try:
            eligible, _, diag = _eligible_surface_from_panel(
                target,
                panel,
                rate_history,
                min_moneyness,
                max_moneyness,
                min_dte,
                max_dte,
                min_price,
                min_iv,
                max_iv,
                min_vega,
                q=q,
            )

            path = out / f"GLD_{slug}_eligible_historical_surface.csv"
            eligible.to_csv(path, index=False)

            status = "ok_64plus" if len(eligible) >= 64 else "ok_lt64"
            summary_rows.append({
                **diag,
                "status": status,
                "surface_file": str(path),
            })
            print(
                f"[{i}/{len(dates)}] {slug}: "
                f"{len(eligible)} points, {diag['expiries']} expiries, "
                f"{diag['strikes']} strikes [{status}]"
            )
        except Exception as exc:
            summary_rows.append({
                "date": slug,
                "status": f"error: {exc}",
                "rows": 0,
                "expiries": 0,
                "strikes": 0,
                "surface_file": "",
            })
            print(f"[{i}/{len(dates)}] {slug}: [SKIP] {exc}")

    summary = pd.DataFrame(summary_rows)
    summary_path = out / "historical_surface_summary.csv"
    summary.to_csv(summary_path, index=False)

    ok = summary["status"].astype(str).str.startswith("ok")
    enough = summary["status"].eq("ok_64plus")
    print("=" * 82)
    print(f"[OK] dates processed       : {len(summary)}")
    print(f"[OK] usable surfaces       : {int(ok.sum())}")
    print(f"[OK] surfaces with >=64    : {int(enough.sum())}")
    print(f"[OK] summary               : {summary_path}")
    print("[OK] IBKR calls             : 0")
    print("=" * 82)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--build-all", action="store_true")
    p.add_argument("--date", default=None)
    p.add_argument("--options", default=DEFAULT_OPTIONS_PATH)
    p.add_argument("--rates", default=DEFAULT_RATES_PATH)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)

    p.add_argument("--min-moneyness", type=float, default=0.85)
    p.add_argument("--max-moneyness", type=float, default=1.15)
    p.add_argument("--min-dte", type=int, default=75)
    p.add_argument("--max-dte", type=int, default=540)
    p.add_argument("--min-price", type=float, default=0.10)
    p.add_argument("--min-iv", type=float, default=0.03)
    p.add_argument("--max-iv", type=float, default=1.50)
    p.add_argument("--min-vega", type=float, default=0.10)
    args = p.parse_args()

    if args.build_all:
        build_all_historical_surfaces(
            options_path=args.options,
            rates_path=args.rates,
            output_dir=args.output_dir,
            start=args.start,
            end=args.end,
            min_moneyness=args.min_moneyness,
            max_moneyness=args.max_moneyness,
            min_dte=args.min_dte,
            max_dte=args.max_dte,
            min_price=args.min_price,
            min_iv=args.min_iv,
            max_iv=args.max_iv,
            min_vega=args.min_vega,
        )
        return

    if args.date is None:
        p.error("use --build-all or provide --date YYYY-MM-DD")

    panel = load_option_panel(args.options)
    rates = load_rate_history(args.rates)
    eligible, _, diag = _eligible_surface_from_panel(
        args.date,
        panel,
        rates,
        args.min_moneyness,
        args.max_moneyness,
        args.min_dte,
        args.max_dte,
        args.min_price,
        args.min_iv,
        args.max_iv,
        args.min_vega,
    )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    slug = pd.Timestamp(args.date).strftime("%Y-%m-%d")
    path = out / f"GLD_{slug}_eligible_historical_surface.csv"
    eligible.to_csv(path, index=False)
    print(diag)
    print(f"[OK] surface: {path}")
    print("[OK] IBKR calls: 0")


if __name__ == "__main__":
    main()
