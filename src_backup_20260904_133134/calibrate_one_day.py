"""
Calibrate BS, Heston, Bates and Full Bates-Hawkes on one daily GLD surface.

Start with one date before launching the rolling OOS chain.

Examples
--------
Quick smoke calibration:
    python src/calibrate_one_day.py --date 2026-07-16 --profile quick

More accurate first-date calibration:
    python src/calibrate_one_day.py --date 2026-07-16 --profile full
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from BnS import BnS
from Heston import Heston
from Bates import Bates
from Hawkes import ExactHawkesCalibration
from surface_builder import build_calibration_surface


def bs_objective(sigma, surface, spot):
    sigma = float(sigma)
    if sigma <= 0.0:
        return 1e8

    errors = []
    for row in surface.itertuples(index=False):
        model = BnS.bs_call_price(
            spot, row.K, row.T, row.rate, sigma
        )
        errors.append((model - row.price) / max(row.vega, 1e-4))
    return float(np.mean(np.square(errors)))


def calibrate_bs(surface, spot):
    started = perf_counter()
    result = minimize_scalar(
        bs_objective,
        bounds=(0.01, 2.0),
        args=(surface, spot),
        method="bounded",
        options={"xatol": 1e-8, "maxiter": 500},
    )
    return {
        "model": "Black-Scholes",
        "parameter_names": ["sigma"],
        "parameters": {"sigma": float(result.x)},
        "objective": float(result.fun),
        "success": bool(result.success),
        "message": str(result.message),
        "elapsed_seconds": float(perf_counter() - started),
    }


def report_payload(report):
    return {
        "model": report.model,
        "parameter_names": list(report.parameter_names),
        "parameters": {
            k: float(v) for k, v in report.params.items()
        },
        "objective": float(report.objective),
        "global_objective": float(report.global_objective),
        "success": bool(report.success),
        "message": str(report.message),
        "evaluations": int(report.evaluations),
        "iterations": int(report.iterations),
        "elapsed_seconds": float(report.elapsed_seconds),
    }


def hawkes_payload(result, elapsed):
    names = [
        "v0", "kappa", "theta", "xi", "rho",
        "lambda0", "lambda_bar", "branching_ratio", "beta",
        "mu_J", "sigma_J",
    ]
    params = {
        name: float(value)
        for name, value in zip(names, np.asarray(result.x, dtype=float))
    }
    params["alpha"] = params["branching_ratio"] * params["beta"]
    return {
        "model": "Full Bates-Hawkes",
        "parameter_names": names,
        "parameters": params,
        "objective": float(result.fun),
        "success": bool(result.success),
        "message": str(result.message),
        "elapsed_seconds": float(elapsed),
    }


def settings(profile):
    if profile == "quick":
        return {
            "heston": dict(maxiter=20, popsize=5, cos_N=128),
            "bates": dict(maxiter=15, popsize=5, cos_N=128),
            "hawkes": dict(
                maxiter=8,
                popsize=4,
                global_cos_N=64,
                local_cos_N=96,
            ),
        }
    return {
        "heston": dict(maxiter=60, popsize=8, cos_N=192),
        "bates": dict(maxiter=50, popsize=8, cos_N=192),
        "hawkes": dict(
            maxiter=20,
            popsize=6,
            global_cos_N=96,
            local_cos_N=128,
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="2026-07-16")
    parser.add_argument(
        "--profile",
        choices=["quick", "full"],
        default="quick",
    )
    parser.add_argument(
        "--models",
        default="bs,heston,bates,hawkes",
        help="comma-separated: bs,heston,bates,hawkes",
    )
    parser.add_argument(
        "--options",
        default="data/processed/options_GLD_daily_60.parquet",
    )
    parser.add_argument(
        "--rates",
        default="data/processed/usd_treasury_history.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/calibrations",
    )
    parser.add_argument("--seed", type=int, default=8)
    args = parser.parse_args()

    requested = {
        item.strip().lower()
        for item in args.models.split(",")
        if item.strip()
    }
    allowed = {"bs", "heston", "bates", "hawkes"}
    unknown = requested.difference(allowed)
    if unknown:
        raise ValueError(f"Unknown models: {sorted(unknown)}")

    surface, spot, diagnostics = build_calibration_surface(
        args.date,
        options_path=args.options,
        rates_path=args.rates,
    )

    date_slug = pd.Timestamp(args.date).strftime("%Y-%m-%d")
    out = Path(args.output_dir) / date_slug
    out.mkdir(parents=True, exist_ok=True)

    surface.to_csv(out / "calibration_surface.csv", index=False)
    (out / "surface_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2),
        encoding="utf-8",
    )

    print("=" * 64)
    print(f"CALIBRATION DATE : {date_slug}")
    print(f"SPOT             : {spot:.4f}")
    print(f"TREASURY CURVE   : {diagnostics['curve_date']}")
    print(f"OPTIONS          : {diagnostics['rows']}")
    print(f"EXPIRIES         : {diagnostics['expiries']}")
    print(
        f"MONEYNESS        : "
        f"{diagnostics['min_moneyness']:.4f} -> "
        f"{diagnostics['max_moneyness']:.4f}"
    )
    print(
        f"DTE              : "
        f"{diagnostics['min_dte']:.0f} -> "
        f"{diagnostics['max_dte']:.0f}"
    )
    print("=" * 64)

    cfg = settings(args.profile)
    results = []
    bates_seed = None

    if "bs" in requested:
        print("\n[1] Calibrating Black-Scholes...")
        result = calibrate_bs(surface, spot)
        results.append(result)
        print(
            f"[OK] BS sigma={result['parameters']['sigma']:.6f}, "
            f"objective={result['objective']:.8g}"
        )

    if "heston" in requested:
        print("\n[2] Calibrating Heston...")
        report = Heston.calibrate_heston(
            surface,
            spot,
            seed=args.seed,
            pricing="cos",
            disp=False,
            return_report=True,
            **cfg["heston"],
        )
        result = report_payload(report)
        results.append(result)
        print(
            f"[OK] Heston objective={result['objective']:.8g}, "
            f"time={result['elapsed_seconds']:.1f}s"
        )

    if "bates" in requested or "hawkes" in requested:
        print("\n[3] Calibrating Bates...")
        report = Bates.calibrate_bates(
            surface,
            spot,
            seed=args.seed,
            pricing="cos",
            disp=False,
            return_report=True,
            **cfg["bates"],
        )
        bates_seed = report.x
        result = report_payload(report)
        if "bates" in requested:
            results.append(result)
        print(
            f"[OK] Bates objective={result['objective']:.8g}, "
            f"time={result['elapsed_seconds']:.1f}s"
        )

    if "hawkes" in requested:
        print("\n[4] Calibrating Full Bates-Hawkes...")
        started = perf_counter()
        result_raw = ExactHawkesCalibration.calibrate_heston(
            surface,
            spot,
            bates_seed=bates_seed,
            seed=args.seed,
            min_branching=0.02,
            **cfg["hawkes"],
        )
        result = hawkes_payload(
            result_raw,
            perf_counter() - started,
        )
        results.append(result)
        print(
            f"[OK] Full Bates-Hawkes objective="
            f"{result['objective']:.8g}, "
            f"time={result['elapsed_seconds']:.1f}s"
        )

    summary_rows = []
    for result in results:
        model_slug = (
            result["model"].lower()
            .replace(" ", "_")
            .replace("-", "_")
        )
        (out / f"{model_slug}.json").write_text(
            json.dumps(result, indent=2),
            encoding="utf-8",
        )
        row = {
            "date": date_slug,
            "model": result["model"],
            "objective": result["objective"],
            "success": result["success"],
            "elapsed_seconds": result["elapsed_seconds"],
        }
        row.update(result["parameters"])
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / "calibration_summary.csv", index=False)

    print("\n" + "=" * 64)
    print("[OK] CALIBRATION RUN COMPLETE")
    print(f"[OK] results: {out}")
    print("=" * 64)


if __name__ == "__main__":
    main()
