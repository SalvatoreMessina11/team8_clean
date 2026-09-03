"""Calibrate Team 8 models directly on an already-selected GLD option surface.

Designed for dense full-surface experiments after node selection with
compare_sampling_all.py.

Key properties
--------------
- Reads the actual selected market observations directly from CSV/Parquet.
- Does NOT rebuild data from options_GLD_daily_60.parquet.
- Saves each model result immediately after that model finishes.
- Resume by default: successful model JSON files are skipped.
- Failed model JSON files are rerun on the next invocation.
- Bates is automatically calibrated/read as a dependency for Full Bates-Hawkes.
- Writes per-date manifest + summary and rebuilds aggregate parameter tables.
- Results are stored under:
      <output-root>/<strategy>/<date>/

Example
-------
python src/calibrate_surface.py ^
  --date 2026-09-02 ^
  --surface outputs/sampling_all_2026-09-02/sample_UU_64.csv ^
  --strategy UU ^
  --profile full ^
  --models bs,heston,bates
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from BnS import BnS
from Heston import Heston
from Bates import Bates
from Hawkes import ExactHawkesCalibration


MODEL_FILES = {
    "bs": "black_scholes.json",
    "heston": "heston.json",
    "bates": "bates.json",
    "hawkes": "full_bates_hawkes.json",
}

MODEL_LABELS = {
    "bs": "Black-Scholes",
    "heston": "Heston",
    "bates": "Bates",
    "hawkes": "Full Bates-Hawkes",
}

ALLOWED_MODELS = tuple(MODEL_FILES)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_write_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, default=_json_default),
        encoding="utf-8",
    )
    tmp.replace(path)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def load_selected_surface(path: str | Path) -> tuple[pd.DataFrame, float, dict[str, Any]]:
    """Load an already-selected calibration surface and validate required data."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Calibration surface not found: {path}")

    if path.suffix.lower() == ".parquet":
        raw = pd.read_parquet(path)
    else:
        raw = pd.read_csv(path)

    required = {"K", "T", "rate", "price", "vega"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(
            f"Selected surface missing required columns: {sorted(missing)}"
        )

    frame = raw.copy()

    numeric = ["K", "T", "rate", "price", "vega"]
    for optional in ["implied_vol", "moneyness", "spot", "dte"]:
        if optional in frame.columns:
            numeric.append(optional)

    for col in numeric:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    frame = frame.dropna(subset=["K", "T", "rate", "price", "vega"]).copy()

    frame = frame.loc[
        frame["K"].gt(0)
        & frame["T"].gt(0)
        & frame["price"].gt(0)
        & frame["vega"].gt(0)
        & np.isfinite(frame["rate"])
    ].copy()

    frame = (
        frame.sort_values(["T", "K"])
        .drop_duplicates(["T", "K"], keep="last")
        .reset_index(drop=True)
    )

    if len(frame) < 8:
        raise ValueError(
            f"Only {len(frame)} valid observations remain; at least 8 are required."
        )

    if "spot" not in frame.columns:
        raise ValueError(
            "Selected surface has no 'spot' column. "
            "Use a sample generated from the full IBKR eligible surface."
        )

    spots = frame["spot"].dropna()
    if spots.empty:
        raise ValueError("Spot column is present but contains no valid values.")

    spot = float(spots.median())
    if not np.isfinite(spot) or spot <= 0:
        raise ValueError(f"Invalid spot value: {spot}")

    expiries = 0
    if "expiry" in frame.columns:
        expiries = int(pd.Series(frame["expiry"]).nunique())

    curve_dates: list[str] = []
    if "curve_date" in frame.columns:
        parsed_curve = pd.to_datetime(frame["curve_date"], errors="coerce")
        curve_dates = sorted(
            {x.strftime("%Y-%m-%d") for x in parsed_curve.dropna()}
        )

    diagnostics = {
        "rows": int(len(frame)),
        "spot": spot,
        "expiries": expiries,
        "unique_strikes": int(frame["K"].nunique()),
        "min_T": float(frame["T"].min()),
        "max_T": float(frame["T"].max()),
        "min_K": float(frame["K"].min()),
        "max_K": float(frame["K"].max()),
        "curve_dates": curve_dates,
    }

    if "implied_vol" in frame.columns and frame["implied_vol"].notna().any():
        diagnostics["min_iv"] = float(frame["implied_vol"].min())
        diagnostics["max_iv"] = float(frame["implied_vol"].max())

    # Models only need these columns, but keeping extra columns in the saved
    # calibration_surface.csv is useful for audit/reproducibility.
    return frame, spot, diagnostics


def calibration_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the columns required by OptionSurface.from_frame plus audit extras."""
    preferred = [
        "K",
        "T",
        "rate",
        "price",
        "vega",
        "implied_vol",
        "moneyness",
        "expiry",
        "dte",
        "spot",
        "curve_date",
        "conId",
        "localSymbol",
        "_row_id",
    ]
    columns = [c for c in preferred if c in frame.columns]
    return frame.loc[:, columns].copy()


def settings(profile: str) -> dict[str, dict[str, Any]]:
    """Preserve the existing Team 8 quick/full numerical profiles."""
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

    if profile == "full":
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

    raise ValueError(f"Unknown profile: {profile}")


def bs_objective(sigma: float, surface: pd.DataFrame, spot: float) -> float:
    sigma = float(sigma)
    if sigma <= 0:
        return 1e8

    errors = []
    for row in surface.itertuples(index=False):
        model = BnS.bs_call_price(
            spot,
            float(row.K),
            float(row.T),
            float(row.rate),
            sigma,
        )
        errors.append(
            (float(model) - float(row.price)) / max(float(row.vega), 1e-4)
        )

    return float(np.mean(np.square(errors)))


def calibrate_bs(surface: pd.DataFrame, spot: float) -> dict[str, Any]:
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


def report_payload(report: Any) -> dict[str, Any]:
    return {
        "model": str(report.model),
        "parameter_names": list(report.parameter_names),
        "parameters": {
            str(k): float(v) for k, v in report.params.items()
        },
        "objective": float(report.objective),
        "global_objective": float(report.global_objective),
        "success": bool(report.success),
        "message": str(report.message),
        "evaluations": int(report.evaluations),
        "iterations": int(report.iterations),
        "elapsed_seconds": float(report.elapsed_seconds),
    }


def hawkes_payload(result: Any, elapsed: float) -> dict[str, Any]:
    names = [
        "v0",
        "kappa",
        "theta",
        "xi",
        "rho",
        "lambda0",
        "lambda_bar",
        "branching_ratio",
        "beta",
        "mu_J",
        "sigma_J",
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


def result_is_successful(path: Path) -> bool:
    if not path.exists():
        return False

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False

    return bool(payload.get("success", False))


def read_result(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def bates_seed_from_payload(payload: dict[str, Any]) -> np.ndarray:
    params = payload.get("parameters", {})

    names = [
        "v0",
        "kappa",
        "theta",
        "sigma",
        "rho",
        "lambd",
        "mu_J",
        "sigma_J",
    ]

    missing = [name for name in names if name not in params]
    if missing:
        raise ValueError(
            f"Stored Bates result missing parameters needed for Hawkes seed: {missing}"
        )

    return np.asarray([float(params[name]) for name in names], dtype=float)


def parse_models(raw: str) -> list[str]:
    requested = [
        item.strip().lower()
        for item in raw.split(",")
        if item.strip()
    ]

    if not requested:
        raise ValueError("No models requested.")

    unknown = [m for m in requested if m not in ALLOWED_MODELS]
    if unknown:
        raise ValueError(
            f"Unknown models: {unknown}. Allowed: {list(ALLOWED_MODELS)}"
        )

    # Keep standard model order irrespective of user input order.
    order = ["bs", "heston", "bates", "hawkes"]
    requested_set = set(requested)

    return [m for m in order if m in requested_set]


def make_failure_payload(
    model: str,
    exc: Exception,
    elapsed: float,
) -> dict[str, Any]:
    return {
        "model": MODEL_LABELS[model],
        "parameter_names": [],
        "parameters": {},
        "objective": None,
        "success": False,
        "message": f"{type(exc).__name__}: {exc}",
        "elapsed_seconds": float(elapsed),
    }


def save_model_result(
    path: Path,
    result: dict[str, Any],
    *,
    date: str,
    strategy: str,
    source_surface: str,
    profile: str,
    seed: int,
    n_points: int,
) -> None:
    payload = dict(result)
    payload.update(
        {
            "date": date,
            "strategy": strategy,
            "source_surface": source_surface,
            "profile": profile,
            "seed": int(seed),
            "n_points": int(n_points),
            "saved_at_utc": utc_now_iso(),
        }
    )
    json_write_atomic(path, payload)


def rebuild_aggregate_tables(output_root: Path) -> None:
    """Rebuild master/long/wide tables by scanning all strategy/date folders."""
    rows = []
    long_rows = []
    wide_rows = []

    for result_path in output_root.glob("*/*/*.json"):
        if result_path.name in {"manifest.json", "surface_diagnostics.json"}:
            continue

        payload = read_result(result_path)
        if not payload or "model" not in payload:
            continue

        row = {
            "date": payload.get("date"),
            "strategy": payload.get("strategy"),
            "model": payload.get("model"),
            "success": payload.get("success"),
            "objective": payload.get("objective"),
            "elapsed_seconds": payload.get("elapsed_seconds"),
            "n_points": payload.get("n_points"),
            "profile": payload.get("profile"),
            "seed": payload.get("seed"),
            "result_file": str(result_path),
        }
        rows.append(row)

        parameters = payload.get("parameters", {}) or {}

        for name, value in parameters.items():
            long_rows.append(
                {
                    "date": payload.get("date"),
                    "strategy": payload.get("strategy"),
                    "model": payload.get("model"),
                    "parameter": name,
                    "value": value,
                }
            )

        wide = dict(row)
        wide.update(parameters)
        wide_rows.append(wide)

    master = pd.DataFrame(rows)
    if not master.empty:
        master = master.sort_values(["date", "strategy", "model"])
    master.to_csv(output_root / "calibration_master.csv", index=False)

    long_df = pd.DataFrame(long_rows)
    if not long_df.empty:
        long_df = long_df.sort_values(
            ["date", "strategy", "model", "parameter"]
        )
    long_df.to_csv(output_root / "parameters_long.csv", index=False)

    wide_df = pd.DataFrame(wide_rows)
    if not wide_df.empty:
        wide_df = wide_df.sort_values(["date", "strategy", "model"])
    wide_df.to_csv(output_root / "parameters_wide.csv", index=False)


def rebuild_date_summary(date_dir: Path) -> None:
    rows = []

    for model in ["bs", "heston", "bates", "hawkes"]:
        path = date_dir / MODEL_FILES[model]
        payload = read_result(path)
        if not payload:
            continue

        row = {
            "date": payload.get("date"),
            "strategy": payload.get("strategy"),
            "model": payload.get("model"),
            "objective": payload.get("objective"),
            "success": payload.get("success"),
            "elapsed_seconds": payload.get("elapsed_seconds"),
        }
        row.update(payload.get("parameters", {}) or {})
        rows.append(row)

    pd.DataFrame(rows).to_csv(
        date_dir / "calibration_summary.csv",
        index=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--date", required=True)
    parser.add_argument("--surface", required=True)
    parser.add_argument("--strategy", required=True)

    parser.add_argument(
        "--profile",
        choices=["quick", "full"],
        default="full",
    )

    parser.add_argument(
        "--models",
        default="bs,heston,bates,hawkes",
        help="Comma-separated subset: bs,heston,bates,hawkes",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--output-root",
        default="outputs/calibrations",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run requested models even when a successful JSON already exists.",
    )

    args = parser.parse_args()

    date_slug = pd.Timestamp(args.date).strftime("%Y-%m-%d")
    strategy = args.strategy.upper()
    source_path = Path(args.surface)
    requested = parse_models(args.models)

    surface_raw, spot, diagnostics = load_selected_surface(source_path)
    surface = calibration_frame(surface_raw)

    output_root = Path(args.output_root)
    date_dir = output_root / strategy / date_slug
    date_dir.mkdir(parents=True, exist_ok=True)

    # Freeze the exact market rows used in this calibration directory.
    frozen_surface_path = date_dir / "calibration_surface.csv"
    surface.to_csv(frozen_surface_path, index=False)

    manifest = {
        "date": date_slug,
        "strategy": strategy,
        "source_surface": str(source_path),
        "source_surface_sha256": file_sha256(source_path),
        "frozen_calibration_surface": str(frozen_surface_path),
        "n_calibration": int(len(surface)),
        "spot": float(spot),
        "curve_dates": diagnostics.get("curve_dates", []),
        "profile": args.profile,
        "seed": int(args.seed),
        "objective": "mean squared price error divided by Vega^2",
        "requested_models": requested,
        "created_or_updated_at_utc": utc_now_iso(),
    }
    json_write_atomic(date_dir / "manifest.json", manifest)
    json_write_atomic(date_dir / "surface_diagnostics.json", diagnostics)

    print("=" * 84)
    print(f"CALIBRATION DATE   : {date_slug}")
    print(f"STRATEGY           : {strategy}")
    print(f"SOURCE             : {source_path}")
    print(f"OBSERVATIONS       : {len(surface)}")
    print(f"SPOT               : {spot:.6f}")
    print(f"PROFILE            : {args.profile}")
    print(f"SEED               : {args.seed}")
    print(f"OUTPUT             : {date_dir}")
    print("=" * 84)

    cfg = settings(args.profile)

    # Bates is a dependency when Hawkes is requested.
    needs_bates_dependency = "hawkes" in requested
    execution_order = list(requested)
    if needs_bates_dependency and "bates" not in execution_order:
        bates_position = execution_order.index("hawkes")
        execution_order.insert(bates_position, "bates")

    for model in execution_order:
        result_path = date_dir / MODEL_FILES[model]

        if not args.force and result_is_successful(result_path):
            print(f"[SKIP] {MODEL_LABELS[model]} already successful: {result_path}")
            continue

        print()
        print(f"[RUN] {MODEL_LABELS[model]}...")

        started = perf_counter()

        try:
            if model == "bs":
                result = calibrate_bs(surface, spot)

            elif model == "heston":
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

            elif model == "bates":
                report = Bates.calibrate_bates(
                    surface,
                    spot,
                    seed=args.seed,
                    pricing="cos",
                    disp=False,
                    return_report=True,
                    **cfg["bates"],
                )
                result = report_payload(report)

            elif model == "hawkes":
                bates_path = date_dir / MODEL_FILES["bates"]
                bates_payload = read_result(bates_path)

                if not bates_payload or not bates_payload.get("success", False):
                    raise RuntimeError(
                        "Full Bates-Hawkes requires a successful Bates result."
                    )

                bates_seed = bates_seed_from_payload(bates_payload)

                hawkes_started = perf_counter()
                raw_result = ExactHawkesCalibration.calibrate_heston(
                    surface,
                    spot,
                    bates_seed=bates_seed,
                    seed=args.seed,
                    min_branching=0.02,
                    **cfg["hawkes"],
                )
                result = hawkes_payload(
                    raw_result,
                    perf_counter() - hawkes_started,
                )

            else:
                raise RuntimeError(f"Unhandled model: {model}")

        except Exception as exc:
            result = make_failure_payload(
                model,
                exc,
                perf_counter() - started,
            )

        save_model_result(
            result_path,
            result,
            date=date_slug,
            strategy=strategy,
            source_surface=str(source_path),
            profile=args.profile,
            seed=args.seed,
            n_points=len(surface),
        )

        if result.get("success", False):
            print(
                f"[OK] {MODEL_LABELS[model]} | "
                f"objective={result.get('objective')} | "
                f"time={result.get('elapsed_seconds', 0.0):.1f}s"
            )
        else:
            print(
                f"[FAIL] {MODEL_LABELS[model]} | "
                f"{result.get('message')}"
            )

        # Save summaries after EVERY model so a long run can be interrupted safely.
        rebuild_date_summary(date_dir)
        rebuild_aggregate_tables(output_root)

    rebuild_date_summary(date_dir)
    rebuild_aggregate_tables(output_root)

    print()
    print("=" * 84)
    print("[DONE] Calibration invocation complete.")
    print(f"[DONE] Date results : {date_dir}")
    print(f"[DONE] Master table : {output_root / 'calibration_master.csv'}")
    print(f"[DONE] Long params  : {output_root / 'parameters_long.csv'}")
    print(f"[DONE] Wide params  : {output_root / 'parameters_wide.csv'}")
    print("=" * 84)


if __name__ == "__main__":
    main()
