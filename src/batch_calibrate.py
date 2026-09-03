"""Batch wrapper for calibrate_surface.py on dense sampling outputs.

Default expected sample path for each date:
    outputs/sampling/<DATE>/sample_<STRATEGY>_64.csv

Each date is delegated to calibrate_surface.py, which handles resume and
immediate per-model saving.

Example
-------
python src/batch_calibrate.py ^
  --dates 2026-09-01,2026-09-02 ^
  --strategy UU ^
  --models bs,heston,bates ^
  --profile full
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


def parse_dates(raw: str) -> list[str]:
    dates = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        dates.append(pd.Timestamp(item).strftime("%Y-%m-%d"))

    if not dates:
        raise ValueError("No dates supplied.")

    return dates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--dates", required=True)
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--models", default="bs,heston,bates,hawkes")
    parser.add_argument("--profile", choices=["quick", "full"], default="full")
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--min-dte", type=int, default=75)
    parser.add_argument("--output-root", default="outputs/calibrations")

    parser.add_argument(
        "--sampling-template",
        default="outputs/sampling/{date}/sample_{strategy}_64.csv",
        help=(
            "Path template. Available placeholders: {date}, {strategy}. "
            "Default matches compare_sampling_all.py outputs."
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
    )

    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop the batch when one date command exits non-zero.",
    )

    args = parser.parse_args()

    dates = parse_dates(args.dates)
    strategy = args.strategy.upper()

    script = Path(__file__).with_name("calibrate_surface.py")

    if not script.exists():
        raise FileNotFoundError(
            f"Missing companion script: {script}"
        )

    print("=" * 84)
    print(f"BATCH DATES     : {', '.join(dates)}")
    print(f"STRATEGY        : {strategy}")
    print(f"MODELS          : {args.models}")
    print(f"PROFILE         : {args.profile}")
    print(f"SEED            : {args.seed}")
    print(f"MIN DTE         : {args.min_dte} days")
    print("=" * 84)

    failures = []

    for i, date in enumerate(dates, start=1):
        sample = Path(
            args.sampling_template.format(
                date=date,
                strategy=strategy,
            )
        )

        print()
        print("=" * 84)
        print(f"[{i}/{len(dates)}] DATE: {date}")
        print(f"SAMPLE: {sample}")
        print("=" * 84)

        if not sample.exists():
            message = f"sample not found: {sample}"
            print(f"[SKIP] {message}")
            failures.append((date, message))
            if args.stop_on_error:
                break
            continue

        cmd = [
            sys.executable,
            str(script),
            "--date",
            date,
            "--surface",
            str(sample),
            "--strategy",
            strategy,
            "--profile",
            args.profile,
            "--models",
            args.models,
            "--seed",
            str(args.seed),
            "--min-dte",
            str(args.min_dte),
            "--output-root",
            args.output_root,
        ]

        if args.force:
            cmd.append("--force")

        completed = subprocess.run(cmd)

        if completed.returncode != 0:
            message = f"calibrate_surface.py returned {completed.returncode}"
            failures.append((date, message))
            print(f"[FAIL] {date}: {message}")

            if args.stop_on_error:
                break

    print()
    print("=" * 84)

    if failures:
        print("[DONE] Batch completed with skipped/failed dates:")
        for date, message in failures:
            print(f"  {date}: {message}")
    else:
        print("[DONE] Batch completed successfully.")

    print("=" * 84)


if __name__ == "__main__":
    main()
