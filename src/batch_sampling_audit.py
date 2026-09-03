from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from Sampling import Sampling


STRATEGIES = {
    "UU": ("uniform", "uniform"),
    "CU": ("chebyshev", "uniform"),
    "UC": ("uniform", "chebyshev"),
    "CC": ("chebyshev", "chebyshev"),
}


def load_dates(path, start, end):
    df = pd.read_csv(path)
    col = "date" if "date" in df.columns else df.columns[0]
    dates = pd.to_datetime(df[col], errors="coerce").dt.normalize().dropna()
    dates = dates[dates.between(pd.Timestamp(start), pd.Timestamp(end))]
    return list(pd.Series(dates.unique()).sort_values())


def load_surface(path):
    df = pd.read_csv(path)
    for c in ["T", "K", "implied_vol"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["T", "K", "implied_vol"])
    df = df[(df["T"] > 0) & (df["K"] > 0) & (df["implied_vol"] > 0)]
    df = (
        df.drop_duplicates(["T", "K"], keep="last")
        .sort_values(["T", "K"])
        .reset_index(drop=True)
    )
    df["_row_id"] = np.arange(len(df), dtype=int)
    return df


def run_fetch(date, args):
    script = Path(__file__).with_name("ibkr_gld_full_date_fetch.py")
    cmd = [
        sys.executable, str(script),
        "--date", date.strftime("%Y-%m-%d"),
        "--host", args.host,
        "--port", str(args.port),
        "--client-id", str(args.client_id),
        "--stock", args.stock,
        "--rates", args.rates,
        "--output-dir", args.full_surface_dir,
        "--min-moneyness", str(args.min_moneyness),
        "--max-moneyness", str(args.max_moneyness),
        "--min-dte", str(args.min_dte),
        "--max-dte", str(args.max_dte),
        "--min-price", str(args.min_price),
        "--min-iv", str(args.min_iv),
        "--max-iv", str(args.max_iv),
        "--min-vega", str(args.min_vega),
        "--pacing-seconds", str(args.pacing_seconds),
    ]
    return subprocess.run(cmd).returncode == 0


def evaluate_date(date, full, out_dir, n_t, n_k):
    need = n_t * n_k
    if len(full) < need:
        return [], {
            "date": date.strftime("%Y-%m-%d"),
            "status": "skip_lt_64",
            "eligible_points": len(full),
        }

    day_out = Path(out_dir) / date.strftime("%Y-%m-%d")
    day_out.mkdir(parents=True, exist_ok=True)

    rows = []
    for strategy, (ts, ks) in STRATEGIES.items():
        sample = Sampling.sample_hybrid(
            full, t_scheme=ts, k_scheme=ks, n_T=n_t, n_K=n_k
        )
        d = Sampling.interpolation_diagnostics(full, sample)
        sample.to_csv(day_out / f"sample_{strategy}_{need}.csv", index=False)

        rows.append({
            "date": date.strftime("%Y-%m-%d"),
            "strategy": strategy,
            "T_sampling": ts,
            "K_sampling": ks,
            "eligible_points": len(full),
            "holdout_n": d["holdout"]["n"],
            "holdout_mae_bps_iv": 10000.0 * d["holdout"]["mae"],
            "holdout_rmse_bps_iv": 10000.0 * d["holdout"]["rmse"],
            "holdout_linf_bps_iv": d["holdout"]["linf_bps_iv"],
        })

    daily = pd.DataFrame(rows).sort_values(
        ["holdout_linf_bps_iv", "holdout_rmse_bps_iv", "holdout_mae_bps_iv"]
    )
    daily["daily_rank"] = np.arange(1, len(daily) + 1)
    daily.to_csv(day_out / "sampling_comparison.csv", index=False)

    return rows, {
        "date": date.strftime("%Y-%m-%d"),
        "status": "ok",
        "eligible_points": len(full),
    }


def make_summary(rows):
    r = pd.DataFrame(rows)
    if r.empty:
        return r, pd.DataFrame()

    s = (
        r.groupby(["strategy", "T_sampling", "K_sampling"], as_index=False)
        .agg(
            dates_used=("date", "nunique"),
            mean_holdout_linf_bps_iv=("holdout_linf_bps_iv", "mean"),
            median_holdout_linf_bps_iv=("holdout_linf_bps_iv", "median"),
            std_holdout_linf_bps_iv=("holdout_linf_bps_iv", "std"),
            max_holdout_linf_bps_iv=("holdout_linf_bps_iv", "max"),
            mean_holdout_rmse_bps_iv=("holdout_rmse_bps_iv", "mean"),
            mean_holdout_mae_bps_iv=("holdout_mae_bps_iv", "mean"),
        )
    )

    winners = (
        r.sort_values(
            ["date", "holdout_linf_bps_iv", "holdout_rmse_bps_iv", "holdout_mae_bps_iv"]
        )
        .groupby("date", as_index=False)
        .first()
        .groupby("strategy")
        .size()
        .rename("daily_wins")
        .reset_index()
    )

    s = s.merge(winners, on="strategy", how="left")
    s["daily_wins"] = s["daily_wins"].fillna(0).astype(int)
    s = s.sort_values(
        ["mean_holdout_linf_bps_iv", "mean_holdout_rmse_bps_iv", "mean_holdout_mae_bps_iv"]
    ).reset_index(drop=True)
    s["overall_rank"] = np.arange(1, len(s) + 1)
    return r, s


def checkpoint(out, statuses, rows):
    out = Path(out)
    pd.DataFrame(statuses).to_csv(out / "date_status.csv", index=False)
    raw, summary = make_summary(rows)
    if not raw.empty:
        raw.to_csv(out / "all_date_sampling_errors.csv", index=False)
        summary.to_csv(out / "sampling_error_summary.csv", index=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dates", default="data/processed/gld_daily_60_dates.csv")
    p.add_argument("--start", default="2026-07-16")
    p.add_argument("--end", default="2026-09-02")
    p.add_argument("--full-surface-dir", default="data/processed/full_surfaces")
    p.add_argument("--output-dir", default="outputs/sampling_audit_all")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7497)
    p.add_argument("--client-id", type=int, default=94)
    p.add_argument("--stock", default="data/processed/gld_daily_history.csv")
    p.add_argument("--rates", default="data/processed/usd_treasury_history.csv")
    p.add_argument("--min-moneyness", type=float, default=0.80)
    p.add_argument("--max-moneyness", type=float, default=1.20)
    p.add_argument("--min-dte", type=int, default=21)
    p.add_argument("--max-dte", type=int, default=730)
    p.add_argument("--min-price", type=float, default=0.05)
    p.add_argument("--min-iv", type=float, default=0.03)
    p.add_argument("--max-iv", type=float, default=1.50)
    p.add_argument("--min-vega", type=float, default=0.10)
    p.add_argument("--pacing-seconds", type=float, default=0.15)
    p.add_argument("--n-t", type=int, default=8)
    p.add_argument("--n-k", type=int, default=8)
    p.add_argument("--reuse-existing", action="store_true")
    args = p.parse_args()

    dates = load_dates(args.dates, args.start, args.end)
    if not dates:
        raise ValueError("No trading dates found in requested window.")

    full_dir = Path(args.full_surface_dir)
    full_dir.mkdir(parents=True, exist_ok=True)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows, statuses = [], []

    print(f"[INFO] dates: {len(dates)} | {dates[0].date()} -> {dates[-1].date()}")

    for i, date in enumerate(dates, 1):
        slug = date.strftime("%Y-%m-%d")
        surface_path = full_dir / f"GLD_{slug}_eligible_full_surface.csv"

        print("\n" + "=" * 90)
        print(f"[DATE {i}/{len(dates)}] {slug}")
        print("=" * 90)

        if args.reuse_existing and surface_path.exists():
            print(f"[REUSE] {surface_path}")
        else:
            if not run_fetch(date, args):
                statuses.append({"date": slug, "status": "fetch_failed", "eligible_points": np.nan})
                checkpoint(out, statuses, rows)
                continue

        if not surface_path.exists():
            statuses.append({"date": slug, "status": "surface_missing", "eligible_points": np.nan})
            checkpoint(out, statuses, rows)
            continue

        try:
            full = load_surface(surface_path)
            new_rows, status = evaluate_date(
                date, full, out, args.n_t, args.n_k
            )
        except Exception as exc:
            statuses.append({
                "date": slug,
                "status": f"error: {exc}",
                "eligible_points": np.nan,
            })
            checkpoint(out, statuses, rows)
            continue

        rows.extend(new_rows)
        statuses.append(status)

        if new_rows:
            daily = pd.DataFrame(new_rows).sort_values("holdout_linf_bps_iv")
            w = daily.iloc[0]
            print(
                f"[OK] eligible={len(full)} | winner={w['strategy']} | "
                f"L_inf={w['holdout_linf_bps_iv']:.3f} IV bp"
            )
        else:
            print(f"[SKIP] eligible={len(full)} < {args.n_t * args.n_k}")

        checkpoint(out, statuses, rows)

    raw, summary = make_summary(rows)
    checkpoint(out, statuses, rows)

    print("\n" + "=" * 110)
    print("FINAL CROSS-DATE SUMMARY")
    print("=" * 110)

    if summary.empty:
        print("[WARN] No valid 64-node dates.")
        return

    cols = [
        "overall_rank",
        "strategy",
        "T_sampling",
        "K_sampling",
        "dates_used",
        "daily_wins",
        "mean_holdout_linf_bps_iv",
        "median_holdout_linf_bps_iv",
        "mean_holdout_rmse_bps_iv",
        "mean_holdout_mae_bps_iv",
    ]
    print(summary[cols].to_string(index=False))

    best = summary.iloc[0]
    print("=" * 110)
    print(
        f"[OK] OVERALL WINNER: {best['strategy']} | "
        f"mean holdout L_inf={best['mean_holdout_linf_bps_iv']:.3f} IV bp | "
        f"dates={int(best['dates_used'])}"
    )
    print(f"[OK] summary: {out / 'sampling_error_summary.csv'}")


if __name__ == "__main__":
    main()
