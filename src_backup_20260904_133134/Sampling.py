"""Sampling strategies for the Team 8 GLD implied-volatility surface."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import Rbf


class Sampling:
    """
    Structured 2-D market-point sampling.

    Supported 8x8 strategies:
        UU = Uniform T, Uniform K
        CU = Chebyshev T, Uniform K
        UC = Uniform T, Chebyshev K
        CC = Chebyshev T, Chebyshev K

    The theoretical grid is never calibrated directly: every grid location is
    mapped to the nearest still-unused ACTUAL market observation.
    """

    @staticmethod
    def chebyshev_roots(n, a, b):
        n = int(n)
        k = np.arange(1, n + 1)
        roots = np.sort(np.cos((2 * k - 1) * np.pi / (2 * n)))
        return 0.5 * (a + b) + 0.5 * (b - a) * roots

    @staticmethod
    def _axis_nodes(series, n, scheme):
        values = pd.to_numeric(series, errors="coerce").dropna()
        if values.empty:
            raise ValueError("Cannot build nodes on an empty axis.")
        a, b = float(values.min()), float(values.max())
        if not np.isfinite(a) or not np.isfinite(b) or b <= a:
            raise ValueError("Sampling axis must have a non-zero finite range.")
        scheme = scheme.lower()
        if scheme == "uniform":
            return np.linspace(a, b, int(n))
        if scheme == "chebyshev":
            return Sampling.chebyshev_roots(int(n), a, b)
        raise ValueError("scheme must be 'uniform' or 'chebyshev'")

    @staticmethod
    def get_nearest_market_points(df, target_T, target_K):
        frame = pd.DataFrame(df).copy().reset_index(drop=True)
        if len(frame) < len(target_T) * len(target_K):
            raise ValueError(
                f"Need at least {len(target_T) * len(target_K)} eligible market "
                f"observations; only {len(frame)} are available."
            )

        if "_row_id" not in frame.columns:
            frame["_row_id"] = np.arange(len(frame), dtype=int)

        t_min, t_max = float(frame["T"].min()), float(frame["T"].max())
        k_min, k_max = float(frame["K"].min()), float(frame["K"].max())
        t_scale = max(t_max - t_min, 1e-12)
        k_scale = max(k_max - k_min, 1e-12)

        selected_indices = set()
        selected = []

        for t_target in np.asarray(target_T, dtype=float):
            for k_target in np.asarray(target_K, dtype=float):
                distance = (
                    ((frame["T"] - t_target) / t_scale) ** 2
                    + ((frame["K"] - k_target) / k_scale) ** 2
                )
                for idx in distance.sort_values(kind="stable").index:
                    if idx not in selected_indices:
                        selected_indices.add(idx)
                        selected.append(idx)
                        break

        sample = frame.loc[selected].copy()
        if len(sample) != len(target_T) * len(target_K):
            raise RuntimeError(
                f"Expected {len(target_T) * len(target_K)} unique nodes, "
                f"obtained {len(sample)}."
            )
        return sample.sort_values(["T", "K"]).reset_index(drop=True)

    @staticmethod
    def sample_hybrid(df, t_scheme, k_scheme, n_T=8, n_K=8):
        target_T = Sampling._axis_nodes(df["T"], n_T, t_scheme)
        target_K = Sampling._axis_nodes(df["K"], n_K, k_scheme)
        return Sampling.get_nearest_market_points(df, target_T, target_K)

    @staticmethod
    def sample_uniform(df, n_T=8, n_K=8):
        return Sampling.sample_hybrid(
            df, "uniform", "uniform", n_T=n_T, n_K=n_K
        )

    @staticmethod
    def sample_chebyshev(df, n_T=8, n_K=8):
        return Sampling.sample_hybrid(
            df, "chebyshev", "chebyshev", n_T=n_T, n_K=n_K
        )

    @staticmethod
    def interpolation_diagnostics(df_full, df_sampled):
        """
        Reconstruct IV via normalized thin-plate RBF.

        Returns:
          - all-surface MAE/RMSE/L_inf, matching the original Team 8 idea;
          - holdout-only MAE/RMSE/L_inf on observations NOT used as nodes.

        Errors are returned in IV decimals, vol points, and IV basis points.
        """
        full = pd.DataFrame(df_full).copy().reset_index(drop=True)
        sample = pd.DataFrame(df_sampled).copy().reset_index(drop=True)

        if "_row_id" not in full.columns:
            full["_row_id"] = np.arange(len(full), dtype=int)
        if "_row_id" not in sample.columns:
            raise ValueError("Sample must preserve _row_id from full surface.")

        t_min, t_max = float(full["T"].min()), float(full["T"].max())
        k_min, k_max = float(full["K"].min()), float(full["K"].max())
        t_scale = max(t_max - t_min, 1e-12)
        k_scale = max(k_max - k_min, 1e-12)

        t_s = (sample["T"].to_numpy(float) - t_min) / t_scale
        k_s = (sample["K"].to_numpy(float) - k_min) / k_scale
        iv_s = sample["implied_vol"].to_numpy(float)

        rbf = Rbf(t_s, k_s, iv_s, function="thin_plate")

        t_f = (full["T"].to_numpy(float) - t_min) / t_scale
        k_f = (full["K"].to_numpy(float) - k_min) / k_scale
        truth = full["implied_vol"].to_numpy(float)
        prediction = np.asarray(rbf(t_f, k_f), dtype=float)
        error = prediction - truth

        sampled_ids = set(sample["_row_id"].astype(int))
        holdout_mask = ~full["_row_id"].astype(int).isin(sampled_ids).to_numpy()

        def metrics(err):
            err = np.asarray(err, dtype=float)
            if err.size == 0:
                return {
                    "n": 0,
                    "mae": np.nan,
                    "rmse": np.nan,
                    "linf": np.nan,
                    "mae_vol_points": np.nan,
                    "rmse_vol_points": np.nan,
                    "linf_vol_points": np.nan,
                    "linf_bps_iv": np.nan,
                }
            absolute = np.abs(err)
            mae = float(np.mean(absolute))
            rmse = float(np.sqrt(np.mean(err ** 2)))
            linf = float(np.max(absolute))
            return {
                "n": int(err.size),
                "mae": mae,
                "rmse": rmse,
                "linf": linf,
                "mae_vol_points": 100.0 * mae,
                "rmse_vol_points": 100.0 * rmse,
                "linf_vol_points": 100.0 * linf,
                "linf_bps_iv": 10000.0 * linf,
            }

        return {
            "all": metrics(error),
            "holdout": metrics(error[holdout_mask]),
        }
