"""Hawkes utilities and exact Heston-Hawkes option-surface calibration.

Cleaned from the Team 8 research code: only functionality needed by the GLD
rolling OOS pipeline is retained here.
"""

import numpy as np
from scipy.optimize import differential_evolution, minimize

from calibration_core import OptionSurface


class Hawkes:
    @staticmethod
    def branching_ratio(alpha, beta):
        if beta <= 0:
            raise ValueError("beta must be positive")
        return alpha / beta

    @staticmethod
    def stationary_mean_intensity(lambda0, alpha, beta):
        ratio = Hawkes.branching_ratio(alpha, beta)
        if not 0 <= ratio < 1:
            raise ValueError("stationarity requires 0 <= alpha / beta < 1")
        return lambda0 / (1.0 - ratio)

    @staticmethod
    def intensity_on_grid(
        time_grid, event_times, lambda_bar, alpha, beta, initial_intensity=None
    ):
        grid = np.asarray(time_grid, dtype=float)
        events = np.asarray(event_times, dtype=float)
        lambda_initial = lambda_bar if initial_intensity is None else initial_intensity
        intensity = lambda_bar + (lambda_initial - lambda_bar) * np.exp(-beta * grid)
        for event_time in events:
            mask = grid >= event_time
            intensity[mask] += alpha * np.exp(-beta * (grid[mask] - event_time))
        return intensity


class ExactHawkesCalibration:
    """Calibration API for the exact affine Heston-Hawkes pricer."""

    MAX_BRANCHING = 0.98

    @staticmethod
    def _pricer():
        from BatesHawkesExact import BatesHawkesExact
        return BatesHawkesExact

    @staticmethod
    def unpack_heston_params(params):
        (
            v0,
            kappa,
            theta,
            xi,
            rho,
            lambda0,
            lambda_bar,
            branching,
            beta,
            mu_j,
            sigma_j,
        ) = (float(v) for v in params)
        alpha = branching * beta
        return {
            "v0": v0,
            "kappa": kappa,
            "theta": theta,
            "xi": xi,
            "rho": rho,
            "lambda0": lambda0,
            "lambda_bar": lambda_bar,
            "stationary_mean_intensity": lambda_bar / max(1.0 - branching, 1e-12),
            "branching_ratio": branching,
            "alpha": alpha,
            "beta": beta,
            "mu_J": mu_j,
            "sigma_J": sigma_j,
        }

    @staticmethod
    def _feller_gap(kappa, theta, xi):
        return 2.0 * float(kappa) * float(theta) - float(xi)**2

    @classmethod
    def _feller_admissible_diffusion_seed(cls, diffusion_seed, safety=0.98):
        seed = np.asarray(diffusion_seed, dtype=float).copy()
        if seed.shape != (5,):
            raise ValueError("diffusion_seed must contain 5 Heston parameters")
        if cls._feller_gap(seed[1], seed[2], seed[3]) < 0.0:
            seed[3] = safety * np.sqrt(max(2.0 * seed[1] * seed[2], 1e-12))
        return seed

    @classmethod
    def objective_heston(
        cls, params, df_market, S0, q=0.0, n_steps=None, cos_N=192
    ):
        try:
            p = cls.unpack_heston_params(params)
        except (TypeError, ValueError):
            return 1e8

        if p["v0"] <= 0 or p["kappa"] <= 0 or p["theta"] <= 0 or p["xi"] <= 0:
            return 1e8
        if cls._feller_gap(p["kappa"], p["theta"], p["xi"]) < 0:
            return 1e8
        if not -0.999 < p["rho"] < 0.999:
            return 1e8
        if p["lambda0"] <= 0 or p["lambda_bar"] <= 0 or p["beta"] <= 0:
            return 1e8
        if p["sigma_J"] <= 0:
            return 1e8
        if not 0.0 <= p["branching_ratio"] < cls.MAX_BRANCHING:
            return 1e8
        if p["alpha"] >= p["beta"]:
            return 1e8

        surface = OptionSurface.from_frame(df_market)
        pricer = cls._pricer()
        error = 0.0
        try:
            for sl in surface.slices:
                model_prices = pricer.hawkes_price_cos(
                    S0,
                    sl.strikes,
                    sl.maturity,
                    p["v0"],
                    p["kappa"],
                    p["theta"],
                    p["xi"],
                    p["rho"],
                    p["lambda0"],
                    p["lambda_bar"],
                    p["alpha"],
                    p["beta"],
                    p["mu_J"],
                    p["sigma_J"],
                    sl.rate,
                    q,
                    N=cos_N,
                    n_steps=n_steps,
                )
                if not np.all(np.isfinite(model_prices)):
                    return 1e8
                error += float(
                    np.sum(((model_prices - sl.market_prices) / sl.safe_vegas) ** 2)
                )
        except (FloatingPointError, OverflowError, ValueError):
            return 1e8
        return error / surface.size

    @classmethod
    def calibrate_heston(
        cls,
        df_market,
        S0,
        q=0.0,
        bates_seed=None,
        maxiter=25,
        popsize=6,
        n_steps=None,
        seed=None,
        global_cos_N=128,
        local_cos_N=192,
        min_branching=0.02,
    ):
        surface = OptionSurface.from_frame(df_market)
        if bates_seed is None:
            bates_seed = (
                0.07997, 1.71643, 0.04410, 0.71955,
                0.22059, 0.89639, -0.19071, 0.10083,
            )
        bates_seed = np.asarray(bates_seed, dtype=float)
        if bates_seed.shape != (8,):
            raise ValueError("bates_seed must contain 8 Bates parameters")
        if not 0.0 <= min_branching < 0.95:
            raise ValueError("min_branching must lie in [0, 0.95)")

        diffusion_seed = cls._feller_admissible_diffusion_seed(bates_seed[:5])
        jump_seed = bates_seed[5:]
        hawkes_bounds = [
            (1e-3, 5.0),
            (1e-3, 5.0),
            (min_branching, 0.95),
            (0.1, 12.0),
            (-0.5, 0.5),
            (1e-3, 0.6),
        ]

        def conditional_objective(hawkes_params):
            full = np.concatenate([diffusion_seed, hawkes_params])
            return cls.objective_heston(full, surface, S0, q, n_steps, global_cos_N)

        result_global = differential_evolution(
            conditional_objective,
            bounds=hawkes_bounds,
            maxiter=maxiter,
            popsize=popsize,
            tol=1e-3,
            polish=False,
            seed=seed,
        )
        full_bounds = [
            (1e-4, 1.0),
            (0.1, 10.0),
            (1e-4, 1.0),
            (0.01, 8.0),
            (-0.99, 0.99),
            *hawkes_bounds,
        ]
        x0 = np.concatenate([diffusion_seed, result_global.x])
        bates_like = np.concatenate([
            diffusion_seed,
            [
                jump_seed[0],
                jump_seed[0] * (1.0 - min_branching),
                min_branching,
                max(result_global.x[3], 1.0),
                jump_seed[1],
                jump_seed[2],
            ],
        ])
        constraints = ({"type": "ineq", "fun": lambda x: 2*x[1]*x[2] - x[3]**2},)
        candidates = []
        for start in (x0, bates_like):
            candidates.append(
                minimize(
                    cls.objective_heston,
                    x0=start,
                    args=(surface, S0, q, n_steps, local_cos_N),
                    method="SLSQP",
                    bounds=full_bounds,
                    constraints=constraints,
                    options={"ftol": 1e-8, "maxiter": 80, "disp": False},
                )
            )
        return min(candidates, key=lambda result: float(result.fun))
