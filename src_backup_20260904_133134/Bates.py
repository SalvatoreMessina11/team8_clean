"""Bates option pricing and reproducible surface calibration."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.optimize import differential_evolution, minimize

from Heston import Heston
from calibration_core import CalibrationReport, OptionSurface, feller_feasible_population
from fourier_pricing import adaptive_cos_call_prices, carr_madan_call


@dataclass(frozen=True)
class BatesParameters:
    v0: float
    kappa: float
    theta: float
    sigma: float
    rho: float
    lambd: float
    mu_J: float
    sigma_J: float

    @classmethod
    def from_array(cls, values):
        values = tuple(float(v) for v in values)
        if len(values) != 8:
            raise ValueError("Bates requires exactly eight parameters")
        return cls(*values)

    def as_tuple(self):
        return tuple(getattr(self, name) for name in Bates.PARAMETER_NAMES)


class Bates(Heston):
    PARAMETER_NAMES = (
        "v0", "kappa", "theta", "sigma", "rho", "lambd", "mu_J", "sigma_J"
    )
    BOUNDS = (
        (1e-4, 1.0),
        (0.1, 10.0),
        (1e-4, 1.0),
        (0.01, 8.0),
        (-0.99, 0.99),
        (0.0, 5.0),
        (-0.5, 0.5),
        (1e-4, 0.5),
    )
    INVALID_OBJECTIVE = 1e8

    @staticmethod
    def bates_charfunc(
        u, S0, v0, kappa, theta, sigma, rho, lambd, mu_J, sigma_J, T, r, q=0.0
    ):
        cf_heston = Heston.heston_charfunc(
            u, S0, v0, kappa, theta, sigma, rho, T, r, q
        )
        omega = -lambd * (np.exp(mu_J + 0.5 * sigma_J**2) - 1.0)
        jump_term = lambd * (
            np.exp(1j * u * mu_J - 0.5 * u**2 * sigma_J**2) - 1.0
        )
        return cf_heston * np.exp((1j * u * omega + jump_term) * T)

    @staticmethod
    def _charfunc_for(S0, T, r, q, parameters):
        return lambda u: Bates.bates_charfunc(u, S0, *parameters.as_tuple(), T, r, q)

    @staticmethod
    def bates_price_fast(
        S0, K, T, v0, kappa, theta, sigma, rho, lambd, mu_J, sigma_J, r, q=0.0
    ):
        p = BatesParameters(v0, kappa, theta, sigma, rho, lambd, mu_J, sigma_J)
        return carr_madan_call(Bates._charfunc_for(S0, T, r, q, p), S0, K, T, r, q)

    @staticmethod
    def bates_prices_cos(
        S0, K, T, v0, kappa, theta, sigma, rho, lambd, mu_J, sigma_J,
        r, q=0.0, N=256, L=12.0
    ):
        strikes = np.atleast_1d(np.asarray(K, dtype=float))
        if T <= 0:
            return np.maximum(S0 - strikes, 0.0)
        p = BatesParameters(v0, kappa, theta, sigma, rho, lambd, mu_J, sigma_J)
        return adaptive_cos_call_prices(
            Bates._charfunc_for(S0, T, r, q, p),
            strikes,
            T,
            r,
            terms=N,
            width_scale=L,
        )

    @staticmethod
    def _admissible(p):
        return (
            p.v0 > 0
            and p.kappa > 0
            and p.theta > 0
            and p.sigma > 0
            and -0.999 <= p.rho <= 0.999
            and 2.0 * p.kappa * p.theta - p.sigma**2 >= 0.0
            and p.lambd >= 0
            and p.sigma_J > 0
        )

    @staticmethod
    def bates_objective(params, df_market, S0, q=0.0, pricing="cos", cos_N=256):
        try:
            p = BatesParameters.from_array(params)
        except (TypeError, ValueError):
            return Bates.INVALID_OBJECTIVE
        if not Bates._admissible(p):
            return Bates.INVALID_OBJECTIVE

        surface = OptionSurface.from_frame(df_market)
        error = 0.0
        try:
            for sl in surface.slices:
                if pricing == "cos":
                    model = Bates.bates_prices_cos(
                        S0, sl.strikes, sl.maturity, *p.as_tuple(), sl.rate, q, N=cos_N
                    )
                elif pricing == "quad":
                    model = np.asarray([
                        Bates.bates_price_fast(
                            S0, k, sl.maturity, *p.as_tuple(), sl.rate, q
                        ) for k in sl.strikes
                    ])
                else:
                    raise ValueError("pricing must be 'cos' or 'quad'")
                if not np.all(np.isfinite(model)):
                    return Bates.INVALID_OBJECTIVE
                error += float(np.sum(((model - sl.market_prices) / sl.safe_vegas) ** 2))
        except (FloatingPointError, OverflowError, ValueError):
            return Bates.INVALID_OBJECTIVE
        return error / surface.size

    @staticmethod
    def calibrate_bates(
        df_market,
        S0,
        q=0.0,
        maxiter=80,
        popsize=8,
        seed=None,
        pricing="cos",
        cos_N=256,
        disp=False,
        return_report=False,
    ):
        started_at = perf_counter()
        surface = OptionSurface.from_frame(df_market)
        args = (surface, S0, q, pricing, cos_N)
        init = feller_feasible_population(Bates.BOUNDS, popsize, seed)
        global_result = differential_evolution(
            Bates.bates_objective,
            bounds=Bates.BOUNDS,
            args=args,
            maxiter=maxiter,
            popsize=popsize,
            tol=1e-3,
            polish=False,
            seed=seed,
            init=init,
            disp=disp,
        )
        constraints = ({"type": "ineq", "fun": lambda x: 2*x[1]*x[2] - x[3]**2},)
        local_result = minimize(
            Bates.bates_objective,
            x0=global_result.x,
            args=args,
            method="SLSQP",
            bounds=Bates.BOUNDS,
            constraints=constraints,
            options={"ftol": 1e-6, "maxiter": 100, "disp": disp},
        )
        report = CalibrationReport.from_optimizer(
            "Bates", Bates.PARAMETER_NAMES, local_result, global_result, started_at
        )
        return report if return_report else report.x
