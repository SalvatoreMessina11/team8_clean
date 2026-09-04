"""Heston pricing and self-contained surface calibration for the clean build."""

from time import perf_counter
import numpy as np
from scipy.optimize import differential_evolution, minimize

from calibration_core import CalibrationReport, OptionSurface, feller_feasible_population
from fourier_pricing import adaptive_cos_call_prices, carr_madan_call


class Heston:
    PARAMETER_NAMES = ("v0", "kappa", "theta", "sigma", "rho")
    BOUNDS = (
        (1e-4, 1.0),
        (0.1, 10.0),
        (1e-4, 1.0),
        (0.01, 8.0),
        (-0.99, 0.99),
    )
    INVALID_OBJECTIVE = 1e8

    @staticmethod
    def heston_charfunc(u, S0, v0, kappa, theta, sigma, rho, T, r, q=0.0):
        """Little-Heston-Trap characteristic function of log(S_T)."""
        alpha = -u**2 / 2 - 1j * u / 2
        beta = kappa - rho * sigma * 1j * u
        gamma = sigma**2 / 2
        d = np.sqrt(beta**2 - 4 * alpha * gamma)
        r_plus = (beta + d) / (2 * gamma)
        r_minus = (beta - d) / (2 * gamma)
        g = r_minus / r_plus
        exp_dT = np.exp(-d * T)
        C = kappa * (
            r_minus * T
            - (2 / sigma**2) * np.log((1 - g * exp_dT) / (1 - g))
        )
        D = r_minus * (1 - exp_dT) / (1 - g * exp_dT)
        return np.exp(
            1j * u * np.log(S0)
            + C * theta
            + D * v0
            + 1j * u * (r - q) * T
        )

    @staticmethod
    def _charfunc_for(S0, T, r, q, values):
        v0, kappa, theta, sigma, rho = values
        return lambda u: Heston.heston_charfunc(
            u, S0, v0, kappa, theta, sigma, rho, T, r, q
        )

    @staticmethod
    def heston_price_fast(S0, K, T, v0, kappa, theta, sigma, rho, r, q=0.0):
        values = (v0, kappa, theta, sigma, rho)
        return carr_madan_call(Heston._charfunc_for(S0, T, r, q, values), S0, K, T, r, q)

    @staticmethod
    def heston_prices_cos(
        S0, K, T, v0, kappa, theta, sigma, rho, r, q=0.0, N=256, L=12.0
    ):
        strikes = np.atleast_1d(np.asarray(K, dtype=float))
        if T <= 0:
            return np.maximum(S0 - strikes, 0.0)
        values = (v0, kappa, theta, sigma, rho)
        return adaptive_cos_call_prices(
            Heston._charfunc_for(S0, T, r, q, values),
            strikes,
            T,
            r,
            terms=N,
            width_scale=L,
        )

    @staticmethod
    def _admissible(values):
        v0, kappa, theta, sigma, rho = values
        return (
            v0 > 0
            and kappa > 0
            and theta > 0
            and sigma > 0
            and -0.999 <= rho <= 0.999
            and 2.0 * kappa * theta - sigma**2 >= 0.0
        )

    @staticmethod
    def heston_objective(params, df_market, S0, q=0.0, pricing="cos", cos_N=256):
        params = np.asarray(params, dtype=float)
        if params.shape != (5,) or not Heston._admissible(params):
            return Heston.INVALID_OBJECTIVE
        surface = OptionSurface.from_frame(df_market)
        error = 0.0
        try:
            for sl in surface.slices:
                if pricing == "cos":
                    model = Heston.heston_prices_cos(
                        S0, sl.strikes, sl.maturity, *params, sl.rate, q, N=cos_N
                    )
                elif pricing == "quad":
                    model = np.asarray([
                        Heston.heston_price_fast(
                            S0, k, sl.maturity, *params, sl.rate, q
                        )
                        for k in sl.strikes
                    ])
                else:
                    raise ValueError("pricing must be 'cos' or 'quad'")
                if not np.all(np.isfinite(model)):
                    return Heston.INVALID_OBJECTIVE
                error += float(np.sum(((model - sl.market_prices) / sl.safe_vegas) ** 2))
        except (FloatingPointError, OverflowError, ValueError):
            return Heston.INVALID_OBJECTIVE
        return error / surface.size

    @staticmethod
    def calibrate_heston(
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
        init = feller_feasible_population(Heston.BOUNDS, popsize, seed)
        global_result = differential_evolution(
            Heston.heston_objective,
            bounds=Heston.BOUNDS,
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
            Heston.heston_objective,
            x0=global_result.x,
            args=args,
            method="SLSQP",
            bounds=Heston.BOUNDS,
            constraints=constraints,
            options={"ftol": 1e-7, "maxiter": 100, "disp": disp},
        )
        report = CalibrationReport.from_optimizer(
            "Heston", Heston.PARAMETER_NAMES, local_result, global_result, started_at
        )
        return report if return_report else report.x
