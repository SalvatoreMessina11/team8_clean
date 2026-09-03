"""Black-Scholes pricing, implied volatility and vega utilities."""

import math
import numpy as np
from scipy.stats import norm


class BnS:
    @staticmethod
    def norm_cdf(x):
        return 0.5 * (1.0 + math.erf(x / np.sqrt(2.0)))

    @staticmethod
    def bs_call_price(S, K, T, r, sigma, q=0.0):
        if T <= 0:
            return max(S - K, 0.0)
        if sigma <= 0:
            return max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
        sqrtT = np.sqrt(T)
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrtT)
        d2 = d1 - sigma * sqrtT
        return float(
            S * np.exp(-q * T) * BnS.norm_cdf(d1)
            - K * np.exp(-r * T) * BnS.norm_cdf(d2)
        )

    @staticmethod
    def implied_vol_call(
        price,
        S,
        K,
        T,
        r,
        q=0.0,
        vol_low=1e-4,
        vol_high=5.0,
        tol=1e-8,
        max_iters=200,
    ):
        if S <= 0 or K <= 0 or T <= 0 or price <= 0:
            return np.nan
        discounted_spot = S * np.exp(-q * T)
        discounted_strike = K * np.exp(-r * T)
        lower_bound = max(discounted_spot - discounted_strike, 0.0)
        if price < lower_bound - 1e-10 or price > discounted_spot + 1e-10:
            return np.nan

        f_low = BnS.bs_call_price(S, K, T, r, vol_low, q) - price
        current_high = vol_high
        f_high = BnS.bs_call_price(S, K, T, r, current_high, q) - price
        for _ in range(12):
            if f_low * f_high <= 0:
                break
            current_high *= 2.0
            f_high = BnS.bs_call_price(S, K, T, r, current_high, q) - price
        else:
            return np.nan

        a, b = vol_low, current_high
        fa = f_low
        for _ in range(max_iters):
            mid = 0.5 * (a + b)
            fm = BnS.bs_call_price(S, K, T, r, mid, q) - price
            if abs(fm) < tol or (b - a) < tol:
                return float(mid)
            if fa * fm <= 0:
                b = mid
            else:
                a, fa = mid, fm
        return float(0.5 * (a + b))

    @staticmethod
    def calculate_bs_vega(S, K, T, r, q, sigma):
        if T <= 0.0 or sigma <= 0.0:
            return 1e-4
        d1 = (
            np.log(S / K) + (r - q + 0.5 * sigma**2) * T
        ) / (sigma * np.sqrt(T))
        vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)
        return float(max(vega, 1e-4))
