"""Exact event-dependent Bates-Hawkes option-pricing engine.

The jump intensity is a state variable:
    d lambda_t = beta (lambda_bar - lambda_t) dt + alpha dN_t
with branching ratio alpha / beta < 1.
"""

import math
import numpy as np

from Heston import Heston
from Bates import Bates
from fourier_pricing import (
    adaptive_cos_call_prices,
    carr_madan_call,
    cos_call_prices_from_values,
    gil_pelaez_call_put,
)

_EXP_REAL_CAP = 700.0


class BatesHawkesExact(Bates):
    @staticmethod
    def validate_params(lambda0, lambda_bar, alpha, beta, sigma=None, sigma_J=None):
        if beta <= 0:
            raise ValueError("beta must be positive")
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        if alpha >= beta:
            raise ValueError("stationarity requires alpha < beta")
        if lambda0 <= 0 or lambda_bar <= 0:
            raise ValueError("lambda0 and lambda_bar must be positive")
        if sigma is not None and sigma <= 0:
            raise ValueError("sigma must be positive")
        if sigma_J is not None and sigma_J < 0:
            raise ValueError("sigma_J must be non-negative")

    @staticmethod
    def branching_ratio(alpha, beta):
        if beta <= 0:
            raise ValueError("beta must be positive")
        return alpha / beta

    @staticmethod
    def kappa_J(mu_J, sigma_J):
        return math.exp(mu_J + 0.5 * sigma_J**2) - 1.0

    @staticmethod
    def _psi_Y(v, mu_J, sigma_J):
        return np.exp(1j * v * mu_J - 0.5 * sigma_J**2 * v**2)

    @staticmethod
    def _jump_ode_solution(
        u_arr, T, lambda_bar, alpha, beta, mu_J, sigma_J, n_steps=None
    ):
        kJ = BatesHawkesExact.kappa_J(mu_J, sigma_J)
        psi = BatesHawkesExact._psi_Y(u_arr, mu_J, sigma_J)
        const = -1j * u_arr * kJ - 1.0
        if n_steps is None:
            n_steps = min(600, max(80, int(math.ceil(120.0 * T))))
        h = T / n_steps
        A = np.zeros_like(u_arr)
        B = np.zeros_like(u_arr)

        def dB(Bc):
            z = alpha * Bc
            expaB = np.exp(np.minimum(z.real, _EXP_REAL_CAP) + 1j * z.imag)
            return psi * expaB - beta * Bc + const

        coeff = beta * lambda_bar
        for _ in range(n_steps):
            k1B = dB(B)
            k2B = dB(B + 0.5 * h * k1B)
            k3B = dB(B + 0.5 * h * k2B)
            k4B = dB(B + h * k3B)
            k1A = coeff * B
            k2A = coeff * (B + 0.5 * h * k1B)
            k3A = coeff * (B + 0.5 * h * k2B)
            k4A = coeff * (B + h * k3B)
            B = B + (h / 6.0) * (k1B + 2*k2B + 2*k3B + k4B)
            A = A + (h / 6.0) * (k1A + 2*k2A + 2*k3A + k4A)
        return A, B

    @staticmethod
    def jump_cf(
        u, T, lambda0, lambda_bar, alpha, beta, mu_J, sigma_J, n_steps=None
    ):
        scalar_input = np.ndim(u) == 0
        u_arr = np.atleast_1d(np.asarray(u, dtype=np.complex128))
        if T <= 0:
            out = np.ones_like(u_arr)
            return complex(out[0]) if scalar_input else out

        kJ = BatesHawkesExact.kappa_J(mu_J, sigma_J)
        if alpha == 0.0:
            c = (
                BatesHawkesExact._psi_Y(u_arr, mu_J, sigma_J)
                - 1.0
                - 1j * u_arr * kJ
            )
            decay = -np.expm1(-beta * T)
            B = c * decay / beta
            A = lambda_bar * c * (T - decay / beta)
        else:
            A, B = BatesHawkesExact._jump_ode_solution(
                u_arr, T, lambda_bar, alpha, beta, mu_J, sigma_J, n_steps
            )
        cf = np.exp(A + B * lambda0)
        return complex(cf[0]) if scalar_input else cf

    @staticmethod
    def _bs_charfunc(u, S0, sigma, T, r, q=0.0):
        drift = math.log(S0) + (r - q - 0.5 * sigma**2) * T
        return np.exp(1j*u*drift - 0.5*sigma**2*u**2*T)

    @staticmethod
    def hawkes_charfunc_constvol(
        u, S0, sigma, lambda0, lambda_bar, alpha, beta,
        mu_J, sigma_J, T, r, q=0.0, n_steps=None
    ):
        cf_diffusion = BatesHawkesExact._bs_charfunc(u, S0, sigma, T, r, q)
        cf_jump = BatesHawkesExact.jump_cf(
            u, T, lambda0, lambda_bar, alpha, beta, mu_J, sigma_J, n_steps
        )
        return cf_diffusion * cf_jump

    @staticmethod
    def hawkes_charfunc(
        u, S0, v0, kappa, theta, sigma, rho, lambda0, lambda_bar,
        alpha, beta, mu_J, sigma_J, T, r, q=0.0, n_steps=None
    ):
        cf_diffusion = Heston.heston_charfunc(
            u, S0, v0, kappa, theta, sigma, rho, T, r, q
        )
        cf_jump = BatesHawkesExact.jump_cf(
            u, T, lambda0, lambda_bar, alpha, beta, mu_J, sigma_J, n_steps
        )
        return cf_diffusion * cf_jump

    @staticmethod
    def hawkes_price_constvol_fast(
        S0, K, T, sigma, lambda0, lambda_bar, alpha, beta,
        mu_J, sigma_J, r, q=0.0, n_steps=None
    ):
        def cf(u):
            return BatesHawkesExact.hawkes_charfunc_constvol(
                u, S0, sigma, lambda0, lambda_bar, alpha, beta,
                mu_J, sigma_J, T, r, q, n_steps
            )
        return carr_madan_call(cf, S0, K, T, r, q)

    @staticmethod
    def hawkes_price_fast(
        S0, K, T, v0, kappa, theta, sigma, rho, lambda0,
        lambda_bar, alpha, beta, mu_J, sigma_J, r, q=0.0, n_steps=None
    ):
        def cf(u):
            return BatesHawkesExact.hawkes_charfunc(
                u, S0, v0, kappa, theta, sigma, rho,
                lambda0, lambda_bar, alpha, beta, mu_J, sigma_J,
                T, r, q, n_steps
            )
        return carr_madan_call(cf, S0, K, T, r, q)

    @staticmethod
    def hawkes_put_price_fast(
        S0, K, T, v0, kappa, theta, sigma, rho, lambda0,
        lambda_bar, alpha, beta, mu_J, sigma_J, r, q=0.0, n_steps=None
    ):
        def cf(u):
            return BatesHawkesExact.hawkes_charfunc(
                u, S0, v0, kappa, theta, sigma, rho,
                lambda0, lambda_bar, alpha, beta, mu_J, sigma_J,
                T, r, q, n_steps
            )
        return gil_pelaez_call_put(cf, S0, K, T, r, q)[1]

    @staticmethod
    def _cos_call_prices(phi_vals, u_k, a, b, K_array, r, T):
        return cos_call_prices_from_values(phi_vals, u_k, a, b, K_array, r, T)

    @staticmethod
    def hawkes_price_constvol_cos(
        S0, K, T, sigma, lambda0, lambda_bar, alpha, beta,
        mu_J, sigma_J, r, q=0.0, N=256, L=12.0, n_steps=None
    ):
        k_arr = np.atleast_1d(np.asarray(K, dtype=float))
        if T <= 0:
            return np.maximum(S0 - k_arr, 0.0)

        def phi(u):
            return BatesHawkesExact.hawkes_charfunc_constvol(
                u, S0, sigma, lambda0, lambda_bar, alpha, beta,
                mu_J, sigma_J, T, r, q, n_steps
            )
        return adaptive_cos_call_prices(phi, k_arr, T, r, terms=N, width_scale=L)

    @staticmethod
    def hawkes_price_cos(
        S0, K, T, v0, kappa, theta, sigma, rho, lambda0,
        lambda_bar, alpha, beta, mu_J, sigma_J, r, q=0.0,
        N=256, L=12.0, n_steps=None
    ):
        k_arr = np.atleast_1d(np.asarray(K, dtype=float))
        if T <= 0:
            return np.maximum(S0 - k_arr, 0.0)

        def phi(u):
            return BatesHawkesExact.hawkes_charfunc(
                u, S0, v0, kappa, theta, sigma, rho,
                lambda0, lambda_bar, alpha, beta, mu_J, sigma_J,
                T, r, q, n_steps
            )
        return adaptive_cos_call_prices(phi, k_arr, T, r, terms=N, width_scale=L)
