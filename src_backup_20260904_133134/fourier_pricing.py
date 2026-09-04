"""Reusable Fourier inversion kernels for Heston, Bates and Bates-Hawkes."""

import math
import numpy as np
from scipy.integrate import quad


def carr_madan_call(charfunc, spot, strike, maturity, rate, dividend_yield=0.0):
    if maturity <= 0.0:
        return float(max(spot - strike, 0.0))

    def integrand(u):
        shifted = charfunc(u - 1j)
        standard = charfunc(u)
        numerator = np.exp(-rate * maturity) * (shifted - strike * standard)
        denominator = 1j * u * strike ** (1j * u)
        return np.real(numerator / denominator)

    integral, _ = quad(
        integrand, 0.0, 100.0, limit=200, epsabs=1e-4, epsrel=1e-4
    )
    price = (
        spot * np.exp(-dividend_yield * maturity)
        - strike * np.exp(-rate * maturity)
    ) / 2.0 + integral / np.pi
    return float(max(np.real(price), 0.0))


def gil_pelaez_call_put(charfunc, spot, strike, maturity, rate, dividend_yield=0.0):
    if maturity <= 0.0:
        call = float(max(spot - strike, 0.0))
        return call, float(max(strike - spot, 0.0))

    log_strike = math.log(strike)
    forward_cf = charfunc(-1j)

    def integrand_p2(u):
        return np.real(np.exp(-1j * u * log_strike) * charfunc(u) / (1j * u))

    def integrand_p1(u):
        numerator = np.exp(-1j * u * log_strike) * charfunc(u - 1j)
        return np.real(numerator / (1j * u * forward_cf))

    p2 = 0.5 + quad(
        integrand_p2, 0.0, 100.0, limit=200, epsabs=1e-4, epsrel=1e-4
    )[0] / np.pi
    p1 = 0.5 + quad(
        integrand_p1, 0.0, 100.0, limit=200, epsabs=1e-4, epsrel=1e-4
    )[0] / np.pi

    discounted_spot = spot * np.exp(-dividend_yield * maturity)
    discounted_strike = strike * np.exp(-rate * maturity)
    call = discounted_spot * p1 - discounted_strike * p2
    put = discounted_strike * (1.0 - p2) - discounted_spot * (1.0 - p1)
    return float(call), float(put)


def cos_call_prices_from_values(phi_values, frequencies, lower, upper, strikes, rate, maturity):
    real_coefficients = np.real(phi_values * np.exp(-1j * frequencies * lower))
    weights = np.ones_like(frequencies)
    weights[0] = 0.5
    factors = weights * real_coefficients

    strike_array = np.atleast_1d(np.asarray(strikes, dtype=float))
    clipped_log_strikes = np.clip(np.log(strike_array), lower, upper)[:, None]
    frequency_row = frequencies[None, :]
    upper_argument = frequency_row * (upper - lower)
    lower_argument = frequency_row * (clipped_log_strikes - lower)
    exp_lower = np.exp(clipped_log_strikes)
    exp_upper = math.exp(upper)
    inverse = 1.0 / (1.0 + frequency_row ** 2)

    chi = inverse * (
        np.cos(upper_argument) * exp_upper
        - np.cos(lower_argument) * exp_lower
        + frequency_row
        * (
            np.sin(upper_argument) * exp_upper
            - np.sin(lower_argument) * exp_lower
        )
    )

    safe_frequencies = frequency_row.copy()
    safe_frequencies[0, 0] = 1.0
    psi = (np.sin(upper_argument) - np.sin(lower_argument)) / safe_frequencies
    psi[:, 0] = upper - clipped_log_strikes[:, 0]

    payoff_coefficients = (2.0 / (upper - lower)) * (
        chi - strike_array[:, None] * psi
    )
    prices = math.exp(-rate * maturity) * (
        payoff_coefficients * factors[None, :]
    ).sum(axis=1)
    return np.maximum(prices, 0.0)


def adaptive_cos_call_prices(
    charfunc,
    strikes,
    maturity,
    rate,
    terms=256,
    width_scale=12.0,
    cumulant_step=0.05,
):
    strike_array = np.atleast_1d(np.asarray(strikes, dtype=float))
    if maturity <= 0.0:
        raise ValueError("adaptive COS pricing requires a positive maturity")
    if terms < 16:
        raise ValueError("COS pricing requires at least 16 terms")

    log_phi = np.log(charfunc(cumulant_step))
    first_cumulant = log_phi.imag / cumulant_step
    second_cumulant = max(-2.0 * log_phi.real / cumulant_step ** 2, 1e-6)
    half_width = width_scale * math.sqrt(second_cumulant)
    lower = first_cumulant - half_width
    upper = first_cumulant + half_width
    frequencies = np.arange(terms) * np.pi / (upper - lower)
    phi_values = charfunc(frequencies)
    return cos_call_prices_from_values(
        phi_values, frequencies, lower, upper, strike_array, rate, maturity
    )
