"""Fast numerical smoke test for the clean Team 8 model stack."""

import numpy as np

from BnS import BnS
from Heston import Heston
from Bates import Bates
from BatesHawkesExact import BatesHawkesExact
from Hawkes import ExactHawkesCalibration


def main():
    S, K, T, r = 100.0, 100.0, 1.0, 0.03

    bs = BnS.bs_call_price(S, K, T, r, 0.20)
    iv = BnS.implied_vol_call(bs, S, K, T, r)
    assert np.isfinite(bs) and abs(iv - 0.20) < 1e-6

    heston = Heston.heston_price_fast(
        S, K, T, 0.04, 2.0, 0.04, 0.30, -0.70, r
    )
    assert np.isfinite(heston) and heston > 0.0

    bates = Bates.bates_price_fast(
        S, K, T, 0.04, 2.0, 0.04, 0.30, -0.70,
        0.40, -0.10, 0.20, r
    )
    assert np.isfinite(bates) and bates > 0.0

    hawkes_limit = BatesHawkesExact.hawkes_price_fast(
        S, K, T, 0.04, 2.0, 0.04, 0.30, -0.70,
        0.40, 0.40, 0.0, 2.0, -0.10, 0.20, r
    )
    assert abs(hawkes_limit - bates) < 1e-6

    full_hawkes = BatesHawkesExact.hawkes_price_cos(
        S,
        np.array([90.0, 100.0, 110.0]),
        0.5,
        0.04, 2.0, 0.04, 0.30, -0.70,
        0.40, 0.32, 0.40, 2.0, -0.10, 0.20,
        r,
        N=128,
        n_steps=80,
    )
    assert np.all(np.isfinite(full_hawkes))

    named = ExactHawkesCalibration.unpack_heston_params(
        [0.04, 2.0, 0.04, 0.30, -0.70, 0.40, 0.32, 0.20, 2.0, -0.10, 0.20]
    )
    assert abs(named["alpha"] - 0.40) < 1e-12

    print("[OK] Black-Scholes price + IV inversion")
    print("[OK] Heston pricing")
    print("[OK] Bates pricing")
    print("[OK] Exact Bates-Hawkes pricing")
    print("[OK] Bates limit of exact Hawkes when alpha=0")
    print("[OK] model stack ready")


if __name__ == "__main__":
    main()
