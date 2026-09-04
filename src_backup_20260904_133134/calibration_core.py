"""Shared contracts for option-surface calibration in the clean Team 8 build."""

from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np

REQUIRED_MARKET_COLUMNS = ("K", "T", "rate", "price", "vega")


@dataclass(frozen=True)
class MaturitySlice:
    maturity: float
    rate: float
    strikes: np.ndarray
    market_prices: np.ndarray
    safe_vegas: np.ndarray
    row_positions: np.ndarray

    @property
    def size(self):
        return int(self.strikes.size)


@dataclass(frozen=True)
class OptionSurface:
    """Validated numeric view of a calibration DataFrame grouped by (T, rate)."""

    slices: tuple
    size: int
    vega_floor: float

    @classmethod
    def from_frame(cls, frame, vega_floor=1e-4):
        if isinstance(frame, cls):
            return frame
        if frame is None or not hasattr(frame, "columns"):
            raise TypeError("market data must be a pandas-like DataFrame")
        missing = [c for c in REQUIRED_MARKET_COLUMNS if c not in frame.columns]
        if missing:
            raise ValueError(f"market data is missing required columns: {missing}")
        if len(frame) == 0:
            raise ValueError("market data must contain at least one option")
        if not np.isfinite(vega_floor) or vega_floor <= 0:
            raise ValueError("vega_floor must be positive and finite")

        values = {
            c: frame[c].to_numpy(dtype=float, copy=True)
            for c in REQUIRED_MARKET_COLUMNS
        }
        for name, arr in values.items():
            if arr.ndim != 1 or arr.size != len(frame):
                raise ValueError(f"column {name!r} is not one-dimensional")
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"column {name!r} contains non-finite values")

        if np.any(values["K"] <= 0):
            raise ValueError("all strikes must be positive")
        if np.any(values["T"] <= 0):
            raise ValueError("all maturities must be positive")
        if np.any(values["price"] < 0):
            raise ValueError("market prices must be non-negative")
        if np.any(values["vega"] < 0):
            raise ValueError("market vegas must be non-negative")

        keys = np.column_stack((values["T"], values["rate"]))
        _, first_positions, inverse = np.unique(
            keys, axis=0, return_index=True, return_inverse=True
        )
        stable_group_order = np.argsort(first_positions)

        slices = []
        for group_id in stable_group_order:
            positions = np.flatnonzero(inverse == group_id)
            order = np.argsort(values["K"][positions], kind="stable")
            positions = positions[order]
            slices.append(
                MaturitySlice(
                    maturity=float(values["T"][positions[0]]),
                    rate=float(values["rate"][positions[0]]),
                    strikes=values["K"][positions],
                    market_prices=values["price"][positions],
                    safe_vegas=np.maximum(values["vega"][positions], vega_floor),
                    row_positions=positions,
                )
            )
        return cls(tuple(slices), int(len(frame)), float(vega_floor))


def vega_weighted_mse(model_prices, market_prices, safe_vegas):
    model = np.asarray(model_prices, dtype=float)
    market = np.asarray(market_prices, dtype=float)
    vegas = np.asarray(safe_vegas, dtype=float)
    if model.shape != market.shape or model.shape != vegas.shape:
        raise ValueError("model prices, market prices and vegas must share a shape")
    if model.size == 0:
        raise ValueError("at least one price is required")
    return float(np.mean(((model - market) / vegas) ** 2))


def feller_feasible_population(bounds, popsize, seed=None):
    """Build a Differential Evolution population satisfying Heston Feller."""
    bounds = np.asarray(bounds, dtype=float)
    if bounds.ndim != 2 or bounds.shape[1] != 2 or bounds.shape[0] < 4:
        raise ValueError("bounds must contain at least four (lower, upper) pairs")
    population_size = max(5, int(popsize) * int(bounds.shape[0]))
    rng = np.random.default_rng(seed)
    population = rng.uniform(bounds[:, 0], bounds[:, 1], (population_size, len(bounds)))
    for row in population:
        kappa = row[1]
        minimum_theta = (bounds[3, 0] / 0.995) ** 2 / (2.0 * kappa)
        row[2] = max(row[2], min(minimum_theta * 1.01, bounds[2, 1]))
        feller_ceiling = 0.995 * np.sqrt(2.0 * row[1] * row[2])
        sigma_upper = min(bounds[3, 1], feller_ceiling)
        if sigma_upper < bounds[3, 0]:
            raise ValueError("bounds contain no Feller-feasible variance point")
        row[3] = rng.uniform(bounds[3, 0], sigma_upper)
    return population


@dataclass(frozen=True)
class CalibrationReport:
    model: str
    parameter_names: tuple
    values: tuple
    objective: float
    global_objective: float
    success: bool
    message: str
    evaluations: int
    iterations: int
    elapsed_seconds: float

    @property
    def x(self):
        return np.asarray(self.values, dtype=float)

    @property
    def params(self):
        return dict(zip(self.parameter_names, self.values))

    def as_dict(self):
        payload = asdict(self)
        payload["parameters"] = self.params
        return payload

    @classmethod
    def from_optimizer(cls, model, parameter_names, local_result, global_result, started_at):
        return cls(
            model=str(model),
            parameter_names=tuple(parameter_names),
            values=tuple(float(v) for v in local_result.x),
            objective=float(local_result.fun),
            global_objective=float(global_result.fun),
            success=bool(local_result.success),
            message=str(local_result.message),
            evaluations=int(getattr(local_result, "nfev", 0)),
            iterations=int(getattr(local_result, "nit", 0)),
            elapsed_seconds=float(perf_counter() - started_at),
        )
