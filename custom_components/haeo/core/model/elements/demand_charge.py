"""Linear demand charges on a connection's measured active-power flow.

Each billing component owns a continuous peak variable.  The objective
prices that peak, and interval constraints make it at least the measured
power in every chargeable interval.  No binary variables are required.
"""

from collections.abc import Mapping
import math
from typing import Final, Literal, TypedDict

import numpy as np
from numpy.typing import NDArray

from custom_components.haeo.core.model.element import Element
from custom_components.haeo.core.model.linear import LinearArray, Rows
from custom_components.haeo.core.model.reactive import constraint, cost
from custom_components.haeo.core.model.solver import Solver

from .connection import Connection

type DemandChargeElementTypeName = Literal["demand_charge"]
ELEMENT_TYPE: Final[DemandChargeElementTypeName] = "demand_charge"
DEMAND_CHARGE_OUTPUT_NAMES: Final[frozenset[str]] = frozenset()


class DemandMeasurementConfig(TypedDict):
    """Weighted average of simulation-step kW for one complete billing measurement."""

    intervals: list[int]
    weights: list[float]


class DemandPeriodConfig(TypedDict):
    """One independently billed peak with weighted demand measurements."""

    measurements: list[DemandMeasurementConfig]
    rate_aud_per_unit_day: float
    billing_days: float
    existing_peak: float


class DemandChargeElementConfig(TypedDict):
    element_type: DemandChargeElementTypeName
    name: str
    connection: str
    unit: Literal["kW", "kVA"]
    power_factor: float
    billing_periods: list[DemandPeriodConfig]


def demand_charge_arguments(config: DemandChargeElementConfig, elements: Mapping[str, object]) -> dict[str, object]:
    """Resolve the measured import connection before constructing the element."""
    connection = elements.get(config["connection"])
    if not isinstance(connection, Connection):
        raise ValueError(f"Demand charge references unknown connection: {config['connection']}")
    return {
        "connection": connection,
        "unit": config["unit"],
        "power_factor": config["power_factor"],
        "billing_periods": config["billing_periods"],
    }


class DemandCharge(Element[str]):
    """Bill the maximum active/apparent demand within configured windows."""

    def __init__(
        self,
        name: str,
        periods: NDArray[np.float64],
        *,
        solver: Solver,
        connection: Connection,
        unit: Literal["kW", "kVA"],
        power_factor: float,
        billing_periods: list[DemandPeriodConfig],
    ) -> None:
        super().__init__(name=name, periods=periods, solver=solver, output_names=DEMAND_CHARGE_OUTPUT_NAMES)
        if unit not in ("kW", "kVA") or not math.isfinite(power_factor) or not 0 < power_factor <= 1:
            raise ValueError("Demand unit must be kW/kVA and power factor must be in (0, 1]")
        if not billing_periods:
            raise ValueError("Demand charge requires at least one billing period")
        self.connection = connection
        self.unit = unit
        self.power_factor = power_factor
        self.billing_periods = tuple(billing_periods)
        for period in self.billing_periods:
            if any(not math.isfinite(float(period[key])) or float(period[key]) < 0
                   for key in ("rate_aud_per_unit_day", "billing_days", "existing_peak")):
                raise ValueError("Demand rate, billing days and existing peak must be finite and nonnegative")
            for measurement in period["measurements"]:
                indices, weights = measurement["intervals"], measurement["weights"]
                if (not indices or len(indices) != len(weights)
                        or len(set(indices)) != len(indices)
                        or any(not isinstance(i, int) or isinstance(i, bool)
                               or i < 0 or i >= self.n_periods for i in indices)):
                    raise ValueError("Invalid demand measurement interval indices")
                if (any(not math.isfinite(float(w)) or float(w) <= 0 for w in weights)
                        or not math.isclose(sum(weights), 1.0, abs_tol=1e-9)):
                    raise ValueError("Demand measurement weights must be positive and sum to one")
        self._peak = solver.add_variables(
            len(self.billing_periods),
            name=f"{name}.billing_peak",
            lower=np.asarray([p["existing_peak"] for p in self.billing_periods], dtype=float),
        )

    @constraint
    def demand_peak_limits(self) -> Rows:
        """Every metered interval must be below its billing-period peak."""
        measured = self.connection.total_measured_power
        factor = 1.0 / self.power_factor if self.unit == "kVA" else 1.0
        rows = []
        for i, period in enumerate(self.billing_periods):
            for measurement in period["measurements"]:
                indices = np.asarray(measurement["intervals"], dtype=np.int64)
                weights = np.asarray(measurement["weights"], dtype=float)
                averaged = (measured[indices] * weights).total()
                rows.append(averaged * factor <= self._peak[i])
        return Rows.concat(rows)

    @cost
    def demand_cost(self) -> LinearArray:
        """Full billing-period peak cost, including any existing peak baseline."""
        rates = np.asarray([
            p["rate_aud_per_unit_day"] * p["billing_days"] for p in self.billing_periods
        ], dtype=float)
        return (self._peak * rates).total()

    @property
    def peak_values(self) -> tuple[float, ...]:
        """Optimised billing peaks, in the configured demand unit."""
        return self.extract_values(self._peak)

    @property
    def incremental_cost_aud(self) -> float:
        """Cost above already-established billing peaks, excluding sunk baseline."""
        return sum(
            (peak - period["existing_peak"]) * period["rate_aud_per_unit_day"] * period["billing_days"]
            for peak, period in zip(self.peak_values, self.billing_periods, strict=True)
        )


__all__ = ["DemandCharge", "DemandChargeElementConfig", "DemandPeriodConfig", "DemandMeasurementConfig", "demand_charge_arguments"]
