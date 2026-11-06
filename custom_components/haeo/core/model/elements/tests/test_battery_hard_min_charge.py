"""Focused regression tests for optional hard battery energy checkpoints."""

from highspy import HighsModelStatus
import numpy as np
import pytest

from custom_components.haeo.core.model.elements.battery import Battery
from custom_components.haeo.core.model.solver import Solver
from custom_components.haeo.core.model.util.solver_rows import add_rows


def _battery(*, initial: float = 2.0, checkpoints: dict[int, float] | None = None) -> tuple[Battery, Solver]:
    solver = Solver()
    solver.setOptionValue("output_flag", False)
    battery = Battery(
        name="truck-01",
        periods=np.array([1.0, 1.0, 1.0]),
        solver=solver,
        capacity=10.0,
        initial_charge=initial,
        hard_min_charge=checkpoints,
    )
    # Apply battery's physical, initial-state and optional checkpoint constraints.
    # No network connection: explicitly prescribe charge energy in each test.
    for name in (
        "battery_initial_charge",
        "battery_initial_discharge",
        "battery_energy_in_flow",
        "battery_energy_out_flow",
        "battery_full",
        "battery_empty",
        "battery_soc_max",
        "battery_soc_min",
        "battery_hard_min_charge",
    ):
        getattr(battery, name)()
    return battery, solver


def _solve_profile(
    *,
    initial: float = 2.0,
    checkpoints: dict[int, float] | None = None,
    charge: tuple[float, float, float] = (3.0, 2.0, 0.0),
) -> tuple[HighsModelStatus, tuple[float, ...] | None]:
    battery, solver = _battery(initial=initial, checkpoints=checkpoints)
    add_rows(solver, battery.power_consumption == np.asarray(charge))
    add_rows(solver, battery.power_production == 0.0)
    solver.run()
    if solver.getModelStatus() != HighsModelStatus.kOptimal:
        return solver.getModelStatus(), None
    return solver.getModelStatus(), battery.extract_values(battery.stored_energy)


def test_missing_and_empty_checkpoints_preserve_battery_behaviour() -> None:
    """The optional feature must not change an existing battery's energy profile."""
    for checkpoints in (None, {}):
        status, energy = _solve_profile(checkpoints=checkpoints)
        assert status == HighsModelStatus.kOptimal
        assert energy == pytest.approx((2.0, 5.0, 7.0, 7.0))


def test_sparse_checkpoints_apply_at_exact_boundaries() -> None:
    """Checkpoint 1 is after the first interval, and 3 is the final state."""
    status, energy = _solve_profile(checkpoints={3: 7.0, 1: 5.0})
    assert status == HighsModelStatus.kOptimal
    assert energy == pytest.approx((2.0, 5.0, 7.0, 7.0))


def test_checkpoint_at_boundary_zero_applies_to_initial_state() -> None:
    """Initial-boundary constraints cannot be met by charging in period zero."""
    assert _solve_profile(checkpoints={0: 2.0})[0] == HighsModelStatus.kOptimal
    assert _solve_profile(checkpoints={0: 3.0})[0] == HighsModelStatus.kInfeasible


def test_checkpoint_is_hard_even_when_soft_minimum_has_slack() -> None:
    """An insufficient fixed charging profile is infeasible, not penalised."""
    assert _solve_profile(checkpoints={1: 5.1})[0] == HighsModelStatus.kInfeasible
    assert _solve_profile(checkpoints={3: 7.1})[0] == HighsModelStatus.kInfeasible


@pytest.mark.parametrize(
    "checkpoints",
    [
        {-1: 1.0},
        {4: 1.0},
        {1.5: 1.0},
        {True: 1.0},
        {1: -0.1},
        {1: 10.1},
        {1: float("nan")},
        {1: float("inf")},
        {1: "invalid"},
    ],
)
def test_invalid_checkpoints_fail_at_construction(checkpoints: dict) -> None:
    with pytest.raises(ValueError, match="hard_min_charge"):
        _battery(checkpoints=checkpoints)
