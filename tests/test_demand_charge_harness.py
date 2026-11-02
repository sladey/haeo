"""Standalone LP regression harness for demand charges (no Home Assistant instance)."""

import numpy as np
import pytest

from custom_components.haeo.core.model.network import Network


def make_network(*, grid_price=0.0, backup_price=None, demand_periods=None, unit="kW", power_factor=1.0):
    network = Network("demand-harness", np.full(4, 0.5))
    network.add({"element_type": "node", "name": "grid", "is_source": True, "is_sink": False})
    network.add({"element_type": "node", "name": "load", "is_source": False, "is_sink": True})
    if backup_price is not None:
        network.add({"element_type": "node", "name": "backup", "is_source": True, "is_sink": False})
        network.add({"element_type": "node", "name": "junction", "is_source": False, "is_sink": False})
        target = "junction"
    else:
        target = "load"
    network.add({
        "element_type": "connection", "name": "site_import",
        "source": "grid", "target": target, "tags": {0},
        "segments": {"price": {"segment_type": "pricing", "price": grid_price},
                     **({} if backup_price is not None else {
                         "fixed": {"segment_type": "power_limit", "max_power": 100.0, "fixed": True}})},
    })
    if backup_price is not None:
        network.add({
            "element_type": "connection", "name": "backup_supply",
            "source": "backup", "target": "junction", "tags": {0},
            "segments": {"price": {"segment_type": "pricing", "price": backup_price}},
        })
        network.add({
            "element_type": "connection", "name": "fixed_load",
            "source": "junction", "target": "load", "tags": {0},
            "segments": {"fixed": {"segment_type": "power_limit", "max_power": 100.0, "fixed": True}},
        })
    if demand_periods is not None:
        demand = network.add({
            "element_type": "demand_charge", "name": "peak",
            "connection": "site_import", "unit": unit, "power_factor": power_factor,
            "billing_periods": demand_periods,
        })
    else:
        demand = None
    return network, demand


def period(indices, *, rate=1.0, days=1.0, existing=0.0):
    return {"measurements": [{"intervals": [i], "weights": [1.0]} for i in indices], "rate_aud_per_unit_day": rate,
            "billing_days": days, "existing_peak": existing}


def test_demand_drives_peak_shaving():
    # Backup energy costs 10 AUD/kWh. Peak charge costs 30 AUD/kW.
    # Optimiser should buy backup rather than create a new 100 kW grid peak.
    network, demand = make_network(
        backup_price=10.0,
        demand_periods=[period([0, 1, 2, 3], rate=1.0, days=30)],
    )
    network.optimize()
    assert demand is not None
    assert demand.peak_values[0] == pytest.approx(0.0, abs=1e-5)
    assert demand.incremental_cost_aud == pytest.approx(0.0, abs=1e-5)


def test_established_peak_prevents_uneconomic_shaving():
    network, demand = make_network(
        backup_price=10.0,
        demand_periods=[period([0, 1, 2, 3], rate=1.0, days=30, existing=100.0)],
    )
    network.optimize()
    assert demand is not None
    assert demand.peak_values[0] == pytest.approx(100.0)
    assert demand.incremental_cost_aud == pytest.approx(0.0)


def test_window_excludes_outside_peak():
    network, demand = make_network(
        backup_price=np.array([10.0, 10.0, 10.0, 10.0]),
        demand_periods=[period([1, 2], rate=1.0, days=30)],
    )
    network.optimize()
    assert demand is not None
    assert demand.peak_values[0] == pytest.approx(0.0, abs=1e-5)


def test_independent_billing_periods_and_kva_conversion():
    network, demand = make_network(
        backup_price=1000.0,
        unit="kVA", power_factor=0.8,
        demand_periods=[
            period([0, 1], rate=1.0, days=2),
            period([2, 3], rate=2.0, days=3),
        ],
    )
    network.optimize()
    assert demand is not None
    assert demand.peak_values == pytest.approx((125.0, 125.0))
    assert demand.incremental_cost_aud == pytest.approx(1000.0)


@pytest.mark.parametrize("indices", [[-1], [4]])
def test_rejects_invalid_window(indices):
    with pytest.raises(ValueError, match="interval"):
        make_network(backup_price=10.0, demand_periods=[period(indices)])


def test_energy_only_regression():
    network, demand = make_network(backup_price=10.0)
    assert demand is None
    network.optimize()
    assert network.elements["site_import"] is not None


def test_constant_100_kw_peak_and_cost():
    network, demand = make_network(demand_periods=[period([0, 1, 2, 3], rate=2.0, days=10)])
    network.optimize()
    assert demand is not None
    assert demand.peak_values == pytest.approx((100.0,))
    assert demand.incremental_cost_aud == pytest.approx(2000.0)


def test_constant_demand_kva_conversion():
    network, demand = make_network(
        demand_periods=[period([0, 1, 2, 3], rate=1.0, days=1)],
        unit="kVA", power_factor=0.8,
    )
    network.optimize()
    assert demand is not None
    assert demand.peak_values == pytest.approx((125.0,))


def test_weighted_average_not_instantaneous_peak():
    network, demand = make_network(
        demand_periods=[{"measurements": [
            {"intervals": [0, 1], "weights": [0.5, 0.5]},
            {"intervals": [2, 3], "weights": [0.5, 0.5]},
        ], "rate_aud_per_unit_day": 1.0, "billing_days": 1.0, "existing_peak": 0.0}]
    )
    network.optimize()
    assert demand is not None
    assert demand.peak_values == pytest.approx((100.0,))


def test_partial_billing_period_with_no_complete_measurements():
    network, demand = make_network(
        demand_periods=[{"measurements": [], "rate_aud_per_unit_day": 1.0,
                         "billing_days": 30.0, "existing_peak": 70.0}]
    )
    network.optimize()
    assert demand is not None
    assert demand.peak_values == pytest.approx((70.0,))


@pytest.mark.parametrize("weights", [[0.25, 0.25], [-0.5, 1.5], [float("nan"), 1.0]])
def test_invalid_measurement_weights(weights):
    with pytest.raises(ValueError, match="weights"):
        make_network(demand_periods=[{"measurements": [{"intervals": [0, 1], "weights": weights}],
            "rate_aud_per_unit_day": 1.0, "billing_days": 1.0, "existing_peak": 0.0}])
