import numpy as np
import pandas as pd

from conftest import EXAMPLE_DIR, ROOT
from ephys_analysis.anomalies import channel_signal_checks, recording_status, signal_flags
from ephys_analysis.utils import load_config

CFG = load_config(ROOT / "01_ephys_meta-org")
RNG = np.random.default_rng(0)


def vm(n=20000, rmp=-60.0):
    return rmp + np.cumsum(RNG.normal(0, 0.05, n)) + RNG.normal(0, 0.3, n)


def flags_of(result):
    return [f for f, *_ in result]


def test_duplicated_voltage_is_flagged():
    v = vm()
    out = signal_flags([v, v + 40 + RNG.normal(0, 0.2, v.size)], ["voltage", "voltage"], ["Vm", "IN 3"], CFG)
    assert flags_of(out) == ["duplicated_voltage_no_current"]


def test_independent_current_channel_is_not_flagged():
    out = signal_flags([vm(), RNG.normal(0, 5, 20000)], ["voltage", "current"], ["Vm", "Im"], CFG)
    assert out == []


def test_gain_error_is_flagged_with_factor():
    out = signal_flags([vm() * 20], ["voltage"], ["Vm"], CFG)
    assert flags_of(out) == ["possible_gain_error"]
    factor = float(out[0][2].split("roughly ")[1].split("x")[0])
    assert 15 < factor < 25


def test_voltage_clamp_files_are_not_checked_for_cc_anomalies():
    out = signal_flags([RNG.normal(-20, 2, 20000), vm()], ["current", "voltage"], ["Im", "Vsec"], CFG)
    assert out == []


def test_status_worst_flag_wins():
    flags = pd.DataFrame({"recording_id": ["a", "a", "b", "c"],
                          "severity": ["review", "needs_fix", "info", "review"]})
    st = recording_status(flags, pd.Series(["a", "b", "c", "d"]))
    assert list(st) == ["needs_fix", "ok", "review", "ok"]


def test_real_current_clamp_example_has_duplicated_voltage(example_abf):
    out = channel_signal_checks(example_abf("20261026_7503_A_0001.abf"), CFG)
    assert "duplicated_voltage_no_current" in flags_of(out)
    assert "possible_gain_error" not in flags_of(out)


def test_real_membrane_test_example_has_no_anomaly(example_abf):
    assert channel_signal_checks(example_abf("20261026_7503_B_0000.abf"), CFG) == []
