import numpy as np
import pytest

from ephys_analysis.abf import first_adc_of_kind, open_abf, standard_scale, summarise_dac, unit_kind


@pytest.mark.parametrize("unit,kind,factor,std", [
    ("mV", "voltage", 1.0, "mV"), ("V", "voltage", 1000.0, "mV"),
    ("pA", "current", 1.0, "pA"), ("nA", "current", 1000.0, "pA"), ("A", "current", 1e12, "pA"),
    ("xyz", "unknown", 1.0, "unknown"), ("", "unknown", 1.0, "unknown"), (None, "unknown", 1.0, "unknown"),
])
def test_units(unit, kind, factor, std):
    assert unit_kind(unit) == kind
    assert standard_scale(unit) == (factor, std)


def test_real_channel_roles_from_units_not_names(example_abf):
    # Ramp file: both ADCs named "I_MTest ..." but record mV (current clamp).
    abf = open_abf(example_abf("20261026_7503_A_0002.abf"))
    assert abf.adcNames[0].startswith("I_MTest")
    assert first_adc_of_kind(abf, "voltage") == 0
    assert first_adc_of_kind(abf, "current") is None
    s = summarise_dac(abf, 1)
    assert s.kind == "current" and s.has_ramp


def test_real_step_protocol_summary(example_abf):
    abf = open_abf(example_abf("20261026_7503_A_0001.abf"))
    s = summarise_dac(abf, 1)
    assert s.amplitude_varies_across_sweeps
    assert np.allclose(sorted(s.step_amplitudes), [-150, -100, -50, 0, 50, 100])
    assert summarise_dac(abf, 0).is_flat
