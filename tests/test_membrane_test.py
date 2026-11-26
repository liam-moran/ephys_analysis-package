import numpy as np
import pytest

from conftest import simulate_rc_pulse
from ephys_analysis.membrane_test import (MTSettings, analyse_recording, analyse_sweep,
                                                   command_edges, edge_segments)


def _onset(rows):
    return next(r for r in rows if r["edge_type"] == "onset")


def test_edge_detection_and_segments():
    cmd = np.zeros(300)
    cmd[100:150] = -10
    edges = command_edges(cmd)
    assert list(edges) == [100, 150]
    # neighbours capped at the pulse width (50 samples)
    assert edge_segments(300, edges) == [(50, 100, 150), (100, 150, 200)]


@pytest.mark.parametrize("Ra,Rm,Cm", [(20, 800, 40), (10, 1500, 25), (35, 300, 60)])
def test_ideal_circuit_is_recovered_exactly(Ra, Rm, Cm):
    t, i, c, tau = simulate_rc_pulse(Ra=Ra, Rm=Rm, Cm=Cm)
    rows = analyse_sweep(t, i, c, MTSettings())
    assert len(rows) == 2
    for r in rows:
        assert r["valid"], r["flags"]
        assert r["Rt_MOhm"] == pytest.approx(Ra + Rm, rel=0.01)
        assert r["tau_ms"] == pytest.approx(tau * 1e3, rel=0.02)
        assert r["Ra_MOhm"] == pytest.approx(Ra, rel=0.03)
        assert r["Rm_MOhm"] == pytest.approx(Rm, rel=0.01)
        assert r["Cm_pF"] == pytest.approx(Cm, rel=0.03)


def test_filtered_noisy_circuit_within_tolerance():
    """10 kHz 4-pole Bessel (as the amplifier filter) + 1.5 pA noise."""
    t, i, c, tau = simulate_rc_pulse(Ra=20, Rm=800, Cm=40, noise_pA=1.5, filter_hz=10_000, seed=3)
    r = _onset(analyse_sweep(t, i, c, MTSettings()))
    assert r["valid"], r["flags"]
    assert r["Rt_MOhm"] == pytest.approx(820, rel=0.05)
    assert r["Cm_pF"] == pytest.approx(40, rel=0.10)
    assert r["Ra_MOhm"] == pytest.approx(20, rel=0.15)
    # the filter attenuates the MEASURED peak, but back-extrapolating the fit recovers it
    assert abs(r["peak_measured_pA"]) < abs(r["peak_extrapolated_pA"])
    assert r["Ra_peak_MOhm"] == pytest.approx(20, rel=0.15)


def test_units_signs_independent_of_pulse_polarity():
    up = _onset(analyse_sweep(*simulate_rc_pulse(dV=+10)[:3], MTSettings()))
    down = _onset(analyse_sweep(*simulate_rc_pulse(dV=-10)[:3], MTSettings()))
    for k in ("Rt_MOhm", "Ra_MOhm", "Cm_pF", "tau_ms"):
        assert up[k] == pytest.approx(down[k], rel=1e-6)
        assert up[k] > 0


def test_no_transient_is_flagged_not_crashed():
    """Pulse applied but no cell response (e.g. lost seal / wrong channel)."""
    t, _, c, _ = simulate_rc_pulse()
    flat = np.random.default_rng(1).normal(-20, 1.5, t.size)
    rows = analyse_sweep(t, flat, c, MTSettings())
    assert rows and not any(r["valid"] for r in rows)
    assert all(r["flags"] for r in rows)
    assert all("Cm_pF" not in r or not np.isfinite(r.get("Cm_pF", np.nan)) or not r["valid"] for r in rows)


def test_biexponential_diagnostic_detects_two_components():
    t, i1, c, _ = simulate_rc_pulse(Ra=20, Rm=800, Cm=40)
    _, i2, _, _ = simulate_rc_pulse(Ra=200, Rm=800, Cm=40)  # slower second component
    r = _onset(analyse_sweep(t, i1 + (i2 - i2[0]) * 0.5, c, MTSettings()))
    assert r["biexp_status"] == "ok"
    assert r["biexp_delta_aic"] > 10
    assert r["biexp_tau_fast_ms"] < r["biexp_tau_slow_ms"]


# ---------------------------------------------------------------- real example data
def test_real_file_uses_active_dac_not_decoy(example_abf):
    """In the example membrane tests DAC0 steps 112->32 mV but drives nothing recorded;
    the real -10 mV test pulse is on DAC1. The old script analysed DAC0."""
    pulses, info = analyse_recording(example_abf("20261026_7503_B_0000.abf"), MTSettings(), "B_0000")
    assert info["command_dac"] == 1
    assert set(pulses["dV_mV"].round(3).abs()) == {10.0}
    good = pulses[pulses["valid"]]
    assert len(good) >= 5
    # physiological sanity ranges (NOT a Clampex validation)
    assert 20 < good["Cm_pF"].median() < 80
    assert 300 < good["Rt_MOhm"].median() < 2000


def test_real_file_without_access_is_flagged(example_abf):
    pulses, info = analyse_recording(example_abf("20261026_7503_B_0006.abf"), MTSettings(), "B_0006")
    assert info["command_dac"] == 1
    assert not pulses["valid"].any()


# --------------------------------------------------------------------------- fast-component Ra
def _two_compartment(Ra, Cs, Rd, Cd, Rm=700.0, fs_out=50_000, up=20, dV=-10.0, filter_hz=2000.0, noise=1.0):
    """Voltage clamp of a soma (Cs, Rm) coupled to a dendrite (Rd, Cd) through Ra; integrated
    at fs_out*up, filtered with a 4-pole Bessel and sampled at fs_out. MOhm, pF, mV, pA."""
    from scipy.signal import bessel, lfilter
    fs = fs_out * up
    dt = 1 / fs
    n = int(0.075 * fs)
    t = np.arange(n) * dt
    cmd = np.where((t >= 0.025) & (t < 0.05), dV, 0.0)
    vs = vd = 0.0
    current = np.empty(n)
    for i in range(n):
        ip = (cmd[i] - vs) / Ra * 1e3
        idd = (vs - vd) / Rd * 1e3
        vs += dt * 1e3 * (ip - vs / Rm * 1e3 - idd) / Cs
        vd += dt * 1e3 * idd / Cd
        current[i] = ip
    b, a = bessel(4, filter_hz / (fs / 2), norm="mag")
    out = lfilter(b, a, current)[::up] + np.random.default_rng(0).normal(0, noise, n // up)
    return t[::up], out, cmd[::up]


@pytest.mark.parametrize("Ra,Cs,Rd,Cd", [(15, 25, 30, 20), (20, 25, 50, 15), (10, 30, 40, 15), (20, 40, 1e6, 1)])
def test_fast_component_ra_recovers_two_compartment_ra(Ra, Cs, Rd, Cd):
    import pandas as pd
    from ephys_analysis.abf import SweepData
    from ephys_analysis.membrane_test import fast_component, mean_transient

    t, current, cmd = _two_compartment(Ra, Cs, Rd, Cd)
    s = MTSettings()
    rows = analyse_sweep(t, current, cmd, s)
    for r in rows:
        r["sweep"] = 0
    pulses = pd.DataFrame(rows)
    sw = SweepData(sweep=0, time_s=t, signal=current, signal_unit="pA", command=cmd, command_unit="mV")
    out = fast_component(mean_transient([sw], pulses, s), 2000.0, s)
    assert out["ra_fast_status"] == "ok"
    assert out["Ra_fast_MOhm"] == pytest.approx(Ra, rel=0.25)
    clampex = np.median([r["Ra_MOhm"] for r in rows])
    assert abs(out["Ra_fast_MOhm"] - Ra) <= abs(clampex - Ra) + 0.5   # never worse than Clampex


def test_unresolved_fast_component_gives_nan():
    from ephys_analysis.membrane_test import fast_component
    t = np.arange(-0.001, 0.025, 2e-5)
    y = np.where(t >= 0, 5.0, 0.0)          # no transient at all, only a step
    y[t >= 0] += 400 * np.exp(-t[t >= 0] / 2e-5)
    out = fast_component((t, y, 10.0, 5.0, 50_000.0), 2000.0, MTSettings())
    assert out["ra_fast_status"] in ("fast_component_unresolved", "no_transient", "fit_not_converged")
    assert np.isnan(out["Ra_fast_MOhm"])
