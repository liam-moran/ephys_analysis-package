"""Current-step analysis (steps.py), detection-first order in 04, and plotting helpers."""

import numpy as np
import pandas as pd
import pytest

from ephys_analysis import baseline as bl
from ephys_analysis import steps as st
from ephys_analysis.spikes import SpikeSettings, detect_attenuated, detect_spikes
from ephys_analysis.utils import envelope

from test_spikes_baseline import _spike

FS = 10_000.0
ON, OFF = 1.0, 1.5          # step window (s)


def _step_sweep(I_pA, rin=300.0, tau_ms=20.0, sag=0.0, v0=-60.0, spikes=(), rebound=(), duration=4.0):
    """Synthetic current-clamp sweep: RC response to a step of I_pA (Rin MOhm, tau ms),
    optional sag (fraction of the peak deflection recovered during the step), spikes at
    the given times and rebound spikes after the step."""
    t = np.arange(0, duration, 1 / FS)
    dv = I_pA * rin / 1e3                       # mV
    v = np.full_like(t, v0)
    inside = (t >= ON) & (t < OFF)
    x = t[inside] - ON
    peak_dv = dv / (1 - sag) if sag else dv
    v[inside] += peak_dv * (1 - np.exp(-x / (tau_ms / 1e3)))
    if sag:
        v[inside] -= (peak_dv - dv) * (1 - np.exp(-np.clip(x - 0.06, 0, None) / 0.05))
    after = t >= OFF
    v[after] += (v[inside][-1] - v0) * np.exp(-(t[after] - OFF) / (tau_ms / 1e3))
    command = np.where(inside, I_pA, 0.0)
    for s in list(spikes) + list(rebound):
        v += _spike(t, s, v0=0.0, thr=20.0, peak=100.0)
    return t, v, command


# --------------------------------------------------------------------------- helpers
def test_envelope_keeps_extremes_and_shrinks():
    t = np.arange(0, 100, 1e-4)
    v = np.zeros_like(t)
    v[123_456] = 50.0
    v[654_321] = -80.0
    te, ve = envelope(t, v, 2000)
    assert te.size == 4000 and ve.max() == 50 and ve.min() == -80
    assert np.all(np.diff(te) >= 0)
    short = np.arange(10.0)
    assert envelope(short, short, 2000)[0].size == 10


def test_step_window_and_amplitudes():
    cmds = [_step_sweep(i)[2] for i in (-100, 0, 50)]
    on, off = st.step_window(cmds)
    assert on == int(ON * FS) and off == int(OFF * FS)
    assert st.step_window([np.zeros(100)]) is None


def test_bursts_adaptation_slope():
    b = st.detect_bursts([0.0, 0.05, 0.1, 1.0, 1.05, 3.0], max_isi_s=0.1, min_spikes=2, min_ibi_s=0.2)
    assert [len(x) for x in b] == [3, 2]
    assert st.adaptation_index([0, 0.1, 0.3, 0.7]) == pytest.approx((1 / 3 + 2 / 6) / 2)
    assert np.isnan(st.adaptation_index([0, 0.1]))
    assert st.linear_slope([-150, -100, -50], [-45, -30, -15]) == pytest.approx(0.3)
    assert np.isnan(st.linear_slope([-50], [-15]))


# --------------------------------------------------------------------------- sweep features
def test_hyperpolarising_sweep_rin_and_sag():
    s = st.StepSettings()
    t, v, _ = _step_sweep(-100, rin=300, sag=0.2)
    f = st.sweep_features(t, v, int(ON * FS), int(OFF * FS), -100.0, detect_spikes(t, v), s)
    assert f["vm_pre_mV"] == pytest.approx(-60, abs=0.01)
    assert f["delta_v_ss_mV"] == pytest.approx(-30, abs=1.0)        # 100 pA x 300 MOhm
    assert f["sag_ratio"] == pytest.approx(0.2, abs=0.05)
    assert f["n_spikes_step"] == 0


def test_depolarising_sweep_rate_latency():
    s = st.StepSettings()
    t, v, _ = _step_sweep(50, spikes=[1.05, 1.15, 1.30, 1.48])
    d = detect_spikes(t, v)
    f = st.sweep_features(t, v, int(ON * FS), int(OFF * FS), 50.0, d, s)
    assert f["n_spikes_step"] == 4 and f["rate_step_Hz"] == pytest.approx(8.0)
    assert f["latency_first_spike_ms"] == pytest.approx(50, abs=1)
    assert f["adaptation_index"] > 0                                   # intervals lengthen


def test_rebound_spikes_after_hyperpolarising_step():
    s = st.StepSettings(rebound_spike_window_s=2.0)   # explicit: test independent of the default
    t, v, _ = _step_sweep(-150, rebound=[1.7, 1.75, 3.8])
    f = st.sweep_features(t, v, int(ON * FS), int(OFF * FS), -150.0, detect_spikes(t, v), s)
    assert f["n_rebound_spikes_short"] == 2 and f["n_rebound_spikes_all"] == 3
    assert f["latency_first_rebound_spike_ms"] == pytest.approx(200, abs=1)
    assert f["n_rebound_bursts"] == 1


def _sweep_table(spike_plan):
    s = st.StepSettings()
    rows, spk = [], []
    for k, (I, times) in enumerate(spike_plan):
        t, v, _ = _step_sweep(I, spikes=times)
        d = detect_spikes(t, v)
        d.insert(0, "sweep", k)
        d["period"] = [st._period(x, ON, OFF) for x in d["threshold_t_s"]]
        spk.append(d)
        f = st.sweep_features(t, v, int(ON * FS), int(OFF * FS), float(I), d, s)
        f.update(sweep=k, n_attenuated_step=0)
        rows.append(f)
    return pd.DataFrame(rows), pd.concat(spk, ignore_index=True), s


def test_recording_features_rheobase_fi_rin():
    sw, spk, s = _sweep_table([(-150, []), (-100, []), (-50, []), (0, []),
                               (50, [1.1, 1.3]), (100, [1.05, 1.15, 1.25, 1.35])])
    f, flags = st.recording_features(sw, spk, s)
    assert f["rin_MOhm"] == pytest.approx(300, rel=0.05)
    assert f["first_spiking_step_pA"] == 50 and f["fi_slope_Hz_per_pA"] == pytest.approx(0.08)
    assert np.isfinite(f["ap_threshold_mV"])
    assert "no_spikes_at_any_positive_step" not in flags
    assert f["adaptation_step_pA"] == 100 and np.isfinite(f["adaptation_index"])


def test_adaptation_needs_four_spikes():
    sw, spk, s = _sweep_table([(-100, []), (-50, []), (50, [1.1]), (100, [1.1, 1.3, 1.45])])
    f, flags = st.recording_features(sw, spk, s)
    assert np.isnan(f["adaptation_index"]) and "too_few_spikes_for_adaptation" in flags
    assert np.isnan(sw.loc[sw["I_step_pA"] == 100, "adaptation_index"].iloc[0])


def test_recording_features_silent_cell():
    sw, spk, s = _sweep_table([(-100, []), (-50, []), (0, []), (50, []), (100, [])])
    f, flags = st.recording_features(sw, spk, s)
    assert np.isnan(f["first_spiking_step_pA"]) and np.isnan(f["ap_threshold_mV"])
    assert "no_spikes_at_any_positive_step" in flags


def test_load_steps_needs_a_current_source():
    with pytest.raises(ValueError, match="no current information"):
        st.load_steps({"current_source": "none", "original_path": "x.abf"}, st.StepSettings())


# --------------------------------------------------------------------------- attenuated events
def test_attenuated_events_are_reported_not_counted():
    t = np.arange(0, 2, 1 / FS)
    v = np.full_like(t, -40.0)
    v += _spike(t, 0.5, v0=0.0, thr=20.0, peak=90.0)          # one full spike
    for c in (0.8, 1.0, 1.2):                                 # slow 40 mV bumps, ~5 mV/ms
        v += 40 * np.exp(-((t - c) / 0.006) ** 2)
    sp = detect_spikes(t, v)
    ev = detect_attenuated(t, v, sp)
    assert len(sp) == 1 and len(ev) == 3
    assert (ev["max_dvdt_mV_ms"] < 20).all()


# --------------------------------------------------------------------------- real files (skipped without data)
def test_baseline_detect_then_features_on_example(example_abf, tmp_path):
    p = example_abf("20261026_7503_A_0000.abf")
    sel = pd.DataFrame([{"recording_id": "A0000", "cell_tag": "A", "abf_filename": p.name, "original_path": str(p),
                         "voltage_source": "ADC0", "voltage_scale": 1.0, "bias_current_pA": np.nan}])
    spikes, det = bl.detect_all(sel, SpikeSettings(), bl.BaselineSettings(), log=lambda m: None,
                                figures_dir=tmp_path, formats=("png", "svg"))
    assert (tmp_path / "A0000.png").exists() and (tmp_path / "A0000.svg").exists()
    feats = bl.feature_table(sel, spikes, bl.BaselineSettings(), log=lambda m: None)
    assert feats.loc[0, "n_spikes"] == det.loc[0, "n_spikes"] == len(spikes)


def test_steps_on_example(example_abf, tmp_path):
    p = example_abf("20261026_7503_A_0001.abf")
    sel = pd.DataFrame([{"recording_id": "A0001", "cell_tag": "A", "abf_filename": p.name, "original_path": str(p),
                         "voltage_source": "ADC0", "voltage_scale": 1.0, "current_source": "command_DAC1",
                         "bias_current_pA": np.nan}])
    spikes, events, det = st.detect_all(sel, SpikeSettings(), st.StepSettings(), log=lambda m: None,
                                        figures_dir=tmp_path)
    assert det.loc[0, "steps_pA"] == "-150, -100, -50, 0, 50, 100"
    assert (tmp_path / "A0001.svg").exists()
    sweeps, feats = st.feature_table(sel, spikes, events, st.StepSettings(), log=lambda m: None)
    assert len(sweeps) == 6 and feats.loc[0, "first_spiking_step_pA"] == 50
    assert "depolarisation_block" in feats.loc[0, "flags"]
