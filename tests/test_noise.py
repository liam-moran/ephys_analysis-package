"""Noise QC and corrections (noise.py) on synthetic signals with known answers."""

import numpy as np
import pandas as pd
import pytest

from ephys_analysis import noise as nz
from ephys_analysis.anomaly_fix import corrections_for, validate_decisions

FS = 10_000.0
S = nz.NoiseSettings()


def _trace(seconds=60.0, seed=0, sd=0.2):
    rng = np.random.default_rng(seed)
    t = np.arange(0, seconds, 1 / FS)
    return t, -55 + rng.normal(0, sd, t.size)


def _spikes(t, times, height=90.0):
    v = np.zeros_like(t)
    for ts in times:
        v += height * np.exp(-((t - ts) / 0.0006) ** 2)
    return v


def _spectrum(x, seg_s=10.0):
    return nz.welch_average([x], FS, int(seg_s * FS))


def test_line_amplitude_known_sine():
    t, v = _trace()
    v = v + 0.25 * np.sin(2 * np.pi * 50 * t) + 0.05 * np.sin(2 * np.pi * 150 * t + 1)   # 0.5 and 0.1 mV pp
    f, p = _spectrum(v)
    assert nz.line_amplitude(f, p, 50) == pytest.approx(0.5, rel=0.1)
    assert nz.line_amplitude(f, p, 150) == pytest.approx(0.1, rel=0.15)
    assert nz.line_amplitude(f, p, 250) < 0.02                         # absent harmonic


def test_quiet_mask_and_spike_free():
    cmd = np.zeros(int(3 * FS))
    cmd[int(1 * FS):int(1.5 * FS)] = -50.0
    q = nz.quiet_mask(cmd, FS, guard_s=0.3)
    assert q[: int(1 * FS)].all() and not q[int(1 * FS):int(1.8 * FS)].any() and q[int(1.85 * FS):].all()
    t, v = _trace(5)
    w, mask = nz.spike_free(v + _spikes(t, [2.0]), FS, 5, 50)
    assert mask[int(2.0 * FS)] and np.abs(w).max() < 60 and not mask[int(1.0 * FS)]


def test_low_frequency_pump_peak_found_but_not_broad_bumps():
    t, v = _trace(120)
    pump = v + 1.0 * np.sin(2 * np.pi * 1.5 * t)                       # 2 mV pp at 1.5 Hz
    f, p = _spectrum(pump)
    peaks = nz.lowf_peaks(f, p, S)
    assert peaks and peaks[0]["freq_Hz"] == pytest.approx(1.5, abs=0.1) and peaks[0]["pp"] == pytest.approx(2, rel=0.2)
    rng = np.random.default_rng(3)
    slow = v + np.convolve(rng.normal(0, 1, t.size), np.ones(20000) / 140, "same")   # broadband slow wander
    f, p = _spectrum(slow)
    assert not [d for d in nz.lowf_peaks(f, p, S) if d["pp"] > 0.5]


def test_artefacts_brief_and_large_only():
    t, v = _trace(30)
    art = v.copy()
    for ts in (5.0, 12.0, 20.0):                                       # 3 brief 6 mV transients (0.3 ms)
        i = int(ts * FS)
        art[i:i + 3] += 6.0
    assert nz.artefact_events(art, FS, 8, 5, min_amp=2.0, max_ms=1.0) == 3
    epsp = v + 4 * np.clip((t - 10) / 0.002, 0, 1) * np.exp(-np.clip(t - 10.002, 0, None) / 0.02)   # slower event
    assert nz.artefact_events(epsp, FS, 8, 5, min_amp=2.0, max_ms=1.0) == 0


def test_remove_mains_keeps_spikes():
    t, v = _trace(20)
    clean = v + _spikes(t, [3.0, 7.5, 12.2])
    hum = 0.4 * np.sin(2 * np.pi * 50 * t + 0.3) + 0.1 * np.sin(2 * np.pi * 150 * t)
    fixed, done = nz.correct(t, clean + hum, {"mains_Hz": 50, "mains_harmonics": 5, "mains_window_s": 1.0})
    assert done == ["mains_removed=50Hz_x5"]
    f, p = _spectrum(fixed)
    assert nz.line_amplitude(f, p, 50) < 0.05                           # from 0.8 mV pp
    for ts in (3.0, 7.5, 12.2):
        i = int(ts * FS)
        assert fixed[i] == pytest.approx(clean[i], abs=0.3)             # spike peak preserved


def test_notch_and_no_correction():
    t, v = _trace(60)
    x = v + 1.0 * np.sin(2 * np.pi * 1.3 * t)
    y, done = nz.correct(t, x, {"notch_Hz": "1.3", "notch_Q": 5})
    f, p = _spectrum(y)
    assert nz.line_amplitude(f, p, 1.3, half_bw=0.2) < 0.2 and done == ["notch=1.3Hz_Q5"]
    same, done = nz.correct(t, x, {"mains_Hz": np.nan, "notch_Hz": ""})
    assert done == [] and np.array_equal(same, x)
    assert nz.parse_freqs("1,3; 2.6") == [1.3, 2.6] and nz.parse_freqs(np.nan) == []


def test_noise_flags_severity():
    m = {"assessed": True, "unit": "mV", "mains_pp": 1.2, "mains_50Hz_pp": 1.0, "mains_150Hz_pp": 0.2, "rms": 0.5,
         "lowf_peak_pp": 0.3, "lowf_peak_Hz": 1.5, "lowf_peaks": "1.50 Hz (0.3 mV)", "artefacts_per_min": 0.0,
         "lowf_assessed": True}
    fl = {f: sev for f, sev, _, _ in nz.noise_flags(m, S)}
    assert fl == {"mains_noise": "review", "periodic_low_frequency_noise": "info"}
    fl = {f: sev for f, sev, _, _ in nz.noise_flags({**m, "mains_pp": 0.1, "rms": 4.0, "artefacts_per_min": 10}, S)}
    assert "mains_noise" not in fl and fl["high_rms_noise"] == "review" and fl["transient_artefacts"] == "review"
    assert nz.noise_flags({"assessed": False, "reason": "x"}, S)[0][0] == "noise_not_assessed"


def test_decisions_and_corrections_columns():
    sheet = pd.DataFrame([
        {"recording_id": "r", "flag": "mains_noise", "decision": "remove_mains", "value": "", "note": "",
         "allowed_actions": "accept, remove_mains, notch, exclude"},
        {"recording_id": "r", "flag": "mains_noise", "decision": "notch", "value": "1.3; 2.6", "note": "",
         "allowed_actions": "accept, remove_mains, notch, exclude"},
        {"recording_id": "r", "flag": "mains_noise", "decision": "notch", "value": "", "note": "",
         "allowed_actions": "accept, remove_mains, notch, exclude"},
        {"recording_id": "r", "flag": "mains_noise", "decision": "remove_mains", "value": "2.5", "note": "",
         "allowed_actions": "accept, remove_mains, notch, exclude"}])
    st = list(validate_decisions(sheet, {"families": []})["decision_status"])
    assert st[0] == "ok" and st[1] == "ok" and st[2].startswith("invalid") and st[3].startswith("invalid")
    rec = pd.Series({"adc_kinds": "voltage | voltage", "abf_family": "spontaneous", "command_dac": "1"})
    c = corrections_for(rec, {"remove_mains": "", "notch": "1.3; 2.6"}, {"mains_Hz": 50, "mains_harmonics": 5})
    assert c["mains_Hz"] == 50 and c["mains_harmonics"] == 5 and c["notch_Hz"] == "1.3; 2.6" and c["notch_Q"] == 30
    none = corrections_for(rec, {})
    assert np.isnan(none["mains_Hz"]) and none["notch_Hz"] == ""


def test_noise_metrics_on_example(example_abf):
    m = nz.noise_metrics(example_abf("20261026_7503_A_0000.abf"), "current_clamp", 1, S)
    assert m["assessed"] and m["unit"] == "mV" and m["quiet_s"] > 60
    assert 0.1 < m["mains_50Hz_pp"] < 1.0                               # mains hum is present in this rig
