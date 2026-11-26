"""Spike detection (spikes.py) and baseline features (baseline.py) on synthetic traces."""

import numpy as np
import pandas as pd
import pytest

from ephys_analysis.baseline import BaselineSettings, analysis_window, baseline_features, classify_firing
from ephys_analysis.spikes import SpikeSettings, detect_spikes, isi_stats, spike_mask

FS = 10_000.0


def _spike(t, t0, v0=-60.0, thr=-40.0, peak=40.0, rise_ms=0.6, decay_ms=1.5, ahp=-10.0):
    """Synthetic AP with a known threshold: a slow linear foot (v0 -> thr over 5 ms,
    4 mV/ms), then from t0 a smooth rise (dV/dt starts at 0, so the 5 % point is at t0),
    exponential decay and an AHP."""
    v = np.zeros_like(t)
    x = (t - t0) * 1e3                       # ms from threshold
    pre = (x > -5) & (x <= 0)
    v[pre] = (thr - v0) * (x[pre] + 5) / 5
    up = (x > 0) & (x <= rise_ms)
    v[up] = (thr - v0) + (peak - thr) * (0.5 - 0.5 * np.cos(np.pi * x[up] / rise_ms))
    dn = x > rise_ms
    xd = x[dn] - rise_ms
    v[dn] = (peak - v0) * np.exp(-xd / decay_ms) + ahp * (1 - np.exp(-xd / 2.0)) * np.exp(-xd / 30.0)
    return v


def _trace(spike_times, duration=10.0, v0=-60.0, noise=0.0, **kw):
    t = np.arange(0, duration, 1 / FS)
    v = np.full_like(t, v0)
    for st in spike_times:
        v += _spike(t, st, v0=v0, **kw)
    if noise:
        v += np.random.default_rng(0).normal(0, noise, t.size)
    return t, v


def test_detects_known_spikes_and_waveform():
    times = [1.0, 2.5, 4.0, 7.2]
    t, v = _trace(times, noise=0.2)
    s = detect_spikes(t, v)
    assert len(s) == 4 and s["valid"].all()
    # threshold time sits just before the fast rise (within 1 ms of the constructed threshold)
    assert np.allclose(s["threshold_t_s"], times, atol=1e-3)
    assert np.allclose(s["peak_mV"], 40.0, atol=1.0)
    assert np.allclose(s["threshold_mV"], -40.0, atol=2.0)
    assert (s["half_width_ms"] > 0.5).all() and (s["half_width_ms"] < 3).all()
    assert (s["ahp_mV"] < 0).all()
    assert np.allclose(s["isi_before_s"].iloc[1:], np.diff(times), atol=1e-3)


def test_silent_trace_returns_empty_table():
    t, v = _trace([], noise=0.3)
    s = detect_spikes(t, v)
    assert s.empty and list(s.columns)[0] == "spike"


def test_small_events_are_not_spikes():
    # an EPSP-like 8 mV bump: dV/dt stays far below 20 mV/ms
    t, v = _trace([])
    v = v + 8 * np.exp(-((t - 5) / 0.005) ** 2)
    assert detect_spikes(t, v).empty


def test_slow_failing_spike_is_counted_but_flagged():
    t, v = _trace([3.0], rise_ms=3.0, decay_ms=8.0)   # upstroke ~ 40 mV/ms, broad
    s = detect_spikes(t, v, SpikeSettings(min_valid_upstroke=50))
    assert len(s) == 1 and not s["valid"].iloc[0]
    assert "slow_upstroke" in s["flags"].iloc[0]


def test_threshold_not_biased_by_upstroke_speed():
    # same threshold, different rise speed: the 5 %-of-own-upstroke threshold stays put
    fast = detect_spikes(*_trace([2.0], rise_ms=0.4))
    slow = detect_spikes(*_trace([2.0], rise_ms=0.9))
    assert fast["threshold_mV"].iloc[0] == pytest.approx(-40, abs=2)
    assert slow["threshold_mV"].iloc[0] == pytest.approx(-40, abs=2)


def test_spike_mask_and_isi_stats():
    t = np.arange(0, 1, 1 / FS)
    keep = spike_mask(t, [0.5], before_ms=5, after_ms=50)
    assert not keep[(t > 0.496) & (t < 0.549)].any() and keep[t < 0.49].all()
    st = isi_stats([0, 1, 2, 3], min_intervals=5)
    assert st["n_isi"] == 3 and np.isnan(st["isi_cv"])
    st = isi_stats(np.arange(10.0), min_intervals=5)
    assert st["isi_cv"] == pytest.approx(0.0)


def test_window_is_last_100_s_and_short_recordings_flagged():
    s = BaselineSettings(window_s=100, min_window_s=60)
    t = np.arange(0, 127.6, 0.01)
    w, flags = analysis_window(t, s)
    assert flags == [] and t[w][0] == pytest.approx(27.6, abs=0.02)
    w, flags = analysis_window(np.arange(0, 50, 0.01), s)
    assert w.all() and "window_below_minimum" in flags


def test_classification_threshold_is_configurable():
    assert classify_firing(10, BaselineSettings(silent_max_spikes=10)) == "silent"
    assert classify_firing(11, BaselineSettings(silent_max_spikes=10)) == "tonic"
    assert classify_firing(3, BaselineSettings(silent_max_spikes=2)) == "tonic"


def test_baseline_features_vm_excludes_spikes():
    times = np.arange(0.5, 9.9, 0.6)                 # 16 spikes, regular
    t, v = _trace(times, v0=-50.0, thr=-35.0)
    sp = detect_spikes(t, v)
    f = baseline_features(t, v, sp, BaselineSettings(silent_max_spikes=10, exclude_after_ms=50))
    assert f["firing_class"] == "tonic" and f["n_spikes"] == 16
    assert f["firing_rate_Hz"] == pytest.approx(1.6, abs=0.01)
    assert f["isi_cv"] == pytest.approx(0.0, abs=0.01)
    # the AHP tail beyond 50 ms is small: Vm close to the -50 mV baseline
    assert f["vm_baseline_mV"] == pytest.approx(-50.0, abs=1.0)
    assert f["n_spikes_waveform"] == 16 and f["ap_peak_mV"] == pytest.approx(40, abs=1)


def test_baseline_features_silent_recording():
    t, v = _trace([], v0=-62.0, noise=0.5)
    f = baseline_features(t, v, detect_spikes(t, v), BaselineSettings())
    assert f["firing_class"] == "silent" and f["n_spikes"] == 0 and f["firing_rate_Hz"] == 0
    assert f["vm_baseline_mV"] == pytest.approx(-62.0, abs=0.1)
    assert np.isnan(f["isi_cv"]) and np.isnan(f["ap_threshold_mV"])
