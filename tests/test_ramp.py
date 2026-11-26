"""Ramp analysis (ramp.py) on synthetic sweeps and the example file."""

import numpy as np
import pandas as pd
import pytest

from ephys_analysis import ramp as rp
from ephys_analysis.spikes import SpikeSettings, detect_spikes

from test_spikes_baseline import _spike

FS = 10_000.0


def _command(duration=10.0, neg=True):
    """0 (2 s) | 0->100 pA (0.5 s) | 100->0 (0.5 s) | 0 (2 s) | 0->-100 (0.5 s) | 0 to the end."""
    t = np.arange(0, duration, 1 / FS)
    c = np.zeros_like(t)
    up = (t >= 2.0) & (t < 2.5)
    c[up] = 200 * (t[up] - 2.0)
    dn = (t >= 2.5) & (t < 3.0)
    c[dn] = 100 - 200 * (t[dn] - 2.5)
    if neg:
        ng = (t >= 5.0) & (t < 5.5)
        c[ng] = -200 * (t[ng] - 5.0)
    return t, c


def _sweep(rin=300.0, v0=-60.0, spikes=(), neg=True):
    t, c = _command(neg=neg)
    v = v0 + c * rin / 1e3                     # instantaneous (no membrane lag)
    for st in spikes:
        v = v + _spike(t, st, v0=0.0, thr=20.0, peak=100.0)
    return t, v, c


def _label(d, seg, t):
    d = d.copy()
    d["period"] = [rp._period((x - t[0]) * FS, seg) for x in d["threshold_t_s"]]
    return d


def test_segments_from_command():
    t, c = _command()
    seg = rp.ramp_segments(c)
    assert seg["up_start"] == pytest.approx(2.0 * FS, abs=2) and seg["up_end"] == pytest.approx(2.5 * FS, abs=2)
    assert seg["down_end"] == pytest.approx(3.0 * FS, abs=2)
    assert seg["neg_start"] == pytest.approx(5.0 * FS, abs=2) and seg["release"] == pytest.approx(5.5 * FS, abs=2)
    assert seg["I_max_pA"] == pytest.approx(100, abs=0.5) and seg["I_min_pA"] == pytest.approx(-100, abs=0.5)
    assert rp.ramp_segments(np.zeros(100)) is None
    assert rp.ramp_segments(_command(neg=False)[1])["neg_start"] is None


def test_first_spike_current_hysteresis_rin():
    # spikes at 2.1 s (20 pA, up) ... and last on the way down at 2.9 s (20 pA)
    t, v, c = _sweep(spikes=[2.1, 2.2, 2.3, 2.6, 2.9])
    seg = rp.ramp_segments(c)
    d = _label(detect_spikes(t, v), seg, t)
    f = rp.sweep_features(t, v, c, seg, d, pd.DataFrame(), rp.RampSettings())
    assert f["n_spikes_up"] == 3 and f["n_spikes_down"] == 2
    assert f["I_first_spike_pA"] == pytest.approx(20, abs=1)
    assert f["I_last_spike_down_pA"] == pytest.approx(20, abs=1)
    assert f["hysteresis_pA"] == pytest.approx(0, abs=2)
    assert f["latency_first_spike_ms"] == pytest.approx(100, abs=1)
    assert f["rin_ramp_MOhm"] == pytest.approx(300, rel=0.02)
    assert f["v_min_neg_mV"] == pytest.approx(-90, abs=0.5)


def test_rebound_after_release_and_silent_ramp():
    t, v, c = _sweep(spikes=[5.8, 5.85, 8.0])          # release at 5.5 s
    seg = rp.ramp_segments(c)
    d = _label(detect_spikes(t, v), seg, t)
    f = rp.sweep_features(t, v, c, seg, d, pd.DataFrame(), rp.RampSettings(rebound_spike_window_s=2.0))
    assert f["n_spikes_up"] == 0 and np.isnan(f["I_first_spike_pA"]) and np.isnan(f["hysteresis_pA"])
    assert f["n_rebound_spikes_short"] == 2 and f["n_rebound_spikes_all"] == 3
    assert f["latency_first_rebound_spike_ms"] == pytest.approx(300, abs=1)
    assert f["n_rebound_bursts"] == 1


def test_quiet_window_before_ramp():
    # spike 0.3 s before the ramp (2.0 s): not quiet; spike 0.8 s before: outside the 0.5 s window
    for st, ok in ((1.7, False), (1.2, True)):
        t, v, c = _sweep(spikes=[st, 2.1])
        seg = rp.ramp_segments(c)
        f = rp.sweep_features(t, v, c, seg, _label(detect_spikes(t, v), seg, t), pd.DataFrame(), rp.RampSettings())
        assert f["rheobase_sweep_ok"] is ok and f["n_spikes_before_ramp"] == (0 if ok else 1)


def test_rheobase_mean_over_quiet_sweeps():
    sw = pd.DataFrame({"rheobase_sweep_ok": [True, False, True, True],
                       "I_first_spike_pA": [10.0, 2.0, 20.0, np.nan]})   # sweep 4: quiet, no spike
    r = rp.rheobase(sw)
    assert r["rheobase_pA"] == pytest.approx(15) and r["n_sweeps_rheobase"] == 2
    assert r["n_sweeps_rheobase_quiet"] == 3 and r["rheobase_sd_pA"] == pytest.approx(np.std([10, 20], ddof=1))
    none = rp.rheobase(sw.assign(rheobase_sweep_ok=False))
    assert np.isnan(none["rheobase_pA"]) and none["n_sweeps_rheobase_quiet"] == 0


def test_no_negative_ramp_leaves_rebound_nan():
    t, v, c = _sweep(neg=False)
    seg = rp.ramp_segments(c)
    f = rp.sweep_features(t, v, c, seg, _label(detect_spikes(t, v), seg, t), pd.DataFrame(), rp.RampSettings())
    assert np.isnan(f["rin_ramp_MOhm"]) and np.isnan(f["rebound_peak_mV"])


def test_load_ramp_needs_current_source():
    with pytest.raises(ValueError, match="no current information"):
        rp.load_ramp({"current_source": "none", "original_path": "x.abf"}, rp.RampSettings())


def test_ramp_on_example(example_abf, tmp_path):
    p = example_abf("20261026_7503_A_0002.abf")
    sel = pd.DataFrame([{"recording_id": "A0002", "cell_tag": "A", "abf_filename": p.name, "original_path": str(p),
                         "voltage_source": "ADC0", "voltage_scale": 1.0, "current_source": "command_DAC1",
                         "bias_current_pA": np.nan}])
    spikes, events, det = rp.detect_all(sel, SpikeSettings(), rp.RampSettings(), log=lambda m: None,
                                        figures_dir=tmp_path)
    assert (tmp_path / "A0002.png").exists() and (tmp_path / "A0002.svg").exists()
    sweeps, feats, waves = rp.feature_table(sel, spikes, events, rp.RampSettings(), log=lambda m: None)
    assert feats.loc[0, "n_sweeps_ap"] >= 1 and set(waves["sweep"]) <= set(sweeps["sweep"])
    assert waves.groupby("sweep").size().nunique() == 1          # same window for every spike
    assert len(sweeps) == 3
    assert 5 < feats.loc[0, "I_first_spike_pA"] < 30
    assert 5 < feats.loc[0, "rheobase_pA"] < 30 and feats.loc[0, "n_sweeps_rheobase"] >= 1
    assert feats.loc[0, "n_rebound_spikes_short"] >= 5        # clear rebound after release
    assert "depolarisation_block" in feats.loc[0, "flags"]


def test_spike_waveforms_window_and_edges():
    from ephys_analysis.spikes import spike_waveforms
    t = np.arange(0, 1.0, 1e-4)
    v = -60 + 80 * np.exp(-((t - 0.5) / 0.0005) ** 2)
    w = spike_waveforms(t, v, [0.4995, 0.001, 0.999], pre_ms=5, post_ms=15)   # 2nd and 3rd too close to edges
    assert set(w["spike"]) == {0} and len(w) == 201
    assert w["t_ms"].iloc[0] == pytest.approx(-5) and w["t_ms"].iloc[-1] == pytest.approx(15)
    assert w.loc[w["v_mV"].idxmax(), "t_ms"] == pytest.approx(0.5, abs=0.11)


def test_ap_spikes_quiet_sweeps_and_fallback():
    sp = pd.DataFrame({"sweep": [0, 0, 1, 2], "threshold_t_s": [1.0, 1.1, 1.2, 1.0], "period": "up",
                       "valid": [False, True, True, True], "threshold_mV": [-30.0, -31, -33, -40]})
    sw = pd.DataFrame({"sweep": [0, 1, 2], "rheobase_sweep_ok": [True, True, False]})
    use, flag = rp.ap_spikes(sp, sw)
    assert list(use["threshold_mV"]) == [-31, -33] and flag == ""      # first VALID spike, quiet sweeps only
    use, flag = rp.ap_spikes(sp, sw.assign(rheobase_sweep_ok=False))
    assert len(use) == 1 and flag == "ap_from_non_quiet_sweep"
    use, flag = rp.ap_spikes(sp.assign(valid=False), sw)
    assert len(use) == 0 and flag == "no_valid_ramp_spike"
