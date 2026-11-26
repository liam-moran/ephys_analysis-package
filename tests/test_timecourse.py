"""Modulation timecourse (timecourse.py): tags, periods, bins, peak response, example file."""

import numpy as np
import pandas as pd
import pytest

from ephys_analysis import timecourse as tc
from ephys_analysis.baseline import BaselineSettings
from ephys_analysis.spikes import SpikeSettings

S = tc.TimecourseSettings()


def test_tags_by_keyword_and_by_order():
    ev, fl = tc.parse_tags([741.1, 154.3], ["washout", "5ht hit bath"], S)      # sorted by time
    assert [e["kind"] for e in ev] == ["drug", "washout"] and ev[0]["compounds"] == ["5-HT"] and fl == []
    ev, fl = tc.parse_tags([10.0, 50.0, 300.0], ["seal ok", "E2 wash in", "wash out"], S)
    assert [e["kind"] for e in ev] == ["other", "drug", "washout"] and "other_tags=1" in fl
    ev, fl = tc.parse_tags([100.0, 500.0], ["A", "B"], S)          # no keyword: by order, flagged
    assert [e["kind"] for e in ev] == ["drug", "washout"] and "tags_assigned_by_order" in fl
    assert "no_drug_tag" in tc.parse_tags([], [], S)[1]


def test_compound_names():
    n = S.drug_names
    assert tc.find_compounds("5HT hit bath", n) == ["5-HT"]
    assert tc.find_compounds("Substance P + TTX in bath", n) == ["SP", "TTX"]
    assert tc.find_compounds("hit bath", n) == []                    # 'ba' must not match 'bath'
    assert tc.find_compounds("A5_CC_5HT_Timecourse", n) == ["5-HT"]
    assert tc.condition_slug("5-HT + Ba2+") == "5HT_Ba2" and tc.condition_slug("5-HT") == "5HT"


def _apps(times, texts, duration, book=None, protocol=""):
    ev, _ = tc.parse_tags(times, texts, S)
    apps = tc.build_applications(ev, duration, S)
    tc.name_applications(apps, book, protocol, S)
    return apps


def test_single_application_windows():
    (a,) = _apps([154.3, 741.1], ["5ht hit bath", "washout"], 1207.6)
    w = a["windows"]
    assert w["baseline"] == pytest.approx((54.3, 154.3)) and w["pre_tag"] is None
    assert w["drug"] == pytest.approx((214.3, 414.3))                # 60 s after the tag, 200 s long
    assert w["washout"] == pytest.approx((1107.6, 1207.6)) and a["flags"] == []
    assert a["condition"] == "5-HT" and a["name_source"] == "tag" and not a["layered"]
    (a,) = _apps([40.0], ["5ht hit bath"], 600.0)
    assert a["windows"]["baseline"] == (0.0, 40.0) and "baseline_period_short=40s" in a["flags"]
    assert a["windows"]["washout"] is None and "no_washout_tag" in a["flags"]
    (a,) = _apps([100.0, 300.0], ["5ht hit bath", "washout"], 600.0)
    assert a["windows"]["drug"] == (160.0, 300.0) and "drug_period_cut_by_next_tag" in a["flags"]


def test_solitary_then_layered_applications():
    apps = _apps([100, 500, 900, 1300, 1700], ["5ht hit bath", "washout", "SP hit bath", "Ba applied", "washout"],
                 2200.0)
    a1, a2, a3 = apps
    # solitary second application: its own pre-tag baseline, after the previous washout
    assert a2["condition"] == "SP" and not a2["layered"] and a2["windows"]["baseline"] == (800.0, 900.0)
    assert a1["windows"]["washout"] == (800.0, 900.0)                # last 100 s before the next drug tag
    # layered: cumulative condition, initial baseline of the chain, pre-tag period reported
    assert a3["condition"] == "SP + Ba2+" and a3["layered"] and a3["compounds_added"] == ["Ba2+"]
    assert a3["windows"]["baseline"] == (800.0, 900.0) and a3["windows"]["pre_tag"] == (1200.0, 1300.0)
    assert a2["windows"]["drug"] == (960.0, 1160.0) and a3["windows"]["washout"] == (2100.0, 2200.0)
    assert "layered_on_previous_compound" in a3["flags"] and "after_previous_application" in a2["flags"]


def test_drug_name_from_book_and_mismatch():
    (a,) = _apps([100.0, 500.0], ["hit bath", "washout"], 900.0, book={"Drug 1": "Substance P", "Drug 2": None})
    assert a["condition"] == "SP" and a["name_source"] == "recording book" and "drug_name_from_book" in a["flags"]
    (a,) = _apps([100.0], ["hit bath"], 900.0)
    assert a["condition"] == "unknown" and "drug_name_unknown" in a["flags"]
    (a,) = _apps([100.0], ["5ht hit bath"], 900.0, book={"Drug 1": "SP"}, protocol="A5_CC_SP_Timecourse")
    assert "drug_name_mismatch_book" in a["flags"] and "drug_name_mismatch_protocol" in a["flags"]


def _trace():
    """600 s at 1 kHz: Vm -50, hyperpolarises to -60 between 200 and 300 s, back by 400 s."""
    t = np.arange(0, 600, 0.001)
    v = np.full_like(t, -50.0)
    v -= 10 * np.clip((t - 200) / 20, 0, 1) * np.clip((400 - t) / 100, 0, 1)
    return t, v


def test_bins_and_peak_response():
    t, v = _trace()
    bins = tc.bin_timecourse(t, v, [5.0, 5.5, 105.0], t0=200.0, bin_s=10.0, keep=np.ones(t.size, bool))
    assert len(bins) == 60 and bins.loc[0, "n_spikes"] == 2 and bins.loc[0, "rate_Hz"] == pytest.approx(0.2)
    assert bins.loc[0, "t_rel_drug_s"] == pytest.approx(-195.0)
    pk = tc.peak_response(bins, 200.0, 450.0, vm_base=-50.0, s=S)
    assert pk["drug_min_vm_mV"] == pytest.approx(-60, abs=0.5)
    assert pk["drug_min_delta_vm_mV"] == pytest.approx(-10, abs=0.5)
    assert 20 <= pk["drug_min_time_s"] <= 100
    assert pk["drug_max_delta_vm_mV"] == pytest.approx(0, abs=0.5)
    assert pk["drug_max_rate_Hz"] == 0.0
    assert pk["drug_peak_direction"] == "hyperpolarising" and pk["drug_peak_delta_vm_mV"] == pytest.approx(-10, abs=0.5)
    up = tc.peak_response(tc.bin_timecourse(t, -100 - v, [], 200.0, 10.0, np.ones(t.size, bool)), 200.0, 450.0,
                          vm_base=-50.0, s=S)                          # mirror image: depolarising by 10 mV
    assert up["drug_peak_direction"] == "depolarising" and up["drug_peak_delta_vm_mV"] == pytest.approx(10, abs=0.5)


def test_peak_response_without_drug_tag():
    t, v = _trace()
    bins = tc.bin_timecourse(t, v, [], 0.0, 10.0, np.ones(t.size, bool))
    pk = tc.peak_response(bins, np.nan, np.nan, -50.0, S)
    assert np.isnan(pk["drug_min_delta_vm_mV"])


def test_timecourse_on_example(example_abf, tmp_path):
    p = example_abf("20261026_7503_A_0004.abf")
    sel = pd.DataFrame([{"recording_id": "A0004", "cell_tag": "A", "abf_filename": p.name, "original_path": str(p),
                         "voltage_source": "ADC0", "voltage_scale": 1.0, "bias_current_pA": np.nan}])
    spikes, det = tc.detect_all(sel, SpikeSettings(), S, BaselineSettings(), log=lambda m: None,
                                figures_dir=tmp_path)
    assert (tmp_path / "A0004.png").exists() and (tmp_path / "A0004.svg").exists()
    periods, summary, bins = tc.feature_table(sel, spikes, S, BaselineSettings(), log=lambda m: None)
    assert list(periods["period"]) == ["baseline", "drug", "washout"]
    assert len(summary) == 1 and summary.loc[0, "condition"] == "5-HT" and summary.loc[0, "application"] == 1
    assert summary.loc[0, "drug_tag_s"] == pytest.approx(154.3, abs=0.1)
    assert summary.loc[0, "drug_min_delta_vm_mV"] < -8           # 5-HT hyperpolarises this cell
    assert summary.loc[0, "drug_start_s"] == pytest.approx(154.3 + 60, abs=0.1)
    assert periods.loc[periods["period"] == "drug", "silent_max_spikes_used"].iloc[0] == 20
    assert summary.loc[0, "delta_drug_vm_baseline_mV"] < 0
