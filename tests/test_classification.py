import json

import pytest

from ephys_analysis.classification import classify_row, label_family
from ephys_analysis.utils import load_config
from ephys_analysis.metadata import extract_abf_metadata

from conftest import EXAMPLE_DIR, ROOT

CFG = load_config(ROOT / "01_ephys_meta-org")


def dac(i, kind, flat=True, ramp=False, amps=(0.0,), pulses=(), varies=False, name=""):
    return {"dac": i, "name": name or f"Cmd {i}", "kind": kind, "is_flat": flat, "has_ramp": ramp,
            "step_amplitudes": list(amps), "pulse_durations_s": list(pulses),
            "amplitude_varies_across_sweeps": varies, "epoch_types": []}


def row(adc_kinds, dacs, dur=180.0, sweeps=1, sweep_len=None, tags=()):
    return {"read_status": "ok", "adc_kinds": " | ".join(adc_kinds), "total_duration_s": dur,
            "sweep_length_s": sweep_len if sweep_len is not None else dur / sweeps, "sweep_count": sweeps,
            "dac_summary_json": json.dumps({str(d["dac"]): d for d in dacs}), "tag_comments": json.dumps(list(tags))}


VC_MT = row(["current", "voltage"],
            [dac(0, "voltage", flat=False, amps=[112, 92, 72], pulses=[0.1]),        # decoy, too large
             dac(1, "voltage", flat=False, amps=[-10] * 5, pulses=[0.025])], dur=10, sweeps=5)


@pytest.mark.parametrize("label,expected", [
    ("A0_VC_Membrane_Test", ["membrane_test"]), ("VC_Membrane_Test", ["membrane_test"]),
    ("A3_epsc", ["epsc"]), ("A3_epsc_washout", ["epsc"]), ("A3_CC_FI_Steps_Baseline", ["current_steps"]),
    ("A4_CC_Ramp_Pyramid_Inverse", ["ramp"]), ("A5_CC_5HT_Timecourse", ["modulation_timecourse"]),
    ("A1_CC_Baseline_Spontaneous", ["spontaneous"]), ("", []), (None, []),
])
def test_label_family(label, expected):
    assert label_family(label, CFG["families"]) == expected


def test_membrane_test_with_decoy_dac():
    r = classify_row(VC_MT, CFG, "A0_VC_Membrane_Test")
    assert (r["abf_family"], r["classification_status"], r["command_dac"]) == ("membrane_test", "confirmed", "1")


def test_label_is_not_used_to_decide_family():
    r = classify_row(VC_MT, CFG, "A1_CC_Baseline_Spontaneous")
    assert r["abf_family"] == "membrane_test" and r["classification_status"] == "mismatch"


def test_missing_label_gives_probable():
    assert classify_row(VC_MT, CFG, None)["classification_status"] == "probable"


def test_epsc_vs_current_clamp_mismatch():
    vc = row(["current", "voltage"], [dac(0, "voltage"), dac(1, "voltage")], dur=180)
    assert classify_row(vc, CFG, "A3_epsc")["abf_family"] == "epsc"
    cc = row(["voltage", "voltage"], [dac(0, "voltage"), dac(1, "current")], dur=180)
    r = classify_row(cc, CFG, "A3_epsc")
    assert (r["abf_family"], r["classification_status"]) == ("spontaneous", "mismatch")


def test_current_clamp_families():
    steps = row(["voltage"], [dac(1, "current", flat=False, amps=[-150, -100, 0, 50, 100], varies=True)], dur=180, sweeps=6)
    ramp = row(["voltage"], [dac(1, "current", flat=False, ramp=True)], dur=66, sweeps=3)
    long = row(["voltage"], [dac(1, "current")], dur=1200, tags=["5HT hit bath"])
    short_tagged = row(["voltage"], [dac(1, "current")], dur=120, tags=["After washout of 5HT"])
    assert classify_row(steps, CFG, None)["abf_family"] == "current_steps"
    assert classify_row(ramp, CFG, None)["abf_family"] == "ramp"
    assert classify_row(long, CFG, None)["abf_family"] == "modulation_timecourse"
    r = classify_row(short_tagged, CFG, "A1_CC_Baseline_Spontaneous")
    assert r["abf_family"] == "spontaneous" and "drug-related tag" in r["review_reason"]


def test_unknown_units_and_unreadable():
    assert classify_row(row(["unknown"], [dac(0, "voltage")]), CFG, "A1_x")["classification_status"] == "unsupported"
    assert classify_row({"read_status": "error", "read_error": "boom"}, CFG, None)["classification_status"] == "unsupported"


def test_no_rule_matches_is_ambiguous():
    tiny = row(["voltage"], [dac(1, "current")], dur=2)
    assert classify_row(tiny, CFG, "A1_CC_Baseline_Spontaneous")["classification_status"] == "ambiguous"


@pytest.mark.parametrize("name,family,status", [
    ("20261026_7503_A_0005.abf", "membrane_test", "confirmed"),
    ("20261026_7503_B_0006.abf", "membrane_test", "confirmed"),
    ("20261026_7503_A_0003.abf", "epsc", "confirmed"),
    ("20261026_7503_B_0005.abf", "spontaneous", "mismatch"),  # labelled A3_epsc, acquired in current clamp
    ("20261026_7503_A_0001.abf", "current_steps", "confirmed"),
    ("20261026_7503_A_0002.abf", "ramp", "confirmed"),
    ("20261026_7503_A_0004.abf", "modulation_timecourse", "confirmed"),
])
def test_real_examples(example_abf, name, family, status):
    meta = extract_abf_metadata(example_abf(name), EXAMPLE_DIR)
    r = classify_row(meta, CFG, meta["abf_protocol"])
    assert (r["abf_family"], r["classification_status"]) == (family, status)
