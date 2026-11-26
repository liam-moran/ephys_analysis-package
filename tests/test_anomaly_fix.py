"""02: auto-resolution, decision sheet, resolution of statuses, organising copies."""

import numpy as np
import pandas as pd
import pytest

from conftest import ROOT
from ephys_analysis import anomaly_fix as af
from ephys_analysis.metadata import sha256_file
from ephys_analysis.utils import load_config

CFG = load_config(ROOT / "02_ephys_anomaly-fix")
META = load_config(ROOT / "01_ephys_meta-org")
GAIN = CFG["auto_resolve"]["possible_gain_error"]


def recordings(tmp_path=None):
    rows = [("r_spont", "spontaneous", "voltage | voltage", "1"), ("r_steps", "current_steps", "voltage | voltage", "1"),
            ("r_mt", "membrane_test", "current | voltage", "1"), ("r_bad", "spontaneous", "voltage | voltage", "1")]
    df = pd.DataFrame(rows, columns=["recording_id", "abf_family", "adc_kinds", "command_dac"])
    df["abf_filename"] = df["recording_id"] + ".abf"
    df["cell_tag"], df["label"], df["clamp_mode"] = "c1", "", ""
    if tmp_path is not None:
        paths, shas = [], []
        for rid in df["recording_id"]:
            p = tmp_path / "raw" / "day1" / f"{rid}.abf"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(rid.encode())
            paths.append(str(p)); shas.append(sha256_file(p))
        df["original_path"], df["sha256"] = paths, shas
    return df


def flags(*rows):
    return pd.DataFrame([dict(zip(["recording_id", "flag", "severity", "value"], r), abf_filename=r[0] + ".abf",
                              detail="d") for r in rows])


@pytest.mark.parametrize("median,expected", [(-1200, 20), (-600, 10), (-5000, 100), (-60, None), (-800, None), (5, None)])
def test_choose_gain(median, expected):
    g, _ = af.choose_gain(median, GAIN)
    assert g == expected


def test_auto_resolve_rules():
    f = flags(("r_spont", "duplicated_voltage_no_current", "needs_fix", np.nan),
              ("r_steps", "duplicated_voltage_no_current", "needs_fix", np.nan),
              ("r_bad", "possible_gain_error", "needs_fix", -1200.0))
    auto = af.auto_resolve(recordings(), f, CFG).set_index("recording_id")
    assert auto.loc["r_spont", "action"] == "voltage_only"
    assert auto.loc["r_steps", "action"] == "command_as_current"
    assert (auto.loc["r_bad", "action"], auto.loc["r_bad", "value"]) == ("rescale", 20)


def test_no_auto_resolution_when_sensitivity_unexpected():
    f = flags(("r_steps", "duplicated_voltage_no_current", "needs_fix", np.nan),
              ("r_steps", "unexpected_command_sensitivity", "review", 2000.0))
    assert af.auto_resolve(recordings(), f, CFG).empty


def _cfg(tmp_path):
    cfg = dict(CFG)
    cfg["decision_sheet"] = str(tmp_path / "decisions.xlsx")
    return cfg


def test_decision_sheet_appends_and_preserves(tmp_path):
    cfg = _cfg(tmp_path)
    f1 = flags(("r_spont", "book_mentions_current", "review", -10.0))
    sheet, n = af.update_decision_sheet(f1, recordings(), cfg)
    assert n == 1 and sheet.loc[0, "suggested_action"] == "bias_current" and sheet.loc[0, "suggested_value"] == "-10"
    sheet.loc[0, ["decision", "value"]] = ["bias_current", "-12"]
    af.write_decision_sheet(sheet, tmp_path / "decisions.xlsx", cfg)
    f2 = pd.concat([f1, flags(("r_bad", "protocol_mismatch", "review", np.nan))])
    sheet, n = af.update_decision_sheet(f2, recordings(), cfg)
    assert n == 1 and len(sheet) == 2
    assert sheet.loc[0, "value"] == "-12"          # the filled-in decision is untouched


def test_stamp_decisions_fills_only_empty_cells():
    s = pd.DataFrame([
        {"decision": "accept", "decided_by": "", "decided_on": ""},
        {"decision": "exclude", "decided_by": "XY", "decided_on": "2026-01-01"},
        {"decision": "", "decided_by": "", "decided_on": ""},
    ])
    out = af.stamp_decisions(s, "LM", today="2026-11-25")
    assert out.loc[0, ["decided_by", "decided_on"]].tolist() == ["LM", "2026-11-25"]
    assert out.loc[1, ["decided_by", "decided_on"]].tolist() == ["XY", "2026-01-01"]
    assert out.loc[2, ["decided_by", "decided_on"]].tolist() == ["", ""]
    assert af.stamp_decisions(s, "", today="2026-11-25").loc[0, "decided_by"] == ""


def test_validation():
    s = pd.DataFrame([
        {"flag": "protocol_mismatch", "allowed_actions": "relabel, accept, exclude", "decision": "relabel", "value": "ramp", "note": ""},
        {"flag": "protocol_mismatch", "allowed_actions": "relabel, accept, exclude", "decision": "relabel", "value": "nonsense", "note": ""},
        {"flag": "protocol_mismatch", "allowed_actions": "relabel, accept, exclude", "decision": "exclude", "value": "", "note": ""},
        {"flag": "book_mentions_current", "allowed_actions": "bias_current, accept, exclude", "decision": "rescale", "value": "2", "note": ""},
        {"flag": "book_mentions_current", "allowed_actions": "bias_current, accept, exclude", "decision": "bias_current", "value": "", "note": ""},
        {"flag": "book_mentions_current", "allowed_actions": "bias_current, accept, exclude", "decision": "", "value": "", "note": ""},
    ])
    st = af.validate_decisions(s, CFG)["decision_status"].tolist()
    assert st[0] == "ok" and st[5] == ""
    assert all(x.startswith("invalid") for x in st[1:5])


def test_resolve_states_and_corrections():
    recs = recordings()
    f = flags(("r_spont", "duplicated_voltage_no_current", "needs_fix", np.nan),
              ("r_spont", "book_mentions_current", "review", -10.0),
              ("r_steps", "duplicated_voltage_no_current", "needs_fix", np.nan),
              ("r_bad", "protocol_mismatch", "review", np.nan),
              ("", "book_rows_without_abf", "info", np.nan))
    auto = af.auto_resolve(recs, f, CFG)
    sheet = pd.DataFrame([
        {**{c: "" for c in af.SHEET_COLUMNS}, "recording_id": "r_spont", "flag": "book_mentions_current",
         "allowed_actions": "bias_current, accept, exclude", "decision": "bias_current", "value": "-10"},
        {**{c: "" for c in af.SHEET_COLUMNS}, "recording_id": "r_bad", "flag": "protocol_mismatch",
         "allowed_actions": "relabel, accept, exclude", "decision": "", "value": ""}])
    status, issues = af.resolve(recs, f, auto, af.validate_decisions(sheet, CFG))
    st = status.set_index("recording_id")
    assert st.loc["r_spont", "state"] == "ready" and st.loc["r_spont", "bias_current_pA"] == -10
    assert st.loc["r_spont", "current_source"] == "none"
    assert st.loc["r_steps", "current_source"] == "command_DAC1"
    assert st.loc["r_mt", "state"] == "ready" and st.loc["r_mt", "current_source"] == "measured_ADC0"
    assert st.loc["r_bad", "state"] == "pending"
    assert set(issues["resolution"]) == {"auto_resolved", "decided", "pending", "info"}
    assert len(issues) == len(f)                  # every flag appears in the report


def test_relabel_exclude_and_rescale_corrections():
    r = recordings().iloc[0]
    c = af.corrections_for(r, {"relabel": "ramp", "rescale": "20"})
    assert c["family"] == "ramp" and c["voltage_scale"] == pytest.approx(0.05)


def test_organise_copies_only_ready_and_removes_stale(tmp_path):
    recs = recordings(tmp_path)
    before = {p: sha256_file(p) for p in recs["original_path"]}
    status = pd.DataFrame({"recording_id": recs["recording_id"], "state": ["ready", "ready", "ready", "pending"],
                           "family": ["spontaneous", "current_steps", "membrane_test", "spontaneous"]})
    out = tmp_path / "out"
    m = af.organise(recs, status, META, out, log=lambda *_: None)
    copied = sorted(p.name for p in (out / "protocol_sorted").rglob("*.abf"))
    assert copied == ["r_mt.abf", "r_spont.abf", "r_steps.abf"]
    assert m["verified_sha256"].all()
    # r_steps becomes pending -> its derived copy is removed
    status.loc[1, "state"] = "pending"
    af.organise(recs, status, META, out, log=lambda *_: None)
    assert sorted(p.name for p in (out / "protocol_sorted").rglob("*.abf")) == ["r_mt.abf", "r_spont.abf"]
    assert {p: sha256_file(p) for p in recs["original_path"]} == before   # originals untouched
