"""Cell-level summary (cell_summary.py): recording selection and join."""

import numpy as np
import pandas as pd

from ephys_analysis.cell_summary import build_summary, select_recordings


def _mt():
    return pd.DataFrame([
        {"cell_tag": "B", "abf_filename": "B_0006.abf", "recording_id": "b6", "qc_status": "fail", "Ra_MOhm": np.nan},
        {"cell_tag": "B", "abf_filename": "B_0000.abf", "recording_id": "b0", "qc_status": "pass", "Ra_MOhm": 26.0},
        {"cell_tag": "B", "abf_filename": "B_0008.abf", "recording_id": "b8", "qc_status": "pass", "Ra_MOhm": 28.0},
        {"cell_tag": "A", "abf_filename": "A_0005.abf", "recording_id": "a5", "qc_status": "fail", "Ra_MOhm": np.nan},
    ])


def test_first_usable_recording_in_file_order():
    sel = select_recordings(_mt(), "membrane_test", [], {}).set_index("cell_tag")
    assert sel.loc["B", "abf_filename"] == "B_0000.abf" and sel.loc["B", "n_recordings"] == 3
    assert "B_0006.abf (membrane test QC fail)" in sel.loc["B", "alternatives"]
    assert sel.loc["A", "abf_filename"] == "" and "no usable recording" in sel.loc["A", "reason"]


def test_flagged_recording_is_second_choice_and_override_wins():
    base = pd.DataFrame([
        {"cell_tag": "B", "abf_filename": "B_0001.abf", "recording_id": "b1", "flags": "abf_tags_present"},
        {"cell_tag": "B", "abf_filename": "B_0010.abf", "recording_id": "b10", "flags": ""},
        {"cell_tag": "C", "abf_filename": "C_0001.abf", "recording_id": "c1", "flags": "abf_tags_present"},
    ])
    sel = select_recordings(base, "baseline", ["abf_tags_present"], {}).set_index("cell_tag")
    assert sel.loc["B", "abf_filename"] == "B_0010.abf"
    assert sel.loc["C", "abf_filename"] == "C_0001.abf" and "used flagged" in sel.loc["C", "reason"]
    sel = select_recordings(base, "baseline", ["abf_tags_present"], {"B": {"baseline": "B_0001.abf"}})
    assert sel.set_index("cell_tag").loc["B", "abf_filename"] == "B_0001.abf"


def test_build_summary_one_row_per_cell_and_missing_protocol():
    cells = pd.DataFrame({"cell_tag": ["A", "B"], "Sex": ["F", "F"]})
    cfg = {"cell_metadata_columns": ["Sex"], "overrides": {},
           "protocols": {"membrane_test": {"prefix": "mt", "columns": ["Ra_MOhm", "qc_status"], "exclude_if_flag": []},
                         "ramp": {"prefix": "ramp", "columns": ["I_first_spike_pA"], "exclude_if_flag": []}}}
    summary, selection = build_summary(cells, {"membrane_test": _mt(), "ramp": pd.DataFrame()}, cfg, log=lambda m: None)
    assert len(summary) == 2 and list(summary["cell_tag"]) == ["A", "B"]
    s = summary.set_index("cell_tag")
    assert s.loc["B", "mt_Ra_MOhm"] == 26.0 and s.loc["B", "mt_recording"] == "B_0000.abf"
    assert np.isnan(s.loc["A", "mt_Ra_MOhm"]) and s.loc["A", "mt_n_recordings"] == 1
    assert (summary["ramp_n_recordings"] == 0).all()


def test_timecourse_split_by_condition():
    cells = pd.DataFrame({"cell_tag": ["A", "B"]})
    tcs = pd.DataFrame([
        {"cell_tag": "A", "abf_filename": "A_0004.abf", "recording_id": "a4", "application": 1, "condition": "5-HT",
         "d": -9.0, "flags": ""},
        {"cell_tag": "A", "abf_filename": "A_0004.abf", "recording_id": "a4", "application": 2, "condition": "SP",
         "d": 3.0, "flags": ""},
        {"cell_tag": "A", "abf_filename": "A_0004.abf", "recording_id": "a4", "application": 3, "condition": "5-HT",
         "d": -7.0, "flags": ""},                                   # repeat in the same recording: not used
        {"cell_tag": "B", "abf_filename": "B_0004.abf", "recording_id": "b4", "application": 1,
         "condition": "5-HT + Ba2+", "d": -2.0, "flags": ""},
    ])
    cfg = {"overrides": {}, "protocols": {"timecourse": {"prefix": "tc", "split_by": "condition",
                                                         "order_by": "application", "columns": ["d"]}}}
    summary, selection = build_summary(cells, {"timecourse": tcs}, cfg, log=lambda m: None)
    s = summary.set_index("cell_tag")
    assert s.loc["A", "tc_5HT_d"] == -9.0 and s.loc["A", "tc_SP_d"] == 3.0 and np.isnan(s.loc["B", "tc_5HT_d"])
    assert s.loc["B", "tc_5HT_Ba2_d"] == -2.0
    assert set(selection["protocol"]) == {"timecourse:5-HT", "timecourse:SP", "timecourse:5-HT + Ba2+"}
