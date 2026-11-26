from pathlib import Path

import pandas as pd

from ephys_analysis.metadata import build_cell_master, make_recording_id, match_recordings, normalise_key


def recs(*stems, folders=None):
    folders = folders or ["d"] * len(stems)
    return pd.DataFrame({"recording_id": [f"{s}__{i}" for i, s in enumerate(stems)],
                         "recording_stem": list(stems), "abf_filename": [s + ".abf" for s in stems],
                         "original_path": [f"/{f}/{s}.abf" for f, s in zip(folders, stems)]})


def book(rows):
    return {"recording_log": pd.DataFrame(rows, columns=["Recording Name", "Cell Tag", "Animal ID", "Slice", "Protocol"])}


def test_normalise_key():
    assert normalise_key("X_0001.abf ") == "x_0001"
    assert normalise_key(r"C:\a\X_0001.ABF") == "x_0001"
    assert normalise_key(None) == ""


def test_recording_id_unique_for_same_filename_in_different_folders(tmp_path):
    a, b = tmp_path / "day1" / "c_0000.abf", tmp_path / "day2" / "c_0000.abf"
    assert make_recording_id(a, tmp_path) != make_recording_id(b, tmp_path)
    assert make_recording_id(a, tmp_path) == make_recording_id(a, tmp_path)
    assert make_recording_id(a, tmp_path).startswith("c_0000__")


def test_match_outcomes():
    r = recs("c_A_0000", "c_A_0001", "c_A_0002", "dup_0000", "dup_0000", folders=list("dddxy"))
    b = book([["c_A_0000", "c_A", "M1", "S1", "A1"],
              ["c_A_0001", "c_A", "M1", "S1", "A3"], ["c_A_0001", "c_A", "M1", "S1", "A4"],
              ["dup_0000", "dup", "M1", "S1", "A1"],
              ["only_in_book", "z", "M1", "S1", "A1"]])
    out, unmatched = match_recordings(r, b)
    st = dict(zip(out["recording_stem"] + out["recording_id"].str[-1], out["match_status"]))
    assert st["c_A_00000"] == "matched"
    assert st["c_A_00011"] == "duplicate_book_entry"
    assert st["c_A_00022"] == "unmatched_abf"
    assert st["dup_00003"] == st["dup_00004"] == "duplicate_filename"
    assert len(out) == 5                                    # no rows lost or multiplied
    # book fields merged only for the unique match
    assert out.loc[out["match_status"] != "matched", "Protocol"].isna().all()
    # Cell Tag never invented from the filename
    missing = out[out["match_status"] != "matched"]
    assert missing["cell_tag"].isna().all()
    assert (missing["cell_tag_source"] == "missing").all()
    assert "c_A" in set(out["cell_tag_candidate_from_filename"])
    assert "only_in_book" in set(unmatched["Recording Name"])


def test_cell_master_links_recordings_and_flags_conflicts():
    rm = pd.DataFrame({"cell_tag": ["c1", "c1", "c2", None],
                       "recording_id": ["r1", "r2", "r3", "r4"], "recording_stem": ["a", "b", "c", "d"],
                       "abf_filename": ["a.abf", "b.abf", "c.abf", "d.abf"],
                       "Animal ID": ["M1", "M2", "M1", "M1"], "abf_family": ["membrane_test", "ramp", "ramp", "ramp"],
                       "classification_status": ["confirmed", "mismatch", "confirmed", "confirmed"]})
    cm = build_cell_master(rm).set_index("cell_tag")
    assert list(cm.index) == ["c1", "c2"]
    assert cm.loc["c1", "n_recordings"] == 2
    assert cm.loc["c1", "recording_ids"] == "r1 | r2"
    assert cm.loc["c1", "metadata_conflicts"] == "Animal ID"
    assert cm.loc["c1", "n_needing_review"] == 1
