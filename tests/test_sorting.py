import os

import pandas as pd

from ephys_analysis.classification import copy_sorted
from ephys_analysis.metadata import sha256_file
from ephys_analysis.utils import load_config

from conftest import ROOT

PROTO = load_config(ROOT / "01_ephys_meta-org")


def setup(tmp_path, status="confirmed", family="membrane_test"):
    raw = tmp_path / "raw" / "day1"
    raw.mkdir(parents=True)
    src = raw / "x_0000.abf"
    src.write_bytes(b"fake abf content")
    cls = pd.DataFrame([{"recording_id": "x_0000__1", "abf_filename": src.name, "original_path": str(src),
                         "label": "A0", "abf_family": family, "classification_status": status, "review_reason": ""}])
    recs = pd.DataFrame([{"recording_id": "x_0000__1", "sha256": sha256_file(src)}])
    return src, cls, recs


def test_copy_preserves_original_and_verifies(tmp_path):
    src, cls, recs = setup(tmp_path)
    before = (sha256_file(src), os.stat(src).st_mtime_ns)
    m = copy_sorted(cls, recs, PROTO, tmp_path / "sorted", log=lambda *_: None)
    dst = tmp_path / "sorted" / "05_membrane_test_A0_M1" / "day1" / "x_0000.abf"
    assert m.loc[0, "copy_status"] == "copied" and m.loc[0, "verified_sha256"]
    assert dst.exists() and src.exists()
    assert (sha256_file(src), os.stat(src).st_mtime_ns) == before


def test_rerun_and_no_overwrite(tmp_path):
    src, cls, recs = setup(tmp_path)
    copy_sorted(cls, recs, PROTO, tmp_path / "sorted", log=lambda *_: None)
    m = copy_sorted(cls, recs, PROTO, tmp_path / "sorted", log=lambda *_: None)
    assert m.loc[0, "copy_status"] == "already_present"
    dst = tmp_path / "sorted" / "05_membrane_test_A0_M1" / "day1" / "x_0000.abf"
    dst.write_bytes(b"something else")
    m = copy_sorted(cls, recs, PROTO, tmp_path / "sorted", log=lambda *_: None)
    assert m.loc[0, "copy_status"] == "skipped_exists_different_content"
    assert dst.read_bytes() == b"something else"
    m = copy_sorted(cls, recs, PROTO, tmp_path / "sorted", overwrite=True, log=lambda *_: None)
    assert m.loc[0, "copy_status"] == "copied" and dst.read_bytes() == b"fake abf content"


def test_review_statuses_go_to_review_folder(tmp_path):
    for status in ("mismatch", "ambiguous", "unsupported"):
        sub = tmp_path / status
        src, cls, recs = setup(sub, status=status)
        m = copy_sorted(cls, recs, PROTO, sub / "sorted", log=lambda *_: None)
        assert m.loc[0, "destination_folder"] == PROTO["review_folder"]


def test_epsc_goes_to_separate_folder(tmp_path):
    src, cls, recs = setup(tmp_path, family="epsc")
    m = copy_sorted(cls, recs, PROTO, tmp_path / "sorted", log=lambda *_: None)
    assert m.loc[0, "destination_folder"] == "06_EPSC_separate_analysis"
