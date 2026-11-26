"""New 01 checks: amplifier header, baseline 0 pA, recording-book comments and dates."""

import json

import numpy as np
import pytest

from conftest import EXAMPLE_DIR, ROOT
from ephys_analysis.anomalies import book_flags, header_flags, parse_current_comment
from ephys_analysis.metadata import extract_abf_metadata
from ephys_analysis.utils import load_config

CFG = load_config(ROOT / "01_ephys_meta-org")
PAT = CFG["expected"]["current_comment"]


def rec(**kw):
    base = {"telegraph_enabled": "[1, 1]", "telegraph_mode": "[1, 1]", "dac_scale_factor": "[20, 400]",
            "dac_holding_level": "[0, 0]", "dac_kinds": "voltage | current",
            "dac_summary_json": json.dumps({"1": {"kind": "current", "per_sweep_min": [0.0], "per_sweep_max": [0.0]}})}
    base.update(kw)
    return base


def names(out):
    return [f for f, *_ in out]


def test_expected_current_clamp_setup_has_no_flags():
    assert header_flags(rec(), "current_clamp", "spontaneous", [1], CFG) == []


def test_telegraph_off_and_mode_mismatch():
    assert names(header_flags(rec(telegraph_enabled="[0, 0]"), "current_clamp", "ramp", [1], CFG)) == ["telegraph_off"]
    assert names(header_flags(rec(telegraph_mode="[0, 0]"), "current_clamp", "ramp", [1], CFG)) == ["telegraph_mode_mismatch"]


def test_unexpected_command_sensitivity():
    out = header_flags(rec(dac_scale_factor="[20, 2000]"), "current_clamp", "current_steps", [1], CFG)
    assert names(out) == ["unexpected_command_sensitivity"] and out[0][3] == 2000


def test_baseline_must_be_zero_pA():
    held = rec(dac_holding_level="[0, -10]")
    assert names(header_flags(held, "current_clamp", "spontaneous", [1], CFG)) == ["baseline_not_zero_current"]
    stepped = rec(dac_summary_json=json.dumps({"1": {"kind": "current", "per_sweep_min": [-50.0], "per_sweep_max": [0.0]}}))
    assert names(header_flags(stepped, "current_clamp", "modulation_timecourse", [1], CFG)) == ["baseline_not_zero_current"]
    # the same command is fine for a step protocol
    assert header_flags(stepped, "current_clamp", "current_steps", [1], CFG) == []


@pytest.mark.parametrize("text,mentioned,value", [
    ("-10pA holding, longer time until effect", True, -10.0),
    ("Injected -10pA current to hyperpolarise", True, -10.0),
    ("held at + 25 pA", True, 25.0),
    ("injected some current", True, np.nan),
    ("Spontaneous firing", False, np.nan),
    ("Looked good, hard to open (Zapped a lot)", False, np.nan),
    (np.nan, False, np.nan),
])
def test_parse_current_comment(text, mentioned, value):
    m, v = parse_current_comment(text, PAT)
    assert m == mentioned
    assert (np.isnan(v) and np.isnan(value)) or v == value


def test_book_comment_only_flagged_in_current_clamp():
    r = {"Protocol Comments": "-10pA holding", "Date Recorded": "2026-10-26", "recording_start": "2026-10-26T15:00:00"}
    out = book_flags(r, "current_clamp", CFG)
    assert names(out) == ["book_mentions_current"] and out[0][3] == -10.0
    assert book_flags(r, "voltage_clamp", CFG) == []


def test_book_date_mismatch():
    r = {"Date Recorded": "2026-10-27 00:00:00", "recording_start": "2026-10-26T15:00:00"}
    assert names(book_flags(r, "voltage_clamp", CFG)) == ["book_date_mismatch"]


def test_real_files_match_expected_setup(example_abf):
    for name, mode, fam in (("20261026_7503_A_0001.abf", "current_clamp", "current_steps"),
                            ("20261026_7503_B_0000.abf", "voltage_clamp", "membrane_test"),
                            ("20261026_7503_B_0004.abf", "current_clamp", "modulation_timecourse")):
        meta = extract_abf_metadata(example_abf(name), EXAMPLE_DIR)
        assert json.loads(meta["telegraph_enabled"])[0] == 1
        assert header_flags(meta, mode, fam, [1], CFG) == []
