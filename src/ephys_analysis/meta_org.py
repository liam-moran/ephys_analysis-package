"""01_ephys_meta-org steps: inventory -> recording book -> classify -> noise -> check -> save.

Each step is one function, so it can be run and reviewed cell by cell in the
notebook, or all at once by ``run.py``. Raw data are only read.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .anomalies import build_flags, flag_summary, recording_status
from .classification import classify_table
from .noise import NoiseSettings, noise_table
from .metadata import (book_columns, build_cell_master, extract_folder, match_recordings, read_recording_book,
                       sha256_file)
from .utils import clear_output, load_config, repo_path, stamp, write_csv

PROJECT = "01_ephys_meta-org"


def setup(repo_root: Path | str, fresh: bool = False) -> tuple[dict, Path]:
    """Load the config. With ``fresh=True`` the subproject's ``output/`` is cleared first
    (derived files only; ``data/`` is never touched)."""
    cfg = load_config(Path(repo_root) / PROJECT)
    out = cfg["_project_dir"] / "output"
    if fresh:
        clear_output(out)
    out.mkdir(parents=True, exist_ok=True)
    return cfg, out


def inventory(cfg: dict, log=print) -> pd.DataFrame:
    """Step 1: read the header of every ABF under ``paths.abf_root`` (one row per file)."""
    root = repo_path(cfg, cfg["paths"]["abf_root"])
    log(f"ABF root: {root}")
    return extract_folder(root, cfg["classification"]["rules"]["flat_command_tolerance"], log=log)


def match_book(recordings: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Step 2: attach recording-book rows. Returns (recordings, book_issues, unmatched_book_rows)."""
    b = cfg["recording_book"]
    book, issues = read_recording_book(repo_path(cfg, cfg["paths"]["recording_book"]), b["sheets"], b["error_values"])
    matched, unmatched = match_recordings(recordings, book, b["key_column"], book_columns(cfg))
    return matched, issues, unmatched


def classify(recordings: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Step 3: protocol family and classification status from the ABF evidence."""
    return classify_table(recordings, cfg, book_columns(cfg)["protocol"])


def noise(recordings: pd.DataFrame, classification: pd.DataFrame, cfg: dict, out: Path | None = None,
          log=print) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Step 4: noise QC of every recording (src/ephys_analysis/noise.py): power spectrum of the
    quiet stretches, RMS noise, mains harmonics, low-frequency (pump) peaks, brief transient
    artefacts. Saves one spectrum figure per recording (``output/noise_spectra/``) when ``out``
    is given. Returns (noise table, noise flags)."""
    s = NoiseSettings.from_config(cfg.get("noise"))
    figs = None if out is None else Path(out) / "noise_spectra"
    tab, rows = noise_table(recordings, classification, s, figs, tuple(cfg.get("figure_formats", ["png", "svg"])), log)
    from .anomalies import FLAG_COLUMNS
    return tab, pd.DataFrame(rows, columns=FLAG_COLUMNS)


def check(recordings: pd.DataFrame, classification: pd.DataFrame, book_issues: pd.DataFrame,
          unmatched: pd.DataFrame, cfg: dict, log=print,
          noise_flags: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Step 5: all checks (plus the noise flags of step 4). Returns (recording table with
    status/flags, flag table)."""
    flags = build_flags(recordings, classification, book_issues, unmatched, cfg, log=log)
    if noise_flags is not None and len(noise_flags):
        flags = pd.concat([flags, noise_flags], ignore_index=True)
    table = recordings.merge(
        classification[["recording_id", "label", "label_source", "clamp_mode", "abf_family", "label_family",
                        "classification_status", "command_dac", "evidence", "review_reason"]],
        on="recording_id", how="left")
    table["status"] = recording_status(flags, table["recording_id"])
    table["flags"] = flag_summary(flags, table["recording_id"])
    first = ["recording_id", "cell_tag", "abf_filename", "abf_family", "classification_status", "status", "flags"]
    return table[first + [c for c in table.columns if c not in first]], flags


def save(table: pd.DataFrame, flags: pd.DataFrame, out: Path, run_id: str,
         noise_tab: pd.DataFrame | None = None) -> list[Path]:
    """Step 6: write recordings.csv, cells.csv, flags.csv (and noise.csv)."""
    paths = [out / "recordings.csv", out / "cells.csv", out / "flags.csv"]
    write_csv(stamp(table, run_id), paths[0])
    write_csv(stamp(build_cell_master(table), run_id), paths[1])
    write_csv(stamp(flags, run_id), paths[2])
    if noise_tab is not None:
        paths.append(out / "noise.csv")
        write_csv(stamp(noise_tab, run_id), paths[3])
    return paths


def originals_unchanged(table: pd.DataFrame) -> list[str]:
    """Re-hash every original ABF; returns the names of files that changed (should be empty)."""
    return [r["abf_filename"] for _, r in table.iterrows() if sha256_file(r["original_path"]) != r["sha256"]]
