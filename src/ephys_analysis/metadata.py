"""Recording-book reading, per-ABF metadata, matching ABFs to the book, cell table."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import hashlib
import json
import re

import pandas as pd

from .abf import open_abf, readable_dacs, summarise_dac, unit_kind


# ===================================================================== workbook_reader
# Read the recording book (Excel) without modifying it.


def read_recording_book(path: Path | str, sheets: dict[str, str],
                        error_values: list[str] | None = None) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Read the configured sheets as text-preserving DataFrames.

    Returns (sheets_by_role, issues). ``issues`` lists spreadsheet error values
    (e.g. ``#REF!``) found in the workbook, with sheet, row and column. Values are
    left as they are; nothing is corrected or written back.
    """
    error_values = set(error_values or [])
    path = Path(path)
    xls = pd.ExcelFile(path)
    frames: dict[str, pd.DataFrame] = {}
    issues = []
    for role, sheet_name in sheets.items():
        if sheet_name not in xls.sheet_names:
            issues.append({"sheet": sheet_name, "row": None, "column": None,
                           "value": None, "issue": "sheet_missing"})
            continue
        df = pd.read_excel(xls, sheet_name=sheet_name, dtype=object)
        df.columns = [str(c).strip() for c in df.columns]
        frames[role] = df
        for col in df.columns:
            for idx, val in df[col].items():
                if isinstance(val, str) and val.strip() in error_values:
                    # Excel row number = DataFrame index + 2 (header row + 1-based)
                    issues.append({"sheet": sheet_name, "row": int(idx) + 2, "column": col,
                                   "value": val, "issue": "spreadsheet_error_value"})
    xls.close()
    return frames, pd.DataFrame(issues, columns=["sheet", "row", "column", "value", "issue"])


# ===================================================================== extraction
# Per-ABF metadata extraction with stable recording identifiers.


def sha256_file(path: Path | str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def make_recording_id(abf_path: Path, abf_root: Path) -> str:
    """Stable, unique recording identifier: '<stem>__<8-hex hash of relative path>'.

    The filename stem keeps it readable; the hash of the path relative to the data
    root keeps it unique when the same filename exists in different folders.
    """
    try:
        rel = abf_path.resolve().relative_to(abf_root.resolve()).as_posix()
    except ValueError:
        rel = abf_path.resolve().as_posix()
    digest = hashlib.sha1(rel.encode("utf-8")).hexdigest()[:8]
    return f"{abf_path.stem}__{digest}"


def extract_abf_metadata(abf_path: Path, abf_root: Path, flat_tol: dict | None = None) -> dict:
    """Header, channel and command-waveform evidence for one ABF.

    Returns a flat dict; nested evidence is JSON-encoded so the row stays tabular.
    Read errors are captured in ``read_status``/``read_error`` instead of raising.
    """
    abf_path = Path(abf_path)
    stat = abf_path.stat()
    row: dict = {
        "recording_id": make_recording_id(abf_path, abf_root),
        "abf_filename": abf_path.name,
        "recording_stem": abf_path.stem,
        "original_path": str(abf_path.resolve()),
        "relative_path": abf_path.resolve().relative_to(abf_root.resolve()).as_posix()
        if abf_path.resolve().is_relative_to(abf_root.resolve()) else "",
        "file_size_bytes": stat.st_size,
        "file_modified": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
        "sha256": sha256_file(abf_path),
    }
    try:
        abf = open_abf(abf_path)
    except Exception as exc:  # unreadable file: record and move on
        row.update(read_status="error", read_error=repr(exc))
        return row

    tags_t = [float(t) for t in getattr(abf, "tagTimesSec", [])]
    tags_c = [str(c) for c in getattr(abf, "tagComments", [])]
    dac_summaries = {}
    for d in range(len(abf.dacNames)):
        if d in readable_dacs(abf):
            try:
                dac_summaries[d] = summarise_dac(abf, d, flat_tol).as_dict()
            except Exception as exc:
                dac_summaries[d] = {"dac": d, "error": repr(exc)}
        else:
            dac_summaries[d] = {"dac": d, "error": "not readable via pyABF (index >= ADC count)"}

    row.update({
        "read_status": "ok",
        "read_error": "",
        "abf_id": abf.abfID,
        "abf_version": str(abf.abfVersionString),
        "abf_protocol": abf.protocol,
        "abf_protocol_path": abf.protocolPath,
        "recording_start": abf.abfDateTime.isoformat() if hasattr(abf.abfDateTime, "isoformat") else str(abf.abfDateTime),
        "sample_rate_Hz": float(abf.dataRate),
        "sweep_count": int(abf.sweepCount),
        "sweep_length_s": float(abf.sweepLengthSec),
        "total_duration_s": float(abf.sweepLengthSec * abf.sweepCount),
        "adc_count": int(abf.channelCount),
        "adc_names": " | ".join(abf.adcNames),
        "adc_units": " | ".join(abf.adcUnits),
        "adc_kinds": " | ".join(unit_kind(u) for u in abf.adcUnits),
        "dac_count": len(abf.dacNames),
        "dac_names": " | ".join(abf.dacNames),
        "dac_units": " | ".join(abf.dacUnits),
        "dac_kinds": " | ".join(unit_kind(u) for u in abf.dacUnits),
        "holding_command": json.dumps([float(x) for x in getattr(abf, "holdingCommand", [])]),
        "tag_times_s": json.dumps(tags_t),
        "tag_comments": json.dumps(tags_c),
        "abf_comment": str(getattr(abf, "abfFileComment", "") or ""),
        "dac_summary_json": json.dumps(dac_summaries, default=float),
    })
    row.update(amplifier_header_fields(abf))
    return row


def amplifier_header_fields(abf) -> dict:
    """Amplifier (MultiClamp telegraph) and Clampex output settings stored in the ABF header.

    Lists are JSON-encoded, one value per ADC (``telegraph_*``) or per DAC (``dac_*``):

    * ``telegraph_enabled``: 1 if Clampex received gains/mode from the amplifier.
    * ``telegraph_mode``: amplifier clamp mode (MultiClamp: 0 = V-clamp, 1 = I-clamp, 2 = I=0).
    * ``telegraph_additional_gain``: telegraphed output gain.
    * ``dac_scale_factor``: command sensitivity per DAC, in DAC units per volt
      (e.g. 400 pA/V in current clamp, 20 mV/V in voltage clamp).
    * ``dac_holding_level``: Clampex holding level per DAC, in DAC units.

    Reads pyABF's private header sections; missing fields are left empty, never invented.
    """
    n_adc, n_dac = abf.channelCount, len(abf.dacNames)

    def _get(section, name, n):
        try:
            return [float(v) for v in list(getattr(section, name))[:n]]
        except Exception:
            return []

    adc, dac = getattr(abf, "_adcSection", None), getattr(abf, "_dacSection", None)
    return {
        "telegraph_enabled": json.dumps(_get(adc, "nTelegraphEnable", n_adc)),
        "telegraph_mode": json.dumps(_get(adc, "nTelegraphMode", n_adc)),
        "telegraph_additional_gain": json.dumps(_get(adc, "fTelegraphAdditGain", n_adc)),
        "dac_scale_factor": json.dumps(_get(dac, "fDACScaleFactor", n_dac)),
        "dac_holding_level": json.dumps(_get(dac, "fDACHoldingLevel", n_dac)),
    }


def extract_folder(abf_root: Path, flat_tol: dict | None = None, log=print) -> pd.DataFrame:
    """Extract metadata for every *.abf under ``abf_root`` (recursive, sorted)."""
    abf_root = Path(abf_root)
    files = sorted(abf_root.rglob("*.abf"))
    rows = []
    for i, f in enumerate(files, 1):
        log(f"[{i}/{len(files)}] metadata: {f.name}")
        rows.append(extract_abf_metadata(f, abf_root, flat_tol))
    df = pd.DataFrame(rows)
    if not df.empty:
        dup = df["abf_filename"].duplicated(keep=False)
        df["duplicate_filename"] = dup
    return df


# ===================================================================== matching
# Transparent matching of ABF recordings to recording-book rows.
#
# Rules:
# * Key = filename stem (case-insensitive, '.abf' stripped) vs. 'Recording Name'.
# * A match is accepted only if it is unique on BOTH sides. Duplicate book rows or
#   duplicate ABF filenames are flagged and their book fields are NOT merged:
#   nothing is chosen arbitrarily.
# * Cell Tag comes from the book only. A filename-derived candidate is kept in a
#   separate column and never used silently.


DEFAULT_BOOK_COLUMNS = {"cell_tag": "Cell Tag", "animal": "Animal ID", "slice": "Slice", "date": "Date Recorded",
                        "protocol": "Protocol"}


def book_columns(cfg: dict | None) -> dict:
    """Recording-book column names: ``recording_book.columns`` in the 01 config, over the defaults."""
    cols = ((cfg or {}).get("recording_book") or {}).get("columns") or {}
    return {**DEFAULT_BOOK_COLUMNS, **cols}


def normalise_key(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = Path(str(value).strip()).name
    if text.lower().endswith(".abf"):
        text = text[:-4]
    return text.strip().lower()


def filename_cell_tag_candidate(stem: str) -> str:
    """Candidate Cell Tag from a filename: strip a trailing '_NNNN' acquisition index."""
    return re.sub(r"_\d{4}$", "", stem)


def _prefixed(df: pd.DataFrame, prefix: str, keep: list[str]) -> pd.DataFrame:
    return df.rename(columns={c: f"{prefix}{c}" for c in df.columns if c not in keep})


def match_recordings(recordings: pd.DataFrame, book: dict[str, pd.DataFrame],
                     key_column: str = "Recording Name",
                     columns: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach book information to recordings. ``columns``: book column names (see
    ``book_columns``; defaults Cell Tag, Animal ID, Slice).

    Returns (recordings_with_book, unmatched_book_rows).
    Adds columns: match_status, match_note, cell_tag, cell_tag_source,
    cell_tag_candidate_from_filename.
    """
    rec = recordings.copy()
    rec["_key"] = rec["recording_stem"].map(normalise_key)
    abf_key_counts = rec["_key"].value_counts()

    log = book.get("recording_log")
    if log is None or key_column not in getattr(log, "columns", []):
        rec["match_status"] = "no_recording_log"
        rec["match_note"] = f"Recording Log or '{key_column}' column missing"
        log = pd.DataFrame(columns=[key_column])
    log = log.copy()
    log["_key"] = log[key_column].map(normalise_key)
    log = log[log["_key"] != ""]
    book_key_counts = log["_key"].value_counts()

    statuses, notes = [], []
    for key in rec["_key"]:
        n_book = int(book_key_counts.get(key, 0))
        n_abf = int(abf_key_counts.get(key, 0))
        if n_book == 0:
            statuses.append("unmatched_abf"); notes.append("no Recording Log row with this name")
        elif n_book > 1:
            statuses.append("duplicate_book_entry"); notes.append(f"{n_book} Recording Log rows share this name; not merged")
        elif n_abf > 1:
            statuses.append("duplicate_filename"); notes.append(f"{n_abf} ABFs share this filename; not merged")
        else:
            statuses.append("matched"); notes.append("")
    if "match_status" not in rec.columns:
        rec["match_status"], rec["match_note"] = statuses, notes

    unique_log = log[log["_key"].map(book_key_counts) == 1]
    mergeable = rec["match_status"] == "matched"
    merged = rec[mergeable].merge(unique_log, on="_key", how="left", validate="one_to_one")
    rec = pd.concat([merged, rec[~mergeable]], ignore_index=True)

    # Animals (one row per animal) and Slices (one row per animal + slice).
    col = {**DEFAULT_BOOK_COLUMNS, **(columns or {})}
    an, sl = col["animal"], col["slice"]
    animals = book.get("animals")
    if animals is not None and an in animals.columns and an in rec.columns:
        a = animals.copy()
        a[an] = a[an].astype("string").str.strip()
        dup_animals = a[an].duplicated(keep=False)
        a = _prefixed(a[~dup_animals], "Animals.", [an])
        rec[an] = rec[an].astype("string").str.strip()
        rec = rec.merge(a, on=an, how="left", validate="many_to_one")
    slices = book.get("slices")
    if slices is not None and {an, sl} <= set(slices.columns) and sl in rec.columns:
        s = slices.copy()
        for k in (an, sl):
            s[k] = s[k].astype("string").str.strip()
            rec[k] = rec[k].astype("string").str.strip()
        dup = s.duplicated([an, sl], keep=False)
        s = _prefixed(s[~dup], "Slices.", [an, sl])
        rec = rec.merge(s, on=[an, sl], how="left", validate="many_to_one")

    # Cell Tag: book value only.
    ct = col["cell_tag"]
    book_tag = rec[ct] if ct in rec.columns else pd.Series(pd.NA, index=rec.index)
    book_tag = book_tag.astype("string").str.strip().replace("", pd.NA)
    rec["cell_tag"] = book_tag
    rec["cell_tag_source"] = book_tag.notna().map({True: "recording_book", False: "missing"})
    rec["cell_tag_candidate_from_filename"] = rec["recording_stem"].map(filename_cell_tag_candidate)

    matched_keys = set(rec.loc[rec["match_status"] == "matched", "_key"])
    unmatched_book = log[~log["_key"].isin(matched_keys)].drop(columns="_key")
    rec = rec.drop(columns="_key").sort_values("recording_id").reset_index(drop=True)
    return rec, unmatched_book.reset_index(drop=True)


# ===================================================================== master_tables
# Recording-level and cell-level master tables.
#
# recording_master: one row per ABF (no rows dropped, no averaging).
# cell_master: one row per Cell Tag, linking all recordings by recording_id.
# Context fields that should be constant within a cell (animal, slice, sex, ...) are
# joined; if they disagree, all values are kept and the conflict is flagged.


CELL_CONTEXT_COLUMNS = [
    "Animal ID", "Slice", "Date Recorded", "Sex", "Age", "Strain",
    "Animals.Strain / Line", "Animals.Genotype", "Animals.Date Sliced", "Animals.Age at Slicing",
]


def _unique_text(values: pd.Series) -> list[str]:
    out = []
    for v in values.dropna():
        s = str(v).strip()
        if s and s not in out:
            out.append(s)
    return out


def build_cell_master(recording_master: pd.DataFrame) -> pd.DataFrame:
    df = recording_master[recording_master["cell_tag"].notna()]
    rows = []
    for tag, g in df.groupby("cell_tag", sort=True):
        g = g.sort_values("recording_stem")
        row = {"cell_tag": tag, "n_recordings": len(g)}
        conflicts = []
        for col in CELL_CONTEXT_COLUMNS:
            if col in g.columns:
                vals = _unique_text(g[col])
                row[col] = " | ".join(vals)
                if len(vals) > 1:
                    conflicts.append(col)
        row["recording_ids"] = " | ".join(g["recording_id"])
        row["abf_filenames"] = " | ".join(g["abf_filename"])
        if "abf_family" in g.columns:
            row["abf_families"] = " | ".join(f"{s}:{f}" for s, f in zip(g["recording_stem"], g["abf_family"]))
            for fam, n in g["abf_family"].value_counts().items():
                row[f"n_{fam}"] = int(n)
        if "classification_status" in g.columns:
            row["n_needing_review"] = int(g["classification_status"].isin(["ambiguous", "mismatch", "unsupported"]).sum())
        row["metadata_conflicts"] = " | ".join(conflicts)
        rows.append(row)
    return pd.DataFrame(rows)
