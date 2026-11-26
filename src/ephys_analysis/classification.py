"""Protocol classification from ABF evidence, and verified copy-sorting by protocol."""

from __future__ import annotations

from pathlib import Path
import json
import re
import shutil

import pandas as pd

from .metadata import sha256_file


# ===================================================================== protocol_detection
# Rule-based protocol classification from ABF acquisition evidence.
#
# The family is derived from the ABF first (clamp mode, command waveform, sweeps,
# duration). The recording-book label is compared afterwards and only sets the
# status (confirmed / probable / mismatch). Filenames are never used.


def label_family(label, families: dict) -> list[str]:
    """Families whose label_patterns match ``label`` (case-insensitive)."""
    if label is None or (isinstance(label, float) and pd.isna(label)) or not str(label).strip():
        return []
    text = str(label).strip()
    hits = []
    for fam, spec in families.items():
        if any(re.search(p, text, flags=re.IGNORECASE) for p in spec.get("label_patterns", [])):
            hits.append(fam)
    return hits


def _families_for_dac(dac: dict, mode: str, row: dict, rules: dict) -> list[tuple[str, str]]:
    """Families supported by one DAC's command; returns (family, evidence) pairs."""
    out = []
    dur = float(row.get("total_duration_s") or 0)
    sweep_len = float(row.get("sweep_length_s") or 0)
    n_sweeps = int(row.get("sweep_count") or 0)
    name = f"DAC{dac['dac']} '{dac.get('name', '')}'"
    if mode == "voltage_clamp":
        r = rules["membrane_test"]
        amps = [abs(a) for a in dac.get("step_amplitudes", [])]
        pulses = dac.get("pulse_durations_s", [])
        if (not dac.get("is_flat") and not dac.get("has_ramp") and pulses and amps
                and max(pulses) <= r["max_pulse_duration_s"]
                and max(amps) <= r["max_abs_pulse_amplitude_mV"]
                and sweep_len <= r["max_sweep_length_s"]):
            out.append(("membrane_test", f"{name}: voltage pulse {max(amps):g} mV, "
                                         f"{max(pulses) * 1000:.0f} ms, sweep {sweep_len:g} s"))
        if dac.get("is_flat") and dur >= rules["epsc"]["min_duration_s"]:
            out.append(("epsc", f"{name}: flat voltage command, {dur:.0f} s"))
    elif mode == "current_clamp":
        if dac.get("has_ramp"):
            out.append(("ramp", f"{name}: ramp command (epochs {dac.get('epoch_types')})"))
        elif not dac.get("is_flat"):
            r = rules["current_steps"]
            if n_sweeps >= r["min_sweeps"] and (dac.get("amplitude_varies_across_sweeps") or not r["require_varying_amplitude"]):
                amps = dac.get("step_amplitudes", [])
                out.append(("current_steps", f"{name}: {n_sweeps} sweeps, steps {min(amps):g} to {max(amps):g} pA"))
        else:
            if dur >= rules["modulation_timecourse"]["min_duration_s"]:
                out.append(("modulation_timecourse", f"{name}: flat current command, {dur:.0f} s"))
            elif dur >= rules["spontaneous"]["min_duration_s"]:
                out.append(("spontaneous", f"{name}: flat current command, {dur:.0f} s"))
    return out


def classify_row(row: dict, protocols_cfg: dict, label=None) -> dict:
    """Classify one metadata row. Returns the classification fields."""
    cls = protocols_cfg["classification"]
    rules = cls["rules"]
    families = protocols_cfg["families"]
    result = {"clamp_mode": "", "abf_family": "", "label_family": "", "classification_status": "",
              "command_dac": "", "evidence": "", "review_reason": ""}
    label_hits = label_family(label, families)
    result["label_family"] = " | ".join(label_hits)

    if row.get("read_status") != "ok":
        result.update(classification_status="unsupported", review_reason=f"ABF unreadable: {row.get('read_error')}")
        return result

    adc_kinds = str(row.get("adc_kinds", "")).split(" | ")
    p = cls.get("primary_adc_index", 0)
    primary = adc_kinds[p] if p < len(adc_kinds) else "unknown"
    mode = {"current": "voltage_clamp", "voltage": "current_clamp"}.get(primary, "")
    result["clamp_mode"] = mode or "unknown"
    if not mode:
        result.update(classification_status="unsupported",
                      review_reason=f"primary ADC {p} has unknown units; clamp mode cannot be inferred")
        return result

    dacs = json.loads(row.get("dac_summary_json") or "{}")
    want = "voltage" if mode == "voltage_clamp" else "current"
    candidates = [d for d in dacs.values() if "error" not in d and d.get("kind") == want]
    found: list[tuple[str, str, int]] = []
    for d in candidates:
        for fam, ev in _families_for_dac(d, mode, row, rules):
            found.append((fam, ev, d["dac"]))
    # EPSC requires ALL voltage DACs to be flat (no test pulse on any of them).
    if any(f == "epsc" for f, _, _ in found) and not all(d.get("is_flat") for d in candidates):
        found = [x for x in found if x[0] != "epsc"]

    evidence = [f"primary ADC{p} records {primary} -> {mode}"] + [ev for _, ev, _ in found]
    tags = json.loads(row.get("tag_comments") or "[]")
    kw = rules["modulation_timecourse"].get("tag_keywords", [])
    drug_tags = [t for t in tags if any(k in t.lower() for k in kw)]
    if drug_tags:
        evidence.append(f"drug-related tags: {drug_tags}")
    result["evidence"] = "; ".join(evidence)

    fams = sorted({f for f, _, _ in found})
    if len(fams) != 1:
        reason = "no classification rule matched" if not fams else f"several families matched: {fams}"
        result.update(classification_status="ambiguous", review_reason=reason)
        return result

    fam = fams[0]
    result["abf_family"] = fam
    result["command_dac"] = ",".join(str(d) for f, _, d in found if f == fam)
    if not label_hits:
        result.update(classification_status="probable", review_reason="no recognised recording-book label")
    elif fam in label_hits:
        result["classification_status"] = "confirmed"
        if fam == "spontaneous" and drug_tags:
            result["review_reason"] = "drug-related tag in a spontaneous recording; check context"
    else:
        result.update(classification_status="mismatch",
                      review_reason=f"ABF indicates {fam}; label '{label}' indicates {label_hits}")
    return result


def classify_table(recordings: pd.DataFrame, protocols_cfg: dict, label_column: str = "Protocol") -> pd.DataFrame:
    """Classify every recording. Label = book Protocol if present, else ABF protocol name."""
    out = []
    for _, r in recordings.iterrows():
        row = r.to_dict()
        book_label = row.get(label_column)
        if book_label is None or (isinstance(book_label, float) and pd.isna(book_label)):
            label, source = row.get("abf_protocol"), "abf_protocol_name"
        else:
            label, source = book_label, "recording_book"
        res = classify_row(row, protocols_cfg, label)
        res.update(recording_id=row["recording_id"], abf_filename=row["abf_filename"],
                   original_path=row["original_path"], label=label, label_source=source,
                   abf_protocol_name=row.get("abf_protocol"))
        out.append(res)
    cols = ["recording_id", "abf_filename", "original_path", "label", "label_source", "abf_protocol_name",
            "clamp_mode", "abf_family", "label_family", "classification_status", "command_dac",
            "evidence", "review_reason"]
    return pd.DataFrame(out)[cols]


# ===================================================================== sorting
# Copy recordings into protocol folders. Originals are never moved or modified.
#
# * Only ``confirmed`` and ``probable`` recordings go to their family folder; all
#   others (ambiguous, mismatch, unsupported) go to the review folder.
# * Copies keep the original filename inside a sub-folder named by the original's
#   parent folder, so identical filenames from different folders cannot collide.
# * An existing destination is never overwritten unless ``overwrite=True``. An
#   existing identical copy (same SHA-256) is reported as 'already_present'.
# * Every copy is verified by SHA-256 against the original.


def destination_for(row: dict, protocols_cfg: dict, sorted_root: Path) -> tuple[Path, str]:
    status = row["classification_status"]
    fam = row.get("abf_family") or ""
    if status in ("confirmed", "probable") and fam in protocols_cfg["families"]:
        folder = protocols_cfg["families"][fam]["sort_folder"]
    else:
        folder = protocols_cfg["review_folder"]
    original = Path(row["original_path"])
    return sorted_root / folder / original.parent.name / original.name, folder


def copy_sorted(classification: pd.DataFrame, recordings: pd.DataFrame, protocols_cfg: dict,
                sorted_root: Path, overwrite: bool = False, log=print) -> pd.DataFrame:
    """Copy each recording to its protocol folder and return the copy manifest."""
    sha_by_id = dict(zip(recordings["recording_id"], recordings["sha256"]))
    rows = []
    for _, r in classification.iterrows():
        row = r.to_dict()
        src = Path(row["original_path"])
        dst, folder = destination_for(row, protocols_cfg, sorted_root)
        entry = {"recording_id": row["recording_id"], "abf_filename": row["abf_filename"],
                 "original_path": str(src), "label": row["label"], "abf_family": row["abf_family"],
                 "classification_status": row["classification_status"], "destination_folder": folder,
                 "destination_path": str(dst), "copy_status": "", "verified_sha256": False,
                 "review_reason": row["review_reason"]}
        if src.resolve() == dst.resolve():
            entry["copy_status"] = "refused_same_path"
            rows.append(entry)
            continue
        original_sha = sha_by_id.get(row["recording_id"]) or sha256_file(src)
        if dst.exists():
            if sha256_file(dst) == original_sha:
                entry.update(copy_status="already_present", verified_sha256=True)
                rows.append(entry)
                continue
            if not overwrite:
                entry["copy_status"] = "skipped_exists_different_content"
                rows.append(entry)
                continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        ok = sha256_file(dst) == original_sha
        entry.update(copy_status="copied" if ok else "copy_verification_failed", verified_sha256=ok)
        log(f"copy {src.name} -> {folder} [{entry['copy_status']}]")
        rows.append(entry)
    return pd.DataFrame(rows)
