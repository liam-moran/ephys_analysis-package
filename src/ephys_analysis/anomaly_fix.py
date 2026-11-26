"""02_ephys_anomaly-fix steps: auto-resolve -> decision sheet -> apply -> organise -> report.

Inputs are the outputs of 01_ephys_meta-org (recordings.csv, flags.csv).

* Known acquisition problems are resolved automatically and logged:
  ``duplicated_voltage_no_current`` and ``possible_gain_error``.
* Every other review / needs_fix flag needs a decision in ``decisions.xlsx``.
* A recording is **ready** when all its flags are resolved, **excluded** when any
  decision is ``exclude``, otherwise **pending**. Only ready recordings are copied
  into ``output/protocol_sorted/``. Originals are never changed; corrections are
  recorded in ``analysis_ready.csv`` for the protocol analyses to apply.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .classification import copy_sorted
from .utils import clear_output, load_config, repo_path

PROJECT = "02_ephys_anomaly-fix"
ACTIONABLE = ("review", "needs_fix")
SHEET_COLUMNS = ["recording_id", "abf_filename", "cell_tag", "protocol", "flag", "severity", "detail",
                 "suggested_action", "suggested_value", "allowed_actions",
                 "decision", "value", "note", "decided_by", "decided_on"]
VALUE_REQUIRED = {"relabel", "bias_current", "rescale", "notch"}
NOISE_DEFAULTS = {"mains_Hz": 50.0, "mains_harmonics": 5, "mains_window_s": 1.0, "notch_Q": 30.0}


# --------------------------------------------------------------------------- setup / load
def setup(repo_root: Path | str, fresh: bool = False) -> tuple[dict, dict, Path]:
    """Load this project's config and the 01 config. ``fresh`` clears ``output/``
    (never the decision sheet, which lives outside ``output/``)."""
    cfg = load_config(Path(repo_root) / PROJECT)
    meta_cfg = load_config(Path(repo_root) / cfg["inputs"]["meta_org_project"])
    out = cfg["_project_dir"] / "output"
    if fresh:
        clear_output(out)
    out.mkdir(parents=True, exist_ok=True)
    return cfg, meta_cfg, out


def load_inputs(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read recordings.csv and flags.csv from 01."""
    rec_path, flag_path = repo_path(cfg, cfg["inputs"]["recordings"]), repo_path(cfg, cfg["inputs"]["flags"])
    for p in (rec_path, flag_path):
        if not p.exists():
            raise FileNotFoundError(f"{p} not found: run 01_ephys_meta-org first.")
    recordings = pd.read_csv(rec_path, dtype={"cell_tag": "string", "command_dac": "string"})
    flags = pd.read_csv(flag_path, dtype={"recording_id": "string"})
    flags["recording_id"] = flags["recording_id"].fillna("")
    flags["abf_filename"] = flags["abf_filename"].fillna("")
    flags["detail"] = flags["detail"].fillna("")
    return recordings, flags


# --------------------------------------------------------------------------- step 1: auto-resolve
def choose_gain(median_mV: float, rules: dict) -> tuple[float | None, str]:
    """Standard gain that brings ``median_mV`` closest to a typical RMP.

    Returns (factor, explanation); factor is None when no gain is clearly best.
    """
    lo, hi = rules["plausible_rmp_mV"]
    typical = rules["typical_rmp_mV"]
    if lo <= median_mV <= hi:
        return None, f"median {median_mV:.0f} mV is already plausible; no rescaling"
    cands = sorted((abs(median_mV / g - typical), g) for g in rules["standard_gains"] if lo <= median_mV / g <= hi)
    if not cands:
        return None, f"no standard gain brings median {median_mV:.0f} mV into {lo} to {hi} mV"
    if len(cands) > 1 and cands[1][0] - cands[0][0] < rules["min_margin_mV"]:
        return None, (f"gains {cands[0][1]:g} and {cands[1][1]:g} are both plausible "
                      f"(corrected medians {median_mV / cands[0][1]:.0f} and {median_mV / cands[1][1]:.0f} mV)")
    g = cands[0][1]
    return g, f"median {median_mV:.0f} mV / {g:g} = {median_mV / g:.1f} mV"


def auto_resolve(recordings: pd.DataFrame, flags: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Resolve duplicated-voltage and gain flags where the rules allow it.

    Returns one row per resolved flag: recording_id, flag, action, value, how.
    """
    rules = cfg["auto_resolve"]
    fam = dict(zip(recordings["recording_id"], recordings["abf_family"].fillna("")))
    rec_flags = flags.groupby("recording_id")["flag"].apply(set).to_dict()
    rows = []
    for _, f in flags[flags["recording_id"] != ""].iterrows():
        rid, family = f["recording_id"], fam.get(f["recording_id"], "")
        if f["flag"] == "duplicated_voltage_no_current":
            d = rules["duplicated_voltage_no_current"]
            others = rec_flags.get(rid, set())
            if family in d["voltage_only_families"]:
                rows.append((rid, f["flag"], "voltage_only", np.nan,
                             f"{family}: 0 pA protocol, no current needed; voltage = primary ADC"))
            elif family in d["command_as_current_families"] and not others & {"unexpected_command_sensitivity", "telegraph_off"}:
                rows.append((rid, f["flag"], "command_as_current", np.nan,
                             f"{family}: current = command waveform (telegraphed sensitivity as expected); "
                             f"command, not measured"))
        elif f["flag"] == "possible_gain_error" and pd.notna(f.get("value")):
            g, how = choose_gain(float(f["value"]), rules["possible_gain_error"])
            if g is not None:
                rows.append((rid, f["flag"], "rescale", g, how))
    return pd.DataFrame(rows, columns=["recording_id", "flag", "action", "value", "how"])


# --------------------------------------------------------------------------- step 2: decision sheet
def pending_flags(flags: pd.DataFrame, auto: pd.DataFrame) -> pd.DataFrame:
    """Recording-level review/needs_fix flags that were not auto-resolved."""
    keys = set(zip(auto["recording_id"], auto["flag"]))
    f = flags[(flags["recording_id"] != "") & flags["severity"].isin(ACTIONABLE)]
    return f[[(r, fl) not in keys for r, fl in zip(f["recording_id"], f["flag"])]]


def read_decision_sheet(path: Path) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame(columns=SHEET_COLUMNS)
    sheet = pd.read_excel(path, dtype=str).fillna("")
    for c in SHEET_COLUMNS:
        if c not in sheet.columns:
            sheet[c] = ""
    return sheet[SHEET_COLUMNS]


def update_decision_sheet(pending: pd.DataFrame, recordings: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, int]:
    """Append a row for every pending flag that is not yet on the sheet. Rows already on the
    sheet are never changed. Writes the sheet (Excel, with dropdowns) and returns (sheet, n_added)."""
    path = repo_path(cfg, cfg["decision_sheet"])
    sheet = read_decision_sheet(path)
    have = set(zip(sheet["recording_id"], sheet["flag"]))
    info = recordings.set_index("recording_id")
    new = []
    for _, f in pending.iterrows():
        if (f["recording_id"], f["flag"]) in have:
            continue
        a = cfg["actions"].get(f["flag"], {"suggest": "accept", "allowed": ["accept", "exclude"]})
        r = info.loc[f["recording_id"]] if f["recording_id"] in info.index else {}
        suggested_value = ""
        if a["suggest"] == "relabel":
            suggested_value = str(r.get("abf_family", "") or "")
        elif a["suggest"] in ("bias_current", "rescale") and pd.notna(f.get("value")):
            suggested_value = f"{float(f['value']):g}"
        new.append({"recording_id": f["recording_id"], "abf_filename": f["abf_filename"],
                    "cell_tag": str(r.get("cell_tag", "") or ""), "protocol": str(r.get("abf_family", "") or ""),
                    "flag": f["flag"], "severity": f["severity"], "detail": f["detail"],
                    "suggested_action": a["suggest"], "suggested_value": suggested_value,
                    "allowed_actions": ", ".join(a["allowed"]),
                    "decision": "", "value": "", "note": "", "decided_by": "", "decided_on": ""})
    if new:
        sheet = pd.concat([sheet, pd.DataFrame(new, columns=SHEET_COLUMNS)], ignore_index=True)
    sheet = stamp_decisions(sheet, cfg.get("decided_by", ""))
    write_decision_sheet(sheet, path, cfg)
    return sheet, len(new)


def stamp_decisions(sheet: pd.DataFrame, initials: str, today: str | None = None) -> pd.DataFrame:
    """Fill in ``decided_by`` (``initials``) and ``decided_on`` (today, YYYY-MM-DD) for rows that
    have a decision but leave those cells empty. Cells already filled in are never changed.
    The date is that of the first run that sees the decision."""
    sheet = sheet.copy()
    today = today or date.today().isoformat()
    has = sheet["decision"].astype(str).str.strip() != ""
    for col, val in (("decided_by", str(initials or "").strip()), ("decided_on", today)):
        empty = sheet[col].astype(str).str.strip() == ""
        if val:
            sheet.loc[has & empty, col] = val
    return sheet


def write_decision_sheet(sheet: pd.DataFrame, path: Path, cfg: dict) -> None:
    """Write the sheet as Excel with a dropdown in the 'decision' column."""
    from openpyxl.utils import get_column_letter
    from openpyxl.worksheet.datavalidation import DataValidation

    path = Path(path)
    try:
        with pd.ExcelWriter(path, engine="openpyxl") as xw:
            sheet.to_excel(xw, sheet_name="decisions", index=False)
            ws = xw.sheets["decisions"]
            actions = sorted({a for spec in cfg["actions"].values() for a in spec["allowed"]} | {"accept", "exclude"})
            col = get_column_letter(SHEET_COLUMNS.index("decision") + 1)
            dv = DataValidation(type="list", formula1='"' + ",".join(actions) + '"', allow_blank=True)
            ws.add_data_validation(dv)
            dv.add(f"{col}2:{col}{max(len(sheet) + 200, 201)}")
            widths = {"detail": 60, "abf_filename": 26, "recording_id": 32, "note": 30, "allowed_actions": 30}
            for i, c in enumerate(SHEET_COLUMNS, 1):
                ws.column_dimensions[get_column_letter(i)].width = widths.get(c, 16)
            ws.freeze_panes = "A2"
    except PermissionError as exc:
        raise PermissionError(f"Close {path} in Excel and run again.") from exc


# --------------------------------------------------------------------------- step 3: apply decisions
def validate_decisions(sheet: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Check filled-in decisions. Returns the sheet with 'decision_status':
    '' (not decided), 'ok', or 'invalid: <reason>'."""
    sheet = sheet.copy()
    status = []
    for _, r in sheet.iterrows():
        d, v = str(r["decision"]).strip(), str(r["value"]).strip()
        allowed = [a.strip() for a in str(r["allowed_actions"]).split(",") if a.strip()]
        if not d:
            status.append("")
        elif allowed and d not in allowed:
            status.append(f"invalid: '{d}' not allowed for {r['flag']} (allowed: {', '.join(allowed)})")
        elif d in VALUE_REQUIRED and not v:
            status.append(f"invalid: '{d}' needs a value")
        elif d == "relabel" and v not in cfg["families"]:
            status.append(f"invalid: unknown family '{v}' (use one of {', '.join(cfg['families'])})")
        elif d == "notch" and not all(_is_number(x, positive=True) for x in v.replace("|", ";").split(";") if x.strip()):
            status.append(f"invalid: notch needs frequencies in Hz, e.g. '1.3' or '1.3; 2.6' (got '{v}')")
        elif d == "remove_mains" and v and not (_is_number(v, positive=True) and float(v.replace(",", ".")).is_integer()):
            status.append(f"invalid: remove_mains value = number of harmonics (whole number) or empty (got '{v}')")
        elif d in ("bias_current", "rescale") and not _is_number(v, positive=(d == "rescale")):
            status.append(f"invalid: '{v}' is not a {'positive ' if d == 'rescale' else ''}number")
        elif d == "exclude" and not str(r["note"]).strip():
            status.append("invalid: 'exclude' needs a note (reason)")
        else:
            status.append("ok")
    sheet["decision_status"] = status
    return sheet


def _is_number(v: str, positive: bool = False) -> bool:
    try:
        x = float(v.replace(",", "."))
    except ValueError:
        return False
    return x > 0 if positive else True


def resolve(recordings: pd.DataFrame, flags: pd.DataFrame, auto: pd.DataFrame,
            sheet: pd.DataFrame, noise_cfg: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Combine auto-resolutions and decisions.

    Returns (status table with one row per recording, issue table with one row per flag
    and its resolution: auto_resolved / decided / pending / invalid / info / book).
    """
    auto_map = {(r.recording_id, r.flag): r for r in auto.itertuples()}
    dec_map = {(r["recording_id"], r["flag"]): r for _, r in sheet.iterrows()}
    issues = []
    for _, f in flags.iterrows():
        key = (f["recording_id"], f["flag"])
        row = f.to_dict()
        if f["recording_id"] == "":
            row.update(resolution="book" if f["severity"] in ACTIONABLE else "info", action="", action_value="",
                       how="fix in the recording book, then re-run 01" if f["severity"] in ACTIONABLE else "")
        elif f["severity"] not in ACTIONABLE:
            row.update(resolution="info", action="", action_value="", how="")
        elif key in auto_map:
            a = auto_map[key]
            row.update(resolution="auto_resolved", action=a.action,
                       action_value="" if pd.isna(a.value) else f"{a.value:g}", how=a.how)
        elif key in dec_map and dec_map[key]["decision_status"] == "ok":
            d = dec_map[key]
            who = " ".join(str(x) for x in (d["decided_by"], d["decided_on"]) if str(x).strip())
            how = " ".join(x for x in (str(d["note"]).strip(), f"({who})" if who else "") if x)
            row.update(resolution="decided", action=d["decision"], action_value=d["value"], how=how)
        elif key in dec_map and str(dec_map[key]["decision_status"]).startswith("invalid"):
            row.update(resolution="invalid", action=dec_map[key]["decision"], action_value=dec_map[key]["value"],
                       how=dec_map[key]["decision_status"])
        else:
            row.update(resolution="pending", action="", action_value="", how="decision needed in decisions.xlsx")
        issues.append(row)
    issues = pd.DataFrame(issues)

    status_rows = []
    for _, r in recordings.iterrows():
        mine = issues[issues["recording_id"] == r["recording_id"]]
        acted = mine[mine["resolution"].isin(["auto_resolved", "decided"])]
        actions = dict(zip(acted["action"], acted["action_value"]))
        if "exclude" in actions:
            state = "excluded"
        elif mine["resolution"].isin(["pending", "invalid"]).any():
            state = "pending"
        else:
            state = "ready"
        status_rows.append({"recording_id": r["recording_id"], "state": state,
                            **corrections_for(r, actions, noise_cfg),
                            "open_issues": "; ".join(mine.loc[mine["resolution"].isin(["pending", "invalid"]), "flag"])})
    return pd.DataFrame(status_rows), issues


def corrections_for(rec: pd.Series, actions: dict, noise_cfg: dict | None = None) -> dict:
    """What a protocol analysis must apply to this recording (originals stay unchanged).
    Noise corrections (decisions ``remove_mains`` [harmonics] and ``notch`` <Hz; Hz>) become
    ``mains_Hz`` / ``mains_harmonics`` / ``mains_window_s`` and ``notch_Hz`` / ``notch_Q``
    (defaults: ``noise_corrections`` in config), applied in memory by noise.correct (04-07)."""
    kinds = str(rec.get("adc_kinds", "")).split(" | ")
    abf_family = rec.get("abf_family")
    family = actions.get("relabel") or (abf_family if isinstance(abf_family, str) else "")
    v_adc = next((i for i, k in enumerate(kinds) if k == "voltage"), None)
    i_adc = next((i for i, k in enumerate(kinds) if k == "current"), None)
    dacs = [d for d in str(rec.get("command_dac") or "").split(",") if d.strip().isdigit()]
    if "command_as_current" in actions:
        current = f"command_DAC{dacs[0]}" if dacs else "command"
    elif "voltage_only" in actions:
        current = "none"
    else:
        current = f"measured_ADC{i_adc}" if i_adc is not None else "none"
    scale = 1.0 / float(str(actions["rescale"]).replace(",", ".")) if "rescale" in actions else 1.0
    bias = float(str(actions["bias_current"]).replace(",", ".")) if "bias_current" in actions else np.nan
    nz = {**NOISE_DEFAULTS, **(noise_cfg or {})}
    mains = {"mains_Hz": np.nan, "mains_harmonics": np.nan, "mains_window_s": np.nan}
    if "remove_mains" in actions:
        v = str(actions["remove_mains"]).strip().replace(",", ".")
        mains = {"mains_Hz": float(nz["mains_Hz"]), "mains_harmonics": int(float(v)) if v else int(nz["mains_harmonics"]),
                 "mains_window_s": float(nz["mains_window_s"])}
    notch = {"notch_Hz": "", "notch_Q": np.nan}
    if "notch" in actions:
        notch = {"notch_Hz": "; ".join(x.strip().replace(",", ".") for x in str(actions["notch"]).replace("|", ";").split(";")
                                       if x.strip()), "notch_Q": float(nz["notch_Q"])}
    text = [f"{k}{' ' + str(v) if str(v) else ''}" for k, v in actions.items()]
    return {"family": family, "voltage_source": f"ADC{v_adc}" if v_adc is not None else "none",
            "current_source": current, "voltage_scale": scale, "bias_current_pA": bias, **mains, **notch,
            "corrections": "; ".join(text)}


# --------------------------------------------------------------------------- step 4: organise
def organise(recordings: pd.DataFrame, status: pd.DataFrame, meta_cfg: dict, out: Path, log=print) -> pd.DataFrame:
    """Copy READY recordings into ``output/protocol_sorted/<family folder>/`` (SHA-256 verified)
    and remove derived copies of recordings that are no longer ready. Returns the copy manifest."""
    sorted_root = out / "protocol_sorted"
    ready = status[status["state"] == "ready"].merge(
        recordings[["recording_id", "abf_filename", "original_path", "label", "sha256"]], on="recording_id")
    cls = pd.DataFrame({"recording_id": ready["recording_id"], "abf_filename": ready["abf_filename"],
                        "original_path": ready["original_path"], "label": ready["label"],
                        "abf_family": ready["family"], "classification_status": "confirmed", "review_reason": ""})
    manifest = copy_sorted(cls, ready, meta_cfg, sorted_root, overwrite=False, log=lambda *_: None)
    expected = {Path(p).resolve() for p in manifest.get("destination_path", [])}
    for f in sorted_root.rglob("*.abf") if sorted_root.exists() else []:
        if f.resolve() not in expected:
            f.unlink()
            log(f"removed stale copy (recording no longer ready or changed protocol): {f.relative_to(out)}")
    for d in sorted(sorted_root.rglob("*"), reverse=True) if sorted_root.exists() else []:
        if d.is_dir() and not any(d.iterdir()):
            d.rmdir()
    return manifest


# --------------------------------------------------------------------------- step 5: report
def write_outputs(recordings: pd.DataFrame, status: pd.DataFrame, issues: pd.DataFrame,
                  manifest: pd.DataFrame, out: Path, run_id: str) -> list[Path]:
    """Write analysis_ready.csv, issues_report.csv and issues_report.md."""
    from . import __version__

    ready = status[status["state"] == "ready"].merge(
        recordings[["recording_id", "cell_tag", "abf_filename", "clamp_mode", "original_path", "sha256"]],
        on="recording_id")
    if not manifest.empty:
        ready = ready.merge(manifest[["recording_id", "destination_path"]].rename(
            columns={"destination_path": "sorted_path"}), on="recording_id", how="left")
    cols = ["recording_id", "cell_tag", "abf_filename", "family", "clamp_mode", "voltage_source", "current_source",
            "voltage_scale", "bias_current_pA", "mains_Hz", "mains_harmonics", "mains_window_s", "notch_Hz", "notch_Q",
            "corrections", "sorted_path", "original_path", "sha256"]
    ready = ready[[c for c in cols if c in ready.columns]].assign(analysis_run_id=run_id, package_version=__version__)
    paths = [out / "analysis_ready.csv", out / "issues_report.csv", out / "issues_report.md"]
    ready.to_csv(paths[0], index=False)
    issues.assign(analysis_run_id=run_id).to_csv(paths[1], index=False)
    paths[2].write_text(report_markdown(recordings, status, issues, run_id), encoding="utf-8")
    return paths


def report_markdown(recordings: pd.DataFrame, status: pd.DataFrame, issues: pd.DataFrame, run_id: str) -> str:
    st = status.merge(recordings[["recording_id", "abf_filename", "cell_tag"]], on="recording_id")
    L = [f"# Issues report ({date.today().isoformat()}, run {run_id})", "",
         f"- Recordings: {len(st)} | ready: {(st['state'] == 'ready').sum()} | "
         f"pending: {(st['state'] == 'pending').sum()} | excluded: {(st['state'] == 'excluded').sum()}", ""]
    ready_fam = st[st["state"] == "ready"]["family"].value_counts()
    if len(ready_fam):
        L += ["Ready per protocol: " + ", ".join(f"{k} {v}" for k, v in ready_fam.items()), ""]
    L += ["## Issues by type", "", "| flag | severity | auto_resolved | decided | pending | invalid | info/book |",
          "|---|---|---|---|---|---|---|"]
    for (flag, sev), g in issues.groupby(["flag", "severity"]):
        c = g["resolution"].value_counts()
        L.append(f"| {flag} | {sev} | {c.get('auto_resolved', 0)} | {c.get('decided', 0)} | {c.get('pending', 0)} | "
                 f"{c.get('invalid', 0)} | {c.get('info', 0) + c.get('book', 0)} |")
    for title, res in (("Automatically resolved", "auto_resolved"), ("Decided", "decided"),
                       ("Pending: fill in decisions.xlsx", "pending"), ("Invalid decisions", "invalid"),
                       ("Recording-book issues: fix in the book, then re-run 01", "book"),
                       ("Notes (no action needed)", "info")):
        g = issues[issues["resolution"] == res]
        if g.empty:
            continue
        L += ["", f"## {title} ({len(g)})", "", "| file | flag | action | how / detail |", "|---|---|---|---|"]
        for _, r in g.iterrows():
            act = f"{r['action']} {r['action_value']}".strip()
            txt = r["how"] if res in ("auto_resolved", "decided", "invalid") else r["detail"]
            L.append(f"| {r['abf_filename'] or '-'} | {r['flag']} | {act or '-'} | {str(txt).replace('|', '/')} |")
    excl = st[st["state"] == "excluded"]
    if len(excl):
        L += ["", f"## Excluded ({len(excl)})", ""] + [f"- {r['abf_filename']}: {r['corrections']}" for _, r in excl.iterrows()]
    return "\n".join(L) + "\n"
