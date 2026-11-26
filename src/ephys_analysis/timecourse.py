"""Pharmacological modulation timecourse (A5 / M6): drug applications with baseline, drug and
washout periods. One recording can hold several applications.

Order, as in 04-06: (1) ``detect_all`` detects spikes over the whole recording and saves one
detection figure per recording; (2) ``feature_table`` computes the period features and the
binned timecourse from that spike table.

Tags (ABF comments) mark the events, in time order (``parse_tags``):
    drug tag     matches ``drug_keywords`` (and no washout keyword); the compound(s) are read
                 from the tag text with the ``drug_names`` aliases (config), e.g. "5ht hit bath"
                 -> 5-HT, "SP + TTX in bath" -> SP, TTX
    washout tag  matches ``washout_keywords``
    other        any other tag (reported, not used)
If no tag matches a keyword, the tags are assigned by order (drug, washout, drug, ...) and the
recording is flagged ``tags_assigned_by_order``.

Applications (``build_applications``): every drug tag starts one application.
    solitary   the previous application (if any) was washed out (washout tag) before this
               drug tag: its baseline is its OWN last ``period_s`` before the tag (not earlier
               than the previous washout tag)
    layered    a drug tag while the previous compound is still in the bath (no washout tag in
               between): the condition is the cumulative set of compounds (e.g. "5-HT + Ba2+")
               and the baseline is the INITIAL baseline of the chain (before its first drug
               tag). The last ``period_s`` before the tag (``pre_tag``) is also reported, with
               the change from it.
    drug       from ``drug_start_after_tag_s`` (60 s) after the tag, ``drug_period_s`` (200 s)
               long (decided 2026-11-26; the same for every cell, not chosen on the response),
               cut at the next tag
    washout    the last ``period_s`` before the next drug tag after the chain's washout tag
               (or before the end of the recording), not earlier than the washout tag
Drug names come from the tag; if the tag names no known compound, from the recording book
(Drug 1, Drug 2, ... in order; flag ``drug_name_from_book``), else "unknown" (flag). A
mismatch between tag, book and protocol name is flagged for review. Names are never invented.

Each period gets the 04 baseline features (spike-excluded Vm, rate, ISI CV, class, AP
waveform). The silent / tonic cut-off is scaled to each period's length (0.1 Hz). Changes:
drug - baseline, washout - baseline. Lowest / highest Vm between the tag and the next tag
(binned timecourse, relative to the baseline Vm) and the baseline drift. Timecourse:
``bin_s`` bins with spike count, rate and spike-excluded median Vm, in time relative to the
first drug tag.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

import numpy as np
import pandas as pd

from .baseline import BaselineSettings, baseline_features, read_voltage, save_figure
from .noise import correct as correct_noise
from .spikes import SPIKE_COLUMNS, SpikeSettings, detect_spikes, spike_mask
from .steps import linear_slope
from .utils import envelope

PERIOD_ORDER = ["baseline", "pre_tag", "drug", "washout"]
DELTA_FEATURES = ["vm_baseline_mV", "firing_rate_Hz", "isi_cv", "ap_threshold_mV", "ap_half_width_ms",
                  "ap_amplitude_mV", "ap_ahp_mV"]
DEFAULT_DRUG_NAMES = {"5-HT": ["5-ht", "5ht", "serotonin"], "SP": ["substance p", "sp"], "TTX": ["ttx"],
                      "Ba2+": ["ba2+", "barium", "ba"], "TQ": ["tertiapin q", "tertiapin", "tq"],
                      "NAD 299": ["nad 299", "nad299", "nad-299"]}


@dataclass
class TimecourseSettings:
    period_s: float = 100.0
    drug_start_after_tag_s: float = 60.0   # drug period starts this long after the drug tag
    drug_period_s: float = 200.0           # and lasts this long
    min_period_s: float = 60.0
    bin_s: float = 10.0
    peak_smooth_bins: int = 3        # lowest / highest point: running mean over 3 bins (30 s)
    max_baseline_drift_mV_per_min: float = 2.0   # flag a drifting baseline period
    drug_keywords: tuple = ("hit bath", "in bath", "bath", "drug", "application", "applied", "wash in", "wash-in")
    washout_keywords: tuple = ("washout", "wash out", "wash-out")
    drug_names: dict = field(default_factory=lambda: dict(DEFAULT_DRUG_NAMES))
    book_drug_prefix: str = "Drug "        # recording-book drug columns: 'Drug 1', 'Drug 2', ...
    book_context_columns: tuple = ("Drug Effects", "Protocol Comments")   # copied for context

    @classmethod
    def from_config(cls, settings: dict | None) -> "TimecourseSettings":
        names = {f.name for f in fields(cls)}
        kw = {k: (tuple(v) if isinstance(v, list) else v) for k, v in (settings or {}).items() if k in names}
        return cls(**kw)


# --------------------------------------------------------------------------- names and tags
def find_compounds(text: str, drug_names: dict) -> list[str]:
    """Canonical compound names in ``text``, in order of appearance (case-insensitive; an
    alias must stand alone, so 'ba' does not match 'bath')."""
    low = str(text or "").lower().replace("_", " ")
    hits = []
    for name, aliases in drug_names.items():
        pos = [m.start() for a in aliases
               for m in re.finditer(r"(?<![a-z0-9])" + re.escape(str(a).lower()) + r"(?![a-z0-9])", low)]
        if pos:
            hits.append((min(pos), name))
    return [n for _, n in sorted(hits)]


def parse_tags(times, comments, s: TimecourseSettings) -> tuple[list[dict], list[str]]:
    """Tags in time order as events {t_s, text, kind: drug / washout / other, compounds}."""
    ev = sorted(zip([float(t) for t in times], [str(c) for c in comments]))
    flags = []
    out = []
    for t, c in ev:
        low = c.lower()
        if any(k in low for k in s.washout_keywords):
            kind = "washout"
        elif any(k in low for k in s.drug_keywords):
            kind = "drug"
        else:
            kind = "other"
        out.append({"t_s": t, "text": c, "kind": kind, "compounds": find_compounds(c, s.drug_names)})
    if out and all(e["kind"] == "other" for e in out):
        for i, e in enumerate(out):
            e["kind"] = "drug" if i % 2 == 0 else "washout"
        flags.append("tags_assigned_by_order")
    n_other = sum(e["kind"] == "other" for e in out)
    if n_other:
        flags.append(f"other_tags={n_other}")
    if not any(e["kind"] == "drug" for e in out):
        flags.append("no_drug_tag")
    return out, flags


def _window(a: float, b: float, s: TimecourseSettings, name: str, flags: list) -> tuple | None:
    if b <= a:
        flags.append(f"no_{name}_period")
        return None
    if b - a < s.min_period_s:
        flags.append(f"{name}_period_short={b - a:.0f}s")
    return (float(a), float(b))


def build_applications(events: list[dict], duration_s: float, s: TimecourseSettings) -> list[dict]:
    """One dict per drug tag: application number, condition, compounds added, layered or
    solitary, tag times, period windows {baseline, pre_tag, drug, washout} and flags."""
    apps, chain, last_wash = [], None, 0.0
    times = [e["t_s"] for e in events]
    for i, e in enumerate(events):
        nxt = times[i + 1] if i + 1 < len(times) else duration_s
        if e["kind"] == "drug":
            layered = chain is not None
            if not layered:
                chain = {"start": e["t_s"], "compounds": [], "apps": [], "lower": last_wash}
            for c in e["compounds"]:
                if c not in chain["compounds"]:
                    chain["compounds"].append(c)
            a = {"application": len(apps) + 1, "drug_tag_s": e["t_s"], "drug_tag_text": e["text"],
                 "compounds_added": list(e["compounds"]), "chain_compounds": list(chain["compounds"]),
                 "layered": layered, "chain_start_s": chain["start"], "chain_lower_s": chain["lower"],
                 "next_tag_s": nxt, "washout_tag_s": np.nan, "washout_tag_text": "", "washout_end_s": np.nan,
                 "prev_tag_s": times[i - 1] if i > 0 else 0.0, "flags": []}
            if layered:
                a["flags"].append("layered_on_previous_compound")
            elif apps:
                a["flags"].append("after_previous_application")
            apps.append(a)
            chain["apps"].append(a)
        elif e["kind"] == "washout":
            if chain is None:
                continue
            nxt_drug = next((x["t_s"] for x in events[i + 1:] if x["kind"] == "drug"), duration_s)
            for a in chain["apps"]:
                a.update(washout_tag_s=e["t_s"], washout_tag_text=e["text"], washout_end_s=nxt_drug)
            chain, last_wash = None, e["t_s"]
    for a in apps:
        fl = a["flags"]
        d = a["drug_tag_s"]
        c0 = a["chain_start_s"]
        win = {"baseline": _window(max(a["chain_lower_s"], c0 - s.period_s), c0, s, "baseline", fl),
               "pre_tag": _window(max(a["prev_tag_s"], d - s.period_s), d, s, "pre_tag", fl) if a["layered"] else None,
               "drug": None, "washout": None}
        start = d + s.drug_start_after_tag_s
        end = min(start + s.drug_period_s, a["next_tag_s"], duration_s)
        if a["next_tag_s"] < start + s.drug_period_s and a["next_tag_s"] < duration_s:
            fl.append("drug_period_cut_by_next_tag")
        win["drug"] = _window(start, end, s, "drug", fl)
        if np.isfinite(a["washout_tag_s"]):
            we = a["washout_end_s"]
            win["washout"] = _window(max(a["washout_tag_s"], we - s.period_s), we, s, "washout", fl)
        else:
            fl.append("no_washout_tag")
        a["windows"] = win
        a["baseline_reference"] = "initial (layered)" if a["layered"] else "own pre-tag"
    return apps


def name_applications(apps: list[dict], book_row: dict | None, protocol: str, s: TimecourseSettings) -> None:
    """Fill ``condition`` (cumulative compounds, e.g. '5-HT + Ba2+') and ``name_source`` of each
    application, in place. Tag first; recording book (Drug 1, Drug 2, ... by order of the
    solitary applications) if the tag names no compound; otherwise 'unknown'. Flags
    mismatches between tag, book and protocol name."""
    book_row = book_row or {}
    pat = re.escape(s.book_drug_prefix) + r"(\d+)"
    drug_cols = sorted((c for c in book_row if re.fullmatch(pat, str(c))),
                       key=lambda c: int(re.fullmatch(pat, str(c)).group(1)))
    book_drugs = [str(book_row[c]).strip() for c in drug_cols
                  if book_row[c] is not None and pd.notna(book_row[c]) and str(book_row[c]).strip()]
    book_set = {c for b in book_drugs for c in (find_compounds(b, s.drug_names) or [b])}
    tag_set = {c for a in apps for c in a["compounds_added"]}
    proto_set = set(find_compounds(protocol, s.drug_names))
    starts = [a for a in apps if not a["layered"]]
    for a in apps:
        a["book_drugs"] = "; ".join(book_drugs)
        if a["chain_compounds"]:
            a["name_source"] = "tag"
        else:
            k = starts.index(a) if a in starts else -1
            if 0 <= k < len(book_drugs):
                a["chain_compounds"] = find_compounds(book_drugs[k], s.drug_names) or [book_drugs[k]]
                a["compounds_added"] = list(a["chain_compounds"])
                a["name_source"] = "recording book"
                a["flags"].append("drug_name_from_book")
            else:
                a["name_source"] = "none"
                a["flags"].append("drug_name_unknown")
        a["condition"] = " + ".join(a["chain_compounds"]) if a["chain_compounds"] else "unknown"
        if tag_set and book_set and not tag_set <= book_set and not book_set <= tag_set:
            a["flags"].append("drug_name_mismatch_book")
        if proto_set and tag_set and not proto_set <= tag_set:
            a["flags"].append("drug_name_mismatch_protocol")


def condition_slug(condition: str) -> str:
    """Column-safe name of a condition: '5-HT' -> '5HT', '5-HT + Ba2+' -> '5HT_Ba2'."""
    return re.sub(r"[^0-9A-Za-z_]", "", str(condition).replace(" + ", "_")) or "unknown"


def bin_timecourse(t: np.ndarray, v: np.ndarray, spike_times, t0: float, bin_s: float, keep: np.ndarray) -> pd.DataFrame:
    """Bins of ``bin_s`` seconds: spike count, rate (Hz) and median Vm (spikes removed),
    with bin centres relative to ``t0`` (the first drug tag)."""
    edges = np.arange(t[0], t[-1] + bin_s, bin_s)
    st = np.asarray(spike_times, dtype=float)
    rows = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (t >= a) & (t < b)
        km = m & keep
        dur = float(min(b, t[-1]) - a)
        n = int(((st >= a) & (st < b)).sum())
        rows.append({"bin_start_s": float(a), "t_rel_drug_s": float((a + b) / 2 - t0), "bin_duration_s": dur,
                     "n_spikes": n, "rate_Hz": n / dur if dur > 0 else np.nan,
                     "vm_median_mV": float(np.median(v[km])) if km.any() else np.nan})
    return pd.DataFrame(rows)


def peak_response(bins: pd.DataFrame, tag_s: float, end_s: float, vm_base: float, s: TimecourseSettings) -> dict:
    """Lowest and highest point after one drug tag, from the binned timecourse between the tag
    and ``end_s`` (the next tag; running mean over ``peak_smooth_bins`` bins):
        drug_min_vm_mV, drug_min_delta_vm_mV, drug_min_time_s
                    lowest smoothed Vm, its difference from the baseline Vm, and when it
                    occurs relative to the tag (bin centre)
        drug_max_vm_mV, drug_max_delta_vm_mV, drug_max_time_s   the same for the highest point
        drug_peak_delta_vm_mV, drug_peak_time_s, drug_peak_direction   whichever of the two is
                    larger in size (the response, for hyperpolarising AND depolarising drugs)
        drug_max_rate_Hz / drug_min_rate_Hz   highest / lowest smoothed rate
    These describe the response; the drug PERIOD is a fixed window (not chosen from them)."""
    keys = ["drug_min_vm_mV", "drug_min_delta_vm_mV", "drug_min_time_s", "drug_max_vm_mV", "drug_max_delta_vm_mV",
            "drug_max_time_s", "drug_peak_delta_vm_mV", "drug_peak_time_s", "drug_max_rate_Hz", "drug_min_rate_Hz"]
    out = {k: np.nan for k in keys}
    out["drug_peak_direction"] = ""
    if not np.isfinite(tag_s) or len(bins) == 0:
        return out
    k = max(1, int(s.peak_smooth_bins))
    vm = bins["vm_median_mV"].rolling(k, center=True, min_periods=1).mean()
    rate = bins["rate_Hz"].rolling(k, center=True, min_periods=1).mean()
    centre = bins["bin_start_s"] + bins["bin_duration_s"] / 2
    end = end_s if np.isfinite(end_s) else np.inf
    inside = (centre >= tag_s) & (centre < end)
    if not inside.any():
        return out
    v_in = vm[inside].dropna()
    if len(v_in):
        i_min, i_max = v_in.idxmin(), v_in.idxmax()
        out.update(drug_min_vm_mV=float(v_in[i_min]), drug_min_time_s=float(centre[i_min] - tag_s),
                   drug_max_vm_mV=float(v_in[i_max]), drug_max_time_s=float(centre[i_max] - tag_s))
        if np.isfinite(vm_base):
            out.update(drug_min_delta_vm_mV=out["drug_min_vm_mV"] - vm_base,
                       drug_max_delta_vm_mV=out["drug_max_vm_mV"] - vm_base)
            down = abs(out["drug_min_delta_vm_mV"]) >= abs(out["drug_max_delta_vm_mV"])
            key = "min" if down else "max"
            out.update(drug_peak_delta_vm_mV=out[f"drug_{key}_delta_vm_mV"], drug_peak_time_s=out[f"drug_{key}_time_s"],
                       drug_peak_direction="hyperpolarising" if down else "depolarising")
    out.update(drug_max_rate_Hz=float(rate[inside].max()), drug_min_rate_Hz=float(rate[inside].min()))
    return out


# --------------------------------------------------------------------------- per recording
def load_recording(row: pd.Series | dict, s: TimecourseSettings, book_row: dict | None = None) -> dict:
    """Read one ready timecourse recording (02 corrections applied), parse its tags and build
    its drug applications."""
    row = dict(row)
    scale = row.get("voltage_scale")
    t, v, abf = read_voltage(row["original_path"], row.get("voltage_source") or "ADC0",
                             scale if scale is not None and pd.notna(scale) else 1.0)
    v, noise_done = correct_noise(t, v, row)       # only if 02 recorded a noise correction
    duration = float(t[-1] - t[0])
    events, flags = parse_tags(abf.tagTimesSec, abf.tagComments, s)
    flags = list(noise_done) + flags
    apps = build_applications(events, duration, s)
    name_applications(apps, book_row, str(abf.protocol), s)
    bias = row.get("bias_current_pA")
    if bias is not None and pd.notna(bias) and float(bias) != 0:
        flags.append("bias_current")
    if abf.sweepCount > 1:
        flags.append("episodic_sweeps_joined")
    first = apps[0]["drug_tag_s"] if apps else np.nan
    info = {"recording_id": row["recording_id"], "cell_tag": row.get("cell_tag"), "abf_filename": row["abf_filename"],
            "abf_protocol": abf.protocol, "recording_duration_s": duration, "bias_current_pA": bias,
            "n_applications": len(apps), "first_drug_tag_s": first,
            "all_tags": "; ".join(f"{e['t_s']:.1f}s {e['text']}" for e in events)}
    return {"t": t, "v": v, "events": events, "applications": apps, "info": info, "flags": flags}


def _label_of(x: float, apps: list[dict]) -> str:
    """'<application>:<period>' of the first window containing time x ('' if none)."""
    for a in apps:
        for p in PERIOD_ORDER:
            w = a["windows"][p]
            if w is not None and w[0] <= x < w[1]:
                return f"{a['application']}:{p}"
    return ""


def _book_row(book: pd.DataFrame | None, recording_id: str) -> dict | None:
    if book is None or "recording_id" not in book:
        return None
    b = book[book["recording_id"] == recording_id]
    return b.iloc[0].to_dict() if len(b) else None


def detect_all(selection: pd.DataFrame, sp: SpikeSettings, s: TimecourseSettings, bs: BaselineSettings, log=print,
               figures_dir: Path | None = None, formats=("png", "svg"),
               book: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Step 1: spikes over the whole recording (+ one detection figure per recording).
    Returns (spikes with a ``period`` column '<application>:<period>', detection: one row
    per recording)."""
    all_sp, summary = [], []
    for _, r in selection.iterrows():
        try:
            rec = load_recording(r, s, _book_row(book, r["recording_id"]))
            d = detect_spikes(rec["t"], rec["v"], sp)
        except Exception as exc:
            log(f"ERROR {r['abf_filename']}: {exc!r}")
            summary.append({"recording_id": r["recording_id"], "abf_filename": r["abf_filename"],
                            "status": f"error: {exc!r}"})
            continue
        apps = rec["applications"]
        d["period"] = [_label_of(x, apps) for x in d["threshold_t_s"]]
        n_bad = int((~d["valid"].astype(bool)).sum()) if len(d) else 0
        conds = "; ".join(f"{a['application']}: {a['condition']}" + (" (layered)" if a["layered"] else "") for a in apps)
        row = {"recording_id": r["recording_id"], "abf_filename": r["abf_filename"], "status": "ok",
               "duration_s": rec["info"]["recording_duration_s"], "tags": rec["info"]["all_tags"],
               "n_applications": len(apps), "applications": conds, "n_spikes_total": len(d),
               "n_flagged_spikes": n_bad, "flags": ";".join(rec["flags"]), "figure": ""}
        if figures_dir is not None:
            row["figure"] = str(plot_recording(rec, d, s, bs, Path(figures_dir) / r["recording_id"], formats=formats)[0])
        summary.append(row)
        d.insert(0, "recording_id", r["recording_id"])
        all_sp.append(d)
        log(f"{r['abf_filename']}: {row['tags'] or 'no tags'} | applications: {conds or 'none'} | spikes {len(d)}"
            + (f" | {row['flags']}" if row["flags"] else ""))
    spikes = (pd.concat(all_sp, ignore_index=True) if all_sp
              else pd.DataFrame(columns=["recording_id"] + SPIKE_COLUMNS + ["period"]))
    return spikes, pd.DataFrame(summary)


def _period_features(t, v, spikes, win, bs: BaselineSettings) -> dict:
    m = (t >= win[0]) & (t < win[1])
    sp_p = spikes[(spikes["threshold_t_s"] >= win[0]) & (spikes["threshold_t_s"] < win[1])] if len(spikes) else spikes
    dur = float(win[1] - win[0])
    bs_p = replace(bs, silent_max_spikes=int(round(bs.silent_max_spikes * dur / bs.window_s)))
    f = baseline_features(t[m], v[m], sp_p.reset_index(drop=True), bs_p)
    f["silent_max_spikes_used"] = bs_p.silent_max_spikes
    return f


def feature_table(selection: pd.DataFrame, spikes: pd.DataFrame, s: TimecourseSettings, bs: BaselineSettings,
                  book: pd.DataFrame | None = None, log=print) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Step 2: (period features: one row per recording x application x period; summary: one
    row per application (one row per recording if it has none) with the period values and
    changes; timecourse bins), from the SAME spikes."""
    period_rows, summary_rows, bins = [], [], []
    book_cols = list(s.book_context_columns)
    for _, r in selection.iterrows():
        brow = _book_row(book, r["recording_id"])
        try:
            rec = load_recording(r, s, brow)
        except Exception as exc:
            log(f"ERROR {r['abf_filename']}: {exc!r}")
            summary_rows.append({"recording_id": r["recording_id"], "cell_tag": r.get("cell_tag"),
                                 "abf_filename": r["abf_filename"], "flags": f"error: {exc!r}"})
            continue
        t, v = rec["t"], rec["v"]
        mine = spikes[spikes["recording_id"] == r["recording_id"]].drop(columns="recording_id") if len(spikes) else spikes
        if len(mine):
            mine = mine.copy()
            mine["valid"] = mine["valid"].astype(bool)
        keep = spike_mask(t, mine["threshold_t_s"].to_numpy() if len(mine) else [], bs.exclude_before_ms,
                          bs.exclude_after_ms)
        t0 = rec["info"]["first_drug_tag_s"]
        bt = bin_timecourse(t, v, mine["threshold_t_s"] if len(mine) else [], t0 if np.isfinite(t0) else 0.0,
                            s.bin_s, keep)
        n_bad = int((~mine["valid"]).sum()) if len(mine) else 0
        rec_flags = rec["flags"] + ([f"flagged_spikes={n_bad}"] if n_bad else [])
        base_info = dict(rec["info"])
        for c in book_cols:
            if brow is not None and c in brow:
                base_info[f"book_{c.lower().replace(' ', '_')}"] = brow[c]
        if not rec["applications"]:
            summary_rows.append({**base_info, "application": 0, "condition": "", "flags": ";".join(rec_flags)})
            log(f"{r['abf_filename']}: no drug application found")
        for a in rec["applications"]:
            summ = {**base_info, "application": a["application"], "condition": a["condition"],
                    "condition_slug": condition_slug(a["condition"]), "compounds_added": " + ".join(a["compounds_added"]),
                    "name_source": a["name_source"], "book_drugs": a["book_drugs"], "layered": a["layered"],
                    "baseline_reference": a["baseline_reference"], "drug_tag_s": a["drug_tag_s"],
                    "drug_tag_text": a["drug_tag_text"], "washout_tag_s": a["washout_tag_s"],
                    "washout_tag_text": a["washout_tag_text"]}
            for p in PERIOD_ORDER:
                w = a["windows"][p]
                summ[f"{p}_start_s"], summ[f"{p}_end_s"] = (w if w else (np.nan, np.nan))
                if w is None:
                    if p != "pre_tag":
                        period_rows.append({"recording_id": r["recording_id"], "abf_filename": r["abf_filename"],
                                            "application": a["application"], "condition": a["condition"], "period": p})
                    continue
                f = _period_features(t, v, mine, w, bs)
                period_rows.append({"recording_id": r["recording_id"], "abf_filename": r["abf_filename"],
                                    "application": a["application"], "condition": a["condition"], "period": p, **f})
                for k in ["firing_class", "vm_sd_mV"] + DELTA_FEATURES:
                    summ[f"{p}_{k}"] = f.get(k, np.nan)
            for p, ref in (("drug", "baseline"), ("washout", "baseline"), ("drug", "pre_tag")):
                name = f"delta_{p}_vs_pre_tag" if ref == "pre_tag" else f"delta_{p}"
                if ref == "pre_tag" and not a["layered"]:
                    continue
                for k in DELTA_FEATURES:
                    x, y = summ.get(f"{p}_{k}", np.nan), summ.get(f"{ref}_{k}", np.nan)
                    summ[f"{name}_{k}"] = x - y if x is not None and y is not None else np.nan
            summ.update(peak_response(bt, a["drug_tag_s"], a["next_tag_s"], summ.get("baseline_vm_baseline_mV", np.nan), s))
            drift = np.nan
            wb = a["windows"]["baseline"]
            if wb is not None:
                inb = (bt["bin_start_s"] >= wb[0]) & (bt["bin_start_s"] + s.bin_s <= wb[1] + 1e-9)
                drift = linear_slope(bt.loc[inb, "bin_start_s"] / 60.0, bt.loc[inb, "vm_median_mV"], min_points=3)
            summ["baseline_drift_mV_per_min"] = drift
            fl = rec_flags + a["flags"]
            if np.isfinite(drift) and abs(drift) > s.max_baseline_drift_mV_per_min:
                fl.append(f"unstable_baseline={drift:+.1f}mV/min")
            summ["flags"] = ";".join(fl)
            summary_rows.append(summ)
            log(f"{r['abf_filename']} #{a['application']} {a['condition']}"
                + (" (layered)" if a["layered"] else "") + f": Vm {summ.get('baseline_vm_baseline_mV', np.nan):.1f} -> "
                f"{summ.get('drug_vm_baseline_mV', np.nan):.1f} -> {summ.get('washout_vm_baseline_mV', np.nan):.1f} mV | "
                f"lowest {summ['drug_min_delta_vm_mV']:+.1f} mV at {summ['drug_min_time_s']:.0f} s")
        bt.insert(0, "recording_id", r["recording_id"])
        bins.append(bt)
    bins = pd.concat(bins, ignore_index=True) if bins else pd.DataFrame()
    return pd.DataFrame(period_rows), pd.DataFrame(summary_rows), bins


# --------------------------------------------------------------------------- figure
def plot_recording(rec: dict, spikes: pd.DataFrame, s: TimecourseSettings, bs: BaselineSettings,
                   out_stem: Path | None = None, formats=("png", "svg"), dpi: int = 200):
    """Detection figure: (A) whole recording (min/max envelope), tags, the periods of every
    application shaded, spikes marked; (B) binned firing rate and (C) binned Vm (spikes
    removed) against time from the first drug tag, with the lowest point of each application;
    (D) spikes aligned to threshold, coloured by period."""
    import matplotlib
    if out_stem is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {"baseline": "tab:blue", "pre_tag": "tab:purple", "drug": "tab:red", "washout": "tab:green"}
    t, v, info, apps = rec["t"], rec["v"], rec["info"], rec["applications"]
    t0 = info["first_drug_tag_s"] if np.isfinite(info["first_drug_tag_s"]) else 0.0
    fig = plt.figure(figsize=(13, 10), constrained_layout=True)
    gs = fig.add_gridspec(4, 2, height_ratios=[3, 1.3, 1.3, 2.5])
    ax = fig.add_subplot(gs[0, :])
    ax_r = fig.add_subplot(gs[1, :])
    ax_v = fig.add_subplot(gs[2, :], sharex=ax_r)
    ax_w = fig.add_subplot(gs[3, 0])
    ax_p = fig.add_subplot(gs[3, 1])
    te, ve = envelope(t, v, 6000)
    ax.plot(te, ve, color="0.3", lw=0.3)
    shaded: dict = {}                     # window -> (colour, labels); a window can serve two roles
    for a in apps:
        for p in PERIOD_ORDER:
            w = a["windows"][p]
            if w is None:
                continue
            col, labs = shaded.setdefault(w, (colours[p], []))
            lab = f"#{a['application']} {a['condition']}: {p}"
            if lab not in labs and not (p == "baseline" and a["layered"]):
                labs.append(lab)
    for w, (col, labs) in shaded.items():
        ax.axvspan(*w, color=col, alpha=0.12, label=" = ".join(labs) + f" ({w[1] - w[0]:.0f} s)")
        for x in (ax_r, ax_v):
            x.axvspan(w[0] - t0, w[1] - t0, color=col, alpha=0.12)
    for e in rec["events"]:
        ax.axvline(e["t_s"], color="k", lw=0.8, ls="--" if e["kind"] != "other" else ":")
        ax.text(e["t_s"], 0.98, f" {e['text']} ({e['t_s']:.0f} s)", fontsize=7, va="top", ha="left", rotation=90,
                transform=ax.get_xaxis_transform())
        for x in (ax_r, ax_v):
            x.axvline(e["t_s"] - t0, color="k", lw=0.8, ls="--" if e["kind"] != "other" else ":")
    if len(spikes):
        ok = spikes["valid"].astype(bool)
        ax.plot(spikes.loc[ok, "peak_t_s"], spikes.loc[ok, "peak_mV"] + 3, "v", color="tab:red", ms=2)
        ax.plot(spikes.loc[~ok, "peak_t_s"], spikes.loc[~ok, "peak_mV"] + 3, "v", color="0.6", ms=2)
    ax.set_ylabel("Vm (mV)")
    ax.set_xlabel("Time (s)")
    ax.legend(fontsize=6, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.set_title("Whole recording: periods shaded (per application), tags dashed, ▼ = spike (grey = flagged)",
                 fontsize=9)

    keep = spike_mask(t, spikes["threshold_t_s"].to_numpy() if len(spikes) else [], bs.exclude_before_ms,
                      bs.exclude_after_ms)
    bt = bin_timecourse(t, v, spikes["threshold_t_s"] if len(spikes) else [], t0, s.bin_s, keep)
    ax_r.step(bt["t_rel_drug_s"], bt["rate_Hz"], where="mid", color="k", lw=0.8)
    ax_r.set_ylabel(f"Rate (Hz, {s.bin_s:g} s bins)")
    ax_v.plot(bt["t_rel_drug_s"], bt["vm_median_mV"], color="k", lw=0.8)
    for a in apps:
        wb = a["windows"]["baseline"]
        if wb is None:
            continue
        mb = (t >= wb[0]) & (t < wb[1]) & keep
        vm_base = float(np.median(v[mb])) if mb.any() else np.nan
        pk = peak_response(bt, a["drug_tag_s"], a["next_tag_s"], vm_base, s)
        if np.isfinite(pk["drug_peak_delta_vm_mV"]):
            down = pk["drug_peak_direction"] == "hyperpolarising"
            ax_v.plot(pk["drug_peak_time_s"] + a["drug_tag_s"] - t0, vm_base + pk["drug_peak_delta_vm_mV"],
                      "v" if down else "^", color="tab:red", ms=6,
                      label=f"#{a['application']} {a['condition']}: {'lowest' if down else 'highest'} "
                            f"{pk['drug_peak_delta_vm_mV']:+.1f} mV at {pk['drug_peak_time_s']:.0f} s "
                            f"(vs {vm_base:.1f} mV)")
    if apps:
        ax_v.legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax_v.set_ylabel("Vm, spikes removed (mV)")
    ax_v.set_xlabel("Time from first drug tag (s)")

    dt = t[1] - t[0]
    pre, post = int(0.005 / dt), int(0.015 / dt)
    for _, sp in spikes.groupby("period", group_keys=False).head(60).iterrows() if len(spikes) else []:
        i = int(round((sp["threshold_t_s"] - t[0]) / dt))
        a_, b_ = max(0, i - pre), min(t.size, i + post)
        p = str(sp["period"]).split(":")[-1]
        col = colours.get(p, "0.7") if sp["valid"] else "0.8"
        ax_w.plot((t[a_:b_] - sp["threshold_t_s"]) * 1e3, v[a_:b_], color=col, lw=0.5, alpha=0.7)
        ax_w.plot(0, sp["threshold_mV"], "o", color="k", ms=2)
        ax_p.plot(v[a_:b_], np.gradient(v[a_:b_], dt * 1e3), color=col, lw=0.5, alpha=0.7)
    ax_w.set_xlabel("Time from threshold (ms)")
    ax_w.set_ylabel("Vm (mV)")
    ax_w.set_title("Spikes: blue baseline, purple pre-tag, red drug,\ngreen washout, grey other (max 60 each)",
                   fontsize=9)
    ax_p.set_xlabel("Vm (mV)")
    ax_p.set_ylabel("dV/dt (mV/ms)")
    ax_p.set_title("Phase plot", fontsize=9)
    for x in (ax, ax_r, ax_v, ax_w, ax_p):
        x.spines[["top", "right"]].set_visible(False)
    n_bad = int((~spikes["valid"].astype(bool)).sum()) if len(spikes) else 0
    fig.suptitle(f"{info['abf_filename']} | {info['abf_protocol']} | {len(apps)} application(s) | {len(spikes)} "
                 f"spikes ({n_bad} flagged)" + (f" | {';'.join(rec['flags'])}" if rec["flags"] else ""), fontsize=9)
    if out_stem is None:
        return fig
    return save_figure(fig, out_stem, formats, dpi)
