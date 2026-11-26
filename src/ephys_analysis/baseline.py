"""Spontaneous baseline (A1 / M2, 0 pA): silent / tonic classification and baseline features.

Per recording:
    window       the last ``window_s`` seconds of the recording (default 100 s)
    spikes       ``spikes.detect_spikes`` on the window
    class        silent if n_spikes <= ``silent_max_spikes``, otherwise tonic
    firing rate  n_spikes / window duration                                      [Hz]
    ISI CV       SD / mean of inter-spike intervals (>= ``min_intervals_for_cv``)
    Vm           median membrane potential of the window after removing every spike
                 from ``exclude_before_ms`` before to ``exclude_after_ms`` after its
                 threshold                                                          [mV]
    AP waveform  median over valid, isolated spikes (previous ISI >= ``isolated_min_isi_s``;
                 the first spike of the window counts as isolated)

Vm is called ``vm_baseline_mV``: in a firing cell it is not a true resting potential.
Liquid junction potential is NOT corrected. A recording with a bias current recorded in
02 (``bias_current_pA``) is flagged ``bias_current``: its Vm is not the 0 pA potential.
A recording with ABF tags is flagged ``abf_tags_present`` (read the tag: e.g. a recording
made after a drug washout is not a drug-naive baseline).
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd

from .abf import open_abf, standard_scale
from .noise import correct as correct_noise
from .spikes import SPIKE_COLUMNS, SpikeSettings, detect_spikes, isi_stats, spike_mask
from .utils import envelope

AP_FEATURES = ["threshold_mV", "peak_mV", "amplitude_mV", "half_width_ms", "upstroke_mV_ms",
               "downstroke_mV_ms", "upstroke_downstroke_ratio", "ahp_mV"]


@dataclass
class BaselineSettings:
    window_s: float = 100.0
    min_window_s: float = 60.0           # shorter recordings are analysed but flagged
    silent_max_spikes: int = 10
    min_intervals_for_cv: int = 5
    exclude_before_ms: float = 5.0
    exclude_after_ms: float = 50.0
    isolated_min_isi_s: float = 0.5

    @classmethod
    def from_config(cls, settings: dict | None) -> "BaselineSettings":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (settings or {}).items() if k in names})


# --------------------------------------------------------------------------- reading
def read_voltage(abf_path: Path | str, voltage_source: str = "ADC0", voltage_scale: float = 1.0):
    """Continuous voltage trace (mV) of one recording, with the 02 corrections applied.

    ``voltage_source`` is 'ADCn' from analysis_ready.csv; ``voltage_scale`` multiplies the
    signal (gain correction). Episodic files are joined end to end (flagged by the caller).
    Returns (t_s, v_mV, abf).
    """
    abf = open_abf(abf_path)
    adc = int(str(voltage_source).replace("ADC", ""))
    unit_scale, unit = standard_scale(abf.adcUnits[adc])
    if unit != "mV":
        raise ValueError(f"ADC{adc} has unit '{abf.adcUnits[adc]}', not a voltage")
    ts, vs, offset = [], [], 0.0
    for s in range(abf.sweepCount):
        abf.setSweep(s, channel=adc)
        t = np.asarray(abf.sweepX, dtype=float)
        ts.append(t + offset)
        vs.append(np.asarray(abf.sweepY, dtype=float) * unit_scale * float(voltage_scale))
        offset = ts[-1][-1] + (t[1] - t[0])
    return np.concatenate(ts), np.concatenate(vs), abf


def analysis_window(t: np.ndarray, s: BaselineSettings) -> tuple[np.ndarray, list[str]]:
    """Boolean mask of the last ``window_s`` seconds, and flags."""
    flags = []
    duration = float(t[-1] - t[0])
    if duration < s.window_s:
        flags.append("recording_shorter_than_window")
        if duration < s.min_window_s:
            flags.append("window_below_minimum")
        return np.ones(t.size, dtype=bool), flags
    return t >= t[-1] - s.window_s, flags


def classify_firing(n_spikes: int, s: BaselineSettings) -> str:
    """'silent' if n_spikes <= silent_max_spikes in the window, else 'tonic'."""
    return "silent" if n_spikes <= s.silent_max_spikes else "tonic"


# --------------------------------------------------------------------------- features
def baseline_features(t: np.ndarray, v: np.ndarray, spikes: pd.DataFrame, s: BaselineSettings) -> dict:
    """Features of one analysis window (t, v already cut to the window)."""
    duration = float(t[-1] - t[0] + (t[1] - t[0]))
    n = len(spikes)
    times = spikes["threshold_t_s"].to_numpy() if n else np.array([])
    keep = spike_mask(t, times, s.exclude_before_ms, s.exclude_after_ms)
    vm = v[keep]
    out = {"window_start_s": float(t[0]), "window_end_s": float(t[-1]), "window_duration_s": duration,
           "n_spikes": n, "n_valid_spikes": int(spikes["valid"].sum()) if n else 0,
           "firing_class": classify_firing(n, s), "firing_rate_Hz": n / duration,
           "vm_baseline_mV": float(np.median(vm)) if vm.size else np.nan,
           "vm_mean_mV": float(np.mean(vm)) if vm.size else np.nan,
           "vm_sd_mV": float(np.std(vm, ddof=1)) if vm.size > 1 else np.nan,
           "fraction_time_excluded": float(1 - keep.mean())}
    out.update(isi_stats(times, s.min_intervals_for_cv))
    # AP waveform: valid, isolated spikes
    if n:
        isi = spikes["isi_before_s"].fillna(np.inf)
        use = spikes[spikes["valid"] & (isi >= s.isolated_min_isi_s)]
    else:
        use = spikes
    out["n_spikes_waveform"] = len(use)
    for f in AP_FEATURES:
        out[f"ap_{f}"] = float(use[f].median()) if len(use) else np.nan
    return out


# --------------------------------------------------------------------------- per recording
def load_window(row: pd.Series | dict, bs: BaselineSettings) -> dict:
    """Read one ready baseline recording (a row of analysis_ready.csv) and its window.

    Returns a dict with t, v (whole recording), window (mask), info (identifiers and
    context for the feature table) and flags (recording-level)."""
    row = dict(row)
    scale = row.get("voltage_scale")
    t, v, abf = read_voltage(row["original_path"], row.get("voltage_source") or "ADC0",
                             scale if scale is not None and pd.notna(scale) else 1.0)
    v, noise_done = correct_noise(t, v, row)       # only if 02 recorded a noise correction
    flags = list(noise_done)
    if abf.sweepCount > 1:
        flags.append("episodic_sweeps_joined")
    bias = row.get("bias_current_pA")
    if bias is not None and pd.notna(bias) and float(bias) != 0:
        flags.append("bias_current")
    if len(abf.tagComments):
        flags.append("abf_tags_present")   # e.g. after a washout: check it is a true baseline
    win, wflags = analysis_window(t, bs)
    info = {"recording_id": row["recording_id"], "cell_tag": row.get("cell_tag"),
            "abf_filename": row["abf_filename"], "recording_duration_s": float(t[-1] - t[0]),
            "sample_rate_Hz": float(abf.dataRate), "bias_current_pA": bias,
            "abf_tags": "; ".join(f"{tt:.1f}s {c}" for tt, c in zip(abf.tagTimesSec, abf.tagComments))}
    return {"t": t, "v": v, "window": win, "info": info, "flags": flags + wflags}


def spikes_of(spikes: pd.DataFrame, recording_id: str) -> pd.DataFrame:
    """The rows of one recording from a multi-recording spike table (without recording_id)."""
    if spikes.empty or "recording_id" not in spikes:
        return pd.DataFrame(columns=SPIKE_COLUMNS)
    s = spikes[spikes["recording_id"] == recording_id].drop(columns="recording_id").reset_index(drop=True)
    s["valid"] = s["valid"].astype(bool)
    return s


# --------------------------------------------------------------------------- step 1: detection
def detect_all(selection: pd.DataFrame, sp: SpikeSettings, bs: BaselineSettings, log=print,
               figures_dir: Path | None = None, formats=("png", "svg")) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Step 1: detect spikes in the window of every recording and save one detection figure
    per recording (``figures_dir/<recording_id>.<fmt>``).

    Returns (spikes: one row per spike, detection: one row per recording with counts and
    flags). Only the spike table is kept, never the traces. Errors are recorded per recording."""
    all_spikes, summary = [], []
    for _, r in selection.iterrows():
        try:
            rec = load_window(r, bs)
            tw, vw = rec["t"][rec["window"]], rec["v"][rec["window"]]
            s = detect_spikes(tw, vw, sp)
        except Exception as exc:
            log(f"ERROR {r['abf_filename']}: {exc!r}")
            summary.append({"recording_id": r["recording_id"], "abf_filename": r["abf_filename"],
                            "status": f"error: {exc!r}"})
            continue
        n_bad = int((~s["valid"]).sum()) if len(s) else 0
        row = {"recording_id": r["recording_id"], "abf_filename": r["abf_filename"], "status": "ok",
               "window_s": float(tw[-1] - tw[0]), "n_spikes": len(s), "n_flagged_spikes": n_bad,
               "flags": ";".join(rec["flags"]), "figure": ""}
        if figures_dir is not None:
            feats = {**rec["info"], **baseline_features(tw, vw, s, bs), "flags": row["flags"]}
            paths = plot_recording(rec, s, feats, Path(figures_dir) / r["recording_id"], formats=formats)
            row["figure"] = str(paths[0])
        summary.append(row)
        s.insert(0, "recording_id", r["recording_id"])
        all_spikes.append(s)
        log(f"{r['abf_filename']}: {len(s)} spikes ({n_bad} flagged) in the last {row['window_s']:.0f} s"
            + (f" | {row['flags']}" if row["flags"] else ""))
    spikes = (pd.concat(all_spikes, ignore_index=True) if all_spikes
              else pd.DataFrame(columns=["recording_id"] + SPIKE_COLUMNS))
    return spikes, pd.DataFrame(summary)


# --------------------------------------------------------------------------- step 2: features
def feature_table(selection: pd.DataFrame, spikes: pd.DataFrame, bs: BaselineSettings, log=print) -> pd.DataFrame:
    """Step 2: one row of baseline features per recording, computed from the SAME spike
    table that step 1 plotted (the ABF is re-read for Vm)."""
    rows = []
    for _, r in selection.iterrows():
        try:
            rec = load_window(r, bs)
        except Exception as exc:
            log(f"ERROR {r['abf_filename']}: {exc!r}")
            rows.append({"recording_id": r["recording_id"], "cell_tag": r.get("cell_tag"),
                         "abf_filename": r["abf_filename"], "flags": f"error: {exc!r}"})
            continue
        s = spikes_of(spikes, r["recording_id"])
        f = {**rec["info"], **baseline_features(rec["t"][rec["window"]], rec["v"][rec["window"]], s, bs)}
        flags = list(rec["flags"])
        n_bad = int((~s["valid"]).sum()) if len(s) else 0
        if n_bad:
            flags.append(f"flagged_spikes={n_bad}")
        f["flags"] = ";".join(flags)
        rows.append(f)
        log(f"{r['abf_filename']}: {f['firing_class']} | {f['n_spikes']} spikes ({f['firing_rate_Hz']:.2f} Hz) | "
            f"Vm {f['vm_baseline_mV']:.1f} mV | ISI CV {f['isi_cv']:.2f}")
    features = pd.DataFrame(rows)
    if not features.empty:
        counts = features.groupby("cell_tag", dropna=False)["recording_id"].transform("count")
        features.insert(3, "n_baseline_recordings_for_cell", counts)
    return features


# --------------------------------------------------------------------------- figure
def save_figure(fig, out_stem: Path, formats=("png", "svg"), dpi: int = 200) -> list[Path]:
    """Save ``fig`` once per format and close it."""
    import matplotlib.pyplot as plt

    out_stem.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        p = out_stem.with_suffix(f".{fmt}")
        fig.savefig(p, dpi=dpi)
        paths.append(p)
    plt.close(fig)
    return paths


def plot_recording(rec: dict, spikes: pd.DataFrame, feats: dict, out_stem: Path | None = None,
                   formats=("png", "svg"), dpi: int = 200):
    """Detection figure: (A) whole recording (min/max envelope) with the analysis window
    shaded and every spike marked; (B) spikes aligned to threshold, threshold marked
    (grey = flagged); (C) phase plot dV/dt vs V.
    Returns the figure, or the saved paths with ``out_stem``."""
    import matplotlib
    if out_stem is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t, v, win = rec["t"], rec["v"], rec["window"]
    fig = plt.figure(figsize=(12, 6.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    ax = fig.add_subplot(gs[0, :])
    te, ve = envelope(t, v, 4000)
    ax.plot(te, ve, color="0.3", lw=0.4)
    ax.axvspan(t[win][0], t[win][-1], color="tab:blue", alpha=0.08, label="analysis window")
    ok = spikes["valid"].astype(bool) if len(spikes) else pd.Series(dtype=bool)
    if len(spikes):
        ax.plot(spikes.loc[ok, "peak_t_s"], spikes.loc[ok, "peak_mV"] + 3, "v", color="tab:red", ms=3, label="spike")
        if (~ok).any():
            ax.plot(spikes.loc[~ok, "peak_t_s"], spikes.loc[~ok, "peak_mV"] + 3, "v", color="0.6", ms=3,
                    label="flagged spike")
    ax.axhline(feats["vm_baseline_mV"], color="tab:blue", lw=0.8, ls="--", label="Vm (spikes excluded)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Vm (mV)")
    ax.legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    n_bad = int((~ok).sum()) if len(spikes) else 0
    ax.set_title(f"{feats['abf_filename']} | {feats['firing_class']} | {feats['n_spikes']} spikes "
                 f"({n_bad} flagged, {feats['firing_rate_Hz']:.2f} Hz) | Vm {feats['vm_baseline_mV']:.1f} mV"
                 + (f" | {feats['flags']}" if feats.get("flags") else ""), fontsize=9)

    ax_w, ax_p = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    dt = t[1] - t[0]
    pre, post = int(0.005 / dt), int(0.015 / dt)
    for _, s in spikes.head(200).iterrows():
        i = int(round((s["threshold_t_s"] - t[0]) / dt))
        a, b = max(0, i - pre), min(t.size, i + post)
        seg = v[a:b]
        col = "tab:red" if s["valid"] else "0.6"
        ax_w.plot((t[a:b] - s["threshold_t_s"]) * 1e3, seg, color=col, lw=0.5, alpha=0.6)
        ax_w.plot(0, s["threshold_mV"], "o", color="k", ms=2)
        ax_p.plot(seg, np.gradient(seg, dt * 1e3), color=col, lw=0.5, alpha=0.6)
    if len(spikes) > 200:
        ax_w.text(0.98, 0.95, f"first 200 of {len(spikes)} spikes", transform=ax_w.transAxes, ha="right", fontsize=7)
    ax_w.set_xlabel("Time from threshold (ms)")
    ax_w.set_ylabel("Vm (mV)")
    ax_w.set_title("Spikes aligned to threshold (black dot = threshold; grey = flagged)", fontsize=9)
    ax_p.set_xlabel("Vm (mV)")
    ax_p.set_ylabel("dV/dt (mV/ms)")
    ax_p.set_title("Phase plot", fontsize=9)
    for a in (ax, ax_w, ax_p):
        a.spines[["top", "right"]].set_visible(False)
    if out_stem is None:
        return fig
    return save_figure(fig, out_stem, formats, dpi)
