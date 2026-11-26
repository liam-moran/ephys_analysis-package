"""Current steps (A3 / M3): sag, input resistance, F-I, rebound, AP at the first spiking step.

Order, as in 04: (1) ``detect_all`` finds spikes in every sweep and saves one detection
figure per recording; (2) ``feature_table`` computes the features from that spike table.

Protocol (read from the command DAC of every file, never hard-coded): one current step per
sweep, the same timing in all sweeps, a different amplitude per sweep. The step amplitude
is the command step; an amplifier bias current from 02 (``bias_current_pA``) is added to
give the total current. The ``command_as_current`` correction from 02 means the command
is used because the current itself was not recorded.

Per sweep (windows in config.yaml):
    vm_pre_mV         median V in the ``pre_window_s`` before the step, spikes removed
    rate_pre_Hz       spikes before the step / time before the step (spontaneous firing)
    I_step_pA         command during the step minus command before it
    n_spikes_step, rate_step_Hz, latency_first_spike_ms (from step onset to threshold)
    isi_cv_step, adaptation_index  (Allen: mean of (ISI[n+1]-ISI[n])/(ISI[n+1]+ISI[n]);
                      NaN with fewer than ``min_spikes_adaptation`` spikes)
    v_min_mV          minimum V during the step (Allen peak deflection; or the first
                      ``sag_window_s`` if set)
    v_ss_mV           median V in the last ``steady_state_s`` of the step (spikes removed)
    sag_ratio         (v_ss - v_min) / (vm_pre - v_min)   hyperpolarising steps only
    rebound_peak_mV   max V (spikes removed) in the first ``rebound_peak_window_s`` after
                      the step, minus vm_pre
    n_rebound_spikes_short / _all, latency_first_rebound_spike_ms, n_rebound_bursts
                      spikes after the step within ``rebound_spike_window_s`` / to the sweep end

Per recording:
    rin_MOhm          slope of (v_ss - vm_pre) vs I_step over hyperpolarising sweeps (>= 2)
    first_spiking_step_pA  smallest positive step with >= 1 spike during the step (coarse: the
                      step increment, e.g. 50 pA)
    fi_slope_Hz_per_pA  linear fit of rate_step vs I_step from the first spiking step upwards (>= 2 sweeps)
    sag / rebound     at the most negative step
    adaptation_index  from the step with the most spikes (>= ``min_spikes_adaptation``);
                      NaN and flag ``too_few_spikes_for_adaptation`` otherwise
    ap_*              waveform of the first valid spike during the first spiking step
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd

from .abf import open_abf, read_sweeps
from .baseline import AP_FEATURES, save_figure
from .membrane_test import command_edges
from .spikes import SPIKE_COLUMNS, SpikeSettings, detect_attenuated, detect_spikes, isi_stats, spike_mask
from .noise import correct as correct_noise
from .utils import envelope


@dataclass
class StepSettings:
    pre_window_s: float = 0.5
    sag_window_s: float | None = None     # None = whole step (Allen peak deflection)
    steady_state_s: float = 0.1
    rebound_peak_window_s: float = 0.5
    rebound_spike_window_s: float = 5.0
    exclude_before_ms: float = 5.0
    exclude_after_ms: float = 50.0
    min_spikes_adaptation: int = 4
    min_intervals_for_cv: int = 3
    burst_max_isi_s: float = 0.1
    burst_min_spikes: int = 2
    burst_min_ibi_s: float = 0.2
    command_tol_pA: float = 1.0
    attenuated_prominence_mV: float = 20.0

    @classmethod
    def from_config(cls, settings: dict | None) -> "StepSettings":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (settings or {}).items() if k in names})


# --------------------------------------------------------------------------- small, testable pieces
def step_window(commands: list[np.ndarray], tol: float = 1.0) -> tuple[int, int] | None:
    """Sample indices (onset, offset) of the current step, from the sweep with the largest
    command change. None if no sweep has exactly one step (two edges)."""
    best = None
    for c in commands:
        e = command_edges(np.asarray(c, dtype=float), tol)
        if e.size == 2:
            size = float(np.max(np.abs(c - c[0])))
            if best is None or size > best[0]:
                best = (size, int(e[0]), int(e[1]))
    return None if best is None else (best[1], best[2])


def detect_bursts(spike_times, max_isi_s: float, min_spikes: int, min_ibi_s: float) -> list[list[float]]:
    """MaxInterval bursts: consecutive spikes <= ``max_isi_s`` apart, >= ``min_spikes`` each;
    bursts closer than ``min_ibi_s`` are merged. (Adapted from the earlier step-analysis notebook.)"""
    st = list(np.asarray(spike_times, dtype=float))
    if len(st) < min_spikes:
        return []
    bursts, cur = [], [st[0]]
    for x in st[1:]:
        if x - cur[-1] <= max_isi_s:
            cur.append(x)
        else:
            if len(cur) >= min_spikes:
                bursts.append(cur)
            cur = [x]
    if len(cur) >= min_spikes:
        bursts.append(cur)
    merged: list[list[float]] = []
    for b in bursts:
        if merged and b[0] - merged[-1][-1] < min_ibi_s:
            merged[-1].extend(b)
        else:
            merged.append(list(b))
    return merged


def adaptation_index(spike_times, min_spikes: int = 4) -> float:
    """Allen adaptation index: mean of (ISI[n+1] - ISI[n]) / (ISI[n+1] + ISI[n]).
    Positive = slowing down. NaN with fewer than ``min_spikes`` spikes."""
    isi = np.diff(np.asarray(spike_times, dtype=float))
    if isi.size + 1 < max(min_spikes, 3):
        return np.nan
    return float(np.mean(np.diff(isi) / (isi[1:] + isi[:-1])))


def linear_slope(x, y, min_points: int = 2) -> float:
    """Least-squares slope of y on x; NaN with fewer than ``min_points`` distinct x."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if np.unique(x[ok]).size < min_points:
        return np.nan
    return float(np.polyfit(x[ok], y[ok], 1)[0])


def sweep_features(t: np.ndarray, v: np.ndarray, on: int, off: int, I_step: float,
                   spikes: pd.DataFrame, s: StepSettings) -> dict:
    """Features of one sweep. ``spikes``: detect_spikes output for the whole sweep."""
    dt = float(t[1] - t[0])
    times = spikes["threshold_t_s"].to_numpy() if len(spikes) else np.array([])
    keep = spike_mask(t, times, s.exclude_before_ms, s.exclude_after_ms)
    t_on, t_off = t[on], t[off]

    def med(a, b):
        m = (t >= a) & (t < b) & keep
        return float(np.median(v[m])) if m.any() else np.nan

    pre = times[times < t_on]
    step = times[(times >= t_on) & (times < t_off)]
    post = times[times >= t_off]
    vm_pre = med(t_on - s.pre_window_s, t_on)
    out = {"I_step_pA": I_step, "step_onset_s": float(t_on), "step_duration_s": float(t_off - t_on),
           "vm_pre_mV": vm_pre, "n_spikes_pre": int(pre.size),
           "rate_pre_Hz": pre.size / max(t_on - t[0], dt),
           "n_spikes_step": int(step.size), "rate_step_Hz": step.size / (t_off - t_on),
           "latency_first_spike_ms": float((step[0] - t_on) * 1e3) if step.size else np.nan}
    st = isi_stats(step, s.min_intervals_for_cv)
    out["isi_cv_step"] = st["isi_cv"]
    out["adaptation_index"] = adaptation_index(step, s.min_spikes_adaptation)

    sag_end = t_off if s.sag_window_s is None else min(t_on + s.sag_window_s, t_off)
    sag_win = (t >= t_on) & (t < sag_end)
    v_min = float(np.min(v[sag_win])) if sag_win.any() else np.nan
    v_ss = med(t_off - s.steady_state_s, t_off)
    out.update(v_min_mV=v_min, v_ss_mV=v_ss, delta_v_ss_mV=v_ss - vm_pre)
    out["sag_ratio"] = ((v_ss - v_min) / (vm_pre - v_min)
                        if I_step < 0 and np.isfinite(v_min) and vm_pre - v_min > 0 else np.nan)

    rb = (t >= t_off) & (t < t_off + s.rebound_peak_window_s) & keep
    out["rebound_peak_mV"] = float(np.max(v[rb]) - vm_pre) if rb.any() else np.nan
    short = post[post < t_off + s.rebound_spike_window_s]
    out.update(n_rebound_spikes_short=int(short.size), n_rebound_spikes_all=int(post.size),
               rate_after_step_Hz=post.size / max(t[-1] - t_off, dt),
               latency_first_rebound_spike_ms=float((post[0] - t_off) * 1e3) if post.size else np.nan,
               n_rebound_bursts=len(detect_bursts(post, s.burst_max_isi_s, s.burst_min_spikes, s.burst_min_ibi_s)))
    return out


# --------------------------------------------------------------------------- per recording
def load_steps(row: pd.Series | dict, s: StepSettings) -> dict:
    """Read one ready step recording: voltage (02 corrections applied), command current,
    step window and per-sweep step amplitude."""
    row = dict(row)
    src = str(row.get("current_source") or "")
    if not src.startswith("command_DAC") and not src.startswith("measured_ADC"):
        raise ValueError(f"no current information (current_source = '{src}')")
    abf = open_abf(row["original_path"])
    adc = int(str(row.get("voltage_source") or "ADC0").replace("ADC", ""))
    if src.startswith("command_DAC"):
        dac = int(src.replace("command_DAC", ""))
    else:  # measured current: the command DAC still defines the step window
        dac = 1 if 1 < len(abf.dacUnits) else 0
    sweeps = read_sweeps(abf, adc, dac)
    scale = row.get("voltage_scale")
    scale = float(scale) if scale is not None and pd.notna(scale) else 1.0
    win = step_window([sw.command for sw in sweeps], s.command_tol_pA)
    if win is None:
        raise ValueError("no single current step found in the command waveform")
    on, off = win
    bias = row.get("bias_current_pA")
    bias = float(bias) if bias is not None and pd.notna(bias) else 0.0
    i_steps = [float(np.median(sw.command[on:off]) - np.median(sw.command[max(0, on - int(0.1 / (sw.time_s[1] - sw.time_s[0]))):on]))
               for sw in sweeps]
    flags = []
    if bias:
        flags.append("bias_current")
    if src.startswith("command_DAC"):
        flags.append("current_from_command")
    info = {"recording_id": row["recording_id"], "cell_tag": row.get("cell_tag"), "abf_filename": row["abf_filename"],
            "n_sweeps": len(sweeps), "sample_rate_Hz": float(abf.dataRate), "bias_current_pA": bias,
            "step_onset_s": float(sweeps[0].time_s[on]), "step_duration_s": float(sweeps[0].time_s[off] - sweeps[0].time_s[on]),
            "current_source": src}
    traces, noise_done = [], []
    for sw in sweeps:                              # noise correction only if 02 recorded one
        v, noise_done = correct_noise(sw.time_s, sw.signal * scale, row)
        traces.append((sw.time_s, v))
    flags += noise_done
    return {"sweeps": traces, "on": on, "off": off,
            "I_steps": i_steps, "info": info, "flags": flags}


def _period(t_thr: float, t_on: float, t_off: float) -> str:
    return "pre" if t_thr < t_on else ("step" if t_thr < t_off else "post")


def detect_all(selection: pd.DataFrame, sp: SpikeSettings, s: StepSettings, log=print,
               figures_dir: Path | None = None, formats=("png", "svg")) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Step 1: spikes and attenuated events in every sweep of every recording (+ one
    detection figure per recording).

    Returns (spikes: one row per spike with sweep, I_step_pA and period pre/step/post;
    events: one row per attenuated event (not a spike, see spikes.detect_attenuated);
    detection: one row per recording)."""
    all_spikes, all_events, summary = [], [], []
    for _, r in selection.iterrows():
        try:
            rec = load_steps(r, s)
        except Exception as exc:
            log(f"ERROR {r['abf_filename']}: {exc!r}")
            summary.append({"recording_id": r["recording_id"], "abf_filename": r["abf_filename"],
                            "status": f"error: {exc!r}"})
            continue
        per, per_ev = [], []
        for k, ((t, v), I) in enumerate(zip(rec["sweeps"], rec["I_steps"])):
            t_on, t_off = t[rec["on"]], t[rec["off"]]
            d = detect_spikes(t, v, sp)
            e = detect_attenuated(t, v, d, sp.min_peak_mV, s.attenuated_prominence_mV)
            for x, time_col in ((d, "threshold_t_s"), (e, "peak_t_s")):
                x.insert(0, "I_step_pA", I)
                x.insert(0, "sweep", k)
                x["period"] = [_period(tt, t_on, t_off) for tt in x[time_col]]
            per.append(d)
            per_ev.append(e)
        sp_rec = pd.concat(per, ignore_index=True)
        ev_rec = pd.concat(per_ev, ignore_index=True)
        n_bad = int((~sp_rec["valid"].astype(bool)).sum()) if len(sp_rec) else 0
        counts = sp_rec.groupby("period").size().to_dict() if len(sp_rec) else {}
        n_att = int((ev_rec["period"] == "step").sum()) if len(ev_rec) else 0
        flags = list(rec["flags"]) + (["depolarisation_block"] if n_att else [])
        row = {"recording_id": r["recording_id"], "abf_filename": r["abf_filename"], "status": "ok",
               "n_sweeps": len(rec["sweeps"]), "steps_pA": ", ".join(f"{i:g}" for i in rec["I_steps"]),
               "n_spikes_pre": counts.get("pre", 0), "n_spikes_step": counts.get("step", 0),
               "n_spikes_post": counts.get("post", 0), "n_flagged_spikes": n_bad,
               "n_attenuated_events_step": n_att, "flags": ";".join(flags), "figure": ""}
        if figures_dir is not None:
            paths = plot_recording(rec, sp_rec, Path(figures_dir) / r["recording_id"], formats=formats, events=ev_rec)
            row["figure"] = str(paths[0])
        summary.append(row)
        sp_rec.insert(0, "recording_id", r["recording_id"])
        ev_rec.insert(0, "recording_id", r["recording_id"])
        all_spikes.append(sp_rec)
        all_events.append(ev_rec)
        log(f"{r['abf_filename']}: steps {row['steps_pA']} pA | spikes pre {row['n_spikes_pre']}, "
            f"step {row['n_spikes_step']}, after {row['n_spikes_post']} ({n_bad} flagged) | "
            f"attenuated events during steps {n_att}")
    cols = ["recording_id", "sweep", "I_step_pA"] + SPIKE_COLUMNS + ["period"]
    spikes = pd.concat(all_spikes, ignore_index=True) if all_spikes else pd.DataFrame(columns=cols)
    ev_cols = ["recording_id", "sweep", "I_step_pA", "peak_t_s", "peak_mV", "prominence_mV", "max_dvdt_mV_ms", "period"]
    events = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame(columns=ev_cols)
    return spikes, events, pd.DataFrame(summary)


def recording_features(sweeps: pd.DataFrame, spikes: pd.DataFrame, s: StepSettings) -> tuple[dict, list[str]]:
    """Recording-level features from the per-sweep table and the spike table."""
    flags = []
    hyp = sweeps[sweeps["I_step_pA"] < -s.command_tol_pA]
    dep = sweeps[sweeps["I_step_pA"] > s.command_tol_pA].sort_values("I_step_pA")
    out = {"vm_pre_mV": float(sweeps["vm_pre_mV"].median()),
           "rate_pre_Hz": float(sweeps["rate_pre_Hz"].mean()),
           "rin_MOhm": linear_slope(hyp["I_step_pA"], hyp["delta_v_ss_mV"]) * 1e3,
           "n_sweeps_rin": len(hyp)}
    if len(hyp) < 2:
        flags.append("rin_needs_2_hyperpolarising_steps")
    if (sweeps["n_spikes_pre"] > 0).any():
        flags.append("spontaneous_before_step")
    zero = sweeps[sweeps["I_step_pA"].abs() <= s.command_tol_pA]
    if len(zero) and zero["n_spikes_step"].sum() > 0:
        flags.append("spikes_during_0pA_step")
    firing = dep[dep["n_spikes_step"] > 0]
    out["first_spiking_step_pA"] = float(firing["I_step_pA"].iloc[0]) if len(firing) else np.nan
    out["step_increment_pA"] = (float(np.median(np.diff(np.sort(sweeps["I_step_pA"].unique()))))
                                         if sweeps["I_step_pA"].nunique() > 1 else np.nan)
    if not len(firing):
        flags.append("no_spikes_at_any_positive_step")
    supra = dep[dep["I_step_pA"] >= out["first_spiking_step_pA"]] if len(firing) else dep.iloc[0:0]
    out["fi_slope_Hz_per_pA"] = linear_slope(supra["I_step_pA"], supra["rate_step_Hz"])
    out["max_rate_step_Hz"] = float(dep["rate_step_Hz"].max()) if len(dep) else np.nan
    adapt = dep[dep["n_spikes_step"] >= s.min_spikes_adaptation]
    if len(adapt):
        best = adapt.loc[adapt["n_spikes_step"].idxmax()]
        out.update(adaptation_index=float(best["adaptation_index"]), adaptation_step_pA=float(best["I_step_pA"]))
    else:
        out.update(adaptation_index=np.nan, adaptation_step_pA=np.nan)
        flags.append("too_few_spikes_for_adaptation")
    block = dep[dep["n_attenuated_step"] > 0]
    out["first_step_with_block_pA"] = float(block["I_step_pA"].iloc[0]) if len(block) else np.nan
    if len(block):
        flags.append("depolarisation_block")
    if len(hyp):
        most = hyp.loc[hyp["I_step_pA"].idxmin()]
        out.update(most_negative_step_pA=float(most["I_step_pA"]), v_min_mV=float(most["v_min_mV"]),
                   sag_ratio=float(most["sag_ratio"]), rebound_peak_mV=float(most["rebound_peak_mV"]),
                   n_rebound_spikes_short=int(most["n_rebound_spikes_short"]),
                   n_rebound_spikes_all=int(most["n_rebound_spikes_all"]),
                   latency_first_rebound_spike_ms=float(most["latency_first_rebound_spike_ms"]),
                   n_rebound_bursts=int(most["n_rebound_bursts"]))
    # AP waveform: first valid spike of the first spiking step
    ap = pd.DataFrame()
    if len(firing) and len(spikes):
        k = int(firing["sweep"].iloc[0])
        ap = spikes[(spikes["sweep"] == k) & (spikes["period"] == "step") & spikes["valid"].astype(bool)].head(1)
    for f in AP_FEATURES:
        out[f"ap_{f}"] = float(ap[f].iloc[0]) if len(ap) else np.nan
    return out, flags


def feature_table(selection: pd.DataFrame, spikes: pd.DataFrame, events: pd.DataFrame, s: StepSettings,
                  log=print) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Step 2: (per-sweep table, per-recording table), from the SAME spikes and attenuated
    events as step 1."""
    sweep_rows, rec_rows = [], []
    for _, r in selection.iterrows():
        try:
            rec = load_steps(r, s)
        except Exception as exc:
            log(f"ERROR {r['abf_filename']}: {exc!r}")
            rec_rows.append({"recording_id": r["recording_id"], "cell_tag": r.get("cell_tag"),
                             "abf_filename": r["abf_filename"], "flags": f"error: {exc!r}"})
            continue
        mine = spikes[spikes["recording_id"] == r["recording_id"]] if len(spikes) else spikes
        my_ev = events[events["recording_id"] == r["recording_id"]] if len(events) else events
        rows = []
        for k, ((t, v), I) in enumerate(zip(rec["sweeps"], rec["I_steps"])):
            d = mine[mine["sweep"] == k] if len(mine) else mine
            e = my_ev[my_ev["sweep"] == k] if len(my_ev) else my_ev
            f = sweep_features(t, v, rec["on"], rec["off"], I, d, s)
            f["n_attenuated_step"] = int((e["period"] == "step").sum()) if len(e) else 0
            rows.append({"recording_id": r["recording_id"], "abf_filename": r["abf_filename"], "sweep": k,
                         "I_total_pA": I + rec["info"]["bias_current_pA"], **f})
        sw = pd.DataFrame(rows)
        sweep_rows.append(sw)
        feats, flags = recording_features(sw, mine, s)
        n_bad = int((~mine["valid"].astype(bool)).sum()) if len(mine) else 0
        if n_bad:
            flags.append(f"flagged_spikes={n_bad}")
        rec_rows.append({**rec["info"], **feats, "flags": ";".join(rec["flags"] + flags)})
        log(f"{r['abf_filename']}: Rin {feats['rin_MOhm']:.0f} MOhm | first spiking step {feats['first_spiking_step_pA']:g} pA | "
            f"F-I {feats['fi_slope_Hz_per_pA']:.2f} Hz/pA | sag {feats.get('sag_ratio', np.nan):.2f} | "
            f"rebound spikes {feats.get('n_rebound_spikes_short', 0)} ({s.rebound_spike_window_s:g} s)"
            + (f" | block from {feats['first_step_with_block_pA']:g} pA" if np.isfinite(feats["first_step_with_block_pA"]) else ""))
    sweeps = pd.concat(sweep_rows, ignore_index=True) if sweep_rows else pd.DataFrame()
    return sweeps, pd.DataFrame(rec_rows)


# --------------------------------------------------------------------------- figure
def plot_recording(rec: dict, spikes: pd.DataFrame, out_stem: Path | None = None,
                   formats=("png", "svg"), dpi: int = 200, events: pd.DataFrame | None = None):
    """Detection figure: (A) the step period of every sweep, colour = step current, spikes
    marked; (B) whole sweeps stacked (min/max envelope), step shaded, spikes marked
    (rebound); (C) spikes aligned to threshold (black dot), grey = flagged."""
    import matplotlib
    if out_stem is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    on, off = rec["on"], rec["off"]
    I = np.asarray(rec["I_steps"])
    norm = plt.Normalize(I.min(), I.max() if I.max() > I.min() else I.min() + 1)
    cmap = plt.get_cmap("coolwarm")
    fig = plt.figure(figsize=(13, 8), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)
    ax_a, ax_b, ax_c = fig.add_subplot(gs[0, :2]), fig.add_subplot(gs[:, 2]), fig.add_subplot(gs[1, :2])
    offset = 0.0
    spread = 120.0
    for k, ((t, v), i) in enumerate(zip(rec["sweeps"], I)):
        col = cmap(norm(i))
        t_on, t_off = t[on], t[off]
        a = (t >= t_on - 0.3) & (t <= t_off + 1.0)
        ta, va = envelope(t[a] - t_on, v[a], 1500)
        ax_a.plot(ta, va, color=col, lw=0.6, label=f"{i:+g} pA")
        te, ve = envelope(t, v, 1500)
        ax_b.plot(te, ve - offset, color=col, lw=0.4)
        ax_b.text(t[-1], -offset, f" {i:+g} pA", fontsize=7, va="center")
        d = spikes[spikes["sweep"] == k] if len(spikes) else spikes
        if len(d):
            ok = d["valid"].astype(bool)
            ina = (d["peak_t_s"] >= t_on - 0.3) & (d["peak_t_s"] <= t_off + 1.0)
            ax_a.plot(d.loc[ina & ok, "peak_t_s"] - t_on, d.loc[ina & ok, "peak_mV"] + 3, "v", color=col, ms=3)
            ax_a.plot(d.loc[ina & ~ok, "peak_t_s"] - t_on, d.loc[ina & ~ok, "peak_mV"] + 3, "v", color="0.6", ms=3)
            ax_b.plot(d.loc[ok, "peak_t_s"], d.loc[ok, "peak_mV"] - offset + 3, "v", color="k", ms=2)
            ax_b.plot(d.loc[~ok, "peak_t_s"], d.loc[~ok, "peak_mV"] - offset + 3, "v", color="0.6", ms=2)
            dt = t[1] - t[0]
            for _, sp in d.head(60).iterrows():
                j = int(round((sp["threshold_t_s"] - t[0]) / dt))
                lo, hi = max(0, j - int(0.005 / dt)), min(t.size, j + int(0.015 / dt))
                ax_c.plot((t[lo:hi] - sp["threshold_t_s"]) * 1e3, v[lo:hi],
                          color=col if sp["valid"] else "0.6", lw=0.5, alpha=0.7)
                ax_c.plot(0, sp["threshold_mV"], "o", color="k", ms=2)
        e = events[events["sweep"] == k] if events is not None and len(events) else None
        if e is not None and len(e):
            ine = (e["peak_t_s"] >= t_on - 0.3) & (e["peak_t_s"] <= t_off + 1.0)
            ax_a.plot(e.loc[ine, "peak_t_s"] - t_on, e.loc[ine, "peak_mV"] + 3, "o", mfc="none", color=col, ms=4)
            ax_b.plot(e["peak_t_s"], e["peak_mV"] - offset + 3, "o", mfc="none", color="0.4", ms=3)
        offset += spread
    t0 = rec["sweeps"][0][0]
    ax_b.axvspan(t0[on], t0[off], color="0.85", zorder=0)
    ax_a.axvspan(0, t0[off] - t0[on], color="0.9", zorder=0)
    ax_a.set_xlabel("Time from step onset (s)")
    ax_a.set_ylabel("Vm (mV)")
    ax_a.set_title("Step period (shaded) and first second after it; ▼ = spike (grey = flagged); ○ = attenuated event (not a spike)", fontsize=9)
    ax_a.legend(fontsize=7, frameon=False, ncol=2)
    ax_b.set_xlabel("Time (s)")
    ax_b.set_yticks([])
    ax_b.set_title("Whole sweeps (offset by 120 mV); rebound after the step", fontsize=9)
    ax_c.set_xlabel("Time from threshold (ms)")
    ax_c.set_ylabel("Vm (mV)")
    ax_c.set_title("Spikes aligned to threshold (black dot = threshold; grey = flagged; max 60 per sweep)", fontsize=9)
    for a in (ax_a, ax_b, ax_c):
        a.spines[["top", "right"]].set_visible(False)
    n = len(spikes)
    n_bad = int((~spikes["valid"].astype(bool)).sum()) if n else 0
    fig.suptitle(f"{rec['info']['abf_filename']} | {len(I)} sweeps, steps {', '.join(f'{i:g}' for i in I)} pA | "
                 f"{n} spikes ({n_bad} flagged)" + (f" | {';'.join(rec['flags'])}" if rec["flags"] else ""), fontsize=9)
    if out_stem is None:
        return fig
    return save_figure(fig, out_stem, formats, dpi)
