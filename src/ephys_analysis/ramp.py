"""Ramp protocol (A4 / M4): pyramid ramp (0 -> +I -> 0), then a negative ramp (0 -> -I)
with an abrupt return to 0 pA (rebound).

Order, as in 04 / 05: (1) ``detect_all`` finds spikes and attenuated events in every sweep
and saves one detection figure per recording; (2) ``feature_table`` computes the features
from that spike table.

Segments are read from each file's command waveform (never hard-coded):
    pre    before the pyramid                    up     rising half of the pyramid
    down   falling half of the pyramid (incl. the short plateau at the top)
    mid    between the pyramid and the negative ramp
    neg    the negative ramp                     post   after the abrupt return to 0 pA
The current at a spike is the command at its threshold time plus any bias current (02).

Per sweep:
    vm_pre_mV, rate_pre_Hz       as in steps (0.5 s before the pyramid, spikes removed)
    I_first_spike_pA             current at the first spike on the up-ramp
    n_spikes_before_ramp, rheobase_sweep_ok   spikes in the ``rheobase_quiet_s`` (0.5 s) before
                                 the up-ramp; the sweep counts for the rheobase only if 0
    v_threshold_first_mV, latency_first_spike_ms (from the start of the up-ramp)
    n_spikes_up, n_spikes_down, max_inst_rate_up_Hz
    I_last_spike_down_pA         current at the last spike on the down-ramp
    hysteresis_pA                I_first_spike - I_last_spike_down (> 0: firing continues
                                 to lower currents on the way down)
    rin_ramp_MOhm                slope of V (spikes removed) vs I over the negative ramp,
                                 after skipping ``rin_skip_ms`` (membrane lag)
    v_min_neg_mV                 minimum V on the negative ramp
    rebound_peak_mV, n_rebound_spikes_short / _all, latency_first_rebound_spike_ms,
    n_rebound_bursts             after the abrupt return to 0 pA (as in steps)
    n_attenuated_up / _down      attenuated events (spikes.detect_attenuated)
Per recording: median over sweeps (the sweeps are repeats of the same protocol within one
recording), n_sweeps, flags;
rheobase_pA = MEAN I_first_spike_pA over the quiet sweeps (``rheobase``; the cell's rheobase:
the 05 steps are too coarse), with rheobase_sd_pA, n_sweeps_rheobase, n_sweeps_rheobase_quiet;
ap_* = MEAN waveform features of the first valid up-ramp spike of each quiet sweep (``ap_spikes``:
the AP at each cell's own rheobase; the cell's AP measure), n_sweeps_ap; their waveforms
(``spikes.spike_waveforms``, ``ap_pre_ms`` / ``ap_post_ms`` around threshold) are returned for
the average-AP and phase-plane figures (ap_waveforms.csv).
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd

from .abf import open_abf, read_sweeps
from .baseline import AP_FEATURES, save_figure
from .spikes import SPIKE_COLUMNS, SpikeSettings, detect_attenuated, detect_spikes, spike_mask, spike_waveforms
from .noise import correct as correct_noise
from .steps import detect_bursts, linear_slope
from .utils import envelope

PERIODS = ["pre", "up", "down", "mid", "neg", "post"]
SWEEP_FEATURES = ["vm_pre_mV", "rate_pre_Hz", "I_first_spike_pA", "v_threshold_first_mV", "latency_first_spike_ms",
                  "n_spikes_up", "n_spikes_down", "max_inst_rate_up_Hz", "I_last_spike_down_pA", "hysteresis_pA",
                  "rin_ramp_MOhm", "v_min_neg_mV", "rebound_peak_mV", "n_rebound_spikes_short",
                  "n_rebound_spikes_all", "latency_first_rebound_spike_ms", "n_rebound_bursts",
                  "n_attenuated_up", "n_attenuated_down"]


@dataclass
class RampSettings:
    pre_window_s: float = 0.5
    rin_skip_ms: float = 50.0
    rebound_peak_window_s: float = 0.5
    rebound_spike_window_s: float = 5.0
    exclude_before_ms: float = 5.0
    exclude_after_ms: float = 50.0
    burst_max_isi_s: float = 0.1
    burst_min_spikes: int = 2
    burst_min_ibi_s: float = 0.2
    command_tol_pA: float = 1.0
    attenuated_prominence_mV: float = 20.0
    rheobase_quiet_s: float = 0.5
    ap_pre_ms: float = 5.0
    ap_post_ms: float = 15.0

    @classmethod
    def from_config(cls, settings: dict | None) -> "RampSettings":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (settings or {}).items() if k in names})


# --------------------------------------------------------------------------- segments
def ramp_segments(command: np.ndarray, tol: float = 1.0) -> dict | None:
    """Sample indices of the protocol segments from one command waveform (pA):
    up_start, up_end (first sample at the maximum), down_start (last sample at the
    maximum), down_end (back at 0), neg_start, neg_end (first sample at the minimum) and
    release (first sample back at 0 after the minimum). None if there is no positive ramp."""
    c = np.asarray(command, dtype=float)
    base = float(np.median(c[: max(1, c.size // 20)]))
    c = c - base
    top, bottom = float(c.max()), float(c.min())
    if top <= tol:
        return None
    # 'zero' must be tight: a 200 pA/s ramp rises only 0.02 pA per sample at 10 kHz
    # (the command waveform is exact, so a tiny tolerance is safe)
    z = 1e-6 * max(abs(top), abs(bottom))
    at_top = np.flatnonzero(c >= top - z)
    up_end, down_start = int(at_top[0]), int(at_top[-1])
    zero_before = np.flatnonzero(np.abs(c[:up_end]) <= z)
    up_start = int(zero_before[-1]) + 1 if zero_before.size else 0
    zero_after = np.flatnonzero(np.abs(c[down_start:]) <= z)
    down_end = down_start + int(zero_after[0]) if zero_after.size else c.size
    seg = {"up_start": up_start, "up_end": up_end, "down_start": down_start, "down_end": down_end,
           "neg_start": None, "neg_end": None, "release": None, "I_max_pA": top, "I_min_pA": bottom}
    if bottom < -tol:
        at_bottom = np.flatnonzero(c <= bottom + z)
        neg_end = int(at_bottom[0])
        zero_b = np.flatnonzero(np.abs(c[down_end:neg_end]) <= z)
        neg_start = down_end + int(zero_b[-1]) + 1 if zero_b.size else down_end
        back = np.flatnonzero(c[at_bottom[-1]:] > -z)
        release = int(at_bottom[-1]) + int(back[0]) if back.size else c.size
        seg.update(neg_start=neg_start, neg_end=neg_end, release=release)
    return seg


def _period(i: float, seg: dict) -> str:
    if i < seg["up_start"]:
        return "pre"
    if i < seg["up_end"]:
        return "up"
    if i < seg["down_end"]:
        return "down"
    if seg["neg_start"] is None or i < seg["neg_start"]:
        return "mid"
    if i < seg["release"]:
        return "neg"
    return "post"


# --------------------------------------------------------------------------- per sweep
def sweep_features(t: np.ndarray, v: np.ndarray, current: np.ndarray, seg: dict, spikes: pd.DataFrame,
                   events: pd.DataFrame, s: RampSettings) -> dict:
    """Features of one ramp sweep. ``current``: injected current (command + bias, pA)."""
    dt = float(t[1] - t[0])
    times = spikes["threshold_t_s"].to_numpy() if len(spikes) else np.array([])
    keep = spike_mask(t, times, s.exclude_before_ms, s.exclude_after_ms)
    per = spikes["period"].to_numpy() if len(spikes) else np.array([])
    t_up = t[seg["up_start"]]
    pre = times[per == "pre"]
    m = (t >= t_up - s.pre_window_s) & (t < t_up) & keep
    out = {k: np.nan for k in SWEEP_FEATURES}
    out.update(vm_pre_mV=float(np.median(v[m])) if m.any() else np.nan,
               rate_pre_Hz=pre.size / max(t_up - t[0], dt))
    n_quiet = int(((times >= t_up - s.rheobase_quiet_s) & (times < t_up)).sum())
    out.update(n_spikes_before_ramp=n_quiet, rheobase_sweep_ok=n_quiet == 0)
    up = spikes[per == "up"] if len(spikes) else spikes
    down = spikes[per == "down"] if len(spikes) else spikes
    out.update(n_spikes_up=len(up), n_spikes_down=len(down))
    idx = np.arange(t.size)
    if len(up):
        first = up.iloc[0]
        out.update(I_first_spike_pA=float(np.interp(first["threshold_t_s"], t, current)),
                   v_threshold_first_mV=float(first["threshold_mV"]),
                   latency_first_spike_ms=float((first["threshold_t_s"] - t_up) * 1e3))
        if len(up) > 1:
            out["max_inst_rate_up_Hz"] = float(1 / np.min(np.diff(up["threshold_t_s"].to_numpy())))
    if len(down):
        out["I_last_spike_down_pA"] = float(np.interp(down.iloc[-1]["threshold_t_s"], t, current))
    if np.isfinite(out["I_first_spike_pA"]) and np.isfinite(out["I_last_spike_down_pA"]):
        out["hysteresis_pA"] = out["I_first_spike_pA"] - out["I_last_spike_down_pA"]
    ev_per = events["period"].to_numpy() if len(events) else np.array([])
    out.update(n_attenuated_up=int((ev_per == "up").sum()), n_attenuated_down=int((ev_per == "down").sum()))

    if seg["neg_start"] is not None:
        a = seg["neg_start"] + int(round(s.rin_skip_ms / 1e3 / dt))
        b = seg["neg_end"]
        sel = (idx >= a) & (idx < b) & keep
        out["rin_ramp_MOhm"] = linear_slope(current[sel], v[sel]) * 1e3
        out["v_min_neg_mV"] = float(np.min(v[seg["neg_start"]:seg["release"]]))
        t_rel = t[seg["release"]] if seg["release"] < t.size else t[-1]
        post = times[per == "post"]
        rb = (t >= t_rel) & (t < t_rel + s.rebound_peak_window_s) & keep
        out["rebound_peak_mV"] = float(np.max(v[rb]) - out["vm_pre_mV"]) if rb.any() else np.nan
        out.update(n_rebound_spikes_short=int((post < t_rel + s.rebound_spike_window_s).sum()),
                   n_rebound_spikes_all=int(post.size),
                   latency_first_rebound_spike_ms=float((post[0] - t_rel) * 1e3) if post.size else np.nan,
                   n_rebound_bursts=len(detect_bursts(post, s.burst_max_isi_s, s.burst_min_spikes, s.burst_min_ibi_s)))
    return out


# --------------------------------------------------------------------------- per recording
def rheobase(sweeps: pd.DataFrame) -> dict:
    """Rheobase of one recording (pA): the MEAN first-spike current on the up-ramp over the
    sweeps with no spike in the ``rheobase_quiet_s`` before the ramp (``rheobase_sweep_ok``).
    A sweep in which the cell was already firing is left out: its first ramp spike can be
    spontaneous rather than evoked. Quiet sweeps without an up-ramp spike are left out of
    the mean (counted in ``n_sweeps_rheobase_quiet`` but not in ``n_sweeps_rheobase``)."""
    ok = sweeps["rheobase_sweep_ok"].astype(bool) if len(sweeps) else pd.Series(dtype=bool)
    vals = pd.to_numeric(sweeps.loc[ok, "I_first_spike_pA"], errors="coerce").dropna() if len(sweeps) else pd.Series(dtype=float)
    return {"rheobase_pA": float(vals.mean()) if len(vals) else np.nan,
            "rheobase_sd_pA": float(vals.std(ddof=1)) if len(vals) > 1 else np.nan,
            "n_sweeps_rheobase": int(len(vals)), "n_sweeps_rheobase_quiet": int(ok.sum())}


def ap_spikes(spikes: pd.DataFrame, sweeps: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """(spikes used for the AP waveform, flag). The first VALID up-ramp spike of every quiet
    sweep (``rheobase_sweep_ok``, the sweeps of the rheobase), so the AP is measured at each
    cell's own threshold current. If no quiet sweep has one: the first valid up-ramp spike of
    any sweep (flag ``ap_from_non_quiet_sweep``); none at all: empty (``no_valid_ramp_spike``)."""
    if len(spikes) == 0:
        return spikes, "no_valid_ramp_spike"
    up = spikes[(spikes["period"] == "up") & spikes["valid"].astype(bool)]
    firsts = up.sort_values(["sweep", "threshold_t_s"]).groupby("sweep").head(1)
    if len(firsts) == 0:
        return firsts, "no_valid_ramp_spike"
    quiet = set(sweeps.loc[sweeps["rheobase_sweep_ok"].astype(bool), "sweep"]) if len(sweeps) else set()
    use = firsts[firsts["sweep"].isin(quiet)]
    if len(use) == 0:
        return firsts.head(1), "ap_from_non_quiet_sweep"
    return use, ""


def load_ramp(row: pd.Series | dict, s: RampSettings) -> dict:
    """Read one ready ramp recording: voltage (02 corrections applied), injected current
    (command + bias) and the segments of every sweep."""
    row = dict(row)
    src = str(row.get("current_source") or "")
    if not (src.startswith("command_DAC") or src.startswith("measured_ADC")):
        raise ValueError(f"no current information (current_source = '{src}')")
    abf = open_abf(row["original_path"])
    adc = int(str(row.get("voltage_source") or "ADC0").replace("ADC", ""))
    dac = int(src.replace("command_DAC", "")) if src.startswith("command_DAC") else (1 if len(abf.dacUnits) > 1 else 0)
    sweeps = read_sweeps(abf, adc, dac)
    scale = row.get("voltage_scale")
    scale = float(scale) if scale is not None and pd.notna(scale) else 1.0
    bias = row.get("bias_current_pA")
    bias = float(bias) if bias is not None and pd.notna(bias) else 0.0
    segs = [ramp_segments(sw.command, s.command_tol_pA) for sw in sweeps]
    if all(x is None for x in segs):
        raise ValueError("no positive ramp found in the command waveform")
    flags = (["bias_current"] if bias else []) + (["current_from_command"] if src.startswith("command_DAC") else [])
    if any(x is not None and x["neg_start"] is None for x in segs):
        flags.append("no_negative_ramp")
    s0 = next(x for x in segs if x is not None)
    t0 = sweeps[0].time_s
    info = {"recording_id": row["recording_id"], "cell_tag": row.get("cell_tag"), "abf_filename": row["abf_filename"],
            "n_sweeps": len(sweeps), "sample_rate_Hz": float(abf.dataRate), "bias_current_pA": bias,
            "current_source": src, "ramp_max_pA": s0["I_max_pA"], "ramp_min_pA": s0["I_min_pA"],
            "up_ramp_s": float(t0[s0["up_end"]] - t0[s0["up_start"]]),
            "up_ramp_rate_pA_s": s0["I_max_pA"] / max(float(t0[s0["up_end"]] - t0[s0["up_start"]]), 1e-9)}
    traces, noise_done = [], []
    for sw in sweeps:                              # noise correction only if 02 recorded one
        v, noise_done = correct_noise(sw.time_s, sw.signal * scale, row)
        traces.append((sw.time_s, v, sw.command + bias))
    return {"sweeps": traces, "segments": segs, "info": info, "flags": flags + noise_done}


def detect_all(selection: pd.DataFrame, sp: SpikeSettings, s: RampSettings, log=print,
               figures_dir: Path | None = None, formats=("png", "svg")) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Step 1: spikes and attenuated events in every sweep (+ one detection figure per
    recording). Returns (spikes, events, detection) with a ``period`` column (see PERIODS)."""
    all_sp, all_ev, summary = [], [], []
    for _, r in selection.iterrows():
        try:
            rec = load_ramp(r, s)
        except Exception as exc:
            log(f"ERROR {r['abf_filename']}: {exc!r}")
            summary.append({"recording_id": r["recording_id"], "abf_filename": r["abf_filename"],
                            "status": f"error: {exc!r}"})
            continue
        per_sp, per_ev = [], []
        for k, ((t, v, cur), seg) in enumerate(zip(rec["sweeps"], rec["segments"])):
            if seg is None:
                continue
            d = detect_spikes(t, v, sp)
            e = detect_attenuated(t, v, d, sp.min_peak_mV, s.attenuated_prominence_mV)
            dt = t[1] - t[0]
            for x, col in ((d, "threshold_t_s"), (e, "peak_t_s")):
                x.insert(0, "sweep", k)
                x["I_pA"] = np.interp(x[col], t, cur) if len(x) else []
                x["period"] = [_period((tt - t[0]) / dt, seg) for tt in x[col]]
            per_sp.append(d)
            per_ev.append(e)
        sp_rec = pd.concat(per_sp, ignore_index=True)
        ev_rec = pd.concat(per_ev, ignore_index=True)
        n_bad = int((~sp_rec["valid"].astype(bool)).sum()) if len(sp_rec) else 0
        counts = sp_rec["period"].value_counts().to_dict() if len(sp_rec) else {}
        n_att = int(ev_rec["period"].isin(["up", "down"]).sum()) if len(ev_rec) else 0
        flags = list(rec["flags"]) + (["depolarisation_block"] if n_att else [])
        row = {"recording_id": r["recording_id"], "abf_filename": r["abf_filename"], "status": "ok",
               "n_sweeps": rec["info"]["n_sweeps"], "ramp_pA": f"0 -> {rec['info']['ramp_max_pA']:g} -> 0, "
                                                               f"0 -> {rec['info']['ramp_min_pA']:g}",
               **{f"n_spikes_{p}": counts.get(p, 0) for p in PERIODS},
               "n_flagged_spikes": n_bad, "n_attenuated_events_ramp": n_att, "flags": ";".join(flags), "figure": ""}
        if figures_dir is not None:
            row["figure"] = str(plot_recording(rec, sp_rec, ev_rec, Path(figures_dir) / r["recording_id"],
                                               formats=formats)[0])
        summary.append(row)
        for x in (sp_rec, ev_rec):
            x.insert(0, "recording_id", r["recording_id"])
        all_sp.append(sp_rec)
        all_ev.append(ev_rec)
        log(f"{r['abf_filename']}: {row['ramp_pA']} pA | spikes " +
            ", ".join(f"{p} {counts.get(p, 0)}" for p in PERIODS) + f" ({n_bad} flagged) | attenuated {n_att}")
    spikes = (pd.concat(all_sp, ignore_index=True) if all_sp
              else pd.DataFrame(columns=["recording_id", "sweep"] + SPIKE_COLUMNS + ["I_pA", "period"]))
    events = (pd.concat(all_ev, ignore_index=True) if all_ev
              else pd.DataFrame(columns=["recording_id", "sweep", "peak_t_s", "peak_mV", "prominence_mV",
                                         "max_dvdt_mV_ms", "I_pA", "period"]))
    return spikes, events, pd.DataFrame(summary)


def feature_table(selection: pd.DataFrame, spikes: pd.DataFrame, events: pd.DataFrame, s: RampSettings,
                  log=print) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Step 2: (per-sweep table, per-recording table: median over sweeps; rheobase and AP as
    the mean over the quiet sweeps; AP waveforms of the spikes used), from the SAME spikes
    and events as step 1."""
    sweep_rows, rec_rows, wave_rows = [], [], []
    for _, r in selection.iterrows():
        try:
            rec = load_ramp(r, s)
        except Exception as exc:
            log(f"ERROR {r['abf_filename']}: {exc!r}")
            rec_rows.append({"recording_id": r["recording_id"], "cell_tag": r.get("cell_tag"),
                             "abf_filename": r["abf_filename"], "flags": f"error: {exc!r}"})
            continue
        mine = spikes[spikes["recording_id"] == r["recording_id"]] if len(spikes) else spikes
        my_ev = events[events["recording_id"] == r["recording_id"]] if len(events) else events
        rows = []
        for k, ((t, v, cur), seg) in enumerate(zip(rec["sweeps"], rec["segments"])):
            if seg is None:
                continue
            d = mine[mine["sweep"] == k] if len(mine) else mine
            e = my_ev[my_ev["sweep"] == k] if len(my_ev) else my_ev
            rows.append({"recording_id": r["recording_id"], "abf_filename": r["abf_filename"], "sweep": k,
                         **sweep_features(t, v, cur, seg, d, e, s)})
        sw = pd.DataFrame(rows)
        sweep_rows.append(sw)
        feats = {f: float(pd.to_numeric(sw[f], errors="coerce").median()) for f in SWEEP_FEATURES}
        feats.update(rheobase(sw))
        flags = list(rec["flags"])
        if feats["n_sweeps_rheobase_quiet"] == 0:
            flags.append("no_quiet_sweep_for_rheobase")
        elif feats["n_sweeps_rheobase"] < feats["n_sweeps_rheobase_quiet"]:
            flags.append("quiet_sweep_without_spike")
        if (sw["n_spikes_up"] == 0).all():
            flags.append("no_spikes_on_up_ramp")
        if (sw["n_attenuated_up"] + sw["n_attenuated_down"] > 0).any():
            flags.append("depolarisation_block")
        if sw["rate_pre_Hz"].gt(0).any():
            flags.append("spontaneous_before_ramp")
        n_bad = int((~mine["valid"].astype(bool)).sum()) if len(mine) else 0
        if n_bad:
            flags.append(f"flagged_spikes={n_bad}")
        use, ap_flag = ap_spikes(mine, sw)
        if ap_flag:
            flags.append(ap_flag)
        for f in AP_FEATURES:
            feats[f"ap_{f}"] = float(pd.to_numeric(use[f], errors="coerce").mean()) if len(use) else np.nan
        feats["n_sweeps_ap"] = int(len(use))
        for _, sp_ in use.iterrows():
            t, v, _ = rec["sweeps"][int(sp_["sweep"])]
            w = spike_waveforms(t, v, [sp_["threshold_t_s"]], s.ap_pre_ms, s.ap_post_ms)
            if len(w):
                w = w.drop(columns="spike")
                w.insert(0, "sweep", int(sp_["sweep"]))
                w.insert(0, "recording_id", r["recording_id"])
                wave_rows.append(w)
        rec_rows.append({**rec["info"], "n_sweeps_analysed": len(sw), **feats, "flags": ";".join(flags)})
        log(f"{r['abf_filename']}: rheobase {feats['rheobase_pA']:.1f} pA ({feats['n_sweeps_rheobase']} of "
            f"{len(sw)} sweeps) | up/down spikes "
            f"{feats['n_spikes_up']:g}/{feats['n_spikes_down']:g} | hysteresis {feats['hysteresis_pA']:.0f} pA | "
            f"Rin {feats['rin_ramp_MOhm']:.0f} MOhm | rebound spikes {feats['n_rebound_spikes_short']:g} "
            f"({s.rebound_spike_window_s:g} s)")
    sweeps = pd.concat(sweep_rows, ignore_index=True) if sweep_rows else pd.DataFrame()
    waves = (pd.concat(wave_rows, ignore_index=True) if wave_rows
             else pd.DataFrame(columns=["recording_id", "sweep", "t_ms", "v_mV", "dvdt_mV_ms"]))
    return sweeps, pd.DataFrame(rec_rows), waves


# --------------------------------------------------------------------------- figure
def plot_recording(rec: dict, spikes: pd.DataFrame, events: pd.DataFrame, out_stem: Path | None = None,
                   formats=("png", "svg"), dpi: int = 200):
    """Detection figure: (A) the ramp part of every sweep with spikes (▼, grey = flagged)
    and attenuated events (○), command below; (B) whole sweeps stacked (rebound);
    (C) spikes aligned to threshold, coloured by period."""
    import matplotlib
    if out_stem is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {"pre": "0.5", "up": "tab:red", "down": "tab:orange", "mid": "0.5", "neg": "tab:blue", "post": "tab:purple"}
    fig = plt.figure(figsize=(13, 9), constrained_layout=True)
    gs = fig.add_gridspec(3, 3, height_ratios=[3, 1, 3])
    ax_a = fig.add_subplot(gs[0, :2])
    ax_i = fig.add_subplot(gs[1, :2], sharex=ax_a)
    ax_b = fig.add_subplot(gs[:, 2])
    ax_c = fig.add_subplot(gs[2, :2])
    seg0 = next(x for x in rec["segments"] if x is not None)
    t_first = rec["sweeps"][0][0]
    t_start = t_first[seg0["up_start"]] - 0.5
    t_stop = t_first[seg0["release"]] + 2.0 if seg0["release"] is not None else t_first[seg0["down_end"]] + 2.0
    offset = 0.0
    cmap = plt.get_cmap("viridis")
    n_sw = max(len(rec["sweeps"]), 1)
    for k, ((t, v, cur), seg) in enumerate(zip(rec["sweeps"], rec["segments"])):
        col = cmap(k / n_sw)
        a = (t >= t_start) & (t <= t_stop)
        ta, va = envelope(t[a], v[a], 2000)
        ax_a.plot(ta, va - offset, color=col, lw=0.6, label=f"sweep {k}")
        if k == 0:
            ti, ci = envelope(t[a], cur[a], 1000)
            ax_i.plot(ti, ci, color="k", lw=0.8)
        te, ve = envelope(t, v, 1500)
        ax_b.plot(te, ve - k * 120.0, color=col, lw=0.4)
        d = spikes[spikes["sweep"] == k] if len(spikes) else spikes
        if len(d):
            ok = d["valid"].astype(bool)
            ina = (d["peak_t_s"] >= t_start) & (d["peak_t_s"] <= t_stop)
            ax_a.plot(d.loc[ina & ok, "peak_t_s"], d.loc[ina & ok, "peak_mV"] - offset + 3, "v", color=col, ms=3)
            ax_a.plot(d.loc[ina & ~ok, "peak_t_s"], d.loc[ina & ~ok, "peak_mV"] - offset + 3, "v", color="0.6", ms=3)
            ax_b.plot(d.loc[ok, "peak_t_s"], d.loc[ok, "peak_mV"] - k * 120.0 + 3, "v", color="k", ms=2)
            dt = t[1] - t[0]
            for _, sp in d.head(80).iterrows():
                j = int(round((sp["threshold_t_s"] - t[0]) / dt))
                lo, hi = max(0, j - int(0.005 / dt)), min(t.size, j + int(0.015 / dt))
                ax_c.plot((t[lo:hi] - sp["threshold_t_s"]) * 1e3, v[lo:hi],
                          color=colours.get(sp["period"], "0.5") if sp["valid"] else "0.75", lw=0.5, alpha=0.7)
                ax_c.plot(0, sp["threshold_mV"], "o", color="k", ms=2)
        e = events[events["sweep"] == k] if len(events) else events
        if len(e):
            ine = (e["peak_t_s"] >= t_start) & (e["peak_t_s"] <= t_stop)
            ax_a.plot(e.loc[ine, "peak_t_s"], e.loc[ine, "peak_mV"] - offset + 3, "o", mfc="none", color=col, ms=4)
    for name, a_, b_ in (("up", "up_start", "up_end"), ("down", "down_start", "down_end"), ("neg", "neg_start", "release")):
        if seg0[a_] is not None:
            for ax in (ax_a, ax_i):
                ax.axvspan(t_first[seg0[a_]], t_first[min(seg0[b_], t_first.size - 1)],
                           color=colours[name], alpha=0.07, zorder=0)
    ax_b.axvspan(t_first[seg0["up_start"]], t_first[min(seg0["release"] or seg0["down_end"], t_first.size - 1)],
                 color="0.85", zorder=0)
    ax_a.set_ylabel("Vm (mV)")
    ax_a.set_title("Ramps, all sweeps overlaid (shaded: up / down / negative ramp); ▼ spike (grey = flagged); "
                   "○ attenuated event", fontsize=9)
    ax_a.legend(fontsize=7, frameon=False)
    ax_i.set_ylabel("I (pA)")
    ax_i.set_xlabel("Time (s)")
    ax_b.set_title("Whole sweeps (offset 120 mV): rebound after release", fontsize=9)
    ax_b.set_xlabel("Time (s)")
    ax_b.set_yticks([])
    ax_c.set_xlabel("Time from threshold (ms)")
    ax_c.set_ylabel("Vm (mV)")
    ax_c.set_title("Spikes aligned to threshold (black dot); red = up-ramp, orange = down-ramp, purple = after "
                   "release, grey = other / flagged", fontsize=9)
    for ax in (ax_a, ax_i, ax_b, ax_c):
        ax.spines[["top", "right"]].set_visible(False)
    n = len(spikes)
    n_bad = int((~spikes["valid"].astype(bool)).sum()) if n else 0
    i = rec["info"]
    fig.suptitle(f"{i['abf_filename']} | {i['n_sweeps']} sweeps | 0 → {i['ramp_max_pA']:g} → 0 pA, 0 → "
                 f"{i['ramp_min_pA']:g} pA | {n} spikes ({n_bad} flagged)"
                 + (f" | {';'.join(rec['flags'])}" if rec["flags"] else ""), fontsize=9)
    if out_stem is None:
        return fig
    return save_figure(fig, out_stem, formats, dpi)
