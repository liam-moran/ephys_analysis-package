"""Action-potential detection and per-spike waveform features (current clamp).

Shared by every current-clamp protocol (baseline, steps, ramp, timecourse).

Method: adapted from the Allen Institute IPFX ``SpikeFeatureExtractor``
(Allen Cell Types Database, Electrophysiology technical white paper, 2017).

1. dV/dt = np.gradient(V, dt)                                            [mV/ms]
2. Candidate spike: dV/dt rises through ``dvdt_cutoff`` (Allen: 20 mV/ms).
3. Peak: maximum of V within ``max_threshold_to_peak_ms`` of the crossing (Allen: 5 ms).
   Accepted if peak >= ``min_peak_mV`` and peak - threshold >= ``min_height_mV``.
4. Upstroke: maximum dV/dt between crossing and peak.
5. Threshold: walking back from the upstroke, the point where dV/dt falls below
   ``threshold_fraction`` x upstroke, linearly interpolated (Allen: 5 %).
   Adaptation: the fraction refers to the spike's OWN upstroke, not the mean upstroke of
   the sweep, so a drug or adaptation that slows the upstroke does not shift the threshold.
6. Half-width: time between the rising and falling crossings of (threshold + peak) / 2,
   both linearly interpolated.
7. Downstroke: minimum dV/dt between peak and trough.
8. Trough (AHP): minimum V between peak and the next spike's threshold, at most
   ``ahp_window_ms`` after the peak. AHP = trough - threshold (negative = hyperpolarising).

Spike-level flags (the spike is still counted, but ``valid`` is False and its waveform
is not used for waveform summaries):
    slow_upstroke         upstroke < ``min_valid_upstroke``  (failing spikes, depolarisation block)
    broad                 half-width > ``max_valid_half_width_ms``
    threshold_not_found   dV/dt never fell below the threshold level within the search window
    no_repolarisation     V did not fall back below the half-amplitude level

Units: time s (ms where stated), voltage mV, dV/dt mV/ms.
Limitation: the example data are filtered at 2 kHz (MultiClamp Bessel) and sampled at
10 kHz, which lowers the measured maximum dV/dt and widens spikes slightly. Values are
comparable within this dataset, not with data filtered at 10 kHz.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np
import pandas as pd

SPIKE_COLUMNS = ["spike", "threshold_t_s", "threshold_mV", "peak_t_s", "peak_mV", "amplitude_mV",
                 "upstroke_mV_ms", "downstroke_mV_ms", "upstroke_downstroke_ratio", "half_width_ms",
                 "trough_t_s", "trough_mV", "ahp_mV", "isi_before_s", "flags", "valid"]


@dataclass
class SpikeSettings:
    dvdt_cutoff: float = 20.0                # mV/ms, detection
    min_peak_mV: float = -20.0               # mV, absolute peak
    min_height_mV: float = 20.0              # mV, threshold to peak
    max_threshold_to_peak_ms: float = 5.0    # ms
    threshold_fraction: float = 0.05         # of the spike's own maximum upstroke
    threshold_search_ms: float = 5.0         # ms, how far back from the upstroke
    ahp_window_ms: float = 50.0              # ms after the peak
    min_valid_upstroke: float = 50.0         # mV/ms
    max_valid_half_width_ms: float = 5.0     # ms

    @classmethod
    def from_config(cls, settings: dict | None) -> "SpikeSettings":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (settings or {}).items() if k in names})


def _interp_x(x0: float, y0: float, y1: float, level: float) -> float:
    """Fractional position between samples x0 and x0+1 where y crosses ``level``."""
    return x0 + (level - y0) / (y1 - y0) if y1 != y0 else float(x0)


def detect_spikes(t: np.ndarray, v: np.ndarray, s: SpikeSettings | None = None) -> pd.DataFrame:
    """Detect action potentials in one voltage trace and measure each waveform.

    t: time (s), evenly sampled.  v: membrane potential (mV).
    Returns one row per spike (columns ``SPIKE_COLUMNS``); empty for a silent trace.
    """
    s = s or SpikeSettings()
    t = np.asarray(t, dtype=float)
    v = np.asarray(v, dtype=float)
    if t.size < 3:
        return pd.DataFrame(columns=SPIKE_COLUMNS)
    dt_ms = float(t[1] - t[0]) * 1e3
    dv = np.gradient(v, dt_ms)
    above = dv >= s.dvdt_cutoff
    crossings = np.flatnonzero(~above[:-1] & above[1:]) + 1
    n_peak = max(1, int(round(s.max_threshold_to_peak_ms / dt_ms)))
    n_search = max(1, int(round(s.threshold_search_ms / dt_ms)))
    n_ahp = max(1, int(round(s.ahp_window_ms / dt_ms)))

    # --- pass 1: accept spikes (crossing, upstroke, peak, end of spike)
    found = []
    resume = 0
    for c in crossings:
        if c < resume:
            continue
        end = min(c + n_peak + 1, v.size)
        pk = c + int(np.argmax(v[c:end]))
        if v[pk] < s.min_peak_mV:
            continue
        up = c + int(np.argmax(dv[c:pk + 1]))
        found.append((c, up, pk))
        # the next spike can only start once V has fallen below the half-way level
        level = (v[c] + v[pk]) / 2
        below = np.flatnonzero(v[pk:] < level)
        resume = pk + (int(below[0]) if below.size else v.size)

    # --- pass 2: threshold, width, trough (bounded by neighbouring spikes)
    rows = []
    for k, (c, up, pk) in enumerate(found):
        flags = []
        prev_pk = found[k - 1][2] if k else 0
        lo = max(prev_pk, up - n_search)
        level = s.threshold_fraction * dv[up]
        j = up
        while j > lo and dv[j] > level:
            j -= 1
        if dv[j] > level:                       # never dropped below the level
            flags.append("threshold_not_found")
            x_thr = float(c)
        else:
            x_thr = _interp_x(j, dv[j], dv[j + 1], level)
        thr_v = float(np.interp(x_thr, np.arange(v.size), v))
        thr_t = float(np.interp(x_thr, np.arange(t.size), t))
        height = v[pk] - thr_v
        if height < s.min_height_mV:
            continue

        half = (thr_v + v[pk]) / 2
        rise = np.flatnonzero(v[int(np.floor(x_thr)):pk + 1] < half)
        x_rise = np.nan
        if rise.size:
            i = int(np.floor(x_thr)) + int(rise[-1])
            x_rise = _interp_x(i, v[i], v[i + 1], half)
        next_c = found[k + 1][0] if k + 1 < len(found) else v.size
        stop = min(pk + n_ahp, next_c, v.size)
        fall = np.flatnonzero(v[pk:stop] < half)
        x_fall = np.nan
        if fall.size:
            i = pk + int(fall[0])
            x_fall = _interp_x(i - 1, v[i - 1], v[i], half)
        else:
            flags.append("no_repolarisation")
        half_width = (x_fall - x_rise) * dt_ms if np.isfinite(x_rise) and np.isfinite(x_fall) else np.nan

        tr = pk + int(np.argmin(v[pk:stop])) if stop > pk else pk
        downstroke = float(np.min(dv[pk:tr + 1])) if tr > pk else np.nan

        if dv[up] < s.min_valid_upstroke:
            flags.append("slow_upstroke")
        if np.isfinite(half_width) and half_width > s.max_valid_half_width_ms:
            flags.append("broad")
        rows.append({"threshold_t_s": thr_t, "threshold_mV": thr_v, "peak_t_s": float(t[pk]),
                     "peak_mV": float(v[pk]), "amplitude_mV": float(height),
                     "upstroke_mV_ms": float(dv[up]), "downstroke_mV_ms": downstroke,
                     "upstroke_downstroke_ratio": float(dv[up] / -downstroke) if downstroke and downstroke < 0 else np.nan,
                     "half_width_ms": half_width, "trough_t_s": float(t[tr]), "trough_mV": float(v[tr]),
                     "ahp_mV": float(v[tr] - thr_v), "flags": ";".join(flags), "valid": not flags})
    df = pd.DataFrame(rows, columns=[c for c in SPIKE_COLUMNS if c not in ("spike", "isi_before_s")])
    df.insert(0, "spike", np.arange(len(df)))
    df.insert(len(df.columns) - 2, "isi_before_s", df["threshold_t_s"].diff())
    return df[SPIKE_COLUMNS]


def detect_attenuated(t: np.ndarray, v: np.ndarray, spikes: pd.DataFrame, min_peak_mV: float = -20.0,
                      min_prominence_mV: float = 20.0, min_distance_ms: float = 5.0) -> pd.DataFrame:
    """Attenuated events: voltage peaks (>= ``min_peak_mV``, prominence >= ``min_prominence_mV``)
    that are NOT detected spikes, e.g. failing, slowly rising events in depolarisation block.
    They are reported separately and never counted as spikes.

    Returns peak_t_s, peak_mV, prominence_mV, max_dvdt_mV_ms (max dV/dt in the 3 ms before the peak)."""
    from scipy.signal import find_peaks

    cols = ["peak_t_s", "peak_mV", "prominence_mV", "max_dvdt_mV_ms"]
    t = np.asarray(t, dtype=float)
    v = np.asarray(v, dtype=float)
    if t.size < 3:
        return pd.DataFrame(columns=cols)
    dt_ms = float(t[1] - t[0]) * 1e3
    pk, props = find_peaks(v, height=min_peak_mV, prominence=min_prominence_mV,
                           distance=max(1, int(round(min_distance_ms / dt_ms))))
    known = spikes["peak_t_s"].to_numpy() if len(spikes) else np.array([])
    dv = np.gradient(v, dt_ms)
    n3 = max(1, int(round(3.0 / dt_ms)))
    rows = []
    for p, prom in zip(pk, props["prominences"]):
        if known.size and np.min(np.abs(known - t[p])) <= 2e-3:
            continue
        rows.append({"peak_t_s": float(t[p]), "peak_mV": float(v[p]), "prominence_mV": float(prom),
                     "max_dvdt_mV_ms": float(dv[max(0, p - n3):p + 1].max())})
    return pd.DataFrame(rows, columns=cols)


WAVEFORM_COLUMNS = ["spike", "t_ms", "v_mV", "dvdt_mV_ms"]


def spike_waveforms(t: np.ndarray, v: np.ndarray, spike_times_s, pre_ms: float = 5.0,
                    post_ms: float = 15.0) -> pd.DataFrame:
    """Waveform of each spike, aligned at its threshold sample (long table: spike index,
    t_ms from threshold, v_mV, dV/dt in mV/ms). The window runs ``pre_ms`` before to
    ``post_ms`` after threshold on the sample grid (all spikes share the same t_ms values,
    so they can be averaged directly). Spikes whose window runs past the trace are skipped."""
    t = np.asarray(t, dtype=float)
    v = np.asarray(v, dtype=float)
    dt_ms = float(t[1] - t[0]) * 1e3
    pre, post = int(round(pre_ms / dt_ms)), int(round(post_ms / dt_ms))
    dv = np.gradient(v, dt_ms)
    rows = []
    for k, ts in enumerate(np.atleast_1d(np.asarray(spike_times_s, dtype=float))):
        i = int(round((ts - t[0]) / (dt_ms / 1e3)))
        if i - pre < 0 or i + post >= v.size:
            continue
        sl = slice(i - pre, i + post + 1)
        rows.append(pd.DataFrame({"spike": k, "t_ms": np.arange(-pre, post + 1) * dt_ms, "v_mV": v[sl],
                                  "dvdt_mV_ms": dv[sl]}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=WAVEFORM_COLUMNS)


def spike_mask(t: np.ndarray, spike_times_s, before_ms: float, after_ms: float) -> np.ndarray:
    """Boolean mask, True for samples OUTSIDE [spike - before_ms, spike + after_ms] for every spike."""
    t = np.asarray(t, dtype=float)
    keep = np.ones(t.size, dtype=bool)
    for st in np.asarray(spike_times_s, dtype=float):
        keep[(t >= st - before_ms / 1e3) & (t <= st + after_ms / 1e3)] = False
    return keep


def isi_stats(spike_times_s, min_intervals: int) -> dict:
    """Inter-spike interval statistics (s). CV = SD / mean (ddof=1); NaN with fewer than
    ``min_intervals`` intervals."""
    isi = np.diff(np.asarray(spike_times_s, dtype=float))
    out = {"n_isi": int(isi.size), "isi_mean_s": np.nan, "isi_median_s": np.nan, "isi_cv": np.nan}
    if isi.size:
        out.update(isi_mean_s=float(isi.mean()), isi_median_s=float(np.median(isi)))
    if isi.size >= max(min_intervals, 2):
        out["isi_cv"] = float(isi.std(ddof=1) / isi.mean())
    return out
