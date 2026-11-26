"""Noise QC (01) and optional, recorded noise corrections (02 decides, 04-07 apply in memory).

QC (``noise_metrics`` -> ``noise_flags``), per recording, on the primary signal (voltage in
current clamp, current in voltage clamp) of QUIET stretches only: the command at its holding
level, away from command changes (``guard_*_s``), and in current clamp with spikes removed
(interpolated over). Then:
    spectrum        Welch power spectral density, averaged over the quiet segments
                    (``segment_s`` long, Hann window)
    rms             SD of the quiet signal in ``rms_band_Hz`` (1-1000 Hz: drift and DC excluded).
                    In current clamp this includes real synaptic activity, so it is an upper
                    bound of the recording noise
    mains           peak-to-peak amplitude of the sinusoid at ``mains_Hz`` and each harmonic:
                    power above the local floor integrated over the peak, A = sqrt(2 P), pp = 2 A
    low frequency   peaks in ``lowf_band_Hz`` (0.05-20 Hz: perfusion / pump) at least
                    ``lowf_min_prominence_dB`` above their surroundings; amplitude as for mains.
                    Only assessed when quiet segments of ``segment_s`` exist (gap-free files)
    artefacts       brief transients: samples more than ``artefact_k`` robust SDs from a running
                    median (``artefact_window_ms``), outside spikes; counted as events per minute
Flags (severity from the thresholds in config; units mV in current clamp, pA in voltage clamp):
    mains_noise, periodic_low_frequency_noise, high_rms_noise, transient_artefacts
    (info below the review threshold), noise_not_assessed (info), noise_check_failed (review).

Corrections (``correct``; only when a decision in 02 asks for it; recorded in analysis_ready.csv):
    remove_mains    subtract fitted sinusoids at mains_Hz x 1..mains_harmonics, fitted by least
                    squares in consecutive ``mains_window_s`` windows (follows slow phase and
                    amplitude drift) on spike-free samples; spikes themselves are not filtered
    notch           narrow zero-phase IIR notch (scipy.signal.iirnotch, quality factor Q) at each
                    listed frequency, e.g. a stable pump frequency. Rings briefly around fast
                    events, so use it only where needed
Never applied by default, never written to the ABF.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

from .abf import open_abf, readable_dacs, read_sweeps, unit_kind


@dataclass
class NoiseSettings:
    mains_Hz: float = 50.0
    n_harmonics: int = 5
    segment_s: float = 10.0            # Welch segment for the spectrum (0.1 Hz resolution)
    min_segment_s: float = 1.0         # shorter quiet stretches are not used
    guard_cc_s: float = 0.3            # skip after a command change (current clamp: membrane settles)
    guard_vc_s: float = 0.02           # ... (voltage clamp: capacitive transient)
    spike_exclude_ms: tuple = (5.0, 200.0)   # around each spike (current clamp): before, after threshold
    rms_band_Hz: tuple = (1.0, 1000.0)
    lowf_band_Hz: tuple = (0.05, 20.0)
    lowf_min_prominence_dB: float = 10.0
    lowf_max_width_Hz: float = 0.5     # a pump gives a narrow line; broad bumps are physiology
    lowf_min_quiet_fraction: float = 0.9   # assess low-f only without current injection (gap-free)
    artefact_k: float = 8.0
    artefact_window_ms: float = 5.0
    artefact_max_ms: float = 1.0       # a switching transient is brief ...
    artefact_min_amp: dict = field(default_factory=lambda: {"mV": 2.0, "pA": 20.0})   # ... and large
    info: dict = field(default_factory=lambda: {"mains_pp": {"mV": 0.2, "pA": 2.0}, "lowf_pp": {"mV": 0.2, "pA": 2.0}})
    review: dict = field(default_factory=lambda: {"mains_pp": {"mV": 1.0, "pA": 10.0}, "lowf_pp": {"mV": 1.0, "pA": 10.0},
                                                  "rms": {"mV": 3.0, "pA": 20.0}, "artefacts_per_min": 6.0})

    @classmethod
    def from_config(cls, settings: dict | None) -> "NoiseSettings":
        names = {f.name for f in fields(cls)}
        kw = {k: (tuple(v) if isinstance(v, list) else v) for k, v in (settings or {}).items() if k in names}
        return cls(**kw)


# --------------------------------------------------------------------------- building blocks
def quiet_mask(command: np.ndarray, fs: float, guard_s: float, tol: float | None = None) -> np.ndarray:
    """True where the command is at its holding level (the value of the first sample, within
    ``tol``) and at least ``guard_s`` after the last change."""
    c = np.asarray(command, dtype=float)
    span = float(np.ptp(c)) if c.size else 0.0
    tol = tol if tol is not None else max(1e-6, 1e-3 * span)
    at_hold = np.abs(c - c[0]) <= tol
    g = int(round(guard_s * fs))
    if g > 0 and not at_hold.all():
        changed = np.flatnonzero(np.diff(at_hold.astype(int)) != 0) + 1
        for i in changed:
            at_hold[i:i + g] = False
    return at_hold


def spike_free(v: np.ndarray, fs: float, before_ms: float, after_ms: float, dvdt_mV_ms: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    """(v with spikes replaced by linear interpolation, mask of the replaced samples). Spikes are
    found from dV/dt > ``dvdt_mV_ms`` and removed from ``before_ms`` to ``after_ms`` around."""
    v = np.asarray(v, dtype=float)
    dv = np.gradient(v) * fs / 1e3
    hits = np.flatnonzero(dv > dvdt_mV_ms)
    mask = np.zeros(v.size, bool)
    b, a = int(before_ms * fs / 1e3), int(after_ms * fs / 1e3)
    for i in hits:
        mask[max(0, i - b):i + a] = True
    out = v.copy()
    if mask.any() and (~mask).sum() > 1:
        idx = np.arange(v.size)
        out[mask] = np.interp(idx[mask], idx[~mask], v[~mask])
    return out, mask


def segments_of(mask: np.ndarray, min_len: int) -> list[slice]:
    """Contiguous True stretches of at least ``min_len`` samples."""
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    d = np.diff(m.astype(int))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return [slice(a, b) for a, b in zip(starts, ends) if b - a >= min_len]


def welch_average(segments: list[np.ndarray], fs: float, nperseg: int) -> tuple[np.ndarray, np.ndarray]:
    """Welch PSD (density, Hann) of each segment (linear trend removed), averaged weighted by length."""
    acc, w, f = None, 0, None
    for x in segments:
        if x.size < nperseg:
            continue
        f, p = signal.welch(x, fs=fs, nperseg=nperseg, detrend="linear")
        acc = p * x.size if acc is None else acc + p * x.size
        w += x.size
    if acc is None:
        return np.array([]), np.array([])
    return f, acc / w


def line_amplitude(f: np.ndarray, p: np.ndarray, f0: float, half_bw: float | None = None) -> float:
    """Peak-to-peak amplitude of a sinusoid at ``f0`` from a density PSD: power above the local
    floor integrated over +-half_bw, A = sqrt(2 P), pp = 2 A. NaN if f0 is out of range."""
    if f.size < 3 or f0 >= f[-1] - 2:
        return np.nan
    df = float(f[1] - f[0])
    half_bw = half_bw or max(0.6, 3 * df)
    pk = (f >= f0 - half_bw) & (f <= f0 + half_bw)
    side = ((f >= f0 - half_bw - 4) & (f < f0 - half_bw - df)) | ((f > f0 + half_bw + df) & (f <= f0 + half_bw + 4))
    if not pk.any() or not side.any():
        return np.nan
    excess = float(np.sum(p[pk] - np.median(p[side])) * df)
    return 2 * float(np.sqrt(2 * excess)) if excess > 0 else 0.0


def band_rms(f: np.ndarray, p: np.ndarray, band: tuple) -> float:
    m = (f >= band[0]) & (f <= band[1])
    return float(np.sqrt(np.trapezoid(p[m], f[m]))) if m.sum() > 1 else np.nan


def lowf_peaks(f: np.ndarray, p: np.ndarray, s: NoiseSettings) -> list[dict]:
    """Peaks in the low-frequency band with their prominence (dB) and amplitude (pp)."""
    m = (f >= s.lowf_band_Hz[0]) & (f <= s.lowf_band_Hz[1])
    if m.sum() < 5:
        return []
    db = 10 * np.log10(np.maximum(p[m], 1e-30))
    df = float(f[1] - f[0])
    idx, props = signal.find_peaks(db, prominence=s.lowf_min_prominence_dB,
                                   width=(None, max(1.0, s.lowf_max_width_Hz / df)), rel_height=0.5)
    fm = f[m]
    out = [{"freq_Hz": float(fm[i]), "prominence_dB": float(pr),
            "pp": line_amplitude(f, p, float(fm[i]), half_bw=max(2 * float(f[1] - f[0]), 0.02))}
           for i, pr in zip(idx, props["prominences"])]
    return sorted(out, key=lambda d: -d["prominence_dB"])


def artefact_events(x: np.ndarray, fs: float, k: float, window_ms: float, exclude: np.ndarray | None = None,
                    min_amp: float = 0.0, max_ms: float = 1.0) -> int:
    """Brief transients: runs of samples more than k robust SDs AND ``min_amp`` from a running
    median (``window_ms``), lasting at most ``max_ms`` (longer deviations are physiology)."""
    from scipy.ndimage import median_filter

    n = max(3, int(round(window_ms * fs / 1e3)) | 1)
    x = np.asarray(x, dtype=float)
    resid = x - median_filter(x, size=n, mode="nearest")
    mad = float(np.median(np.abs(resid - np.median(resid)))) * 1.4826
    if mad <= 0:
        return 0
    hit = np.abs(resid) > max(k * mad, min_amp)
    if exclude is not None:
        hit &= ~exclude
    if not hit.any():
        return 0
    runs = segments_of(hit, 1)
    max_len = max(1, int(round(max_ms * fs / 1e3)))
    return int(sum(1 for r in runs if r.stop - r.start <= max_len))


# --------------------------------------------------------------------------- QC per recording
def noise_metrics(path, clamp_mode: str, command_dac: int | None, s: NoiseSettings) -> dict:
    """Noise measures of one recording (see the module docstring), plus the spectrum
    (keys ``f``, ``psd``) for the figure."""
    abf = open_abf(path)
    kinds = [unit_kind(u) for u in abf.adcUnits]
    want = "voltage" if clamp_mode == "current_clamp" else "current"
    if want not in kinds:
        return {"assessed": False, "reason": f"no {want} channel"}
    adc = kinds.index(want)
    unit = "mV" if want == "voltage" else "pA"
    fs = float(abf.dataRate)
    dacs = readable_dacs(abf)
    dac = command_dac if command_dac in dacs else (dacs[0] if dacs else None)
    guard = s.guard_cc_s if want == "voltage" else s.guard_vc_s
    segs, n_art, minutes, total_s = [], 0, 0.0, 0.0
    if dac is not None:
        sweeps = [(sw.signal, sw.command) for sw in read_sweeps(abf, adc, dac)]
    else:                       # no readable command: every sample counts as holding
        from .abf import standard_scale
        sc, _ = standard_scale(abf.adcUnits[adc])
        sweeps = []
        for k in range(abf.sweepCount):
            abf.setSweep(k, channel=adc)
            y = np.asarray(abf.sweepY, float) * sc
            sweeps.append((y, np.zeros(y.size)))
    for sig, cmd in sweeps:
        x = np.asarray(sig, dtype=float)
        q = quiet_mask(cmd, fs, guard)
        spk = np.zeros(x.size, bool)
        if want == "voltage":
            x, spk = spike_free(x, fs, *s.spike_exclude_ms)
        n_art += artefact_events(np.where(q, x, np.median(x[q]) if q.any() else 0.0), fs, s.artefact_k,
                                 s.artefact_window_ms, exclude=spk | ~q, min_amp=s.artefact_min_amp[unit],
                                 max_ms=s.artefact_max_ms)
        minutes += q.sum() / fs / 60
        total_s += x.size / fs
        segs += [x[sl] for sl in segments_of(q, int(s.min_segment_s * fs))]
    longest = max((x.size for x in segs), default=0)
    if longest == 0:
        return {"assessed": False, "reason": "no quiet stretch of at least "
                                             f"{s.min_segment_s:g} s (command never at holding)", "unit": unit}
    nper = int(min(s.segment_s * fs, longest))
    f, p = welch_average(segs, fs, nper)
    out = {"assessed": True, "unit": unit, "fs_Hz": fs, "quiet_s": sum(x.size for x in segs) / fs,
           "resolution_Hz": float(f[1] - f[0]), "rms": band_rms(f, p, s.rms_band_Hz),
           "artefacts_per_min": n_art / minutes if minutes > 0 else np.nan, "f": f, "psd": p}
    for k in range(1, s.n_harmonics + 1):
        out[f"mains_{int(k * s.mains_Hz)}Hz_pp"] = line_amplitude(f, p, k * s.mains_Hz)
    out["mains_pp"] = float(np.nansum([out[f"mains_{int(k * s.mains_Hz)}Hz_pp"] for k in range(1, s.n_harmonics + 1)]))
    out["quiet_fraction"] = out["quiet_s"] / total_s if total_s else np.nan
    lowf_ok = (out["resolution_Hz"] <= max(s.lowf_band_Hz[0] * 2, 0.2)
               and out["quiet_fraction"] >= s.lowf_min_quiet_fraction)
    peaks = lowf_peaks(f, p, s) if lowf_ok else []
    out.update(lowf_assessed=lowf_ok, lowf_peak_Hz=peaks[0]["freq_Hz"] if peaks else np.nan,
               lowf_peak_pp=peaks[0]["pp"] if peaks else np.nan,
               lowf_peak_prominence_dB=peaks[0]["prominence_dB"] if peaks else np.nan,
               lowf_peaks=("; ".join(f"{d['freq_Hz']:.2f} Hz ({d['pp']:.2g} {unit})" for d in peaks[:5])))
    return out


def noise_flags(m: dict, s: NoiseSettings) -> list[tuple]:
    """(flag, severity, detail, value) tuples from ``noise_metrics``."""
    if not m.get("assessed"):
        return [("noise_not_assessed", "info", m.get("reason", ""), np.nan)]
    u = m["unit"]
    out = []
    mp = m["mains_pp"]
    if np.isfinite(mp) and mp >= s.info["mains_pp"][u]:
        harm = ", ".join(f"{k}: {v:.2g}" for k, v in m.items() if k.startswith("mains_") and k.endswith("Hz_pp")
                         and np.isfinite(v) and v >= s.info["mains_pp"][u] / 4)
        out.append(("mains_noise", "review" if mp >= s.review["mains_pp"][u] else "info",
                    f"{s.mains_Hz:g} Hz + harmonics: {mp:.2g} {u} peak-to-peak in total ({harm} {u})", mp))
    lp = m.get("lowf_peak_pp", np.nan)
    if np.isfinite(lp) and lp >= s.info["lowf_pp"][u]:
        out.append(("periodic_low_frequency_noise", "review" if lp >= s.review["lowf_pp"][u] else "info",
                    f"low-frequency peaks (e.g. perfusion pump): {m['lowf_peaks']}", m["lowf_peak_Hz"]))
    if np.isfinite(m["rms"]) and m["rms"] >= s.review["rms"][u]:
        out.append(("high_rms_noise", "review",
                    f"RMS {m['rms']:.2g} {u} in {s.rms_band_Hz[0]:g}-{s.rms_band_Hz[1]:g} Hz (quiet stretches; "
                    "includes synaptic activity in current clamp)", m["rms"]))
    a = m.get("artefacts_per_min", np.nan)
    if np.isfinite(a) and a > 0:
        out.append(("transient_artefacts", "review" if a >= s.review["artefacts_per_min"] else "info",
                    f"{a:.2g} brief transients per minute (> {s.artefact_k:g} robust SD from a running median)", a))
    if not m.get("lowf_assessed"):
        out.append(("noise_not_assessed", "info", "low-frequency band not assessed (current injection or command "
                                                  "pulses in the recording, or quiet stretches too short)", np.nan))
    return out


def plot_spectrum(m: dict, title: str, s: NoiseSettings, out_stem: Path | None = None, formats=("png", "svg")):
    """Power spectrum (log-log) with the mains harmonics and low-frequency peaks marked."""
    import matplotlib
    if out_stem is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .baseline import save_figure

    fig, ax = plt.subplots(figsize=(7, 3.6), constrained_layout=True)
    f, p = m["f"], m["psd"]
    ok = f > 0
    ax.loglog(f[ok], p[ok], color="k", lw=0.6)
    for k in range(1, s.n_harmonics + 1):
        ax.axvline(k * s.mains_Hz, color="tab:red", lw=0.6, ls=":", alpha=0.7)
    if np.isfinite(m.get("lowf_peak_Hz", np.nan)):
        ax.axvline(m["lowf_peak_Hz"], color="tab:blue", lw=0.8, ls="--")
    ax.axvspan(*s.lowf_band_Hz, color="tab:blue", alpha=0.05, lw=0)
    u = m["unit"]
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(f"PSD ({u}²/Hz)")
    ax.set_title(f"{title} | quiet {m['quiet_s']:.0f} s | RMS {m['rms']:.2g} {u} ({s.rms_band_Hz[0]:g}-"
                 f"{s.rms_band_Hz[1]:g} Hz) | mains {m['mains_pp']:.2g} {u} pp | artefacts "
                 f"{m['artefacts_per_min']:.2g}/min", fontsize=7)
    ax.text(0.01, 0.02, f"red: {s.mains_Hz:g} Hz harmonics; blue band: low-frequency (pump) range"
                        + (f"; dashed: peak {m['lowf_peak_Hz']:.2f} Hz" if np.isfinite(m.get('lowf_peak_Hz', np.nan)) else ""),
            transform=ax.transAxes, fontsize=6)
    if out_stem is None:
        return fig
    return save_figure(fig, out_stem, formats)


def noise_table(recordings: pd.DataFrame, classification: pd.DataFrame, s: NoiseSettings, figures_dir: Path | None = None,
                formats=("png", "svg"), log=print) -> tuple[pd.DataFrame, list[dict]]:
    """Noise QC of every readable recording: (one row of measures per recording, flag rows
    {recording_id, abf_filename, flag, severity, detail, value})."""
    cls = classification.set_index("recording_id")
    rows, flags = [], []
    for _, r in recordings.iterrows():
        if r.get("read_status") != "ok" or r["recording_id"] not in cls.index:
            continue
        c = cls.loc[r["recording_id"]]
        dacs = [int(d) for d in str(c.get("command_dac", "")).split(",") if str(d).strip().isdigit()]
        try:
            m = noise_metrics(r["original_path"], c.get("clamp_mode", ""), dacs[0] if dacs else None, s)
            fl = noise_flags(m, s)
        except Exception as exc:   # noqa: BLE001 - recorded as a flag, never silent
            m, fl = {"assessed": False, "reason": repr(exc)}, [("noise_check_failed", "review", repr(exc), np.nan)]
        row = {"recording_id": r["recording_id"], "abf_filename": r["abf_filename"],
               "clamp_mode": c.get("clamp_mode", ""), **{k: v for k, v in m.items() if k not in ("f", "psd")},
               "figure": ""}
        if figures_dir is not None and m.get("assessed"):
            row["figure"] = str(plot_spectrum(m, r["abf_filename"], s, Path(figures_dir) / r["recording_id"], formats)[0])
        rows.append(row)
        flags += [{"recording_id": r["recording_id"], "abf_filename": r["abf_filename"], "flag": f, "severity": sev,
                   "detail": d, "value": v} for f, sev, d, v in fl]
        if m.get("assessed"):
            log(f"{r['abf_filename']}: RMS {m['rms']:.2g} {m['unit']} | mains {m['mains_pp']:.2g} {m['unit']} pp | "
                f"low-f {m['lowf_peaks'] or '-'} | artefacts {m['artefacts_per_min']:.2g}/min")
        else:
            log(f"{r['abf_filename']}: noise not assessed ({m.get('reason', '')})")
    return pd.DataFrame(rows), flags


# --------------------------------------------------------------------------- corrections
def remove_mains(v: np.ndarray, fs: float, f0: float = 50.0, n_harmonics: int = 5, window_s: float = 1.0,
                 exclude: np.ndarray | None = None) -> np.ndarray:
    """Subtract sinusoids at f0 x 1..n_harmonics, fitted by least squares (with an offset and a
    linear trend) in consecutive ``window_s`` windows on the samples not in ``exclude``."""
    v = np.asarray(v, dtype=float)
    out = v.copy()
    n = v.size
    w = max(int(round(window_s * fs)), 2)
    starts = list(range(0, n, w))
    if len(starts) > 1 and n - starts[-1] < w // 2:
        starts.pop()
    bounds = list(zip(starts, starts[1:] + [n]))
    freqs = [k * f0 for k in range(1, n_harmonics + 1) if k * f0 < fs / 2]
    for a, b in bounds:
        t = np.arange(a, b) / fs
        cols = [np.ones_like(t), t - t.mean()]
        for fk in freqs:
            cols += [np.cos(2 * np.pi * fk * t), np.sin(2 * np.pi * fk * t)]
        X = np.column_stack(cols)
        use = np.ones(b - a, bool) if exclude is None else ~np.asarray(exclude[a:b], bool)
        if use.sum() < X.shape[1] * 4:
            continue
        coef, *_ = np.linalg.lstsq(X[use], v[a:b][use], rcond=None)
        out[a:b] -= X[:, 2:] @ coef[2:]
    return out


def notch(v: np.ndarray, fs: float, freqs, Q: float = 30.0) -> np.ndarray:
    """Zero-phase narrow notch (scipy.signal.iirnotch) at each frequency in ``freqs``."""
    out = np.asarray(v, dtype=float)
    for f0 in freqs:
        if 0 < f0 < fs / 2:
            b, a = signal.iirnotch(f0, Q, fs=fs)
            out = signal.filtfilt(b, a, out)
    return out


def parse_freqs(value) -> list[float]:
    """'1.3; 2.6' -> [1.3, 2.6] (commas as decimal points allowed); '' / NaN -> []."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    return [float(x.replace(",", ".")) for x in str(value).replace("|", ";").split(";") if x.strip()]


def correct(t: np.ndarray, v: np.ndarray, row: dict | pd.Series) -> tuple[np.ndarray, list[str]]:
    """Apply the noise corrections recorded for this recording in analysis_ready.csv
    (``mains_Hz``, ``mains_harmonics``, ``mains_window_s``, ``notch_Hz``, ``notch_Q``), in memory.
    Returns (corrected v, descriptions for the flags). Nothing recorded -> v unchanged."""
    row = dict(row)
    fs = 1.0 / float(t[1] - t[0])
    done = []
    mains = row.get("mains_Hz")
    if mains is not None and pd.notna(mains) and float(mains) > 0:
        n_h = int(row.get("mains_harmonics") or 5)
        win = float(row.get("mains_window_s") or 1.0)
        _, spk = spike_free(v, fs, 2.0, 10.0)
        v = remove_mains(v, fs, float(mains), n_h, win, exclude=spk)
        done.append(f"mains_removed={float(mains):g}Hz_x{n_h}")
    freqs = parse_freqs(row.get("notch_Hz"))
    if freqs:
        q = row.get("notch_Q")
        q = float(q) if q is not None and pd.notna(q) else 30.0
        v = notch(v, fs, freqs, q)
        done.append("notch=" + "+".join(f"{f:g}" for f in freqs) + f"Hz_Q{q:g}")
    return v, done
