"""Membrane-test analysis (Clampex algorithm) and an optional per-recording figure."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .abf import first_adc_of_kind, open_abf, read_sweeps, readable_dacs, unit_kind


# ===================================================================== membrane_test
# Membrane-test analysis (voltage clamp): Rt, tau, Ra, Rm, Cm per pulse edge.
#
# Method: Clampex Membrane Test algorithm, pCLAMP 10 User Guide (1-2500-0180 Rev. A),
# "Membrane Test", pp. 163-166. Full derivation in 03_ephys_membrane-test/README.md.
#
# For each command edge (onset and offset of each test pulse):
#
#     I2  = mean current over the last 20 % of the segment BEFORE the edge  (baseline)
#     I1  = mean current over the last 20 % of the segment AFTER the edge   (steady state)
#     dI  = I1 - I2                                            [direct]
#     Rt  = dV / dI                                            [direct]
#     tau : single-exponential (log-linear) fit of (I - I1) from 80 % to 10 % of the
#           transient peak                                     [fitted]
#     Q1  = integral of (I - I1) over the post-edge segment    [direct]
#     Q2  = dI * tau   (charge lost while the membrane settles)  [derived]
#     Qt  = Q1 + Q2
#     Ra  = tau * dV / Qt                                      [derived, Clampex eq. 4]
#     Rm  = Rt - Ra                                            [derived, eq. 6]
#     Cm  = Qt * Rt / (dV * Rm)                                [derived, eq. 3]
#
# For an ideal Ra-(Rm||Cm) circuit these expressions are exact. An alternative
# Ra = dV / I_peak(extrapolated) is also reported (``Ra_peak_MOhm``) for comparison.
#
# Units: time s, voltage mV, current pA, charge pC.
# mV/pA = GOhm -> x1000 = MOhm;  pC/mV = nF -> x1000 = pF.


METRICS = ["Rt_MOhm", "Ra_MOhm", "Rm_MOhm", "Cm_pF", "tau_ms"]
CRITICAL_FLAGS = {"fit_failed", "too_few_fit_points", "low_fit_r_squared", "low_snr",
                  "nonpositive_Rm", "nonpositive_Qt", "zero_dI"}


@dataclass
class MTSettings:
    steady_state_fraction: float = 0.20
    fit_upper_fraction: float = 0.80
    fit_lower_fraction: float = 0.10
    edges: str = "both"
    min_fit_points: int = 5
    min_fit_r_squared: float = 0.70
    min_transient_snr: float = 20.0
    report_biexponential: bool = True
    plausible_ranges: dict | None = None
    command_dac: object = "auto"
    current_adc: object = "auto"
    ra_model: str = "fast"            # 'fast' (two-exponential, filter-aware) or 'clampex'
    min_tau_fast_ms: float = 0.1      # fast component faster than this is not resolved (2 kHz filter)
    fit_pre_ms: float = 1.0           # baseline kept before each edge in the mean transient
    filter_upsample: int = 10         # model is filtered on a grid this many times finer

    @classmethod
    def from_config(cls, settings: dict) -> "MTSettings":
        s = dict(settings or {})
        known = {k: v for k, v in s.items() if k in cls.__dataclass_fields__}
        return cls(**known)


# --------------------------------------------------------------------------- edges
def command_edges(command: np.ndarray, tol: float = 0.05) -> np.ndarray:
    """Indices of the first sample after each command change."""
    return np.where(np.abs(np.diff(command)) > tol)[0] + 1


def edge_segments(n: int, edges: np.ndarray) -> list[tuple[int, int, int]]:
    """For each edge return (prev_start, edge, next_end).

    Neighbouring segments are capped at the shortest interior segment (the pulse
    width), so baseline and steady-state windows sit next to the edge, as in Clampex.
    """
    bounds = np.r_[0, edges, n]
    interior = np.diff(edges) if edges.size > 1 else np.array([], dtype=int)
    cap = int(interior.min()) if interior.size else n
    out = []
    for k, e in enumerate(edges):
        prev_start = max(int(bounds[k]), int(e) - cap)
        next_end = min(int(bounds[k + 2]), int(e) + cap)
        out.append((prev_start, int(e), next_end))
    return out


# --------------------------------------------------------------------------- one edge
def analyse_edge(t: np.ndarray, current: np.ndarray, command: np.ndarray,
                 prev_start: int, edge: int, next_end: int, s: MTSettings) -> dict:
    """Clampex-style analysis of one capacitive transient. Never raises on bad data."""
    dt = float(t[1] - t[0])
    dV = float(np.median(command[edge:next_end]) - np.median(command[prev_start:edge]))
    n_prev, n_next = edge - prev_start, next_end - edge
    k_prev = max(2, int(round(n_prev * s.steady_state_fraction)))
    k_next = max(2, int(round(n_next * s.steady_state_fraction)))
    base = current[edge - k_prev:edge]
    I2 = float(np.mean(base))
    I1 = float(np.mean(current[next_end - k_next:next_end]))
    noise = float(np.std(base, ddof=1)) if base.size > 2 else np.nan
    dI = I1 - I2
    out: dict = {"edge_time_s": float(t[edge]), "dV_mV": dV, "I2_baseline_pA": I2,
                 "I1_steady_pA": I1, "holding_current_pA": (I1 + I2) / 2, "dI_pA": dI,
                 "noise_sd_pA": noise, "segment_before_ms": n_prev * dt * 1e3,
                 "segment_after_ms": n_next * dt * 1e3}
    flags: list[str] = []

    if dI != 0:
        out["Rt_MOhm"] = dV / dI * 1e3
    else:
        flags.append("zero_dI")

    sign = 1.0 if dV >= 0 else -1.0
    y = (current[edge:next_end] - I1) * sign          # transient, made positive
    tt = t[edge:next_end] - t[edge]                   # time from edge
    search = max(3, int(0.3 * y.size))
    pk = int(np.argmax(y[:search]))
    peak = float(y[pk])
    out["peak_measured_pA"] = peak * sign
    if noise and noise > 0:
        out["transient_snr"] = peak / noise
    else:  # noise-free (synthetic) baseline
        out["transient_snr"] = np.inf if peak > 0 else np.nan
    if np.isnan(out["transient_snr"]) or out["transient_snr"] < s.min_transient_snr:
        flags.append("low_snr")

    # Fit window: from the first point <= upper*peak to the first point <= lower*peak.
    after = y[pk:]
    i_hi = np.argmax(after <= s.fit_upper_fraction * peak) if np.any(after <= s.fit_upper_fraction * peak) else None
    i_lo = np.argmax(after <= s.fit_lower_fraction * peak) if np.any(after <= s.fit_lower_fraction * peak) else None
    tau = np.nan
    if peak <= 0 or i_hi is None or i_lo is None or i_lo - i_hi < s.min_fit_points:
        flags.append("too_few_fit_points" if peak > 0 else "fit_failed")
    else:
        a, b = pk + i_hi, pk + i_lo
        yy, tw = y[a:b], tt[a:b]
        pos = yy > 0
        if pos.sum() < s.min_fit_points:
            flags.append("too_few_fit_points")
        else:
            slope, intercept = np.polyfit(tw[pos], np.log(yy[pos]), 1)
            if slope >= 0:
                flags.append("fit_failed")
            else:
                tau = -1.0 / slope
                amp0 = float(np.exp(intercept))            # extrapolated to edge time
                model = amp0 * np.exp(-tw / tau)
                ss_res = float(np.sum((yy - model) ** 2))
                ss_tot = float(np.sum((yy - yy.mean()) ** 2))
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
                out.update(tau_ms=tau * 1e3, fit_start_ms=float(tw[0] * 1e3), fit_end_ms=float(tw[-1] * 1e3),
                           fit_n_points=int(b - a), fit_r_squared=r2, peak_extrapolated_pA=amp0 * sign)
                if not np.isfinite(r2) or r2 < s.min_fit_r_squared:
                    flags.append("low_fit_r_squared")
                if amp0 > 0:
                    out["Ra_peak_MOhm"] = abs(dV) / amp0 * 1e3

    if s.report_biexponential and peak > 0 and y.size - pk > 20:
        out.update(_biexponential_diagnostic(tt[pk:], y[pk:], tau, peak))

    if np.isfinite(tau) and dI != 0:
        Q1 =float(np.trapezoid(current[edge:next_end] - I1, tt))  # pC (pA*s)
        Q2 = dI * tau
        Qt = Q1 + Q2
        out.update(Q1_pC=Q1, Q2_pC=Q2, Qt_pC=Qt)
        if Qt * sign <= 0:
            flags.append("nonpositive_Qt")
        else:
            Ra = tau * dV / Qt * 1e3
            Rt = out["Rt_MOhm"]
            Rm = Rt - Ra
            out.update(Ra_MOhm=Ra, Rm_MOhm=Rm)
            if Rm <= 0:
                flags.append("nonpositive_Rm")
            else:
                out["Cm_pF"] = Qt * Rt / (dV * Rm) * 1e3

    for key, (lo, hi) in (s.plausible_ranges or {}).items():
        v = out.get(key)
        if v is not None and np.isfinite(v) and not (lo <= v <= hi):
            flags.append(f"outside_plausible_{key}")
    out["flags"] = ";".join(flags)
    out["valid"] = not (set(flags) & CRITICAL_FLAGS)
    return out


def _biexponential_diagnostic(t: np.ndarray, y: np.ndarray, tau_mono_s: float, peak: float) -> dict:
    """Diagnostic only: fit A1*exp(-t/tau1) + A2*exp(-t/tau2) from the peak to the
    end of the segment, and compare with a mono-exponential fit over the SAME data
    (AIC). Not used to derive Ra/Cm (that choice is a scientific decision)."""
    from scipy.optimize import curve_fit

    def mono(t, a, tau):
        return a * np.exp(-t / tau)

    def bi(t, a1, t1, a2, t2):
        return a1 * np.exp(-t / t1) + a2 * np.exp(-t / t2)

    t0 = np.nan_to_num(tau_mono_s, nan=1e-3)
    n = y.size
    res = {}
    try:
        pm, _ = curve_fit(mono, t, y, p0=[peak, t0], bounds=([0, 1e-6], [np.inf, 1.0]), maxfev=5000)
        pb, _ = curve_fit(bi, t, y, p0=[0.7 * peak, t0 / 4, 0.3 * peak, t0 * 3],
                          bounds=([0, 1e-6, 0, 1e-6], [np.inf, 1.0, np.inf, 1.0]), maxfev=20000)
    except Exception as exc:
        return {"biexp_status": f"failed: {type(exc).__name__}"}
    if pb[1] > pb[3]:                       # order: fast first
        pb = np.array([pb[2], pb[3], pb[0], pb[1]])
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    rss_m = float(np.sum((y - mono(t, *pm)) ** 2))
    rss_b = float(np.sum((y - bi(t, *pb)) ** 2))
    aic_m = n * np.log(rss_m / n) + 2 * 2
    aic_b = n * np.log(rss_b / n) + 2 * 4
    res.update(biexp_status="ok", biexp_tau_fast_ms=pb[1] * 1e3, biexp_tau_slow_ms=pb[3] * 1e3,
               biexp_fast_fraction=pb[0] / (pb[0] + pb[2]) if pb[0] + pb[2] > 0 else np.nan,
               biexp_r_squared=1 - rss_b / ss_tot if ss_tot > 0 else np.nan,
               monoexp_fullwindow_r_squared=1 - rss_m / ss_tot if ss_tot > 0 else np.nan,
               biexp_delta_aic=aic_m - aic_b)  # > 10: strong support for two components
    return res


def analyse_sweep(t, current, command, s: MTSettings, tol: float = 0.05) -> list[dict]:
    edges = command_edges(command, tol)
    rows = []
    for k, (a, e, b) in enumerate(edge_segments(len(command), edges)):
        dV = float(command[e] - command[e - 1])
        rising_from_holding = abs(command[e - 1] - command[0]) <= tol
        edge_type = "onset" if rising_from_holding else "offset"
        if s.edges == "onset" and edge_type != "onset":
            continue
        r = analyse_edge(t, current, command, a, e, b, s)
        r.update(edge_number=k, edge_type=edge_type)
        rows.append(r)
    return rows


# --------------------------------------------------------------------------- channel choice
def choose_command_dac(abf, current_adc: int, s: MTSettings) -> tuple[int | None, str]:
    """Pick the voltage DAC whose edges evoke the largest current transient (sweep 0)."""
    if s.command_dac != "auto":
        return int(s.command_dac), f"command DAC {s.command_dac} forced by configuration"
    scores = {}
    for d in readable_dacs(abf):
        if unit_kind(abf.dacUnits[d]) != "voltage":
            continue
        sw = read_sweeps(abf, current_adc, d)[0]
        edges = command_edges(sw.command)
        if edges.size == 0:
            scores[d] = 0.0
            continue
        resp = []
        for a, e, b in edge_segments(len(sw.command), edges):
            base = sw.signal[max(a, e - 50):e]
            noise = np.std(base) or 1.0
            win = sw.signal[e:min(b, e + 50)]
            resp.append(np.max(np.abs(win - np.mean(base))) / noise)
        scores[d] = float(np.median(resp))
    if not scores or max(scores.values()) <= 0:
        return None, f"no voltage DAC with command edges (scores {scores})"
    best = max(scores, key=scores.get)
    ranked = sorted(scores.values(), reverse=True)
    note = "; ".join(f"DAC{d} '{abf.dacNames[d]}' response/noise={v:.1f}" for d, v in scores.items())
    if len(ranked) > 1 and ranked[1] > 0.5 * ranked[0]:
        note += "; WARNING: second DAC response comparable"
    return best, f"auto: chose DAC{best}. {note}"


# ===================================================================== fast-component Ra
# The transients are two-exponential (soma + slower dendritic charging). A single
# exponential over 80-10 % of the peak mixes both and overestimates Ra (Clampex method),
# and extrapolating the fast component to the edge underestimates it, because the
# amplifier's 4-pole Bessel filter (telegraphed, 2 kHz here) rounds the start of the
# transient. The fit below models that filter explicitly:
#
#   I(t) - I2 = H_bessel * [ dI + A1 exp(-(t-t0)/tau1) + A2 exp(-(t-t0)/tau2) ],  t >= t0
#
# fitted to the mean sign-normalised transient of all valid edges, from 1 ms before the
# edge to the end of the segment (free: A1, tau1, A2, tau2, edge shift t0).
#   Ra_fast = dV / (A1 + A2 + dI)            (current at the edge, before filtering)
# Validation (two-compartment simulation, 2 kHz filter, noise): within about +0-19 % of the
# true Ra; a fast component below ``min_tau_fast_ms`` is not resolved -> NaN + flag.


def mean_transient(sweeps, pulses: pd.DataFrame, s: MTSettings):
    """Mean sign-normalised transient (relative to the pre-edge baseline I2) over all VALID
    edges. Returns (t_rel_s, y_pA, dV_mV, dI_pA, fs) or None."""
    ok = set(zip(pulses.loc[pulses["valid"], "sweep"], pulses.loc[pulses["valid"], "edge_number"]))
    traces, dIs, dVs = [], [], []
    fs = None
    for sw in sweeps:
        dt = float(sw.time_s[1] - sw.time_s[0])
        fs = 1.0 / dt
        npre = max(1, int(round(s.fit_pre_ms * 1e-3 * fs)))
        edges = command_edges(sw.command)
        for k, (a, e, b) in enumerate(edge_segments(len(sw.command), edges)):
            if (sw.sweep, k) not in ok or e - npre < a:
                continue
            dv = float(np.median(sw.command[e:b]) - np.median(sw.command[a:e]))
            sg = 1.0 if dv >= 0 else -1.0
            k_prev = max(2, int(round((e - a) * s.steady_state_fraction)))
            k_next = max(2, int(round((b - e) * s.steady_state_fraction)))
            I2 = float(np.mean(sw.signal[e - k_prev:e]))
            I1 = float(np.mean(sw.signal[b - k_next:b]))
            traces.append((sw.signal[e - npre:b] - I2) * sg)
            dIs.append((I1 - I2) * sg)
            dVs.append(abs(dv))
    if not traces:
        return None
    n = min(len(x) for x in traces)
    y = np.mean([x[:n] for x in traces], axis=0)
    t_rel = (np.arange(n) - int(round(s.fit_pre_ms * 1e-3 * fs))) / fs
    return t_rel, y, float(np.median(dVs)), float(np.mean(dIs)), fs


def filtered_biexp(p, t_rel, fs, filter_hz, dI, up=10):
    """Model: step of dI plus two exponentials starting at t0 = p[4], passed through a
    4-pole Bessel low-pass at ``filter_hz`` (computed on a grid ``up`` times finer)."""
    from scipy.signal import bessel, lfilter

    a1, t1, a2, t2, t0 = p
    fsu = fs * up
    tu = np.arange(t_rel[0], t_rel[-1] + 1 / fs, 1 / fsu)
    x = np.clip(tu - t0, 0, None)
    y = np.where(tu >= t0, dI + a1 * np.exp(-x / t1) + a2 * np.exp(-x / t2), 0.0)
    if filter_hz and filter_hz < fsu / 2:
        b, a = bessel(4, filter_hz / (fsu / 2), norm="mag")
        y = lfilter(b, a, y)
    return np.interp(t_rel, tu, y)


def fast_component(mean, filter_hz, s: MTSettings) -> dict:
    """Fit ``filtered_biexp`` to the mean transient and derive Ra_fast (MOhm)."""
    from scipy.optimize import least_squares

    out = {"Ra_fast_MOhm": np.nan, "tau_fast_ms": np.nan, "tau_slow_ms": np.nan, "fast_fraction": np.nan,
           "edge_shift_us": np.nan, "filter_Hz": filter_hz, "ra_fast_status": ""}
    if mean is None:
        out["ra_fast_status"] = "no_valid_edges"
        return out
    t_rel, y, dV, dI, fs = mean
    y0 = float(np.max(y))
    if y0 <= 0 or dV == 0:
        out["ra_fast_status"] = "no_transient"
        return out

    def resid(p):
        return filtered_biexp(p, t_rel, fs, filter_hz, dI, s.filter_upsample) - y

    try:
        r = least_squares(resid, [2 * y0, 3e-4, 0.3 * y0, 3e-3, 0.0],
                          bounds=([0, 1e-5, 0, 2e-4, -2e-4], [1e5, 5e-3, 1e5, 0.2, 2e-4]),
                          x_scale="jac", max_nfev=3000)
    except Exception as exc:
        out["ra_fast_status"] = f"fit_failed: {type(exc).__name__}"
        return out
    a1, t1, a2, t2, t0 = r.x
    if t1 > t2:
        a1, t1, a2, t2 = a2, t2, a1, t1
    out.update(tau_fast_ms=t1 * 1e3, tau_slow_ms=t2 * 1e3,
               fast_fraction=a1 / (a1 + a2) if a1 + a2 > 0 else np.nan, edge_shift_us=t0 * 1e6)
    if not filter_hz:
        out["ra_fast_status"] = "filter_unknown"
    elif not r.success:
        out["ra_fast_status"] = "fit_not_converged"
    elif t1 * 1e3 < s.min_tau_fast_ms:
        out["ra_fast_status"] = "fast_component_unresolved"
    else:
        out["Ra_fast_MOhm"] = dV / (a1 + a2 + dI) * 1e3
        out["ra_fast_status"] = "ok"
    return out


# --------------------------------------------------------------------------- recording
def analyse_recording(abf_path: Path | str, s: MTSettings, recording_id: str = "") -> tuple[pd.DataFrame, dict]:
    """Analyse all sweeps of one membrane-test ABF. Returns (pulse_table, recording_info)."""
    abf_path = Path(abf_path)
    info = {"recording_id": recording_id, "abf_filename": abf_path.name, "status": "ok", "note": ""}
    abf = open_abf(abf_path)
    adc = first_adc_of_kind(abf, "current") if s.current_adc == "auto" else int(s.current_adc)
    if adc is None:
        info.update(status="no_current_channel", note=f"ADC units {abf.adcUnits}")
        return pd.DataFrame(), info
    dac, note = choose_command_dac(abf, adc, s)
    info.update(current_adc=adc, current_adc_name=abf.adcNames[adc], current_adc_unit=abf.adcUnits[adc],
                command_dac=dac, command_dac_selection=note, sample_rate_Hz=float(abf.dataRate))
    if dac is None:
        info.update(status="no_command_pulse", note=note)
        return pd.DataFrame(), info
    info.update(command_dac_name=abf.dacNames[dac], command_dac_unit=abf.dacUnits[dac])
    rows = []
    sweeps = read_sweeps(abf, adc, dac)
    for sw in sweeps:
        for r in analyse_sweep(sw.time_s, sw.signal, sw.command, s):
            r.update(recording_id=recording_id, abf_filename=abf_path.name, sweep=sw.sweep)
            rows.append(r)
    df = pd.DataFrame(rows)
    info["ra_model"] = s.ra_model
    if not df.empty:
        first = ["recording_id", "abf_filename", "sweep", "edge_number", "edge_type", "valid", "flags"]
        df = df[first + [c for c in df.columns if c not in first]]
        try:
            filter_hz = float(abf._adcSection.fTelegraphFilter[adc]) if abf._adcSection.nTelegraphEnable[adc] else None
        except Exception:
            filter_hz = None
        info.update(fast_component(mean_transient(sweeps, df, s) if df["valid"].any() else None, filter_hz, s))
    return df, info


def final_values(pulses: pd.DataFrame, info: dict) -> dict:
    """The final membrane-test values of one recording.

    Clampex values (``*_clampex``) are medians over VALID pulse edges. The main Ra is
    ``Ra_fast`` (two-exponential, filter-aware) unless ``ra_model`` is 'clampex'; then
    Rm = Rt - Ra and Cm = median over valid edges of Qt * Rt / (dV * Rm), with Qt the
    Clampex charge of each edge.
    """
    n = len(pulses)
    valid = pulses[pulses["valid"]] if n else pulses
    row = {"recording_id": info.get("recording_id"), "abf_filename": info.get("abf_filename"),
           "n_pulses": n, "n_valid_pulses": len(valid)}

    def med(col):
        vals = pd.to_numeric(valid.get(col, pd.Series(dtype=float)), errors="coerce").dropna()
        return vals.median() if len(vals) else np.nan

    clampex = {m: med(m) for m in METRICS}
    row["Rt_MOhm"] = clampex["Rt_MOhm"]
    use_fast = info.get("ra_model", "fast") == "fast"
    ra = info.get("Ra_fast_MOhm", np.nan) if use_fast else clampex["Ra_MOhm"]
    row["Ra_MOhm"] = ra
    if use_fast and len(valid) and np.isfinite(ra):
        rt_e = valid["Rt_MOhm"].to_numpy(float)
        ok = rt_e > ra
        cm_e = (np.abs(valid["Qt_pC"].to_numpy(float)) * rt_e
                / (np.abs(valid["dV_mV"].to_numpy(float)) * (rt_e - ra)) * 1e3)
        row["Rm_MOhm"] = row["Rt_MOhm"] - ra
        row["Cm_pF"] = float(np.nanmedian(cm_e[ok])) if ok.any() else np.nan
    elif use_fast:
        row["Rm_MOhm"] = row["Cm_pF"] = np.nan
    else:
        row["Rm_MOhm"], row["Cm_pF"] = clampex["Rm_MOhm"], clampex["Cm_pF"]
    for k in ("tau_fast_ms", "tau_slow_ms", "fast_fraction"):
        row[k] = info.get(k, np.nan)
    row["ra_fast_status"] = info.get("ra_fast_status", "")
    names = {"Rt_MOhm": "Rt_clampex_MOhm", "Ra_MOhm": "Ra_clampex_MOhm", "Rm_MOhm": "Rm_clampex_MOhm",
             "Cm_pF": "Cm_clampex_pF", "tau_ms": "tau_clampex_ms"}
    for m in METRICS:
        if m != "Rt_MOhm":
            row[names[m]] = clampex[m]
    row["holding_current_pA"] = med("holding_current_pA")
    row["qc_status"] = ("fail" if row["n_valid_pulses"] == 0 else
                        "review" if row["n_valid_pulses"] < 0.5 * max(n, 1)
                        or (use_fast and row["ra_fast_status"] != "ok") else "pass")
    if n:
        flags = pulses["flags"].fillna("").str.split(";").explode()
        row["pulse_flags"] = "; ".join(f"{k}={v}" for k, v in flags[flags != ""].value_counts().items())
    row.update(command_dac=info.get("command_dac"), current_adc=info.get("current_adc"), filter_Hz=info.get("filter_Hz"),
               analysis_status=info.get("status"),
               method=("Ra: two-exponential with amplifier filter; Rt, Qt: Clampex" if use_fast
                       else "clampex_single_exponential"),
               clampex_validation="not_validated")
    return row


def analyse_table(selection: pd.DataFrame, s: MTSettings, log=print, figures_dir: Path | None = None) -> pd.DataFrame:
    """Final values for every recording in ``selection`` (needs recording_id, cell_tag,
    abf_filename, original_path). Errors are recorded per recording, never hidden."""
    rows = []
    for _, r in selection.iterrows():
        try:
            pulses, info = analyse_recording(r["original_path"], s, r["recording_id"])
        except Exception as exc:
            log(f"ERROR {r['abf_filename']}: {exc!r}")
            rows.append({"recording_id": r["recording_id"], "abf_filename": r["abf_filename"],
                         "analysis_status": f"error: {exc!r}", "qc_status": "fail"})
            continue
        row = final_values(pulses, info)
        row["cell_tag"] = r.get("cell_tag")
        rows.append(row)
        log(f"{r['abf_filename']}: {row['n_valid_pulses']}/{row['n_pulses']} valid | Cm {row['Cm_pF']:.1f} pF | "
            f"Ra {row['Ra_MOhm']:.1f} MOhm (Clampex {row['Ra_clampex_MOhm']:.1f}) | Rt {row['Rt_MOhm']:.0f} MOhm | "
            f"tau fast/slow {row['tau_fast_ms']:.2f}/{row['tau_slow_ms']:.2f} ms"
            + (f" | Ra fast: {row['ra_fast_status']}" if row["ra_fast_status"] not in ("ok", "") else ""))
        if figures_dir is not None and not pulses.empty:
            plot_recording(r["original_path"], pulses, info, Path(figures_dir) / r["recording_id"])
    values = pd.DataFrame(rows)
    if values.empty:
        return values
    first = ["recording_id", "cell_tag", "abf_filename"]
    return values[first + [c for c in values.columns if c not in first]]


# ===================================================================== figure (optional)


def plot_recording(abf_path: Path, pulses: pd.DataFrame, info: dict, out_stem: Path | None = None,
                   formats=("png", "svg"), dpi: int = 300):
    """Figure: (A) every transient aligned to its edge with the exponential fit,
    (B) Rt, Ra, Cm, tau per pulse (filled = valid, open = flagged).

    With ``out_stem`` the figure is saved (one file per format) and the paths are
    returned; without it the figure itself is returned (for display in a notebook).
    """
    import matplotlib
    if out_stem is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    abf = open_abf(abf_path)
    sweeps = {sw.sweep: sw for sw in read_sweeps(abf, info["current_adc"], info["command_dac"])}
    fig = plt.figure(figsize=(11, 6.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 4)
    ax_on, ax_off = fig.add_subplot(gs[0, :2]), fig.add_subplot(gs[0, 2:])
    cmap = plt.get_cmap("viridis")
    n_sw = max(len(sweeps), 1)
    for _, p in pulses.iterrows():
        sw = sweeps[int(p["sweep"])]
        ax = ax_on if p["edge_type"] == "onset" else ax_off
        e = int(np.searchsorted(sw.time_s, p["edge_time_s"]))
        dt = sw.time_s[1] - sw.time_s[0]
        a, b = max(0, e - int(0.002 / dt)), min(len(sw.time_s), e + int(0.012 / dt))
        t_ms = (sw.time_s[a:b] - sw.time_s[e]) * 1e3
        col = cmap(int(p["sweep"]) / n_sw)
        ax.plot(t_ms, sw.signal[a:b], color=col, lw=0.8, alpha=0.8 if p["valid"] else 0.3,
                label=f"sweep {int(p['sweep'])}")
        if pd.notna(p.get("tau_ms")):
            tf = np.linspace(p["fit_start_ms"], p["fit_end_ms"], 100)
            ax.plot(tf, p["I1_steady_pA"] + p["peak_extrapolated_pA"] * np.exp(-tf / p["tau_ms"]),
                    color="red", lw=1.2)
    for ax, title in ((ax_on, "Pulse onset"), (ax_off, "Pulse offset")):
        ax.set_title(f"{title} (red: exponential fit, 80-10 % of peak)", fontsize=9)
        ax.set_xlabel("Time from command edge (ms)")
        ax.set_ylabel("Current (pA)")
        ax.spines[["top", "right"]].set_visible(False)
    ax_on.legend(fontsize=7, frameon=False)

    for i, (m, lab) in enumerate((("Rt_MOhm", "Rt (MΩ)"), ("Ra_MOhm", "Ra, Clampex (MΩ)"),
                                  ("Cm_pF", "Cm (pF)"), ("tau_ms", "τ, Clampex (ms)"))):
        ax = fig.add_subplot(gs[1, i])
        for edge, marker in (("onset", "o"), ("offset", "s")):
            d = pulses[pulses["edge_type"] == edge]
            if m not in d:
                continue
            v = d[d["valid"]]
            f = d[~d["valid"]]
            ax.plot(v["sweep"], v[m], marker, color="k", ms=5, label=f"{edge} (valid)")
            ax.plot(f["sweep"], f[m], marker, mfc="none", color="tab:red", ms=5, label=f"{edge} (flagged)")
        ax.set_xlabel("Sweep")
        ax.set_ylabel(lab)
        ax.spines[["top", "right"]].set_visible(False)
        if i == 0:
            ax.legend(fontsize=6, frameon=False)
    fig.suptitle(f"{info['abf_filename']}  |  current ADC{info['current_adc']} '{info.get('current_adc_name')}', "
                 f"command DAC{info['command_dac']} '{info.get('command_dac_name')}'  |  NOT validated against Clampex",
                 fontsize=9)
    if out_stem is None:
        return fig
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        p = out_stem.with_suffix(f".{fmt}")
        fig.savefig(p, dpi=dpi)
        paths.append(p)
    plt.close(fig)
    return paths
