"""Generate docs/parameter_guide.html: every extracted parameter, a schematic, and what it means.

Schematics are drawn from synthetic traces (not real data) as small inline SVGs. The text is
written by hand below (SECTIONS): update it when a definition in the package changes.
Usage: python docs/make_param_guide.py
"""
import html
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator

DOCS = Path(__file__).resolve().parent
W, H = 240, 120


def f1(x):
    return f"{x:.1f}".rstrip("0").rstrip(".")


class Box:
    def __init__(self, xr, yr, px=(12, 232), py=(10, 108)):
        (self.x0, self.x1), (self.y0, self.y1), self.px, self.py = xr, yr, px, py

    def X(self, x):
        return self.px[0] + (x - self.x0) / (self.x1 - self.x0) * (self.px[1] - self.px[0])

    def Y(self, y):
        return self.py[1] - (y - self.y0) / (self.y1 - self.y0) * (self.py[1] - self.py[0])


class Svg:
    def __init__(self):
        self.el = []

    def path(self, b, xs, ys, cls="tr"):
        pts = [f"{f1(b.X(x))} {f1(b.Y(y))}" for x, y in zip(xs, ys)]
        self.el.append(f'<path class="{cls}" d="M{" L".join(pts)}"/>')

    def line(self, b, x0, y0, x1, y1, cls="hl", arrow=None):
        m = ""
        if arrow in ("end", "both"):
            m += ' marker-end="url(#ah)"'
        if arrow == "both":
            m += ' marker-start="url(#ah)"'
        self.el.append(f'<line class="{cls}" x1="{f1(b.X(x0))}" y1="{f1(b.Y(y0))}" '
                       f'x2="{f1(b.X(x1))}" y2="{f1(b.Y(y1))}"{m}/>')

    def rect(self, b, x0, x1, y0, y1, cls="sh"):
        xa, xb = sorted((b.X(x0), b.X(x1)))
        ya, yb = sorted((b.Y(y0), b.Y(y1)))
        self.el.append(f'<rect class="{cls}" x="{f1(xa)}" y="{f1(ya)}" width="{f1(max(xb - xa, 1.5))}" '
                       f'height="{f1(max(yb - ya, 1))}"/>')

    def prect(self, x, y, w, h, cls="sh", extra=""):
        self.el.append(f'<rect class="{cls}" x="{f1(x)}" y="{f1(y)}" width="{f1(w)}" height="{f1(h)}"{extra}/>')

    def dot(self, b, x, y, cls="dt", r=2.6):
        self.el.append(f'<circle class="{cls}" cx="{f1(b.X(x))}" cy="{f1(b.Y(y))}" r="{r}"/>')

    def text(self, b, x, y, s, cls="lh", anchor="start", dx=0, dy=0):
        self.ptext(b.X(x) + dx, b.Y(y) + dy, s, cls, anchor)

    def ptext(self, x, y, s, cls="lb", anchor="start"):
        self.el.append(f'<text class="{cls}" x="{f1(x)}" y="{f1(y)}" text-anchor="{anchor}">{html.escape(s)}</text>')

    def svg(self, title):
        return (f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(title)}">'
                + "".join(self.el) + "</svg>")


def spiky(t, v, spikes, peaks=None, dt_frac=1 / 500):
    """Insert narrow spikes (threshold -> peak -> trough) into a smooth trace."""
    e = (t[-1] - t[0]) * dt_frac
    xs, ys = list(t), list(v)
    keep = np.ones(len(t), bool)
    add = []
    for k, ts in enumerate(spikes):
        pk = 20.0 if peaks is None else peaks[k]
        v0 = float(np.interp(ts, t, v))
        keep &= ~((t > ts) & (t < ts + 2 * e))
        add += [(ts, v0), (ts + e, pk), (ts + 2 * e, v0 - 6)]
    pts = [(x, y) for x, y, k in zip(xs, ys, keep) if k] + add
    pts.sort()
    return np.array([p[0] for p in pts]), np.array([p[1] for p in pts])


def smooth_noise(n, sd, seed):
    rng = np.random.default_rng(seed)
    return np.convolve(rng.normal(0, 1, n), np.ones(9) / 9, "same") * sd * 3


# ------------------------------------------------------------------ action potential
AP_T = np.linspace(0, 12, 400)
AP_V = PchipInterpolator([0, 2, 3, 3.25, 3.6, 4.0, 4.8, 5.6, 6.6, 9, 12],
                         [-56, -52, -45, -15, 25, 12, -30, -55, -62, -58.5, -56])(AP_T)
_dv = np.gradient(AP_V, AP_T)
_up = int(np.argmax(_dv))
_i = _up
while _i > 0 and _dv[_i - 1] >= 0.05 * _dv[_up]:
    _i -= 1
AP = dict(thr_t=AP_T[_i], thr=AP_V[_i], pk=int(np.argmax(AP_V)), up=_up, dn=int(np.argmin(_dv)))
AP["pk_t"], AP["pk_v"] = AP_T[AP["pk"]], AP_V[AP["pk"]]
_tr = AP["pk"] + int(np.argmin(AP_V[AP["pk"]:]))
AP["tr_t"], AP["tr_v"] = AP_T[_tr], AP_V[_tr]
AP["half"] = (AP["thr"] + AP["pk_v"]) / 2
AP["x_rise"] = np.interp(AP["half"], AP_V[_i:AP["pk"] + 1], AP_T[_i:AP["pk"] + 1])
AP["x_fall"] = np.interp(AP["half"], AP_V[AP["pk"]:_tr][::-1], AP_T[AP["pk"]:_tr][::-1])


def fig_ap(hl):
    s, b = Svg(), Box((0, 12), (-72, 36))
    s.path(b, AP_T, AP_V)
    thr, pk_v = AP["thr"], AP["pk_v"]

    def thr_line():
        s.line(b, 0.3, thr, 11.7, thr, "gd")
        s.dot(b, AP["thr_t"], thr)

    def tangent(i, half_ms, lab, dx):
        sl = _dv[i]
        s.line(b, AP_T[i] - half_ms, AP_V[i] - half_ms * sl, AP_T[i] + half_ms, AP_V[i] + half_ms * sl, "hl")
        s.text(b, AP_T[i], AP_V[i], lab, dx=dx, dy=3)

    if hl == "threshold":
        thr_line()
        s.text(b, 0.4, thr, "threshold: dV/dt = 5 % of max", dy=-4)
    elif hl == "peak":
        s.dot(b, AP["pk_t"], pk_v)
        s.text(b, AP["pk_t"], pk_v, "peak", dx=6, dy=4)
    elif hl == "amplitude":
        thr_line()
        s.line(b, 1.5, pk_v, 11.7, pk_v, "gd")
        s.line(b, 1.8, thr, 1.8, pk_v, "hl", "both")
        s.text(b, 1.8, (thr + pk_v) / 2, "peak − threshold", dx=-4, anchor="end") if False else \
            s.text(b, 5.3, (thr + pk_v) / 2 + 10, "amplitude = peak − threshold")
    elif hl == "half_width":
        thr_line()
        s.line(b, AP["x_rise"], AP["half"], AP["x_fall"], AP["half"], "hl", "both")
        s.text(b, AP["x_fall"], AP["half"], "width at ½ amplitude", dx=6, dy=3)
    elif hl == "upstroke":
        tangent(AP["up"], 0.35, "max rise (mV/ms)", 8)
    elif hl == "downstroke":
        tangent(AP["dn"], 0.6, "max fall (mV/ms)", 10)
    elif hl == "ratio":
        tangent(AP["up"], 0.35, "rise", 8)
        tangent(AP["dn"], 0.6, "fall", 10)
        s.text(b, 6.5, 28, "ratio = rise / |fall|")
    elif hl == "ahp":
        thr_line()
        s.dot(b, AP["tr_t"], AP["tr_v"])
        s.line(b, AP["tr_t"], thr, AP["tr_t"], AP["tr_v"], "hl", "both")
        s.text(b, AP["tr_t"], AP["tr_v"], "AHP = trough − threshold", dx=-6, dy=13, anchor="start") if False else \
            s.text(b, AP["tr_t"] + 0.3, (thr + AP["tr_v"]) / 2 + 12, "AHP = trough − thr.")
        s.ptext(b.X(AP["tr_t"]) - 14, b.Y(AP["tr_v"]) + 10, "trough (≤ 50 ms)", "lb")
    s.ptext(4, 116, "schematic, ~12 ms", "lb")
    return s


# ------------------------------------------------------------------ spontaneous trace
SP_SPK = [0.8, 1.9, 2.7, 4.1, 5.0, 6.6, 7.3, 8.9]


def spont(t0=0, t1=10, spikes=SP_SPK, seed=1, vrest=-50.0, n=500):
    t = np.linspace(t0, t1, n)
    v = vrest + smooth_noise(n, 0.6, seed)
    for ts in spikes:
        m = t > ts
        v[m] -= 7 * np.exp(-(t[m] - ts) / (0.012 * (t1 - t0)))
    return t, v


def fig_spont(hl):
    s = Svg()
    if hl == "class":
        b1, b2 = Box((0, 10), (-62, 25), py=(8, 52)), Box((0, 10), (-62, 25), py=(64, 108))
        t, v = spont(spikes=[3.2], seed=3, vrest=-56)
        s.path(b1, *spiky(t, v, [3.2]))
        s.ptext(232, 18, "silent: ≤ 10 spikes / 100 s", "lh", "end")
        t, v = spont()
        s.path(b2, *spiky(t, v, SP_SPK))
        s.ptext(232, 72, "tonic: > 10 spikes / 100 s", "lh", "end")
        return s
    if hl == "window":
        b = Box((0, 120), (-65, 25), py=(10, 96))
        spk = list(np.arange(3, 120, 4.1) + np.sin(np.arange(29)) * 1.2)
        t, v = spont(0, 120, spk, n=700)
        s.rect(b, 20, 120, -65, 25)
        s.path(b, *spiky(t, v, spk, dt_frac=1 / 1400))
        s.line(b, 20, -62, 120, -62, "hl", "both")
        s.text(b, 70, -62, "last 100 s analysed", anchor="middle", dy=12)
        s.text(b, 0, -62, "0", "lb", dy=12)
        s.text(b, 120, -62, "120 s", "lb", "end", dy=12)
        return s
    b = Box((0, 10), (-64, 25))
    t, v = spont()
    xs, ys = spiky(t, v, SP_SPK)
    if hl in ("vm", "excluded", "vm_sd"):
        for ts in SP_SPK:
            s.rect(b, ts - 0.05, ts + 0.5, -64, 25, "sh")
    s.path(b, xs, ys)
    mask = np.ones_like(t, bool)
    for ts in SP_SPK:
        mask &= ~((t > ts - 0.05) & (t < ts + 0.5))
    med = float(np.median(v[mask]))
    if hl == "rate":
        for ts in SP_SPK:
            s.dot(b, ts + 0.02, 20)
        s.text(b, 0.2, -60, "rate = n spikes / window length")
    elif hl == "isi":
        s.line(b, 4.1, 23, 5.0, 23, "hl", "both")
        s.text(b, 4.55, 23, "ISI", anchor="middle", dy=-4)
    elif hl == "isi_cv":
        for a, c in zip(SP_SPK[:-1], SP_SPK[1:]):
            s.line(b, a, 23, c, 23, "hl", "both")
        s.text(b, 0.2, -60, "CV = SD(ISI) / mean(ISI)")
    elif hl == "vm":
        s.line(b, 0, med, 10, med, "hl")
        s.text(b, 0.2, -60, "median Vm, spikes removed (shaded)")
    elif hl == "vm_sd":
        sd = float(np.std(v[mask]))
        s.rect(b, 0, 10, med - 3 * sd, med + 3 * sd, "hb")
        s.line(b, 0, med, 10, med, "hl")
        s.text(b, 0.2, -60, "spread of Vm between spikes (SD)")
    elif hl == "excluded":
        s.text(b, 0.2, -60, "5 ms before to 50 ms after each spike (widened)")
    return s


# ------------------------------------------------------------------ current steps
ST_T = np.linspace(0, 2.2, 600)
TS, TE = 0.6, 1.1


def hyper_trace():
    t, vpre = ST_T, -50.0
    v = np.full_like(t, vpre) + smooth_noise(t.size, 0.15, 5)
    a = (t >= TS) & (t < TE)
    tau = t[a] - TS
    v[a] += -24 * (1 - np.exp(-tau / 0.035)) + 7 * (1 - np.exp(-tau / 0.15)) * (1 - np.exp(-tau / 0.035))
    vend = vpre - 24 + 7 * (1 - np.exp(-0.5 / 0.15))
    p = t >= TE
    tau2 = t[p] - TE
    v[p] = vend + (vpre - vend) * (1 - np.exp(-tau2 / 0.035)) + 13 * np.exp(-tau2 / 0.25) * (1 - np.exp(-tau2 / 0.03))
    return t, v, vpre


REB_SPK = [TE + 0.07, TE + 0.12, TE + 0.42]


def fig_step_neg(hl):
    s = Svg()
    b = Box((0, 2.2), (-78, 25), py=(8, 86))
    bc = Box((0, 2.2), (-1.3, 0.3), py=(94, 112))
    t, v, vpre = hyper_trace()
    a = (t >= TS) & (t < TE)
    i_min = np.where(a)[0][np.argmin(v[a])]
    ss = (t >= TE - 0.1) & (t < TE)
    vss = float(np.median(v[ss]))
    p = (t >= TE) & (t < TE + 0.5)
    i_rb = np.where(p)[0][np.argmax(v[p])]
    if hl == "vm_pre":
        s.rect(b, TS - 0.5, TS, -78, 25)
    if hl == "v_ss":
        s.rect(b, TE - 0.1, TE, -78, 25)
    if hl in ("rebound_peak",):
        s.rect(b, TE, TE + 0.5, -78, 25)
    if hl == "rebound_spikes":
        s.rect(b, TE, 2.2, -78, 25)
    s.path(b, *spiky(t, v, REB_SPK))
    s.path(bc, [0, TS, TS, TE, TE, 2.2], [0, 0, -1, -1, 0, 0], "cm")
    s.ptext(232, 104, "current", "lb", "end")
    if hl == "vm_pre":
        s.line(b, TS - 0.5, vpre, TS, vpre, "hl")
        s.text(b, 0.05, vpre, "Vm before step", dy=-6)
    elif hl == "v_min":
        s.dot(b, t[i_min], v[i_min])
        s.text(b, t[i_min], v[i_min], "minimum", dx=-4, dy=10, anchor="start")
    elif hl == "v_ss":
        s.line(b, TE - 0.1, vss, TE, vss, "hl")
        s.text(b, TE, vss, "steady state", dx=4, dy=10)
    elif hl == "delta_v_ss":
        s.line(b, 0.3, vpre, 1.05, vpre, "gd")
        s.line(b, 1.05, vpre, 1.05, vss, "hl", "both")
        s.text(b, 1.05, (vpre + vss) / 2, "ΔV", dx=4, dy=3)
    elif hl == "sag":
        s.line(b, 0.3, vpre, TE, vpre, "gd")
        s.line(b, t[i_min], v[i_min], TE, v[i_min], "gd")
        s.line(b, TS + 0.03, vpre, TS + 0.03, v[i_min], "hl", "both")
        s.line(b, TE - 0.03, v[i_min], TE - 0.03, vss, "hl", "both")
        s.text(b, TS + 0.03, (vpre + v[i_min]) / 2, "b", dx=-8, dy=3)
        s.text(b, TE - 0.03, (v[i_min] + vss) / 2, "a", dx=4, dy=3)
        s.text(b, 1.3, 15, "sag ratio = a / b")
    elif hl == "rebound_peak":
        s.line(b, 0.3, vpre, 2.2, vpre, "gd")
        s.line(b, t[i_rb] + 0.1, vpre, t[i_rb] + 0.1, v[i_rb], "hl", "both")
        s.text(b, t[i_rb] + 0.1, v[i_rb], "peak − Vm before", dx=5, dy=-2)
    elif hl == "rebound_spikes":
        for x in REB_SPK:
            s.dot(b, x + 0.004, 20)
        s.text(b, TE + 0.05, 8, "spikes after the step (5 s)")
    elif hl == "rebound_latency":
        s.line(b, TE, 5, REB_SPK[0], 5, "hl", "end")
        s.text(b, TE, 5, "latency", dx=-2, dy=-5, anchor="end")
    elif hl == "bursts":
        s.line(b, REB_SPK[0], 24, REB_SPK[1] + 0.01, 24, "hl")
        s.text(b, REB_SPK[0], 24, "burst (ISI ≤ 100 ms)", dx=-2, dy=-3, anchor="start")
    return s


DEP_SPK = [TS + x for x in (0.03, 0.08, 0.15, 0.24, 0.35, 0.48)]


def dep_trace(seed=7, plateau=12.0):
    t, vpre = ST_T, -55.0
    v = np.full_like(t, vpre) + smooth_noise(t.size, 0.15, seed)
    a = (t >= TS) & (t < TE)
    v[a] += plateau * (1 - np.exp(-(t[a] - TS) / 0.02))
    p = t >= TE
    v[p] += (plateau * (1 - np.exp(-0.5 / 0.02))) * np.exp(-(t[p] - TE) / 0.03) - 3 * np.exp(-(t[p] - TE) / 0.3) * (1 - np.exp(-(t[p] - TE) / 0.03))
    return t, v, vpre


def fig_step_pos(hl):
    s = Svg()
    b = Box((0, 2.2), (-66, 25), py=(8, 86))
    bc = Box((0, 2.2), (-0.3, 1.3), py=(94, 112))
    t, v, vpre = dep_trace()
    spk = list(DEP_SPK)
    peaks = None
    if hl == "rate_pre":
        spk = [0.22] + spk + [1.7]
    if hl == "block":
        spk = [TS + x for x in (0.02, 0.05, 0.09, 0.14, 0.2, 0.27, 0.35, 0.44)]
        peaks = [22, 18, 12, -6, -13, -18, -22, -25]
        t, v, vpre = dep_trace(plateau=20)
    if hl == "rate_pre":
        s.rect(b, 0, TS, -66, 25)
    if hl == "rate_after":
        s.rect(b, TE, 2.2, -66, 25)
    if hl == "rate_step":
        s.rect(b, TS, TE, -66, 25)
    s.path(b, *spiky(t, v, spk, peaks))
    s.path(bc, [0, TS, TS, TE, TE, 2.2], [0, 0, 1, 1, 0, 0], "cm")
    s.ptext(232, 104, "current", "lb", "end")
    if hl == "latency":
        s.line(b, TS, 0, spk[0], 0, "hl", "end")
        s.line(b, TS, -66, TS, 25, "gd")
        s.text(b, TS, 0, "latency", dx=-3, dy=3, anchor="end")
    elif hl in ("rate_step", "rate_pre", "rate_after"):
        sel = {"rate_step": lambda x: TS <= x < TE, "rate_pre": lambda x: x < TS,
               "rate_after": lambda x: x >= TE}[hl]
        for x in spk:
            if sel(x):
                s.dot(b, x + 0.004, 20)
        s.text(b, 0.05, -60, {"rate_step": "spikes in step / 0.5 s", "rate_pre": "spikes before the step",
                              "rate_after": "spikes after the step"}[hl])
    elif hl == "adaptation":
        for k, (a_, c_) in enumerate(zip(spk[:-1], spk[1:])):
            s.line(b, a_, 23 - 0 * k, c_, 23 - 0 * k, "hl", "both")
        s.text(b, 1.2, 14, "ISIs lengthen → index > 0")
    elif hl == "isi_cv_step":
        for a_, c_ in zip(spk[:-1], spk[1:]):
            s.line(b, a_, 23, c_, 23, "hl", "both")
        s.text(b, 1.2, 14, "CV = SD / mean of ISIs")
    elif hl == "ap_rheobase":
        s.el.append(f'<circle class="ring" cx="{f1(b.X(spk[0]))}" cy="{f1(b.Y(0))}" r="12"/>')
        s.text(b, spk[0], 20, "first valid spike", dx=14, dy=-2)
    elif hl == "block":
        for x, pk in zip(spk[3:], peaks[3:]):
            s.el.append(f'<circle class="ring" cx="{f1(b.X(x + 0.004))}" cy="{f1(b.Y(pk))}" r="4"/>')
        s.text(b, 1.15, 12, "attenuated events:")
        s.text(b, 1.15, 12, "not counted as spikes", dy=10)
    return s


def fig_rheobase(hl):
    s = Svg()
    rows = [(0, []), (50, [TS + 0.32]), (100, list(DEP_SPK))]
    for k, (cur, spk) in enumerate(rows):
        b = Box((0, 2.2), (-66, 25), px=(40, 232), py=(6 + 36 * k, 38 + 36 * k))
        t, v, _ = dep_trace(seed=10 + k, plateau=4 + cur / 10)
        if k == 1:
            s.rect(b, 0, 2.2, -66, 25, "hb")
        s.path(b, *spiky(t, v, spk))
        s.ptext(36, 26 + 36 * k, f"+{cur} pA", "lh" if k == 1 else "lb", "end")
    s.ptext(236, 46, "first step", "lh", "end")
    s.ptext(236, 55, "with ≥ 1 spike", "lh", "end")
    return s


def axes(s, b, xlab, ylab):
    s.line(b, b.x0, 0, b.x1, 0, "ax")
    s.line(b, 0, b.y0, 0, b.y1, "ax")
    s.ptext(b.px[1], b.Y(0) - 4, xlab, "lb", "end")
    if b.X(0) > 150:
        s.ptext(b.X(0) - 4, b.py[1] - 2, ylab, "lb", "end")
    else:
        s.ptext(b.X(0) + 4, b.py[0] + 6, ylab, "lb")


def fig_iv(hl):
    s = Svg()
    b = Box((-170, 20), (-32, 6), py=(10, 108))
    axes(s, b, "step (pA)", "ΔV ss (mV)")
    I = np.array([-150, -100, -50])
    dV = np.array([-27.5, -17.8, -9.6])
    k, c = np.polyfit(I, dV, 1)
    s.line(b, -165, k * -165 + c, 0, c, "hl")
    for x, y in zip(I, dV):
        s.dot(b, x, y)
    s.text(b, -150, -6, "slope = Rin")
    return s


def fig_fi(hl):
    s = Svg()
    b = Box((-20, 115), (-2, 16), px=(24, 228), py=(10, 104))
    axes(s, b, "step (pA)", "rate (Hz)")
    pts = [(0, 0), (50, 2), (100, 12)]
    for x, y in pts:
        s.dot(b, x, y, "dm" if hl == "fi_slope" and x == 0 else "dt")
    if hl == "fi_slope":
        s.line(b, 50, 2, 100, 12, "hl")
        s.text(b, 55, 9, "slope (Hz/pA)", anchor="end", dx=-4)
        s.ptext(228, 112, "fit from first spiking step up", "lb", "end")
    else:
        s.el.append(f'<circle class="ring" cx="{f1(b.X(100))}" cy="{f1(b.Y(12))}" r="6"/>')
        s.text(b, 100, 12, "max rate", dx=-9, dy=3, anchor="end")
    return s


# ------------------------------------------------------------------ ramp
R_T = np.linspace(0, 7, 900)
R_UP0, R_UP1, R_DN1, R_NEG0, R_REL = 0.6, 1.6, 2.6, 4.6, 5.1


def ramp_current(t):
    return np.interp(t, [0, R_UP0, R_UP1, R_DN1, R_NEG0, R_REL, R_REL + 1e-6, 7],
                     [0, 0, 100, 0, 0, -100, 0, 0])


UP_SPK = [1.0, 1.17, 1.3, 1.4, 1.48, 1.55]
DN_SPK = [1.63, 1.72, 1.83, 1.97, 2.15, 2.38]
RB_SPK = [5.2, 5.32]


def ramp_trace():
    t = R_T
    I = ramp_current(t)
    target = -55 + 0.18 * I
    v = np.empty_like(t)
    v[0] = target[0]
    dt = t[1] - t[0]
    for i in range(1, t.size):
        v[i] = v[i - 1] + (target[i] - v[i - 1]) * dt / 0.05
    m = t > R_REL
    v[m] += 11 * np.exp(-(t[m] - R_REL) / 0.3) * (1 - np.exp(-(t[m] - R_REL) / 0.04))
    return t, v + smooth_noise(t.size, 0.12, 9), I


def fig_ramp(hl):
    s = Svg()
    b = Box((0, 7), (-80, 25), py=(8, 78))
    bc = Box((0, 7), (-110, 110), py=(84, 114))
    t, v, I = ramp_trace()
    if hl == "n_up_down":
        s.rect(b, R_UP0, R_UP1, -80, 25)
        s.rect(b, R_UP1, R_DN1, -80, 25, "hb")
    s.path(b, *spiky(t, v, UP_SPK + DN_SPK + RB_SPK, dt_frac=1 / 900))
    s.path(bc, t, I, "cm")
    s.ptext(236, 110, "current", "lb", "end")
    i_at = lambda x: float(ramp_current(np.array([x]))[0])
    if hl in ("I_first", "hysteresis"):
        x = UP_SPK[0]
        s.line(b, x, 20, x, -80, "gd")
        s.dot(bc, x, i_at(x))
        s.text(bc, x, i_at(x), "I at 1st spike", dx=5, dy=3) if hl == "I_first" else None
    if hl in ("I_last", "hysteresis"):
        x = DN_SPK[-1]
        s.line(b, x, 20, x, -80, "gd")
        s.dot(bc, x, i_at(x))
        if hl == "I_last":
            s.text(bc, x, i_at(x), "I at last spike", dx=5, dy=3)
    if hl == "hysteresis":
        s.line(bc, R_DN1 + 0.25, i_at(DN_SPK[-1]), R_DN1 + 0.25, i_at(UP_SPK[0]), "hl", "both")
        s.line(bc, UP_SPK[0], i_at(UP_SPK[0]), R_DN1 + 0.3, i_at(UP_SPK[0]), "gd")
        s.text(bc, R_DN1 + 0.35, 40, "hysteresis = I first − I last")
    elif hl == "n_up_down":
        s.text(b, R_UP0, -70, "up")
        s.text(b, R_UP1 + 0.1, -70, "down")
    elif hl == "max_inst_rate":
        s.line(b, UP_SPK[-2], 23, UP_SPK[-1], 23, "hl", "both")
        s.text(b, R_DN1 + 0.1, 12, "shortest ISI → 1/ISI")
    elif hl == "v_thr_first":
        v0 = float(np.interp(UP_SPK[0], t, v))
        s.line(b, 0.1, v0, 3.5, v0, "gd")
        s.dot(b, UP_SPK[0], v0)
        s.text(b, 0.1, v0, "threshold, 1st spike", dy=11)
    elif hl == "latency":
        s.line(b, R_UP0, -66, UP_SPK[0], -66, "hl", "end")
        s.line(b, R_UP0, -80, R_UP0, 25, "gd")
        s.text(b, UP_SPK[0], -66, "latency", dx=4, dy=3)
    elif hl == "rin_ramp":
        a = (t > R_NEG0 + 0.05) & (t < R_REL)
        s.path(b, t[a], v[a], "hl")
        s.text(b, R_NEG0, -78, "slope ΔV/ΔI", dx=-4, dy=-2, anchor="end")
    elif hl == "v_min_neg":
        a = (t > R_NEG0) & (t < R_REL + 0.01)
        i = np.where(a)[0][np.argmin(v[a])]
        s.dot(b, t[i], v[i])
        s.text(b, t[i], v[i], "minimum", dx=-5, dy=3, anchor="end")
    elif hl == "rebound":
        for x in RB_SPK:
            s.dot(b, x + 0.004, 20)
        s.text(b, R_REL + 0.25, 2, "rebound")
    elif hl == "rheobase":
        s.rect(b, R_UP0 - 0.5, R_UP0, -80, 25)
        s.text(b, R_UP0 - 0.5, -70, "0.5 s: no spike", dx=-2, anchor="start")
        x = UP_SPK[0]
        s.line(b, x, 20, x, -80, "gd")
        s.dot(bc, x, i_at(x))
        s.text(bc, x, i_at(x), "I at 1st spike → mean", dx=5, dy=3)
    elif hl == "vm_pre":
        s.rect(b, 0.1, R_UP0, -80, 25)
        s.text(b, 0.1, -70, "before ramp")
    return s


# ------------------------------------------------------------------ timecourse
TC_B = np.arange(0, 1200, 10.0)
TAG_D, TAG_W = 160.0, 800.0


def tc_data():
    x = np.clip((TC_B + 5 - TAG_D) / 90.0, 0, None)
    resp = -9 * x * np.exp(1 - x) - 2.0 * (TC_B > TAG_D) * (TC_B < TAG_W)
    vm = -50 + resp + smooth_noise(TC_B.size, 0.35, 4)
    rate = np.clip(1.6 + resp * 0.2 + smooth_noise(TC_B.size, 0.15, 6), 0, None)
    return vm, rate


def fig_tc(hl):
    s = Svg()
    b = Box((0, 1200), (-62, -44), py=(10, 70))
    br = Box((0, 1200), (0, 3), py=(80, 110))
    vm, rate = tc_data()
    win = {"baseline": (TAG_D - 100, TAG_D), "drug": (TAG_D + 60, TAG_D + 260), "washout": (1100, 1200)}
    show = {"periods": list(win), "baseline": ["baseline"], "drug": list(win), "washout": ["washout"],
            "delta_drug": ["baseline", "drug"], "delta_washout": ["baseline", "washout"],
            "drift": ["baseline"]}.get(hl, [])
    for p in show:
        s.rect(b, *win[p], -62, -44)
        s.rect(br, *win[p], 0, 3)
    s.path(b, TC_B + 5, vm)
    s.path(br, np.repeat(TC_B, 2)[1:], np.repeat(rate, 2)[:-1], "cm")
    for x, lab in ((TAG_D, "drug tag"), (TAG_W, "washout tag")):
        s.line(b, x, -62, x, -44, "gd")
        s.line(br, x, 0, x, 3, "gd")
        s.text(b, x, -44, lab, "lb", dx=3, dy=8)
    s.ptext(236, 88, "rate", "lb", "end")
    s.ptext(236, 20, "Vm", "lb", "end")
    med = lambda p: float(np.median(vm[(TC_B >= win[p][0]) & (TC_B < win[p][1])]))
    if hl == "periods":
        s.text(b, TAG_D - 50, -61, "base", anchor="middle", dy=-2)
        s.text(b, TAG_D + 160, -61, "drug", anchor="middle", dy=-2)
        s.text(b, 1150, -61, "wash", anchor="middle", dy=-2)
    elif hl == "drug":
        s.line(br, TAG_D, 2.7, TAG_D + 60, 2.7, "hl", "end")
        s.text(br, TAG_D + 70, 2.7, "+60 s, then 200 s", dy=3)
    elif hl in ("delta_drug", "delta_washout"):
        q = "drug" if hl == "delta_drug" else "washout"
        s.line(b, win["baseline"][0], med("baseline"), win[q][1], med("baseline"), "gd")
        s.line(b, *win[q], med(q), med(q), "hl") if False else s.line(b, win[q][0], med(q), win[q][1], med(q), "hl")
        xm = win[q][1] + 15
        s.line(b, xm, med("baseline"), xm, med(q), "hl", "both")
        s.text(b, xm, (med("baseline") + med(q)) / 2, "Δ", dx=4, dy=3)
    elif hl == "drug_min":
        sm = np.convolve(vm, np.ones(3) / 3, "same")
        inside = (TC_B >= TAG_D) & (TC_B < TAG_W)
        i = np.where(inside)[0][np.argmin(sm[inside])]
        s.line(b, 0, med("baseline"), 1200, med("baseline"), "gd")
        s.line(b, TC_B[i] + 5, med("baseline"), TC_B[i] + 5, sm[i], "hl", "both")
        s.text(b, TC_B[i] + 5, sm[i], "▼ lowest point", dx=6, dy=4)
        s.line(br, TAG_D, 2.7, TC_B[i] + 5, 2.7, "hl", "end")
        s.text(br, TC_B[i] + 20, 2.7, "time from tag", dy=3)
    elif hl == "drug_max":
        sm = np.convolve(vm, np.ones(3) / 3, "same")
        inside = (TC_B >= TAG_D) & (TC_B < TAG_W)
        i = np.where(inside)[0][np.argmax(sm[inside])]
        s.line(b, 0, med("baseline"), 1200, med("baseline"), "gd")
        s.dot(b, TC_B[i] + 5, sm[i])
        s.text(b, TC_B[i] + 5, sm[i], "▲ highest point", dx=6, dy=-3)
    elif hl == "drug_max_rate":
        sr = np.convolve(rate, np.ones(3) / 3, "same")
        inside = (TC_B >= TAG_D) & (TC_B < TAG_W)
        for f, lab in ((np.argmax, "max"), (np.argmin, "min")):
            i = np.where(inside)[0][f(sr[inside])]
            s.dot(br, TC_B[i] + 5, sr[i])
            s.text(br, TC_B[i] + 5, sr[i], lab, dx=5, dy=3)
    elif hl == "drift":
        sel = (TC_B >= win["baseline"][0]) & (TC_B < win["baseline"][1])
        k, c = np.polyfit(TC_B[sel], vm[sel], 1)
        s.line(b, win["baseline"][0] - 40, k * (win["baseline"][0] - 40) + c, win["baseline"][1] + 40,
               k * (win["baseline"][1] + 40) + c, "hl")
        s.text(b, 230, -48, "baseline slope (mV/min)", dy=4)
    elif hl == "bins":
        for x in TC_B[30:45]:
            s.line(b, x, -62, x, -59.5, "hl")
        s.text(b, 300, -59, "10 s bins", dy=-3)
    return s


# ------------------------------------------------------------------ membrane test
MT_T = np.concatenate([np.linspace(0, 4, 60), np.linspace(4, 14, 500), np.linspace(14, 20, 260)[1:]])


def mt_trace():
    t = MT_T
    hold, dI, A1, A2, t1, t2 = -20.0, 30.0, 170.0, 45.0, 0.12, 0.8
    I = np.full_like(t, hold)
    on = (t >= 4) & (t < 14)
    tau = t[on] - 4
    I[on] = hold + dI + A1 * np.exp(-tau / t1) + A2 * np.exp(-tau / t2)
    off = t >= 14
    tau = t[off] - 14
    I[off] = hold - A1 * np.exp(-tau / t1) - A2 * np.exp(-tau / t2)
    return t, I, hold, dI


def fig_mt(hl):
    s = Svg()
    bv = Box((0, 20), (-0.2, 1.2), py=(8, 24))
    b = Box((0, 20), (-250, 260), py=(30, 112))
    t, I, hold, dI = mt_trace()
    on = (t >= 4) & (t < 14)
    if hl == "Cm":
        xs = np.concatenate([[4], t[on], [14]])
        ys = np.concatenate([[hold + dI], I[on], [hold + dI]])
        s.path(b, xs, ys, "fa")
    s.path(bv, [0, 4, 4, 14, 14, 20], [0, 0, 1, 1, 0, 0], "cm")
    s.ptext(236, 18, "ΔV", "lb", "end")
    s.path(b, t, I)
    s.ptext(236, 108, "current", "lb", "end")
    if hl == "holding":
        s.line(b, 0, hold, 20, hold, "hl")
        s.text(b, 0.3, hold, "holding current", dy=14)
    elif hl in ("Rt", "Rm"):
        s.line(b, 8, hold, 15.5, hold, "gd")
        s.line(b, 8, hold + dI, 14, hold + dI, "gd")
        s.line(b, 12.5, hold, 12.5, hold + dI, "hl", "both")
        s.text(b, 8, hold + dI, "ΔI (steady state)", dy=-5)
        s.text(b, 15, -120, "Rt = ΔV / ΔI" if hl == "Rt" else "Rm = Rt − Ra")
    elif hl == "Ra":
        s.dot(b, 4.02, float(I[on][0]))
        s.text(b, 4.02, float(I[on][0]), "fast peak", dx=6, dy=8)
        s.text(b, 8, -120, "Ra from the fast component", anchor="middle")
    elif hl == "Cm":
        s.text(b, 5.5, 90, "charge Q (area)")
    elif hl == "tau":
        a = (t >= 4.05) & (t < 7)
        s.path(b, t[a], I[a], "hl")
        s.text(b, 5.5, 60, "decay: τ fast + τ slow")
    return s


# ------------------------------------------------------------------ synaptic currents (planned)
def fig_psc(hl):
    s = Svg()
    b = Box((0, 2), (-38, 38), py=(10, 104))
    t = np.linspace(0, 2, 1600)
    exc = hl.startswith("e")
    times = [0.12, 0.31, 0.38, 0.66, 0.93, 1.12, 1.41, 1.55, 1.83] if exc else [0.2, 0.55, 0.98, 1.3, 1.74]
    amps = [18, 12, 26, 15, 22, 10, 30, 16, 20] if exc else [14, 22, 11, 18, 15]
    tau = 0.012 if exc else 0.035
    sign = -1 if exc else 1
    I = smooth_noise(t.size, 0.6, 21 if exc else 22)
    for ts, a in zip(times, amps):
        m = t >= ts
        I[m] += sign * a * (1 - np.exp(-(t[m] - ts) / 0.0015)) * np.exp(-(t[m] - ts) / tau)
    s.path(b, t, I)
    s.ptext(8, 116, ("sEPSCs: inward (down)" if exc else "sIPSCs: outward (up), depolarised holding") + "; schematic", "lb")
    peaks = [t[(t >= ts) & (t < ts + 0.01)][np.argmax(sign * I[(t >= ts) & (t < ts + 0.01)])] for ts in times]
    pv = [float(np.interp(x, t, I)) for x in peaks]
    if hl.endswith("freq"):
        for x, y in zip(peaks, pv):
            s.dot(b, x, y + sign * 4)
        s.text(b, 0.02, -sign * 32, "frequency = n events / time", dy=4)
    else:
        k = 6 if exc else 1
        s.line(b, times[k] - 0.08, 0, times[k] + 0.12, 0, "gd")
        s.line(b, peaks[k] + 0.03, 0, peaks[k] + 0.03, pv[k], "hl", "both")
        s.text(b, peaks[k] - 0.03, pv[k] / 2, "amplitude", dy=3, anchor="end")
        s.text(b, 0.02, -sign * 32, "peak − local baseline", dy=4)
    return s


# ------------------------------------------------------------------ cell level
def fig_cell(hl):
    s = Svg()
    if hl == "select":
        for k, (lab, ok) in enumerate((("recording 1", False), ("recording 2", True), ("recording 3", True))):
            y = 14 + 30 * k
            s.prect(14, y, 92, 22, "bx" if ok else "bxm", ' rx="3"')
            s.ptext(60, y + 15, lab + ("" if ok else " ✗"), "lb", "middle")
        s.el.append('<line class="hl" x1="106" y1="55" x2="140" y2="55" marker-end="url(#ah)"/>')
        s.prect(144, 44, 84, 22, "bxh", ' rx="3"')
        s.ptext(186, 59, "one row per cell", "lh", "middle")
        s.ptext(14, 110, "first usable recording (or an override)", "lb")
        return s
    if hl == "zscore":
        rng = np.random.default_rng(2)
        z = rng.normal(0, 1, (6, 9))
        z[2, 4] = np.nan
        for i in range(6):
            for j in range(9):
                x, y = 40 + j * 20, 10 + i * 14
                if np.isnan(z[i, j]):
                    s.prect(x, y, 19, 13, "hatch")
                    continue
                c = np.clip(z[i, j] / 2.5, -1, 1)
                col = f"rgb({int(255 - 120 * max(-c, 0) - 0 * c)},{int(255 - 150 * abs(c))},{int(255 - 120 * max(c, 0))})"
                s.prect(x, y, 19, 13, "", f' style="fill:{col}"')
            s.ptext(36, y + 10, f"C{i + 1}", "lb", "end")
        s.ptext(40, 106, "z = (value − mean of cells) / SD", "lb")
        s.ptext(40, 116, "red higher, blue lower, hatched missing", "lb")
        return s
    return s


# ------------------------------------------------------------------ content
TPL = {"psc": fig_psc, "ap": fig_ap, "sp": fig_spont, "sn": fig_step_neg, "spos": fig_step_pos, "rh": fig_rheobase,
       "iv": fig_iv, "fi": fig_fi, "rp": fig_ramp, "tc": fig_tc, "mt": fig_mt, "cell": fig_cell}

IN_09 = {"mt_Cm_pF", "step_rin_MOhm", "base_vm_baseline_mV", "step_sag_ratio", "base_firing_rate_Hz", "base_isi_cv",
         "ramp_ap_threshold_mV", "ramp_ap_half_width_ms", "ramp_ap_amplitude_mV", "ramp_ap_upstroke_mV_ms",
         "ramp_ap_ahp_mV", "ramp_rheobase_pA", "step_fi_slope_Hz_per_pA",
         "step_max_rate_step_Hz", "ramp_n_rebound_spikes_short", "ramp_rebound_peak_mV", "step_rebound_peak_mV",
         "tc_delta_drug_vm_baseline_mV", "tc_drug_min_delta_vm_mV"}   # per condition in 08: tc_<condition>_...

# Each card: (title, columns, unit, (template, highlight), how, meaning, note, tags)
SECTIONS = [
 ("mt", "Membrane test", "03_ephys_membrane-test", "mt_",
  "Voltage clamp. A small square voltage pulse (ΔV) is applied and the current response is measured. "
  "Values are the median over pulses; table <code>membrane_test_values.csv</code>.",
  [
   ("Membrane capacitance", ["Cm_pF"], "pF", ("mt", "Cm"),
    "Charge carried by the capacitive transient (area above the steady state, corrected for the charge lost while "
    "the membrane settles), scaled by the resistances: Cm = Q·Rt / (ΔV·Rm).",
    "Proportional to the membrane area the pulse can charge (about 1 µF/cm²). A larger value means a larger cell or "
    "more proximal dendrite.",
    "Distant dendrites charge slowly and are only partly included, so Cm underestimates the total membrane area.", []),
   ("Membrane resistance", ["Rm_MOhm"], "MΩ", ("mt", "Rm"),
    "Total resistance minus access resistance: Rm = Rt − Ra.",
    "How few channels are open at the holding potential. A high Rm means a small current moves Vm a lot. It is the "
    "voltage-clamp counterpart of the input resistance from the steps (Rin).",
    "", []),
   ("Total resistance", ["Rt_MOhm"], "MΩ", ("mt", "Rt"),
    "ΔV divided by the steady-state current change ΔI.",
    "The resistance in series from pipette to ground: access plus membrane. With a good recording, Rt ≈ Rm.",
    "", []),
   ("Access (series) resistance", ["Ra_MOhm", "Ra_clampex_MOhm"], "MΩ", ("mt", "Ra"),
    "From the fast component of a two-exponential fit to the mean transient. The fit models the 2 kHz recording "
    "filter: Ra = ΔV / (A_fast + A_slow + ΔI). The Clampex value is kept as <code>Ra_clampex_MOhm</code>.",
    "Describes the <b>recording</b>, not the cell: the resistance between the pipette and the cell interior. A high "
    "Ra slows and filters voltage, and distorts Cm and Rm.",
    "Use it for quality control. Not validated against the Clampex readout of these recordings yet.", ["QC"]),
   ("Transient time constants", ["tau_fast_ms", "tau_slow_ms", "fast_fraction"], "ms, fraction", ("mt", "tau"),
    "Two decay time constants of the capacitive transient; fast_fraction = A_fast / (A_fast + A_slow).",
    "The fast component mostly reflects charging of the soma through Ra. The slow component reflects membrane "
    "farther away, such as dendrites. A large slow share points to a lot of electrically distant membrane.",
    "Descriptive; also used to check the Ra fit (τ_fast < 0.1 ms is rejected).", ["QC"]),
   ("Holding current", ["holding_current_pA"], "pA", ("mt", "holding"),
    "Mean current at the holding potential, around the pulses.",
    "The current needed to keep the cell at the holding potential. A large negative value usually means a leaky seal "
    "or an unhealthy cell.",
    "", ["QC"]),
  ]),
 ("base", "Spontaneous firing (baseline)", "04_ephys_baseline", "base_",
  "Current clamp at 0 pA, no stimulus. The cell's own activity in the slice; table <code>baseline_features.csv</code>.",
  [
   ("Analysis window", ["window_start_s", "window_end_s", "window_duration_s"], "s", ("sp", "window"),
    "The last 100 s of the ~120 s recording (configurable; at least 60 s).",
    "Skips the first seconds, when the cell may still be settling after the protocol starts.", "", []),
   ("Firing rate", ["n_spikes", "firing_rate_Hz"], "count, Hz", ("sp", "rate"),
    "Number of detected spikes divided by the window length.",
    "How active the cell is on its own, without injected current.",
    "In the 09 overview the rate is log-transformed, because it is very skewed (0 to several Hz).", []),
   ("Firing class", ["firing_class"], "silent / tonic", ("sp", "class"),
    "Silent: ≤ 10 spikes in 100 s (0.1 Hz). Tonic: more. The cut-off is in config.yaml.",
    "Whether the cell fires spontaneously. Silent cells can still fire when depolarised (see the ramp rheobase).",
    "Colours the 09 overview: blue silent, red tonic.", []),
   ("Membrane potential", ["vm_baseline_mV", "vm_mean_mV"], "mV", ("sp", "vm"),
    "Median (and mean) Vm of the window after removing each spike from 5 ms before to 50 ms after its threshold.",
    "In a silent cell this is the resting membrane potential. In a firing cell it is the average potential between "
    "spikes, not a true resting potential.",
    "Not corrected for the liquid junction potential (+15.9 mV): the true Vm is ≈ 15.9 mV more negative. Flagged when a bias current was applied.", []),
   ("Vm fluctuation", ["vm_sd_mV"], "mV", ("sp", "vm_sd"),
    "Standard deviation of Vm after removing spikes.",
    "Size of the voltage fluctuations between spikes: synaptic input, subthreshold oscillations or an unstable "
    "recording.", "", []),
   ("Time excluded", ["fraction_time_excluded"], "fraction", ("sp", "excluded"),
    "Share of the window removed around spikes before computing Vm.",
    "Quality check for Vm: in a fast-firing cell much of the trace is removed and Vm is less certain.", "", ["QC"]),
   ("Inter-spike interval", ["n_isi", "isi_mean_s", "isi_median_s"], "s", ("sp", "isi"),
    "Time between consecutive spikes; mean and median.",
    "The typical pause between spikes. The median is less affected by occasional long pauses.", "", []),
   ("ISI coefficient of variation", ["isi_cv"], "—", ("sp", "isi_cv"),
    "SD / mean of the inter-spike intervals; needs ≥ 5 intervals.",
    "Regularity of firing. About 0 = clock-like (pacemaker-like); about 1 = irregular, as random; above 1 = "
    "clustered spikes or long pauses.",
    "Missing for silent cells.", []),
   ("AP waveform (spontaneous)", ["ap_*"], "", ("ap", "none"),
    "Median waveform features of valid, isolated spontaneous spikes (previous ISI ≥ 0.5 s). Definitions: see "
    "<a href='#ap'>Action potential</a>.",
    "AP shape without injected current. Only available for cells that fire spontaneously.", "", []),
  ]),
 ("ap", "Action potential waveform", "spike detection (all protocols)", "",
  "Measured on every detected spike (spikes.py, adapted from the Allen Institute IPFX method). Reported as "
  "<code>ap_*</code> in 04 (spontaneous), 05 (first spike at the first spiking step), 06 (<b>AP at ramp rheobase</b>: "
  "mean of the first valid up-ramp spike of each quiet sweep, the sweeps of the rheobase) and 07 (per period). The "
  "cell's AP measure, used in 08, 09 and the 10 figures (average AP, phase plane), is the ramp AP "
  "(<code>ramp_ap_*</code>): every cell at its own threshold current. "
  "Absolute voltages (threshold, peak, trough) are not corrected for the +15.9 mV liquid junction potential; "
  "amplitude, AHP, half-width and the dV/dt values are unaffected.",
  [
   ("Threshold", ["ap_threshold_mV"], "mV", ("ap", "threshold"),
    "Walking back from the steepest rise, the Vm where dV/dt falls below 5 % of that spike's own maximum rise.",
    "The voltage at which the spike takes off. A more negative threshold means a spike starts closer to rest; it "
    "depends on how many sodium channels are available.", "", []),
   ("Peak", ["ap_peak_mV"], "mV", ("ap", "peak"),
    "Maximum Vm of the spike.", "The overshoot of the spike.", "", []),
   ("Amplitude", ["ap_amplitude_mV"], "mV", ("ap", "amplitude"),
    "Peak − threshold.",
    "Size of the spike. It shrinks when sodium channels inactivate, e.g. during fast firing or depolarisation block.",
    "", []),
   ("Half-width", ["ap_half_width_ms"], "ms", ("ap", "half_width"),
    "Width at half the amplitude (between threshold and peak), interpolated.",
    "How fast the spike repolarises. Broad spikes let in more calcium; narrow spikes allow faster firing.",
    "Spikes broader than 5 ms are flagged and left out of the medians.", []),
   ("Upstroke", ["ap_upstroke_mV_ms"], "mV/ms", ("ap", "upstroke"),
    "Maximum dV/dt during the rise.",
    "Reflects the sodium current available. Below 50 mV/ms the spike is flagged (slow_upstroke).", "", []),
   ("Downstroke", ["ap_downstroke_mV_ms"], "mV/ms", ("ap", "downstroke"),
    "Most negative dV/dt during the fall (a negative number).",
    "Reflects the repolarising potassium current.", "", []),
   ("Upstroke/downstroke ratio", ["ap_upstroke_downstroke_ratio"], "—", ("ap", "ratio"),
    "Upstroke / |downstroke|.",
    "One number for the spike's shape. A high ratio means slow repolarisation relative to the rise. Used in cell-type "
    "classification (Allen Cell Types).", "", []),
   ("Afterhyperpolarisation (AHP)", ["ap_ahp_mV"], "mV", ("ap", "ahp"),
    "Lowest Vm within 50 ms after the peak, minus threshold (negative = below threshold).",
    "How far the cell is pulled down after a spike, mostly by potassium currents. A deep AHP spaces spikes further "
    "apart.", "", []),
  ]),
 ("step", "Current steps", "05_ephys_steps", "step_",
  "Current clamp. One 500 ms step per sweep: −150 to +100 pA in 50 pA steps, then ~25 s without current. Per sweep: "
  "<code>step_sweeps.csv</code>; per recording: <code>step_features.csv</code>.",
  [
   ("Vm and rate before the step", ["vm_pre_mV", "rate_pre_Hz"], "mV, Hz", ("spos", "rate_pre"),
    "Spike-free median Vm in the 0.5 s before the step; spikes before the step / time.",
    "The state of the cell at the start of each sweep. The reference for ΔV, sag and rebound.", "", []),
   ("Input resistance", ["rin_MOhm"], "MΩ", ("iv", "rin"),
    "Slope of the steady-state voltage change (ΔV) against step current, over the hyperpolarising steps (≥ 2).",
    "How many mV the cell moves per pA of current. A high Rin means a small or electrically tight cell; synaptic "
    "currents then move Vm more.",
    "Measured below rest, so active currents such as I_h contribute. Compare with Rm (03) and the ramp Rin (06).", []),
   ("Steady-state voltage change", ["v_ss_mV", "delta_v_ss_mV"], "mV", ("sn", "delta_v_ss"),
    "Median Vm in the last 100 ms of the step (spikes removed), and its difference from Vm before.",
    "The settled response to each step; the points used for Rin.", "", []),
   ("Minimum during the step", ["v_min_mV"], "mV", ("sn", "v_min"),
    "Lowest Vm during the hyperpolarising step (whole step, Allen convention).",
    "The peak hyperpolarisation, before sag pulls Vm back.", "", []),
   ("Sag ratio", ["sag_ratio"], "fraction (0–1)", ("sn", "sag"),
    "(steady state − minimum) / (Vm before − minimum), at the most negative step.",
    "The share of the hyperpolarisation that recovers during the step. Sag is typical of the HCN current I_h. "
    "0 = no sag.",
    "Which current causes it has not been tested pharmacologically here.", []),
   ("Rebound depolarisation", ["rebound_peak_mV"], "mV", ("sn", "rebound_peak"),
    "Highest Vm (spikes removed) within 0.5 s after the step, minus Vm before. At the most negative step.",
    "How far the cell overshoots above its starting Vm after being hyperpolarised. Often due to I_h or to "
    "T-type calcium channels recovering from inactivation.", "", []),
   ("Rebound spikes", ["n_rebound_spikes_short", "n_rebound_spikes_all"], "count", ("sn", "rebound_spikes"),
    "Spikes within 5 s after the step (short) or until the end of the sweep (all).",
    "Whether release from inhibition makes the cell fire.", "", []),
   ("Rebound latency", ["latency_first_rebound_spike_ms"], "ms", ("sn", "rebound_latency"),
    "Time from the end of the step to the first rebound spike.", "How quickly the cell fires after inhibition ends.",
    "", []),
   ("Rebound bursts", ["n_rebound_bursts"], "count", ("sn", "bursts"),
    "Groups of ≥ 2 spikes with intervals ≤ 100 ms (bursts closer than 200 ms are merged).",
    "Whether rebound firing comes as bursts rather than single spikes.", "", []),
   ("First spiking step", ["first_spiking_step_pA", "step_increment_pA"], "pA", ("rh", "rheobase"),
    "Smallest positive step with at least one spike during the step.",
    "<b>Not the rheobase</b>: with 50 pA steps it only says that the rheobase lies between the previous step and this "
    "one. It is the start of the F–I fit and the step whose first spike gives the AP waveform.",
    "The cell's rheobase comes from the ramp (<a href='#ramp'>06</a>, <code>rheobase_pA</code>).", []),
   ("F–I slope", ["fi_slope_Hz_per_pA"], "Hz/pA", ("fi", "fi_slope"),
    "Linear fit of firing rate against step current, from the first spiking step upwards (≥ 2 steps).",
    "Gain: how much faster the cell fires per extra pA.",
    "With +50 and +100 pA as the only positive steps, the slope often rests on 2 points.", []),
   ("Maximum step rate", ["max_rate_step_Hz"], "Hz", ("fi", "max_rate"),
    "Highest firing rate over all steps.",
    "Maximal sustained output up to +100 pA. Can be limited by depolarisation block.", "", []),
   ("Rate during the step", ["n_spikes_step", "rate_step_Hz"], "count, Hz", ("spos", "rate_step"),
    "Spikes during the step / 0.5 s (per sweep).", "The F–I points.", "", []),
   ("First-spike latency", ["latency_first_spike_ms"], "ms", ("spos", "latency"),
    "Time from step onset to the threshold of the first spike (per sweep).",
    "How quickly the cell responds. A long delay can indicate a transient potassium current (I_A).", "", []),
   ("Adaptation index", ["adaptation_index", "adaptation_step_pA"], "—", ("spos", "adaptation"),
    "Allen: mean of (ISIₙ₊₁ − ISIₙ)/(ISIₙ₊₁ + ISIₙ), from the step with the most spikes; needs ≥ 4 spikes.",
    "0 = constant rate; > 0 = the cell slows during the step (spike-frequency adaptation); < 0 = it speeds up.",
    "Missing (flag too_few_spikes_for_adaptation) when no step has 4 spikes.", []),
   ("ISI CV during the step", ["isi_cv_step"], "—", ("spos", "isi_cv_step"),
    "SD / mean of the intervals during the step (≥ 3 intervals).", "Regularity of evoked firing.", "", []),
   ("Rate after the step", ["rate_after_step_Hz"], "Hz", ("spos", "rate_after"),
    "Spikes after the step / time (per sweep).", "Firing that persists or appears after the step.", "", []),
   ("Depolarisation block", ["first_step_with_block_pA", "n_attenuated_step"], "pA, count", ("spos", "block"),
    "Attenuated events: peaks ≥ 20 mV prominent and above −20 mV that are not detected as spikes (e.g. rise slower "
    "than 20 mV/ms). They are <b>counted</b>, in their own column (<code>n_attenuated_step</code>); the smallest step "
    "with such events is reported.",
    "Strong depolarisation keeps sodium channels inactivated, so the cell can no longer fire full spikes. Shows how "
    "much drive the cell tolerates.",
    "Kept out of the spike count, firing rate and AP medians: they are failed spikes, and it is unknown whether "
    "they propagate along the axon. Marked B in the 09 overview.", []),
   ("AP at the first spiking step (secondary)", ["ap_*"], "", ("spos", "ap_rheobase"),
    "Waveform of the first valid spike at the first spiking step. Definitions: see "
    "<a href='#ap'>Action potential</a>.",
    "A square-step AP. With 50 pA steps, cells get very different overdrive, so it is kept as a secondary "
    "measure; the cell's AP measure is the ramp AP (06).", "", []),
  ]),
 ("ramp", "Current ramp", "06_ephys_ramp", "ramp_",
  "Current clamp. 0 → +100 → 0 pA (500 ms each), 2 s at 0, 0 → −100 pA over 500 ms, abrupt return to 0. "
  "Three sweeps; the rheobase is the mean over the quiet sweeps, the other values the median over sweeps.",
  [
   ("Rheobase", ["rheobase_pA", "rheobase_sd_pA", "n_sweeps_rheobase"], "pA", ("rp", "rheobase"),
    "Mean current at the first spike on the rising ramp, over the sweeps with no spike in the 0.5 s before the "
    "ramp. A sweep in which the cell was already firing is left out: its first ramp spike may be spontaneous.",
    "How much current the cell needs to start firing. Lower = more excitable. This is the cell's rheobase.",
    "Depends on the ramp speed (200 pA/s) and is slightly overestimated because Vm lags the ramp. NaN with the "
    "flag no_quiet_sweep_for_rheobase if the cell fired before every ramp.", []),
   ("First-spike current per sweep", ["I_first_spike_pA", "n_spikes_before_ramp", "rheobase_sweep_ok"],
    "pA, count", ("rp", "I_first"),
    "Per sweep: current at the first up-ramp spike, and the spikes in the 0.5 s before the ramp. "
    "<code>rheobase_sweep_ok</code> = no spike there.",
    "The values the rheobase is averaged from; the spread between sweeps shows how stable it is.",
    "The recording-level <code>I_first_spike_pA</code> is the median over all sweeps, quiet or not.", []),
   ("AP at ramp rheobase", ["ap_*", "n_sweeps_ap", "ap_waveforms.csv"], "", ("rp", "I_first"),
    "Mean waveform features of the first VALID up-ramp spike of each quiet sweep (the sweeps of the rheobase); "
    "their waveforms (−5 to +15 ms) are saved for the average AP and phase plane. Definitions: see "
    "<a href='#ap'>Action potential</a>.",
    "The cell's AP measure: every cell at its own threshold current, under the same condition.",
    "A slow ramp inactivates some sodium channels: ramp APs can have a higher threshold and slower upstroke than "
    "step APs. Fallback to a non-quiet sweep is flagged (ap_from_non_quiet_sweep).", []),
   ("Threshold of the first ramp spike", ["v_threshold_first_mV"], "mV", ("rp", "v_thr_first"),
    "AP threshold of that first spike.",
    "Threshold during slow depolarisation. It can be higher than during a step, because some sodium channels "
    "inactivate on the way up.", "", []),
   ("Ramp latency", ["latency_first_spike_ms"], "ms", ("rp", "latency"),
    "Time from the start of the rising ramp to the first spike.", "Same information as the first-spike current, in time.",
    "", []),
   ("Spikes up / down", ["n_spikes_up", "n_spikes_down"], "count", ("rp", "n_up_down"),
    "Spikes on the rising and falling halves.", "Output during increasing and decreasing drive.", "", []),
   ("Max instantaneous rate", ["max_inst_rate_up_Hz"], "Hz", ("rp", "max_inst_rate"),
    "1 / shortest interval on the rising ramp.", "The fastest the cell fires under ramp drive.", "", []),
   ("Current at the last spike", ["I_last_spike_down_pA"], "pA", ("rp", "I_last"),
    "Current at the last spike on the falling ramp.", "The current at which firing stops.", "", []),
   ("Hysteresis", ["hysteresis_pA"], "pA", ("rp", "hysteresis"),
    "First-spike current − last-spike current.",
    "> 0: the cell keeps firing at lower currents on the way down than it needed to start, as with persistent inward "
    "currents. < 0: firing stops early, as with adaptation or inactivation.",
    "Which currents are involved is not tested here.", []),
   ("Ramp input resistance", ["rin_ramp_MOhm"], "MΩ", ("rp", "rin_ramp"),
    "Slope of Vm against current on the negative ramp, skipping the first 50 ms.",
    "Input resistance measured with a slow, continuous change. Compare with the step Rin.", "", []),
   ("Minimum on the negative ramp", ["v_min_neg_mV"], "mV", ("rp", "v_min_neg"),
    "Lowest Vm on the negative ramp.", "Depth of the hyperpolarisation before release.", "", []),
   ("Ramp rebound", ["rebound_peak_mV", "n_rebound_spikes_short", "n_rebound_spikes_all",
                     "latency_first_rebound_spike_ms", "n_rebound_bursts"], "mV, count, ms", ("rp", "rebound"),
    "As for the steps, after the abrupt return from −100 pA to 0 pA (peak within 0.5 s, spikes within 5 s).",
    "Firing after release from hyperpolarisation.",
    "Marked R in the 09 overview when at least one rebound spike occurs.", []),
   ("Vm and rate before the ramp", ["vm_pre_mV", "rate_pre_Hz"], "mV, Hz", ("rp", "vm_pre"),
    "Spike-free median Vm and rate in the 0.5 s before the ramp.", "The starting state of each sweep.", "", []),
   ("Attenuated events", ["n_attenuated_up", "n_attenuated_down"], "count", ("spos", "block"),
    "As for the steps, on the rising and falling halves; counted, but not as spikes.",
    "Depolarisation block during the ramp.", "", []),
  ]),
 ("tc", "Drug timecourse", "07_ephys_timecourse", "tc_",
  "Current clamp, long recording, drugs applied to the bath. Periods are placed from the ABF tags. A recording can "
  "hold several <b>applications</b>; every value below exists per application (one row each in "
  "<code>timecourse_summary.csv</code>), and 08 keeps one column set per drug condition "
  "(<code>tc_5HT_…</code>, <code>tc_SP_…</code>). The response describes the drug; it says nothing about which "
  "receptor is involved.",
  [
   ("Applications and conditions", ["application", "condition", "compounds_added", "layered", "baseline_reference",
                                    "name_source"], "", ("tc", "periods"),
    "Every drug tag starts an application. <b>Solitary</b>: the previous compound was washed out first; condition = "
    "the compound(s) named in the tag. <b>Layered</b>: added while the previous compound is still in the bath; "
    "condition = all compounds (e.g. <code>5-HT + Ba2+</code>). Names come from the tag text (aliases in config), "
    "else from the recording book, else <code>unknown</code> (flagged).",
    "Which drug(s) the cell was exposed to, in which order. A later application can be affected by the earlier one "
    "(incomplete washout, desensitisation); this is flagged, never averaged away.",
    "Write the compound in every drug tag, e.g. \"SP hit bath\", \"Ba applied\".", []),
   ("Periods", ["baseline_start_s", "pre_tag_start_s", "drug_start_s", "washout_start_s", "…_end_s"], "s",
    ("tc", "drug"),
    "Baseline: last 100 s before the tag (solitary) or before the first tag of the chain (layered). Pre-tag (layered "
    "only): last 100 s before its own tag. Drug: from 60 s after the tag, 200 s long, cut at the next tag. Washout: "
    "last 100 s before the next drug tag or the end of the recording.",
    "Fixed windows, the same for every cell, so they are not chosen from the response.", "", []),
   ("Period features", ["baseline_*", "pre_tag_*", "drug_*", "washout_*"], "", ("tc", "periods"),
    "The 04 features (Vm, rate, ISI CV, class, AP waveform) for each period. The silent cut-off is scaled to the "
    "period length (0.1 Hz).",
    "The state of the cell before, during and after the drug.", "", []),
   ("Drug effect", ["delta_drug_*"], "as feature", ("tc", "delta_drug"),
    "Drug period − baseline period, e.g. <code>delta_drug_vm_baseline_mV</code>.",
    "The average effect over the fixed drug window: negative ΔVm = hyperpolarisation.",
    "For a layered application this includes what the earlier compound still does (initial baseline).", []),
   ("Effect of the added compound (layered)", ["delta_drug_vs_pre_tag_*"], "as feature", ("tc", "delta_drug"),
    "Drug period − pre-tag period of a layered application.",
    "The change caused by the compound added on top, e.g. what Ba²⁺ does in the presence of SP.",
    "Missing for solitary applications.", []),
   ("Washout change", ["delta_washout_*"], "as feature", ("tc", "delta_washout"),
    "Washout period − baseline period.",
    "Whether the effect reverses. Near 0 = recovered; close to the drug effect = no recovery (or drift).", "", []),
   ("Peak response", ["drug_peak_delta_vm_mV", "drug_peak_time_s", "drug_peak_direction"], "mV, s",
    ("tc", "drug_min"),
    "Whichever of the lowest and highest point (below) is larger in size, relative to the baseline Vm.",
    "The size and speed of the response for hyperpolarising (5-HT) and depolarising drugs alike.",
    "Selected per cell, so biased away from zero: a cell without a response still has a peak. Check the figure.", []),
   ("Lowest point", ["drug_min_vm_mV", "drug_min_delta_vm_mV", "drug_min_time_s"], "mV, s", ("tc", "drug_min"),
    "Lowest Vm of the binned timecourse (smoothed over 3 bins, 30 s) between this drug tag and the next tag; "
    "relative to the baseline Vm; time after the tag.",
    "The peak hyperpolarisation, wherever it falls, and how quickly it is reached.", "", []),
   ("Highest point", ["drug_max_vm_mV", "drug_max_delta_vm_mV", "drug_max_time_s"], "mV, s", ("tc", "drug_max"),
    "As the lowest point, for the highest Vm.", "Captures depolarising responses.", "", []),
   ("Rate extremes", ["drug_max_rate_Hz", "drug_min_rate_Hz"], "Hz", ("tc", "drug_max_rate"),
    "Highest and lowest smoothed rate between this drug tag and the next tag.",
    "Whether the drug silences or speeds up firing at its strongest point.", "", []),
   ("Baseline drift", ["baseline_drift_mV_per_min"], "mV/min", ("tc", "drift"),
    "Slope of the binned Vm over the baseline period. Above 2 mV/min it is flagged unstable_baseline.",
    "Quality check: if Vm was already drifting, part of the drug effect may be drift.", "", ["QC"]),
   ("Binned timecourse", ["timecourse_bins.csv"], "per 10 s", ("tc", "bins"),
    "Spike count, rate and spike-free median Vm per 10 s bin, with time relative to the first drug tag.",
    "The full response over time, for plotting and for the lowest point.", "", []),
  ]),
 ("psc", "Synaptic currents (planned)", "separate miniML workflow", "",
  "Voltage clamp, spontaneous postsynaptic currents (PSCs: currents, not spikes). Detection and measurement are "
  "<b>not</b> part of this package: they will come from the miniML workflow, matched to these cells by "
  "<code>recording_id</code> and <code>cell_tag</code>. Definitions below are the usual ones and still to be fixed in "
  "that workflow. Holding potentials are as set on the amplifier (not corrected for the +15.9 mV junction potential), "
  "which matters for the driving force of each current.",
  [
   ("sEPSC frequency", ["sEPSC_frequency_Hz"], "Hz", ("psc", "e_freq"),
    "Number of detected excitatory events / analysed time.",
    "How much excitatory synaptic input the cell receives: number of active inputs and their release probability.",
    "With TTX the same measure gives miniature (mEPSC) frequency, which reflects release without action potentials.",
    ["planned"]),
   ("sEPSC amplitude", ["sEPSC_amplitude_pA"], "pA", ("psc", "e_amp"),
    "Peak of each event relative to the local baseline just before it; median over events.",
    "Strength of single excitatory inputs, mostly set by the postsynaptic receptors (number, conductance).",
    "Small events are lost in the noise, so the amplitude depends on the detection threshold.", ["planned"]),
   ("sIPSC frequency", ["sIPSC_frequency_Hz"], "Hz", ("psc", "i_freq"),
    "Number of detected inhibitory events / analysed time.",
    "How much inhibitory synaptic input the cell receives.",
    "Inhibitory currents are separated by holding potential or by receptor blockers; which one is used still has to "
    "be decided.", ["planned"]),
   ("sIPSC amplitude", ["sIPSC_amplitude_pA"], "pA", ("psc", "i_amp"),
    "Peak of each event relative to the local baseline; median over events.",
    "Strength of single inhibitory inputs. Depends on the holding potential relative to the chloride reversal "
    "potential.", "", ["planned"]),
  ]),
 ("cell", "Cell level", "08_ephys_cell-summary, 09_ephys_cell-overview", "",
  "The cell (Cell Tag) is the unit of analysis.",
  [
   ("Recording selection", ["selection.csv"], "", ("cell", "select"),
    "Per cell and protocol, the first usable recording in file order, or an override from config. Recordings are "
    "never averaged. Columns get a prefix: mt_, base_, step_, ramp_, tc_.",
    "One value per cell per property, without pseudoreplication.", "", []),
   ("Z-score (09 heatmap)", ["cell_overview_zscores.csv"], "SD", ("cell", "zscore"),
    "(value − mean over cells) / SD over cells, per property, clipped at ±2.5. The firing rate is log-transformed first.",
    "Shows whether a cell is high or low compared with the other cells. It is relative to the cells in the "
    "dataset, not an absolute value.", "", []),
  ]),
]


def card(sec_prefix, title, cols, unit, fig, how, meaning, note, tags):
    tpl, hl = fig
    svg = TPL[tpl](hl).svg(title)
    full = [sec_prefix + c if sec_prefix and not c.endswith(".csv") and "*" not in c else c for c in cols]
    badges = ""
    if any(f in IN_09 for f in full) or (sec_prefix == "step_" and cols == ["ap_*"]):
        badges += '<span class="badge b09">in 09 overview</span>'
    if "planned" in tags:
        badges += '<span class="badge bpl">planned · miniML, not in this package</span>'
    if "QC" in tags:
        badges += '<span class="badge bqc">recording quality</span>'
    col_html = " ".join(f"<code>{html.escape(c)}</code>" for c in cols)
    unit_html = f'<span class="unit">{html.escape(unit)}</span>' if unit else ""
    note_html = f'<p class="note">{note}</p>' if note else ""
    search = html.escape(" ".join([title, *cols, *full]).lower())
    return f"""<article class="card" data-s="{search}">
<div class="fig">{svg}</div>
<div class="body"><h3>{html.escape(title)} {unit_html}</h3>
<div class="cols">{col_html}</div>{badges and f'<div class="badges">{badges}</div>'}
<dl><dt>How it is measured</dt><dd>{how}</dd><dt>What it says about the cell</dt><dd>{meaning}</dd></dl>{note_html}
</div></article>"""


CSS = """
:root{--bg:#f7f6f3;--card:#fff;--ink:#1f2328;--mut:#6a6f76;--line:#e3e1dc;--hl:#c2410c;--hlbg:rgba(194,65,12,.10);
--sh:rgba(37,99,235,.10);--cm:#8a8f96;--code:#f0eee9;--b09:#1d4ed8;--bqc:#7c5e10}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#15171a;--card:#1d2024;--ink:#e6e6e3;--mut:#9aa0a6;
--line:#30343a;--hl:#fb923c;--hlbg:rgba(251,146,60,.14);--sh:rgba(96,165,250,.14);--cm:#7d838a;--code:#262a2f;--b09:#93b4ff;--bqc:#e3c46b}}
:root[data-theme="dark"]{--bg:#15171a;--card:#1d2024;--ink:#e6e6e3;--mut:#9aa0a6;--line:#30343a;--hl:#fb923c;
--hlbg:rgba(251,146,60,.14);--sh:rgba(96,165,250,.14);--cm:#7d838a;--code:#262a2f;--b09:#93b4ff;--bqc:#e3c46b}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
header{max-width:1180px;margin:0 auto;padding:32px 16px 8px}
h1{font-size:26px;margin:0 0 6px;letter-spacing:-.01em}
header p{color:var(--mut);margin:4px 0;max-width:820px}
nav{position:sticky;top:0;z-index:5;background:var(--bg);border-bottom:1px solid var(--line)}
nav .in{max-width:1180px;margin:0 auto;padding:10px 16px;display:flex;gap:6px 14px;flex-wrap:wrap;align-items:center}
nav a{color:var(--ink);text-decoration:none;font-size:14px;padding:2px 0;border-bottom:2px solid transparent}
nav a:hover{border-color:var(--hl)}
nav input{margin-left:auto;min-width:200px;flex:0 1 260px;padding:6px 10px;border:1px solid var(--line);border-radius:6px;
background:var(--card);color:var(--ink);font:inherit;font-size:14px}
main{max-width:1180px;margin:0 auto;padding:8px 16px 60px}
section{margin-top:28px;scroll-margin-top:60px}
h2{font-size:20px;margin:0 0 2px}
.src{color:var(--mut);font-size:13px;margin:0 0 4px}
.intro{margin:0 0 14px;max-width:860px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden;display:flex;flex-direction:column}
.fig{border-bottom:1px solid var(--line);padding:6px 10px 2px}
.fig svg{display:block;width:100%;max-width:320px;height:auto;margin:0 auto}
.body{padding:10px 14px 14px}
h3{font-size:16px;margin:0 0 4px}
.unit{font-weight:400;color:var(--mut);font-size:13px;margin-left:4px}
.cols{margin:0 0 6px;line-height:1.9}
code{font:12.5px ui-monospace,SFMono-Regular,Consolas,monospace;background:var(--code);padding:1px 5px;border-radius:4px;word-break:break-word}
.badges{margin:0 0 6px}
.badge{display:inline-block;font-size:11.5px;padding:1px 7px;border-radius:99px;border:1px solid;margin-right:6px}
.b09{color:var(--b09)}.bqc{color:var(--bqc)}.bpl{color:var(--mut)}
dl{margin:0}dt{font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:var(--mut);margin-top:6px}
dd{margin:2px 0 0}
.note{margin:8px 0 0;font-size:13.5px;color:var(--mut);border-left:3px solid var(--line);padding-left:8px}
a{color:var(--b09)}
.ljp{border-left:3px solid var(--hl);padding-left:10px;color:var(--ink)!important}
footer{max-width:1180px;margin:0 auto;padding:0 16px 40px;color:var(--mut);font-size:13px}
.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:13px;color:var(--mut);margin-top:8px}
.legend span{display:inline-flex;align-items:center;gap:6px}
.sw{display:inline-block;width:18px;height:3px;border-radius:2px}
svg .tr{fill:none;stroke:var(--ink);stroke-width:1.1;stroke-linejoin:round}
svg .cm{fill:none;stroke:var(--cm);stroke-width:1.1}
svg .hl{fill:none;stroke:var(--hl);stroke-width:1.6}
svg path.hl{stroke-width:2.4}
svg .gd{stroke:var(--mut);stroke-width:.8;stroke-dasharray:3 3;fill:none}
svg .ax{stroke:var(--mut);stroke-width:.8}
svg .sh{fill:var(--sh)}
svg .hb{fill:var(--hlbg)}
svg .fa{fill:var(--hlbg);stroke:none}
svg .dt{fill:var(--hl)}
svg .dm{fill:var(--mut)}
svg .ring{fill:none;stroke:var(--hl);stroke-width:1.4}
svg .lb{fill:var(--mut);font:9px system-ui,sans-serif}
svg .lh{fill:var(--hl);font:600 9.5px system-ui,sans-serif}
svg .bx{fill:var(--card);stroke:var(--ink);stroke-width:.8}
svg .bxm{fill:var(--card);stroke:var(--mut);stroke-width:.8;stroke-dasharray:3 2}
svg .bxh{fill:var(--hlbg);stroke:var(--hl);stroke-width:1}
svg .hatch{fill:url(#hatch)}
.hide{display:none}
@media (max-width:600px){nav input{margin-left:0;flex:1 1 100%}h1{font-size:22px}}
"""

JS = """
const q=document.getElementById('q');
q.addEventListener('input',()=>{const s=q.value.trim().toLowerCase();
document.querySelectorAll('.card').forEach(c=>c.classList.toggle('hide',s&&!c.dataset.s.includes(s)));
document.querySelectorAll('section').forEach(sec=>{const any=sec.querySelector('.card:not(.hide)');sec.classList.toggle('hide',!any);});});
"""

DEFS = ('<svg width="0" height="0" style="position:absolute" aria-hidden="true"><defs>'
        '<marker id="ah" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
        '<path d="M0 0L8 4L0 8z" style="fill:var(--hl)"/></marker>'
        '<pattern id="hatch" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
        '<rect width="5" height="5" style="fill:var(--code)"/><line x1="0" y1="0" x2="0" y2="5" style="stroke:var(--mut);stroke-width:1"/></pattern>'
        '</defs></svg>')


def build():
    nav = "".join(f'<a href="#{k}">{html.escape(t)}</a>' for k, t, *_ in SECTIONS)
    secs = []
    n = 0
    for key, title, src, prefix, intro, cards in SECTIONS:
        body = "".join(card(prefix, *c) for c in cards)
        n += len(cards)
        shown = prefix + "&lt;condition&gt;_" if key == "tc" else prefix
        pre = f" · in the 08 cell summary these columns start with <code>{shown}</code>" if prefix else ""
        secs.append(f'<section id="{key}"><h2>{html.escape(title)}</h2><p class="src">{html.escape(src)}{pre}</p>'
                    f'<p class="intro">{intro}</p><div class="grid">{body}</div></section>')
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PMvDAT+ Parameter Guide</title><style>{CSS}</style></head><body>{DEFS}
<header><h1>PMvDAT+ intrinsic electrophysiology: parameter guide</h1>
<p>Every parameter the pipeline extracts (03–09): how it is measured and what it says about the cell. The sketches are
schematic, drawn from synthetic traces, not from recordings. The highlighted part (orange) is what the parameter measures.</p>
<p class="ljp"><b>Liquid junction potential:</b> +15.9 mV (K-gluconate internal, standard ACSF, 35 °C, Ag/AgCl pellet), <b>not compensated</b> and not corrected here. All absolute voltages (Vm, threshold, peak, minimum) are as recorded; the true membrane potential is ≈ 15.9 mV more negative (V<sub>true</sub> = V<sub>recorded</sub> − 15.9 mV). Differences (amplitude, AHP, ΔV, sag, Rin, drug changes) are unaffected.</p>
<p>Levels: <b>spike</b> → <b>sweep</b> → <b>recording</b> (one ABF) → <b>cell</b> (one Cell Tag). Thresholds and windows
are set in each subproject's <code>config.yaml</code>; the values given here are the current defaults.</p>
<div class="legend"><span><i class="sw" style="background:var(--ink)"></i>voltage</span>
<span><i class="sw" style="background:var(--cm)"></i>injected current / command</span>
<span><i class="sw" style="background:var(--hl)"></i>what is measured</span>
<span><i class="sw" style="background:var(--sh);height:10px"></i>analysis window</span></div></header>
<nav><div class="in">{nav}<input id="q" type="search" placeholder="Filter, e.g. sag or rin_MOhm" aria-label="Filter parameters"></div></nav>
<main>{''.join(secs)}</main>
<footer>{n} entries. Flags, run IDs and bookkeeping columns are not listed. Generated from the package code definitions
(spikes.py, baseline.py, steps.py, ramp.py, timecourse.py, membrane_test.py, cell_overview.py).</footer>
<script>{JS}</script></body></html>"""
    out = DOCS / "parameter_guide.html"
    out.write_text(page, encoding="utf-8")
    print(out, len(page) // 1024, "kB", n, "entries")


build()
