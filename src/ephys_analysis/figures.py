"""Publication-style figures of the intrinsic properties of PMvDAT+ neurons (10). Descriptive.

Reads the outputs of 02 and 05-09 (one recording per cell and protocol, as chosen by 08);
ABFs are read only for the example traces. Every cell is shown; a mean is only drawn
together with the individual cells; n is given in every panel. The only statistics are the
PROVISIONAL paired drug tests (drug_stats.py), marked as such on the figure.

    fig1_examples       example traces: silent and tonic spontaneous, step family, ramp
    fig2_labels         silent / tonic, block, rebound: stacked count bars + cell x label dot matrix
    fig3_properties     passive and firing properties: one dot per cell, median and IQR
    fig4_ap             AP at ramp rheobase: average waveform, phase plane, AP properties
    fig5_excitability   F-I per cell, block onset (censored above the largest step), rebound vs sag
    fig6_drug_<cond>    per drug condition: dVm timecourse, paired estimation plot (DABEST style:
                        effect size + bootstrap 95 % CI as the headline), peak vs sham peak
    fig7_estimation_<cond>  optional: estimation plot per secondary outcome
    summary_table       n, mean +- SD, median [IQR], range per property
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .baseline import read_voltage, save_figure
from .cell_overview import CLASS_COLOURS, _order
from .drug_stats import CAVEAT
from .utils import envelope

# Study settings (10 config: liquid_junction_potential_mV, ljp_solution, cell_type_label); set by ``configure``.
SETTINGS = {"ljp_mV": 15.9, "ljp_solution": "K-gluconate, 35 °C", "cell_type_label": "PMvDAT+"}


def configure(cfg: dict) -> None:
    """Take the study settings from the 10 config (missing keys keep their defaults)."""
    for key, name in (("ljp_mV", "liquid_junction_potential_mV"), ("ljp_solution", "ljp_solution"),
                      ("cell_type_label", "cell_type_label")):
        if cfg.get(name) is not None:
            SETTINGS[key] = cfg[name]


def ljp_note() -> str:
    v = float(SETTINGS["ljp_mV"])
    return (f"Membrane potentials are not corrected for the liquid junction potential ({v:+.1f} mV, "
            f"{SETTINGS['ljp_solution']}): the true Vm is ~{abs(v):.1f} mV {'more negative' if v > 0 else 'more positive'}.")


def _plt(save: bool, style: dict | None = None):
    import matplotlib
    if save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"svg.fonttype": "none", "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
                         "axes.spines.top": False, "axes.spines.right": False, **(style or {})})
    return plt


def _note(fig, text: str, width: int | None = None, height: float = 0.06) -> None:
    """Footnote below the panels: reserve space at the bottom of the constrained layout.
    ``width``: characters per line (default: from the figure width, ~19 per inch at 6 pt)."""
    import textwrap
    wrapped = textwrap.fill(text, width or int(19 * fig.get_figwidth()))
    n_lines = wrapped.count("\n") + 1
    height = max(height, (0.11 * n_lines + 0.08) / fig.get_figheight())   # inches of text -> figure fraction
    fig.get_layout_engine().set(rect=(0, height, 1, 1 - height))       # (left, bottom, width, height)
    fig.text(0.01, 0.005, wrapped, fontsize=6, va="bottom", ha="left")


def _finish(fig, out_stem, formats):
    if out_stem is None:
        return fig
    return save_figure(fig, out_stem, formats)


def _colour(labels: pd.DataFrame, cell) -> str:
    return CLASS_COLOURS.get(labels.loc[cell, "firing_class"], "0.5") if cell in labels.index else "0.5"


# --------------------------------------------------------------------------- tables
def summary_table(summary: pd.DataFrame, props: list[dict]) -> pd.DataFrame:
    """Per property (``props``: column, label, unit): n, mean, SD, median, Q1, Q3, min, max
    over cells (NaN excluded)."""
    rows = []
    for p in props:
        x = pd.to_numeric(summary[p["column"]], errors="coerce").dropna() if p["column"] in summary else pd.Series(dtype=float)
        rows.append({"property": p["label"], "column": p["column"], "unit": p.get("unit", ""), "n": int(x.size),
                     "mean": x.mean() if x.size else np.nan, "sd": x.std(ddof=1) if x.size > 1 else np.nan,
                     "median": x.median() if x.size else np.nan, "q1": x.quantile(0.25) if x.size else np.nan,
                     "q3": x.quantile(0.75) if x.size else np.nan, "min": x.min() if x.size else np.nan,
                     "max": x.max() if x.size else np.nan})
    out = pd.DataFrame(rows)
    out["note"] = ljp_note()
    return out


# --------------------------------------------------------------------------- panels
def strip(ax, values: pd.Series, labels: pd.DataFrame, title: str, log: bool = False, seed: int = 0,
          markers: pd.Series | None = None, show_summary: bool = True):
    """One dot per cell (colour = firing class), median line and IQR bar, n in the title."""
    x = pd.to_numeric(values, errors="coerce").dropna()
    rng = np.random.default_rng(seed)
    for cell, val in x.items():
        m = markers.get(cell, "o") if markers is not None else "o"
        ax.plot(rng.uniform(-0.18, 0.18), val, m, ms=5, mfc=_colour(labels, cell), mec="k", mew=0.4, alpha=0.9)
    if len(x) and show_summary:
        q1, med, q3 = x.quantile([0.25, 0.5, 0.75])
        ax.plot([-0.32, 0.32], [med, med], color="k", lw=1.2)
        ax.plot([0.38, 0.38], [q1, q3], color="k", lw=1.0)
    if log:
        ax.set_yscale("symlog", linthresh=0.01)
    ax.set_xlim(-0.5, 0.5)
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    ax.set_title(f"{title}\n(n = {len(x)})", fontsize=8)


def fig_labels(labels: pd.DataFrame, out_stem: Path | None = None, formats=("png", "svg")):
    """Stacked count bars (silent / tonic, block, rebound) and a cell x label dot matrix."""
    plt = _plt(out_stem is not None)
    rows = _order(labels)
    lab = labels.loc[rows]
    fig, (ax_b, ax_m) = plt.subplots(1, 2, figsize=(7.2, max(2.6, 0.22 * len(rows) + 1.2)),
                                     gridspec_kw={"width_ratios": [1.6, 1]}, constrained_layout=True)
    n = len(lab)
    bars = [("Firing at rest", [("silent", (lab["firing_class"] == "silent").sum(), CLASS_COLOURS["silent"]),
                                ("tonic", (lab["firing_class"] == "tonic").sum(), CLASS_COLOURS["tonic"])]),
            ("Depolarisation block", [("block", lab["block"].sum(), "0.25"), ("no block", (~lab["block"]).sum(), "0.85")]),
            ("Rebound spikes", [("rebound", lab["rebound"].sum(), "0.25"), ("none", (~lab["rebound"]).sum(), "0.85")])]
    for i, (name, parts) in enumerate(bars):
        left = 0
        for txt, k, col in parts:
            k = int(k)
            if k:
                ax_b.barh(i, k, left=left, color=col, edgecolor="k", lw=0.5, height=0.6)
                ax_b.text(left + k / 2, i, f"{txt}\n{k}/{n}", ha="center", va="center", fontsize=7,
                          color="w" if col in ("0.25", CLASS_COLOURS["tonic"], CLASS_COLOURS["silent"]) else "k")
            left += k
    ax_b.set_yticks(range(len(bars)))
    ax_b.set_yticklabels([b[0] for b in bars])
    ax_b.invert_yaxis()
    ax_b.set_xlim(0, max(n, 1))
    from matplotlib.ticker import MaxNLocator
    ax_b.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_b.set_xlabel("Cells")
    ax_b.set_title(f"Intrinsic labels (n = {n} {SETTINGS['cell_type_label']} cells)")
    cols = ["firing_class", "block", "rebound"]
    for i, c in enumerate(rows):
        ax_m.plot(0, i, "o", ms=7, color=_colour(labels, c))
        for j, k in enumerate(cols[1:], 1):
            ax_m.plot(j, i, "o", ms=7, mfc="0.2" if lab.loc[c, k] else "w", mec="0.2", mew=0.8)
    ax_m.set_xticks(range(3))
    ax_m.set_xticklabels(["silent (blue) /\ntonic (red)", "block", "rebound"], fontsize=7)
    ax_m.set_yticks(range(len(rows)))
    ax_m.set_yticklabels(rows, fontsize=6)
    ax_m.set_ylim(len(rows) - 0.5, -0.5)
    ax_m.set_xlim(-0.5, 2.5)
    ax_m.set_title("Per cell (filled = present)")
    return _finish(fig, out_stem, formats)


def fig_properties(summary: pd.DataFrame, labels: pd.DataFrame, props: list[dict], out_stem: Path | None = None,
                   formats=("png", "svg"), ncols: int = 4):
    """Strip plots in real units for ``props`` (column, label, log)."""
    plt = _plt(out_stem is not None)
    nrows = int(np.ceil(len(props) / ncols)) or 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.8 * ncols, 2.2 * nrows), constrained_layout=True, squeeze=False)
    s = summary.set_index("cell_tag")
    for k, p in enumerate(props):
        ax = axes[k // ncols][k % ncols]
        vals = s[p["column"]] if p["column"] in s else pd.Series(dtype=float)
        strip(ax, vals, labels, p["label"], log=p.get("log", False), seed=k)
    for k in range(len(props), nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")
    _note(fig, "One dot per cell (blue silent, red tonic at rest); line = median, bar = IQR. " + ljp_note(),
          height=0.07)
    return _finish(fig, out_stem, formats)


def cell_mean_waveforms(waves: pd.DataFrame, rec_to_cell: dict) -> pd.DataFrame:
    """Mean AP waveform per cell (over its sweeps): columns cell_tag, t_ms, v_mV, dvdt_mV_ms."""
    w = waves[waves["recording_id"].isin(rec_to_cell)].copy()
    if w.empty:
        return pd.DataFrame(columns=["cell_tag", "t_ms", "v_mV", "dvdt_mV_ms"])
    w["cell_tag"] = w["recording_id"].map(rec_to_cell)
    w["t_ms"] = w["t_ms"].round(4)
    return w.groupby(["cell_tag", "t_ms"], as_index=False)[["v_mV", "dvdt_mV_ms"]].mean()


def fig_ap(waves: pd.DataFrame, summary: pd.DataFrame, labels: pd.DataFrame, rec_to_cell: dict, ap_props: list[dict],
           out_stem: Path | None = None, formats=("png", "svg")):
    """Average AP aligned at threshold and phase plane (each cell thin, mean bold), plus
    strips of the AP properties (``ap_props``)."""
    plt = _plt(out_stem is not None)
    cw = cell_mean_waveforms(waves, rec_to_cell)
    fig = plt.figure(figsize=(7.2, 5.4), constrained_layout=True)
    gs = fig.add_gridspec(2, max(len(ap_props), 2))
    half = max(len(ap_props), 2) // 2
    ax_w = fig.add_subplot(gs[0, :half])
    ax_p = fig.add_subplot(gs[0, half:])
    cells = list(cw["cell_tag"].unique())
    for c in cells:
        g = cw[cw["cell_tag"] == c]
        ax_w.plot(g["t_ms"], g["v_mV"], color=_colour(labels, c), lw=0.7, alpha=0.6)
        ax_p.plot(g["v_mV"], g["dvdt_mV_ms"], color=_colour(labels, c), lw=0.7, alpha=0.6)
    if cells:
        m = cw.groupby("t_ms")[["v_mV", "dvdt_mV_ms"]].mean().reset_index()
        ax_w.plot(m["t_ms"], m["v_mV"], color="k", lw=1.8, label="mean")
        ax_p.plot(m["v_mV"], m["dvdt_mV_ms"], color="k", lw=1.8)
        ax_w.legend(frameon=False)
    ax_w.axvline(0, color="0.6", lw=0.6, ls=":")
    ax_w.set_xlabel("Time from threshold (ms)")
    ax_w.set_ylabel("Vm (mV)")
    ax_w.set_title(f"AP at ramp rheobase (n = {len(cells)} cells)")
    ax_p.set_xlabel("Vm (mV)")
    ax_p.set_ylabel("dV/dt (mV/ms)")
    ax_p.set_title("Phase plane")
    s = summary.set_index("cell_tag")
    for k, p in enumerate(ap_props):
        ax = fig.add_subplot(gs[1, k])
        strip(ax, s[p["column"]] if p["column"] in s else pd.Series(dtype=float), labels, p["label"], seed=k)
    fig.suptitle("First valid up-ramp spike of each quiet ramp sweep, averaged per cell (blue silent, red tonic)",
                 fontsize=8)
    return _finish(fig, out_stem, formats)


def fig_excitability(step_sweeps: pd.DataFrame, summary: pd.DataFrame, labels: pd.DataFrame, rec_to_cell: dict,
                     out_stem: Path | None = None, formats=("png", "svg")):
    """F-I per cell (open markers = steps with attenuated events), block onset (cells without
    block plotted above the largest step), rebound spikes vs sag ratio."""
    plt = _plt(out_stem is not None)
    fig, (ax_f, ax_b, ax_r) = plt.subplots(1, 3, figsize=(7.2, 2.6), constrained_layout=True,
                                           gridspec_kw={"width_ratios": [2, 1, 1.6]})
    st = step_sweeps[step_sweeps["recording_id"].isin(rec_to_cell)].copy()
    st["cell_tag"] = st["recording_id"].map(rec_to_cell)
    dep = st[st["I_step_pA"] >= -1e-6]
    for c, g in dep.groupby("cell_tag"):
        g = g.sort_values("I_step_pA")
        col = _colour(labels, c)
        ax_f.plot(g["I_step_pA"], g["rate_step_Hz"], "-", color=col, lw=0.8)
        blk = g["n_attenuated_step"] > 0 if "n_attenuated_step" in g else pd.Series(False, index=g.index)
        ax_f.plot(g.loc[~blk, "I_step_pA"], g.loc[~blk, "rate_step_Hz"], "o", color=col, ms=4)
        ax_f.plot(g.loc[blk, "I_step_pA"], g.loc[blk, "rate_step_Hz"], "o", mfc="w", mec=col, ms=5, mew=1)
    ax_f.set_xlabel("Step current (pA)")
    ax_f.set_ylabel("Firing rate (Hz)")
    ax_f.set_title(f"F–I (n = {dep['cell_tag'].nunique()}; open = block)")
    s = summary.set_index("cell_tag")
    top = float(dep["I_step_pA"].max()) if len(dep) else 100.0
    onset = s.get("step_first_step_with_block_pA", pd.Series(dtype=float))
    has_steps = s.index[s.get("step_recording", pd.Series("", index=s.index)).fillna("").astype(str) != ""]
    vals = pd.Series({c: (onset.get(c) if pd.notna(onset.get(c, np.nan)) else top + 25) for c in has_steps})
    marks = pd.Series({c: ("o" if pd.notna(onset.get(c, np.nan)) else "^") for c in has_steps})
    strip(ax_b, vals, labels, "Block onset (pA)", markers=marks, show_summary=False)
    ax_b.axhline(top + 12.5, color="0.6", lw=0.6, ls=":")
    ticks = [y for y in ax_b.get_yticks() if y <= top + 1e-9 and y >= ax_b.get_ylim()[0]] + [top + 25]
    ax_b.set_yticks(ticks)
    ax_b.set_yticklabels([f"{y:g}" for y in ticks[:-1]] + [f"> {top:g}\n(no block)"])
    x = pd.to_numeric(s.get("step_sag_ratio", pd.Series(dtype=float)), errors="coerce")
    y = pd.to_numeric(s.get("ramp_n_rebound_spikes_short", pd.Series(dtype=float)), errors="coerce")
    ok = x.notna() & y.notna()
    for c in x[ok].index:
        ax_r.plot(x[c], y[c], "o", ms=5, mfc=_colour(labels, c), mec="k", mew=0.4)
    ax_r.set_xlabel("Sag ratio (−150 pA step)")
    ax_r.set_ylabel("Rebound spikes after ramp (5 s)")
    ax_r.set_title(f"Rebound vs sag (n = {int(ok.sum())})")
    return _finish(fig, out_stem, formats)


def aligned_timecourse(apps: pd.DataFrame, bins: pd.DataFrame, t_min: float = -150, t_max: float = 900,
                       step: float = 10.0) -> pd.DataFrame:
    """Binned Vm of each application, relative to its baseline Vm, on a common grid of time
    from its drug tag (linear interpolation of the bin centres): cell_tag, t_s, dvm_mV."""
    grid = np.arange(t_min, t_max + step / 2, step)
    rows = []
    for _, a in apps.iterrows():
        b = bins[bins["recording_id"] == a["recording_id"]]
        if b.empty or not np.isfinite(a.get("drug_tag_s", np.nan)):
            continue
        tc = b["bin_start_s"] + b["bin_duration_s"] / 2 - a["drug_tag_s"]
        ok = b["vm_median_mV"].notna()
        y = np.interp(grid, tc[ok], b.loc[ok, "vm_median_mV"], left=np.nan, right=np.nan)
        rows.append(pd.DataFrame({"cell_tag": a["cell_tag"], "t_s": grid, "dvm_mV": y - a["baseline_vm_baseline_mV"]}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["cell_tag", "t_s", "dvm_mV"])


def _effect_text(est: dict, unit: str, kind: str = "mean_diff") -> str:
    """'Δ = -8.1 mV [95 % CI -9.2, -7.0]' (or the median / Hedges' g) from ``estimate``."""
    lo, hi, name, u = {"mean_diff": ("ci_low", "ci_high", "Δ mean", unit),
                       "median_diff": ("median_ci_low", "median_ci_high", "Δ median", unit),
                       "hedges_g": ("g_ci_low", "g_ci_high", "g", "")}[kind]
    v = est.get(kind, np.nan)
    if not np.isfinite(est.get(lo, np.nan)):
        return f"{name} = {v:+.2f} {u}".strip() + (f" ({est['note']})" if est.get("note") else "")
    return f"{name} = {v:+.2f} {u} [95 % CI {est[lo]:+.2f}, {est[hi]:+.2f}]".replace("  ", " ")


def estimation_plot(ax_raw, ax_eff, values: pd.DataFrame, labels: pd.DataFrame, s, unit: str = "mV",
                    ylabel: str = "Vm (mV)") -> list[dict]:
    """Paired Cumming estimation plot (Ho et al. 2019, DABEST style; drawn here without the
    dabest dependency). ``values``: cells x levels, the FIRST level is the shared reference.
    Top (``ax_raw``): each cell's values connected across the levels. Bottom (``ax_eff``): for
    every other level, the paired difference from the reference: bootstrap distribution
    (half-violin), mean difference (dot) and its 95 % CI (bar; BCa by default).
    Returns the ``drug_stats.estimate`` result of every contrast."""
    from scipy.stats import gaussian_kde

    from .drug_stats import estimate

    levels = list(values.columns)
    for c, r in values.iterrows():
        ax_raw.plot(range(len(levels)), r.to_numpy(dtype=float), "o-", color=_colour(labels, c), ms=3.5, lw=0.8,
                    alpha=0.85)
    ax_raw.set_xticks(range(len(levels)))
    ax_raw.set_xticklabels([f"{lv}\n(n = {int(values[lv].notna().sum())})" for lv in levels], fontsize=7)
    ax_raw.set_xlim(-0.5, len(levels) - 0.5)
    ax_raw.set_ylabel(ylabel)
    ests = []
    ref = values[levels[0]].to_numpy(dtype=float)
    for k, lv in enumerate(levels[1:], 1):
        est = estimate(ref, values[lv].to_numpy(dtype=float), s)
        ests.append(est)
        boots = est["boot_mean_diff"]
        if boots.size > 10 and np.ptp(boots) > 0:
            grid = np.linspace(boots.min(), boots.max(), 200)
            dens = gaussian_kde(boots)(grid)
            ax_eff.fill_betweenx(grid, k, k + 0.35 * dens / dens.max(), color="0.6", alpha=0.5, lw=0)
        if np.isfinite(est["mean_diff"]):
            ax_eff.plot(k, est["mean_diff"], "o", color="k", ms=5, zorder=3)
        if np.isfinite(est["ci_low"]):
            ax_eff.plot([k, k], [est["ci_low"], est["ci_high"]], color="k", lw=1.5, zorder=3)
    ax_eff.axhline(0, color="0.5", lw=0.6, ls="--")
    ax_eff.set_xticks(range(1, len(levels)))
    ax_eff.set_xticklabels([f"{lv} −\n{levels[0]}" for lv in levels[1:]], fontsize=7)
    ax_eff.set_xlim(-0.5, len(levels) - 0.5)
    ax_eff.set_ylabel(f"Paired Δ ({unit})" if unit else "Paired Δ")
    return ests


def fig_drug(condition: str, apps: pd.DataFrame, bins: pd.DataFrame, labels: pd.DataFrame, stats_rows: pd.DataFrame,
             drug_window: tuple[float, float], out_stem: Path | None = None, formats=("png", "svg"),
             sham: pd.DataFrame | None = None, s=None):
    """One drug condition: (A) ΔVm from the tag, each cell thin, mean bold where every cell
    has data, drug window shaded; (B) paired estimation plot of Vm: reference (baseline; pre-tag
    for a layered application) -> drug -> washout per cell, and the paired differences from the
    reference with their bootstrap distribution and 95 % CI; (C) peak vs sham-peak excursion.
    The effect size with its CI is the headline; p-values are only in drug_stats.csv
    (PROVISIONAL)."""
    from .drug_stats import DrugStatsSettings

    s = s or DrugStatsSettings()
    plt = _plt(out_stem is not None)
    fig = plt.figure(figsize=(7.2, 4.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[2.2, 1.5, 1.0], height_ratios=[1.3, 1])
    ax_t = fig.add_subplot(gs[:, 0])
    ax_raw = fig.add_subplot(gs[0, 1])
    ax_eff = fig.add_subplot(gs[1, 1])
    ax_s = fig.add_subplot(gs[:, 2])
    tcx = aligned_timecourse(apps, bins)
    for c, g in tcx.groupby("cell_tag"):
        ax_t.plot(g["t_s"], g["dvm_mV"], color=_colour(labels, c), lw=0.7, alpha=0.7)
    if len(tcx):
        m = tcx.pivot_table(index="t_s", columns="cell_tag", values="dvm_mV", dropna=False).mean(axis=1, skipna=False)
        ax_t.plot(m.index, m.values, color="k", lw=1.6, label="mean")
        ax_t.legend(frameon=False, loc="lower right")
    ax_t.axvspan(*drug_window, color="tab:red", alpha=0.1, lw=0)
    ax_t.axvline(0, color="k", lw=0.6, ls="--")
    ax_t.axhline(0, color="0.6", lw=0.5)
    ax_t.set_xlabel(f"Time from {condition} tag (s)")
    ax_t.set_ylabel("ΔVm from baseline (mV)")
    ax_t.set_title(f"{condition} (n = {tcx['cell_tag'].nunique()} {SETTINGS['cell_type_label']} cells)")
    layered = bool(apps["layered"].astype(bool).any()) if "layered" in apps and len(apps) else False
    ref = "pre_tag" if layered else "baseline"
    cols = {ref: f"{ref}_vm_baseline_mV", "drug": "drug_vm_baseline_mV", "washout": "washout_vm_baseline_mV"}
    vals = pd.DataFrame({k: (apps.set_index("cell_tag")[c] if c in apps else np.nan) for k, c in cols.items()})
    ests = estimation_plot(ax_raw, ax_eff, vals.rename(columns={"pre_tag": "pre-tag"}), labels, s)
    ax_raw.set_title(f"Drug − {ref.replace('_', '-')}:\n{_effect_text(ests[0], 'mV', s.effect_size)}", fontsize=7)
    if sham is not None and len(sham):
        for _, r in sham.iterrows():
            ax_s.plot([0, 1], [abs(r["sham"]), abs(r["peak"])], "o-", color=_colour(labels, r["cell_tag"]), ms=4, lw=0.8)
    ax_s.set_xticks([0, 1])
    ax_s.set_xticklabels(["sham", "drug"])
    ax_s.set_xlim(-0.4, 1.4)
    ax_s.set_ylabel("|peak excursion| (mV)")
    ax_s.set_title("Peak vs sham peak", fontsize=8)
    ci = ests[0].get("ci_method") or "bootstrap"
    _note(fig, f"{CAVEAT}. Estimation plot: dot = mean paired difference, bar = {ci} 95 % CI "
               f"({s.n_boot} resamples), grey = bootstrap distribution; p-values in drug_stats.csv. Shaded: drug period. "
               f"Bold: mean where every cell has data. {ljp_note()}", height=0.1)
    return _finish(fig, out_stem, formats)


def fig_estimation_secondary(condition: str, apps: pd.DataFrame, labels: pd.DataFrame, outcomes: list[str], s,
                             out_stem: Path | None = None, formats=("png", "svg")):
    """One paired estimation plot (reference -> drug) per secondary outcome (``outcomes``)."""
    plt = _plt(out_stem is not None)
    n = max(1, len(outcomes))
    fig, axes = plt.subplots(2, n, figsize=(1.8 * n + 0.6, 3.6), constrained_layout=True, squeeze=False,
                             gridspec_kw={"height_ratios": [1.3, 1]})
    layered = bool(apps["layered"].astype(bool).any()) if "layered" in apps and len(apps) else False
    ref = "pre_tag" if layered else "baseline"
    a = apps.set_index("cell_tag")
    for k, o in enumerate(outcomes):
        vals = pd.DataFrame({ref.replace("_", "-"): a.get(f"{ref}_{o}", np.nan), "drug": a.get(f"drug_{o}", np.nan)})
        vals = vals.apply(pd.to_numeric, errors="coerce")
        est = estimation_plot(axes[0][k], axes[1][k], vals, labels, s, unit="", ylabel=o)
        axes[0][k].set_title(_effect_text(est[0], "", s.effect_size), fontsize=6)
    fig.suptitle(f"{condition}: secondary outcomes (exploratory)", fontsize=8)
    _note(fig, f"{CAVEAT}. Paired difference drug − {ref.replace('_', '-')}: dot = mean, bar = 95 % CI, grey = bootstrap "
               "distribution.", height=0.08)
    return _finish(fig, out_stem, formats)


def fig_examples(examples: dict, labels: pd.DataFrame, step_settings, ramp_settings, out_stem: Path | None = None,
                 formats=("png", "svg"), spont_s: float = 30.0):
    """Example traces. ``examples``: {'silent': row, 'tonic': row, 'steps': row, 'ramp': row}
    (rows of analysis_ready.csv, any may be None)."""
    from .ramp import load_ramp
    from .steps import load_steps

    plt = _plt(out_stem is not None)
    fig = plt.figure(figsize=(7.2, 5.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    for k, kind in enumerate(("silent", "tonic")):
        ax = fig.add_subplot(gs[0, k])
        row = examples.get(kind)
        if row is None:
            ax.text(0.5, 0.5, f"no {kind} cell", ha="center", transform=ax.transAxes)
            ax.axis("off")
            continue
        sc = row.get("voltage_scale")
        t, v, _ = read_voltage(row["original_path"], row.get("voltage_source") or "ADC0",
                               float(sc) if sc is not None and pd.notna(sc) else 1.0)
        from .noise import correct as correct_noise
        v, _ = correct_noise(t, v, row)            # same noise correction as the analysis (if any)
        m = t >= t[-1] - spont_s
        te, ve = envelope(t[m] - t[m][0], v[m], 3000)
        ax.plot(te, ve, color=CLASS_COLOURS[kind], lw=0.4)
        ax.set_title(f"Spontaneous, {kind} ({row['cell_tag']})")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Vm (mV)")
    ax = fig.add_subplot(gs[1, 0])
    row = examples.get("steps")
    if row is not None:
        rec = load_steps(row, step_settings)
        t0 = rec["sweeps"][0][0]
        on, off = rec["on"], rec["off"]
        dt = t0[1] - t0[0]
        a, b = max(0, on - int(0.2 / dt)), min(t0.size, off + int(1.0 / dt))
        order = np.argsort(rec["I_steps"])
        cmap = plt.get_cmap("coolwarm")
        for i, j in enumerate(order):
            t, v = rec["sweeps"][j]
            ax.plot(t[a:b] - t[on], v[a:b], color=cmap(i / max(1, len(order) - 1)), lw=0.5,
                    label=f"{rec['I_steps'][j]:+.0f} pA")
        ax.legend(frameon=False, fontsize=5.5, ncol=2, loc="upper right")
        ax.set_title(f"Current steps ({row['cell_tag']})")
        ax.set_xlabel("Time from step onset (s)")
        ax.set_ylabel("Vm (mV)")
    else:
        ax.axis("off")
    sub = gs[1, 1].subgridspec(2, 1, height_ratios=[4, 1], hspace=0.05)
    ax, ax_c = fig.add_subplot(sub[0]), None
    row = examples.get("ramp")
    if row is not None:
        ax_c = fig.add_subplot(sub[1], sharex=ax)
        rec = load_ramp(row, ramp_settings)
        t, v, cur = rec["sweeps"][0]
        te, ve = envelope(t, v, 3000)
        ax.plot(te, ve, color="k", lw=0.4)
        ax.set_title(f"Ramp, sweep 1 ({row['cell_tag']})")
        ax.set_ylabel("Vm (mV)")
        ax.tick_params(labelbottom=False)
        ax_c.plot(t, cur, color="0.4", lw=0.7)
        ax_c.set_ylabel("I (pA)")
        ax_c.set_xlabel("Time (s)")
    else:
        ax.axis("off")
    _note(fig, ljp_note(), height=0.035)
    return _finish(fig, out_stem, formats)


# --------------------------------------------------------------------------- orchestration
def load_inputs(cfg: dict, repo_path) -> dict:
    """Read every input table named in the 10 config (missing files -> empty tables)."""
    def read(key, **kw):
        p = repo_path(cfg, cfg["inputs"][key])
        return pd.read_csv(p, **kw) if Path(p).exists() else pd.DataFrame()
    return {"summary": read("cell_summary", dtype={"cell_tag": "string"}),
            "selection": read("selection", dtype={"cell_tag": "string"}),
            "ready": read("analysis_ready", dtype={"cell_tag": "string"}),
            "step_sweeps": read("step_sweeps"), "waves": read("ap_waveforms"),
            "tc_summary": read("timecourse_summary", dtype={"cell_tag": "string"}),
            "tc_bins": read("timecourse_bins")}


def recordings_of(selection: pd.DataFrame, protocol: str) -> dict:
    """{recording_id: cell_tag} of the recordings 08 selected for ``protocol``."""
    if selection.empty:
        return {}
    s = selection[(selection["protocol"] == protocol) & (selection["recording_id"].fillna("") != "")]
    return dict(zip(s["recording_id"], s["cell_tag"]))


def example_rows(cfg_examples: dict, labels: pd.DataFrame, selection: pd.DataFrame, ready: pd.DataFrame) -> dict:
    """analysis_ready rows of the example recordings. ``cfg_examples``: {silent, tonic, steps,
    ramp: cell_tag or 'auto'}; auto = the first silent / tonic cell (cell-tag order); steps and
    ramp default to the silent example (else the tonic one)."""
    order = sorted(labels.index, key=str)
    pick = {}
    for kind in ("silent", "tonic"):
        want = (cfg_examples or {}).get(kind, "auto")
        cands = [c for c in order if labels.loc[c, "firing_class"] == kind]
        pick[kind] = cands[0] if want in (None, "auto") and cands else (None if want in (None, "auto") else want)
    for kind in ("steps", "ramp"):
        want = (cfg_examples or {}).get(kind, "auto")
        pick[kind] = (pick["silent"] or pick["tonic"]) if want in (None, "auto") else want
    proto = {"silent": "baseline", "tonic": "baseline", "steps": "steps", "ramp": "ramp"}
    out = {}
    for kind, cell in pick.items():
        rid = {v: k for k, v in recordings_of(selection, proto[kind]).items()}.get(cell)
        r = ready[ready["recording_id"] == rid] if rid is not None and len(ready) else ready.iloc[0:0]
        out[kind] = r.iloc[0].to_dict() if len(r) else None
    return out
