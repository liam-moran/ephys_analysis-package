"""Paired, within-cell statistics of drug applications (10). PROVISIONAL.

Every number this module produces carries ``CAVEAT``: the tests and rules are pre-specified
defaults (10_ephys_figures/config.yaml, ``drug_stats``) that must be reviewed before anything
is reported. Explainer: 10_ephys_figures/README.md, "Drug statistics: what and why".

Design: each cell is its own control (baseline -> drug -> washout), one application per cell
and condition (the 08 selection). Per condition:
    primary     drug vs reference (baseline; for a layered application its pre-tag period) of
                the primary outcome (default Vm). ESTIMATION first (``estimate``, as DABEST, Ho
                et al. 2019): mean and median paired difference and Hedges' g_av, each with a
                BCa bootstrap 95 % CI, and dz; then, secondary, Wilcoxon signed-rank and paired t
    periods     Friedman over baseline / drug / washout, then pairwise Wilcoxon (drug vs
                baseline, washout vs baseline), Holm-corrected
    secondary   drug vs reference for each secondary outcome, Wilcoxon, Holm-corrected
                across the secondary set
    firing      silent <-> tonic changes: exact McNemar test on the discordant cells
    responders  per cell: |delta Vm| > k x baseline Vm SD and |baseline drift| below the
                limit; count with a Clopper-Pearson 95 % CI
    peak        the peak (largest smoothed excursion from the baseline Vm) is NOT tested
                against 0 (it is never 0). It is compared with a SHAM peak: the same
                procedure on the bins just before the tag, over the same length of time as
                the peak search window after the drug period start (paired Wilcoxon on
                the absolute excursions)
    pairs       optional: two conditions in the same cells (config ``condition_pairs``),
                paired Wilcoxon of their responses
With fewer than ``min_cells`` cells no p-value is computed ("too few cells").
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields

import numpy as np
import pandas as pd
from scipy import stats

CAVEAT = "PROVISIONAL: pre-specified defaults, to be reviewed before reporting"


@dataclass
class DrugStatsSettings:
    primary: str = "vm_baseline_mV"
    secondary: tuple = ("firing_rate_Hz", "isi_cv", "ap_threshold_mV", "ap_half_width_ms", "ap_ahp_mV")
    alpha: float = 0.05
    n_boot: int = 10000
    seed: int = 0
    ci_method: str = "BCa"                 # bootstrap CI: BCa (as DABEST) or percentile
    effect_size: str = "mean_diff"         # headline effect in the figures: mean_diff / median_diff / hedges_g
    estimation_secondary: bool = False     # fig7: one estimation plot per secondary outcome
    min_cells: int = 5
    responder_sd_factor: float = 3.0
    responder_max_drift_mV_per_min: float = 2.0
    exclude_flags: tuple = ("unstable_baseline",)
    smooth_bins: int = 3
    condition_pairs: list = field(default_factory=list)
    animal_column: str = "Animal ID"       # 08 cell-summary column identifying the animal

    @classmethod
    def from_config(cls, settings: dict | None) -> "DrugStatsSettings":
        names = {f.name for f in fields(cls)}
        kw = {k: (tuple(v) if isinstance(v, list) and k != "condition_pairs" else v)
              for k, v in (settings or {}).items() if k in names}
        return cls(**kw)


# --------------------------------------------------------------------------- building blocks
def holm(p) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (NaN kept as NaN)."""
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    if ok.size == 0:
        return out
    order = ok[np.argsort(p[ok])]
    m = order.size
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        out[i] = running
    return out


def _mean_diff(a, b, axis=-1):
    return np.mean(b - a, axis=axis)


def _median_diff(a, b, axis=-1):
    return np.median(b - a, axis=axis)


def _hedges_g_av(a, b, axis=-1):
    """Hedges' g_av for paired data (Lakens 2013): mean difference / average SD of the two
    conditions, x (1 - 3 / (8 n - 9)) small-sample correction."""
    n = np.shape(a)[axis]
    s_av = np.sqrt((np.var(a, ddof=1, axis=axis) + np.var(b, ddof=1, axis=axis)) / 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.mean(b - a, axis=axis) / s_av * (1 - 3 / (8 * n - 9))


def estimate(x, y, s: DrugStatsSettings) -> dict:
    """Estimation statistics of the paired difference y - x (cells with both values), as in
    DABEST (Ho et al. 2019), without the dependency:
        mean_diff, ci_low, ci_high          mean difference and its 95 % CI
        median_diff, median_ci_low / high   median difference and its 95 % CI
        hedges_g, g_ci_low / high           Hedges' g_av (paired; Lakens 2013) and its 95 % CI
        dz                                  mean difference / SD of the differences
        ci_method                           'BCa' (bias-corrected and accelerated bootstrap,
                                            Efron 1987; scipy.stats.bootstrap, paired) or
                                            'percentile' (fallback when BCa is degenerate)
        boot_mean_diff                      bootstrap distribution of the mean difference (plot)
    ``s.n_boot`` resamples, seed ``s.seed`` (reproducible)."""
    import warnings

    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    d = y - x
    keys = ["mean_diff", "ci_low", "ci_high", "median_diff", "median_ci_low", "median_ci_high", "hedges_g",
            "g_ci_low", "g_ci_high", "dz"]
    out = {k: np.nan for k in keys}
    out.update(n=int(d.size), ci_method="", boot_mean_diff=np.array([]), note="")
    if d.size:
        out.update(mean_diff=float(d.mean()), median_diff=float(np.median(d)))
    if d.size < s.min_cells:
        out["note"] = f"too few cells (n = {d.size} < {s.min_cells})"
        return out
    sd = float(d.std(ddof=1))
    out["dz"] = float(d.mean() / sd) if sd > 0 else np.nan
    out["hedges_g"] = float(_hedges_g_av(x, y))
    method = getattr(s, "ci_method", "BCa")
    notes = []
    for name, fn, lo, hi in (("mean", _mean_diff, "ci_low", "ci_high"),
                             ("median", _median_diff, "median_ci_low", "median_ci_high"),
                             ("g", _hedges_g_av, "g_ci_low", "g_ci_high")):
        res, used = None, method
        for m in ([method, "percentile"] if method != "percentile" else ["percentile"]):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    r = stats.bootstrap((x, y), fn, paired=True, vectorized=True, n_resamples=s.n_boot,
                                        method=m, rng=np.random.default_rng(s.seed))
                except Exception:   # noqa: BLE001 - degenerate data: try the next method
                    continue
            if np.isfinite(r.confidence_interval.low) and np.isfinite(r.confidence_interval.high):
                res, used = r, m
                break
        if res is None:
            notes.append(f"no CI for the {name} (degenerate data)")
            continue
        if used != method:
            notes.append(f"{name}: {method} degenerate, percentile CI used")
        out[lo], out[hi] = float(res.confidence_interval.low), float(res.confidence_interval.high)
        if name == "mean":
            out["ci_method"] = used
            out["boot_mean_diff"] = np.asarray(res.bootstrap_distribution, dtype=float)
    out["note"] = "; ".join(notes)
    return out


def paired(x, y, s: DrugStatsSettings) -> dict:
    """Paired comparison y - x: the estimation statistics of ``estimate`` (primary: effect size
    with its 95 % CI) plus, secondary, the Wilcoxon signed-rank and paired t-tests."""
    est = estimate(x, y, s)
    out = {k: v for k, v in est.items() if k != "boot_mean_diff"}
    out.update(wilcoxon_W=np.nan, p_wilcoxon=np.nan, t=np.nan, p_t=np.nan)
    if est["n"] < s.min_cells:
        return out
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    d = y[ok] - x[ok]
    sd = float(d.std(ddof=1))
    def note(msg):
        out["note"] = "; ".join(x for x in (out["note"], msg) if x)

    if np.all(d == 0):
        note("all differences are 0")
        return out
    w = stats.wilcoxon(d)
    out.update(wilcoxon_W=float(w.statistic), p_wilcoxon=float(w.pvalue))
    if sd > 1e-9 * max(1.0, abs(float(d.mean()))):
        tt = stats.ttest_1samp(d, 0.0)
        out.update(t=float(tt.statistic), p_t=float(tt.pvalue))
    else:
        note("identical differences: no t-test")
    return out


def friedman(*periods, s: DrugStatsSettings) -> dict:
    """Friedman test over repeated periods (cells with all values)."""
    a = np.column_stack([np.asarray(p, dtype=float) for p in periods])
    a = a[np.all(np.isfinite(a), axis=1)]
    out = {"n": int(a.shape[0]), "chi2": np.nan, "p": np.nan, "note": ""}
    if a.shape[0] < s.min_cells:
        out["note"] = f"too few cells (n = {a.shape[0]} < {s.min_cells})"
        return out
    r = stats.friedmanchisquare(*a.T)
    out.update(chi2=float(r.statistic), p=float(r.pvalue))
    return out


def mcnemar_exact(before, after, positive: str = "tonic") -> dict:
    """Exact McNemar test of a paired yes/no state (e.g. silent / tonic) before and after:
    binomial test on the discordant cells (b: no -> yes, c: yes -> no)."""
    pairs = [(a, b) for a, b in zip(before, after) if isinstance(a, str) and isinstance(b, str) and a and b]
    b = sum(a != positive and z == positive for a, z in pairs)
    c = sum(a == positive and z != positive for a, z in pairs)
    p = float(stats.binomtest(b, b + c, 0.5).pvalue) if b + c else np.nan
    return {"n": len(pairs), "became_" + positive: int(b), "stopped_" + positive: int(c), "p": p}


def proportion_ci(k: int, n: int) -> tuple[float, float]:
    """Clopper-Pearson (exact) 95 % CI of k / n."""
    if n == 0:
        return np.nan, np.nan
    ci = stats.binomtest(k, n).proportion_ci(confidence_level=0.95, method="exact")
    return float(ci.low), float(ci.high)


def excursion(vm: pd.Series, base: float, k: int) -> float:
    """Signed largest excursion of the smoothed (running mean over k bins) Vm from ``base``."""
    v = vm.rolling(max(1, k), center=True, min_periods=1).mean().dropna()
    if len(v) == 0 or not np.isfinite(base):
        return np.nan
    dev = v - base
    return float(dev.iloc[int(np.argmax(np.abs(dev.to_numpy())))])


def peak_and_sham(bins: pd.DataFrame, row: pd.Series, s: DrugStatsSettings) -> tuple[float, float]:
    """(drug peak, sham peak) of one application, over windows of equal length L: the sham in
    the L s before the tag (L = available pre-tag time, at most the drug period), the drug
    peak in the L s from the start of the drug period; both relative to the baseline Vm."""
    tag, d0, d1 = row.get("drug_tag_s"), row.get("drug_start_s"), row.get("drug_end_s")
    b0, base = row.get("baseline_start_s"), row.get("baseline_vm_baseline_mV")
    if not all(np.isfinite(x) for x in (tag, d0, d1, b0, base)) or len(bins) == 0:
        return np.nan, np.nan
    L = min(tag - b0, d1 - d0)
    if L <= 0:
        return np.nan, np.nan
    start = bins["bin_start_s"]
    end = start + bins["bin_duration_s"]
    sham = bins.loc[(start >= tag - L - 1e-9) & (end <= tag + 1e-9), "vm_median_mV"]
    drug = bins.loc[(start >= d0 - 1e-9) & (end <= d0 + L + 1e-9), "vm_median_mV"]
    return excursion(drug, base, s.smooth_bins), excursion(sham, base, s.smooth_bins)


def peak_sham_table(apps: pd.DataFrame, bins: pd.DataFrame, s: DrugStatsSettings) -> pd.DataFrame:
    """Per application: cell_tag, peak and sham excursion (mV, signed; see ``peak_and_sham``)."""
    rows = []
    for _, a in apps.iterrows():
        p, q = peak_and_sham(bins[bins["recording_id"] == a["recording_id"]], a, s)
        rows.append({"cell_tag": a["cell_tag"], "peak": p, "sham": q})
    return pd.DataFrame(rows, columns=["cell_tag", "peak", "sham"])


# --------------------------------------------------------------------------- per condition
def select_applications(tc_summary: pd.DataFrame, selection: pd.DataFrame | None) -> pd.DataFrame:
    """One application per cell and condition: the first application of the condition in
    the recording that 08 selected (``timecourse:<condition>`` rows of selection.csv); without
    a selection, the first recording in file order."""
    t = tc_summary[tc_summary["condition"].fillna("").astype(str) != ""]
    t = t.sort_values(["abf_filename", "application"]).drop_duplicates(["cell_tag", "condition", "abf_filename"])
    if selection is not None and len(selection):
        sel = selection[selection["protocol"].astype(str).str.startswith("timecourse:")].copy()
        sel["condition"] = sel["protocol"].str.slice(len("timecourse:"))
        t = t.merge(sel[["cell_tag", "condition", "abf_filename"]], on=["cell_tag", "condition", "abf_filename"])
    return t.drop_duplicates(["cell_tag", "condition"]).reset_index(drop=True)


def _excluded(df: pd.DataFrame, s: DrugStatsSettings) -> pd.Series:
    flags = df["flags"].fillna("").astype(str) if "flags" in df else pd.Series("", index=df.index)
    return flags.apply(lambda f: any(x and x in f for x in s.exclude_flags))


def condition_stats(apps: pd.DataFrame, bins: pd.DataFrame, s: DrugStatsSettings,
                    cells: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(stats rows, per-cell responder table) for the applications of ONE condition."""
    cond = str(apps["condition"].iloc[0]) if len(apps) else ""
    excl = _excluded(apps, s)
    used = apps[~excl]
    excluded_cells = "; ".join(f"{c} ({f})" for c, f in zip(apps.loc[excl, "cell_tag"], apps.loc[excl, "flags"]))
    layered = bool(used["layered"].astype(bool).any()) if "layered" in used and len(used) else False
    ref = "pre_tag" if layered else "baseline"
    n_animals, per_animal = np.nan, ""
    if cells is not None and s.animal_column in cells and len(used):
        a = cells.set_index("cell_tag").reindex(used["cell_tag"])[s.animal_column]
        n_animals = int(a.nunique())
        per_animal = "; ".join(f"{k}: {v}" for k, v in a.value_counts().items())
    common = {"condition": cond, "reference": ref, "n_cells_excluded": int(excl.sum()),
              "excluded_cells": excluded_cells, "n_animals": n_animals, "cells_per_animal": per_animal,
              "caveat": CAVEAT}
    rows = []

    def col(period, outcome):
        c = f"{period}_{outcome}"
        return used[c] if c in used else pd.Series(np.nan, index=used.index)

    # primary
    r = paired(col(ref, s.primary), col("drug", s.primary), s)
    rows.append({**common, "role": "primary", "outcome": s.primary, "comparison": f"drug - {ref}", **r,
                 "p_holm": r["p_wilcoxon"]})
    # baseline / drug / washout
    fr = friedman(col("baseline", s.primary), col("drug", s.primary), col("washout", s.primary), s=s)
    rows.append({**common, "role": "periods", "outcome": s.primary, "comparison": "Friedman baseline / drug / washout",
                 "n": fr["n"], "statistic": fr["chi2"], "p_wilcoxon": fr["p"], "p_holm": fr["p"], "note": fr["note"]})
    pw = [("drug - baseline", paired(col("baseline", s.primary), col("drug", s.primary), s)),
          ("washout - baseline", paired(col("baseline", s.primary), col("washout", s.primary), s))]
    ph = holm([x["p_wilcoxon"] for _, x in pw])
    for (name, x), p_adj in zip(pw, ph):
        rows.append({**common, "role": "periods", "outcome": s.primary, "comparison": name, **x, "p_holm": p_adj})
    # secondary
    sec = [(o, paired(col(ref, o), col("drug", o), s)) for o in s.secondary]
    ph = holm([x["p_wilcoxon"] for _, x in sec])
    for (o, x), p_adj in zip(sec, ph):
        rows.append({**common, "role": "secondary", "outcome": o, "comparison": f"drug - {ref}", **x, "p_holm": p_adj})
    # firing state
    mc = mcnemar_exact(col(ref, "firing_class").tolist(), col("drug", "firing_class").tolist())
    rows.append({**common, "role": "secondary", "outcome": "firing_class", "comparison": f"silent/tonic, {ref} -> drug",
                 "n": mc["n"], "statistic": mc["became_tonic"] - mc["stopped_tonic"], "p_wilcoxon": mc["p"],
                 "p_holm": mc["p"], "note": f"exact McNemar: {mc['became_tonic']} became tonic, "
                                            f"{mc['stopped_tonic']} became silent"})
    # peak vs sham
    ps = peak_sham_table(used, bins, s)
    r = paired(ps["sham"].abs(), ps["peak"].abs(), s)
    rows.append({**common, "role": "peak", "outcome": "|peak excursion| (mV)", "comparison": "drug peak - sham peak",
                 **r, "p_holm": r["p_wilcoxon"],
                 "note": (r["note"] + "; " if r["note"] else "") + "equal-length windows; peak never tested against 0"})
    # responders
    d = col("drug", s.primary) - col(ref, s.primary)
    sd = col(ref, "vm_sd_mV") if f"{ref}_vm_sd_mV" in used else col("baseline", "vm_sd_mV")
    drift = used["baseline_drift_mV_per_min"] if "baseline_drift_mV_per_min" in used else pd.Series(np.nan, index=used.index)
    resp = pd.DataFrame({"cell_tag": used["cell_tag"], "abf_filename": used["abf_filename"], "condition": cond,
                         "delta_mV": d, "threshold_mV": s.responder_sd_factor * sd, "baseline_drift_mV_per_min": drift})
    stable = resp["baseline_drift_mV_per_min"].abs() < s.responder_max_drift_mV_per_min
    resp["responder"] = ((resp["delta_mV"].abs() > resp["threshold_mV"]) & stable).astype(object)
    resp.loc[resp["delta_mV"].isna() | resp["threshold_mV"].isna(), "responder"] = np.nan
    resp["direction"] = np.where(resp["responder"] == True, np.where(resp["delta_mV"] < 0, "hyperpolarising",  # noqa: E712
                                                                      "depolarising"), "")
    resp["caveat"] = CAVEAT
    k, n = int((resp["responder"] == True).sum()), int(resp["responder"].notna().sum())  # noqa: E712
    lo, hi = proportion_ci(k, n)
    rows.append({**common, "role": "responders", "outcome": s.primary,
                 "comparison": f"|delta| > {s.responder_sd_factor:g} x baseline SD, |drift| < "
                               f"{s.responder_max_drift_mV_per_min:g} mV/min",
                 "n": n, "statistic": k, "mean_diff": k / n if n else np.nan, "ci_low": lo, "ci_high": hi,
                 "note": f"{k} of {n} cells; Clopper-Pearson 95 % CI of the proportion"})
    return pd.DataFrame(rows), resp


def pair_stats(apps: pd.DataFrame, pairs: list, s: DrugStatsSettings) -> pd.DataFrame:
    """Paired comparison of the primary response between two conditions in the same cells."""
    rows = []
    for a, b in pairs or []:
        resp = {}
        for c in (a, b):
            x = apps[apps["condition"] == c]
            lay = x["layered"].astype(bool) if "layered" in x else pd.Series(False, index=x.index)
            ref = np.where(lay, x.get(f"pre_tag_{s.primary}", np.nan), x.get(f"baseline_{s.primary}", np.nan))
            resp[c] = pd.Series(x[f"drug_{s.primary}"].to_numpy() - ref, index=x["cell_tag"])
        both = resp[a].index.intersection(resp[b].index)
        r = paired(resp[a].reindex(both), resp[b].reindex(both), s)
        rows.append({"condition": f"{b} vs {a}", "reference": "own reference per condition", "role": "pair",
                     "outcome": s.primary, "comparison": f"response({b}) - response({a})", **r,
                     "p_holm": r["p_wilcoxon"], "caveat": CAVEAT})
    return pd.DataFrame(rows)


def drug_statistics(tc_summary: pd.DataFrame, bins: pd.DataFrame, s: DrugStatsSettings,
                    selection: pd.DataFrame | None = None,
                    cells: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(drug_stats: one row per condition x outcome x comparison; responders: one row per
    cell and condition). See the module docstring; every row carries ``CAVEAT``."""
    apps = select_applications(tc_summary, selection)
    out, resp = [], []
    for cond, g in apps.groupby("condition", sort=True):
        st, rp = condition_stats(g, bins, s, cells)
        out.append(st)
        resp.append(rp)
    if s.condition_pairs:
        out.append(pair_stats(apps[~_excluded(apps, s)], s.condition_pairs, s))
    stats_df = pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["condition", "caveat"])
    resp_df = pd.concat(resp, ignore_index=True) if resp else pd.DataFrame(columns=["cell_tag", "condition"])
    return stats_df, resp_df
