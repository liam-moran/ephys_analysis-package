"""Provisional paired drug statistics (drug_stats.py) on synthetic data with known answers."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from ephys_analysis import drug_stats as ds

S = ds.DrugStatsSettings(n_boot=2000)


def test_holm_known_values():
    p = ds.holm([0.01, 0.04, 0.03, np.nan])
    assert p[0] == pytest.approx(0.03) and p[2] == pytest.approx(0.06) and p[1] == pytest.approx(0.06)
    assert np.isnan(p[3])


def test_paired_shift_and_no_effect():
    rng = np.random.default_rng(1)
    base = rng.normal(-50, 3, 12)
    r = ds.paired(base, base - 10 + rng.normal(0, 1, 12), S)
    assert r["n"] == 12 and r["mean_diff"] == pytest.approx(-10, abs=1) and r["p_wilcoxon"] < 0.001
    assert r["ci_low"] < r["mean_diff"] < r["ci_high"] < 0 and r["dz"] < -3
    r0 = ds.paired(base, base + rng.normal(0, 1, 12), S)
    assert r0["ci_low"] < 0 < r0["ci_high"] and r0["p_wilcoxon"] > 0.05
    few = ds.paired([1, 2, 3], [2, 3, 4], S)
    assert np.isnan(few["p_wilcoxon"]) and "too few cells" in few["note"]


def test_friedman_and_mcnemar_and_ci():
    b = np.arange(8.0)
    fr = ds.friedman(b, b - 5, b - 1, s=S)
    assert fr["n"] == 8 and fr["p"] < 0.01
    mc = ds.mcnemar_exact(["silent"] * 6 + ["tonic"] * 2, ["tonic"] * 6 + ["tonic"] * 2)
    assert mc["became_tonic"] == 6 and mc["stopped_tonic"] == 0
    assert mc["p"] == pytest.approx(stats.binomtest(6, 6, 0.5).pvalue)            # 2 x 0.5^6
    lo, hi = ds.proportion_ci(7, 10)
    assert lo == pytest.approx(0.3475, abs=1e-3) and hi == pytest.approx(0.9333, abs=1e-3)


def test_excursion_is_signed_largest():
    vm = pd.Series([-50, -50, -58, -58, -58, -50, -49])
    assert ds.excursion(vm, -50.0, 1) == pytest.approx(-8)
    assert ds.excursion(vm, -50.0, 3) < 0


def _apps(n=8, shift=-8.0, seed=0):
    rng = np.random.default_rng(seed)
    rows, bins = [], []
    for i in range(n):
        base = -50 + rng.normal(0, 2)
        resp = shift + rng.normal(0, 1) if shift else 0.0          # shift 0: a true null (no response)
        rid = f"r{i}"
        rows.append({"recording_id": rid, "cell_tag": f"C{i}", "abf_filename": f"C{i}_0004.abf", "application": 1,
                     "condition": "5-HT", "layered": False, "flags": "unstable_baseline=+3.0mV/min" if i == 0 else "",
                     "drug_tag_s": 150.0, "baseline_start_s": 50.0, "drug_start_s": 210.0, "drug_end_s": 410.0,
                     "baseline_vm_baseline_mV": base, "drug_vm_baseline_mV": base + resp,
                     "washout_vm_baseline_mV": base + 0.5, "baseline_vm_sd_mV": 0.5, "baseline_drift_mV_per_min": 0.2,
                     "baseline_firing_class": "tonic" if i % 2 else "silent", "drug_firing_class": "silent",
                     "baseline_firing_rate_Hz": 0.5, "drug_firing_rate_Hz": 0.1})
        t = np.arange(0, 600, 10.0)
        vm = base + rng.normal(0, 0.3, t.size) + np.where((t >= 160) & (t < 500), resp, 0.0)
        bins.append(pd.DataFrame({"recording_id": rid, "bin_start_s": t, "bin_duration_s": 10.0, "vm_median_mV": vm}))
    return pd.DataFrame(rows), pd.concat(bins, ignore_index=True)


def test_condition_stats_end_to_end():
    apps, bins = _apps()
    cells = pd.DataFrame({"cell_tag": [f"C{i}" for i in range(8)], "Animal ID": ["a1"] * 4 + ["a2"] * 4})
    st, resp = ds.drug_statistics(apps, bins, S, cells=cells)
    assert (st["caveat"] == ds.CAVEAT).all() and (resp["caveat"] == ds.CAVEAT).all()
    prim = st[st["role"] == "primary"].iloc[0]
    assert prim["n"] == 7 and prim["n_cells_excluded"] == 1 and "C0" in prim["excluded_cells"]
    assert prim["mean_diff"] == pytest.approx(-8, abs=1) and prim["p_wilcoxon"] < 0.05
    assert prim["n_animals"] == 2
    peak = st[st["role"] == "peak"].iloc[0]
    assert peak["mean_diff"] > 5 and peak["p_wilcoxon"] < 0.05             # drug excursion >> sham excursion
    assert (resp["responder"] == True).sum() == 7                           # noqa: E712
    r = st[st["role"] == "responders"].iloc[0]
    assert r["statistic"] == 7 and r["n"] == 7


def test_sham_matches_peak_without_effect():
    apps, bins = _apps(shift=0.0, seed=3)
    st, _ = ds.drug_statistics(apps, bins, S)
    peak = st[st["role"] == "peak"].iloc[0]
    assert abs(peak["mean_diff"]) < 1 and peak["p_wilcoxon"] > 0.05


def test_too_few_cells_gives_no_p():
    apps, bins = _apps(n=3)
    st, _ = ds.drug_statistics(apps, bins, S)
    prim = st[st["role"] == "primary"].iloc[0]
    assert np.isnan(prim["p_wilcoxon"]) and "too few cells" in prim["note"]


def test_estimate_bca_ci_and_reproducible():
    rng = np.random.default_rng(7)
    base = rng.normal(-50, 3, 10)
    e = ds.estimate(base, base - 8 + rng.normal(0, 1.5, 10), S)
    assert e["ci_method"] == "BCa" and e["ci_low"] < -8 < e["ci_high"] < 0
    assert e["median_ci_low"] <= e["median_diff"] <= e["median_ci_high"] and e["g_ci_high"] < 0
    assert e["boot_mean_diff"].size == S.n_boot
    y = base - 8 + np.random.default_rng(8).normal(0, 1.5, 10)
    first, second = ds.estimate(base, y, S), ds.estimate(base, y, S)
    assert first["ci_low"] == second["ci_low"] and first["ci_high"] == second["ci_high"]   # same seed, same CI
    noise = rng.normal(0, 1.5, 10)
    null = ds.estimate(base, base + noise - noise.mean(), S)          # mean difference exactly 0
    assert null["ci_low"] < 0 < null["ci_high"]


def test_hedges_g_av_hand_calculation():
    a, b = np.array([1, 2, 3, 4, 5.0]), np.array([2, 4, 5, 7, 7.0])
    # mean diff 2; SD_av = sqrt((2.5 + 4.5) / 2); correction 1 - 3 / (8 * 5 - 9)
    expected = 2 / np.sqrt(3.5) * (1 - 3 / 31)
    assert ds.estimate(a, b, S)["hedges_g"] == pytest.approx(expected)


def test_estimate_degenerate_and_too_few():
    a = np.arange(8.0)
    e = ds.estimate(a, a + 2, S)                                       # identical differences
    assert e["mean_diff"] == 2 and "percentile" in e["note"] and e["ci_low"] == e["ci_high"] == 2
    few = ds.estimate([1, 2], [2, 3], S)
    assert np.isnan(few["ci_low"]) and "too few cells" in few["note"]
