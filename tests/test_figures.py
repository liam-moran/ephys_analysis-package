"""Publication figures (figures.py): tables, helpers and that every figure is written."""

import numpy as np
import pandas as pd
import pytest

from ephys_analysis import figures as fg
from ephys_analysis.cell_overview import cell_labels
from ephys_analysis.drug_stats import DrugStatsSettings, drug_statistics, peak_sham_table, select_applications

RULES = {"firing_class": "base_firing_class", "block_if_not_missing": "step_first_step_with_block_pA",
         "rebound_if_positive": "ramp_n_rebound_spikes_short"}
PROPS = [{"column": "base_vm_baseline_mV", "label": "Vm (mV)", "unit": "mV"},
         {"column": "base_firing_rate_Hz", "label": "Rate (Hz)", "unit": "Hz", "log": True}]
AP = [{"column": "ramp_ap_threshold_mV", "label": "Threshold (mV)", "unit": "mV"}]


def synthetic(n=12, seed=0):
    """Synthetic outputs of 05-08 for n cells (cell summary, selection, step sweeps, AP
    waveforms, timecourse summary and bins)."""
    rng = np.random.default_rng(seed)
    cells = [f"C{i:02d}" for i in range(n)]
    tonic = rng.random(n) < 0.4
    summary = pd.DataFrame({
        "cell_tag": cells, "Animal ID": [f"a{i // 3}" for i in range(n)],
        "base_firing_class": np.where(tonic, "tonic", "silent"),
        "base_firing_rate_Hz": np.where(tonic, rng.uniform(0.2, 3, n), rng.uniform(0, 0.08, n)),
        "base_vm_baseline_mV": rng.normal(-55, 4, n), "step_sag_ratio": rng.uniform(0, 0.2, n),
        "step_first_step_with_block_pA": np.where(rng.random(n) < 0.4, 50.0, np.nan),
        "ramp_n_rebound_spikes_short": rng.integers(0, 6, n), "ramp_ap_threshold_mV": rng.normal(-35, 3, n),
        "step_recording": [f"{c}_steps" for c in cells]})
    sel = pd.DataFrame([{"cell_tag": c, "protocol": p, "recording_id": f"{c}_{p}", "abf_filename": f"{c}_{p}.abf"}
                        for c in cells for p in ("steps", "ramp", "timecourse:5-HT")])
    sel.loc[sel["protocol"] == "timecourse:5-HT", "recording_id"] = [f"{c}_tc" for c in cells]
    sel.loc[sel["protocol"] == "timecourse:5-HT", "abf_filename"] = [f"{c}_tc.abf" for c in cells]
    steps = pd.DataFrame([{"recording_id": f"{c}_steps", "I_step_pA": I, "rate_step_Hz": max(0, (I - 20) * 0.15),
                           "n_attenuated_step": 0} for c in cells for I in (-150, -100, -50, 0, 50, 100)])
    t = np.arange(-50, 151) * 0.1
    w = []
    for c in cells:
        for sw in range(2):
            v = -40 + 85 * np.exp(-((t - 1.2) / 0.8) ** 2) - 12 * (t > 2) * np.exp(-(t - 2) / 8)
            w.append(pd.DataFrame({"recording_id": f"{c}_ramp", "sweep": sw, "t_ms": t, "v_mV": v,
                                   "dvdt_mV_ms": np.gradient(v, 0.1)}))
    waves = pd.concat(w, ignore_index=True)
    tcs, bins = [], []
    for i, c in enumerate(cells):
        base, resp = summary.loc[i, "base_vm_baseline_mV"], rng.normal(-9, 2)
        tcs.append({"recording_id": f"{c}_tc", "cell_tag": c, "abf_filename": f"{c}_tc.abf", "application": 1,
                    "condition": "5-HT", "layered": False, "flags": "", "drug_tag_s": 150.0, "baseline_start_s": 50.0,
                    "drug_start_s": 210.0, "drug_end_s": 410.0, "baseline_vm_baseline_mV": base,
                    "drug_vm_baseline_mV": base + resp, "washout_vm_baseline_mV": base + resp / 4,
                    "baseline_vm_sd_mV": 0.6, "baseline_drift_mV_per_min": 0.3, "baseline_firing_class": "silent",
                    "drug_firing_class": "silent"})
        tb = np.arange(0, 1200, 10.0)
        x = np.clip((tb + 5 - 150) / 90, 0, None)
        bins.append(pd.DataFrame({"recording_id": f"{c}_tc", "bin_start_s": tb, "bin_duration_s": 10.0,
                                  "vm_median_mV": base + resp * x * np.exp(1 - x) + rng.normal(0, 0.3, tb.size)}))
    return summary, sel, steps, waves, pd.DataFrame(tcs), pd.concat(bins, ignore_index=True)


def test_summary_table_known_values():
    df = pd.DataFrame({"cell_tag": list("abcd"), "x": [1.0, 2.0, 3.0, np.nan]})
    t = fg.summary_table(df, [{"column": "x", "label": "X", "unit": "u"}, {"column": "y", "label": "Y"}])
    r = t.iloc[0]
    assert r["n"] == 3 and r["median"] == 2 and r["q1"] == 1.5 and r["q3"] == 2.5 and r["sd"] == pytest.approx(1)
    assert t.iloc[1]["n"] == 0 and "junction potential" in r["note"]


def test_cell_mean_waveforms_and_alignment():
    summary, sel, steps, waves, tcs, bins = synthetic(3)
    cw = fg.cell_mean_waveforms(waves, fg.recordings_of(sel, "ramp"))
    assert cw["cell_tag"].nunique() == 3 and len(cw) == 3 * 201
    tcx = fg.aligned_timecourse(tcs, bins)
    near0 = tcx[(tcx["t_s"] >= -100) & (tcx["t_s"] <= -20)]["dvm_mV"].abs()
    assert near0.max() < 1.5                                            # baseline-subtracted before the tag
    assert tcx.groupby("cell_tag")["dvm_mV"].min().max() < -5            # each cell hyperpolarises after it


@pytest.mark.parametrize("n", [12, 2])
def test_every_figure_is_written(tmp_path, n):
    summary, sel, steps, waves, tcs, bins = synthetic(n)
    labels = cell_labels(summary, RULES)
    fg.fig_labels(labels, tmp_path / "f2")
    fg.fig_properties(summary, labels, PROPS, tmp_path / "f3")
    fg.fig_ap(waves, summary, labels, fg.recordings_of(sel, "ramp"), AP, tmp_path / "f4")
    fg.fig_excitability(steps, summary, labels, fg.recordings_of(sel, "steps"), tmp_path / "f5")
    s = DrugStatsSettings(n_boot=500)
    st, _ = drug_statistics(tcs, bins, s, sel, summary)
    apps = select_applications(tcs, sel)
    fg.fig_drug("5-HT", apps, bins, labels, st, (60, 260), tmp_path / "f6", sham=peak_sham_table(apps, bins, s))
    for f in ("f2", "f3", "f4", "f5", "f6"):
        assert (tmp_path / f"{f}.png").exists() and (tmp_path / f"{f}.svg").exists()


def test_examples_on_real_files(example_abf, tmp_path):
    from ephys_analysis.ramp import RampSettings
    from ephys_analysis.steps import StepSettings
    rows = {"silent": {"original_path": str(example_abf("20261026_7503_A_0000.abf")), "cell_tag": "A",
                       "voltage_source": "ADC0", "voltage_scale": 1.0},
            "tonic": None,
            "steps": {"recording_id": "s", "cell_tag": "A", "abf_filename": "x", "original_path":
                      str(example_abf("20261026_7503_A_0001.abf")), "voltage_source": "ADC0", "voltage_scale": 1.0,
                      "current_source": "command_DAC1", "bias_current_pA": np.nan},
            "ramp": {"recording_id": "r", "cell_tag": "A", "abf_filename": "y", "original_path":
                     str(example_abf("20261026_7503_A_0002.abf")), "voltage_source": "ADC0", "voltage_scale": 1.0,
                     "current_source": "command_DAC1", "bias_current_pA": np.nan}}
    labels = pd.DataFrame({"firing_class": ["silent"], "block": [True], "rebound": [True]}, index=["A"])
    fg.fig_examples(rows, labels, StepSettings(), RampSettings(), tmp_path / "f1")
    assert (tmp_path / "f1.png").exists() and (tmp_path / "f1.svg").exists()


def test_estimation_plot_and_secondary_figure(tmp_path):
    summary, sel, steps, waves, tcs, bins = synthetic(12)
    labels = cell_labels(summary, RULES)
    tcs["baseline_firing_rate_Hz"], tcs["drug_firing_rate_Hz"] = 1.0, 0.2
    s = DrugStatsSettings(n_boot=500)
    fg.fig_estimation_secondary("5-HT", tcs, labels, ["firing_rate_Hz", "vm_baseline_mV"], s, tmp_path / "f7")
    assert (tmp_path / "f7.png").exists() and (tmp_path / "f7.svg").exists()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a, b) = plt.subplots(2, 1)
    vals = tcs.set_index("cell_tag")[["baseline_vm_baseline_mV", "drug_vm_baseline_mV", "washout_vm_baseline_mV"]]
    ests = fg.estimation_plot(a, b, vals, labels, s)
    assert len(ests) == 2 and ests[0]["ci_high"] < 0                 # drug hyperpolarises in the synthetic data
    plt.close(fig)
