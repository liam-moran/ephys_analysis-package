"""Cell overview (cell_overview.py): table, z-scores, labels, figures."""

import numpy as np
import pandas as pd
import pytest

from ephys_analysis.cell_overview import cell_labels, feature_table, plot_heatmap, plot_strips, zscore_table


def _summary(n=12, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"cell_tag": [f"C{i:02d}" for i in range(n)],
                       "base_firing_class": ["tonic", "silent"] * (n // 2),
                       "a": rng.normal(10, 2, n), "rate": rng.uniform(0.01, 2, n),
                       "const": np.ones(n), "block_pA": [50.0, np.nan] * (n // 2),
                       "rebound_n": [0, 3] * (n // 2)})
    df.loc[1, "a"] = np.nan
    return df


def test_feature_table_keeps_nan_and_reports_missing():
    v, m = feature_table(_summary(), ["a", "rate", "not_there"])
    assert list(v.columns) == ["a", "rate"] and np.isnan(v.loc["C01", "a"])
    mm = m.set_index("feature")
    assert mm.loc["a", "n_missing"] == 1 and mm.loc["not_there", "note"] == "column not found"


def test_zscores_log_and_degenerate_features():
    v, _ = feature_table(_summary(), ["a", "rate", "const"])
    z = zscore_table(v, log_features=["rate"])
    assert z["a"].mean() == pytest.approx(0, abs=1e-9) and z["a"].std(ddof=1) == pytest.approx(1)
    logged = np.log10(v["rate"] + 0.01)
    assert z["rate"].corr(logged) == pytest.approx(1)
    assert z["const"].isna().all()                                   # zero variance -> n/a
    one = zscore_table(pd.DataFrame({"x": [1.0, np.nan, np.nan]}))
    assert one["x"].isna().all()                                     # single value -> n/a


def test_labels():
    lab = cell_labels(_summary(), {"firing_class": "base_firing_class", "block_if_not_missing": "block_pA",
                                   "rebound_if_positive": "rebound_n"})
    assert lab.loc["C00", "firing_class"] == "tonic" and lab.loc["C00", "block"] and not lab.loc["C00", "rebound"]
    assert lab.loc["C01", "firing_class"] == "silent" and not lab.loc["C01", "block"] and lab.loc["C01", "rebound"]


@pytest.mark.parametrize("n", [12, 2])
def test_figures_written(tmp_path, n):
    df = _summary(12).head(n)
    v, _ = feature_table(df, ["a", "rate", "const"])
    z = zscore_table(v, ["rate"])
    lab = cell_labels(df, {"firing_class": "base_firing_class", "block_if_not_missing": "block_pA",
                           "rebound_if_positive": "rebound_n"})
    names = {"a": "A (mV)", "rate": "Rate (Hz)", "const": "Const"}
    areas = {"a": "X", "rate": "X", "const": "Y"}
    plot_heatmap(z, lab, names, areas, tmp_path / "heat")
    plot_strips(v, lab, names, tmp_path / "strips")
    for f in ("heat.png", "heat.svg", "strips.png", "strips.svg"):
        assert (tmp_path / f).exists()
