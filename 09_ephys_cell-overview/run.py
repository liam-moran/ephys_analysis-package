"""09_ephys_cell-overview: heatmap and strip plots of every cell's key properties.

Needs 08_ephys_cell-summary/output/cell_summary.csv. Step by step: open cell-overview.ipynb.
    python 09_ephys_cell-overview/run.py              # refuses to overwrite
    python 09_ephys_cell-overview/run.py --overwrite

Output: output/figures/cell_overview_heatmap.(png|svg), output/figures/cell_overview_strips.(png|svg),
output/cell_overview_values.csv, output/cell_overview_zscores.csv, output/missing.csv.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from ephys_analysis.cell_overview import cell_labels, feature_table, plot_heatmap, plot_strips, zscore_table
from ephys_analysis.utils import OutputExistsError, RunLog, check_writable, load_config, repo_path, stamp, write_csv

PROJECT_DIR = Path(__file__).resolve().parent


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(PROJECT_DIR)
    src = repo_path(cfg, cfg["cell_summary"])
    if not src.exists():
        print(f"Missing {src}. Run 08 first.")
        return 1
    out = PROJECT_DIR / "output"
    targets = [out / "cell_overview_values.csv", out / "cell_overview_zscores.csv", out / "missing.csv"]
    try:
        check_writable(targets, args.overwrite)
    except OutputExistsError as exc:
        print(exc)
        return 2

    log = RunLog(cfg, out)
    summary = pd.read_csv(src, dtype={"cell_tag": "string"})
    feats = [f["column"] for f in cfg["features"]]
    names = {f["column"]: f["label"] for f in cfg["features"]}
    areas = {f["column"]: f["area"] for f in cfg["features"]}
    values, missing = feature_table(summary, feats)
    z = zscore_table(values, cfg.get("log_features"))
    labels = cell_labels(summary, cfg["labels"])
    log(f"{len(values)} cells x {values.shape[1]} features; missing values: {int(values.isna().sum().sum())}")
    fmt = tuple(cfg["figure_formats"])
    plot_heatmap(z, labels, names, areas, out / "figures" / "cell_overview_heatmap", cfg.get("clip_sd", 2.5), fmt)
    plot_strips(values, labels, names, out / "figures" / "cell_overview_strips", formats=fmt)
    write_csv(stamp(values.join(labels).reset_index(), log.run_id), targets[0])
    write_csv(stamp(z.join(labels).reset_index(), log.run_id), targets[1])
    write_csv(stamp(missing, log.run_id), targets[2])
    log("written: figures/cell_overview_heatmap, figures/cell_overview_strips, " + ", ".join(t.name for t in targets))
    log.finish(targets)
    return 0


if __name__ == "__main__":
    sys.exit(main())
