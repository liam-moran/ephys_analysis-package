"""04_ephys_baseline: silent / tonic classification and baseline features of every ready
spontaneous recording (last 100 s).

Needs 02_ephys_anomaly-fix/output/analysis_ready.csv. Step by step: open baseline.ipynb.
    python 04_ephys_baseline/run.py              # refuses to overwrite
    python 04_ephys_baseline/run.py --overwrite
    python 04_ephys_baseline/run.py --no-figures # skip the detection figures

Order: step 1 detects spikes and saves a detection figure per recording (PNG + SVG,
output/figures/); step 2 builds the feature table from that same spike table.
Output: output/spikes.csv, output/detection.csv, output/baseline_features.csv.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from ephys_analysis.baseline import BaselineSettings, detect_all, feature_table
from ephys_analysis.spikes import SpikeSettings
from ephys_analysis.utils import OutputExistsError, RunLog, check_writable, load_config, repo_path, stamp, write_csv

PROJECT_DIR = Path(__file__).resolve().parent


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no-figures", action="store_true", help="do not save the detection figures")
    args = ap.parse_args(argv)

    cfg = load_config(PROJECT_DIR)
    table_path = repo_path(cfg, cfg["recordings_table"])
    if not table_path.exists():
        print(f"Missing {table_path}. Run 01 and 02 first.")
        return 1
    out = PROJECT_DIR / "output"
    targets = [out / "spikes.csv", out / "detection.csv", out / "baseline_features.csv"]
    try:
        check_writable(targets, args.overwrite)
    except OutputExistsError as exc:
        print(exc)
        return 2

    log = RunLog(cfg, out)
    ready = pd.read_csv(table_path, dtype={"cell_tag": "string"})
    selection = ready[ready["family"].isin(cfg["families"])]
    log(f"{len(selection)} ready baseline recordings")
    sp = SpikeSettings.from_config(cfg["spikes"])
    bs = BaselineSettings.from_config(cfg["baseline"])
    figures = None if args.no_figures else out / "figures"
    log("step 1: spike detection" + ("" if figures is None else f" + figures ({', '.join(cfg['figure_formats'])})"))
    spikes, detection = detect_all(selection, sp, bs, log, figures, tuple(cfg["figure_formats"]))
    write_csv(stamp(spikes, log.run_id), targets[0])
    write_csv(stamp(detection, log.run_id), targets[1])
    log("step 2: baseline features")
    features = feature_table(selection, spikes, bs, log)
    write_csv(stamp(features, log.run_id), targets[2])
    log("written: " + ", ".join(str(t) for t in targets))
    log.finish(targets)
    return 0


if __name__ == "__main__":
    sys.exit(main())
