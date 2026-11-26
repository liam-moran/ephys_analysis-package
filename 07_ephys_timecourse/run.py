"""07_ephys_timecourse: every drug application (baseline / drug / washout) of every ready A5 / M6
recording; one recording can hold several applications.

Needs 02_ephys_anomaly-fix/output/analysis_ready.csv. Step by step: open timecourse.ipynb.
    python 07_ephys_timecourse/run.py              # refuses to overwrite
    python 07_ephys_timecourse/run.py --overwrite
    python 07_ephys_timecourse/run.py --no-figures # skip the detection figures

Order: step 1 detects spikes over the whole recording and saves a detection figure per
recording (PNG + SVG, output/figures/); step 2 computes the period features, changes and the
binned timecourse from that same spike table.
Output: output/spikes.csv, output/detection.csv, output/period_features.csv,
        output/timecourse_summary.csv, output/timecourse_bins.csv.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from ephys_analysis.baseline import BaselineSettings
from ephys_analysis.spikes import SpikeSettings
from ephys_analysis.timecourse import TimecourseSettings, detect_all, feature_table
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
    targets = [out / "spikes.csv", out / "detection.csv", out / "period_features.csv",
               out / "timecourse_summary.csv", out / "timecourse_bins.csv"]
    try:
        check_writable(targets, args.overwrite)
    except OutputExistsError as exc:
        print(exc)
        return 2

    log = RunLog(cfg, out)
    ready = pd.read_csv(table_path, dtype={"cell_tag": "string"})
    selection = ready[ready["family"].isin(cfg["families"])]
    book_path = repo_path(cfg, cfg["book_table"])
    book = pd.read_csv(book_path) if book_path.exists() else None
    log(f"{len(selection)} ready timecourse recordings")
    sp = SpikeSettings.from_config(cfg["spikes"])
    tc = TimecourseSettings.from_config(cfg["timecourse"])
    bs = BaselineSettings.from_config(cfg["baseline"])
    figures = None if args.no_figures else out / "figures"
    log("step 1: spike detection" + ("" if figures is None else f" + figures ({', '.join(cfg['figure_formats'])})"))
    spikes, detection = detect_all(selection, sp, tc, bs, log, figures, tuple(cfg["figure_formats"]), book)
    write_csv(stamp(spikes, log.run_id), targets[0])
    write_csv(stamp(detection, log.run_id), targets[1])
    log("step 2: period features and timecourse")
    periods, summary, bins = feature_table(selection, spikes, tc, bs, book, log)
    for df, p in zip((periods, summary, bins), targets[2:]):
        write_csv(stamp(df, log.run_id), p)
    log("written: " + ", ".join(str(t) for t in targets))
    log.finish(targets)
    return 0


if __name__ == "__main__":
    sys.exit(main())
