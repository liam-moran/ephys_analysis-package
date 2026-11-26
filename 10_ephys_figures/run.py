"""10_ephys_figures: publication-style figures (descriptive) and PROVISIONAL paired drug statistics.

Needs the outputs of 02 and 05-08. Step by step: open figures.ipynb.
    python 10_ephys_figures/run.py              # refuses to overwrite
    python 10_ephys_figures/run.py --overwrite

Output: output/figures/fig1_examples ... fig6_drug_<condition> (+ fig7_estimation_<condition> if enabled; PNG + SVG), output/summary_table.csv,
output/drug_stats.csv, output/drug_responders.csv (both PROVISIONAL).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ephys_analysis import figures as fg
from ephys_analysis.cell_overview import cell_labels
from ephys_analysis.drug_stats import DrugStatsSettings, drug_statistics, peak_sham_table, select_applications
from ephys_analysis.ramp import RampSettings
from ephys_analysis.steps import StepSettings
from ephys_analysis.timecourse import condition_slug
from ephys_analysis.utils import OutputExistsError, RunLog, check_writable, load_config, repo_path, stamp, write_csv

PROJECT_DIR = Path(__file__).resolve().parent


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(PROJECT_DIR)
    fg.configure(cfg)
    data = fg.load_inputs(cfg, repo_path)
    if data["summary"].empty:
        print("Missing the 08 cell summary. Run 01-08 first.")
        return 1
    out = PROJECT_DIR / "output"
    figs = out / "figures"
    targets = [out / "summary_table.csv", out / "drug_stats.csv", out / "drug_responders.csv"]
    try:
        check_writable(targets, args.overwrite)
    except OutputExistsError as exc:
        print(exc)
        return 2
    log = RunLog(cfg, out)
    fmt = tuple(cfg["figure_formats"])
    summary, selection = data["summary"], data["selection"]
    labels = cell_labels(summary, cfg["labels"])
    log(f"{len(summary)} cells")

    table = fg.summary_table(summary, cfg["properties"] + cfg["ap_properties"])
    s = DrugStatsSettings.from_config(cfg["drug_stats"])
    stats_df, resp = (drug_statistics(data["tc_summary"], data["tc_bins"], s, selection, summary)
                      if not data["tc_summary"].empty else (None, None))

    rows = fg.example_rows(cfg.get("examples"), labels, selection, data["ready"])
    fg.fig_examples(rows, labels, StepSettings(), RampSettings(), figs / "fig1_examples", fmt,
                    cfg.get("example_spontaneous_s", 30))
    fg.fig_labels(labels, figs / "fig2_labels", fmt)
    fg.fig_properties(summary, labels, cfg["properties"], figs / "fig3_properties", fmt)
    fg.fig_ap(data["waves"], summary, labels, fg.recordings_of(selection, "ramp"), cfg["ap_properties"],
              figs / "fig4_ap", fmt)
    fg.fig_excitability(data["step_sweeps"], summary, labels, fg.recordings_of(selection, "steps"),
                        figs / "fig5_excitability", fmt)
    if stats_df is not None:
        apps = select_applications(data["tc_summary"], selection)
        for cond, g in apps.groupby("condition"):
            fg.fig_drug(cond, g, data["tc_bins"], labels, stats_df[stats_df["condition"] == cond],
                        tuple(cfg["drug_window_s"]), figs / f"fig6_drug_{condition_slug(cond)}", fmt,
                        peak_sham_table(g, data["tc_bins"], s), s)
            if s.estimation_secondary:
                fg.fig_estimation_secondary(cond, g, labels, list(s.secondary), s,
                                            figs / f"fig7_estimation_{condition_slug(cond)}", fmt)
            log(f"fig6 {cond}: {g['cell_tag'].nunique()} cells")
    write_csv(stamp(table, log.run_id), targets[0])
    if stats_df is not None:
        write_csv(stamp(stats_df, log.run_id), targets[1])
        write_csv(stamp(resp, log.run_id), targets[2])
        log("drug statistics written: PROVISIONAL, review before reporting")
    log("written: figures fig1-fig7, " + ", ".join(t.name for t in targets))
    log.finish(targets)
    return 0


if __name__ == "__main__":
    sys.exit(main())
