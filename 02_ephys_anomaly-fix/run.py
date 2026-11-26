"""02_ephys_anomaly-fix: resolve flagged problems, then organise the analysis-ready files.

Needs 01_ephys_meta-org first. Step by step: open anomaly-fix.ipynb. All at once:
    python 02_ephys_anomaly-fix/run.py            # re-run after filling in decisions.xlsx
    python 02_ephys_anomaly-fix/run.py --fresh    # clear output/ first (decisions.xlsx is kept)

Every run regenerates output/ from 01's results and the current decisions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ephys_analysis import anomaly_fix as af
from ephys_analysis.utils import RunLog

REPO = Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fresh", action="store_true", help="clear output/ first (decisions.xlsx is kept)")
    args = ap.parse_args(argv)

    cfg, meta_cfg, out = af.setup(REPO, fresh=args.fresh)
    log = RunLog(cfg, out)
    recordings, flags = af.load_inputs(cfg)

    auto = af.auto_resolve(recordings, flags, cfg)
    log(f"auto-resolved: {len(auto)} issues")
    sheet, n_new = af.update_decision_sheet(af.pending_flags(flags, auto), recordings, cfg)
    sheet = af.validate_decisions(sheet, cfg)
    log(f"decision sheet: {n_new} new rows; {(sheet['decision_status'] == '').sum()} undecided; "
        f"{sheet['decision_status'].str.startswith('invalid').sum()} invalid")
    status, issues = af.resolve(recordings, flags, auto, sheet, cfg.get("noise_corrections"))
    manifest = af.organise(recordings, status, meta_cfg, out, log)
    paths = af.write_outputs(recordings, status, issues, manifest, out, log.run_id)

    log("Recordings: " + ", ".join(f"{k}={v}" for k, v in status["state"].value_counts().items()))
    if (status["state"] == "pending").any():
        log(f"Pending recordings need decisions: fill in {af.repo_path(cfg, cfg['decision_sheet'])} and run again.")
    log(f"Report: {paths[2]}")
    log.finish(paths)
    return 0


if __name__ == "__main__":
    sys.exit(main())
