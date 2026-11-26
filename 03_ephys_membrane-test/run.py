"""03_ephys_membrane-test: final Cm, Ra, Rt, Rm and tau for every ready membrane-test recording.

Needs 02_ephys_anomaly-fix/output/analysis_ready.csv. Step by step: open membrane-test.ipynb.
    python 03_ephys_membrane-test/run.py              # refuses to overwrite
    python 03_ephys_membrane-test/run.py --overwrite
    python 03_ephys_membrane-test/run.py --figures    # also save a check figure per recording

Output: output/membrane_test_values.csv (one row per recording). Not yet validated against Clampex.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from ephys_analysis.membrane_test import MTSettings, analyse_table
from ephys_analysis.utils import OutputExistsError, RunLog, check_writable, load_config, repo_path, stamp, write_csv

PROJECT_DIR = Path(__file__).resolve().parent


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--figures", action="store_true", help="save a check figure per recording (output/figures/)")
    args = ap.parse_args(argv)

    cfg = load_config(PROJECT_DIR)
    table_path = repo_path(cfg, cfg["recordings_table"])
    if not table_path.exists():
        print(f"Missing {table_path}. Run 01 and 02 first.")
        return 1
    out = PROJECT_DIR / "output"
    target = out / "membrane_test_values.csv"
    try:
        check_writable([target], args.overwrite)
    except OutputExistsError as exc:
        print(exc)
        return 2

    log = RunLog(cfg, out, save=False)  # only the final values are saved
    ready = pd.read_csv(table_path, dtype={"cell_tag": "string"})
    selection = ready[ready["family"] == cfg["family"]]
    log(f"{len(selection)} ready membrane-test recordings")
    values = analyse_table(selection, MTSettings.from_config(cfg["membrane_test"]), log,
                           out / "figures" if args.figures else None)
    write_csv(stamp(values, log.run_id), target)
    log(f"written: {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
