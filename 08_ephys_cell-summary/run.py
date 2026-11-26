"""08_ephys_cell-summary: one row per cell from the final values of 03-07.

Needs the outputs of 01 and 03-07 (missing protocol outputs are skipped). Step by step: open
cell-summary.ipynb.
    python 08_ephys_cell-summary/run.py              # refuses to overwrite
    python 08_ephys_cell-summary/run.py --overwrite

Output: output/cell_summary.csv (one row per cell), output/selection.csv (which recording was
used for each cell and protocol, the alternatives and why).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from ephys_analysis.cell_summary import build_summary, load_tables
from ephys_analysis.utils import OutputExistsError, RunLog, check_writable, load_config, repo_path, stamp, write_csv

PROJECT_DIR = Path(__file__).resolve().parent


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(PROJECT_DIR)
    cells_path = repo_path(cfg, cfg["cells_table"])
    if not cells_path.exists():
        print(f"Missing {cells_path}. Run 01 first.")
        return 1
    out = PROJECT_DIR / "output"
    targets = [out / "cell_summary.csv", out / "selection.csv"]
    try:
        check_writable(targets, args.overwrite)
    except OutputExistsError as exc:
        print(exc)
        return 2

    log = RunLog(cfg, out)
    cells = pd.read_csv(cells_path, dtype={"cell_tag": "string"})
    tables = load_tables(cfg, repo_path)
    for name, t in tables.items():
        log(f"{name}: {len(t)} recordings" + ("" if len(t) else " (output not found or empty)"))
    summary, selection = build_summary(cells, tables, cfg, log)
    write_csv(stamp(summary, log.run_id), targets[0])
    write_csv(stamp(selection, log.run_id), targets[1])
    log("written: " + ", ".join(str(t) for t in targets))
    log.finish(targets)
    return 0


if __name__ == "__main__":
    sys.exit(main())
