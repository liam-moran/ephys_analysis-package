"""01_ephys_meta-org: read every ABF, match the recording book, classify protocols and
flag anything unusual. Nothing is copied or changed here (sorting happens in 02).

Step by step: open meta-org.ipynb. All at once:
    python 01_ephys_meta-org/run.py            # refuses to overwrite existing outputs
    python 01_ephys_meta-org/run.py --fresh    # clear output/ and start from scratch
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ephys_analysis import meta_org
from ephys_analysis.utils import OutputExistsError, RunLog, check_writable

REPO = Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fresh", action="store_true", help="clear output/ first (raw data are never touched)")
    args = ap.parse_args(argv)

    cfg, out = meta_org.setup(REPO, fresh=args.fresh)
    try:
        check_writable([out / "recordings.csv"], overwrite=False)
    except OutputExistsError as exc:
        print(f"{exc}\n(use --fresh)")
        return 2
    log = RunLog(cfg, out)

    recordings = meta_org.inventory(cfg, log)
    recordings, book_issues, unmatched = meta_org.match_book(recordings, cfg)
    classification = meta_org.classify(recordings, cfg)
    noise_tab, noise_flags = meta_org.noise(recordings, classification, cfg, out, log)
    table, flags = meta_org.check(recordings, classification, book_issues, unmatched, cfg, log, noise_flags)
    paths = meta_org.save(table, flags, out, log.run_id, noise_tab)

    log("Status: " + ", ".join(f"{k}={v}" for k, v in table["status"].value_counts().items()))
    changed = meta_org.originals_unchanged(table)
    log("All original ABFs unchanged (SHA-256)." if not changed else f"ERROR: originals changed: {changed}")
    log.finish(paths)
    return 3 if changed else 0


if __name__ == "__main__":
    sys.exit(main())
