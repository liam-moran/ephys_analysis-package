# data/AGENTS.md

This folder holds **raw inputs only**, and they are read-only:

- `<recording day>/*.abf` (e.g. `example_data/…`): original ABF recordings, one folder per recording day.
- `Recording_Book_*.xlsx`: the recording book.

Rules:
- Never edit, re-save, rename, move or delete anything here. Open the recording book read-only. Excel auto-save counts as editing.
- Do not write outputs here. Derived files go to the subproject's `output/` folder (sorted copies: `02_ephys_anomaly-fix/output/protocol_sorted/`).
- Problems in the recording book (`#REF!`, duplicate Recording Names, missing Cell Tags) are reported in `01_ephys_meta-org/output/flags.csv`. The user corrects the book.
- Problems in recordings (duplicated voltage / missing current, gain errors, amplifier-set bias current) are flagged by `01_ephys_meta-org` and resolved by `02_ephys_anomaly-fix` as **recorded corrections** (`analysis_ready.csv`). The originals are never changed.
- Note any current set on the amplifier (holding / bias) in the recording book: it is not stored in the ABF.
- New recording days get their own sub-folder directly under `data/` (`paths.abf_root` in `01_ephys_meta-org/config.yaml`). Keep original filenames.
- Raw data and `*.xlsx` files are git-ignored. Never commit them.
