# notebooks/AGENTS.md

Notebooks are for **exploring and checking**: viewing individual recordings, comparing candidate methods before a decision, and checking extracted values against known recordings. The package (`src/ephys_analysis`) is the source of truth for calculations.

- Import from `ephys_analysis`. Do not copy analysis code into notebooks; if logic is reusable, move it into the package with tests.
- Use the kernel "Python (jaap_ephys)". Notebooks must run top to bottom: no `input()`, no out-of-order state, no hard-coded absolute data paths.
- Only read raw data. Do not save results from notebooks into `data/`.
- Write decisions and observations in markdown cells. Lasting decisions go into the relevant subproject `README.md`.
- `check_scripts/`: short read-only inspection scripts requested by the user, each with a docstring stating its purpose.
- Name files in lowercase without spaces, prefixed with a date or topic, e.g. `2026-11-25_membrane_test_check.ipynb`.
