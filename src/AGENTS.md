# src/AGENTS.md

Shared code for all subprojects: `src/ephys_analysis/`. It is deliberately flat, with one module per topic:

| Module | Contents |
|---|---|
| `abf.py` | read ABFs (pyABF); channel roles from units; unit conversion; command-waveform summaries |
| `metadata.py` | recording book, per-ABF metadata, `recording_id`, ABF-to-book matching, cell table |
| `classification.py` | protocol classification from ABF evidence; verified copy-sorting |
| `anomalies.py` | flags: recording-book, amplifier-header and signal checks |
| `noise.py` | 01 noise QC (spectrum, RMS, mains, low-frequency peaks, artefacts; spectrum figure) and the noise corrections decided in 02 (`correct`: mains sinusoid removal, notch), applied by 04–07 |
| `meta_org.py` | 01 step functions (inventory, match book, classify, check, save) |
| `anomaly_fix.py` | 02 step functions (auto-resolve, decision sheet, resolve, organise, report) |
| `membrane_test.py` | Clampex membrane-test algorithm; optional check figure |
| `spikes.py` | shared AP detection, per-spike waveform features, attenuated (non-spike) events |
| `baseline.py` | 04 baseline: analysis window, silent / tonic, Vm, rate, ISI CV, AP waveform summary; detection figure |
| `steps.py` | 05 current steps: step window, sag, Rin, rheobase, F–I, rebound, bursts; detection figure |
| `ramp.py` | 06 ramp: segments from the command, first-spike current, hysteresis, Rin, rebound; detection figure |
| `timecourse.py` | 07 drug timecourse: tag classification, periods, changes, lowest point, bins; detection figure |
| `cell_summary.py` | 08 one row per cell: recording selection (first usable / override) and join |
| `cell_overview.py` | 09 descriptive heatmap (z-scores across cells) and strip plots |
| `figures.py` | 10 publication-style figures and the summary table (reads 02, 05–08 outputs) |
| `drug_stats.py` | 10 PROVISIONAL paired within-cell drug statistics (Wilcoxon, bootstrap CI, Friedman, Holm, McNemar, responders, sham peak) |
| `utils.py` | subproject config loading, run log / provenance, safe CSV writing, min/max `envelope` for plotting |

When a new protocol subproject is built, add **one** module for it (e.g. `spontaneous.py`). The shared spike detector is `spikes.py`: reuse it, do not write another. Do not create sub-packages.

## Rules

- Functions do one thing and take arrays and parameters as arguments. Keep calculations independent of pyABF, so they can be tested with synthetic signals.
- Give public functions type hints and docstrings with **units** and the method or equation.
- Work in mV, pA, s (derived MΩ, pF, ms). Convert from the ABF's units and record the factor.
- No magic numbers. Thresholds and windows come from the subproject `config.yaml`. Defaults in code must match it.
- Record failures (status, flags, reason) and continue. Never swallow exceptions silently.
- Silent cells, failed fits and missing values are results (0 or NaN with a reason). Never drop the row.
- Add tests in `tests/` for every new function: known-answer synthetic signals, edge cases and, where possible, the real example files. Run `python -m pytest -q` before finishing.
- Style: PEP 8, `snake_case`, `pathlib`, `is None`. Prefer readable code to abstraction.
