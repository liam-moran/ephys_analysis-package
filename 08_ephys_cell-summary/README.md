# 08_ephys_cell-summary: one row per cell

This step joins the final values of **03 membrane test, 04 baseline, 05 steps, 06 ramp and 07 timecourse** into **one row per cell**, identified by the Cell Tag from the recording book. The cell is the unit of analysis. When a cell has several recordings of one protocol, **one** of them is used; recordings are never averaged.

**Run step by step:** open `cell-summary.ipynb`.
**Run all at once:** `python 08_ephys_cell-summary/run.py` (add `--overwrite`).
Run 01–07 first. A protocol whose output is missing is left empty for every cell.

## Which recording is used

For each cell and protocol, the recording is chosen in this order:
1. **Override:** the recording named in `config.yaml` → `overrides: {cell_tag: {protocol: abf_filename}}`.
2. **First usable recording**, in recording order (file name). The earliest recording is the one closest to break-in and before any drug. A recording is not usable if:
   - its analysis failed;
   - for membrane tests, its `qc_status` is `fail`;
   - it carries an `exclude_if_flag` flag. For baselines these are `abf_tags_present` (e.g. recorded after a drug washout) and `bias_current`.
3. **Fallback:** a flagged recording, used only if no unflagged one exists. The reason then says so.

Every choice is listed in `selection.csv`, with the alternatives and why each was not used.

## Output (`output/`)

| File | Content |
|---|---|
| `cell_summary.csv` | One row per cell, in three parts. **Metadata** from the recording book (`cell_metadata_columns`). **Values**, prefixed by protocol: `mt_` (membrane test), `base_` (baseline), `step_`, `ramp_`, `tc_<condition>_` (timecourse, **one column set per drug condition**, e.g. `tc_5HT_`, `tc_SP_`, `tc_5HT_Ba2_` for a layered application; override key `timecourse:<condition>`). **For each protocol:** the recording used (`<prefix>_recording`), the number of recordings available (`<prefix>_n_recordings`) and its flags. |
| `selection.csv` | One row per cell × protocol: the recording used, the number of recordings, the reason, and the alternatives |
| `log.txt`, `run_info.json` | Run log and provenance |

The columns kept per protocol are listed in `config.yaml` → `protocols: <name>: columns`. Add a column there to include it.

## Current result (example data)

| Protocol | Cell A | Cell B |
|---|---|---|
| Membrane test | A_0005 (only one) | **B_0000**: first of 4. B_0006 and B_0007 fail QC; B_0008 also passes |
| Baseline | A_0000 | **B_0001**. B_0010 is skipped: it was recorded "after washout of 5HT" |
| Steps / ramp / timecourse | A_0001 / A_0002 / A_0004 | B_0002 / B_0003 / B_0004 |

| | Cell A | Cell B |
|---|---|---|
| Ra / Cm / Rt | 54 MΩ / 36.6 pF / 745 MΩ | 26 MΩ / 39.1 pF / 745 MΩ |
| Baseline | silent (0.04 Hz), Vm −55.7 mV | tonic (1.12 Hz, ISI CV 0.76), Vm −46.0 mV |
| AP threshold / half-width (baseline) | −34.4 mV / 1.79 ms | −36.1 mV / 1.96 ms |
| Rheobase: steps / ramp | 50 pA / 12.8 pA | 50 pA / 18.5 pA |
| Rin: steps / ramp | 385 / 859 MΩ | 331 / 636 MΩ |
| Depolarisation block | yes, from +50 pA | no |
| Rebound | after the ramp: 8 spikes, first at 243 ms | after the steps: 1 spike at 203 ms |
| 5-HT: change in drug period / lowest point | −11.8 / −13.9 mV | −6.6 / −11.3 mV |

## Notes

- **Statistics:** none are run here (`AGENTS.md`: no group comparisons unless asked). The table is the input for them, with one row per cell.
- **Recording-book metadata** (sex, age, genotype …) is copied for context only. Experience-group labels are not used.
- **Validation:** values come from subprojects that are not yet validated against manual measurements (see their READMEs).
