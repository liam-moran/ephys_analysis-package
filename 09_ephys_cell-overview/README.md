# 09_ephys_cell-overview: every cell's key properties side by side

A **descriptive** overview of the 08 cell summary (one row per cell). It shows:
- how heterogeneous the PMvDAT+ cells are;
- which cells stand out;
- how the intrinsic labels line up with the other properties. The labels are silent / tonic, depolarisation block and rebound.

**There are no statistics, no PCA and no LDA** (see below).

**Run step by step:** open `cell-overview.ipynb`.
**Run all at once:** `python 09_ephys_cell-overview/run.py` (add `--overwrite`).
Run 01–08 first.

## Figures (`output/figures/`, PNG + SVG)

| Figure | What it shows |
|---|---|
| `cell_overview_heatmap` | Cells (rows, silent first) × properties (columns, grouped by area). Each property is **z-scored across cells**, so 0 = the average cell, red = higher, blue = lower, in SD, clipped at ±2.5. The firing rate is log-transformed first. Missing values are hatched. A side strip shows the class (blue silent, red tonic), **B** = depolarisation block (steps) and **R** = rebound spikes (ramp). |
| `cell_overview_strips` | One panel per property in **real units**. Each dot is a cell: colour = class, square = block, filled = rebound. The line is the median; n is in each panel title. |

**Tables:**
- `cell_overview_values.csv`: the plotted values, in real units, with the labels.
- `cell_overview_zscores.csv`: the heatmap values.
- `missing.csv`: missing values per property.

## Properties shown (`config.yaml` → `features`)

| Area | Properties |
|---|---|
| Passive | Cm (03), Rin from steps (05), Vm baseline (04), sag ratio (05) |
| Spontaneous | firing rate (log), ISI CV (04; missing for silent cells) |
| AP at ramp rheobase | threshold, half-width, amplitude, upstroke, AHP (06; mean of the first valid up-ramp spike of each quiet sweep, the sweeps of the rheobase, so every cell is measured at its own threshold current) |
| Excitability | rheobase (06 ramp: mean over quiet sweeps), F–I slope, max step rate (05) |
| Rebound | rebound spikes after the ramp (06), rebound peak after the ramp (06) and after the steps (05) |
| Drug | 5-HT: ΔVm in the drug period, lowest ΔVm (07) |

- **Editing the list:** add or remove a property by editing `features` (`column` from `08_ephys_cell-summary/output/cell_summary.csv`, plus a label and an area).
- **Left out:** Ra and other recording-quality measures. They describe the recording, not the cell.
- **Missing values:** nothing is dropped or filled in; gaps are shown.

## What B and R mean

Both labels mark a yes/no property of a cell. A z-scored colour column would hide such a property, so it is shown as a mark. The rules are set under `labels` in `config.yaml`.

**B: depolarisation block (square marker in the strip plots)**
- **Source:** the depolarising current steps (05, up to +100 pA).
- **What it is:** during strong depolarisation, some cells stop firing full action potentials and make small, broad events instead. 05 detects these as **attenuated events**:
  - the voltage rises ≥ 20 mV and peaks above −20 mV;
  - but it fails the spike criterion (dV/dt ≥ 20 mV/ms).

  These events are reported apart and are never counted as spikes.
- **Rule:** B when `step_first_step_with_block_pA` (the smallest step current with attenuated events) is not missing.
- **Why it is marked:** block explains a flat or falling F–I curve and a low maximum rate. Without the mark, such a cell would only look "less excitable".

**R: rebound (filled marker in the strip plots)**
- **Source:** the ramp (06). Spikes are counted within 5 s after the abrupt release from −100 pA back to 0 pA.
- **Rule:** R when `ramp_n_rebound_spikes_short` > 0.
- **Why it is marked:** rebound firing often goes with a larger sag and a higher rebound peak. It usually reflects currents activated or de-inactivated by hyperpolarisation (e.g. I_h or T-type Ca²⁺). This package does not test which current is responsible.

**Why the two are shown differently**
- **Block** is a label only. The block current is missing for every cell without block, so as a heatmap column it would be mostly hatched.
- **Rebound** is shown both as a label and as three heatmap columns: rebound spikes after the ramp, rebound peak after the ramp, and rebound peak after the steps.

**Caveats**
- Both labels are yes/no. For example, one rebound spike counts the same as ten.
- Both labels come from the one recording per cell that 08 selected (`08_ephys_cell-summary/output/selection.csv`).

## Why no PCA or LDA

- **LDA** separates predefined groups. All animals are naive, so there are no groups. Experience-group comparisons are also excluded from this package (`AGENTS.md`). An earlier notebook (not in the repository) ran LDA on experience groups.
- **PCA** needs clearly more cells than features to give stable components. A common rule of thumb is ≥ 5–10 cells per feature. With the expected ~10 cells and ~20 properties there are fewer cells than features. The components would mostly reflect which cells happen to be in the dataset, and correlations between properties would be noisy: at n = 10, r must exceed ~0.63 to be distinguishable from chance.

**When to revisit:**
- **Full PCA:** from ~20–30 cells, on a reduced set of properties.
- **Earlier:** an exploratory PCA on ~5 key properties, with a leave-one-cell-out stability check.

## Example data

With the 2 example cells, every property has 2 values, so the z-scores are only ±0.71. The figures show that the pipeline works but carry no information. A synthetic 12-cell table was used to check the layout during development.
