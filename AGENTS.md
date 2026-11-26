# AGENTS.md: PMvDAT+ intrinsic electrophysiology

Instructions for Codex agents. Folder-specific rules are in `data/AGENTS.md`, `src/AGENTS.md` and `notebooks/AGENTS.md`. Each subproject's `README.md` describes what it does.

## Project

This package analyses the **intrinsic electrophysiology of PMvDAT+ neurons**: dopamine-transporter-expressing neurons of the hypothalamic ventral premammillary nucleus, recorded by whole-cell patch clamp (`.abf`). Reference: Stagkourakis *et al.* (2018), *Nat Neurosci* 21:834–842, https://pubmed.ncbi.nlm.nih.gov/29802391/.

- Write **PMvDAT+ neurons** (PMv^DAT). Never call them "dopaminergic" or "dopamine-producing"; DAT expression alone does not establish that.
- Terms: **recording** = one ABF, **sweep** = one episode, **cell** = one Cell Tag. Keep these levels separate.

## Layout: one folder per subproject, run in order

| Folder | Subproject | Status |
|---|---|---|
| `01_ephys_meta-org/` | inventory, recording-book matching, protocol classification, checks → flags (no copying) | implemented |
| `02_ephys_anomaly-fix/` | auto-resolve duplicated voltage / gain errors, decision sheet for the rest, copy ready files into protocol folders, issues report | implemented |
| `03_ephys_membrane-test/` | final Cm, Ra, Rt, Rm, tau per recording | implemented (not validated against Clampex) |
| `04_ephys_baseline/` | spontaneous (A1/M2): spike detection, silent / tonic, Vm, rate, ISI CV, AP waveform | implemented (not validated against manual measurements) |
| `05_ephys_steps/` | current steps (A3/M3): sag, Rin, F–I, rebound, AP at the first spiking step | implemented (not validated) |
| `06_ephys_ramp/` | ramp (A4/M4): rheobase (quiet sweeps, mean), up/down firing, hysteresis, Rin, rebound | implemented (not validated) |
| `07_ephys_timecourse/` | modulation timecourse (A5/M6): drug applications (solitary / layered), baseline / drug / washout, peak response, binned timecourse | implemented (not validated) |
| `08_ephys_cell-summary/` | one row per cell from 03–07; recording selection logged | implemented |
| `09_ephys_cell-overview/` | descriptive heatmap + strip plots per cell (no PCA / LDA at n ≈ 10, no groups) | implemented |
| `10_ephys_figures/` | publication-style figures (descriptive) + paired within-cell drug statistics | implemented (drug statistics PROVISIONAL, to be reviewed) |

Each subproject folder contains only `README.md`, one step-by-step notebook, `config.yaml`, `run.py` and `output/` (02 also holds the user-edited `decisions.xlsx`, which must never be overwritten). Notebook and `run.py` call the same step functions in the package; no analysis logic in notebooks. Everything a subproject produces stays in its own `output/`. Protocol subprojects detect events first and save a detection figure (PNG + SVG) per recording, then compute features from that same event table. Protocol analyses read `02_ephys_anomaly-fix/output/analysis_ready.csv` and apply its corrections (`current_source`, `voltage_scale`, `bias_current_pA`). Shared code lives in `src/ephys_analysis/`; `docs/` holds the parameter guide (regenerate after changing a definition). Dataset-specific settings live only in the subproject `config.yaml` files (see the root README, "Using the pipeline on a new dataset"); data, outputs and `decisions.xlsx` are never committed. Keep it simple: no extra folders or files without a reason.

## Scope

- **In scope:** passive properties, RMP, spontaneous firing, AP properties, excitability / F-I, adaptation, ramp responses, pharmacological modulation timecourses.
- **Excluded:** behavioural experience-group comparisons. Group labels may stay in metadata but never drive filtering, grouping or statistics.
- **Excluded:** synaptic connectivity / EPSC analysis. That is a separate miniML workflow. EPSC files are only catalogued and copied. Keep `recording_id` and `cell_tag` compatible with it.
- **Excluded:** machine learning, unless it is explicitly requested with a scientific reason.

## Non-negotiable rules

1. **Raw data are read-only.** Never modify, move, rename or overwrite ABFs or the recording book. Corrections are recorded, never applied to the files. Sorting (in 02, ready files only) makes SHA-256-verified copies.
2. **Flag, never delete.** Every flag has a severity (`needs_fix` / `review` / `info`) and a reason. Only the documented auto-resolution rules may resolve a flag without a user decision; every resolution appears in the issues report.
3. **The ABF decides the protocol, not the label.** Infer clamp mode from channel units. Never assume ADC 0 or DAC 0. Never classify from filenames. Ambiguous or mismatched files go to review.
4. **Identifiers.** `recording_id = <stem>__<hash of relative path>`. `cell_tag` comes only from the recording book. Never invent metadata. Never average repeated recordings silently.
5. **No silent overwrites.** Outputs are replaced only with `--overwrite` / `--fresh` (02 rebuilds its output from 01 and the decisions on every run). Never overwrite filled-in decisions.
6. **Configuration over code.** Thresholds and windows live in the subproject's `config.yaml`.
7. **Validation means reference data.** Passing synthetic tests is not scientific validation. Ask when a scientific definition is ambiguous (e.g. spike threshold, membrane-test model); do not pick one arbitrarily.
8. **Statistics.** The cell is the unit of analysis. Avoid pseudoreplication. Run no group comparisons unless asked.

## Environment and commands

```powershell
conda activate jaap_ephys          # or call the jaap_ephys environment's python directly
python -m pip install -e .
python -m pytest -q
python 01_ephys_meta-org/run.py [--fresh]
python 02_ephys_anomaly-fix/run.py [--fresh]
python 03_ephys_membrane-test/run.py [--overwrite] [--figures]
python 04_ephys_baseline/run.py [--overwrite] [--no-figures]
python 05_ephys_steps/run.py [--overwrite] [--no-figures]
python 06_ephys_ramp/run.py [--overwrite] [--no-figures]
python 07_ephys_timecourse/run.py [--overwrite] [--no-figures]
python 08_ephys_cell-summary/run.py [--overwrite]
python 09_ephys_cell-overview/run.py [--overwrite]
python 10_ephys_figures/run.py [--overwrite]
```

Do not install packages globally or change sandbox permissions. Do not run unreviewed third-party scripts.

## Agent behaviour

- Inspect before changing. Make small, incremental changes. Do not rewrite working code unnecessarily.
- Ask before creating, moving or deleting folders.
- Test new functions with synthetic signals with known answers and with edge cases (silent cells, failed fits, missing channels).
- Update the relevant `README.md` when functionality changes. Keep implemented and planned clearly separate.
- Scientific prose follows the writing guide in the Codex workspace (`Downloads\CODEX-scafold\Text scientific writing.txt`).
- Finish with: files changed, tests run (results), open issues.
