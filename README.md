# PMvDAT+ intrinsic electrophysiology

Analysis of whole-cell patch-clamp recordings (`.abf`) from **PMvDAT+ neurons**: dopamine-transporter-expressing neurons of the hypothalamic ventral premammillary nucleus (Stagkourakis *et al.*, 2018, *Nat Neurosci* 21:834–842, [PubMed 29802391](https://pubmed.ncbi.nlm.nih.gov/29802391/)). DAT expression alone does not make these neurons dopaminergic, so they are called PMvDAT+.

**Scope:** intrinsic properties only. These are passive properties, RMP, spontaneous firing, AP properties, excitability / F-I, ramp responses and drug timecourses.

**Not included:**
- Behavioural experience-group comparisons.
- Synaptic connectivity. EPSC recordings are only catalogued for the separate **miniML** workflow.

**Parameter guide:** `docs/parameter_guide.html` (open in a browser) lists every parameter with a sketch, how it is measured and what it says about the cell. Regenerate it with `python docs/make_param_guide.py` after changing a definition.

**Licence:** none yet; all rights reserved. Decide with the lab before making the repository public.

## Structure

```
ephys_analysis-package/
├── AGENTS.md                  rules for coding agents (+ data/, src/, notebooks/AGENTS.md)
├── data/                      raw ABFs (one folder per recording day) + recording book: read-only
├── 01_ephys_meta-org/         inventory, recording book, protocol, checks → flags
├── 02_ephys_anomaly-fix/      auto-resolve + your decisions → analysis-ready, sorted files + issues report
├── 03_ephys_membrane-test/    final Cm, Ra, Rt, Rm, tau per recording
├── 04_ephys_baseline/         spike detection, silent / tonic, Vm, rate, ISI CV, AP waveform
├── 05_ephys_steps/            current steps: sag, Rin, F–I, rebound, AP at the first spiking step
├── 06_ephys_ramp/             ramp: first-spike current, up/down firing, hysteresis, Rin, rebound
├── 07_ephys_timecourse/       drug applications: baseline / drug / washout, peak response, binned timecourse
├── 08_ephys_cell-summary/     one row per cell from 03–07 (one recording per protocol, never averaged)
├── 09_ephys_cell-overview/    heatmap + strip plots of every cell's properties (descriptive; no PCA / LDA)
├── 10_ephys_figures/          publication-style figures + PROVISIONAL paired drug statistics
├── src/ephys_analysis/        shared code used by all subprojects
├── notebooks/                 exploration / check notebooks (optional)
├── docs/                      parameter guide (HTML) and its generator
└── tests/
```

Each subproject folder contains:
- `README.md`: what it does, checks, outputs, current results;
- a step-by-step **notebook**;
- `config.yaml`;
- `run.py`, which runs the same steps in one go;
- `output/`.

`02` also contains your **`decisions.xlsx`**. Each protocol analysis is its own subproject: baseline (04), current steps (05), ramp (06) and the modulation timecourse (07) are built.

In 04–07, spikes are **detected first** and a **detection figure (PNG + SVG) is saved for every recording**. The features are only computed afterwards, from the same spike table, so the detection can be reviewed first.

## Install (once)

```powershell
conda env create -f environment.yml          # environment: jaap_ephys
conda activate jaap_ephys
python -m pip install -e .
python -m pytest -q
```

To open the notebooks in VS Code or JupyterLab, select the **jaap_ephys** Python environment as the kernel.

## Running from scratch with new data

1. **Copy the data.** Copy each recording day as its own folder into `data/` (e.g. `data/2026-11-26/…abf`), and copy the recording book into `data/`. These files are only ever read. To leave the practice recordings out, move `data/example_data` elsewhere.
2. **Check the settings** (see "Using the pipeline on a new dataset" below; for more data of the same study, usually only the recording-book file name).
3. **Close outputs.** Close any output files open in Excel.
4. **Run the notebooks in order, cell by cell, reviewing each step:**
   1. `01_ephys_meta-org/meta-org.ipynb`, with `FRESH = True`.
   2. `02_ephys_anomaly-fix/anomaly-fix.ipynb`. If decisions are listed, fill in `decisions.xlsx`, close it, and re-run from Step 2.
   3. `03_ephys_membrane-test/membrane-test.ipynb`.
   4. `04_ephys_baseline/baseline.ipynb`.
   5. `05_ephys_steps/steps.ipynb`.
   6. `06_ephys_ramp/ramp.ipynb`.
   7. `07_ephys_timecourse/timecourse.ipynb`.
   8. `08_ephys_cell-summary/cell-summary.ipynb`.
   9. `09_ephys_cell-overview/cell-overview.ipynb`.
   10. `10_ephys_figures/figures.ipynb`.

The same steps without notebooks:

```powershell
python 01_ephys_meta-org/run.py --fresh
python 02_ephys_anomaly-fix/run.py --fresh
python 03_ephys_membrane-test/run.py --overwrite
python 04_ephys_baseline/run.py --overwrite
python 05_ephys_steps/run.py --overwrite
python 06_ephys_ramp/run.py --overwrite
python 07_ephys_timecourse/run.py --overwrite
python 08_ephys_cell-summary/run.py --overwrite
python 09_ephys_cell-overview/run.py --overwrite
python 10_ephys_figures/run.py --overwrite
```

**Step by step or all at once?**
- For a new batch of data, go **step by step**. After 01, read the flags. After 02, read `issues_report.md` and fill in the decisions. Only then run 03.
- Once the checks and decisions are settled, the `run.py` commands repeat the whole run in one go.

## Using the pipeline on a new dataset

The pipeline works for **ABF recordings of this lab's protocol types** (membrane test, spontaneous, current steps, ramp, drug timecourse, EPSC), recognised from the command waveform, not from file names. It is not a general ephys toolbox: other file formats or new protocol types need new code.

**One dataset per copy of the repository.** Clone the repository for each study, put that study's data in `data/`, and keep its `decisions.xlsx` and outputs in that copy. They are never committed to the package.

**Settings to check, all in existing, commented `config.yaml` files:**

| Where | What | When to change it |
|---|---|---|
| `01_ephys_meta-org/config.yaml` | `paths.recording_book` | always: the file name of the new recording book |
| | `recording_book.sheets`, `.key_column`, `.columns` | if the book's sheet or column names differ (Recording Log, Recording Name, Cell Tag, Animal ID, Slice, Date Recorded, Protocol) |
| | `expected` | another amplifier or other command sensitivities |
| | `noise.mains_Hz` | 60 in North America |
| `07_ephys_timecourse/config.yaml` | `drug_names` | add every compound you write in tags (e.g. `SP: [substance p, sp]`) |
| | `drug_keywords`, `washout_keywords`, `book_drug_prefix` | if tags or book columns are worded differently |
| `08_ephys_cell-summary/config.yaml` | `cell_metadata_columns` | the book columns to carry into the cell table |
| `10_ephys_figures/config.yaml` | `cell_type_label`, `liquid_junction_potential_mV`, `ljp_solution` | another cell type, internal solution or temperature |
| | `drug_stats` | decide the primary outcome and rules **before** looking at the results |

Analysis windows and thresholds (04–07) are in the same files; change them only for a scientific reason, and for all cells at once.

**How to recall later which settings were used:**
- every `config.yaml` explains its settings in comments;
- git keeps the history of every change;
- every run writes the exact settings it used to its `output/run_info.json` (and a log to `output/log.txt`).

## Status

| Subproject | Status |
|---|---|
| 01 meta-org | ✅ Recording-book, header (MultiClamp 700B) and signal checks; **noise QC** (spectrum per recording: RMS, 50 Hz mains, pump peaks, transient artefacts). Example data: all 10 current-clamp files have the voltage recorded twice; two files have a −10 pA bias noted in the book; one label mismatch. |
| 02 anomaly-fix | ✅ Auto-resolves duplicated-voltage and gain problems, with a decision sheet for the rest (including optional noise corrections, `remove_mains` / `notch`, applied in memory in 04–07), sorting of ready files and an issues report. Gain auto-resolution is tested on synthetic data only, since no example file has a gain error. |
| 03 membrane-test | ✅ Main Ra comes from the fast component of a two-exponential fit that models the amplifier filter; the Clampex values are kept for comparison. ⚠️ Validated on simulations only, not yet against Clampex readouts. |
| 04 baseline | ✅ Spike detection (Allen-style, threshold at 5 % of each spike's upstroke), silent / tonic (≤ 10 spikes in the last 100 s), Vm with spikes removed, rate, ISI CV, AP waveform. Not validated against manual measurements. |
| 05 steps | ✅ Sag (Allen, whole step), Rin, first spiking step (coarse, 50 pA; not the rheobase), F–I, adaptation (Allen; NaN below 4 spikes), rebound spikes (5 s) and bursts, AP at the first spiking step; attenuated events reported separately (depolarisation block). Not validated against manual measurements. |
| 06 ramp | ✅ **Rheobase** (mean first-spike current over the sweeps with no spike in the 0.5 s before the ramp), **AP at ramp rheobase** (mean of the first valid up-ramp spike of the same sweeps; waveforms saved), up/down spikes, hysteresis, Rin from the negative ramp, rebound after release, AP at the first spike; attenuated events reported separately. Not validated against manual measurements. |
| 07 timecourse | ✅ Several drug applications per recording (solitary or layered; drug names from the tags, else the book); per application baseline (100 s before the drug; layered: initial baseline), drug (60–260 s after the tag) and washout periods with the 04 features, changes from baseline, lowest point relative to baseline Vm, baseline drift, 10 s binned timecourse. Not validated against manual measurements. |
| 08 cell summary | ✅ One row per cell joining 03–07, with one column set per drug condition. One recording per protocol: the first usable one, or an override; every choice is listed in `selection.csv`. No statistics. |
| 09 cell overview | ✅ Heatmap (z-scored across cells) and strip plots (real units) of ~20 properties per cell, coloured by silent / tonic, block and rebound. Descriptive only. PCA is not used at the expected n ≈ 10 (fewer cells than features); LDA is not possible without groups. |
| 10 figures | ✅ Example traces, silent / tonic + block + rebound (count bars, per-cell dot matrix), properties (one dot per cell, median, IQR), AP at ramp rheobase (average AP, phase plane), F–I and block onset, drug timecourse per condition; summary table. ⚠️ Drug statistics (paired, within-cell) are **provisional** defaults to be reviewed before reporting. |

**Liquid junction potential:** +15.9 mV (K-gluconate internal, standard ACSF, 35 °C, Ag/AgCl pellet). It was **not** compensated during recording and is **not** corrected by the pipeline. Absolute voltages (Vm, threshold, peak, v_min, …) are as recorded; the true membrane potential is ≈ 15.9 mV more negative (V_true = V_recorded − 15.9 mV). Differences (amplitude, AHP, ΔV, sag ratio, Rin, drug changes) are unaffected.

**Protocols as recorded** (from the ABF command waveforms, example data):
- **Current steps (A3):** 6 sweeps of 30 s; a 500 ms step (3.97–4.47 s) of −150, −100, −50, 0, +50 and +100 pA; then 25 s at 0 pA (rebound).
- **Ramp (A4):** 3 sweeps of 22 s; 0 → +100 pA over 500 ms, back to 0 over 500 ms, 2 s at 0, 0 → −100 pA over 500 ms, abrupt return to 0, then 15 s at 0 (rebound).
- **Modulation timecourse (A5):** 1200 s gap-free; tag 1 = drug reaches the bath ("5ht hit bath"), tag 2 = "washout".

Raw data, recording books, every `output/` folder and all `.xlsx` files (including `decisions.xlsx`, which belongs to a dataset, not to the package) are excluded from git.
