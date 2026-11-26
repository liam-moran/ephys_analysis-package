# 04_ephys_baseline: spike detection, silent / tonic, baseline features

This step analyses the **spontaneous baseline recordings** (A1 / M2, no current injected). For each recording it takes the **last 100 s** and:
- detects action potentials;
- classifies the recording as **silent** or **tonic**;
- measures **Vm** (spikes removed), **firing rate**, **ISI CV** and the **AP waveform**.

**Order (notebook and run.py):**
1. **Detect** spikes in every recording and save one **detection figure per recording (PNG + SVG)**.
2. **Review** the figures. If detection is wrong, change `spikes:` in `config.yaml` and detect again.
3. **Features**: the feature table is computed from the *same* spike table that was plotted.

**Run step by step:** open `baseline.ipynb`. It stops after detection with all figures shown.
**Run all at once:** `python 04_ephys_baseline/run.py` (add `--overwrite`; `--no-figures` skips the figures).

Input is `02_ephys_anomaly-fix/output/analysis_ready.csv`, recordings with `family = spontaneous` only. Gain corrections (`voltage_scale`) are applied. Settings are in `config.yaml`.

## Output (`output/`)

| File | Content |
|---|---|
| `figures/<recording_id>.png` + `.svg` | detection figure, always made (formats: `figure_formats` in config). Whole recording, drawn as a min/max envelope so every spike shows, with the window shaded and spikes marked (grey = flagged); spikes aligned to threshold (black dot); phase plot |
| `spikes.csv` | step 1: one row per detected spike: threshold, peak, amplitude, half-width, upstroke, downstroke, AHP, ISI before, flags |
| `detection.csv` | step 1: one row per recording: spike count, flagged spikes, recording flags, figure path |
| `baseline_features.csv` | step 2: one row per recording (columns below) |
| `log.txt`, `run_info.json` | run log and provenance |

| Column | Meaning |
|---|---|
| `recording_id`, `cell_tag`, `abf_filename` | identifiers |
| `n_baseline_recordings_for_cell` | several baseline recordings of one cell are **not averaged**; decide which to use |
| `window_start_s`, `window_end_s`, `window_duration_s` | the analysis window (last 100 s) |
| `firing_class` | `silent` (≤ `silent_max_spikes` spikes in the window) or `tonic` |
| `n_spikes`, `n_valid_spikes`, `firing_rate_Hz` | all detected spikes count towards the rate |
| `n_isi`, `isi_mean_s`, `isi_median_s`, `isi_cv` | ISI CV = SD / mean, only with ≥ 5 intervals |
| `vm_baseline_mV`, `vm_mean_mV`, `vm_sd_mV`, `fraction_time_excluded` | Vm with spikes removed |
| `n_spikes_waveform`, `ap_*` | AP waveform: median over valid, isolated spikes |
| `abf_tags`, `bias_current_pA`, `flags` | context and recording-level flags |

## Method

### Spike detection (`src/ephys_analysis/spikes.py`)

Adapted from the Allen Institute IPFX `SpikeFeatureExtractor`.

| Step | Definition | Setting |
|---|---|---|
| Candidate | dV/dt rises above the cutoff | `dvdt_cutoff` 20 mV/ms |
| Accepted | peak within 5 ms; peak ≥ −20 mV; peak − threshold ≥ 20 mV | `max_threshold_to_peak_ms`, `min_peak_mV`, `min_height_mV` |
| **Threshold** | walking back from the maximum upstroke: where dV/dt = **5 % of that spike's own upstroke** (interpolated) | `threshold_fraction` 0.05 |
| Half-width | width at (threshold + peak) / 2, both crossings interpolated | |
| Upstroke / downstroke | max / min dV/dt of the spike | |
| AHP | lowest V within 50 ms after the peak (or before the next spike), minus threshold | `ahp_window_ms` |

**Difference from Allen.** Allen takes 5 % of the *mean* upstroke of the sweep. Here it is 5 % of each spike's *own* upstroke, so a drug or adaptation that slows the upstroke does not shift the threshold of the other spikes.

**Why not a fixed dV/dt level as threshold.** The earlier analysis notebook (not in the repository) used the voltage at 20 mV/ms. That value depends on how fast the upstroke is. On these recordings it lies 2–5 mV above the 5 % threshold.

**Flagged spikes.** These are counted, but their waveform is not used:

| Flag | Condition | Meaning |
|---|---|---|
| `slow_upstroke` | upstroke < 50 mV/ms | failing spike, e.g. near depolarisation block |
| `broad` | half-width > 5 ms | |
| `threshold_not_found` | no threshold point found within the search window | |
| `no_repolarisation` | the voltage does not fall back below half-amplitude | |

**Detection margin on the example data:**
- dV/dt noise between spikes: SD 0.4 mV/ms, maximum 3–10 mV/ms.
- Spike upstrokes: 55–180 mV/ms.

### Silent / tonic

A recording is **silent** if it has ≤ `silent_max_spikes` (default 10) spikes in the 100 s window, i.e. ≤ 0.1 Hz. Otherwise it is **tonic**. The firing rate is always reported, so the cut-off can be changed later.

A silent recording is a result, not a failure: it keeps its row, with rate 0 (or the few spikes it has) and Vm.

### Vm

**Vm** is the median of the window after removing each spike, from 5 ms before to 50 ms after its threshold. Removing spikes changes it by < 0.3 mV on the example data, because firing is sparse.

Caveats:
- In a firing cell this is not a true resting potential, so it is called `vm_baseline_mV`.
- The liquid junction potential (+15.9 mV, K-gluconate, 35 °C) was not compensated and is **not** corrected: the true Vm is ≈ 15.9 mV more negative.
- A recording with an amplifier bias current (`bias_current_pA` from 02) is flagged `bias_current`.

### AP waveform

The waveform is the median over **valid, isolated** spikes, meaning spikes ≥ 0.5 s after the previous one; the first spike always counts. Spike shape depends on the preceding interval: during sustained firing the upstroke drops by up to a third. Silent cells usually have no waveform here.

The standard comparison across *all* cells (silent and tonic) will be the first spike at rheobase in the current-step protocol (next subproject).

### Recording flags

| Flag | Meaning |
|---|---|
| `recording_shorter_than_window` / `window_below_minimum` | shorter than 100 s / 60 s; the whole recording is used |
| `abf_tags_present` | the ABF has tags. Read them: e.g. B_0010 is tagged "After washout of 5HT and fixing seal", so it is not a drug-naive baseline. |
| `bias_current` | Vm is not at 0 pA |
| `flagged_spikes=n` | n spikes flagged (see above) |

### Limitations

- The data are filtered at **2 kHz** (MultiClamp Bessel) and sampled at **10 kHz**. This lowers the maximum dV/dt and widens spikes slightly.
- Values are comparable within this dataset, but not with Allen data, which were filtered at 10 kHz and sampled at 50 kHz.
- Not validated against manual measurements yet.

## Current result (example data)

| Recording | Class | Spikes / 100 s | Rate (Hz) | ISI CV | Vm (mV) | AP threshold (mV) | Half-width (ms) | Flags |
|---|---|---|---|---|---|---|---|---|
| A_0000 | silent | 4 | 0.04 | – | −55.7 | −34.4 (1 spike) | 1.79 | |
| B_0001 | tonic | 112 | 1.12 | 0.76 | −46.0 | −36.1 | 1.96 | |
| B_0010 | tonic | 28 | 0.28 | 0.94 | −48.6 | −38.7 | 2.10 | `abf_tags_present` (after 5-HT washout) |

- A_0000's 4 spikes fall within ~200 ms of each other: a single short burst in an otherwise silent cell.
- B_0001 and B_0010 are the same cell. B_0010 was recorded after 5-HT washout.
