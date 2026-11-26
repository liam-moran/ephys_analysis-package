# 05_ephys_steps: current steps (sag, Rin, rheobase, F–I, rebound)

This step analyses the **current-step recordings** (A3 / M3). From each recording it extracts:
- sag and input resistance (hyperpolarising steps);
- rheobase, F–I and AP waveform (depolarising steps);
- **post-inhibitory rebound** in the 25 s after each step.

**Order (notebook and run.py), as in 04:**
1. **Detect** spikes (and attenuated events) in every sweep, and save one **detection figure per recording (PNG + SVG)**.
2. **Review** the figures. If detection is wrong, change `spikes:` in `config.yaml` and detect again.
3. **Features** are computed from the *same* spike table.

**Run step by step:** open `steps.ipynb`.
**Run all at once:** `python 05_ephys_steps/run.py` (add `--overwrite`; `--no-figures` skips the figures).

Input is `02_ephys_anomaly-fix/output/analysis_ready.csv`, with `family = current_steps`.
- **Step current.** The current was not recorded (correction `command_as_current`), so the injected current is the **command** waveform of the DAC named in `current_source`.
- **Bias current.** A bias decided in 02 (`bias_current_pA`) is added to give `I_total_pA`. Rin and F–I use the step amplitude, so the bias cancels.

## Protocol as recorded (example data)

| | |
|---|---|
| Sweeps | 6 × 30 s |
| Step | 500 ms, from 3.97 to 4.47 s |
| Amplitudes | −150, −100, −50, 0, +50, +100 pA |
| After the step | 25 s at 0 pA (rebound) |

The step window and amplitudes are read from each file's command waveform, never hard-coded:
- **Window:** taken from the sweep with the largest step.
- **Amplitude:** the command during the step minus the command just before it.

## Output (`output/`)

| File | Content |
|---|---|
| `figures/<recording_id>.png` + `.svg` | detection figure (see below) |
| `spikes.csv` | one row per spike: sweep, step current, period (`pre` / `step` / `post`), waveform, flags |
| `attenuated_events.csv` | peaks that are **not** spikes (see below) |
| `detection.csv` | one row per recording: steps found, spike counts per period, attenuated events, flags, figure |
| `step_sweeps.csv` | one row per sweep (features below) |
| `step_features.csv` | one row per recording (features below) |
| `log.txt`, `run_info.json` | run log and provenance |

**The detection figure has three panels:**
- **Top:** the step period of every sweep (colour = current). ▼ marks a spike (grey = flagged); ○ marks an attenuated event.
- **Right:** whole sweeps stacked, so rebound firing over 25 s is visible.
- **Bottom:** every spike aligned to its threshold (black dot).

## Method

**Spikes** use the same detector and settings as 04 (`src/ephys_analysis/spikes.py`; keep `spikes:` identical in both configs). They are detected over the whole sweep and labelled by period: `pre` (spontaneous, before the step), `step`, or `post` (after the step).

**Attenuated events.** During strong depolarising steps, some cells fire one or two full spikes and then go into **depolarisation block**. The following peaks shrink and rise slowly: in A_0001 they reach +7 to +20 mV at 2–17 mV/ms. By the spike definition (dV/dt ≥ 20 mV/ms) these are **not spikes**. Instead of ignoring them silently:
- every voltage peak ≥ −20 mV with a prominence ≥ 20 mV that is not a spike is listed in `attenuated_events.csv`;
- they are counted per sweep (`n_attenuated_step`);
- the recording is flagged `depolarisation_block`, and `first_step_with_block_pA` gives the first step where it happens.

They are **not** counted in the firing rate. Whether they should be is open (see below).

### Per sweep (`step_sweeps.csv`)

| Column | Definition |
|---|---|
| `I_step_pA`, `I_total_pA` | step amplitude (command), and step + bias |
| `vm_pre_mV` | median V in the 0.5 s before the step, spikes removed |
| `n_spikes_pre`, `rate_pre_Hz` | spontaneous spikes before the step |
| `n_spikes_step`, `rate_step_Hz` | spikes during the step / step duration |
| `latency_first_spike_ms` | step onset to the first threshold |
| `isi_cv_step` | ISI CV during the step (≥ 3 intervals) |
| `adaptation_index` | Allen definition: mean of (ISIₙ₊₁ − ISIₙ)/(ISIₙ₊₁ + ISIₙ). Needs ≥ 4 spikes (3 intervals, 2 ratios); with fewer it is **NaN**, never 0 |
| `v_min_mV` | minimum V during the whole step (Allen peak deflection) |
| `v_ss_mV`, `delta_v_ss_mV` | median V in the last 100 ms of the step (spikes removed), and minus vm_pre |
| `sag_ratio` | (v_ss − v_min) / (vm_pre − v_min), hyperpolarising steps only; 0 = no sag |
| `rebound_peak_mV` | max V (spikes removed) in the first 500 ms after the step, minus vm_pre |
| `n_rebound_spikes_short`, `n_rebound_spikes_all` | spikes after the step: within 5 s, and to the end of the sweep (25 s) |
| `latency_first_rebound_spike_ms`, `n_rebound_bursts` | first spike after the step; bursts (MaxInterval: ISI ≤ 100 ms, ≥ 2 spikes, merged if < 200 ms apart) |
| `n_attenuated_step` | attenuated events during the step |

### Per recording (`step_features.csv`)

| Column | Definition |
|---|---|
| `rin_MOhm` | slope of `delta_v_ss_mV` against `I_step_pA` over the hyperpolarising steps (linear fit, ≥ 2 steps) |
| `first_spiking_step_pA` | smallest positive step with ≥ 1 spike during the step. **Coarse:** to the step increment (`step_increment_pA`, 50 pA here). **Not the rheobase:** the cell's rheobase comes from the ramp (06, `rheobase_pA`). It is used here as the start of the F–I fit and for the AP waveform |
| `fi_slope_Hz_per_pA`, `max_rate_step_Hz` | linear fit of `rate_step_Hz` from the first spiking step upwards; highest step rate |
| `first_step_with_block_pA` | first positive step with attenuated events |
| `sag_ratio`, `v_min_mV`, `rebound_*` | at the most negative step (−150 pA) |
| `adaptation_index`, `adaptation_step_pA` | from the step with the most spikes (≥ 4). If no step reaches 4 spikes: NaN, flag `too_few_spikes_for_adaptation` |
| `ap_*` | waveform of the **first valid spike at the first spiking step**: the standard AP comparison across cells, silent or tonic |
| `flags` | `bias_current`, `current_from_command`, `spontaneous_before_step`, `spikes_during_0pA_step`, `depolarisation_block`, `no_spikes_at_any_positive_step`, `rin_needs_2_hyperpolarising_steps`, `flagged_spikes=n` |

## Current result (example data)

| Recording | Vm pre (mV) | Rin (MΩ) | First spiking step (pA) | F–I (Hz/pA) | Max rate (Hz) | Block from | Sag (−150 pA) | Rebound (−150 pA, 5 s) | Adaptation | AP threshold / half-width, first spiking step |
|---|---|---|---|---|---|---|---|---|---|---|
| A_0001 | −57.2 | 385 | 50 | 0.04 | 6 | +50 pA | 0.08 | none | NaN (≤ 3 full spikes) | −32.4 mV / 2.56 ms |
| B_0002 (bias −10 pA) | −58.1 | 331 | 50 | 0.16 | 28 | – | 0.02 | 1 spike at 203 ms | 0.06 (at +100 pA) | −41.5 mV / 1.59 ms |

- **Little sag** in either cell at −150 pA (whole-step minimum).
- **Rebound activity:**
  - B_0002 fires a rebound spike ~0.2 s after −150 and −100 pA.
  - Several sweeps show spontaneous bursts 5–25 s after the step. These are counted in `n_rebound_spikes_all`; compare them with `rate_pre_Hz` before calling them rebound.
- **A_0001 goes into depolarisation block** from +50 pA. Its F–I slope (0.04 Hz/pA) therefore reflects block, not low excitability.
- **Rin differs from 03.** Rin from the steps (330–390 MΩ) is lower than the membrane-test Rt (~745 MΩ, voltage clamp). The two are measured differently and are not expected to match exactly.

## Decisions (2026-11-26)

- **Sag:** Allen style. v_min is the minimum over the whole step (`sag_window_s: null`; a number restricts it to the first N s).
- **Rebound:** "short" rebound spikes within **5 s** of the step end. The peak depolarisation is taken within 500 ms, and the steady state is the last 100 ms of the step.
- **Adaptation:** Allen index. It is NaN with fewer than 4 spikes, and the per-recording value comes from the step with the most spikes.
- **Depolarisation block:** attenuated events are **reported separately**. They are not counted in the firing rate or F–I.
- **Rheobase resolution** stays coarse with 50 pA steps. The ramp protocol can refine it.

## Limitations

- The injected current is the **command**, not a measured current.
- The 2 kHz filter and 10 kHz sampling affect absolute spike values (see 04).
- Not validated against manual measurements.
