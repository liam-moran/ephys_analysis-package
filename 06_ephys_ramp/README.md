# 06_ephys_ramp: ramp rheobase, up/down firing, hysteresis, Rin, rebound

This step analyses the **ramp recordings** (A4 / M4). It measures:
- the current at which firing starts on a rising ramp (a finer rheobase than the 50 pA steps);
- how firing on the way up compares with the way down (**hysteresis**);
- input resistance from a negative ramp;
- **post-inhibitory rebound** after an abrupt release from −100 pA.

**Order (notebook and run.py), as in 04 and 05:**
1. **Detect** spikes and attenuated events in every sweep and save one **detection figure per recording** (PNG + SVG).
2. **Review** the figures.
3. **Features**, computed from the *same* spike table.

**Run step by step:** open `ramp.ipynb`.
**Run all at once:** `python 06_ephys_ramp/run.py` (add `--overwrite`; `--no-figures` skips the figures).

**Input:** `02_ephys_anomaly-fix/output/analysis_ready.csv`, recordings with `family = ramp`.
- **Injected current:** the **command** waveform (`command_as_current`), plus any bias current decided in 02.

## Protocol as recorded (example data)

| Segment | Time in sweep | Current |
|---|---|---|
| pre | 0 – 2.34 s | 0 pA |
| **up** | 2.34 – 2.84 s | 0 → +100 pA (200 pA/s) |
| **down** | 2.84 – 3.35 s | +100 pA for 5 ms, then +100 → 0 pA |
| mid | 3.35 – 5.35 s | 0 pA |
| **neg** | 5.35 – 5.85 s | 0 → −100 pA |
| **post** | 5.85 – 22 s | abrupt return to 0 pA (rebound) |

There are 3 sweeps of 22 s. The segments are found in each sweep's command waveform, not hard-coded.

## Output (`output/`)

| File | Content |
|---|---|
| `figures/<recording_id>.png` + `.svg` | detection figure (described below) |
| `spikes.csv` | one row per spike: sweep, period, current at threshold (`I_pA`), waveform, flags |
| `attenuated_events.csv` | peaks that are not spikes (depolarisation block), with period and current |
| `detection.csv` | one row per recording: spikes per period, attenuated events, flags, figure |
| `ramp_sweeps.csv` | one row per sweep (features below) |
| `ramp_features.csv` | one row per recording: **median over the sweeps**; rheobase and AP as the **mean over the quiet sweeps** |
| `ap_waveforms.csv` | the waveform (−5 to +15 ms around threshold: t_ms, v_mV, dvdt_mV_ms) of every spike used for the AP, for the average-AP and phase-plane figures (10) |
| `log.txt`, `run_info.json` | run log and provenance |

**The detection figure has three panels:**
- **Ramps:** all sweeps overlaid, with the command below. ▼ marks a spike (grey = flagged) and ○ an attenuated event.
- **Whole sweeps:** stacked, so the rebound is visible.
- **Aligned spikes:** every spike aligned to its threshold, coloured by period (red = up-ramp, orange = down-ramp, purple = after release).

## Method

Spike detection is the same as in 04 and 05 (`spikes.py`; the `spikes:` settings are identical). Attenuated events follow 05.

### Per sweep (`ramp_sweeps.csv`)

| Column | Definition |
|---|---|
| `vm_pre_mV`, `rate_pre_Hz` | median V in the 0.5 s before the up-ramp (spikes removed); spontaneous rate before the ramp |
| `I_first_spike_pA` | current at the threshold of the first spike on the up-ramp |
| `n_spikes_before_ramp`, `rheobase_sweep_ok` | spikes in the 0.5 s before the up-ramp (`rheobase_quiet_s`); the sweep counts for the rheobase only if there are none |
| `v_threshold_first_mV`, `latency_first_spike_ms` | its threshold, and the time from the start of the up-ramp |
| `n_spikes_up`, `n_spikes_down`, `max_inst_rate_up_Hz` | spikes on the rising and falling ramp; highest instantaneous rate on the rising ramp |
| `I_last_spike_down_pA` | current at the last spike on the down-ramp |
| `hysteresis_pA` | `I_first_spike_pA` − `I_last_spike_down_pA`. **> 0:** firing continues to lower currents on the way down (e.g. persistent inward current). **< 0:** firing stops earlier on the way down (e.g. inactivation, block). NaN if no spike on either side |
| `rin_ramp_MOhm` | slope of V against I over the negative ramp (spikes removed, first 50 ms skipped for the membrane lag) |
| `v_min_neg_mV` | lowest V on the negative ramp |
| `rebound_peak_mV` | max V (spikes removed) in the first 500 ms after release, minus vm_pre |
| `n_rebound_spikes_short`, `n_rebound_spikes_all` | spikes after release: within 5 s, and to the end of the sweep |
| `latency_first_rebound_spike_ms`, `n_rebound_bursts` | as in 05 |
| `n_attenuated_up`, `n_attenuated_down` | attenuated events on the rising and falling ramp |

**Counting rules:**
- Flagged spikes (slow upstroke or broad) **count** as spikes here, as in the other subprojects.
- Attenuated events do **not** count as spikes.

### Per recording (`ramp_features.csv`)

- **Rheobase (`rheobase_pA`):** the **mean** `I_first_spike_pA` over the sweeps with no spike in the 0.5 s before the ramp. A sweep in which the cell was already firing is left out, because its first ramp spike may be spontaneous rather than evoked. Also reported: `rheobase_sd_pA`, `n_sweeps_rheobase` (sweeps used) and `n_sweeps_rheobase_quiet` (quiet sweeps). Flags: `no_quiet_sweep_for_rheobase` (NaN), `quiet_sweep_without_spike` (a quiet sweep reached +100 pA without a spike). This is the cell's rheobase; the 05 steps are too coarse (50 pA).
- **Other features:** the median of each sweep feature over the sweeps. The sweeps repeat the same protocol within one recording, so this is a within-recording summary, not an average over recordings or cells.
- **AP (`ap_*`, the cell's AP measure, used in 08–10):** the **mean** over the first **valid** up-ramp spike of each **quiet** sweep (the sweeps of the rheobase), so every cell is measured at its own threshold current (`n_sweeps_ap`). If no quiet sweep has a valid spike, the first valid up-ramp spike of any sweep is used (flag `ap_from_non_quiet_sweep`); none at all: NaN (`no_valid_ramp_spike`). A slow ramp inactivates some sodium channels, so ramp APs can have a slightly higher threshold and slower upstroke than step APs (05, kept as a secondary measure).
- **Flags:**
  - `bias_current`, `current_from_command`, `no_negative_ramp`;
  - `no_spikes_on_up_ramp`, `spontaneous_before_ramp`, `depolarisation_block`;
  - `no_quiet_sweep_for_rheobase`, `quiet_sweep_without_spike`;
  - `flagged_spikes=n`.

## Current result (example data)

| | A_0002 | B_0003 |
|---|---|---|
| Vm before the ramp | −45.9 mV | −52.2 mV |
| **Rheobase** (mean over quiet sweeps) | **13.3 pA** (3 of 3 sweeps, SD 1.0) | **20.2 pA** (3 of 3 sweeps, SD 5.9) |
| First spike at (median sweep) | 12.8 pA (threshold −32.6 mV, 64 ms into the ramp) | 18.5 pA (−39.3 mV, 92 ms) |
| Spikes up / down | 4 / 0 | 9 / 3 |
| Max rate on the up-ramp | 19 Hz | 28 Hz |
| Hysteresis | – (no down-ramp spikes) | −58 pA (last spike on the way down at 75 pA) |
| Depolarisation block | yes: 5 attenuated events on the up-ramp | no, but late spikes broaden (flagged) |
| Rin (negative ramp) | 859 MΩ | 636 MΩ |
| **Rebound after release** | **8 spikes within 5 s**, first at 243 ms | none: one spike at 1.8 s, as in spontaneous firing |
| AP at ramp rheobase (mean of 3 quiet sweeps; threshold / half-width / amplitude) | −32.5 mV / 2.07 ms / 82.7 mV | −39.7 mV / 1.80 ms / 88.4 mV |

**How these compare with the other subprojects:**
- **Rheobase:** the ramp gives 13 and 20 pA; the 05 steps can only say "between 0 and 50 pA". In cell B the three sweeps differ more (15–27 pA): it fires spontaneously, so the first ramp spike is less tightly locked to the current.
- **Same cells in 05:**
  - Cell A (A_0001 there) also goes into block during the steps.
  - Cell B (B_0002 there) fires its rebound spike after the hyperpolarising *steps*, but not after this ramp.
- **Rin:** ramp Rin (640–860 MΩ, from 0 to −100 pA, near rest) is higher than step Rin (330–390 MΩ, a fit from −50 to −150 pA). The difference points to inward rectification at more negative potentials. They are different measures, so do not pool them.

## Limitations

- The injected current is the **command**, not a measured current.
- **Membrane lag:** with a membrane time constant of tens of ms, V lags a 200 pA/s ramp by a few pA. `I_first_spike_pA` is therefore slightly overestimated.
- The 2 kHz filter and 10 kHz sampling affect absolute spike values (see 04).
- Not validated against manual measurements.
