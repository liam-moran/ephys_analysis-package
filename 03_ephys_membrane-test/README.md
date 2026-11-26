# 03_ephys_membrane-test: final membrane-test values

This step calculates **Cm, Ra, Rt, Rm and tau** for every membrane-test recording (A0 / M1) and saves **only the final values**: one row per recording.

**Run step by step:** open `membrane-test.ipynb`. It first shows one recording in detail (chosen command channel, every pulse edge, fit figure), then analyses all of them.
**Run all at once:** `python 03_ephys_membrane-test/run.py` (add `--overwrite`, or `--figures` for check plots).

Input is `02_ephys_anomaly-fix/output/analysis_ready.csv`: only recordings whose problems are resolved, with `family = membrane_test`. Settings are in `config.yaml`.

## Output: `output/membrane_test_values.csv`

| Column | Meaning |
|---|---|
| `recording_id`, `cell_tag`, `abf_filename` | identifiers |
| `n_pulses`, `n_valid_pulses` | pulse edges analysed and edges passing QC (onset and offset × sweeps) |
| `Rt_MOhm` | median over valid pulse edges |
| **`Ra_MOhm`** | **fast-component Ra**: two-exponential fit that models the amplifier filter (see below) |
| `Rm_MOhm`, `Cm_pF` | Rt − Ra; and the median over valid edges of Qt·Rt / (ΔV·Rm), using that Ra |
| `tau_fast_ms`, `tau_slow_ms`, `fast_fraction`, `ra_fast_status` | the two-exponential fit; status `ok`, or the reason Ra is NaN |
| `Ra_clampex_MOhm`, `Rm_clampex_MOhm`, `Cm_clampex_pF`, `tau_clampex_ms` | Clampex single-exponential values (median over valid edges), for comparison |
| `holding_current_pA`, `filter_Hz` | holding current; telegraphed amplifier filter cutoff |
| `qc_status` | `pass` (≥ 50 % of edges valid and the Ra fit ok), `review`, or `fail` (no valid edge) |
| `pulse_flags` | counts of pulse-level flags |
| `command_dac`, `current_adc`, `method`, `clampex_validation` | how the values were obtained |
| `analysis_run_id`, `package_version` | provenance |

## Method (Clampex Membrane Test algorithm)

Source: pCLAMP 10 User Guide (1-2500-0180 Rev. A), "Membrane Test", pp. 163–166. Each pulse edge (onset and offset, every sweep) is analysed:

| Quantity | Definition | Kind |
|---|---|---|
| I2, I1 | mean current over the last 20 % of the segment before / after the edge | direct |
| **Rt** | ΔV / (I1 − I2) = Ra + Rm | direct |
| **tau** | single-exponential (log-linear) fit of (I − I1) from 80 % to 10 % of the peak | fitted |
| Qt | ∫(I − I1)dt + (I1 − I2)·tau | derived |
| **Ra** | tau · ΔV / Qt | derived |
| **Rm** | Rt − Ra | derived |
| **Cm** | Qt · Rt / (ΔV · Rm) | derived |

For an ideal Ra-(Rm∥Cm) circuit these equations are exact. Tests recover known values within 3 %, and within 15 % after 10 kHz filtering with noise.

**Command channel.** The test pulse is found from the response, not the channel index. In these recordings, DAC 0 (`Cmd 0`) carries a 112 → 32 mV step that evokes no current. The real pulse is on **DAC 1: −10 mV, 25 ms**. The original metadata script used DAC 0, which is why its values were wrong.

**Pulse QC.** A pulse edge is invalid if:
- the fit fails, or has R² < 0.70 (provisional);
- signal-to-noise < 20;
- Rm ≤ 0.

Values outside plausible ranges are warnings only.

### Main Ra: fast component with the filter modelled (decided 2026-11-26)

The transients are **two-exponential**:
- **fast component:** ~0.35–0.7 ms, 75–85 % of the amplitude. This is the soma charging through Ra.
- **slow component:** 2.5–5.5 ms, most likely dendritic membrane charging.

Two simpler approaches both get Ra wrong:
- **Clampex single exponential (80→10 %):** mixes the two components, so τ and Ra come out too high.
- **Extrapolating the fast component back to the edge:** comes out too low, because the MultiClamp 4-pole Bessel filter (2 kHz, read from the telegraph) rounds off the start of the transient.

The main Ra therefore comes from a fit that **models the filter**. The mean sign-normalised transient of all valid edges, from 1 ms before the edge to the end of the segment, is fitted with

I(t) − I2 = Bessel(2 kHz) ∗ [ΔI + A1·exp(−(t−t0)/τ1) + A2·exp(−(t−t0)/τ2)]

**Ra = ΔV / (A1 + A2 + ΔI)**: the current at the edge, before filtering. The free parameters are A1, τ1, A2, τ2 and a small edge shift t0.

**Validation (simulation).** A soma–dendrite model was filtered at 2 kHz with 1 pA noise added:

| Method | Error relative to true Ra |
|---|---|
| Filter-aware fit | +0 to +19 % |
| Clampex | +25 to +45 % |
| Extrapolated fast component | −25 to −35 % |

**Quality check.** If τ1 < 0.1 ms, the 2 kHz filter cannot resolve the fast component. Ra is then NaN, `ra_fast_status` is `fast_component_unresolved`, and the recording goes to `review`.

To make the Clampex values the main ones again, set `ra_model: clampex` in `config.yaml`.

## Current result (example data)

| Recording | Valid | Rt (MΩ) | **Ra (MΩ)** | Rm (MΩ) | Cm (pF) | τ fast / slow (ms) | Ra Clampex (MΩ) |
|---|---|---|---|---|---|---|---|
| A_0005 | 10/10 | 745 | **54.1** | 690 | 36.6 | 0.68 / 5.56 | 75.9 |
| B_0000 | 10/10 | 745 | **26.2** | 719 | 39.1 | 0.35 / 4.28 | 40.4 |
| B_0006 | 0/10 | – | – | – | – | – | – |
| B_0007 | 0/10 | – | – | – | – | – | – |
| B_0008 | 9/10 | 287 | **28.5** | 259 | 27.2 | 0.36 / 2.68 | 38.6 |

- **B_0000 and B_0008:** within the expected 10–30 MΩ.
- **A_0005:** 54 MΩ, with a slower fast component (0.68 ms). This is consistent with poorer access in that recording.
- **B_0006 and B_0007:** no capacitive transient (Rt ≈ 3–4 GΩ), which suggests no whole-cell access.

## Open issues

- **Not validated against Clampex.** To validate, note the live Cm / Ra / Rt / Tau readout from the Clampex Membrane Test window for a few cells and compare it with this table.
- **Ra (decided 2026-11-26).** The fast-component Ra is now the main value. It has only been validated on simulations; compare it with the Clampex Membrane Test readout on the rig for a few cells.
- **Holding potential.** `Vm2_sec` reads about −22 mV at a 0 mV command. Please confirm the holding potential.
