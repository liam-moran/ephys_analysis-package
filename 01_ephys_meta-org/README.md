# 01_ephys_meta-org: inventory, recording book, protocols and checks

This step runs on **all** ABF files. For each file it:
- reads the file header;
- matches the file to the recording book;
- works out which protocol was actually recorded;
- checks the file against what the data should look like.

It only **flags** problems. Nothing is changed, copied or analysed here. Sorting into protocol folders happens in `02_ephys_anomaly-fix`, once the problems are resolved.

**Run step by step:** open `meta-org.ipynb` and run the cells in order. There is a review cell after each step.
**Run all at once:** `python 01_ephys_meta-org/run.py --fresh`.

Settings are in `config.yaml`: data paths, recording-book sheets, classification rules, anomaly thresholds and the `expected` set-up.

## Steps

1. **Inventory.** Opens every `.abf` under `data/` read-only and records:
   - channel names and units;
   - sampling rate, sweeps and duration;
   - tags;
   - a summary of the command waveform;
   - the amplifier telegraph settings;
   - a SHA-256 fingerprint of the file.
2. **Recording book.** Matches each ABF to the Recording Log by file name. The Cell Tag is taken only from the book, and duplicates are never guessed.
3. **Protocol.** The ABF decides; the label is only compared afterwards.
   - The clamp mode comes from the channel units: pA means voltage clamp, mV means current clamp.
   - The protocol family comes from the command waveform:

     | Clamp mode | Command waveform | Family |
     |---|---|---|
     | Voltage clamp | small short pulse (≤ 20 mV, ≤ 0.2 s) | membrane test |
     | Voltage clamp | flat, ≥ 30 s | EPSC (catalogue only) |
     | Current clamp | ramp | ramp |
     | Current clamp | ≥ 3 sweeps with varying steps | current steps |
     | Current clamp | flat, ≥ 600 s | modulation timecourse |
     | Current clamp | flat, 10–600 s | spontaneous |

   - Statuses:
     - `confirmed`: ABF and label agree;
     - `probable`: no label;
     - `mismatch`: ABF and label disagree;
     - `ambiguous`: no rule, or several rules, matched;
     - `unsupported`: unreadable file or unknown units.
4. **Noise.** A power spectrum per recording (see "Noise check" below), with one spectrum figure each.
5. **Checks.** See the table below.
6. **Save.** Writes the outputs and re-hashes every original to prove nothing was changed.

## What is checked

Every flag has a **severity**:
- **needs_fix:** must be corrected before analysis.
- **review:** a person must decide.
- **info:** noted only.

| Source | Check (what the data should look like) | Flag | Severity |
|---|---|---|---|
| Recording book | each ABF has exactly one Recording Log row | `book_unmatched_abf`, `book_duplicate_book_entry`, `book_duplicate_filename` | review |
| Recording book | Cell Tag present | `missing_cell_tag` | review |
| Recording book | protocol label agrees with the acquisition | `protocol_mismatch` / `_ambiguous` / `_unsupported` | review |
| Recording book | 'Date Recorded' equals the ABF date | `book_date_mismatch` | review |
| Recording book | in current clamp, no comment mentions current set on the amplifier (e.g. "-10pA holding"); the value is parsed | `book_mentions_current` | review |
| Recording book | no spreadsheet errors (`#REF!`, …) | `recording_book_spreadsheet_error_value` | review |
| ABF header | amplifier telegraph on (gains and mode confirmed by the MultiClamp) | `telegraph_off` | review |
| ABF header | amplifier clamp mode agrees with the channel units | `telegraph_mode_mismatch` | review |
| ABF header | command sensitivity: 400 pA/V in current clamp, 20 mV/V in voltage clamp (MultiClamp 700B) | `unexpected_command_sensitivity` | review |
| ABF header + command | baseline protocols (spontaneous, timecourse) inject 0 pA: holding and command are 0 | `baseline_not_zero_current` | needs_fix |
| Signal | current clamp records a current channel, not the same voltage twice (correlation ≥ 0.90) | `duplicated_voltage_no_current` | needs_fix |
| Signal | current-clamp voltage is physiological (median −120…20 mV; 1–99 % span ≤ 300 mV); otherwise a gain error such as ~20× | `possible_gain_error` | needs_fix |
| ABF | file readable, units known, sampling rate ≥ 5 kHz | `unreadable_file`, `unknown_channel_units`, `low_sample_rate` | needs_fix / review |
| Notes | drug tag in a spontaneous file; recording-book rows without an ABF | `protocol_note`, `book_rows_without_abf` | info |
| Noise | mains hum (50 Hz + harmonics, peak-to-peak) below the review level (current clamp 1 mV, voltage clamp 10 pA) | `mains_noise` | info / review |
| Noise | no narrow low-frequency peak (0.05–20 Hz: perfusion / pump), in recordings without current injection | `periodic_low_frequency_noise` | info / review |
| Noise | RMS noise (1–1000 Hz) below 3 mV / 20 pA | `high_rms_noise` | review |
| Noise | no brief (≤ 1 ms), large (≥ 2 mV / 20 pA) transients, e.g. from switching equipment | `transient_artefacts` | info / review |
| Noise | spectrum could be computed (quiet stretch ≥ 1 s) | `noise_not_assessed`, `noise_check_failed` | info / review |

A recording's `status` is its worst flag.

**Limitation:** a current set on the amplifier itself (front panel or Commander "holding") is **not** stored in the ABF. It can only be detected through recording-book comments, so always note it in the book.

## Noise check

Measured on the **quiet stretches** of each recording: the command at its holding level, ≥ 0.3 s (current clamp) or 20 ms (voltage clamp) after any command change, and in current clamp with spikes removed. Settings and thresholds: `config.yaml` → `noise` (units mV in current clamp, pA in voltage clamp).

| Measure (`noise.csv`) | How |
|---|---|
| spectrum | Welch power spectral density, 10 s segments (0.1 Hz resolution), averaged over the quiet stretches; one figure per recording in `output/noise_spectra/` (log–log, mains harmonics marked) |
| `rms` | SD of the quiet signal in 1–1000 Hz (drift excluded). **In current clamp it includes real synaptic activity**, so it is an upper bound of the recording noise |
| `mains_50Hz_pp` … `mains_250Hz_pp`, `mains_pp` | peak-to-peak amplitude of the sinusoid at each mains harmonic (power above the local floor, A = √(2P), pp = 2A), and their sum |
| `lowf_peaks` | narrow peaks (≥ 10 dB prominent, ≤ 0.5 Hz wide) in 0.05–20 Hz with their amplitude. Only assessed in recordings without current injection (steps and ramps are dominated by their own responses) |
| `artefacts_per_min` | brief transients: > 8 robust SD and ≥ 2 mV / 20 pA from a 5 ms running median, lasting ≤ 1 ms |

**What the flags lead to:**
- An `info` flag only notes the noise.
- A `review` flag needs a decision in 02: `accept`, `remove_mains`, `notch` or `exclude`.
- **Nothing is filtered here.** A correction is applied only when decided in 02, only in memory, in 04–07 (current clamp).

**Why not filter by default:**
- **The data are already filtered:** 2 kHz Bessel on the amplifier.
- **Extra filtering changes AP shape:** a lowpass slows the upstroke and widens the half-width; a notch rings around fast events.
- **Medians absorb small hum:** small mains hum averages out of the Vm medians and binned timecourses.

**Never high-pass a current-clamp recording to remove pump noise:** it removes the membrane potential and the drug responses. The bench fix (grounding the pump, a drip chamber or gravity perfusion, keeping the pump away from the rig) is always better than a filter. The spectrum shows whether it worked.

## Output (`output/`)

| File | Content |
|---|---|
| `recordings.csv` | One row per ABF: identifiers, Cell Tag, protocol, status, flags, channels, amplifier settings, recording-book fields. |
| `cells.csv` | One row per Cell Tag, linking all its recordings (no averaging). |
| `flags.csv` | Every flag, with severity, detail and value. |
| `noise.csv`, `noise_spectra/` | Noise measures per recording (see above) and one spectrum figure (PNG + SVG) each. |
| `log.txt`, `run_info.json` | Run log; versions and config of the latest run. |

## Current result (example data, 17 files)

- **Noise:** 50 Hz mains hum (with 150 and 250 Hz) in every recording.
  - **Current clamp:** about 0.2–0.3 mV peak-to-peak, flagged `info`. B_0004 (1.0 mV) and B_0005 (2.0 mV) are flagged `review`.
  - **Voltage clamp:** 2–4 pA.
  - **Not found:** no low-frequency pump peaks and no transient artefacts.
  - **Other lines:** the spectra show further narrow lines that are not mains harmonics (e.g. ~35 Hz, many between 100 Hz and 4 kHz). They probably come from other equipment; they are small and not flagged.
  - **B_0004 checked:** removing the hum there (tested in memory) leaves the 5-HT results unchanged (ΔVm −6.62 → −6.61 mV).

- All 17 files matched the recording book.
- 16 are `confirmed`. **B_0005** is a `mismatch`: labelled `A3_epsc` but recorded in current clamp (book: "Recorded in CC, super weird").
- **All 10 current-clamp files:** `duplicated_voltage_no_current`. ADC 1 (`IN 3`) is a copy of the voltage, offset by +36 to +56 mV, and no current was recorded.
- **B_0002 and B_0004:** `book_mentions_current` (−10 pA). In both files the Clampex holding and command are 0 pA, so the bias was set on the amplifier.
- The amplifier settings match the expected set-up in every file: telegraph on, modes consistent, 400 pA/V and 20 mV/V. There are no gain errors.
- One `#REF!` in the recording book (Slices sheet, row 3).
