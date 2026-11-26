# 02_ephys_anomaly-fix: resolve problems, then organise the files

This step takes the flags from `01_ephys_meta-org`. It resolves the known acquisition problems automatically, asks you to decide on the rest, and then copies **only analysis-ready recordings** into protocol folders.

The original ABFs are never changed. Corrections are *written down* in `analysis_ready.csv`, and the protocol analyses apply them.

**Run step by step:** open `anomaly-fix.ipynb`.
**Run all at once:** `python 02_ephys_anomaly-fix/run.py` (add `--fresh` to clear `output/`).
Re-run whenever you have filled in decisions. Each run rebuilds `output/` from 01's results and your current decisions.

## Workflow

```
01 flags  →  1 auto-resolve  →  2 decision sheet  →  (you fill in decisions.xlsx)  →  3 apply  →  4 organise  →  5 report
```

### 1. Automatic resolution (logged in the report)

| Problem | Condition | Resolution |
|---|---|---|
| Voltage recorded twice, no current (`duplicated_voltage_no_current`) | baseline protocol: spontaneous or modulation timecourse (0 pA) | **voltage_only.** Voltage from the primary ADC. The current is not needed. |
| | current steps or ramp, with no `unexpected_command_sensitivity` or `telegraph_off` flag | **command_as_current.** The current is taken from the command waveform (DAC, telegraphed 400 pA/V). It is labelled as the *command*, not a measured current. |
| Gain error (`possible_gain_error`) | exactly one standard amplifier gain (2, 5, 10, 20, 50, …) brings the median voltage clearly closest to −60 mV, and within −100…−20 mV | **rescale** by that gain. The factor and the evidence are logged. |

Anything that does not meet these conditions goes to the decision sheet. The rules are in `config.yaml` → `auto_resolve`.

### 2. Decision sheet: `decisions.xlsx` (in this folder)

Every other review or needs_fix flag gets one row, with the evidence and a **suggested action**. Open the sheet in Excel and fill in:

| Column | What to enter |
|---|---|
| `decision` | one of `allowed_actions` (dropdown) |
| `value` | `bias_current`: pA (e.g. `-10`); `relabel`: family (e.g. `spontaneous`); `rescale`: gain factor; `notch`: frequencies in Hz (e.g. `1.3` or `1.3; 2.6`); `remove_mains`: number of harmonics, or empty for the default (5) |
| `note` | the reason (**required** for `exclude`) |
| `decided_by`, `decided_on` | optional: left empty, they are filled in on the next run with `decided_by` from `config.yaml` and the run date |

| Action | Effect |
|---|---|
| `accept` | keep as is; the flag does not affect analysis |
| `exclude` | not used in analysis and not copied; stays in the report |
| `relabel` | use another protocol family |
| `bias_current` | record a current set on the amplifier (not visible in the ABF) |
| `rescale` | divide the voltage by a gain factor |
| `voltage_only` / `command_as_current` | as in the automatic rules |
| `remove_mains` | **noise:** subtract fitted 50 Hz sinusoids (and harmonics), re-fitted every 1 s on spike-free samples; spikes are not filtered. For `mains_noise` |
| `notch` | **noise:** narrow zero-phase notch (Q = 30) at the given frequencies, e.g. a stable pump frequency. Rings briefly around fast events, so use it only where needed. For `mains_noise`, `periodic_low_frequency_noise` |

**Noise corrections:**
- **Where they apply:** in memory in 04–07 (current clamp), never to the ABF.
- **What is recorded:** the settings (`noise_corrections` in `config.yaml`) are copied into `analysis_ready.csv`, so every analysis applies exactly the same correction.
- **What shows in the outputs:** the analyses add a flag such as `mains_removed=50Hz_x5` to the recording.
- **Choosing:** `accept` is the right decision when the noise does not affect the measures (compare the spectrum in `01…/output/noise_spectra/`).

New flags are appended on each run. **Rows you have filled in are never changed.** Invalid entries (unknown action, missing value or note) are reported, and that recording stays pending.

### 3–5. Apply, organise, report

- **Recording state:**
  - **ready:** all of its flags are resolved;
  - **excluded:** any decision is `exclude`;
  - otherwise **pending**.
- **Organising:** only ready recordings are copied, SHA-256-verified, to `output/protocol_sorted/<protocol folder>/<recording day>/`. Copies of recordings that are no longer ready are removed.

## Output (`output/`)

| File | Content |
|---|---|
| `issues_report.md` | Readable report: counts per issue; automatically resolved, decided, pending and invalid items; recording-book issues; notes; excluded files. |
| `issues_report.csv` | Every flag from 01 with its `resolution` (`auto_resolved`, `decided`, `pending`, `invalid`, `book`, `info`), action and explanation. |
| `analysis_ready.csv` | Ready recordings with `family`, `voltage_source`, `current_source` (`measured_ADCn` / `command_DACn` / `none`), `voltage_scale`, `bias_current_pA`, noise corrections (`mains_Hz`, `mains_harmonics`, `mains_window_s`, `notch_Hz`, `notch_Q`) and `corrections`. **Protocol analyses read this file.** |
| `protocol_sorted/` | Copies of the ready recordings, by protocol. |
| `log.txt`, `run_info.json` | Run log and provenance. |

## Current result (example data)

- **10 issues resolved automatically:**
  - 6 baseline files → `voltage_only`;
  - 4 step and ramp files → `command_as_current`.
- **3 decisions needed:**
  - B_0002 and B_0004: −10 pA bias noted in the book; suggested `bias_current −10`;
  - B_0005: protocol mismatch; the book says "Recorded in CC… probably lost the cell".
- **14 recordings ready** before those decisions.

## The MultiClamp 700B command sensitivity

The ABF header already contains the command sensitivity, sent by the amplifier through the telegraph:
- **400 pA/V** for Cmd 1 in current clamp;
- **20 mV/V** in voltage clamp.

These are the 700B's standard settings. 01 flags any file where they differ. To confirm by hand on the rig:
- **MultiClamp 700B Commander:** in I-Clamp, check the *External Command Sensitivity* (400 pA/V or 2 nA/V).
- **Clampex:** Configure → Lab Bench → Output Signals → Cmd 1. With telegraphs on, the scale factor shows as telegraphed.

## Note for future recordings

In the current set-up, the second input (`IN 3`) records the membrane **voltage** a second time. To record the injected current, set the MultiClamp **secondary output** to *membrane current* in I-Clamp, and give `IN 3` current units (pA) in Clampex. 01 will then find a measured current and no longer flag these files.
