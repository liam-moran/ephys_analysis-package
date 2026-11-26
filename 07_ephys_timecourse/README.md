# 07_ephys_timecourse: drug applications (baseline, drug and washout)

This step analyses the **drug timecourse recordings** (A5 / M6). Each is a long gap-free current-clamp recording (~20 min, 0 pA) in which a drug is washed in and out. Tags in the ABF mark the events.

A recording can hold **several drug applications** (e.g. 5-HT, washout, then SP; or a compound layered on top of another). Usually there is one drug per recording, and other drugs are in separate recordings of the same cell.

For each **application** it gives:
- the 04 baseline features in its **periods** (baseline, drug, washout; plus pre-tag for layered applications);
- the **changes** from baseline;
- the **lowest and highest point** after the tag, relative to the baseline membrane potential, and the **peak response** (whichever is larger);
- a **binned timecourse** aligned to drug arrival.

**Order (notebook and run.py), as in 04–06:**
1. **Detect** spikes over the whole recording and save one **detection figure per recording** (PNG + SVG).
2. **Review** the tags, periods and spikes in the figures.
3. **Features**, computed from the *same* spike table.

**Run step by step:** open `timecourse.ipynb`.
**Run all at once:** `python 07_ephys_timecourse/run.py` (add `--overwrite`; `--no-figures` skips the figures).

**Inputs:**
- `02_ephys_anomaly-fix/output/analysis_ready.csv`, with `family = modulation_timecourse`.
- `01_ephys_meta-org/output/recordings.csv`, for the recording-book columns `Drug 1`, `Drug 2`, … (used for the drug name only when the tag names no compound), `Drug Effects` and `Protocol Comments` (context only).

## Tags, applications and periods

**Tags** are read in time order and matched by keyword (case-insensitive), set in `config.yaml`:

| Tag | Rule | Example data |
|---|---|---|
| **Drug** | contains a `drug_keywords` entry (hit bath, in bath, bath, drug, application, applied, wash in) and no washout keyword | "5ht hit bath" at 154 s; "5HT hit bath" at 168 s |
| **Washout** | contains a `washout_keywords` entry (washout, wash out, wash-out) | "washout" at 741 s and 883 s |
| other | anything else: listed in `all_tags`, counted (`other_tags=n`), not used | – |

If no tag matches a keyword, the tags are assigned by order (drug, washout, drug, …) and flagged `tags_assigned_by_order`.

**Drug names.**
- **Source:** read from the tag text with the aliases in `drug_names` (config), as whole words: "5ht hit bath" → 5-HT, "SP + TTX in bath" → SP, TTX. Add every compound you use to that list.
- **Fallback:** if a tag names no known compound, the recording book is used (`Drug 1` for the first solitary application, `Drug 2` for the second, …; flag `drug_name_from_book`). Otherwise the name is `unknown` (flag `drug_name_unknown`). Names are never guessed.
- **Mismatch flags:** `drug_name_mismatch_book` when tags and book disagree, and `drug_name_mismatch_protocol` when the protocol name names a compound that no tag does (e.g. `A5_CC_5HT_Timecourse` with an "SP" tag).

**Applications.** Every drug tag starts one application, numbered 1, 2, … in the recording:

| Type | When | Condition | Baseline |
|---|---|---|---|
| **Solitary** | the previous application (if any) was washed out (washout tag) before this drug tag | the compound(s) in this tag, e.g. `SP` | its **own** last 100 s before the tag (not earlier than the previous washout tag). Flag `after_previous_application` for application 2, 3, … |
| **Layered** | a drug tag while the previous compound is still in the bath (no washout tag in between) | the cumulative set, e.g. `5-HT + Ba2+` | the **initial** baseline of the chain (before its first drug tag). The last 100 s before this tag is also measured as `pre_tag`. Flag `layered_on_previous_compound` |

**Periods of each application.** Baseline, pre-tag and washout are 100 s each, the same as the 04 baseline window. The drug period is 200 s:

| Period | Window | Flag if not possible |
|---|---|---|
| **baseline** | see the table above | `baseline_period_short` (< 60 s) |
| **pre_tag** | layered only: last 100 s before its tag (not before the previous tag) | `pre_tag_period_short` |
| **drug** | **from 60 s after the drug tag, 200 s long** (`drug_start_after_tag_s`, `drug_period_s`; decided 2026-11-26), cut at the **next tag** of any kind. The same window for every cell, so it is not chosen from the response | `drug_period_cut_by_next_tag`, `no_drug_period` |
| **washout** | last 100 s before the next drug tag after the washout tag (or before the end of the recording); a layered chain shares one washout | `no_washout_tag` |

A window can serve two roles. For example, the washout of 5-HT is also the baseline of a following solitary SP application. The figure legend then shows both.

## Output (`output/`)

| File | Content |
|---|---|
| `figures/<recording_id>.png` + `.svg` | detection figure (described below) |
| `spikes.csv` | one row per spike, with its `period` as `<application>:<period>` (e.g. `2:drug`; empty outside all periods) |
| `detection.csv` | one row per recording: tags, applications, spikes, flags, figure |
| `period_features.csv` | one row per recording × application × period: the 04 baseline features (Vm with spikes removed, `vm_sd_mV`, rate, class, ISI CV, AP waveform) |
| `timecourse_summary.csv` | **one row per application** (a recording without a drug tag keeps one row, `application` = 0): `application`, `condition`, `condition_slug`, `compounds_added`, `layered`, `baseline_reference`, `name_source`, `book_drugs`, tags, period windows, period values, changes, lowest / highest / peak point, baseline drift, recording-book comments, flags. 08 makes one column set per `condition` from it |
| `timecourse_bins.csv` | 10 s bins: spike count, rate and Vm (spikes removed), with time relative to the **first** drug tag |
| `log.txt`, `run_info.json` | run log and provenance |

**The detection figure has four panels:**
1. The whole recording, with all tags (dashed lines), the periods of every application (shaded, labelled `#n condition: period`) and spikes (▼).
2. Binned firing rate against time from the first drug tag.
3. Binned Vm against time from the first drug tag, with the **peak response** of each application (▼ lowest for a hyperpolarisation, ▲ highest for a depolarisation).
4. The spikes of each period aligned to threshold, with a phase plot.

## Method

- **Period features:** the 04 definitions (`baseline.baseline_features`):
  - Vm: median with each spike removed (−5 to +50 ms around threshold);
  - firing rate;
  - silent / tonic: ≤ 10 spikes per 100 s. The cut-off scales with period length (≤ 20 in the 200 s drug period; `silent_max_spikes_used`), so the same 0.1 Hz rule applies to every period;
  - ISI CV (≥ 5 intervals);
  - AP waveform from valid, isolated spikes.
- **Changes:** `delta_drug_*` = drug − baseline and `delta_washout_*` = washout − baseline, for Vm, rate, ISI CV, AP threshold, half-width, amplitude and AHP.
- **Layered applications:** `delta_drug_*` is measured from the **initial** baseline, so it includes what the earlier compound(s) still do. The change caused by the added compound alone is `delta_drug_vs_pre_tag_*` (drug − pre-tag). Example: SP depolarises by 5 mV and Ba²⁺ added on top brings Vm 3 mV back down. Then `delta_drug_vm_baseline_mV` = +2 mV and `delta_drug_vs_pre_tag_vm_baseline_mV` = −3 mV.
- **Lowest point after the tag:** taken from the 10 s bins between this drug tag and the next tag, smoothed with a running mean over 3 bins (30 s).
  - `drug_min_vm_mV`: the lowest smoothed Vm.
  - `drug_min_delta_vm_mV`: that Vm minus the baseline-period Vm (the drop relative to the membrane potential before the drug).
  - `drug_min_time_s`: when it happens, relative to drug arrival.
  - `drug_max_vm_mV`, `drug_max_delta_vm_mV`, `drug_max_time_s`: the same for the highest point, which is the relevant extreme for a depolarising drug.
  - `drug_peak_delta_vm_mV`, `drug_peak_time_s`, `drug_peak_direction`: whichever of the lowest and highest point is larger in size, so one column describes the response of both hyperpolarising (5-HT) and depolarising drugs.
  - `drug_max_rate_Hz`, `drug_min_rate_Hz`: the highest and lowest smoothed firing rate during the drug.

  These describe the extreme of the response. They do **not** set the drug period. That stays a fixed window, so the period comparison is not biased by picking the extreme in each cell.
- **Baseline drift:** `baseline_drift_mV_per_min` is the slope of binned Vm over the baseline period. The recording is flagged `unstable_baseline` above 2 mV/min.
- **Bias current:** a bias decided in 02 is flagged `bias_current`. Vm is then not the 0 pA potential, but changes within the recording remain valid.

## Current result (example data: 5-HT)

| | A_0004 | B_0004 (−10 pA bias) |
|---|---|---|
| Firing | silent in every period (8 spikes ~800–1000 s, outside the periods) | silent in every period |
| Drug period (60–260 s after the tag) | 214–414 s | 228–428 s |
| Vm baseline → drug → washout | −52.1 → −63.9 → −43.5 mV | −62.7 → −69.3 → −67.3 mV |
| **Change in the drug period** | **−11.8 mV** | **−6.6 mV** |
| **Lowest point (relative to baseline Vm)** | **−66.0 mV = −13.9 mV, at 141 s** after 5-HT reaches the bath (just before the drug period starts) | **−74.0 mV = −11.3 mV, at 267 s** (inside the drug period) |
| Change at washout | **+8.5 mV** (above baseline) | −4.7 mV |
| Baseline drift | −0.8 mV/min | −0.2 mV/min |
| Recording book | "Was this 5HT application?" | "−10pA holding, longer time until effect, not as clear effect. Unsure if recording was stable" |

**Reading the example data:**
- **Both cells hyperpolarise in 5-HT.**
  - A_0004 responds fast: its lowest point is at 141 s. The 60–260 s window still holds most of its response (−11.8 mV against −13.9 mV at the lowest point).
  - B_0004 responds more slowly, with its lowest point at 267 s; this matches the book comment. The window covers most of its rise towards that point (−6.6 mV against −11.3 mV).
  - The window is the same for every cell. Change it only in `config.yaml`, for all cells at once.
- **A_0004 does not return to baseline.** It depolarises ~8 mV *above* baseline from ~900 s and fires a few spikes. This could be a rebound after 5-HT or a deteriorating recording. The figures show it, but no rule decides between the two.

## How to organise the pharmacology data

- **Unit of analysis:** the cell. Every application is its own row in 07; 08 uses **one application per cell and condition** (the first usable recording; within a recording, the first application of that condition) and never averages repeats.
- **Order effects:** a second application in the same cell can be affected by the first (incomplete washout, desensitisation). It is flagged (`after_previous_application`, `layered_on_previous_compound`) and its baseline is shown in the figure.
- **Which measure:**
  - The **drug period** change is the unbiased comparison: a fixed window, the same for every cell.
  - The **lowest point** and its time describe the size and speed of the response. They are selected per cell, so label them as such.
- **Recovery:** report `delta_washout_*` and look at the figure. A washout value beyond baseline in the opposite direction (as in A_0004) needs a decision about stability before the cell is used.
- **Drug identity:** write the compound in every drug tag (e.g. "SP hit bath", "Ba applied"), and keep the recording book's `Drug 1`, `Drug 2`, … filled in as a check. No drug or group comparison is run here.
- **Voltage clamp:** drug-induced currents (holding-current distributions, I–V subtraction with ramps) are a voltage-clamp analysis and are not part of this current-clamp step.

## Limitations

- **Periods are fixed:** windows are anchored to the tags (baseline and washout 100 s, drug 60–260 s after drug arrival). Bath exchange time is not modelled, so the tag is taken as drug arrival.
- **Lowest point:** a single smoothed extreme of noisy bins. It is biased away from zero: a cell with no response still has a lowest point. Check it in the figure.
- **Not validated** against manual measurements.
