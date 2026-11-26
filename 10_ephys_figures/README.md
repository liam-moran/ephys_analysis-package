# 10_ephys_figures: publication-style figures and provisional drug statistics

This step makes the figures that describe what PMvDAT+ neurons are like and how much they vary. There are no groups: every cell is shown, a mean is only drawn together with the individual cells, and every panel gives n. The only statistics are the **provisional paired drug tests** (below). They are clearly marked and must be reviewed before anything is reported.

**Run step by step:** open `figures.ipynb`.
**Run all at once:** `python 10_ephys_figures/run.py` (add `--overwrite`).
Run 01–08 first. 10 reads only their outputs, plus the ABFs of the example traces.

## Figures (`output/figures/`, PNG + SVG)

SVG text stays editable (Illustrator / Inkscape). Colours: blue = silent, red = tonic at rest.

| Figure | Panels | Source |
|---|---|---|
| `fig1_examples` | spontaneous activity of a silent and a tonic cell (last 30 s); step family (−150…+100 pA); ramp with its command | ABFs of the recordings 08 selected; cells in `config.yaml` → `examples` (auto: first silent / first tonic cell) |
| `fig2_labels` | stacked count bars: silent / tonic, depolarisation block, rebound (counts, e.g. 6/10); **dot matrix**: which labels occur together in which cell | 08 cell summary |
| `fig3_properties` | one dot per cell with median (line) and IQR (bar), in real units: Vm, Cm, Rin (steps and ramp), sag, spontaneous rate (log axis), ISI CV, rheobase, F–I slope, max rate, rebound spikes, hysteresis | 08 (`properties` in config) |
| `fig4_ap` | **AP at ramp rheobase**: average waveform aligned at threshold and phase plane (each cell thin, mean bold); threshold, amplitude, half-width, upstroke, downstroke, AHP | 06 `ap_waveforms.csv` + 08 `ramp_ap_*` |
| `fig5_excitability` | F–I per cell (open markers = steps with attenuated events, i.e. block); block onset (cells without block shown above the largest step, "> 100"); rebound spikes against sag ratio (descriptive) | 05 `step_sweeps.csv`, 08 |
| `fig6_drug_<condition>` | per drug condition: ΔVm from baseline against time from the tag (each cell thin, mean bold where all cells have data, drug period shaded); **estimation plot** (baseline → drug → washout per cell; below, the paired differences with their bootstrap distribution and BCa 95 % CI; the effect size with its CI is the panel title); peak vs sham-peak excursion | 07, 08 selection |
| `fig7_estimation_<condition>` | (if `estimation_secondary: true`) one estimation plot per secondary outcome: firing rate, ISI CV, AP threshold, half-width, AHP (exploratory) | 07, 08 selection |

**Tables (`output/`):**
- `summary_table.csv`: n, mean ± SD, median [Q1, Q3] and range per property.
- `drug_stats.csv` and `drug_responders.csv`: **PROVISIONAL**; every row says so.

All membrane potentials are as recorded. The liquid junction potential (+15.9 mV, K-gluconate, 35 °C) is not corrected; the true Vm is ≈ 15.9 mV more negative. Every figure with absolute voltages carries this note.

## Which AP

The AP waveform and all AP parameters come from the **first valid spike on the rising ramp**. Its mean is taken over the **quiet ramp sweeps**, the same sweeps as the rheobase (06). Every cell is therefore measured at its own threshold current, under the same condition.
- **Why not the steps:** the step AP (05, first spike at the first step with spikes) comes at very different overdrive in different cells, because the steps are 50 pA apart.
- **Ramp caveat:** a slow ramp inactivates some sodium channels, so ramp spikes can have a slightly higher threshold and slower upstroke than step spikes. Compare values only with other ramp-evoked APs.

## How to read block and rebound

- **Depolarisation block:** how much drive the cell can take before its spikes fail. Sodium channels stay inactivated and spikes shrink into small events. A cell that blocks at +50 pA has a narrow output range: it can signal "on", but cannot encode stronger input with higher rates. In `fig5` this shows as an F–I curve that flattens, with open markers.
- **Rebound:** whether release from inhibition makes the cell fire. This usually comes from currents opened or de-inactivated by hyperpolarisation, e.g. I_h (which also causes sag) or T-type Ca²⁺. A cell with rebound can turn a pause in its inhibitory input into output. Which current is responsible is not tested here.
- **The dot matrix** (`fig2`) shows which properties occur together in the same cell, which separate pie charts cannot.

## Estimation statistics: effect sizes first

The drug results are reported as **estimation statistics** (Ho et al. 2019, *Nat Methods* 16:565; the DABEST approach). The headline is **how large the effect is and how precisely it is known**: an effect size with its 95 % confidence interval. Whether p < 0.05 comes second.

**How to read the estimation plot** (`fig6`, `fig7`):
- **Top:** every cell's value in each period, connected per cell. All data are visible, not just a mean.
- **Bottom:** the paired difference from the reference (drug − baseline, washout − baseline):
  - **dot:** mean difference;
  - **bar:** 95 % CI;
  - **grey half-violin:** the bootstrap distribution, i.e. how much the mean difference would vary if the experiment were repeated on cells like these.
- **The panel title gives the number to report**, e.g. "Δ mean = −8.05 mV [95 % CI −9.27, −7.10]".

**How it is computed** (`drug_stats.estimate`, scipy only; DABEST itself is not used because it does not run on this environment's Python 3.13 / pandas 3 / matplotlib 3.11):

| Quantity | Definition |
|---|---|
| mean / median difference | paired, drug − reference, over the cells with both values |
| 95 % CI | **BCa** bootstrap (bias-corrected and accelerated; Efron 1987), resampling cells as pairs, 10 000 resamples, fixed seed (`ci_method`, `n_boot`, `seed`). BCa corrects for skew in small samples; when it cannot be computed (e.g. identical differences) the percentile CI is used and noted |
| Hedges' g_av | standardised effect: mean difference / average SD of the two conditions, small-sample corrected (Lakens 2013). Use it to compare effects measured in different units; report the raw difference (mV) as the main result |
| dz | mean difference / SD of the differences (paired standardised effect, used for power) |

**Reporting suggestion:** "5-HT hyperpolarised PMvDAT+ neurons by 8.1 mV (mean paired difference; 95 % CI −9.3 to −7.1 mV, BCa bootstrap; n = 12 cells from N animals)". The Wilcoxon p-value can follow in brackets.

**References:**
- Ho et al. 2019, Moving beyond P values: data analysis with estimation graphics, *Nat Methods*.
- Calin-Jageman & Cumming 2019, Estimation for better inference in neuroscience, *eNeuro*.
- Efron 1987, Better bootstrap confidence intervals, *JASA*.
- Lakens 2013, Calculating and reporting effect sizes, *Front Psychol*.

## Drug statistics: what and why (PROVISIONAL)

> **PROVISIONAL: pre-specified defaults, to be reviewed before reporting.** The rules are in `config.yaml` → `drug_stats`. Set them before looking at the results, and review them with the final dataset (and, ideally, a statistician) before anything is reported.

**Design: paired, within-cell.** Each cell is its own control (baseline → drug → washout). That removes the large differences between cells from the comparison. One application per cell and condition is used (the 08 selection); repeats are never averaged.

| Question | Test (default) | Why |
|---|---|---|
| **Primary:** how much does the drug change Vm? | drug − baseline (layered application: drug − pre-tag): **mean and median difference, Hedges' g, each with a BCa bootstrap 95 % CI** (the headline); secondary: Wilcoxon signed-rank and paired t-test | One outcome decided in advance. The effect size with its CI says how large the effect is; Wilcoxon needs no normality assumption, which cannot be checked at n ≈ 10. |
| Does it recover? | **Friedman** over baseline / drug / washout, then pairwise Wilcoxon (drug − baseline, washout − baseline), **Holm-corrected** | Repeated measures in the same cells. Incomplete recovery is common and is reported, not hidden. |
| Secondary outcomes | firing rate, ISI CV, AP threshold, half-width, AHP: Wilcoxon, **Holm-corrected across the set** | Several tests raise the chance of a false positive, so they are corrected or read as exploratory. |
| Does firing switch on or off? | **exact McNemar** on silent ↔ tonic changes | Many silent cells have a rate of 0; a yes/no change is clearer than a skewed rate. |
| How many cells respond? | per cell: \|ΔVm\| > 3 × its baseline Vm SD **and** baseline drift < 2 mV/min; count with **Clopper–Pearson 95 % CI** | A rule fixed in advance, from each cell's own noise. |
| Is the peak response real? | the peak is **never tested against 0**: it is the extreme of a noisy trace and never 0. It is compared with a **sham peak**: the same procedure on the same length of baseline before the tag (paired Wilcoxon on the sizes) | Removes the bias of selecting an extreme. |
| Two drugs in the same cells | optional `condition_pairs` (e.g. 5-HT vs 5-HT + Ba²⁺): paired Wilcoxon of the responses | Within-cell comparison, as in the published 5-HT / Ba²⁺ experiments. Include a re-application to rule out desensitisation. |

**Other points:**
- **Too few cells:** below `min_cells` (5) no p-value is computed; the row says "too few cells". The example data (2 cells) are such a case.
- **Exclusions:** applications flagged `unstable_baseline` are left out and listed (`excluded_cells`). Decide further exclusion criteria (e.g. an Ra change > 20 %) before looking at the responses.
- **The animal:** cells from the same animal are not fully independent. `n_animals` and `cells_per_animal` are reported. With ≥ ~5 animals, a mixed model (response ~ condition + (1 | animal)) is the robust option; it is not run here.
- **Power:** a paired design with n = 10 detects an effect of dz ≈ 1 with 80 % power. A 5-HT hyperpolarisation of ~10 mV with modest spread is easily detectable. Small secondary effects (e.g. 1 mV in AP threshold) are not, so read those as exploratory.

**To review before reporting:**
1. the primary outcome and the drug window;
2. the responder rule;
3. the exclusion criteria;
4. the set of secondary outcomes;
5. whether a mixed model is needed (number of animals);
6. any condition pairs.

## Example data

With 2 cells every figure is drawn, but the drug tests return "too few cells". The layout at the expected n was checked on a synthetic 12-cell dataset (`tests/test_figures.py`).
