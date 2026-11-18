# CODE_SUMMARY: CTG Risk Monitor (final submission)

_Author: Rahul Jain. Last updated: 18 Nov 2026. Covers the modelling pipeline (Day 1)
and the demo app (Day 2, documented further in `app/README.md`)._

**Prototype on retrospective, non-Indian data. Not yet clinically validated.**

---

## 1. Description

A two-layer intrapartum CTG risk monitor, built and cross-validated on the PhysioNet
CTU-UHB database (552 labours), with a self-supervised encoder pre-trained on JNU-CTG
(20,769 antenatal recordings, signals only).

- **Layer 1** is a causal guideline rule engine (R1–R5, two notes and a signal-quality
  flag), evaluated every 5 seconds. It learns nothing and would run on any ward today.
  It is the baseline everything else is measured against.
- **Layer 2** is a hand-written Naive Bayes. It starts from the pregnancy history and
  updates at every reading with the CTG evidence of the last 15 minutes, shrunk by λ and
  faded by γ, so that old evidence decays and overlapping windows are not double-counted.
- The two feed one **zone**: GREEN, RED, or RED – HIGH RISK.
  - RED if any rule fires or the probability crosses θ.
  - HIGH RISK after K = 3 consecutive RED readings, or immediately on a deceleration
    lasting 9 minutes.
  - Readings every 15 minutes while GREEN, every 5 minutes once RED or in second stage.

**Outcome:** adverse = pH < 7.15 OR Apgar5 < 7, in 113 of 552 labours (20.5%).

Two CTG representations were compared head to head on identical machinery:
- **Track F:** 12 guideline-type window features.
- **Track E:** a 128-d embedding from the JNU-CTG encoder.

**Track F was selected** by the plan's rule (higher alarm sensitivity), although Track E
ranks better on AUROC. See Section 6.

---

## 2. Methods

### Data
- **CTU-UHB:** 552 labours, FHR + UC at 4 Hz, last ≤ 90 min before birth.
  - `dbID` is dropped in the loader and never written.
  - Outcome and post-labour fields are excluded from the predictors by construction
    (`FORBIDDEN_PREDICTORS` in `config.py`).
- **JNU-CTG:** 20,769 antenatal recordings → 77,714 pre-training windows from 20,470
  pregnancies.
  - Only the raw FHR/UC columns are read.
  - `metadata.csv` (215 clinical columns, 108 pre-extracted features) is never opened;
    the patient id comes from the filename.

### Preprocessing (causal)
- 0 and out-of-range values → NaN.
- Spike removal against a trailing 5-s median.
- 1 Hz series = mean of the valid 4 Hz samples (NaN if fewer than 2).
- Gaps ≤ 15 s linearly interpolated, with the validity mask kept separately.
- Layer 1 runs on the **un-interpolated** series.

### Clinical prior (Layer 2, part 1)
- `SkippingCategoricalNB`, written by hand. One log likelihood ratio per category,
  Laplace smoothing (α = 1).
- **Missing items are skipped**: they contribute nothing rather than being imputed.
- 11 items: age band, nulliparity, gestation band, diabetes, hypertension,
  pre-eclampsia, PROM, pyrexia, meconium, induction, non-cephalic presentation.
  Gravidity is excluded because it duplicates parity.

### CTG likelihood (Layer 2, part 2)
- Pipeline per 15-min window: median-impute → standardise → PCA (8 components) →
  Gaussian NB.
- Each labour is weighted once, however many windows it has. The per-window |log LR|
  is capped at 3.
- **Track F:** baseline, baseline shift, STV, per-minute amplitude, % low-variability
  minutes, accelerations, decelerations, deceleration area, maximum depth, longest
  deceleration, % time below baseline − 15, contractions per 10 min.
- **Track E:** a small 1-D CNN masked autoencoder. About 40% of the FHR is hidden in
  30–90 s spans while contractions stay visible, and the network learns to fill the gaps.
  It is trained on JNU-CTG only, so no CTU-UHB labour can leak into the representation.

### Validation
**Nested cross-validation split by labour**, not a fixed train/test split.
- **Outer:** 5-fold, stratified on the outcome.
  - About 441 labours train and 110 test per fold, with about 22–23 adverse cases in each
    test fold.
  - Every labour is tested exactly once, by a model that never saw it.
- **Inner:** 3-fold inside each outer training set. It produces out-of-fold scores for
  calibration and the operating point.
- Everything label-dependent is fitted inside training folds only: both Naive Bayes
  tables, the scaler, PCA, Platt calibration, and θ.
- The encoder's own hold-out is 5% of JNU-CTG **pregnancies**, never split mid-pregnancy.
- Fold splits are fixed by seed 42.

### Causality
Every output at time *t* uses only signal at or before *t*, with one documented exception:
interpolating a gap of ≤ 15 s uses the first valid sample after it. This is used for
Layer 2 windows, never for Layer 1. Unit tests replace all signal after minute 40 with
noise and assert that every Layer 1 state and window feature up to minute 40 is unchanged.

---

## 3. Paths (this submission folder)

```
codes/
├── README.md                    # entry point for reviewers
├── CODE_SUMMARY.md              # this file
├── requirements.txt             # full pipeline; app/requirements.txt for the app only
├── clinical_schema.json         # history form for the app (21 fields, 11 in the model)
├── src/ctg/                     # library
│   ├── config.py                # single source of truth for paths and parameters
│   ├── utils.py                 # run context, logger, ETA, decisions log
│   ├── io_ctu.py, io_jnu.py     # loaders (dbID dropped; JNU metadata never read)
│   ├── preprocess.py            # causal cleaning, 1 Hz conversion, windows
│   ├── labels.py                # outcome and the 11 Naive Bayes items
│   ├── detect.py                # baseline, decelerations, variability, contractions
│   ├── rules.py                 # Layer 1 R1–R5, N1–N2, Q
│   ├── features.py              # Track F window features
│   ├── encoder.py               # Track E masked autoencoder
│   ├── nb.py                    # SkippingCategoricalNB (clinical prior)
│   ├── ctg_nb.py                # CtgNB (CTG likelihood)
│   ├── sequential.py            # sequential update and zone simulator
│   ├── evaluate.py              # metrics, bootstrap CIs, thresholds, DCA
│   ├── eda.py, plots_results.py, html_report.py
│   └── engine.py                # CTGRiskEngine: the live API used by the app
├── scripts/                     # numbered pipeline 00–09 (+ report_parts.py)
│   └── x01–x03_*.py             # diagnostics behind the v1 → v3 decision (Section 6)
├── tests/                       # pytest, 45 tests
├── app/                         # Streamlit app, form helpers, 6 de-identified sample cases
├── output/models/
│   ├── engine_bundle_20261117_2221/   # FINAL model: clinical_nb.json, ctg_nb.json, params.json
│   ├── fold{0-4}_{clinical,F,E}_20261117_2151.json   # per-fold models of the sealed CV run
│   └── encoder_best_20261117_1612.pt  # Track E encoder weights
├── output/reports/              # EDA + results HTML, summaries, decisions log
└── docs/                        # concept note, plan, EDA/results specifications, methods notes
```

Not included: the datasets (download them; see Section 4), intermediate arrays, per-run
logs, superseded runs, and the per-labour demo timelines.

---

## 4. How to run

### Demo app (no dataset needed)
```bash
pip install -r app/requirements.txt
streamlit run app/ctg_monitor_app.py
```
The app picks up the newest `output/models/engine_bundle_*` (the shipped final bundle)
and offers the six sample cases in `app/sample_data/`.

### Full pipeline
1. `pip install -r requirements.txt`. The sealed build ran on Python 3.9.0, Windows 10,
   with an NVIDIA Quadro P4000.
2. Download the data:
   - `data/ctu-uhb/`: [CTU-UHB](https://physionet.org/content/ctu-uhb-ctgdb/1.0.0/)
     (ODC-By 1.0)
   - `data/jnu-ctg/`: [JNU-CTG](https://zenodo.org/records/21800730) (CC BY 4.0)
3. Run the scripts in this order:

```bat
python scripts\00_check_env.py             :: M0  environment + data check
python scripts\01_prepare_ctu.py           :: M1  load, clean, label, windows
python scripts\01b_eda_clinical.py         :: M1b clinical EDA -> HTML
python scripts\02_prepare_jnu.py           :: M2  pre-training windows (~11 min)
python scripts\04_pretrain_encoder.py      :: M5  encoder (~16 min on a P4000)
python scripts\03_layer1_rules.py          :: M3  rule engine on all 552 labours
python scripts\05_embed_windows.py         :: Track F features + Track E embeddings
python scripts\06_cross_validate.py        :: M7  nested CV (~22 min)
python scripts\09_results_report.py        :: M7b evaluation -> HTML
python scripts\07_fit_final_and_export.py  :: M9  final fit + engine bundle
python scripts\08_make_demo_samples.py     :: M9  six demo cases
pytest -q
```

- Scripts resolve their inputs by run name from `config.py` (`DATA_RUN`, `JNU_RUN`,
  `ENCODER_RUN`, `CLEAN_RUN`), or take the newest matching file and log which one.
- `RUN_NAME` is a timestamp and goes into every output filename, so reruns never
  overwrite earlier results.

---

## 5. Configuration

All defaults live in `src/ctg/config.py`. The ones that matter most:

| Parameter | Value | What it does |
|---|---|---|
| `OUTCOME_MODE`, `PH_CUTOFF`, `APGAR5_CUTOFF` | composite, 7.15, 7 | the outcome definition |
| `WINDOW_MIN`, `GRID_MIN` | 15, 5 | window length and reading grid (min) |
| `MIN_VALID_FRAC_WINDOW` | 0.50 | below this a window is "signal poor"; no Layer 2 update |
| `THRESHOLD_RULE`, `TARGET_ALARM_RATE` | target_alarm_rate, 0.30 | **the alarm budget sets the operating point**; accuracy is never a target |
| `EVIDENCE_AGGREGATOR` | decaying_max | worst recent 15-min window, faded so an alarm can clear |
| `MU_GRID`, `LAMBDA_GRID`, `GAMMA_GRID` | 0.25, 0.7, 0.85 | **design constants, not tuned** (see Section 6) |
| `TUNE_OBJECTIVE`, `CALIBRATION` | auroc, platt | rank first, then recalibrate inside the folds |
| `K_HIGH_RISK`, `N_CLEAR` | 3, 2 | consecutive REDs to escalate / clear readings to stand down |
| `N_PCA`, `LOGLR_CLIP` | 8, 3.0 | CTG likelihood capacity and per-window evidence cap |
| `DEPLOY_PREVALENCE` | None | set to a local adverse rate to shift the base log odds |
| `N_OUTER`, `N_INNER`, `N_BOOT` | 5, 3, 2000 | cross-validation and bootstrap |
| `SEED` | 42 | numpy, torch, sklearn and the fold splits |

---

## 6. Results (aggregate, pooled out-of-fold)

**Sealed build:** cross-validation run `20261117_2151`, final engine bundle
`engine_bundle_20261117_2221` (Track F, μ 0.25, λ 0.70, γ 0.85, θ 0.20, `decaying_max`).
- 552 labours, 113 adverse (20.5%).
- Every threshold was set inside the training folds and applied to the held-out fold.

| Model | AUROC | Sensitivity | Specificity | Adverse caught | Missed |
|---|---|---|---|---|---|
| L1 (rules alone, ever RED) | – | 0.381 | 0.831 | 43/113 | 70 |
| CLIN (history alone) | 0.588 | – | – | – | – |
| CTG-F (trace alone) | 0.634 | 0.416 | 0.811 | 47/113 | 66 |
| CTG-E (trace alone) | 0.676 | 0.354 | 0.825 | 40/113 | 73 |
| SEQ-E (Layer 2, best ranking) | **0.685** | 0.381 | 0.816 | 43/113 | 70 |
| **SYS-F (final system)** | 0.637 | **0.478** | **0.793** | **54/113** | **59** |

**Final system (SYS-F):**
- **Selection:** the plan's rule selects on sensitivity for ever HIGH RISK, and SYS-F
  leads SYS-E, 0.478 vs 0.425 (gap 0.053 > 0.03). SEQ-E ranks better (AUROC 0.685 vs
  0.638); the rule optimises the alarm, not the ranking. That tension is reported rather
  than smoothed over.
- **PPV and NPV:** PPV 54/145 = 37.2%; NPV 348/407 = 85.5%.
- **Operating point:** a 30% alarm budget. The system alarms on **26.3%** of labours
  (145/552).
- **Zones grade risk:** the adverse share by final zone is GREEN 12.9% (n = 342), RED
  24.0% (n = 75), HIGH RISK 37.8% (n = 135).

**Encoder:** validation masked-L1 of **4.955 bpm vs 5.739 bpm** for linear
interpolation (13.7% better). Embedding SD 0.222, so no collapse.

**Fold parameters:** identical in all five folds by construction (μ 0.25, λ 0.70,
γ 0.85, aggregator `decaying_max`). θ ranges 0.20–0.22.

### How the combination rule was settled (three builds)

| Build | Combination rule | SEQ-E AUROC | Operating point |
|---|---|---|---|
| v1 | faded sum, μ forced to 1, log-loss tuning | 0.659 | alarmed on 87% of labours |
| v2 | all tunable, 175-point grid | **0.588** | sane, but ranking collapsed |
| **v3 (sealed)** | decaying max, μ/λ/γ fixed | **0.685** | 30% alarm budget |

**v2 is the instructive failure.**
- Making μ, λ and γ tunable **raised** the inner-fold AUROC the tuner selected on
  (0.63–0.71), while the real out-of-fold score **fell** to 0.588.
- That gap is tuning overfitting. About 90 events per inner split cannot support 175
  choices, and the chosen values swung widely across folds.
- Diagnostic `x03` evaluated fixed configurations applied unchanged to every fold. The
  surface is flat: on Track E with `decaying_max`, AUROC runs 0.684–0.694 across every
  λ/γ at μ = 0.25.
- So the design changes were kept and the tuning freedom was removed: μ, λ and γ are now
  design constants. **v3 has no label-driven per-fold tuning.** The only per-fold choice
  is θ, which the alarm-budget rule sets from the score distribution alone.

**Declared optimism:** μ and the aggregator were chosen by inspecting pooled out-of-fold
results. That is worth at most about 0.01 AUROC of selection bias, against about 0.07 of
tuning overfitting avoided.

### Success criteria

| # | Criterion | Result | Met |
|---|---|---|---|
| 0 | Stays inside the 30% alarm budget | alarms on 26.3% of labours | **MET** |
| 1 | Sensitivity ≥ 0.90 for ever HIGH RISK | 0.478 | **NOT MET** |
| 2 | More sensitive than Layer 1 alone | 0.478 vs 0.381 | **MET** |
| 3 | % adverse rises GREEN → RED → HIGH RISK | 12.9% → 24.0% → 37.8% (n = 342 / 75 / 135) | **MET** |
| 4 | Layer 2 beats history and trace alone on AUROC | SEQ-E 0.685 vs CLIN 0.588, CTG-E 0.676 | **PARTIAL** |
| 5 | Specificity and false alarms beside every sensitivity | throughout | **MET** |
| 6 | n printed beside every percentage | throughout | **MET** |

**Criterion 1 is not met and should not be.**
- v1 met it only by alarming on 87% of labours.
- At AUROC ≈ 0.68, a monitor cannot catch 90% of adverse labours without alarming on most
  normal ones. The alarm budget makes that trade explicit.
- More data (full-labour recordings, Indian cohorts, richer history) is the route to a
  higher-sensitivity operating point.

**Criterion 4 is partial:** the clinical prior adds little, because CTU-UHB records only
11 thin history items.

---

## 7. Deliverables in this folder

- **`output/models/engine_bundle_20261117_2221/`** (the final model):
  `clinical_nb.json`, `ctg_nb.json`, `params.json`, `README.md`. It is Track F, so it
  loads without PyTorch.
- **`src/ctg/engine.py`:** `CTGRiskEngine` with `set_clinical`, `add_clinical`,
  `push_signal`, `state()`, `replay()`. It contains no plotting or UI code.
- **`app/`:** the Streamlit app.
  - Page 1 builds the starting risk from the history form.
  - Page 2 replays a CTG with the zone, the risk trajectory, the active rule and the
    contribution of each item.
  - Six de-identified sample cases (signal only) are included. All six were out-of-fold
    in the sealed run, including one adverse labour the system **missed**.
- **`clinical_schema.json`:** 21 history fields in three blocks. 11 are `in_model: true`
  and map to a Naive Bayes item. The rest are collected but cannot move the probability
  until local data exist.
- **`output/reports/`:**
  - `eda_clinical_20261117_1556.html` and `results_20261117_2151.html`: self-contained;
    they open offline.
  - `results_summary_20261117_2151.md`, `layer1_review_20261117_2149.md`,
    `data_summary_20261117_1542.md`.
  - `decisions_log.md`: every judgment call, with its reasoning.

---

## 8. Testing

45 tests in `tests/` cover:
- **R1 timing:** RED at 23:00, expedite at 29:00, baseline held at 140 ± 3 through a
  10-minute drop.
- **Other rules:** R2, R3 in both directions, the signal-poor state hold, deceleration
  geometry, and R5 second-stage gating.
- **Naive Bayes:** missing items skipped, a hand-computed toy example, and a JSON round
  trip without sklearn.
- **Sequential update:** λ = 0 reproduces the prior; two 5-minute readings weigh ⅔ of
  one 15-minute reading.
- **Causality:** future noise cannot change past outputs.
- **Leakage:** no labour in two folds, no CTU-UHB id in the encoder index, no forbidden
  predictor, no `dbID` in any written file.
- **Engine ≡ simulator:** checked on real labours, each with its own fold model.

**Run on this submission folder as shipped (no datasets):**
- `pytest -q` → **35 passed, 10 skipped**.
- The 10 skipped tests need the downloaded datasets, the cross-validation outputs or the
  demo timelines. They run once the pipeline in Section 4 has been executed.

---

## 9. Known gaps

- **The clinical prior contributes little** (μ fixed at 0.25; history alone scores AUROC
  0.588). With CTU-UHB's 11 history items that is the honest result; a richer history
  would likely change it.
- **Sensitivity analyses not run:** pH < 7.05 / 7.10 / 7.15 alone, Apgar-only, and
  `N_PCA` 4 and 12. The K = 2/3/4 sweep *was* run and is in the report.
- **Optional per-fold encoder fine-tuning** on CTU-UHB training folds (M5.6) was not run.
- **Results report coverage:** it has Parts 0–5 with 29 figures. Not built: the
  per-labour waterfall charts, the time-dependent AUROC and the ΔAUROC forest.
- **R3 never fires on CTU-UHB** (reduced variability for ≥ 30 consecutive minutes). This
  was verified not to be a bug: the detector fires on a synthetic flat trace, but only
  1.26% of minutes in this cohort are below 5 bpm, and the longest flat run is 2 minutes.
- **Layer 1 thresholds still need clinician confirmation,** in particular the upper
  baseline limit, the definition of late decelerations, and the R4 implementation.
- **Data scope:** non-Indian, retrospective, last 90 minutes of labour only. The start of
  each recording is treated as the start of monitoring.

---

## 10. Change history

| Date | Change |
|---|---|
| 17 Nov 2026 | Initial build, M0–M10. Nested CV, both tracks, engine bundle and six demo cases. |
| 17 Nov 2026 | **Causality fix:** window features had counted decelerations before they reached their minimum duration, and had measured depth and area over the *whole* episode, including the part after the window end. Per-minute variability could also be changed retroactively by an episode confirmed later. Both were genuine look-ahead. Fixed, and Layer 1, the feature matrix and the cross-validation were regenerated. Found by test 7. |
| 17 Nov 2026 | **Parity cleaning rule narrowed** from `Parity < Gravidity` to `Parity > Gravidity`. The written rule would have blanked 115 values (20.8% of the cohort). `Parity > Gravidity` occurs in 0 records while equality occurs in 115, which is a coding convention, not 115 data-entry errors. |
| 17 Nov 2026 | **Methodology v2 rejected, v3 sealed.** Tuning μ/λ/γ per fold over 175 combinations overfitted the tuner (out-of-fold AUROC 0.659 → 0.588). Kept the decaying-max aggregator, the clinical weight, the rank-then-calibrate split and the alarm budget; made μ/λ/γ design constants. SEQ-E 0.659 → 0.685. |
| 17 Nov 2026 | **Second causality fix:** contraction peaks were not causal. `scipy.find_peaks` measures prominence against the surrounding trace, including the future, so Track F's contraction feature changed when later signal arrived. Caught by a streaming-vs-batch engine test. Bounded the prominence window (`wlen`) and raised the confirmation delay to 60 s; the causality test no longer exempts that feature. |
| 17 Nov 2026 | Demo-sample folder cleared of stale cases. The engine now refuses a Track E bundle with no encoder instead of silently computing the wrong features. |
| 17 Nov 2026 | Added `UC_MAX_WIDTH_S = 300`: half-prominence contraction widths ran to 517 s, which is UC baseline drift, not one contraction. Affects note N2 only. |
| 18 Nov 2026 | **Portability fix** in `nb.py` and `ctg_nb.py`: `from_json` checked whether a JSON string was an existing file path. On Linux and macOS a long string raises `OSError: File name too long` instead of returning False. JSON text is now recognised first. Found when testing this package on Linux; no model output changes. |
| 18 Nov 2026 | Packaged for submission as `codes/`: final bundle `engine_bundle_20261117_2221`, the per-fold models of run `20261117_2151`, the final reports and the docs. Corrected stale run references in this summary. |

No participant-level data appears in this file.
