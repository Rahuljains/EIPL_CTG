"""Single source of truth for paths and parameters of the CTG Risk Monitor.

Every script imports from here. Nothing below a script CONFIG block may
hard-code a path or a threshold.

Author: Rahul Jain
Date:   17 Nov 2026
Usage:  from ctg import config as C
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

# ---------- paths ----------
ROOT = Path(__file__).resolve().parents[2]                   # D:\Rahul\B1
DATA_DIR = ROOT / "data"
CTU_DIR = DATA_DIR / "ctu-uhb"                               # searched recursively for *.hea
JNU_DIR = DATA_DIR / "jnu-ctg"
OUTPUT_DIR = ROOT / "output"
OUTPUT_SUBDIRS = ["figures", "sheets", "models", "logs", "reports", "arrays"]
SEED = 42

# Run names of upstream outputs. None = pick the latest matching file and log it.
DATA_RUN: Optional[str] = None        # run name of 01_prepare_ctu.py outputs
JNU_RUN: Optional[str] = None         # run name of 02_prepare_jnu.py outputs
ENCODER_RUN: Optional[str] = None     # run name of 04_pretrain_encoder.py outputs
CLEAN_RUN: Optional[str] = None       # run name of the EDA cleaned clinical csv

# ---------- signal ----------
FS_RAW = 4                   # Hz, both datasets
FS_WORK = 1                  # Hz, working resolution for windows/encoder/rules
FHR_VALID_MIN, FHR_VALID_MAX = 50, 220   # bpm; outside -> missing
SPIKE_BPM = 25               # sample > 25 bpm from trailing 5-s median -> missing
SPIKE_WIN_S = 5              # trailing median window for spike removal (s)
MAX_INTERP_S = 15            # gaps up to 15 s linearly interpolated (mask kept)
WINDOW_MIN = 15              # window length (min)
GRID_MIN = 5                 # window-end grid (min): 15, 20, 25, ...
MIN_VALID_FRAC_WINDOW = 0.50     # window used for Layer 2 only if >= 50% valid FHR
MIN_VALID_FRAC_PRETRAIN = 0.70
MIN_1HZ_VALID_SAMPLES = 2    # >= 2 valid 4 Hz samples needed to form a 1 Hz sample

# ---------- outcome ----------
OUTCOME_MODE = "composite"   # "composite" | "ph_only" | "apgar_only"
PH_CUTOFF = 7.15             # acidaemia if pH < cutoff
APGAR5_CUTOFF = 7            # low Apgar if Apgar5 < cutoff
SENSITIVITY_OUTCOMES = [("ph_only", 7.05), ("ph_only", 7.10), ("ph_only", 7.15),
                        ("apgar_only", None)]

# ---------- CTU-UHB header keys ----------
# Spellings as found in the .hea comment lines (verified on record 1001, M0).
HEADER_KEYS_OUTCOME = ["pH", "BDecf", "pCO2", "BE", "Apgar1", "Apgar5"]
HEADER_KEYS_NEONATAL = ["NICU days", "Seizures", "HIE", "Intubation",
                        "Main diag.", "Other diag."]       # flagged !NotReadyYet! upstream
HEADER_KEYS_FETUS = ["Gest. weeks", "Weight(g)", "Sex"]
HEADER_KEYS_MATERNAL = ["Age", "Gravidity", "Parity", "Diabetes", "Hypertension",
                        "Preeclampsia", "Liq. praecox", "Pyrexia", "Meconium"]
HEADER_KEYS_DELIVERY = ["Presentation", "Induced", "I.stage", "NoProgress", "CK/KP",
                        "II.stage", "Deliv. type"]
HEADER_KEYS_SIGNAL = ["Rec. type", "Pos. II.st.", "Sig2Birth"]
HEADER_KEY_DROP = ["dbID"]   # hospital identifier: dropped at load, never written

# Fields that must never be used as predictors (outcomes, post-labour, metadata).
FORBIDDEN_PREDICTORS = (HEADER_KEYS_OUTCOME + HEADER_KEYS_NEONATAL
                        + ["Weight(g)", "Sex", "Deliv. type", "I.stage", "II.stage",
                           "NoProgress", "CK/KP", "Sig2Birth", "Rec. type", "Pos. II.st."])

PRESENTATION_CEPHALIC = 1    # expected coding; confirmed against value_counts in M1
REC_TYPE_LABELS = {1: "ultrasound", 2: "scalp electrode"}

# ---------- clinical NB ----------
CLIN_ITEMS = {               # model name : header key
    "age_band": "Age", "nullipara": "Parity", "gest_band": "Gest. weeks",
    "diabetes": "Diabetes", "hypertension": "Hypertension", "preeclampsia": "Preeclampsia",
    "prom": "Liq. praecox", "pyrexia": "Pyrexia", "meconium": "Meconium",
    "induced": "Induced", "non_cephalic": "Presentation"}
AGE_BINS = [0, 20, 35, 100]      # <20, 20-34, >=35
GEST_BINS = [0, 39, 41, 50]      # 37-38, 39-40, >=41
NB_ALPHA = 1.0                   # Laplace smoothing
DEPLOY_PREVALENCE: Optional[float] = None   # None = training prevalence

# ---------- CTG NB ----------
N_PCA = 8                    # components fed to Gaussian NB (sensitivity: 4, 12)
LOGLR_CLIP = 3.0             # per-window |log LR| cap
WEIGHT_WINDOWS_PER_LABOUR = True

# ---------- Layer 1 (clinician to confirm) ----------
BASELINE_WIN_MIN = 10        # running baseline = robust median of last 10 min
BASELINE_EXCL_BPM = 15       # exclude samples > 15 bpm from preliminary median
DECEL_DROP_BPM = 15          # deceleration: >= 15 bpm below baseline ...
DECEL_MIN_S = 15             # ... for >= 15 s
ACCEL_RISE_BPM = 15
ACCEL_MIN_S = 15
PROLONGED_DECEL_MIN = 3      # R1 red
EXPEDITE_DECEL_MIN = 9       # R1 -> HIGH RISK immediately
DECEL_GAP_TOL_S = 10         # missing gaps <= 10 s do not reset the decel timer
LOW_BASELINE_BPM = 100       # R2
LOW_VAR_BPM = 5              # R3: per-minute amplitude < 5 bpm ...
LOW_VAR_MIN = 30             # ... for >= 30 consecutive valid minutes
PERSIST_DECEL_MIN = 30       # R4 look-back
RISING_BASELINE_BPM = 20     # R5 (red in 2nd stage) / note N1 (1st stage)
EXCESS_UC_PER_10MIN = 5      # note N2
LONG_UC_S = 120              # note N2
SIGNAL_POOR_FRAC = 0.50      # flag if < 50% valid FHR in last 60 s
SIGNAL_POOR_WIN_S = 60
RULE_STEP_S = 5              # Layer 1 evaluated every 5 s
MIN_VALID_FRAC_MINUTE = 0.50  # a minute with < 50% valid data is "unknown"
UC_PROMINENCE = 15           # contraction peak prominence (UC units)
UC_MIN_DIST_S = 60
UC_MIN_WIDTH_S = 30
UC_MAX_WIDTH_S = 300         # wider than this at half prominence = UC baseline drift,
                             # not one contraction; such peaks are discarded (M3 finding)
UC_SMOOTH_S = 30             # trailing mean applied to UC before peak finding
UC_UNRELIABLE_FRAC = 0.50    # > 50% missing UC -> contractions unreliable
ENABLE_TACHY_RULE = False
ENABLE_SINUSOIDAL_RULE = False

# ---------- encoder (Track E) ----------
FHR_NORM_CENTRE, FHR_NORM_SCALE = 140.0, 30.0
UC_NORM_SCALE, UC_NORM_CLIP = 100.0, 1.5
MASK_FRAC = 0.40             # fraction of FHR hidden during pre-training
MASK_SPAN_S = (30, 90)       # masked span length range (s)
EMB_DIM = 128
PRETRAIN_EPOCHS_GPU, PRETRAIN_EPOCHS_CPU = 30, 15
PRETRAIN_BATCH_GPU, PRETRAIN_BATCH_CPU = 256, 128
PRETRAIN_SUBSAMPLE_CPU = 30_000    # windows sampled per CPU epoch
PRETRAIN_LR, PRETRAIN_WD = 1e-3, 1e-4
PRETRAIN_PATIENCE = 4
MAX_TRAIN_MIN = 45           # hard time cap
JNU_VAL_FRAC = 0.05          # pregnancies held out for SSL validation

# ---------- sequential update & zones ----------
# How per-window CTG evidence becomes one running score. The x01 diagnostics showed
# the faded sum dilutes a single bad stretch across the whole labour: a decaying
# running maximum raised pooled AUROC 0.659 -> 0.694 and specificity at 90%
# sensitivity 0.235 -> 0.319 on Track E. "decaying_max" reduces to a plain running
# max when gamma = 1, so the tuner can still choose that.
EVIDENCE_AGGREGATOR = "decaying_max"     # "decaying_max" | "faded_sum" | "topk"
TOP_K_WINDOWS = 3                        # for the "topk" aggregator

# Grids. In the first build lambda, gamma and theta all sat on a grid edge, which
# means the grid was binding rather than the data. All three are widened, and the
# clinical prior gets a tunable weight instead of being forced to 1.
# These three are FIXED, not tuned. The v2 build tuned them per fold over a
# 175-point grid and pooled out-of-fold AUROC FELL from 0.659 to 0.588, while the
# inner-fold AUROC it selected on rose to 0.63-0.71 - the gap is tuning
# overfitting, which ~90 inner events cannot support. Diagnostic x03 evaluated
# fixed configurations with no tuning at all and found the surface is flat:
# on Track E with decaying_max, AUROC runs 0.684-0.694 across every lambda/gamma
# at mu = 0.25. There is nothing to gain by tuning them and real variance to lose,
# so they are design constants. Chosen on pooled out-of-fold data, which is a mild
# optimism of at most ~0.01 AUROC, against ~0.07 of tuning overfitting avoided.
MU_GRID = [0.25]                         # weight on the clinical prior (fixed)
LAMBDA_GRID = [0.7]                      # CTG evidence shrinkage (fixed)
GAMMA_GRID = [0.85]                      # fading per 15 min (fixed)
THETA_GRID = [round(0.02 + 0.02 * i, 3) for i in range(49)]   # 0.02 ... 0.98

# Tuning objective. Log-loss punishes confident errors, which is what drove lambda
# to its floor in the first build. Rank first, then restore calibration with a
# Platt scaler fitted inside the training folds, so decision curve analysis still
# has honest probabilities without the ranking being compromised to get them.
TUNE_OBJECTIVE = "auroc"                 # "auroc" | "log_loss"
CALIBRATION = "platt"                    # "platt" | "none"

# Threshold rule. "target_alarm_rate" picks the most sensitive monitor that stays
# inside an alarm budget, which avoids the degenerate corner the sensitivity target
# produced (452 of 552 labours ending in HIGH RISK).
THRESHOLD_RULE = "target_alarm_rate"     # "target_alarm_rate" | "target_sens" | "youden"
TARGET_ALARM_RATE = 0.30                 # at most this share of labours may alarm
ALARM_RATE_TARGETS_REPORT = [0.20, 0.30, 0.40, 0.50]
TARGET_SENS = 0.90
SENS_TARGETS_REPORT = [0.80, 0.85, 0.90, 0.95]
TUNE_LOSS = "log_loss"                   # retained for the comparison runs
K_HIGH_RISK = 3              # consecutive red 5-min readings -> HIGH RISK
N_CLEAR = 2                  # consecutive non-red readings -> GREEN
EXPEDITE_TRIGGERS_HIGH_RISK = True
K_SWEEP = [2, 3, 4]          # K values swept for the results report

# ---------- CV ----------
N_OUTER = 5
N_INNER = 3
N_REPEATS = 1
N_BOOT = 2000
N_BOOT_FAST = 1000           # for per-variable AUROC CIs in the EDA

# ---------- reporting ----------
FIG_DPI_FILE = 200           # figures written to output/figures
FIG_DPI_HTML = 150           # figures embedded in the HTML reports
NOT_FOR_CLINICAL_USE = "Prototype, not for clinical use"
