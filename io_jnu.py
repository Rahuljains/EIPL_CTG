"""Loading the JNU-CTG antenatal cardiotocography dataset (waveforms only).

JNU-CTG ships 20,769 thirty-minute recordings at 4 Hz as CSV files with FHR and
UC columns, plus a 215-column clinical metadata registry and 108 pre-extracted
features. **None of the metadata may enter the model**: this loader reads the
raw waveform columns and nothing else, and takes the patient id straight from
the filename so `metadata.csv` is never opened.

Format confirmed in M2 against the dataset README:
  waveforms/csv_raw_7200/{patient_id}_{gravidity}_{timestamp}.csv
  7,201 rows (1 header + 7,200 samples), columns FHR,UC, 4 Hz, 0 = missing.

Author: Rahul Jain
Date:   17 Nov 2026
Usage:  from ctg.io_jnu import find_jnu_files, load_jnu_waveform
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from . import config as C

WAVEFORM_SUBDIR = "csv_raw_7200"
EXPECTED_SAMPLES = 7200          # 30 min at 4 Hz
WAVEFORM_COLUMNS = ["FHR", "UC"]


def find_jnu_files(jnu_dir: Path = C.JNU_DIR) -> List[Path]:
    """Return every JNU-CTG waveform csv, sorted by filename.

    The zip may add a nested folder, so the waveform directory is located by
    searching for the documented subdirectory name rather than being assumed.
    """
    if not jnu_dir.exists():
        raise FileNotFoundError(f"JNU-CTG folder not found: {jnu_dir}")

    candidates = [p for p in jnu_dir.rglob(WAVEFORM_SUBDIR) if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(
            f"No '{WAVEFORM_SUBDIR}' directory under {jnu_dir}. "
            "Check that JNU-CTG.zip extracted completely.")

    files = sorted(candidates[0].glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No csv waveforms in {candidates[0]}")
    return files


def patient_id_from_path(path: Path) -> str:
    """Extract the patient id from '{patient_id}_{gravidity}_{timestamp}.csv'.

    Taken from the filename so the clinical metadata file is never read.
    """
    return path.stem.split("_")[0]


def load_jnu_waveform(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read one recording. Return (fhr_4hz, uc_4hz) in bpm / a.u. with NaN for missing.

    The dataset encodes missing samples as 0 in both channels; FHR 0 is
    physiologically impossible and UC 0 is ambiguous, so both become NaN, which
    is the same convention the CTU-UHB loader uses.
    """
    frame = pd.read_csv(path, usecols=WAVEFORM_COLUMNS, dtype=np.float32,
                        encoding="utf-8-sig")
    fhr = frame["FHR"].to_numpy(dtype=np.float64)
    uc = frame["UC"].to_numpy(dtype=np.float64)

    fhr[fhr == 0] = np.nan
    uc[uc == 0] = np.nan
    return fhr, uc


def split_patients(patient_ids: np.ndarray, val_frac: float = C.JNU_VAL_FRAC,
                   seed: int = C.SEED) -> np.ndarray:
    """Return a boolean mask marking the validation patients (grouped, never split).

    Holding out whole pregnancies, not recordings, keeps repeat visits by the
    same patient out of both sides of the self-supervised validation split.
    """
    unique = np.unique(patient_ids)
    rng = np.random.default_rng(seed)
    n_val = max(1, int(round(len(unique) * val_frac)))
    val_patients = set(rng.choice(unique, size=n_val, replace=False).tolist())
    return np.array([pid in val_patients for pid in patient_ids], dtype=bool)
