"""Outcome labels and the clinical items the Naive Bayes prior sees.

The outcome is a surrogate: cord acidaemia (pH) and/or a low 5-minute Apgar.
Nothing in this module may be used as a predictor; the derived clinical items
are built only from fields allowed by the plan (Section 5.3).

Author: Rahul Jain
Date:   17 Nov 2026
Usage:  from ctg.labels import derive_labels, derive_clinical_items
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from . import config as C

AGE_BAND_LABELS = ["<20", "20-34", ">=35"]
GEST_BAND_LABELS = ["<39", "39-40", ">=41"]


def derive_labels(frame: pd.DataFrame, mode: str = C.OUTCOME_MODE,
                  ph_cutoff: float = C.PH_CUTOFF,
                  apgar5_cutoff: float = C.APGAR5_CUTOFF) -> pd.DataFrame:
    """Add acid, low_apgar and adverse columns. Rows with no component stay NaN.

    mode is 'composite' (either component), 'ph_only' or 'apgar_only'.
    """
    out = frame.copy()
    ph = pd.to_numeric(out.get("pH"), errors="coerce")
    apgar5 = pd.to_numeric(out.get("Apgar5"), errors="coerce")

    out["acid"] = np.where(ph.notna(), (ph < ph_cutoff).astype(float), np.nan)
    out["low_apgar"] = np.where(apgar5.notna(),
                                (apgar5 < apgar5_cutoff).astype(float), np.nan)

    if mode == "ph_only":
        adverse = out["acid"]
    elif mode == "apgar_only":
        adverse = out["low_apgar"]
    elif mode == "composite":
        # Either component positive is enough; NaN only when both are missing.
        either = out[["acid", "low_apgar"]].max(axis=1, skipna=True)
        both_missing = out[["acid", "low_apgar"]].isna().all(axis=1)
        adverse = either.where(~both_missing, np.nan)
    else:
        raise ValueError(f"Unknown OUTCOME_MODE: {mode!r}")

    out["adverse"] = adverse
    return out


def outcome_label(mode: str = C.OUTCOME_MODE, ph_cutoff: float = C.PH_CUTOFF,
                  apgar5_cutoff: float = C.APGAR5_CUTOFF) -> str:
    """Return a human-readable description of the outcome definition."""
    if mode == "ph_only":
        return f"pH < {ph_cutoff}"
    if mode == "apgar_only":
        return f"Apgar5 < {apgar5_cutoff:g}"
    return f"pH < {ph_cutoff} OR Apgar5 < {apgar5_cutoff:g}"


def _band(values: pd.Series, bins: list, labels: list) -> pd.Series:
    """Cut a numeric series into labelled bands, keeping NaN as NaN."""
    return pd.cut(pd.to_numeric(values, errors="coerce"), bins=bins,
                  labels=labels, right=False, include_lowest=True).astype(object)


def _binary(values: pd.Series) -> pd.Series:
    """Map a 0/1 header field to float 0/1; anything else becomes NaN."""
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.where(numeric.isin([0, 1]), np.nan)


def derive_clinical_items(frame: pd.DataFrame,
                          presentation_cephalic: int = C.PRESENTATION_CEPHALIC
                          ) -> pd.DataFrame:
    """Add the Naive Bayes clinical items derived from the raw header fields.

    Items: age_band, gest_band, nullipara, non_cephalic, the binary risk flags,
    and any_risk_factor. Gravidity is deliberately not used (redundant with
    parity; see plan M4).
    """
    out = frame.copy()
    out["age_band"] = _band(out["Age"], C.AGE_BINS, AGE_BAND_LABELS)
    out["gest_band"] = _band(out["Gest. weeks"], C.GEST_BINS, GEST_BAND_LABELS)

    parity = pd.to_numeric(out["Parity"], errors="coerce")
    out["nullipara"] = np.where(parity.notna(), (parity == 0).astype(float), np.nan)

    presentation = pd.to_numeric(out["Presentation"], errors="coerce")
    out["non_cephalic"] = np.where(presentation.notna(),
                                   (presentation != presentation_cephalic).astype(float),
                                   np.nan)

    for item, key in [("diabetes", "Diabetes"), ("hypertension", "Hypertension"),
                      ("preeclampsia", "Preeclampsia"), ("prom", "Liq. praecox"),
                      ("pyrexia", "Pyrexia"), ("meconium", "Meconium"),
                      ("induced", "Induced")]:
        out[item] = _binary(out[key])

    gest = pd.to_numeric(out["Gest. weeks"], errors="coerce")
    out["post_term"] = np.where(gest.notna(), (gest >= 42).astype(float), np.nan)

    risk_items = ["diabetes", "hypertension", "preeclampsia", "prom", "pyrexia",
                  "meconium", "induced", "non_cephalic", "post_term"]
    out["n_risk_factors"] = out[risk_items].sum(axis=1, min_count=1)
    out["any_risk_factor"] = np.where(out["n_risk_factors"].notna(),
                                      (out["n_risk_factors"] > 0).astype(float), np.nan)
    return out


def nb_item_frame(frame: pd.DataFrame,
                  items: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """Return only the columns the clinical Naive Bayes consumes, as categoricals."""
    items = items or C.CLIN_ITEMS
    return frame[list(items.keys())].copy()
