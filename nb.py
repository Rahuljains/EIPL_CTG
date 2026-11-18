"""The clinical prior: a hand-written categorical Naive Bayes that skips missing items.

Written by hand rather than taken from sklearn for three reasons the plan calls
for: a missing item must contribute exactly nothing (not an imputed category),
each item's contribution must be inspectable so the app can show why a
probability moved, and the fitted bundle must load without sklearn.

Author: Rahul Jain
Date:   17 Nov 2026
Usage:  from ctg.nb import SkippingCategoricalNB
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from . import config as C


def logit(probability: float) -> float:
    """Log odds of a probability, clipped away from 0 and 1."""
    probability = min(max(float(probability), 1e-9), 1 - 1e-9)
    return math.log(probability / (1 - probability))


def sigmoid(value: float) -> float:
    """Inverse of logit, overflow-safe."""
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def _is_missing(value: object) -> bool:
    """True when a clinical item has no recorded value."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return isinstance(value, str) and value.strip() == ""


class SkippingCategoricalNB:
    """Categorical Naive Bayes over clinical items, with missing items skipped.

    Attributes after fit:
        log_lr[item][category]  log P(cat | adverse) - log P(cat | normal)
        base_log_odds           prior log odds of the adverse outcome
    """

    def __init__(self, alpha: float = C.NB_ALPHA,
                 deploy_prevalence: Optional[float] = C.DEPLOY_PREVALENCE) -> None:
        self.alpha = float(alpha)
        self.deploy_prevalence = deploy_prevalence
        self.log_lr: Dict[str, Dict[str, float]] = {}
        self.categories: Dict[str, List[str]] = {}
        self.counts: Dict[str, Dict[str, Dict[str, int]]] = {}
        self.base_log_odds: float = 0.0
        self.train_prevalence: float = float("nan")
        self.items: List[str] = []
        self.n_train: int = 0

    # ------------------------------------------------------------------ fit
    def fit(self, frame: pd.DataFrame, y: Sequence[float],
            items: Optional[Sequence[str]] = None) -> "SkippingCategoricalNB":
        """Fit the per-category likelihood ratios and the base log odds."""
        y = pd.Series(y).reset_index(drop=True)
        frame = frame.reset_index(drop=True)
        known = y.notna()
        frame, y = frame.loc[known], y.loc[known].astype(int)

        self.items = list(items) if items is not None else list(frame.columns)
        self.n_train = int(len(y))
        self.train_prevalence = float(y.mean())
        prevalence = (self.train_prevalence if self.deploy_prevalence is None
                      else float(self.deploy_prevalence))
        self.base_log_odds = logit(prevalence)

        adverse_mask = (y == 1).to_numpy()
        for item in self.items:
            values = frame[item].map(lambda v: None if _is_missing(v) else str(v))
            observed = values.notna()
            categories = sorted(values[observed].unique())
            self.categories[item] = categories
            if not categories:
                self.log_lr[item] = {}
                continue

            n_categories = len(categories)
            table: Dict[str, float] = {}
            raw: Dict[str, Dict[str, int]] = {}
            adverse_total = int((observed.to_numpy() & adverse_mask).sum())
            normal_total = int((observed.to_numpy() & ~adverse_mask).sum())

            for category in categories:
                in_category = (values == category).to_numpy()
                n_adverse = int((in_category & adverse_mask).sum())
                n_normal = int((in_category & ~adverse_mask).sum())

                p_adverse = ((n_adverse + self.alpha)
                             / (adverse_total + self.alpha * n_categories))
                p_normal = ((n_normal + self.alpha)
                            / (normal_total + self.alpha * n_categories))
                table[category] = math.log(p_adverse) - math.log(p_normal)
                raw[category] = {"n": n_adverse + n_normal, "n_adverse": n_adverse,
                                 "n_normal": n_normal}
            self.log_lr[item] = table
            self.counts[item] = raw
        return self

    # -------------------------------------------------------------- predict
    def item_contributions(self, row: Mapping[str, object]) -> Dict[str, float]:
        """Return {item: log likelihood ratio}; missing or unseen items are skipped."""
        contributions: Dict[str, float] = {}
        for item in self.items:
            value = row.get(item) if hasattr(row, "get") else None
            if _is_missing(value):
                continue
            table = self.log_lr.get(item, {})
            key = str(value)
            if key in table:
                contributions[item] = table[key]
        return contributions

    def prior_logit(self, row: Mapping[str, object]) -> float:
        """Base log odds plus every observed item's contribution."""
        return self.base_log_odds + sum(self.item_contributions(row).values())

    def prior_prob(self, row: Mapping[str, object]) -> float:
        """Probability of the adverse outcome given the clinical items only."""
        return sigmoid(self.prior_logit(row))

    def predict_logit(self, frame: pd.DataFrame) -> np.ndarray:
        """Vectorised prior_logit over a dataframe of items."""
        return np.array([self.prior_logit(row) for row in
                         frame.to_dict(orient="records")], dtype=float)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        """Vectorised prior_prob over a dataframe of items."""
        return np.array([sigmoid(value) for value in self.predict_logit(frame)])

    # ----------------------------------------------------------------- i/o
    def to_dict(self) -> Dict[str, object]:
        """Serialise the fitted model to plain python types."""
        return {"model": "SkippingCategoricalNB", "alpha": self.alpha,
                "items": self.items, "categories": self.categories,
                "log_lr": self.log_lr, "counts": self.counts,
                "base_log_odds": self.base_log_odds,
                "train_prevalence": self.train_prevalence,
                "deploy_prevalence": self.deploy_prevalence,
                "n_train": self.n_train}

    def to_json(self, path: Optional[Path] = None) -> str:
        """Serialise to JSON, optionally writing it to disk."""
        text = json.dumps(self.to_dict(), indent=2)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "SkippingCategoricalNB":
        """Rebuild a fitted model from its dictionary form. No sklearn needed."""
        model = cls(alpha=float(payload["alpha"]),
                    deploy_prevalence=payload.get("deploy_prevalence"))
        model.items = list(payload["items"])
        model.categories = {k: list(v) for k, v in payload["categories"].items()}
        model.log_lr = {k: {c: float(x) for c, x in v.items()}
                        for k, v in payload["log_lr"].items()}
        model.counts = payload.get("counts", {})
        model.base_log_odds = float(payload["base_log_odds"])
        model.train_prevalence = float(payload["train_prevalence"])
        model.n_train = int(payload.get("n_train", 0))
        return model

    @classmethod
    def from_json(cls, source) -> "SkippingCategoricalNB":
        """Rebuild from a JSON string or a path."""
        is_json_text = isinstance(source, str) and source.lstrip().startswith("{")
        if isinstance(source, Path) or (isinstance(source, str) and not is_json_text):
            source = Path(source).read_text(encoding="utf-8")
        return cls.from_dict(json.loads(source))

    # --------------------------------------------------------------- tables
    def table(self) -> pd.DataFrame:
        """Return the fitted log LR table, one row per item and category."""
        rows: List[Dict[str, object]] = []
        for item in self.items:
            for category, value in self.log_lr.get(item, {}).items():
                counts = self.counts.get(item, {}).get(category, {})
                n = counts.get("n", 0)
                n_adverse = counts.get("n_adverse", 0)
                rows.append({"item": item, "category": category, "n": n,
                             "n_adverse": n_adverse,
                             "pct_adverse": round(100 * n_adverse / n, 1) if n else np.nan,
                             "log_lr": round(value, 4),
                             "lr": round(math.exp(value), 3)})
        return pd.DataFrame(rows)
