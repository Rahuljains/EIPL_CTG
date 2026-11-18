"""The CTG likelihood: a Gaussian Naive Bayes on PCA components of a window.

Identical for Track F (guideline-type features) and Track E (encoder embeddings)
- only the input matrix changes, so the two tracks are compared head to head on
exactly the same machinery.

Each labour contributes total weight 1 regardless of how many windows it has, so
a long recording cannot outvote a short one.

Everything is fitted inside the training folds only, and the fitted bundle
serialises to JSON so inference needs no sklearn.

Author: Rahul Jain
Date:   17 Nov 2026
Usage:  from ctg.ctg_nb import CtgNB
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from . import config as C


class CtgNB:
    """Median impute -> standardise -> PCA -> Gaussian NB, giving a per-window log LR."""

    def __init__(self, n_components: int = C.N_PCA,
                 loglr_clip: float = C.LOGLR_CLIP,
                 weight_per_labour: bool = C.WEIGHT_WINDOWS_PER_LABOUR) -> None:
        self.n_components = int(n_components)
        self.loglr_clip = float(loglr_clip)
        self.weight_per_labour = bool(weight_per_labour)

        self.medians: Optional[np.ndarray] = None
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None
        self.components_: Optional[np.ndarray] = None
        self.pca_mean_: Optional[np.ndarray] = None
        self.theta_: Optional[np.ndarray] = None       # (2, n_components) class means
        self.var_: Optional[np.ndarray] = None         # (2, n_components) class vars
        self.feature_names: Optional[Sequence[str]] = None
        self.n_fit_windows: int = 0
        self.explained_variance_ratio_: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ fit
    def fit(self, X: np.ndarray, y: Sequence[float], labour_ids: Sequence,
            valid_frac: Optional[np.ndarray] = None,
            feature_names: Optional[Sequence[str]] = None) -> "CtgNB":
        """Fit on the training windows only. Windows below the validity floor are dropped."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=float)
        labour_ids = np.asarray(labour_ids)

        keep = np.isfinite(y)
        if valid_frac is not None:
            keep &= np.asarray(valid_frac, dtype=float) >= C.MIN_VALID_FRAC_WINDOW
        X, y, labour_ids = X[keep], y[keep].astype(int), labour_ids[keep]
        if X.size == 0:
            raise ValueError("CtgNB.fit received no usable windows")

        self.feature_names = list(feature_names) if feature_names is not None else None
        self.n_fit_windows = int(len(X))

        # 1. median impute (training medians only)
        self.medians = np.nanmedian(X, axis=0)
        self.medians = np.where(np.isfinite(self.medians), self.medians, 0.0)
        X = self._impute(X)

        # 2. standardise
        self.mean_ = X.mean(axis=0)
        scale = X.std(axis=0, ddof=0)
        self.scale_ = np.where(scale > 1e-12, scale, 1.0)
        Z = (X - self.mean_) / self.scale_

        # 3. PCA by SVD, no whitening
        n_components = int(min(self.n_components, Z.shape[0], Z.shape[1]))
        self.pca_mean_ = Z.mean(axis=0)
        centred = Z - self.pca_mean_
        _, singular, right = np.linalg.svd(centred, full_matrices=False)
        self.components_ = right[:n_components]
        variances = singular ** 2 / max(len(centred) - 1, 1)
        total = variances.sum()
        self.explained_variance_ratio_ = (variances[:n_components] / total
                                          if total > 0 else None)
        scores = centred @ self.components_.T

        # 4. Gaussian NB with per-labour weights
        weights = self._labour_weights(labour_ids)
        self.theta_ = np.zeros((2, n_components))
        self.var_ = np.zeros((2, n_components))
        for label in (0, 1):
            mask = y == label
            if not mask.any():
                raise ValueError(f"CtgNB.fit: no training windows for class {label}")
            w = weights[mask]
            w = w / w.sum()
            class_scores = scores[mask]
            mean = (w[:, None] * class_scores).sum(axis=0)
            variance = (w[:, None] * (class_scores - mean) ** 2).sum(axis=0)
            self.theta_[label] = mean
            self.var_[label] = np.maximum(variance, 1e-9)
        return self

    def _labour_weights(self, labour_ids: np.ndarray) -> np.ndarray:
        """Weight each window by 1 / (number of windows of its labour)."""
        if not self.weight_per_labour:
            return np.ones(len(labour_ids))
        unique, counts = np.unique(labour_ids, return_counts=True)
        lookup = dict(zip(unique.tolist(), counts.tolist()))
        return np.array([1.0 / lookup[lid] for lid in labour_ids.tolist()])

    def _impute(self, X: np.ndarray) -> np.ndarray:
        """Fill non-finite entries with the training medians."""
        X = np.asarray(X, dtype=np.float64).copy()
        bad = ~np.isfinite(X)
        if bad.any():
            X[bad] = np.take(self.medians, np.nonzero(bad)[1])
        return X

    # -------------------------------------------------------------- predict
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Impute, standardise and project onto the fitted PCA components."""
        X = self._impute(X)
        Z = (X - self.mean_) / self.scale_
        return (Z - self.pca_mean_) @ self.components_.T

    def log_lr(self, X: np.ndarray) -> np.ndarray:
        """Per-window log likelihood ratio, clipped. Class priors are NOT included."""
        scores = self.transform(X)
        log_density = []
        for label in (0, 1):
            variance = self.var_[label]
            deviation = scores - self.theta_[label]
            log_density.append(-0.5 * np.sum(np.log(2 * np.pi * variance)
                                             + deviation ** 2 / variance, axis=1))
        ratio = log_density[1] - log_density[0]
        return np.clip(ratio, -self.loglr_clip, self.loglr_clip)

    # ----------------------------------------------------------------- i/o
    def to_dict(self) -> Dict[str, object]:
        """Serialise to plain python types (JSON-safe)."""
        return {"model": "CtgNB", "n_components": self.n_components,
                "loglr_clip": self.loglr_clip,
                "weight_per_labour": self.weight_per_labour,
                "medians": self.medians.tolist(), "mean": self.mean_.tolist(),
                "scale": self.scale_.tolist(),
                "pca_mean": self.pca_mean_.tolist(),
                "components": self.components_.tolist(),
                "theta": self.theta_.tolist(), "var": self.var_.tolist(),
                "feature_names": list(self.feature_names) if self.feature_names else None,
                "n_fit_windows": self.n_fit_windows,
                "explained_variance_ratio":
                    self.explained_variance_ratio_.tolist()
                    if self.explained_variance_ratio_ is not None else None}

    def to_json(self, path: Optional[Path] = None) -> str:
        """Serialise to JSON, optionally writing it to disk."""
        text = json.dumps(self.to_dict(), indent=2)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "CtgNB":
        """Rebuild a fitted model without sklearn."""
        model = cls(n_components=int(payload["n_components"]),
                    loglr_clip=float(payload["loglr_clip"]),
                    weight_per_labour=bool(payload["weight_per_labour"]))
        model.medians = np.asarray(payload["medians"], dtype=float)
        model.mean_ = np.asarray(payload["mean"], dtype=float)
        model.scale_ = np.asarray(payload["scale"], dtype=float)
        model.pca_mean_ = np.asarray(payload["pca_mean"], dtype=float)
        model.components_ = np.asarray(payload["components"], dtype=float)
        model.theta_ = np.asarray(payload["theta"], dtype=float)
        model.var_ = np.asarray(payload["var"], dtype=float)
        model.feature_names = payload.get("feature_names")
        model.n_fit_windows = int(payload.get("n_fit_windows", 0))
        ratio = payload.get("explained_variance_ratio")
        model.explained_variance_ratio_ = (np.asarray(ratio, dtype=float)
                                           if ratio is not None else None)
        return model

    @classmethod
    def from_json(cls, source) -> "CtgNB":
        """Rebuild from a JSON string or a path."""
        is_json_text = isinstance(source, str) and source.lstrip().startswith("{")
        if isinstance(source, Path) or (isinstance(source, str) and not is_json_text):
            source = Path(source).read_text(encoding="utf-8")
        return cls.from_dict(json.loads(source))

    # --------------------------------------------------------------- tables
    def component_table(self, top_loadings: int = 3) -> pd.DataFrame:
        """Per component: class means and variances, plus the top feature loadings."""
        rows = []
        for index in range(self.theta_.shape[1]):
            row = {"component": index + 1,
                   "mean_normal": round(float(self.theta_[0, index]), 4),
                   "mean_adverse": round(float(self.theta_[1, index]), 4),
                   "var_normal": round(float(self.var_[0, index]), 4),
                   "var_adverse": round(float(self.var_[1, index]), 4),
                   "separation": round(float(
                       abs(self.theta_[1, index] - self.theta_[0, index])
                       / np.sqrt(0.5 * (self.var_[0, index] + self.var_[1, index]))), 4)}
            if self.explained_variance_ratio_ is not None:
                row["explained_var_pct"] = round(
                    100 * float(self.explained_variance_ratio_[index]), 2)
            if self.feature_names is not None:
                loadings = self.components_[index]
                order = np.argsort(-np.abs(loadings))[:top_loadings]
                row["top_loadings"] = ", ".join(
                    f"{self.feature_names[i]} ({loadings[i]:+.2f})" for i in order)
            rows.append(row)
        return pd.DataFrame(rows)
