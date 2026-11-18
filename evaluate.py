"""Metrics, bootstrap confidence intervals and threshold selection.

Primary target: high sensitivity. Every function that picks a threshold picks it
for sensitivity first, and every alarm metric is returned with specificity and
the false-alarm burden beside it so the cost is never hidden.

Accuracy is deliberately absent: with a 3.9 : 1 class ratio it rewards a monitor
that never alarms.

Author: Rahul Jain
Date:   17 Nov 2026
Usage:  from ctg import evaluate
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import config as C


# ------------------------------------------------------------------ ranking
def auroc(scores: Sequence[float], labels: Sequence[float]) -> float:
    """Area under the ROC curve, via the rank-sum identity."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    keep = np.isfinite(scores) & np.isfinite(labels)
    scores, labels = scores[keep], labels[keep]
    positives, negatives = scores[labels == 1], scores[labels == 0]
    if positives.size == 0 or negatives.size == 0:
        return float("nan")
    order = np.argsort(np.concatenate([positives, negatives]))
    ranks = np.empty(order.size, dtype=float)
    ranks[order] = np.arange(1, order.size + 1)
    # average ranks for ties
    combined = np.concatenate([positives, negatives])
    for value in np.unique(combined):
        tied = combined == value
        if tied.sum() > 1:
            ranks[tied] = ranks[tied].mean()
    rank_sum = ranks[:positives.size].sum()
    return float((rank_sum - positives.size * (positives.size + 1) / 2)
                 / (positives.size * negatives.size))


def auprc(scores: Sequence[float], labels: Sequence[float]) -> float:
    """Average precision (area under the precision-recall curve)."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    keep = np.isfinite(scores) & np.isfinite(labels)
    scores, labels = scores[keep], labels[keep]
    if labels.sum() == 0:
        return float("nan")
    order = np.argsort(-scores)
    labels = labels[order]
    true_positives = np.cumsum(labels)
    precision = true_positives / np.arange(1, len(labels) + 1)
    return float((precision * labels).sum() / labels.sum())


def brier(scores: Sequence[float], labels: Sequence[float]) -> float:
    """Mean squared error of the predicted probabilities."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    keep = np.isfinite(scores) & np.isfinite(labels)
    if keep.sum() == 0:
        return float("nan")
    return float(np.mean((scores[keep] - labels[keep]) ** 2))


def log_loss(scores: Sequence[float], labels: Sequence[float],
             epsilon: float = 1e-9) -> float:
    """Binary cross-entropy; the loss (lambda, gamma) are tuned on."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    keep = np.isfinite(scores) & np.isfinite(labels)
    scores, labels = np.clip(scores[keep], epsilon, 1 - epsilon), labels[keep]
    if scores.size == 0:
        return float("nan")
    return float(-np.mean(labels * np.log(scores)
                          + (1 - labels) * np.log(1 - scores)))


def roc_curve(scores: Sequence[float], labels: Sequence[float]
              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (false positive rate, true positive rate, threshold)."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    keep = np.isfinite(scores) & np.isfinite(labels)
    scores, labels = scores[keep], labels[keep]
    order = np.argsort(-scores)
    scores, labels = scores[order], labels[order]

    true_positives = np.cumsum(labels)
    false_positives = np.cumsum(1 - labels)
    n_positive, n_negative = labels.sum(), (1 - labels).sum()
    tpr = np.concatenate([[0.0], true_positives / max(n_positive, 1)])
    fpr = np.concatenate([[0.0], false_positives / max(n_negative, 1)])
    thresholds = np.concatenate([[np.inf], scores])
    return fpr, tpr, thresholds


def pr_curve(scores: Sequence[float], labels: Sequence[float]
             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (recall, precision, threshold)."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    keep = np.isfinite(scores) & np.isfinite(labels)
    scores, labels = scores[keep], labels[keep]
    order = np.argsort(-scores)
    scores, labels = scores[order], labels[order]
    true_positives = np.cumsum(labels)
    precision = true_positives / np.arange(1, len(labels) + 1)
    recall = true_positives / max(labels.sum(), 1)
    return recall, precision, scores


# ------------------------------------------------------------- alarm metrics
def alarm_metrics(predicted: Sequence[bool], labels: Sequence[float]
                  ) -> Dict[str, float]:
    """2x2 table plus sensitivity, specificity, PPV, NPV, F1 and the misses."""
    predicted = np.asarray(predicted, dtype=bool)
    labels = np.asarray(labels, dtype=float)
    keep = np.isfinite(labels)
    predicted, labels = predicted[keep], labels[keep].astype(int)

    tp = int(np.sum(predicted & (labels == 1)))
    fp = int(np.sum(predicted & (labels == 0)))
    fn = int(np.sum(~predicted & (labels == 1)))
    tn = int(np.sum(~predicted & (labels == 0)))

    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    npv = tn / (tn + fn) if (tn + fn) else float("nan")
    f1 = (2 * ppv * sensitivity / (ppv + sensitivity)
          if np.isfinite(ppv) and np.isfinite(sensitivity) and (ppv + sensitivity)
          else float("nan"))

    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "n": tp + fp + fn + tn, "n_adverse": tp + fn,
            "sensitivity": sensitivity, "specificity": specificity,
            "ppv": ppv, "npv": npv, "f1": f1,
            "adverse_missed": fn,
            "false_alarms_per_100_normal": 100 * fp / (fp + tn) if (fp + tn) else np.nan}


def bootstrap_ci(values: Sequence[float], labels: Sequence[float],
                 statistic, n_boot: int = C.N_BOOT, seed: int = C.SEED,
                 alpha: float = 0.05) -> Tuple[float, float, float]:
    """Percentile bootstrap over labours. Returns (point, low, high)."""
    values = np.asarray(values)
    labels = np.asarray(labels, dtype=float)
    point = statistic(values, labels)

    rng = np.random.default_rng(seed)
    draws: List[float] = []
    n = len(values)
    for _ in range(n_boot):
        index = rng.integers(0, n, n)
        sample_labels = labels[index]
        if len(np.unique(sample_labels[np.isfinite(sample_labels)])) < 2:
            continue
        try:
            draws.append(statistic(values[index], sample_labels))
        except (ValueError, ZeroDivisionError):
            continue
    draws = [d for d in draws if np.isfinite(d)]
    if not draws:
        return point, float("nan"), float("nan")
    return (point, float(np.percentile(draws, 100 * alpha / 2)),
            float(np.percentile(draws, 100 * (1 - alpha / 2))))


# ------------------------------------------------------- threshold selection
def threshold_for_sensitivity(scores: Sequence[float], labels: Sequence[float],
                              target: float = C.TARGET_SENS) -> float:
    """Largest threshold whose sensitivity still reaches the target.

    The largest such threshold is the one that raises the fewest false alarms
    while still meeting the sensitivity the plan asks for. If no threshold
    reaches the target, the smallest observed score is returned so the alarm is
    as sensitive as the model allows.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    keep = np.isfinite(scores) & np.isfinite(labels)
    scores, labels = scores[keep], labels[keep]
    if scores.size == 0:
        return float("nan")

    candidates = np.unique(scores)
    best = None
    for threshold in candidates:
        sensitivity = np.mean(scores[labels == 1] >= threshold)
        if sensitivity >= target:
            best = threshold if best is None else max(best, threshold)
    return float(best) if best is not None else float(candidates.min())


def threshold_for_alarm_rate(scores: Sequence[float],
                             max_rate: float = C.TARGET_ALARM_RATE) -> float:
    """Smallest threshold whose alarm rate still fits inside the alarm budget.

    This is the deployable framing: a ward can absorb only so many alerts, so fix
    the budget and be as sensitive as possible within it. Unlike a sensitivity
    target it cannot collapse into "alarm on everything", because the budget binds
    from the other side. Note it uses the score distribution only, never the
    labels, so it carries no outcome information.
    """
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        return float("nan")
    candidates = np.unique(scores)
    for threshold in candidates:                     # ascending = most sensitive first
        if np.mean(scores >= threshold) <= max_rate:
            return float(threshold)
    return float(candidates.max())


def platt_scaling(scores: Sequence[float], labels: Sequence[float]
                  ) -> Tuple[float, float]:
    """Fit p = sigmoid(a * logit + b) by maximum likelihood. Returns (a, b).

    Used to restore calibration after the sequential parameters have been tuned
    for ranking. Fitted inside the training folds only.
    """
    x = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if x.size < 10 or len(np.unique(y)) < 2:
        return 1.0, 0.0

    design = np.column_stack([x, np.ones_like(x)])
    beta = np.array([1.0, 0.0])
    for _ in range(100):
        eta = design @ beta
        probability = 1.0 / (1.0 + np.exp(-np.clip(eta, -500, 500)))
        weights = np.maximum(probability * (1 - probability), 1e-9)
        gradient = design.T @ (y - probability)
        hessian = design.T @ (design * weights[:, None])
        try:
            step = np.linalg.solve(hessian + 1e-9 * np.eye(2), gradient)
        except np.linalg.LinAlgError:
            break
        beta += step
        if np.max(np.abs(step)) < 1e-9:
            break
    if not np.all(np.isfinite(beta)):
        return 1.0, 0.0
    return float(beta[0]), float(beta[1])


def youden_threshold(scores: Sequence[float], labels: Sequence[float]) -> float:
    """Threshold maximising sensitivity + specificity - 1 (comparison only)."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    keep = np.isfinite(scores) & np.isfinite(labels)
    scores, labels = scores[keep], labels[keep]
    candidates = np.unique(scores)
    best_value, best_threshold = -np.inf, float("nan")
    for threshold in candidates:
        predicted = scores >= threshold
        metrics = alarm_metrics(predicted, labels)
        value = metrics["sensitivity"] + metrics["specificity"] - 1
        if np.isfinite(value) and value > best_value:
            best_value, best_threshold = value, float(threshold)
    return best_threshold


def calibration_table(scores: Sequence[float], labels: Sequence[float],
                      n_bins: int = 10) -> pd.DataFrame:
    """Predicted vs observed risk by decile of predicted probability."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    keep = np.isfinite(scores) & np.isfinite(labels)
    scores, labels = scores[keep], labels[keep]
    if scores.size == 0:
        return pd.DataFrame()

    quantiles = np.quantile(scores, np.linspace(0, 1, n_bins + 1))
    quantiles[0] -= 1e-9
    bins = np.unique(quantiles)
    assignment = np.clip(np.digitize(scores, bins[1:-1], right=True), 0,
                         len(bins) - 2)

    rows = []
    for index in range(len(bins) - 1):
        mask = assignment == index
        if not mask.any():
            continue
        rows.append({"bin": index + 1, "n": int(mask.sum()),
                     "mean_predicted": float(scores[mask].mean()),
                     "observed": float(labels[mask].mean()),
                     "n_adverse": int(labels[mask].sum())})
    return pd.DataFrame(rows)


def calibration_slope_intercept(scores: Sequence[float], labels: Sequence[float]
                                ) -> Tuple[float, float]:
    """Logistic recalibration slope and intercept (1 and 0 mean well calibrated)."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    keep = np.isfinite(scores) & np.isfinite(labels)
    scores, labels = np.clip(scores[keep], 1e-6, 1 - 1e-6), labels[keep]
    if scores.size < 10 or len(np.unique(labels)) < 2:
        return float("nan"), float("nan")

    x = np.log(scores / (1 - scores))
    beta = np.zeros(2)
    design = np.column_stack([np.ones_like(x), x])
    for _ in range(50):                      # Newton-Raphson
        eta = design @ beta
        probability = 1 / (1 + np.exp(-eta))
        weights = np.maximum(probability * (1 - probability), 1e-9)
        gradient = design.T @ (labels - probability)
        hessian = design.T @ (design * weights[:, None])
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            break
        beta += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return float(beta[1]), float(beta[0])


def net_benefit(scores: Sequence[float], labels: Sequence[float],
                thresholds: Sequence[float]) -> pd.DataFrame:
    """Decision curve analysis: net benefit of the model, treat-all and treat-none."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    keep = np.isfinite(scores) & np.isfinite(labels)
    scores, labels = scores[keep], labels[keep]
    n = len(labels)
    prevalence = labels.mean() if n else np.nan

    rows = []
    for threshold in thresholds:
        if threshold <= 0 or threshold >= 1:
            continue
        predicted = scores >= threshold
        tp = int(np.sum(predicted & (labels == 1)))
        fp = int(np.sum(predicted & (labels == 0)))
        odds = threshold / (1 - threshold)
        rows.append({
            "threshold": threshold,
            "net_benefit_model": tp / n - (fp / n) * odds,
            "net_benefit_all": prevalence - (1 - prevalence) * odds,
            "net_benefit_none": 0.0})
    return pd.DataFrame(rows)
