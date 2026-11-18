"""Figures for the evaluation report (milestone M7b).

One palette across every figure, fixed by Results_Instructions Section 0.3, so a
colour means the same thing on every page: Layer 1 is always grey, the combined
system always red.

Titles state the finding, not the variable, and every figure carries n and the
event count.

Author: Rahul Jain
Date:   17 Nov 2026
Usage:  from ctg import plots_results as pr
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402

from . import config as C                                        # noqa: E402
from . import evaluate as ev                                     # noqa: E402
from .eda import wilson_ci                                       # noqa: E402

# Fixed palette (Results_Instructions 0.3)
COL_L1 = "#4D4D4D"          # Layer 1
COL_CLIN = "#1B9E77"        # clinical prior
COL_CTG = "#7570B3"         # CTG only
COL_SEQ = "#1F78B4"         # Layer 2 sequential
COL_SYS = "#D7301F"         # combined system
COL_ADVERSE = "#D7301F"
COL_NORMAL = "#1F78B4"
COL_ZON = "#2B8CBE"
CI_ALPHA = 0.20

MODEL_COLOURS = {"L1": COL_L1, "CLIN": COL_CLIN, "CTG-F": COL_CTG, "CTG-E": "#9E8FD0",
                 "SEQ-F": COL_SEQ, "SEQ-E": "#6BAED6", "SYS-F": COL_SYS,
                 "SYS-E": "#F16913", "ZON-F": COL_ZON, "ZON-E": "#74A9CF"}

plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                     "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 10.5, "axes.titlesize": 12,
                     "axes.labelsize": 10.5, "legend.fontsize": 9,
                     "xtick.labelsize": 9.5, "ytick.labelsize": 9.5})


def finish(fig, title: str, subtitle: str = "") -> None:
    """Apply the shared title and subtitle treatment."""
    if subtitle:
        fig.suptitle(title, fontsize=12.5, y=1.02)
        fig.text(0.5, 0.975, subtitle, ha="center", fontsize=9.6, color="#5b6472")
    else:
        fig.suptitle(title, fontsize=12.5)
    fig.tight_layout()


def model_colour(name: str) -> str:
    """Colour for a model name, falling back to grey."""
    return MODEL_COLOURS.get(name, "#8C96A8")


# -------------------------------------------------------------------- curves
def bootstrap_roc_band(scores: np.ndarray, labels: np.ndarray, n_boot: int = 400,
                       seed: int = C.SEED) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bootstrap a ROC confidence band on a common false-positive-rate grid."""
    grid = np.linspace(0, 1, 101)
    rng = np.random.default_rng(seed)
    curves = []
    for _ in range(n_boot):
        index = rng.integers(0, len(scores), len(scores))
        sample_labels = labels[index]
        if len(np.unique(sample_labels)) < 2:
            continue
        fpr, tpr, _ = ev.roc_curve(scores[index], sample_labels)
        curves.append(np.interp(grid, fpr, tpr))
    if not curves:
        return grid, np.full_like(grid, np.nan), np.full_like(grid, np.nan)
    stacked = np.vstack(curves)
    return grid, np.percentile(stacked, 2.5, axis=0), np.percentile(stacked, 97.5, axis=0)


def plot_roc(models: Dict[str, Tuple[np.ndarray, np.ndarray]], title: str,
             subtitle: str = "", band_for: Optional[str] = None,
             operating_points: Optional[List[Dict[str, object]]] = None,
             sens_targets: Sequence[float] = ()):
    """ROC overlay with an optional bootstrap band and marked operating points."""
    fig, axis = plt.subplots(figsize=(7.0, 6.2))
    axis.plot([0, 1], [0, 1], ls=":", lw=1.2, color="#999", label="chance")

    for name, (scores, labels) in models.items():
        scores = np.asarray(scores, dtype=float)
        labels = np.asarray(labels, dtype=float)
        keep = np.isfinite(scores) & np.isfinite(labels)
        if keep.sum() == 0:
            continue
        fpr, tpr, _ = ev.roc_curve(scores[keep], labels[keep])
        point, low, high = ev.bootstrap_ci(scores[keep], labels[keep], ev.auroc,
                                           n_boot=500)
        axis.plot(fpr, tpr, lw=2.0, color=model_colour(name),
                  label=f"{name}  AUROC {point:.3f} ({low:.3f}-{high:.3f})")

        if band_for == name:
            grid, lower, upper = bootstrap_roc_band(scores[keep], labels[keep])
            axis.fill_between(grid, lower, upper, color=model_colour(name),
                              alpha=CI_ALPHA, lw=0)
        for target in sens_targets:
            index = int(np.argmax(tpr >= target)) if (tpr >= target).any() else None
            if index:
                axis.plot(fpr[index], tpr[index], "o", ms=6, mfc="white",
                          mec=model_colour(name), mew=1.6)
                axis.annotate(f"sens {target:.2f}", (fpr[index], tpr[index]),
                              fontsize=8, xytext=(6, -10), textcoords="offset points",
                              color="#5b6472")

    for point in operating_points or []:
        axis.plot(1 - point["specificity"], point["sensitivity"],
                  point.get("marker", "*"), ms=point.get("size", 18),
                  color=point.get("colour", COL_SYS), mec="white", mew=1.2,
                  label=point["label"], zorder=5)
        if "sens_ci" in point and "spec_ci" in point:
            axis.errorbar(1 - point["specificity"], point["sensitivity"],
                          yerr=[[point["sensitivity"] - point["sens_ci"][0]],
                                [point["sens_ci"][1] - point["sensitivity"]]],
                          xerr=[[point["spec_ci"][1] and
                                 (1 - point["spec_ci"][1]) * -1 + (1 - point["specificity"])],
                                [(1 - point["spec_ci"][0]) - (1 - point["specificity"])]],
                          fmt="none", ecolor=point.get("colour", COL_SYS),
                          elinewidth=1.3, capsize=3, alpha=0.8)

    axis.set_xlabel("1 - specificity (false alarm rate)")
    axis.set_ylabel("Sensitivity (adverse labours caught)")
    axis.set_xlim(-0.02, 1.02)
    axis.set_ylim(-0.02, 1.02)
    axis.legend(loc="lower right", frameon=False, fontsize=8.5)
    finish(fig, title, subtitle)
    return fig


def plot_pr(models: Dict[str, Tuple[np.ndarray, np.ndarray]], prevalence: float,
            title: str, subtitle: str = "", sens_targets: Sequence[float] = ()):
    """Precision-recall overlay with the prevalence baseline."""
    fig, axis = plt.subplots(figsize=(7.0, 5.6))
    axis.axhline(prevalence, ls="--", lw=1.3, color="#999",
                 label=f"prevalence {prevalence:.3f}")

    for name, (scores, labels) in models.items():
        scores = np.asarray(scores, dtype=float)
        labels = np.asarray(labels, dtype=float)
        keep = np.isfinite(scores) & np.isfinite(labels)
        if keep.sum() == 0:
            continue
        recall, precision, _ = ev.pr_curve(scores[keep], labels[keep])
        point, low, high = ev.bootstrap_ci(scores[keep], labels[keep], ev.auprc,
                                           n_boot=500)
        axis.plot(recall, precision, lw=2.0, color=model_colour(name),
                  label=f"{name}  AUPRC {point:.3f} ({low:.3f}-{high:.3f})")
        for target in sens_targets:
            if (recall >= target).any():
                index = int(np.argmax(recall >= target))
                axis.plot(recall[index], precision[index], "o", ms=6, mfc="white",
                          mec=model_colour(name), mew=1.5)

    axis.set_xlabel("Recall (sensitivity)")
    axis.set_ylabel("Precision (PPV)")
    axis.set_xlim(-0.02, 1.02)
    axis.set_ylim(0, 1.02)
    axis.legend(loc="upper right", frameon=False, fontsize=8.5)
    finish(fig, title, subtitle)
    return fig


def plot_threshold_curve(scores: np.ndarray, labels: np.ndarray,
                         cv_threshold: float, title: str, subtitle: str = "",
                         target: float = C.TARGET_SENS):
    """Two-panel threshold curve: metrics above, the clinical trade-off below."""
    thresholds = np.arange(0.02, 0.99, 0.01)
    rows = []
    for threshold in thresholds:
        metrics = ev.alarm_metrics(scores >= threshold, labels)
        metrics["threshold"] = threshold
        metrics["pct_flagged"] = 100 * np.mean(scores >= threshold)
        metrics["missed_per_100_adverse"] = (
            100 * metrics["fn"] / max(metrics["n_adverse"], 1))
        rows.append(metrics)
    table = pd.DataFrame(rows)

    sens_low, sens_high, spec_low, spec_high = [], [], [], []
    for _, row in table.iterrows():
        low, high = wilson_ci(int(row["tp"]), int(row["tp"] + row["fn"]))
        sens_low.append(low)
        sens_high.append(high)
        low, high = wilson_ci(int(row["tn"]), int(row["tn"] + row["fp"]))
        spec_low.append(low)
        spec_high.append(high)

    fig, axes = plt.subplots(2, 1, figsize=(9.5, 8.0), sharex=True,
                             gridspec_kw={"height_ratios": [1.35, 1.0]})

    axis = axes[0]
    meets = table["sensitivity"] >= target
    if meets.any():
        axis.axvspan(table.loc[meets, "threshold"].min(),
                     table.loc[meets, "threshold"].max(), color="#D7301F", alpha=0.07,
                     lw=0, label=f"sensitivity >= {target:.2f}")
    axis.plot(table["threshold"], table["sensitivity"], lw=2.2, color=COL_ADVERSE,
              label="sensitivity")
    axis.fill_between(table["threshold"], sens_low, sens_high, color=COL_ADVERSE,
                      alpha=CI_ALPHA, lw=0)
    axis.plot(table["threshold"], table["specificity"], lw=2.2, color=COL_NORMAL,
              label="specificity")
    axis.fill_between(table["threshold"], spec_low, spec_high, color=COL_NORMAL,
                      alpha=CI_ALPHA, lw=0)
    axis.plot(table["threshold"], table["ppv"], lw=1.6, color="#7570B3", label="PPV")
    axis.plot(table["threshold"], table["npv"], lw=1.6, color="#1B9E77", label="NPV")
    axis.plot(table["threshold"], table["pct_flagged"] / 100, lw=1.4, ls="--",
              color="#8C96A8", label="% labours flagged")
    axis.axhline(target, ls=":", lw=1.2, color="#D7301F")
    if np.isfinite(cv_threshold):
        axis.axvline(cv_threshold, lw=2.0, color="#111",
                     label=f"CV-tuned threshold {cv_threshold:.3f}")
    axis.set_ylabel("Metric value")
    axis.set_ylim(0, 1.02)
    axis.legend(loc="center right", frameon=False, fontsize=8.5, ncol=2)

    axis = axes[1]
    axis.plot(table["threshold"], table["false_alarms_per_100_normal"], lw=2.2,
              color=COL_NORMAL, label="false alarms per 100 normal labours")
    axis.plot(table["threshold"], table["missed_per_100_adverse"], lw=2.2,
              color=COL_ADVERSE, label="missed adverse per 100 adverse")
    if np.isfinite(cv_threshold):
        axis.axvline(cv_threshold, lw=2.0, color="#111")
    axis.set_xlabel("Probability threshold")
    axis.set_ylabel("Per 100 labours")
    axis.legend(frameon=False, fontsize=9)
    finish(fig, title, subtitle)
    return fig, table


def plot_confusion(metrics: Dict[str, float], title: str, subtitle: str = ""):
    """2x2 confusion matrix with counts and row percentages."""
    counts = np.array([[metrics["tn"], metrics["fp"]],
                       [metrics["fn"], metrics["tp"]]], dtype=float)
    row_pct = 100 * counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)

    fig, axis = plt.subplots(figsize=(5.6, 4.8))
    axis.imshow(row_pct, cmap="Blues", vmin=0, vmax=100)
    for i in range(2):
        for j in range(2):
            axis.text(j, i, f"{int(counts[i, j])}\n{row_pct[i, j]:.1f}%", ha="center",
                      va="center", fontsize=14,
                      color="white" if row_pct[i, j] > 55 else "#1a1d23",
                      fontweight="bold" if (i, j) in [(1, 1), (1, 0)] else "normal")
    axis.set_xticks([0, 1])
    axis.set_xticklabels(["No alarm", "Alarm"])
    axis.set_yticks([0, 1])
    axis.set_yticklabels(["Normal", "Adverse"])
    axis.set_xlabel("Monitor output")
    axis.set_ylabel("Actual outcome")
    axis.grid(False)
    finish(fig, title, subtitle)
    return fig


def plot_calibration(scores: np.ndarray, labels: np.ndarray, title: str,
                     subtitle: str = ""):
    """Reliability diagram with Wilson CIs and a prediction histogram beneath."""
    table = ev.calibration_table(scores, labels)
    slope, intercept = ev.calibration_slope_intercept(scores, labels)
    brier = ev.brier(scores, labels)

    fig, axes = plt.subplots(2, 1, figsize=(6.6, 7.0), sharex=True,
                             gridspec_kw={"height_ratios": [3.0, 1.0]})
    axis = axes[0]
    axis.plot([0, 1], [0, 1], ls=":", lw=1.3, color="#999", label="perfect calibration")

    if not table.empty:
        lows, highs = [], []
        for _, row in table.iterrows():
            low, high = wilson_ci(int(row["n_adverse"]), int(row["n"]))
            lows.append(row["observed"] - low)
            highs.append(high - row["observed"])
        axis.errorbar(table["mean_predicted"], table["observed"],
                      yerr=np.vstack([np.abs(lows), np.abs(highs)]), fmt="o", ms=7,
                      color=COL_SEQ, ecolor=COL_SEQ, capsize=3, lw=1.4,
                      label="observed by decile (Wilson 95% CI)")
        axis.plot(table["mean_predicted"], table["observed"], lw=1.2, color=COL_SEQ,
                  alpha=0.55)
        for _, row in table.iterrows():
            axis.annotate(f"n={int(row['n'])}", (row["mean_predicted"], row["observed"]),
                          fontsize=7.5, xytext=(5, -11), textcoords="offset points",
                          color="#5b6472")

    axis.set_ylabel("Observed adverse rate")
    axis.set_xlim(0, max(1.0, float(np.nanmax(scores)) * 1.05))
    axis.set_ylim(0, 1.0)
    axis.legend(loc="upper left", frameon=False, fontsize=8.5,
                title=f"slope {slope:.2f}, intercept {intercept:.2f}, "
                      f"Brier {brier:.4f}", title_fontsize=8.5)

    axes[1].hist(scores[np.isfinite(scores)], bins=30, color="#8C96A8",
                 edgecolor="white")
    axes[1].set_xlabel("Predicted probability")
    axes[1].set_ylabel("Labours")
    finish(fig, title, subtitle)
    return fig, table, slope, intercept


def plot_dca(models: Dict[str, Tuple[np.ndarray, np.ndarray]], title: str,
             subtitle: str = "", band_for: Optional[str] = None,
             clinical_range: Tuple[float, float] = (0.05, 0.30)):
    """Decision curve analysis against treat-all and treat-none."""
    thresholds = np.arange(0.01, 0.61, 0.01)
    fig, axis = plt.subplots(figsize=(8.2, 5.6))
    axis.axvspan(*clinical_range, color="#D7301F", alpha=0.06, lw=0,
                 label=f"clinically relevant range ({clinical_range[0]:.2f}-"
                       f"{clinical_range[1]:.2f})")

    reference_done = False
    for name, (scores, labels) in models.items():
        scores = np.asarray(scores, dtype=float)
        labels = np.asarray(labels, dtype=float)
        keep = np.isfinite(scores) & np.isfinite(labels)
        if keep.sum() == 0:
            continue
        curve = ev.net_benefit(scores[keep], labels[keep], thresholds)
        axis.plot(curve["threshold"], curve["net_benefit_model"], lw=2.1,
                  color=model_colour(name), label=name)
        if not reference_done:
            axis.plot(curve["threshold"], curve["net_benefit_all"], lw=1.6, ls="--",
                      color="black", label="treat all (senior review for everyone)")
            axis.axhline(0, lw=1.4, color="black", label="treat none")
            reference_done = True
        if band_for == name:
            rng = np.random.default_rng(C.SEED)
            draws = []
            for _ in range(300):
                index = rng.integers(0, keep.sum(), keep.sum())
                sample = ev.net_benefit(scores[keep][index], labels[keep][index],
                                        thresholds)
                draws.append(sample["net_benefit_model"].to_numpy())
            stacked = np.vstack(draws)
            axis.fill_between(thresholds, np.percentile(stacked, 2.5, axis=0),
                              np.percentile(stacked, 97.5, axis=0),
                              color=model_colour(name), alpha=CI_ALPHA, lw=0)

    axis.set_xlabel("Threshold probability (risk at which you would act)")
    axis.set_ylabel("Net benefit")
    axis.set_xlim(0, 0.6)
    axis.legend(frameon=False, fontsize=8.5)
    finish(fig, title, subtitle)
    return fig


def plot_grouped_metrics(names: Sequence[str], metrics: Sequence[Dict[str, float]],
                         title: str, subtitle: str = "",
                         target: float = C.TARGET_SENS):
    """Grouped sensitivity/specificity bars with Wilson CIs and the target line."""
    fig, axis = plt.subplots(figsize=(max(7.0, 1.9 * len(names) + 3), 5.0))
    positions = np.arange(len(names))
    width = 0.36

    for offset, key, colour, label in [(-width / 2, "sensitivity", COL_ADVERSE,
                                        "sensitivity (adverse caught)"),
                                       (width / 2, "specificity", COL_NORMAL,
                                        "specificity (normal cleared)")]:
        values, lows, highs = [], [], []
        for metric in metrics:
            value = metric[key]
            if key == "sensitivity":
                low, high = wilson_ci(int(metric["tp"]), int(metric["tp"] + metric["fn"]))
            else:
                low, high = wilson_ci(int(metric["tn"]), int(metric["tn"] + metric["fp"]))
            values.append(value)
            lows.append(value - low)
            highs.append(high - value)
        axis.bar(positions + offset, values, width, color=colour, alpha=0.9,
                 edgecolor="white", label=label)
        axis.errorbar(positions + offset, values,
                      yerr=np.vstack([np.abs(lows), np.abs(highs)]), fmt="none",
                      ecolor="#333", capsize=4, lw=1.2)
        for x, (value, metric) in enumerate(zip(values, metrics)):
            count = (f"{int(metric['tp'])}/{int(metric['tp'] + metric['fn'])}"
                     if key == "sensitivity"
                     else f"{int(metric['tn'])}/{int(metric['tn'] + metric['fp'])}")
            axis.annotate(f"{value:.3f}\n{count}", (x + offset, value), ha="center",
                          va="bottom", fontsize=8.5, xytext=(0, 7),
                          textcoords="offset points")

    axis.axhline(target, ls="--", lw=1.5, color="#111",
                 label=f"sensitivity target {target:.2f}")
    axis.set_xticks(positions)
    axis.set_xticklabels(names)
    axis.set_ylabel("Metric value")
    axis.set_ylim(0, 1.18)
    axis.legend(frameon=False, fontsize=9, loc="upper right", ncol=1)
    finish(fig, title, subtitle)
    return fig


def plot_zone_rates(labels: Sequence[str], events: Sequence[int],
                    totals: Sequence[int], prevalence: float, title: str,
                    subtitle: str = "", xlabel: str = "Zone"):
    """% adverse per zone with Wilson CIs and n on the bars."""
    colours = {"GREEN": "#1B9E77", "RED": "#F16913", "RED - HIGH RISK": "#D7301F"}
    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    positions = np.arange(len(labels))

    rates, lows, highs = [], [], []
    for event, total in zip(events, totals):
        rate = 100 * event / total if total else np.nan
        low, high = wilson_ci(int(event), int(total))
        rates.append(rate)
        lows.append(rate - 100 * low)
        highs.append(100 * high - rate)

    axis.bar(positions, rates, color=[colours.get(l, "#8C96A8") for l in labels],
             alpha=0.9, edgecolor="white")
    axis.errorbar(positions, rates, yerr=np.vstack([np.abs(lows), np.abs(highs)]),
                  fmt="none", ecolor="#333", capsize=4, lw=1.2)
    axis.axhline(100 * prevalence, ls="--", lw=1.4, color="#111",
                 label=f"overall prevalence {100 * prevalence:.1f}%")
    for x, (rate, event, total) in enumerate(zip(rates, events, totals)):
        axis.annotate(f"{rate:.1f}%", (x, rate), ha="center", va="bottom",
                      fontsize=11, fontweight="bold", xytext=(0, 8),
                      textcoords="offset points")
        axis.annotate(f"{int(event)} of {int(total)}", (x, 0), ha="center",
                      va="bottom", fontsize=9, color="white", xytext=(0, 5),
                      textcoords="offset points")
    axis.set_xticks(positions)
    axis.set_xticklabels(labels)
    axis.set_xlabel(xlabel)
    axis.set_ylabel("% adverse")
    axis.set_ylim(0, max([r + h for r, h in zip(rates, highs)] + [10]) * 1.3)
    axis.legend(frameon=False)
    finish(fig, title, subtitle)
    return fig


def plot_venn(only_a: int, only_b: int, both: int, missed: int, name_a: str,
              name_b: str, title: str, subtitle: str = ""):
    """Two-circle Venn drawn in matplotlib, with counts and the missed total beside it."""
    fig, axis = plt.subplots(figsize=(7.4, 5.2))
    circle_a = plt.Circle((-0.42, 0), 0.78, color=COL_L1, alpha=0.42, lw=0)
    circle_b = plt.Circle((0.42, 0), 0.78, color=COL_SEQ, alpha=0.42, lw=0)
    axis.add_patch(circle_a)
    axis.add_patch(circle_b)

    axis.text(-0.82, 0, str(only_a), ha="center", va="center", fontsize=19,
              fontweight="bold")
    axis.text(0.82, 0, str(only_b), ha="center", va="center", fontsize=19,
              fontweight="bold")
    axis.text(0, 0, str(both), ha="center", va="center", fontsize=19,
              fontweight="bold")
    axis.text(-0.82, 0.92, name_a, ha="center", fontsize=11, color=COL_L1,
              fontweight="bold")
    axis.text(0.82, 0.92, name_b, ha="center", fontsize=11, color=COL_SEQ,
              fontweight="bold")
    axis.text(0, -1.12, f"Caught by neither: {missed}", ha="center", fontsize=11.5,
              color=COL_ADVERSE, fontweight="bold")

    axis.set_xlim(-1.6, 1.6)
    axis.set_ylim(-1.35, 1.25)
    axis.set_aspect("equal")
    axis.axis("off")
    finish(fig, title, subtitle)
    return fig


def plot_lead_time(lead_times: np.ndarray, title: str, subtitle: str = ""):
    """Histogram of warning lead time with the median and IQR marked."""
    values = np.asarray(lead_times, dtype=float)
    values = values[np.isfinite(values)]
    fig, axis = plt.subplots(figsize=(7.6, 4.6))
    if values.size:
        axis.hist(values, bins=min(25, max(6, values.size // 2)), color=COL_SYS,
                  alpha=0.82, edgecolor="white")
        median = np.median(values)
        q1, q3 = np.percentile(values, [25, 75])
        axis.axvline(median, lw=2.2, color="#111",
                     label=f"median {median:.0f} min (IQR {q1:.0f}-{q3:.0f})")
        axis.axvspan(q1, q3, color="#111", alpha=0.07, lw=0)
        axis.legend(frameon=False)
    axis.set_xlabel("Minutes from first HIGH RISK to the end of the recording")
    axis.set_ylabel("Adverse labours")
    finish(fig, title, subtitle)
    return fig


def plot_cumulative_detection(curves: Dict[str, np.ndarray], total_adverse: int,
                              title: str, subtitle: str = ""):
    """% of adverse labours already alerted, against minutes before the end."""
    fig, axis = plt.subplots(figsize=(8.4, 5.2))
    grid = np.arange(-90, 1, 1.0)
    for name, first_times in curves.items():
        values = np.asarray(first_times, dtype=float)
        values = values[np.isfinite(values)]
        detected = [100 * np.sum(-values <= point) / max(total_adverse, 1)
                    for point in grid]
        axis.plot(grid, detected, lw=2.2, color=model_colour(name), label=name)
    axis.set_xlabel("Minutes before the end of the recording")
    axis.set_ylabel("% of adverse labours already alerted")
    axis.set_xlim(-90, 0)
    axis.set_ylim(0, 100)
    axis.legend(frameon=False)
    finish(fig, title, subtitle)
    return fig


def plot_risk_over_time(readings: pd.DataFrame, threshold: float, title: str,
                        subtitle: str = "", bin_min: int = 5):
    """Mean probability against minutes before the end, by outcome, with CI bands."""
    frame = readings.copy()
    frame["bin"] = (-frame["minutes_before_end"] // bin_min * bin_min).astype(float)

    fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.4), sharex=True,
                             gridspec_kw={"height_ratios": [3.0, 1.0]})
    for label, colour, name in [(1, COL_ADVERSE, "Adverse"), (0, COL_NORMAL, "Normal")]:
        group = frame[frame["adverse"] == label]
        if group.empty:
            continue
        stats = group.groupby("bin")["p_post"].agg(["mean", "std", "count"])
        stats = stats[stats["count"] >= 5]
        error = 1.96 * stats["std"] / np.sqrt(stats["count"])
        axes[0].plot(stats.index, stats["mean"], lw=2.2, color=colour,
                     label=f"{name} (n={group['record_id'].nunique()} labours)")
        axes[0].fill_between(stats.index, stats["mean"] - error, stats["mean"] + error,
                             color=colour, alpha=CI_ALPHA, lw=0)
        axes[1].bar(stats.index, stats["count"], width=bin_min * 0.85, color=colour,
                    alpha=0.5)

    axes[0].axhline(threshold, ls="--", lw=1.5, color="#111",
                    label=f"CV-tuned threshold {threshold:.3f}")
    axes[0].set_ylabel("Mean predicted probability")
    axes[0].legend(frameon=False)
    axes[1].set_xlabel("Minutes before the end of the recording")
    axes[1].set_ylabel("Readings")
    finish(fig, title, subtitle)
    return fig


def plot_forest(labels: Sequence[str], values: Sequence[float],
                lows: Sequence[float], highs: Sequence[float], title: str,
                subtitle: str = "", xlabel: str = "", reference: Optional[float] = None,
                colours: Optional[Sequence[str]] = None, markers: Optional[Sequence[str]] = None):
    """Generic horizontal forest plot."""
    fig, axis = plt.subplots(figsize=(8.4, max(3.6, 0.36 * len(labels) + 1.6)))
    positions = np.arange(len(labels))
    colours = colours or [COL_SEQ] * len(labels)

    for position, (value, low, high, colour) in enumerate(
            zip(values, lows, highs, colours)):
        axis.plot([low, high], [position, position], lw=1.8, color=colour, alpha=0.8)
        marker = markers[position] if markers else "o"
        axis.plot(value, position, marker, ms=8, color=colour, mec="white", mew=1.0)

    if reference is not None:
        axis.axvline(reference, ls="--", lw=1.4, color="#111")
    axis.set_yticks(positions)
    axis.set_yticklabels(labels, fontsize=9.5)
    axis.invert_yaxis()
    axis.set_xlabel(xlabel)
    finish(fig, title, subtitle)
    return fig
