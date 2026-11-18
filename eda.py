"""Statistics and figures for the clinical EDA (milestone M1b).

Everything here is descriptive. Associations with the outcome are shown for
understanding only and must never be used to add, drop or re-bin a predictor
(the predictor list is fixed in config.py).

Author: Rahul Jain
Date:   17 Nov 2026
Usage:  from ctg import eda
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402

from . import config as C                                        # noqa: E402

# Shared palette: one colour per outcome group, used in every figure.
COLOUR_NORMAL = "#4C78A8"
COLOUR_ADVERSE = "#B8232F"
COLOUR_ACCENT = "#2F5D9E"
COLOUR_NEUTRAL = "#8C96A8"
COLOUR_LIMIT = "#C0392B"
GROUP_COLOURS = {0: COLOUR_NORMAL, 1: COLOUR_ADVERSE}
GROUP_NAMES = {0: "Normal", 1: "Adverse"}

plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                     "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 10.5, "axes.titlesize": 11.5,
                     "axes.labelsize": 10.5, "legend.fontsize": 9.5,
                     "xtick.labelsize": 9.5, "ytick.labelsize": 9.5})


# --------------------------------------------------------------------- stats
def wilson_ci(successes: int, total: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Wilson score interval for a proportion; (nan, nan) when total is 0."""
    if total <= 0:
        return (float("nan"), float("nan"))
    z = stats.norm.ppf(1 - alpha / 2)
    phat = successes / total
    denom = 1 + z ** 2 / total
    centre = (phat + z ** 2 / (2 * total)) / denom
    halfwidth = z * np.sqrt(phat * (1 - phat) / total
                            + z ** 2 / (4 * total ** 2)) / denom
    return (max(0.0, centre - halfwidth), min(1.0, centre + halfwidth))


def describe_continuous(series: pd.Series) -> Dict[str, float]:
    """Return n, missing, mean, SD, quartiles, range, skewness, kurtosis, mode."""
    values = pd.to_numeric(series, errors="coerce")
    clean = values.dropna()
    total = len(values)
    out = {"n": int(clean.size), "n_missing": int(total - clean.size),
           "pct_missing": 100.0 * (total - clean.size) / total if total else np.nan,
           "n_unique": int(clean.nunique())}
    if clean.empty:
        return {**out, **{k: np.nan for k in
                          ["mean", "sd", "median", "q1", "q3", "iqr", "min", "max",
                           "skewness", "excess_kurtosis", "mode"]}}
    q1, q3 = float(clean.quantile(0.25)), float(clean.quantile(0.75))
    modes = clean.mode()
    out.update({"mean": float(clean.mean()), "sd": float(clean.std(ddof=1)),
                "median": float(clean.median()), "q1": q1, "q3": q3, "iqr": q3 - q1,
                "min": float(clean.min()), "max": float(clean.max()),
                "skewness": float(stats.skew(clean)),
                "excess_kurtosis": float(stats.kurtosis(clean)),
                "mode": float(modes.iloc[0]) if not modes.empty else np.nan})
    return out


def outliers_tukey(series: pd.Series, factor: float = 1.5) -> np.ndarray:
    """Boolean mask of values outside [Q1 - factor*IQR, Q3 + factor*IQR]."""
    values = pd.to_numeric(series, errors="coerce")
    q1, q3 = values.quantile(0.25), values.quantile(0.75)
    iqr = q3 - q1
    if not np.isfinite(iqr) or iqr == 0:
        return np.zeros(len(values), dtype=bool)
    return ((values < q1 - factor * iqr) | (values > q3 + factor * iqr)).fillna(
        False).to_numpy()


def outliers_mad(series: pd.Series, cutoff: float = 3.5) -> np.ndarray:
    """Boolean mask of values with a robust z-score (MAD based) above cutoff."""
    values = pd.to_numeric(series, errors="coerce")
    median = values.median()
    mad = (values - median).abs().median()
    if not np.isfinite(mad) or mad == 0:
        return np.zeros(len(values), dtype=bool)
    robust_z = (values - median).abs() / (1.4826 * mad)
    return (robust_z > cutoff).fillna(False).to_numpy()


def outliers_implausible(series: pd.Series,
                         limits: Tuple[float, float]) -> np.ndarray:
    """Boolean mask of values outside the clinically plausible range."""
    values = pd.to_numeric(series, errors="coerce")
    low, high = limits
    return ((values < low) | (values > high)).fillna(False).to_numpy()


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Single-variable AUROC via the Mann-Whitney U statistic; NaNs dropped."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    keep = np.isfinite(scores) & np.isfinite(labels)
    scores, labels = scores[keep], labels[keep]
    positives, negatives = scores[labels == 1], scores[labels == 0]
    if positives.size == 0 or negatives.size == 0:
        return float("nan")
    ranks = stats.rankdata(np.concatenate([positives, negatives]))
    rank_sum = ranks[:positives.size].sum()
    u_stat = rank_sum - positives.size * (positives.size + 1) / 2
    return float(u_stat / (positives.size * negatives.size))


def auroc_ci(scores: np.ndarray, labels: np.ndarray, n_boot: int = C.N_BOOT_FAST,
             seed: int = C.SEED) -> Tuple[float, float, float]:
    """AUROC with a percentile bootstrap CI over records."""
    point = auroc(scores, labels)
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    keep = np.isfinite(scores) & np.isfinite(labels)
    scores, labels = scores[keep], labels[keep]
    if scores.size == 0 or len(np.unique(labels)) < 2:
        return point, float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        index = rng.integers(0, scores.size, scores.size)
        if len(np.unique(labels[index])) < 2:
            continue
        draws.append(auroc(scores[index], labels[index]))
    if not draws:
        return point, float("nan"), float("nan")
    return point, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def mann_whitney_p(values: pd.Series, labels: pd.Series) -> float:
    """Two-sided Mann-Whitney U p-value between the outcome groups."""
    values = pd.to_numeric(values, errors="coerce")
    frame = pd.DataFrame({"x": values, "y": labels}).dropna()
    group_a = frame.loc[frame["y"] == 1, "x"]
    group_b = frame.loc[frame["y"] == 0, "x"]
    if group_a.empty or group_b.empty:
        return float("nan")
    return float(stats.mannwhitneyu(group_a, group_b, alternative="two-sided")[1])


def odds_ratio(exposed_events: int, exposed_total: int,
               unexposed_events: int, unexposed_total: int
               ) -> Tuple[float, float, float]:
    """Odds ratio with a 95% CI; Haldane +0.5 correction when any cell is zero."""
    a = exposed_events
    b = exposed_total - exposed_events
    c = unexposed_events
    d = unexposed_total - unexposed_events
    if min(a, b, c, d) == 0:
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    ratio = (a * d) / (b * c)
    se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    return (float(ratio), float(ratio * np.exp(-1.96 * se)),
            float(ratio * np.exp(1.96 * se)))


def fisher_p(table: np.ndarray) -> float:
    """Fisher exact p for a 2x2 table; chi-square for larger tables."""
    table = np.asarray(table)
    if table.shape == (2, 2):
        return float(stats.fisher_exact(table)[1])
    if (table.sum(axis=0) == 0).any() or (table.sum(axis=1) == 0).any():
        return float("nan")
    return float(stats.chi2_contingency(table)[1])


def cramers_v(a: pd.Series, b: pd.Series) -> float:
    """Cramer's V between two categorical series (bias-corrected)."""
    table = pd.crosstab(a, b)
    if table.size == 0 or table.shape[0] < 2 or table.shape[1] < 2:
        return float("nan")
    chi2 = stats.chi2_contingency(table)[0]
    n = table.to_numpy().sum()
    phi2 = chi2 / n
    rows, cols = table.shape
    phi2_corr = max(0.0, phi2 - (cols - 1) * (rows - 1) / (n - 1))
    rows_corr = rows - (rows - 1) ** 2 / (n - 1)
    cols_corr = cols - (cols - 1) ** 2 / (n - 1)
    denom = min(cols_corr - 1, rows_corr - 1)
    return float(np.sqrt(phi2_corr / denom)) if denom > 0 else float("nan")


def category_table(series: pd.Series, outcome: pd.Series) -> pd.DataFrame:
    """Counts, %, adverse events and Wilson CI per category of a variable."""
    frame = pd.DataFrame({"cat": series, "y": outcome})
    rows = []
    total = int(frame["cat"].notna().sum())
    for category, group in frame.dropna(subset=["cat"]).groupby("cat", dropna=True):
        known = group["y"].notna()
        n_cat = int(len(group))
        events = int(group.loc[known, "y"].sum())
        n_known = int(known.sum())
        low, high = wilson_ci(events, n_known)
        rows.append({"category": category, "n": n_cat,
                     "pct": 100.0 * n_cat / total if total else np.nan,
                     "n_adverse": events,
                     "pct_adverse": 100.0 * events / n_known if n_known else np.nan,
                     "ci_low": 100 * low, "ci_high": 100 * high})
    return pd.DataFrame(rows).sort_values("category").reset_index(drop=True)


def category_flags(table: pd.DataFrame, n_total: int,
                   expected_codes: Optional[Sequence] = None) -> List[str]:
    """Return RARE / SPARSE-EVENTS / NEAR-CONSTANT / UNEXPECTED-CODE flags."""
    flags: List[str] = []
    if table.empty:
        return ["NO-DATA"]
    if (table["pct"] < 5).any():
        flags.append("RARE")
    if (table["n_adverse"] < 5).any():
        flags.append("SPARSE-EVENTS")
    if (table["pct"] >= 95).any():
        flags.append("NEAR-CONSTANT")
    if expected_codes is not None:
        unexpected = [c for c in table["category"] if c not in expected_codes]
        if unexpected:
            flags.append("UNEXPECTED-CODE")
    return flags


# -------------------------------------------------------------------- figures
def _finish(fig, title: str, subtitle: str = "") -> None:
    """Apply the shared title/subtitle treatment and tighten the layout."""
    if subtitle:
        fig.suptitle(title, fontsize=12.5, y=1.015)
        fig.text(0.5, 0.985, subtitle, ha="center", fontsize=9.8,
                 color="#5b6472")
    else:
        fig.suptitle(title, fontsize=12.5)
    fig.tight_layout()


def plot_continuous_panel(values: pd.Series, outcome: pd.Series, name: str,
                          unit: str = "", limits: Optional[Tuple[float, float]] = None,
                          with_qq: bool = True):
    """Four-panel card for a continuous variable: histogram, box, by outcome, Q-Q."""
    clean = pd.to_numeric(values, errors="coerce")
    data = clean.dropna()
    n_panels = 4 if with_qq else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(4.0 * n_panels, 3.4))

    # 1. histogram with KDE, median and IQR
    axis = axes[0]
    if data.size:
        bins = min(30, max(8, int(np.sqrt(data.size))))
        axis.hist(data, bins=bins, color=COLOUR_NORMAL, edgecolor="white", alpha=0.9)
        if data.nunique() > 2:
            kde = stats.gaussian_kde(data)
            grid = np.linspace(data.min(), data.max(), 200)
            scale = len(data) * (data.max() - data.min()) / bins
            axis.plot(grid, kde(grid) * scale, color=COLOUR_ACCENT, lw=1.6)
        q1, q3 = data.quantile(0.25), data.quantile(0.75)
        axis.axvspan(q1, q3, color=COLOUR_ACCENT, alpha=0.10)
        axis.axvline(data.median(), color=COLOUR_ACCENT, lw=1.6, ls="--")
    axis.set_xlabel(f"{name}{f' ({unit})' if unit else ''}")
    axis.set_ylabel("Labours")
    axis.set_title("Distribution")

    # 2. box plot with plausibility limits
    axis = axes[1]
    if data.size:
        axis.boxplot(data, vert=False, widths=0.55,
                     flierprops={"marker": "o", "markersize": 3.5,
                                 "markerfacecolor": COLOUR_NEUTRAL,
                                 "markeredgecolor": "none", "alpha": 0.6},
                     medianprops={"color": COLOUR_ACCENT, "lw": 1.8},
                     boxprops={"color": "#44506b"},
                     whiskerprops={"color": "#44506b"}, capprops={"color": "#44506b"})
    if limits:
        for limit in limits:
            axis.axvline(limit, color=COLOUR_LIMIT, ls="--", lw=1.3)
        axis.set_title(f"Spread (red = plausible range {limits[0]:g}-{limits[1]:g})",
                       fontsize=10)
    else:
        axis.set_title("Spread")
    axis.set_xlabel(f"{name}{f' ({unit})' if unit else ''}")
    axis.set_yticks([])

    # 3. by outcome
    axis = axes[2]
    groups = [clean[(outcome == value) & clean.notna()].to_numpy() for value in (0, 1)]
    labels = [f"{GROUP_NAMES[v]}\nn={groups[v].size}" for v in (0, 1)]
    if all(g.size for g in groups):
        parts = axis.boxplot(groups, widths=0.5, patch_artist=True,
                             medianprops={"color": "white", "lw": 1.8},
                             flierprops={"marker": "o", "markersize": 3,
                                         "alpha": 0.5, "markeredgecolor": "none",
                                         "markerfacecolor": COLOUR_NEUTRAL})
        for patch, value in zip(parts["boxes"], (0, 1)):
            patch.set_facecolor(GROUP_COLOURS[value])
            patch.set_alpha(0.85)
            patch.set_edgecolor("none")
    axis.set_ylabel(f"{name}{f' ({unit})' if unit else ''}")
    axis.set_title("By outcome")

    # 4. Q-Q
    if with_qq:
        axis = axes[3]
        if data.size > 3:
            stats.probplot(data, dist="norm", plot=axis)
            axis.get_lines()[0].set(markerfacecolor=COLOUR_NORMAL,
                                    markeredgecolor="none", markersize=3.5)
            axis.get_lines()[1].set(color=COLOUR_LIMIT, lw=1.4)
        axis.set_title("Normal Q-Q")
        axis.set_xlabel("Theoretical quantiles")
        axis.set_ylabel("Observed")

    _finish(fig, f"{name}", f"n = {data.size} non-missing of {len(clean)} labours")
    return fig


def plot_categorical_panel(series: pd.Series, outcome: pd.Series, name: str,
                           table: pd.DataFrame, prevalence: float):
    """Three-panel card: category counts, adverse rate with CIs, contingency tiles."""
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.6))
    categories = [str(c) for c in table["category"]]
    positions = np.arange(len(categories))

    # 1. counts
    axis = axes[0]
    axis.bar(positions, table["n"], color=COLOUR_NEUTRAL, edgecolor="white")
    for x, (count, pct) in enumerate(zip(table["n"], table["pct"])):
        axis.annotate(f"{int(count)}\n{pct:.1f}%", (x, count), ha="center",
                      va="bottom", fontsize=9)
    axis.set_xticks(positions)
    axis.set_xticklabels(categories)
    axis.set_ylabel("Labours")
    axis.set_ylim(0, table["n"].max() * 1.28 if len(table) else 1)
    axis.set_title("Category counts")

    # 2. adverse rate with Wilson CIs
    axis = axes[1]
    errors = np.vstack([table["pct_adverse"] - table["ci_low"],
                        table["ci_high"] - table["pct_adverse"]])
    axis.bar(positions, table["pct_adverse"], color=COLOUR_ADVERSE, alpha=0.85,
             edgecolor="white")
    axis.errorbar(positions, table["pct_adverse"], yerr=np.abs(errors), fmt="none",
                  ecolor="#44506b", capsize=4, lw=1.2)
    axis.axhline(100 * prevalence, ls="--", lw=1.3, color=COLOUR_ACCENT,
                 label=f"overall {100 * prevalence:.1f}%")
    for x, (rate, n_adv, n_cat) in enumerate(zip(table["pct_adverse"],
                                                 table["n_adverse"], table["n"])):
        axis.annotate(f"{n_adv}/{n_cat}", (x, 0), ha="center", va="bottom",
                      fontsize=8.5, color="white", xytext=(0, 3),
                      textcoords="offset points")
    axis.set_xticks(positions)
    axis.set_xticklabels(categories)
    axis.set_ylabel("% adverse")
    axis.legend(frameon=False, loc="upper right")
    axis.set_title("Adverse rate per category (95% Wilson CI)")

    # 3. contingency tiles
    axis = axes[2]
    counts = np.vstack([(table["n"] - table["n_adverse"]).to_numpy(),
                        table["n_adverse"].to_numpy()])
    row_pct = counts / np.maximum(counts.sum(axis=0, keepdims=True), 1) * 100
    axis.imshow(row_pct, cmap="Blues", aspect="auto", vmin=0, vmax=100)
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            axis.text(j, i, f"{int(counts[i, j])}\n{row_pct[i, j]:.0f}%",
                      ha="center", va="center", fontsize=9.5,
                      color="white" if row_pct[i, j] > 55 else "#1a1d23")
    axis.set_xticks(positions)
    axis.set_xticklabels(categories)
    axis.set_yticks([0, 1])
    axis.set_yticklabels(["Normal", "Adverse"])
    axis.grid(False)
    axis.set_title("Counts (column %)")

    _finish(fig, name, f"n = {int(table['n'].sum())} labours with a recorded value")
    return fig


def plot_heatmap(matrix: pd.DataFrame, title: str, subtitle: str = "",
                 threshold: float = 0.3, cmap: str = "RdBu_r",
                 vmin: float = -1, vmax: float = 1, figsize=(9.0, 7.5)):
    """Annotated correlation-style heat map; |value| >= threshold is outlined."""
    fig, axis = plt.subplots(figsize=figsize)
    image = axis.imshow(matrix.to_numpy(dtype=float), cmap=cmap, vmin=vmin, vmax=vmax)
    labels = [str(c) for c in matrix.columns]
    axis.set_xticks(range(len(labels)))
    axis.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    axis.set_yticks(range(len(matrix.index)))
    axis.set_yticklabels([str(i) for i in matrix.index], fontsize=9)
    axis.grid(False)

    values = matrix.to_numpy(dtype=float)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            if not np.isfinite(value):
                continue
            axis.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7.6,
                      color="white" if abs(value) > 0.55 else "#1a1d23")
            if abs(value) >= threshold and i != j:
                axis.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                             edgecolor="#111", lw=1.4))
    fig.colorbar(image, ax=axis, shrink=0.8, label="coefficient")
    _finish(fig, title, subtitle)
    return fig


def plot_missingness_matrix(frame: pd.DataFrame, outcome: pd.Series, title: str):
    """Binary image of missing values; rows sorted by outcome, no record labels."""
    order = np.argsort(outcome.fillna(-1).to_numpy(), kind="stable")
    missing = frame.isna().to_numpy()[order].astype(float)
    fig, axis = plt.subplots(figsize=(11.0, 5.0))
    axis.imshow(missing, aspect="auto", cmap="Greys", vmin=0, vmax=1,
                interpolation="nearest")
    axis.set_xticks(range(frame.shape[1]))
    axis.set_xticklabels(frame.columns, rotation=45, ha="right", fontsize=8.5)
    axis.set_ylabel("Labours (sorted by outcome; no identifiers)")
    axis.set_yticks([])
    axis.grid(False)
    _finish(fig, title, "black = missing, white = recorded")
    return fig


def plot_rate_bars(labels: Sequence[str], events: Sequence[int], totals: Sequence[int],
                   title: str, subtitle: str = "", xlabel: str = "",
                   reference: Optional[float] = None):
    """Bar chart of event rates with Wilson CIs, n printed under each bar."""
    rates, lows, highs = [], [], []
    for event, total in zip(events, totals):
        rate = 100.0 * event / total if total else np.nan
        low, high = wilson_ci(int(event), int(total))
        rates.append(rate)
        lows.append(rate - 100 * low)
        highs.append(100 * high - rate)

    fig, axis = plt.subplots(figsize=(max(6.5, 1.5 * len(labels) + 2), 4.0))
    positions = np.arange(len(labels))
    axis.bar(positions, rates, color=COLOUR_ADVERSE, alpha=0.85, edgecolor="white")
    axis.errorbar(positions, rates, yerr=np.vstack([np.abs(lows), np.abs(highs)]),
                  fmt="none", ecolor="#44506b", capsize=4, lw=1.2)
    if reference is not None:
        axis.axhline(reference, ls="--", lw=1.3, color=COLOUR_ACCENT,
                     label=f"overall {reference:.1f}%")
        axis.legend(frameon=False)
    for x, (rate, event, total) in enumerate(zip(rates, events, totals)):
        axis.annotate(f"{rate:.1f}%", (x, rate), ha="center", va="bottom",
                      fontsize=9.5, xytext=(0, 6), textcoords="offset points")
        axis.annotate(f"{int(event)}/{int(total)}", (x, 0), ha="center", va="bottom",
                      fontsize=8.5, color="white", xytext=(0, 3),
                      textcoords="offset points")
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, rotation=0 if len(labels) <= 6 else 30,
                         ha="center" if len(labels) <= 6 else "right")
    axis.set_ylabel("% adverse")
    axis.set_xlabel(xlabel)
    axis.set_ylim(0, max([r + h for r, h in zip(rates, highs)] + [1]) * 1.25)
    _finish(fig, title, subtitle)
    return fig
