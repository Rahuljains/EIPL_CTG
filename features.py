"""Track F: guideline-type features computed causally inside one 15-minute window.

These are the quantities a clinician reads off a trace - baseline, variability,
how many decelerations and how deep - computed with the same detectors the
Layer 1 rules use. Track F is the fallback that is guaranteed to finish, and
the head-to-head comparison for the learned Track E embedding.

`valid_frac` is a filter only and is never a predictor: a window is used or it
is not, but how much signal it had must not itself predict the outcome.

Author: Rahul Jain
Date:   17 Nov 2026
Usage:  from ctg.features import window_features, FEATURE_NAMES
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from . import config as C
from .detect import DetectorOutput

FEATURE_NAMES: List[str] = [
    "baseline_bpm",            # 1. baseline at the end of the window
    "baseline_shift_bpm",      # 2. baseline minus the recording's initial baseline
    "stv_bpm",                 # 3. short-term variation, decelerations excluded
    "mean_minute_amplitude",   # 4. mean per-minute amplitude
    "pct_minutes_flat",        # 5. % of minutes with amplitude < 5 bpm
    "n_accelerations",         # 6.
    "n_decelerations",         # 7.
    "decel_area_bpm_min",      # 8. total deceleration area
    "max_decel_depth_bpm",     # 9.
    "longest_decel_s",         # 10.
    "pct_time_below_baseline",  # 11. % of the window >= 15 bpm below baseline
    "contractions_per_10min",  # 12. NaN when the UC is unreliable
]


def _window_slice(t_end_min: float, window_min: int = C.WINDOW_MIN):
    """Return the (start, end) second indices covered by a window."""
    end = int(round(t_end_min * 60))
    return max(0, end - window_min * 60), end


def window_features(detectors: DetectorOutput, fhr_1hz: np.ndarray,
                    valid_1hz: np.ndarray, t_end_min: float,
                    window_min: int = C.WINDOW_MIN) -> Dict[str, float]:
    """Compute the twelve Track F features for the window ending at t_end_min.

    Every quantity uses only samples inside the window, which itself ends at
    t_end_min, so nothing after the reading time can influence the result.
    """
    start, end = _window_slice(t_end_min, window_min)
    fhr = fhr_1hz[start:end]
    valid = valid_1hz[start:end].astype(bool)
    observed = valid & np.isfinite(fhr)

    features = {name: np.nan for name in FEATURE_NAMES}
    if fhr.size == 0:
        return features

    # ---- 1-2. baseline and its shift
    baseline_series = detectors.baseline[start:end]
    baseline = np.nan
    finite_baseline = baseline_series[np.isfinite(baseline_series)]
    if finite_baseline.size:
        baseline = float(finite_baseline[-1])
    features["baseline_bpm"] = baseline
    if np.isfinite(baseline) and np.isfinite(detectors.initial_baseline):
        features["baseline_shift_bpm"] = baseline - detectors.initial_baseline

    # ---- episodes overlapping this window, strictly as known at t_end
    #
    # Two things here keep the features causal. An episode counts only once it
    # has already reached its minimum duration by the window end, so whether the
    # last few seconds of a window belong to a deceleration never depends on
    # whether the drop happens to continue afterwards. And its geometry is
    # measured on the part inside the window only, so an episode that runs past
    # t_end contributes just the depth and area seen so far.
    decels = [e for e in detectors.decelerations
              if e.start_s < end and e.end_s > start
              and e.start_s + C.DECEL_MIN_S <= end]
    accels = [e for e in detectors.accelerations
              if e.start_s < end and e.end_s > start
              and e.start_s + C.ACCEL_MIN_S <= end]

    episode_mask = np.zeros(fhr.size, dtype=bool)
    for event in decels + accels:
        low = max(event.start_s, start) - start
        high = min(event.end_s, end) - start
        episode_mask[low:high] = True

    # ---- 3. short-term variation, decelerations and accelerations excluded
    usable = observed & ~episode_mask
    if usable.sum() >= 2:
        values = fhr.copy()
        values[~usable] = np.nan
        differences = np.abs(np.diff(values))
        differences = differences[np.isfinite(differences)]
        if differences.size:
            features["stv_bpm"] = float(np.mean(differences))

    # ---- 4-5. per-minute amplitude inside the window
    first_minute, last_minute = start // 60, end // 60
    amplitudes = detectors.minute_amplitude[first_minute:last_minute]
    known = detectors.minute_known[first_minute:last_minute]
    if known.any():
        known_amplitudes = amplitudes[known]
        features["mean_minute_amplitude"] = float(np.mean(known_amplitudes))
        features["pct_minutes_flat"] = float(
            100.0 * np.mean(known_amplitudes < C.LOW_VAR_BPM))

    # ---- 6-10. episode counts and geometry, truncated to the window
    features["n_accelerations"] = float(len(accels))
    features["n_decelerations"] = float(len(decels))

    areas, depths, durations = [], [], []
    for event in decels:
        low, high = max(event.start_s, start), min(event.end_s, end)
        segment = fhr_1hz[low:high]
        reference = detectors.baseline[low:high]
        drop = reference - segment
        drop = drop[np.isfinite(drop)]
        areas.append(float(np.sum(drop) / 60.0) if drop.size else 0.0)
        depths.append(float(np.max(drop)) if drop.size else 0.0)
        durations.append(float(high - low))

    features["decel_area_bpm_min"] = float(sum(areas))
    features["max_decel_depth_bpm"] = float(max(depths)) if depths else 0.0
    features["longest_decel_s"] = float(max(durations)) if durations else 0.0

    # ---- 11. time spent well below the baseline
    if np.isfinite(baseline) and observed.any():
        below = observed & ((baseline - fhr) >= C.DECEL_DROP_BPM)
        features["pct_time_below_baseline"] = float(
            100.0 * below.sum() / observed.sum())

    # ---- 12. contractions, NaN when the UC cannot be trusted
    if detectors.contractions_reliable:
        peaks = detectors.contraction_peaks_s
        in_window = int(np.sum((peaks >= start) & (peaks < end)))
        features["contractions_per_10min"] = float(
            in_window * 10.0 / (window_min if window_min else 1))

    return features


def feature_matrix(detectors: DetectorOutput, fhr_1hz: np.ndarray,
                   valid_1hz: np.ndarray, window_times_min: np.ndarray,
                   window_min: int = C.WINDOW_MIN) -> np.ndarray:
    """Stack the Track F features for every window of one labour."""
    rows = [window_features(detectors, fhr_1hz, valid_1hz, t_end, window_min)
            for t_end in window_times_min]
    if not rows:
        return np.empty((0, len(FEATURE_NAMES)))
    return np.array([[row[name] for name in FEATURE_NAMES] for row in rows],
                    dtype=np.float64)
