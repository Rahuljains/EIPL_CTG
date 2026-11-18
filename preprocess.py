"""Causal signal cleaning, shared by CTU-UHB and JNU-CTG.

Every step here uses only past and present samples, with one documented
exception: linear interpolation of a gap of <= MAX_INTERP_S seconds uses the
first valid sample after the gap. That is at most 15 s of look-ahead and is used
only for the Layer 2 windows; Layer 1 runs on the un-interpolated series.

Author: Rahul Jain
Date:   17 Nov 2026
Usage:  from ctg.preprocess import clean_fhr, clean_uc, to_1hz, interpolate_short_gaps
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from . import config as C


def clean_fhr(fhr_4hz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Clean a 4 Hz FHR trace causally. Return (cleaned bpm, originally-valid mask).

    Steps: 0 and out-of-range values become NaN, then a sample more than
    SPIKE_BPM away from the trailing SPIKE_WIN_S median becomes NaN.
    """
    fhr = np.asarray(fhr_4hz, dtype=np.float64).copy()
    fhr[fhr == 0] = np.nan
    fhr[(fhr < C.FHR_VALID_MIN) | (fhr > C.FHR_VALID_MAX)] = np.nan

    window = int(C.SPIKE_WIN_S * C.FS_RAW)
    trailing_median = (pd.Series(fhr)
                       .rolling(window=window, min_periods=3)
                       .median()
                       .to_numpy())
    spike = np.abs(fhr - trailing_median) > C.SPIKE_BPM   # NaN comparisons give False
    fhr[spike] = np.nan

    return fhr, np.isfinite(fhr)


def clean_uc(uc_4hz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Clean a 4 Hz UC trace: exact zeros become NaN. Return (cleaned, valid mask)."""
    uc = np.asarray(uc_4hz, dtype=np.float64).copy()
    uc[uc == 0] = np.nan
    return uc, np.isfinite(uc)


def to_1hz(signal_4hz: np.ndarray,
           min_valid: int = C.MIN_1HZ_VALID_SAMPLES) -> Tuple[np.ndarray, np.ndarray]:
    """Average each second of a 4 Hz signal. Return (1 Hz series, valid mask).

    A second with fewer than `min_valid` valid 4 Hz samples is NaN.
    """
    signal = np.asarray(signal_4hz, dtype=np.float64)
    n_seconds = len(signal) // C.FS_RAW
    trimmed = signal[:n_seconds * C.FS_RAW].reshape(n_seconds, C.FS_RAW)

    valid = np.isfinite(trimmed)
    counts = valid.sum(axis=1)
    sums = np.where(valid, trimmed, 0.0).sum(axis=1)

    out = np.full(n_seconds, np.nan)
    enough = counts >= min_valid
    out[enough] = sums[enough] / counts[enough]
    return out, enough


def interpolate_short_gaps(signal_1hz: np.ndarray,
                           max_gap_s: int = C.MAX_INTERP_S) -> np.ndarray:
    """Linearly fill gaps of at most max_gap_s seconds. Longer gaps stay NaN.

    Leading and trailing NaN runs are never filled (no value to interpolate from).
    """
    signal = np.asarray(signal_1hz, dtype=np.float64).copy()
    missing = ~np.isfinite(signal)
    if not missing.any():
        return signal

    if np.count_nonzero(~missing) < 2:
        return signal

    for start, stop in _runs(missing):
        if stop - start > max_gap_s:
            continue
        if start == 0 or stop >= len(signal):
            continue                                    # no bracketing valid sample
        left, right = signal[start - 1], signal[stop]
        if not (np.isfinite(left) and np.isfinite(right)):
            continue
        steps = stop - start + 1
        signal[start:stop] = left + (right - left) * np.arange(1, steps) / steps
    return signal


def _runs(flags: np.ndarray) -> list:
    """Return [start, stop) index pairs of each True run in a boolean array."""
    padded = np.concatenate(([False], flags.astype(bool), [False]))
    changes = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return list(zip(starts.tolist(), stops.tolist()))


def prepare_signals(fhr_4hz: np.ndarray, uc_4hz: np.ndarray) -> dict:
    """Run the full cleaning pipeline and return every array the later stages need.

    Keys: fhr_4hz, fhr_mask_4hz, uc_4hz, uc_mask_4hz, fhr_1hz_raw, valid_1hz,
    fhr_1hz (gap-interpolated), uc_1hz, uc_valid_1hz.
    """
    fhr_clean, fhr_mask = clean_fhr(fhr_4hz)
    uc_clean, uc_mask = clean_uc(uc_4hz)

    fhr_1hz_raw, valid_1hz = to_1hz(fhr_clean)
    uc_1hz, uc_valid_1hz = to_1hz(uc_clean, min_valid=1)
    fhr_1hz = interpolate_short_gaps(fhr_1hz_raw)

    return {"fhr_4hz": fhr_clean, "fhr_mask_4hz": fhr_mask,
            "uc_4hz": uc_clean, "uc_mask_4hz": uc_mask,
            "fhr_1hz_raw": fhr_1hz_raw, "valid_1hz": valid_1hz,
            "fhr_1hz": fhr_1hz, "uc_1hz": uc_1hz, "uc_valid_1hz": uc_valid_1hz}


def window_grid(n_seconds: int, window_min: int = C.WINDOW_MIN,
                grid_min: int = C.GRID_MIN) -> np.ndarray:
    """Return window-end times in minutes: window_min, +grid_min, ... <= recording end."""
    length_min = n_seconds / 60.0
    if length_min < window_min:
        return np.array([], dtype=float)
    n_steps = int((length_min - window_min) // grid_min)
    return window_min + grid_min * np.arange(n_steps + 1, dtype=float)


def extract_window(signal_1hz: np.ndarray, t_end_min: float,
                   window_min: int = C.WINDOW_MIN) -> np.ndarray:
    """Return the samples covering [t_end - window_min, t_end) at 1 Hz."""
    end = int(round(t_end_min * 60))
    start = end - window_min * 60
    return signal_1hz[max(start, 0):end]
