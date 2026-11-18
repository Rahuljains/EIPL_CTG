"""Causal CTG pattern detectors: baseline, decelerations, accelerations, variability,
contractions.

Shared by the Layer 1 rule engine (M3) and the Track F window features (M6), so
both see exactly the same definition of a deceleration.

Causality: the engine walks forward one second at a time and every quantity at
time t is computed from samples at or before t. The baseline is recomputed on
the minute from a trailing 10-minute window and is frozen while a drop is in
progress, so a prolonged deceleration cannot drag the baseline down with it.
Contractions are the one place where a peak is only reported once its shape has
fully passed (peak time + CONTRACTION_CONFIRM_S), which keeps them causal too.

Author: Rahul Jain
Date:   17 Nov 2026
Usage:  from ctg.detect import run_detectors, DetectorOutput
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import signal as sp_signal

from . import config as C

# A peak is only "known" this long after it occurs, and its prominence and width
# are evaluated inside a window of +/- this many seconds (find_peaks `wlen`). Both
# halves matter: without the bounded window, scipy measures prominence against the
# whole trace, so a peak detected at minute 10 could change once minute 40 arrived.
# That made Track F's contraction feature depend on the future, which a live
# monitor cannot do - caught by the streaming-vs-batch engine test.
CONTRACTION_CONFIRM_S = 60
MIN_BASELINE_SAMPLES = 60       # valid 1 Hz samples needed before a baseline exists


@dataclass
class Event:
    """One deceleration or acceleration episode."""

    start_s: int
    end_s: int
    duration_s: int
    depth_bpm: float             # max drop below (or rise above) the baseline
    area_bpm_min: float          # sum of |baseline - FHR| over the event, in bpm.min

    def as_dict(self, kind: str) -> Dict[str, object]:
        """Flatten for a csv row."""
        return {"kind": kind, "start_s": self.start_s, "end_s": self.end_s,
                "duration_s": self.duration_s, "depth_bpm": round(self.depth_bpm, 2),
                "area_bpm_min": round(self.area_bpm_min, 3)}


@dataclass
class DetectorOutput:
    """Everything the rule engine and the Track F features need, on a 1 Hz grid."""

    baseline: np.ndarray                 # bpm per second, NaN before it is established
    initial_baseline: float              # the baseline at BASELINE_WIN_MIN minutes
    in_decel: np.ndarray                 # bool: a confirmed deceleration is active
    decel_duration_s: np.ndarray         # seconds the current drop has lasted
    in_accel: np.ndarray
    decelerations: List[Event] = field(default_factory=list)
    accelerations: List[Event] = field(default_factory=list)
    minute_amplitude: np.ndarray = field(default_factory=lambda: np.empty(0))
    minute_known: np.ndarray = field(default_factory=lambda: np.empty(0, bool))
    contraction_peaks_s: np.ndarray = field(default_factory=lambda: np.empty(0, int))
    contraction_widths_s: np.ndarray = field(default_factory=lambda: np.empty(0))
    contractions_reliable: bool = True


def _robust_baseline(window: np.ndarray, exclude: np.ndarray) -> float:
    """Median of a trailing FHR window, repeated after dropping outlying samples.

    `exclude` marks samples inside a detected deceleration or acceleration; they
    never contribute, so an episode cannot pull the baseline towards itself.
    """
    usable = window[np.isfinite(window) & ~exclude]
    if usable.size < MIN_BASELINE_SAMPLES:
        return np.nan
    preliminary = np.median(usable)
    refined = usable[np.abs(usable - preliminary) <= C.BASELINE_EXCL_BPM]
    if refined.size < MIN_BASELINE_SAMPLES // 2:
        return float(preliminary)
    return float(np.median(refined))


def _minute_amplitudes(fhr_4hz: np.ndarray, decelerations: List[Event],
                       accelerations: List[Event],
                       n_minutes: int) -> Tuple[np.ndarray, np.ndarray]:
    """Per-minute amplitude (max - min) of the valid 4 Hz FHR, episodes excluded.

    A minute with less than MIN_VALID_FRAC_MINUTE valid data is "unknown": it
    neither extends nor resets a low-variability run.

    Causality: a minute's amplitude is finalised at the end of that minute, so it
    may only exclude episode samples that were already confirmed by then. An
    episode is confirmed once it has lasted its minimum duration, and its extent
    is truncated to the end of the minute. Without this, a drop beginning in the
    last seconds of a minute would retroactively change that minute's amplitude
    according to whether it continued afterwards.
    """
    amplitude = np.full(n_minutes, np.nan)
    known = np.zeros(n_minutes, dtype=bool)
    samples_per_min = 60 * C.FS_RAW
    episodes = ([(event, C.DECEL_MIN_S) for event in decelerations]
                + [(event, C.ACCEL_MIN_S) for event in accelerations])

    for minute in range(n_minutes):
        start = minute * samples_per_min
        chunk = fhr_4hz[start:start + samples_per_min]
        if chunk.size < samples_per_min:
            break

        minute_start_s, minute_end_s = minute * 60, (minute + 1) * 60
        episode_seconds = np.zeros(60, dtype=bool)
        for event, min_duration in episodes:
            if event.start_s + min_duration > minute_end_s:
                continue                       # not yet confirmed at this minute
            low = max(event.start_s, minute_start_s)
            high = min(event.end_s, minute_end_s)
            if high > low:
                episode_seconds[low - minute_start_s:high - minute_start_s] = True
        episode_4hz = np.repeat(episode_seconds, C.FS_RAW)[:chunk.size]

        usable = chunk[np.isfinite(chunk) & ~episode_4hz]
        if usable.size < C.MIN_VALID_FRAC_MINUTE * samples_per_min:
            continue
        amplitude[minute] = float(usable.max() - usable.min())
        known[minute] = True
    return amplitude, known


def _detect_contractions(uc_1hz: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Find contraction peaks in the UC trace. Return (peak seconds, widths, reliable).

    The UC is smoothed with a trailing 30-second mean (causal), then peaks are
    located. A peak is reported at peak_time + CONTRACTION_CONFIRM_S so that no
    rule ever uses a peak before the data defining it has arrived.
    """
    finite = np.isfinite(uc_1hz)
    reliable = bool(finite.mean() >= (1 - C.UC_UNRELIABLE_FRAC))
    if not reliable or finite.sum() < C.UC_SMOOTH_S * 2:
        return np.empty(0, dtype=int), np.empty(0), reliable

    filled = np.where(finite, uc_1hz, np.nan)
    kernel = np.ones(C.UC_SMOOTH_S)
    values = np.nan_to_num(filled, nan=0.0)
    counts = np.convolve(finite.astype(float), kernel, mode="full")[:len(filled)]
    sums = np.convolve(values, kernel, mode="full")[:len(filled)]
    smoothed = np.divide(sums, counts, out=np.full_like(sums, np.nan),
                         where=counts > 0)

    usable = np.nan_to_num(smoothed, nan=np.nanmin(smoothed) if np.isfinite(
        smoothed).any() else 0.0)
    peaks, properties = sp_signal.find_peaks(
        usable, prominence=C.UC_PROMINENCE, distance=C.UC_MIN_DIST_S,
        width=C.UC_MIN_WIDTH_S, rel_height=0.5,
        wlen=2 * CONTRACTION_CONFIRM_S)
    widths = np.asarray(properties.get("widths", np.empty(0)), dtype=float)

    # A half-prominence width beyond UC_MAX_WIDTH_S is UC baseline drift or several
    # merged contractions, not one contraction lasting that long. Measured on
    # CTU-UHB the widths run to 517 s, which is not physiological, so these peaks
    # are discarded rather than reported as very long contractions.
    if widths.size:
        plausible = widths <= C.UC_MAX_WIDTH_S
        peaks, widths = peaks[plausible], widths[plausible]
    return peaks.astype(int), widths, reliable


def run_detectors(fhr_1hz: np.ndarray, valid_1hz: np.ndarray, fhr_4hz: np.ndarray,
                  uc_1hz: Optional[np.ndarray] = None) -> DetectorOutput:
    """Run every causal detector over one recording and return the 1 Hz series.

    `fhr_1hz` is the gap-interpolated working series; `valid_1hz` marks the
    seconds that were genuinely observed. Decelerations are measured against the
    frozen running baseline.
    """
    n = len(fhr_1hz)
    baseline = np.full(n, np.nan)
    in_decel = np.zeros(n, dtype=bool)
    in_accel = np.zeros(n, dtype=bool)
    decel_duration = np.zeros(n, dtype=float)
    episode_mask = np.zeros(n, dtype=bool)      # decel or accel, for baseline exclusion

    decelerations: List[Event] = []
    accelerations: List[Event] = []

    current_baseline = np.nan
    initial_baseline = np.nan

    # candidate-run state for a drop and for a rise
    drop_start: Optional[int] = None
    drop_last_seen: Optional[int] = None
    rise_start: Optional[int] = None
    rise_last_seen: Optional[int] = None
    window_samples = C.BASELINE_WIN_MIN * 60

    for t in range(n):
        value = fhr_1hz[t]
        observed = bool(valid_1hz[t]) and np.isfinite(value)

        # ---- baseline, recomputed on the minute and frozen inside a drop
        if t % 60 == 0 and t >= 60 and drop_start is None:
            start = max(0, t - window_samples)
            current_baseline = _robust_baseline(fhr_1hz[start:t],
                                                episode_mask[start:t])
            if t == C.BASELINE_WIN_MIN * 60 and np.isfinite(current_baseline):
                initial_baseline = current_baseline
        baseline[t] = current_baseline

        if not np.isfinite(current_baseline):
            continue

        # ---- deceleration: >= DECEL_DROP_BPM below baseline for >= DECEL_MIN_S
        below = observed and (current_baseline - value) >= C.DECEL_DROP_BPM
        if below:
            if drop_start is None:
                drop_start = t
            drop_last_seen = t
        elif drop_start is not None:
            gap = t - (drop_last_seen if drop_last_seen is not None else drop_start)
            recovered = observed and gap > 0
            missing_too_long = (not observed) and gap > C.DECEL_GAP_TOL_S
            if recovered or missing_too_long:
                _close_episode(decelerations, fhr_1hz, baseline, drop_start,
                               drop_last_seen, C.DECEL_MIN_S, sign=1)
                drop_start = drop_last_seen = None

        if drop_start is not None:
            duration = t - drop_start + 1
            decel_duration[t] = duration
            if duration >= C.DECEL_MIN_S:
                in_decel[t] = True
                episode_mask[drop_start:t + 1] = True

        # ---- acceleration: >= ACCEL_RISE_BPM above baseline for >= ACCEL_MIN_S
        above = observed and (value - current_baseline) >= C.ACCEL_RISE_BPM
        if above:
            if rise_start is None:
                rise_start = t
            rise_last_seen = t
        elif rise_start is not None:
            gap = t - (rise_last_seen if rise_last_seen is not None else rise_start)
            if (observed and gap > 0) or ((not observed) and gap > C.DECEL_GAP_TOL_S):
                _close_episode(accelerations, fhr_1hz, baseline, rise_start,
                               rise_last_seen, C.ACCEL_MIN_S, sign=-1)
                rise_start = rise_last_seen = None

        if rise_start is not None and (t - rise_start + 1) >= C.ACCEL_MIN_S:
            in_accel[t] = True
            episode_mask[rise_start:t + 1] = True

    # close anything still open at the end of the recording
    if drop_start is not None:
        _close_episode(decelerations, fhr_1hz, baseline, drop_start, drop_last_seen,
                       C.DECEL_MIN_S, sign=1)
    if rise_start is not None:
        _close_episode(accelerations, fhr_1hz, baseline, rise_start, rise_last_seen,
                       C.ACCEL_MIN_S, sign=-1)

    n_minutes = n // 60
    amplitude, known = _minute_amplitudes(fhr_4hz, decelerations, accelerations,
                                          n_minutes)

    if uc_1hz is None:
        peaks, widths, reliable = np.empty(0, int), np.empty(0), False
    else:
        peaks, widths, reliable = _detect_contractions(uc_1hz)

    return DetectorOutput(
        baseline=baseline, initial_baseline=initial_baseline, in_decel=in_decel,
        decel_duration_s=decel_duration, in_accel=in_accel,
        decelerations=decelerations, accelerations=accelerations,
        minute_amplitude=amplitude, minute_known=known,
        contraction_peaks_s=peaks, contraction_widths_s=widths,
        contractions_reliable=reliable)


def _close_episode(store: List[Event], fhr: np.ndarray, baseline: np.ndarray,
                   start: int, last_seen: Optional[int], min_duration: int,
                   sign: int) -> None:
    """Record a completed episode if it lasted long enough. sign 1 = decel, -1 = accel."""
    end = (last_seen if last_seen is not None else start) + 1
    duration = end - start
    if duration < min_duration:
        return
    segment = fhr[start:end]
    reference = baseline[start:end]
    difference = (reference - segment) * sign
    difference = difference[np.isfinite(difference)]
    if difference.size == 0:
        return
    store.append(Event(start_s=start, end_s=end, duration_s=duration,
                       depth_bpm=float(np.nanmax(difference)),
                       area_bpm_min=float(np.nansum(difference) / 60.0)))


def contraction_times_known_by(output: DetectorOutput, t_s: int) -> np.ndarray:
    """Return the contraction peak times that are already confirmed at time t_s."""
    if output.contraction_peaks_s.size == 0:
        return np.empty(0, dtype=int)
    confirmed = output.contraction_peaks_s + CONTRACTION_CONFIRM_S <= t_s
    return output.contraction_peaks_s[confirmed]
