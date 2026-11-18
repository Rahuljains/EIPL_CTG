"""Layer 1: the guideline rule engine.

Five RED rules (R1-R5), two notes (N1, N2) and a signal-quality flag (Q),
evaluated every RULE_STEP_S seconds on the causal detector output. Layer 1 is
RED while any R-rule is active; R1 at EXPEDITE_DECEL_MIN minutes raises HIGH
RISK immediately.

These thresholds still need clinician confirmation (plan Section 12).

Author: Rahul Jain
Date:   17 Nov 2026
Usage:  from ctg.rules import evaluate_rules
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import config as C
from .detect import DetectorOutput, contraction_times_known_by

RULE_NAMES = ["R1", "R2", "R3", "R4", "R5"]
NOTE_NAMES = ["N1", "N2"]
RULE_DESCRIPTIONS = {
    "R1": f"Deceleration lasting >= {C.PROLONGED_DECEL_MIN} min "
          f"(>= {C.EXPEDITE_DECEL_MIN} min = expedite birth)",
    "R2": f"Baseline < {C.LOW_BASELINE_BPM} bpm",
    "R3": f"Amplitude < {C.LOW_VAR_BPM} bpm for >= {C.LOW_VAR_MIN} consecutive "
          "valid minutes",
    "R4": f"A deceleration in each 10-min block of the last {C.PERSIST_DECEL_MIN} min, "
          f"with a rising baseline or reduced variability",
    "R5": f"Second stage: baseline risen >= {C.RISING_BASELINE_BPM} bpm above the "
          "initial baseline",
    "N1": f"First stage: baseline risen >= {C.RISING_BASELINE_BPM} bpm (note only)",
    "N2": f">= {C.EXCESS_UC_PER_10MIN} contractions in 10 min, or one lasting "
          f"> {C.LONG_UC_S} s (note only)",
}


def _low_variability_run(amplitude: np.ndarray, known: np.ndarray,
                         minute_index: int) -> int:
    """Count consecutive valid minutes ending at minute_index with amplitude < threshold.

    Unknown minutes are skipped: they neither extend nor reset the run.
    """
    run = 0
    for minute in range(minute_index, -1, -1):
        if not known[minute]:
            continue
        if amplitude[minute] < C.LOW_VAR_BPM:
            run += 1
        else:
            break
    return run


def _reduced_variability_recent(amplitude: np.ndarray, known: np.ndarray,
                                minute_index: int, n_recent: int = 10,
                                n_required: int = 5) -> bool:
    """True when at least n_required of the last n_recent valid minutes are flat."""
    collected: List[float] = []
    for minute in range(minute_index, -1, -1):
        if not known[minute]:
            continue
        collected.append(amplitude[minute])
        if len(collected) == n_recent:
            break
    if not collected:
        return False
    return sum(value < C.LOW_VAR_BPM for value in collected) >= n_required


def _decel_in_each_block(decel_starts: np.ndarray, t_s: int) -> bool:
    """True when a deceleration started in each 10-minute block of the last 30 min."""
    window_start = t_s - C.PERSIST_DECEL_MIN * 60
    if window_start < 0:
        return False
    block_length = (C.PERSIST_DECEL_MIN * 60) // 3
    for block in range(3):
        low = window_start + block * block_length
        high = low + block_length
        if not np.any((decel_starts >= low) & (decel_starts < high)):
            return False
    return True


def evaluate_rules(detectors: DetectorOutput, valid_1hz: np.ndarray,
                   stage2_start_s: int = -1,
                   step_s: int = C.RULE_STEP_S) -> pd.DataFrame:
    """Evaluate Layer 1 every step_s seconds. Return one row per step.

    Columns: t_s, baseline, in_decel, decel_dur_s, R1..R5, N1, N2, signal_poor,
    l1_red, l1_expedite.
    """
    n = len(valid_1hz)
    decel_starts = np.array([event.start_s for event in detectors.decelerations],
                            dtype=int)
    decel_durations = np.array([event.duration_s for event in detectors.decelerations],
                               dtype=int)
    initial = detectors.initial_baseline

    rows: List[Dict[str, object]] = []
    previous: Optional[Dict[str, object]] = None

    for t_s in range(0, n, step_s):
        minute_index = t_s // 60 - 1            # last fully completed minute
        baseline = detectors.baseline[t_s]
        decel_dur = float(detectors.decel_duration_s[t_s])

        # ---- Q: signal quality over the last 60 s
        window_start = max(0, t_s - C.SIGNAL_POOR_WIN_S + 1)
        recent_valid = valid_1hz[window_start:t_s + 1]
        valid_frac = float(np.mean(recent_valid)) if recent_valid.size else 0.0
        signal_poor = valid_frac < C.SIGNAL_POOR_FRAC

        if signal_poor and previous is not None:
            # rules that cannot be evaluated hold their last state
            carried = dict(previous)
            carried.update({"t_s": t_s, "baseline": baseline,
                            "in_decel": bool(detectors.in_decel[t_s]),
                            "decel_dur_s": decel_dur, "signal_poor": True,
                            "valid_frac": round(valid_frac, 3)})
            rows.append(carried)
            continue

        # ---- R1: a prolonged deceleration in progress
        r1 = bool(detectors.in_decel[t_s]
                  and decel_dur >= C.PROLONGED_DECEL_MIN * 60)
        expedite = bool(detectors.in_decel[t_s]
                        and decel_dur >= C.EXPEDITE_DECEL_MIN * 60)
        if not expedite and decel_durations.size:
            # a completed deceleration that already reached the expedite length
            reached = decel_durations[decel_starts + decel_durations <= t_s]
            expedite = bool(np.any(reached >= C.EXPEDITE_DECEL_MIN * 60))

        # ---- R2: baseline bradycardia
        r2 = bool(np.isfinite(baseline) and baseline < C.LOW_BASELINE_BPM)

        # ---- R3: sustained reduced variability
        r3 = False
        if minute_index >= 0 and minute_index < len(detectors.minute_amplitude):
            r3 = _low_variability_run(detectors.minute_amplitude,
                                      detectors.minute_known,
                                      minute_index) >= C.LOW_VAR_MIN

        # ---- R5 / N1: a rising baseline against the initial baseline
        rising = bool(np.isfinite(baseline) and np.isfinite(initial)
                      and (baseline - initial) >= C.RISING_BASELINE_BPM)
        in_second_stage = 0 <= stage2_start_s <= t_s
        r5 = rising and in_second_stage
        n1 = rising and not in_second_stage

        # ---- R4: persistent decelerations plus a secondary abnormality
        secondary = rising
        if not secondary and 0 <= minute_index < len(detectors.minute_amplitude):
            secondary = _reduced_variability_recent(detectors.minute_amplitude,
                                                    detectors.minute_known,
                                                    minute_index)
        r4 = bool(_decel_in_each_block(decel_starts, t_s) and secondary)

        # ---- N2: excessive or prolonged contractions
        n2 = False
        if detectors.contractions_reliable:
            known_peaks = contraction_times_known_by(detectors, t_s)
            recent = known_peaks[known_peaks >= t_s - 600]
            long_contraction = False
            if detectors.contraction_widths_s.size:
                confirmed = np.isin(detectors.contraction_peaks_s, known_peaks)
                long_contraction = bool(np.any(
                    detectors.contraction_widths_s[confirmed] > C.LONG_UC_S))
            n2 = bool(len(recent) >= C.EXCESS_UC_PER_10MIN or long_contraction)

        row = {"t_s": t_s, "baseline": baseline,
               "in_decel": bool(detectors.in_decel[t_s]), "decel_dur_s": decel_dur,
               "R1": r1, "R2": r2, "R3": r3, "R4": r4, "R5": r5,
               "N1": n1, "N2": n2, "signal_poor": False,
               "valid_frac": round(valid_frac, 3)}
        row["l1_red"] = bool(r1 or r2 or r3 or r4 or r5)
        row["l1_expedite"] = bool(expedite and C.EXPEDITE_TRIGGERS_HIGH_RISK)
        rows.append(row)
        previous = row

    frame = pd.DataFrame(rows)
    if frame.empty:
        frame = pd.DataFrame(columns=["t_s", "baseline", "in_decel", "decel_dur_s"]
                             + RULE_NAMES + NOTE_NAMES
                             + ["signal_poor", "valid_frac", "l1_red", "l1_expedite"])
    # once the expedite condition is met it stays met (birth is being expedited)
    if "l1_expedite" in frame and len(frame):
        frame["l1_expedite"] = frame["l1_expedite"].astype(bool).cummax()
    return frame


def summarise_record(states: pd.DataFrame, record_id: int) -> Dict[str, object]:
    """One summary row per labour: ever red, first red minute, minutes red per rule."""
    summary: Dict[str, object] = {"record_id": record_id}
    red = states["l1_red"].to_numpy(dtype=bool)
    step_min = C.RULE_STEP_S / 60.0

    summary["ever_red"] = bool(red.any())
    summary["first_red_min"] = (float(states.loc[red, "t_s"].iloc[0] / 60.0)
                                if red.any() else np.nan)
    summary["minutes_red"] = float(red.sum() * step_min)
    summary["ever_expedite"] = bool(states["l1_expedite"].any())
    summary["first_expedite_min"] = (
        float(states.loc[states["l1_expedite"], "t_s"].iloc[0] / 60.0)
        if states["l1_expedite"].any() else np.nan)
    summary["pct_signal_poor"] = float(100 * states["signal_poor"].mean())

    for rule in RULE_NAMES + NOTE_NAMES:
        flags = states[rule].to_numpy(dtype=bool)
        summary[f"ever_{rule}"] = bool(flags.any())
        summary[f"minutes_{rule}"] = float(flags.sum() * step_min)
        summary[f"first_{rule}_min"] = (float(states.loc[flags, "t_s"].iloc[0] / 60.0)
                                        if flags.any() else np.nan)
    return summary


def rule_events(states: pd.DataFrame, record_id: int) -> List[Dict[str, object]]:
    """One row per rule activation: record, rule, start_min, end_min."""
    events: List[Dict[str, object]] = []
    times = states["t_s"].to_numpy()
    for rule in RULE_NAMES + NOTE_NAMES:
        flags = states[rule].to_numpy(dtype=bool)
        if not flags.any():
            continue
        padded = np.concatenate(([False], flags, [False]))
        changes = np.diff(padded.astype(np.int8))
        for start, stop in zip(np.flatnonzero(changes == 1),
                               np.flatnonzero(changes == -1)):
            events.append({"record_id": record_id, "rule": rule,
                           "start_min": round(float(times[start]) / 60.0, 2),
                           "end_min": round(float(times[min(stop, len(times) - 1)])
                                            / 60.0, 2),
                           "duration_min": round(float(times[min(stop, len(times) - 1)]
                                                       - times[start]) / 60.0, 2)})
    return events
