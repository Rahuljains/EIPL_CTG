"""The sequential update and the zone simulator: how the two layers combine.

The clinical prior sets the starting log odds. Each 15-minute CTG window adds
its log likelihood ratio, shrunk by lambda and faded by gamma, so old evidence
decays and overlapping windows do not count as independent observations. The
posterior after one reading is the prior shown at the next.

    w_r = dt / 15                                 window weight
    S_r = gamma^(dt/15) * S_{r-1} + w_r * logLR_r
    L_r = L_clin(T_r) + lambda * S_r
    p_r = sigmoid(L_r)

An invalid window contributes no evidence (logLR = 0) but the fading still
applies, so a poor signal never reads as reassuring.

Author: Rahul Jain
Date:   17 Nov 2026
Usage:  from ctg.sequential import simulate_labour, sequential_probabilities
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from . import config as C
from .nb import sigmoid

ZONE_GREEN = "GREEN"
ZONE_RED = "RED"
ZONE_HIGH_RISK = "RED - HIGH RISK"


@dataclass
class SequentialParams:
    """The tuned parameters of the sequential engine and the zone logic."""

    lam: float = 1.0                 # CTG evidence shrinkage
    gamma: float = 1.0               # fading per 15 minutes
    theta: float = 0.5               # red probability threshold
    mu: float = 1.0                  # weight on the clinical prior
    aggregator: str = C.EVIDENCE_AGGREGATOR
    calib_a: float = 1.0             # Platt slope, fitted inside the training folds
    calib_b: float = 0.0             # Platt intercept
    k_high_risk: int = C.K_HIGH_RISK
    n_clear: int = C.N_CLEAR

    def as_dict(self) -> Dict[str, float]:
        """Flatten for logging and json."""
        return {"lambda": self.lam, "gamma": self.gamma, "theta": self.theta,
                "mu": self.mu, "aggregator": self.aggregator,
                "calib_a": self.calib_a, "calib_b": self.calib_b,
                "K": self.k_high_risk, "N_CLEAR": self.n_clear}

    def probability(self, clinical_logit: float, evidence: float) -> float:
        """Combine the prior and the CTG evidence, then apply the Platt scaler."""
        raw = self.mu * clinical_logit + self.lam * evidence
        return sigmoid(self.calib_a * raw + self.calib_b)


def update_evidence(previous: Optional[float], contribution: Optional[float],
                    delta_min: float, params: SequentialParams,
                    history: Optional[List[float]] = None) -> float:
    """Advance the running CTG evidence by one reading.

    `contribution` is None when the window was unusable, in which case the
    evidence only decays - a poor signal must never read as reassuring.

    - `faded_sum`: the original accumulation, weighted by the gap length.
    - `decaying_max`: the worst window seen recently, decayed so it can clear.
      With gamma = 1 this is the plain running maximum.
    - `topk`: the mean of the k largest window log LRs so far.
    """
    decay = params.gamma ** (delta_min / C.WINDOW_MIN)

    if params.aggregator == "faded_sum":
        weight = delta_min / C.WINDOW_MIN
        return decay * (previous or 0.0) + weight * (contribution or 0.0)

    if params.aggregator == "topk":
        if contribution is not None and history is not None:
            history.append(float(contribution))
        if not history:
            return 0.0
        top = sorted(history, reverse=True)[:C.TOP_K_WINDOWS]
        return float(sum(top) / len(top))

    # decaying_max
    decayed = decay * previous if previous is not None else None
    if contribution is None:
        return decayed if decayed is not None else 0.0
    if decayed is None:
        return float(contribution)
    return float(max(decayed, contribution))


def sequential_probabilities(times_min: Sequence[float], log_lrs: Sequence[float],
                             clinical_logit: float, params: SequentialParams,
                             valid: Optional[Sequence[bool]] = None) -> Dict[str, np.ndarray]:
    """Run the sequential update over one labour's readings.

    Returns p_prior (the posterior carried in from the previous reading),
    p_post, the running evidence S, and the log LR actually applied.
    """
    times = np.asarray(times_min, dtype=float)
    log_lrs = np.asarray(log_lrs, dtype=float)
    valid = (np.ones(len(times), dtype=bool) if valid is None
             else np.asarray(valid, dtype=bool))

    n = len(times)
    p_prior = np.zeros(n)
    p_post = np.zeros(n)
    evidence = np.zeros(n)
    applied = np.zeros(n)

    running: Optional[float] = None
    history: List[float] = []
    previous_posterior = params.probability(clinical_logit, 0.0)
    previous_time = times[0] - C.WINDOW_MIN if n else 0.0

    for index in range(n):
        delta = times[index] - previous_time
        if delta <= 0:
            delta = float(C.WINDOW_MIN)

        usable = bool(valid[index]) and bool(np.isfinite(log_lrs[index]))
        contribution = float(log_lrs[index]) if usable else None
        running = update_evidence(running, contribution, delta, params, history)

        p_prior[index] = previous_posterior
        evidence[index] = running
        applied[index] = contribution if contribution is not None else 0.0
        p_post[index] = params.probability(clinical_logit, running)

        previous_posterior = p_post[index]
        previous_time = times[index]

    return {"p_prior": p_prior, "p_post": p_post, "S": evidence, "logLR": applied}


def simulate_labour(record_id: int, layer1_times_s: np.ndarray,
                    layer1_red: np.ndarray, layer1_expedite: np.ndarray,
                    layer1_poor: np.ndarray, window_times_min: np.ndarray,
                    window_log_lr: np.ndarray, window_valid: np.ndarray,
                    clinical_logit: float, params: SequentialParams,
                    stage2_start_s: int = -1, duration_min: float = np.inf,
                    use_layer1: bool = True,
                    rules_active: Optional[Dict[str, np.ndarray]] = None
                    ) -> pd.DataFrame:
    """Walk one labour forward reading by reading and assign a zone to each.

    The reading schedule adapts: every 15 minutes while GREEN in the first
    stage, every 5 minutes once RED or once the second stage starts. Layer 1
    turning RED between readings pulls the next reading forward.

    `use_layer1=False` gives the ZON variants: the same simulator with the rule
    layer switched off, which is what isolates each layer's contribution.
    """
    lookup = {float(t): index for index, t in enumerate(window_times_min)}
    rows: List[Dict[str, object]] = []

    running: Optional[float] = None
    history: List[float] = []
    previous_posterior = params.probability(clinical_logit, 0.0)
    previous_time = float(C.WINDOW_MIN) - C.WINDOW_MIN
    red_streak = 0
    clear_streak = 0
    zone = ZONE_GREEN
    high_risk = False
    reading_time = float(C.WINDOW_MIN)
    first_high_risk_trigger = ""

    while reading_time <= duration_min + 1e-9:
        t_s = int(round(reading_time * 60))

        # ---- Layer 1 state at this reading, and whether it fired since the last one
        l1_red = False
        l1_expedite = False
        l1_between = False
        active_rules: List[str] = []
        if use_layer1 and layer1_times_s.size:
            at = int(np.searchsorted(layer1_times_s, t_s, side="right")) - 1
            if at >= 0:
                l1_red = bool(layer1_red[at])
                l1_expedite = bool(layer1_expedite[at])
                if rules_active:
                    active_rules = [name for name, flags in rules_active.items()
                                    if flags[at]]
            since = (layer1_times_s > previous_time * 60) & (layer1_times_s <= t_s)
            l1_between = bool(layer1_red[since].any()) if since.any() else False

        # ---- Layer 2 update
        index = lookup.get(float(reading_time))
        has_window = index is not None
        window_ok = bool(has_window and window_valid[index])
        delta = max(reading_time - previous_time, 1e-9)

        contribution = None
        if window_ok and np.isfinite(window_log_lr[index]):
            contribution = float(window_log_lr[index])
        running = update_evidence(running, contribution, delta, params, history)
        p_post = params.probability(clinical_logit, running)

        signal_poor = bool(has_window and not window_valid[index]) or not has_window
        if signal_poor:
            p_post = previous_posterior      # carry the zone and probability forward

        # ---- zone at this reading
        l2_red = bool(p_post >= params.theta)
        is_red = bool((l1_red or l1_between) or l2_red)

        if is_red:
            red_streak += 1
            clear_streak = 0
        else:
            clear_streak += 1
            if clear_streak >= params.n_clear:
                red_streak = 0

        became_high_risk = (red_streak >= params.k_high_risk
                            or (l1_expedite and C.EXPEDITE_TRIGGERS_HIGH_RISK))
        if became_high_risk and not high_risk:
            if l1_expedite or l1_red or l1_between:
                first_high_risk_trigger = "L1+L2" if l2_red else "L1"
            else:
                first_high_risk_trigger = "L2"
        high_risk = bool(became_high_risk)

        if high_risk:
            zone = ZONE_HIGH_RISK
        elif is_red:
            zone = ZONE_RED
        elif clear_streak >= params.n_clear:
            zone = ZONE_GREEN
            high_risk = False

        in_second_stage = 0 <= stage2_start_s <= t_s
        rows.append({
            "record_id": record_id, "T_min": reading_time,
            "minutes_before_end": duration_min - reading_time,
            "p_prior": previous_posterior,
            "logLR": contribution if contribution is not None else 0.0,
            "S": running,
            "p_post": p_post, "l1_red": l1_red or l1_between,
            "l1_rules_active": "|".join(active_rules), "l2_red": l2_red,
            "zone": zone, "high_risk": high_risk, "signal_poor": signal_poor,
            "stage": 2 if in_second_stage else 1,
            "red_streak": red_streak})

        previous_posterior = p_post
        previous_time = reading_time

        # ---- next reading: 5 minutes when RED or in second stage, else 15
        if is_red or high_risk or in_second_stage:
            reading_time += C.GRID_MIN
        else:
            reading_time += C.WINDOW_MIN

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame.attrs["first_high_risk_trigger"] = first_high_risk_trigger
    return frame


def summarise_simulation(readings: pd.DataFrame, record_id: int, adverse: float,
                         duration_min: float, sig2birth_min: float = 0.0
                         ) -> Dict[str, object]:
    """Collapse one labour's readings into the summary row the evaluation uses."""
    if readings.empty:
        return {"record_id": record_id, "adverse": adverse, "final_p": np.nan,
                "max_p": np.nan, "ever_red": False, "ever_high_risk": False,
                "final_zone": ZONE_GREEN, "n_readings": 0}

    red = readings["zone"].ne(ZONE_GREEN).to_numpy()
    high = readings["high_risk"].to_numpy(dtype=bool)
    first_red = readings.loc[red, "T_min"].iloc[0] if red.any() else np.nan
    first_high = readings.loc[high, "T_min"].iloc[0] if high.any() else np.nan

    # count distinct RED episodes
    padded = np.concatenate(([False], red, [False]))
    episodes = int((np.diff(padded.astype(np.int8)) == 1).sum())

    trigger_red = ""
    if red.any():
        row = readings.loc[red].iloc[0]
        trigger_red = ("L1+L2" if row["l1_red"] and row["l2_red"]
                       else ("L1" if row["l1_red"] else "L2"))

    return {
        "record_id": record_id, "adverse": adverse,
        "final_p": float(readings["p_post"].iloc[-1]),
        "max_p": float(readings["p_post"].max()),
        "ever_red": bool(red.any()), "ever_high_risk": bool(high.any()),
        "final_zone": str(readings["zone"].iloc[-1]),
        "first_red_min": float(first_red) if np.isfinite(first_red) else np.nan,
        "first_high_risk_min": float(first_high) if np.isfinite(first_high) else np.nan,
        "lead_time_to_end_min": (float(duration_min - first_high)
                                 if np.isfinite(first_high) else np.nan),
        "lead_time_to_birth_min": (float(duration_min - first_high + sig2birth_min)
                                   if np.isfinite(first_high) else np.nan),
        "n_red_episodes": episodes,
        "minutes_red": float(red.sum() * C.GRID_MIN),
        "n_readings": int(len(readings)),
        "pct_signal_poor": float(100 * readings["signal_poor"].mean()),
        "trigger_first_red": trigger_red,
        "trigger_first_high_risk": readings.attrs.get("first_high_risk_trigger", ""),
    }
