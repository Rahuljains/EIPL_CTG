"""CTGRiskEngine: the live API tomorrow's interface calls.

Holds the fitted bundle and walks a labour forward in real time: signal is
pushed in as it arrives, Layer 1 re-evaluates every 5 seconds, and Layer 2
updates on the reading schedule. `state()` returns everything a screen needs,
including why the probability is where it is.

No plotting and no UI code belongs in this module.

Author: Rahul Jain
Date:   17 Nov 2026
Usage:  from ctg.engine import CTGRiskEngine
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Generator, List, Optional, Sequence

import numpy as np

from . import config as C
from .ctg_nb import CtgNB
from .detect import run_detectors
from .features import FEATURE_NAMES, window_features
from .nb import SkippingCategoricalNB, sigmoid
from .preprocess import clean_fhr, clean_uc, interpolate_short_gaps, to_1hz
from .rules import NOTE_NAMES, RULE_NAMES, evaluate_rules
from .sequential import (SequentialParams, ZONE_GREEN, ZONE_HIGH_RISK, ZONE_RED,
                         update_evidence)


class CTGRiskEngine:
    """Live two-layer risk engine, loaded from an exported bundle directory."""

    def __init__(self, bundle_dir: Path) -> None:
        self.bundle_dir = Path(bundle_dir)
        if not self.bundle_dir.exists():
            raise FileNotFoundError(f"Bundle directory not found: {self.bundle_dir}")

        self.params_payload = json.loads(
            (self.bundle_dir / "params.json").read_text(encoding="utf-8"))
        self.clinical_nb = SkippingCategoricalNB.from_json(
            self.bundle_dir / "clinical_nb.json")
        self.ctg_nb = CtgNB.from_json(self.bundle_dir / "ctg_nb.json")

        self.track = self.params_payload.get("track", "F")
        self.params = SequentialParams(
            lam=float(self.params_payload["lambda"]),
            gamma=float(self.params_payload["gamma"]),
            theta=float(self.params_payload["theta"]),
            mu=float(self.params_payload.get("mu", 1.0)),
            aggregator=str(self.params_payload.get("aggregator",
                                                   C.EVIDENCE_AGGREGATOR)),
            calib_a=float(self.params_payload.get("calib_a", 1.0)),
            calib_b=float(self.params_payload.get("calib_b", 0.0)),
            k_high_risk=int(self.params_payload.get("K", C.K_HIGH_RISK)),
            n_clear=int(self.params_payload.get("N_CLEAR", C.N_CLEAR)))

        self._encoder = None
        if self.track == "E":
            if not (self.bundle_dir / "encoder.pt").exists():
                raise FileNotFoundError(
                    f"{self.bundle_dir} declares track E but has no encoder.pt. "
                    "Without it the engine would compute Track F features against a "
                    "Track E model, so the bundle is refused rather than used.")
            self._load_encoder()

        self.reset()

    # ------------------------------------------------------------------ setup
    def _load_encoder(self) -> None:
        """Load the Track E encoder lazily; Track F never needs torch."""
        import torch
        from .encoder import MaskedCTGAutoencoder

        self._torch = torch
        self._encoder = MaskedCTGAutoencoder()
        payload = torch.load(self.bundle_dir / "encoder.pt", map_location="cpu",
                             weights_only=False)
        self._encoder.load_state_dict(payload["state_dict"])
        self._encoder.eval()

    def reset(self) -> None:
        """Clear all state so the engine can replay another labour."""
        self.fhr_4hz = np.empty(0, dtype=float)
        self.uc_4hz = np.empty(0, dtype=float)
        self.clinical_items: Dict[str, object] = {}
        self.not_in_model: List[str] = []
        self.clinical_timeline: List[Dict[str, object]] = []
        self.readings: List[Dict[str, object]] = []
        self.event_log: List[Dict[str, object]] = []

        self._evidence = None
        self._evidence_history = []
        self._red_streak = 0
        self._clear_streak = 0
        self._zone = ZONE_GREEN
        self._high_risk = False
        self._next_reading_min = float(C.WINDOW_MIN)
        self._last_reading_min = 0.0
        self._stage2_start_s = -1
        self._detectors = None
        self._states = None

    # --------------------------------------------------------------- clinical
    def set_clinical(self, items: Dict[str, object]) -> Dict[str, List[str]]:
        """Set the history at time 0. Unknown fields are accepted and listed back."""
        known = set(self.clinical_nb.items)
        self.clinical_items = {k: v for k, v in items.items() if k in known}
        self.not_in_model = sorted(set(items) - known)
        self.clinical_timeline = [{"t_min": 0.0, "item": k, "value": v}
                                  for k, v in self.clinical_items.items()]
        return {"in_model": sorted(self.clinical_items),
                "not_in_model": self.not_in_model}

    def add_clinical(self, item: str, value: object, t_min: float) -> bool:
        """Add or update one history item part-way through labour."""
        if item not in set(self.clinical_nb.items):
            self.not_in_model.append(item)
            return False
        self.clinical_items[item] = value
        self.clinical_timeline.append({"t_min": float(t_min), "item": item,
                                       "value": value})
        self.event_log.append({"t_min": float(t_min), "kind": "clinical",
                               "detail": f"{item} = {value}"})
        return True

    def set_second_stage(self, t_min: float) -> None:
        """Mark the start of the second stage, which tightens the schedule."""
        self._stage2_start_s = int(round(t_min * 60))
        self.event_log.append({"t_min": float(t_min), "kind": "stage",
                               "detail": "second stage started"})

    @property
    def clinical_logit(self) -> float:
        """Current clinical prior log odds from the items known so far."""
        return self.clinical_nb.prior_logit(self.clinical_items)

    # ----------------------------------------------------------------- signal
    def push_signal(self, fhr_4hz: Sequence[float],
                    uc_4hz: Optional[Sequence[float]] = None) -> Dict[str, object]:
        """Append new 4 Hz samples and bring every layer up to date."""
        new_fhr = np.asarray(fhr_4hz, dtype=float)
        new_uc = (np.asarray(uc_4hz, dtype=float) if uc_4hz is not None
                  else np.full(new_fhr.shape, np.nan))
        self.fhr_4hz = np.concatenate([self.fhr_4hz, new_fhr])
        self.uc_4hz = np.concatenate([self.uc_4hz, new_uc])
        self._recompute()
        return self.state()

    def _recompute(self) -> None:
        """Re-run the detectors, the rules and any readings that are now due."""
        fhr_clean, _ = clean_fhr(self.fhr_4hz)
        uc_clean, _ = clean_uc(self.uc_4hz)
        fhr_1hz_raw, valid_1hz = to_1hz(fhr_clean)
        uc_1hz, _ = to_1hz(uc_clean, min_valid=1)
        if fhr_1hz_raw.size == 0:
            return

        self._fhr_1hz_raw = fhr_1hz_raw
        self._fhr_1hz = interpolate_short_gaps(fhr_1hz_raw)
        self._uc_1hz = uc_1hz
        self._valid_1hz = valid_1hz

        self._detectors = run_detectors(fhr_1hz_raw, valid_1hz, fhr_clean, uc_1hz)
        self._states = evaluate_rules(self._detectors, valid_1hz,
                                      stage2_start_s=self._stage2_start_s)

        now_min = len(fhr_1hz_raw) / 60.0
        while self._next_reading_min <= now_min + 1e-9:
            self._take_reading(self._next_reading_min)

    def _layer1_at(self, t_min: float) -> Dict[str, object]:
        """Layer 1 state at a time, plus whether it fired since the last reading."""
        if self._states is None or self._states.empty:
            return {"l1_red": False, "l1_expedite": False, "rules": []}
        t_s = int(round(t_min * 60))
        times = self._states["t_s"].to_numpy()
        index = int(np.searchsorted(times, t_s, side="right")) - 1
        if index < 0:
            return {"l1_red": False, "l1_expedite": False, "rules": []}

        row = self._states.iloc[index]
        since = (times > self._last_reading_min * 60) & (times <= t_s)
        fired_between = bool(self._states.loc[since, "l1_red"].any()) if since.any() \
            else False
        return {"l1_red": bool(row["l1_red"]) or fired_between,
                "l1_expedite": bool(row["l1_expedite"]),
                "rules": [r for r in RULE_NAMES if bool(row[r])],
                "notes": [n for n in NOTE_NAMES if bool(row[n])],
                "signal_poor": bool(row["signal_poor"]),
                "baseline": float(row["baseline"])}

    def _window_log_lr(self, t_min: float) -> Dict[str, object]:
        """Compute this window's CTG log likelihood ratio, or mark it unusable."""
        end = int(round(t_min * 60))
        start = end - C.WINDOW_MIN * 60
        if start < 0 or end > len(self._fhr_1hz):
            return {"log_lr": 0.0, "valid": False, "valid_frac": 0.0}

        valid_frac = float(np.mean(self._valid_1hz[start:end]))
        if valid_frac < C.MIN_VALID_FRAC_WINDOW:
            return {"log_lr": 0.0, "valid": False, "valid_frac": valid_frac}

        if self.track == "E" and self._encoder is not None:
            from .encoder import build_ctu_inputs
            channels = build_ctu_inputs(self._fhr_1hz[start:end],
                                        self._uc_1hz[start:end],
                                        self._valid_1hz[start:end])
            with self._torch.no_grad():
                vector = self._encoder.embed(
                    self._torch.from_numpy(channels[None, ...])).numpy()
            features = vector
        else:
            values = window_features(self._detectors, self._fhr_1hz_raw,
                                     self._valid_1hz, t_min)
            features = np.array([[values[name] for name in FEATURE_NAMES]])

        return {"log_lr": float(self.ctg_nb.log_lr(features)[0]), "valid": True,
                "valid_frac": valid_frac}

    def _take_reading(self, t_min: float) -> Dict[str, object]:
        """Run one Layer 2 reading and update the zone."""
        layer1 = self._layer1_at(t_min)
        window = self._window_log_lr(t_min)

        delta = max(t_min - self._last_reading_min, 1e-9)
        previous = (self.readings[-1]["p_post"] if self.readings
                    else self.params.probability(self.clinical_logit, 0.0))

        contribution = window["log_lr"] if window["valid"] else None
        self._evidence = update_evidence(self._evidence, contribution, delta,
                                         self.params, self._evidence_history)
        p_post = self.params.probability(self.clinical_logit, self._evidence)
        if not window["valid"]:
            p_post = previous

        l2_red = bool(p_post >= self.params.theta)
        is_red = bool(layer1["l1_red"] or l2_red)

        if is_red:
            self._red_streak += 1
            self._clear_streak = 0
        else:
            self._clear_streak += 1
            if self._clear_streak >= self.params.n_clear:
                self._red_streak = 0

        self._high_risk = bool(self._red_streak >= self.params.k_high_risk
                               or (layer1["l1_expedite"]
                                   and C.EXPEDITE_TRIGGERS_HIGH_RISK))
        if self._high_risk:
            self._zone = ZONE_HIGH_RISK
        elif is_red:
            self._zone = ZONE_RED
        elif self._clear_streak >= self.params.n_clear:
            self._zone = ZONE_GREEN

        reading = {"T_min": float(t_min), "p_prior": float(previous),
                   "p_post": float(p_post),
                   "logLR": float(contribution) if contribution is not None else 0.0,
                   "S": float(self._evidence or 0.0), "zone": self._zone,
                   "high_risk": self._high_risk, "l1_red": layer1["l1_red"],
                   "l2_red": l2_red, "signal_poor": not window["valid"],
                   "l1_rules_active": layer1.get("rules", []),
                   "valid_frac": window["valid_frac"]}
        self.readings.append(reading)
        self.event_log.append({"t_min": float(t_min), "kind": "reading",
                               "detail": f"{self._zone} (p = {p_post:.3f})"})

        self._last_reading_min = t_min
        in_second_stage = 0 <= self._stage2_start_s <= t_min * 60
        step = (C.GRID_MIN if (is_red or self._high_risk or in_second_stage)
                else C.WINDOW_MIN)
        self._next_reading_min = t_min + step
        return reading

    # ------------------------------------------------------------------ state
    def state(self) -> Dict[str, object]:
        """Everything the interface needs to draw the current screen."""
        last = self.readings[-1] if self.readings else None
        contributions = self.clinical_nb.item_contributions(self.clinical_items)
        layer1 = self._layer1_at(len(getattr(self, "_fhr_1hz_raw", [])) / 60.0) \
            if self._states is not None else {"rules": [], "notes": [],
                                              "signal_poor": False}
        return {
            "zone": self._zone,
            "high_risk": self._high_risk,
            "signal_poor": bool(layer1.get("signal_poor", False)),
            "p_current": (float(last["p_post"]) if last
                          else self.params.probability(self.clinical_logit, 0.0)),
            "p_prior": (float(last["p_prior"]) if last
                        else self.params.probability(self.clinical_logit, 0.0)),
            "trend": [round(float(r["p_post"]), 4) for r in self.readings[-4:]],
            "next_reading_min": float(self._next_reading_min),
            "l1_rules_active": layer1.get("rules", []),
            "l1_notes": layer1.get("notes", []),
            "contributions": {"clinical": {k: round(float(v), 4)
                                           for k, v in contributions.items()},
                              "clinical_total": round(float(sum(contributions.values())), 4),
                              "base_log_odds": round(float(self.clinical_nb.base_log_odds), 4),
                              "ctg_total": round(float(self.params.lam
                                                        * (self._evidence or 0.0)), 4),
                              "aggregator": self.params.aggregator},
            "not_in_model": self.not_in_model,
            "n_readings": len(self.readings),
            "event_log": self.event_log[-20:],
            "params": self.params.as_dict(),
            "track": self.track}

    # ----------------------------------------------------------------- replay
    def replay(self, fhr_4hz: Sequence[float], uc_4hz: Sequence[float],
               clinical: Dict[str, object], stage2_start_s: int = -1,
               chunk_s: int = 60) -> Generator[Dict[str, object], None, None]:
        """Replay a recorded labour, yielding the state at each reading. For demos."""
        self.reset()
        self.set_clinical(clinical)
        if stage2_start_s >= 0:
            self.set_second_stage(stage2_start_s / 60.0)

        fhr = np.asarray(fhr_4hz, dtype=float)
        uc = np.asarray(uc_4hz, dtype=float)
        step = chunk_s * C.FS_RAW

        for start in range(0, len(fhr), step):
            before = len(self.readings)
            self.push_signal(fhr[start:start + step], uc[start:start + step])
            if len(self.readings) > before:
                yield self.state()

    def timeline(self) -> List[Dict[str, object]]:
        """Every reading taken so far, for writing a demo timeline csv."""
        return list(self.readings)
