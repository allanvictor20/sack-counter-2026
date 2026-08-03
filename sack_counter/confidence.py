"""
confidence.py — Confidence scoring for Sack Counter v13.

Exports:
    confidence_class(score, cfg) -> str   ("HIGH" / "MEDIUM" / "LOW")
    ConfidenceTracker
"""

from collections import defaultdict


def confidence_class(score: float, cfg: dict) -> str:
    """Return HIGH / MEDIUM / LOW label for an ownership/delivery confidence score."""
    if score >= cfg["conf_class_high"]:
        return "HIGH"
    if score >= cfg["conf_class_medium"]:
        return "MEDIUM"
    return "LOW"


class ConfidenceTracker:
    def __init__(self, cfg: dict):
        self.w_det  = cfg["conf_weight_detection"]
        self.w_trk  = cfg["conf_weight_tracking"]
        self.w_own  = cfg["conf_weight_ownership"]
        self.w_crs  = cfg["conf_weight_crossing"]
        self.min    = cfg["min_delivery_confidence"]

        self._det:  dict[int, float] = {}
        self._hits: dict[int, int]   = defaultdict(int)
        self._own:  dict[int, float] = {}
        self._crs:  dict[int, float] = {}

    def record_detection(self, sid: int, det_conf: float):
        self._det[sid] = det_conf
        self._hits[sid] += 1

    def record_ownership(self, sid: int, score: float):
        self._own[sid] = score

    def record_approach(self, sid: int):
        """Sack has reached the door approach zone — half crossing credit."""
        self._crs[sid] = 0.5

    def record_crossing(self, sid: int):
        """
        Sack has crossed the door — full crossing credit.

        Nothing used to call this (nor ``record_approach``), so the
        crossing term — 15% of the score by default — was permanently 0
        and every delivery_confidence was capped at 0.85.  The door-crossing
        commit path now calls it.
        """
        self._crs[sid] = 1.0

    # Gate-era names, kept so existing call-sites and tests keep working.
    record_gate_a = record_approach
    record_gate_b = record_crossing

    def delivery_confidence(self, sid: int) -> float:
        det  = self._det.get(sid, 0.5)
        hits = min(self._hits.get(sid, 1), 30)
        trk  = hits / 30.0
        own  = self._own.get(sid, 0.5)
        crs  = self._crs.get(sid, 0.0)
        return (self.w_det * det
              + self.w_trk * trk
              + self.w_own * own
              + self.w_crs * crs)

    def is_high_confidence(self, sid: int) -> bool:
        return self.delivery_confidence(sid) >= self.min

    def remove(self, sid: int) -> None:
        """
        Remove all tracking data for a single sack ID.

        Use this instead of ``cleanup({sid})`` when expiring a single sack —
        ``cleanup`` expects the set of sacks to *keep*, so passing a
        single-element set of the expiring sack is semantically backwards
        (BUG #5 fix).

        Args:
            sid: Sack tracker ID to remove.
        """
        self._det.pop(sid, None)
        self._hits.pop(sid, None)
        self._own.pop(sid, None)
        self._crs.pop(sid, None)

    def cleanup(self, active_sids: set):
        """
        Prune data for all sacks not in *active_sids*.

        Args:
            active_sids: Set of sack IDs that are still active (to KEEP).
                         All other entries are removed.
        """
        for sid in list(self._det):
            if sid not in active_sids:
                del self._det[sid]
        for sid in list(self._hits):
            if sid not in active_sids:
                del self._hits[sid]
        for sid in list(self._own):
            if sid not in active_sids:
                del self._own[sid]
        for sid in list(self._crs):
            if sid not in active_sids:
                del self._crs[sid]
