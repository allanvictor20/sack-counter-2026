"""
box_pipeline.py — Box (class-2) counting pipeline for Sack Counter v13.

BoxCountingPipeline tracks cardboard boxes through confirmation,
ownership assignment, and gate crossing — mirroring the sack pipeline
but using a simpler (no-appearance) association score.
"""

from collections import defaultdict

from .assignment import association_score
from .trackers import OwnershipMemory
from .confidence import ConfidenceTracker, confidence_class


class BoxCountingPipeline:
    def __init__(self, confirm_frames: int, miss_frames: int,
                 ownership_mem: OwnershipMemory,
                 conf_tracker: ConfidenceTracker):
        self.confirm_frames   = confirm_frames
        self.miss_frames      = miss_frames
        self.ownership        = ownership_mem
        self.conf_tracker     = conf_tracker

        self._hit_count:   dict[int, int]   = defaultdict(int)
        self._miss_count:  dict[int, int]   = defaultdict(int)
        self._confirmed:   set              = set()
        self._prev_cx:     dict[int, int]   = {}
        self._owner:       dict[int, int]   = {}
        self._delivered:   dict[int, set]   = defaultdict(set)
        self.total_counted: int             = 0
        self.delivery_log:  list            = []

    def update(self, raw_boxes, person_boxes, fn, fps,
               cfg: dict = None,
               sack_owner_scores: dict = None) -> dict:
        active_bids = {b[0] for b in raw_boxes}

        for (bid, *_) in raw_boxes:
            self._hit_count[bid]  += 1
            self._miss_count[bid]  = 0
            if self._hit_count[bid] >= self.confirm_frames:
                self._confirmed.add(bid)

        for bid in list(self._confirmed):
            if bid not in active_bids:
                self._miss_count[bid] += 1
                if self._miss_count[bid] > self.miss_frames:
                    self._confirmed.discard(bid)
                    self._hit_count.pop(bid, None)
                    self._miss_count.pop(bid, None)

        box_owner = {}
        for (bid, bx1, by1, bx2, by2, sc) in raw_boxes:
            if bid not in self._confirmed:
                continue
            bcx = (bx1 + bx2) // 2
            bcy = (by1 + by2) // 2
            best_pid   = None
            best_score = 0.0
            for pid, pbox in person_boxes.items():
                s = association_score(bx1, by1, bx2, by2, bcx, bcy, pbox,
                                      {"max_assoc_dist":  350,
                                       "carry_zone_top": -2.0,
                                       "carry_zone_bot":  0.6,
                                       "w_distance":      0.35,
                                       "w_overlap":       0.30,
                                       "w_height":        0.35,
                                       "w_appearance":    0.0})
                if s > best_score:
                    best_score = s
                    best_pid   = pid
            if best_pid is not None and best_score > 0.05:
                box_owner[bid] = self.ownership.update(bid, best_pid, best_score)
                self.conf_tracker.record_detection(bid, sc)
                self.conf_tracker.record_ownership(bid, best_score)

        # v21: Box gate crossing removed. Boxes are tracked for ownership
        # display purposes; delivery counting via door-zone follows sack logic.

        return box_owner

    def delivery_counts(self) -> dict:
        return {pid: len(bids) for pid, bids in self._delivered.items()}
