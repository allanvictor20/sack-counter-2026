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
                 conf_tracker: ConfidenceTracker,
                 cfg: dict = None):
        self.confirm_frames   = confirm_frames
        self.miss_frames      = miss_frames
        self.ownership        = ownership_mem
        self.conf_tracker     = conf_tracker
        # Retained for confidence_class() at delivery time.  update() also
        # refreshes it, so a cfg passed per-frame still works.
        self._cfg             = cfg or {}

        self._hit_count:   dict[int, int]   = defaultdict(int)
        self._miss_count:  dict[int, int]   = defaultdict(int)
        self._confirmed:   set              = set()
        # bid -> pid for boxes currently held, used to credit a carrier
        # when they are committed at the door.
        self._owner:       dict[int, int]   = {}
        self._delivered:   dict[int, set]   = defaultdict(set)
        self.total_counted: int             = 0
        self.delivery_log:  list            = []

    def update(self, raw_boxes, person_boxes, fn, fps,
               cfg: dict = None,
               sack_owner_scores: dict = None) -> dict:
        if cfg:
            self._cfg = cfg
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
                resolved = self.ownership.update(bid, best_pid, best_score)
                box_owner[bid] = resolved
                self._owner[bid] = resolved
                self.conf_tracker.record_detection(bid, sc)
                self.conf_tracker.record_ownership(bid, best_score)

        # Drop ownership for boxes whose track has been evicted, so a
        # carrier is not later credited with a box that no longer exists.
        for bid in list(self._owner):
            if bid not in self._confirmed:
                self._owner.pop(bid, None)

        return box_owner

    def commit_delivery(self, pid: int, fn: int) -> int:
        """
        Credit every box currently owned by *pid* as delivered.

        v21 removed box gate-crossing and never replaced it, so
        ``_delivered`` was never written: ``delivery_counts()`` always
        returned an empty dict and ``total_counted`` was permanently 0,
        even though the report printed Boxes columns and the JSON log wrote
        a ``total_boxes`` field.  Boxes now settle on the same trigger as
        sacks — the door-crossing commit for the carrier holding them.

        Args:
            pid: Carrier person ID being committed at the door.
            fn:  Current frame number.

        Returns:
            Number of boxes newly credited to *pid*.
        """
        held = [bid for bid, owner in self._owner.items() if owner == pid]
        new  = [bid for bid in held if bid not in self._delivered[pid]]
        if not new:
            return 0

        for bid in new:
            self._delivered[pid].add(bid)
            self._owner.pop(bid, None)
        self.total_counted += len(new)

        for bid in new:
            conf = self.conf_tracker.delivery_confidence(bid)
            self.delivery_log.append({
                "frame":            fn,
                "box_id":           bid,
                "person_id":        pid,
                "confidence":       round(conf, 3),
                "confidence_class": confidence_class(conf, self._cfg),
                "high_conf":        conf >= self.conf_tracker.min,
                "type":             "box",
                "trigger":          "door_crossing",
            })
        return len(new)

    def delivery_counts(self) -> dict:
        return {pid: len(bids) for pid, bids in self._delivered.items()}
