"""
trackers.py — Stateful tracker classes for Sack Counter v20.

Classes:
    SackMotionTracker   — detects whether a sack is stationary
    IdentityRelinker    — deep ReID re-linking for person tracks
    OwnershipMemory     — hysteresis / flicker-guard for sack ownership
    GhostSacks          — extrapolates sack positions during brief occlusions
"""

import numpy as np
import logging
from collections import defaultdict, deque

from .embedder import DeepEmbedder


# ─────────────────────────────────────────────────────────
# SackMotionTracker
# ─────────────────────────────────────────────────────────

class SackMotionTracker:
    """
    Classifies a sack as stationary from its mean per-frame displacement.

    Stillness is decided purely by ``speed_thresh`` over ``speed_window``
    samples.  The constructor used to also accept ``window`` and
    ``thresh_px`` and silently discard both, which made the
    ``still_window`` and ``still_thresh_ratio`` config keys look like live
    tuning knobs when they did nothing at all; they have been removed from
    the config rather than left as decoration.
    """

    def __init__(self, speed_thresh: float, speed_window: int):
        self.speed_thresh = speed_thresh
        self.speed_window = speed_window
        self.history: dict[int, deque] = defaultdict(
            lambda: deque(maxlen=speed_window + 1))

    def update(self, sack_id: int, cx: int, cy: int):
        self.history[sack_id].append((cx, cy))

    def is_still(self, sack_id: int) -> bool:
        h = self.history.get(sack_id)
        if not h or len(h) < 3:
            return False
        pts    = list(h)
        speeds = [np.hypot(pts[i][0]-pts[i-1][0], pts[i][1]-pts[i-1][1])
                  for i in range(1, len(pts))]
        return float(np.mean(speeds)) < self.speed_thresh

    def cleanup(self, active_ids: set):
        for sid in list(self.history):
            if sid not in active_ids:
                del self.history[sid]


# ─────────────────────────────────────────────────────────
# IdentityRelinker
# Uses rebalanced weights: appearance=0.50 (dominant)
# ─────────────────────────────────────────────────────────

class IdentityRelinker:
    def __init__(self, cfg: dict, embedder: DeepEmbedder):
        self.embedder            = embedder
        self.reid_memory_frames  = 0
        self.threshold           = cfg["reid_threshold"]
        self.w_pos               = cfg["reid_w_position"]    # 0.25
        self.w_vel               = cfg["reid_w_velocity"]    # 0.15
        self.w_bbox              = cfg["reid_w_bbox_size"]   # 0.10
        self.w_app               = cfg["reid_w_appearance"]  # 0.50 ← dominant
        self.max_dist            = cfg["max_assoc_dist"]
        self.lost_registry:  dict = {}
        self.id_map:         dict = {}
        self.reid_events:    list = []

    def set_memory_frames(self, frames: int):
        self.reid_memory_frames = frames

    def canonical(self, pid: int) -> int:
        visited = set()
        while pid in self.id_map and pid not in visited:
            visited.add(pid)
            pid = self.id_map[pid]
        return pid

    def register_lost(self, pid: int, cx: int, cy: int,
                      vx: float, vy: float, bbox_area: float,
                      embedding: np.ndarray, frame_no: int):
        can = self.canonical(pid)
        self.lost_registry[can] = {
            "last_cx":   cx,
            "last_cy":   cy,
            "vx":        vx,
            "vy":        vy,
            "bbox_area": bbox_area,
            "embedding": embedding,
            "frame":     frame_no,
        }

    def try_relink(self, new_pid: int,
                   cx: int, cy: int,
                   vx: float, vy: float,
                   bbox_area: float,
                   embedding: np.ndarray,
                   frame_no: int,
                   logger) -> int:
        if new_pid in self.id_map:
            return self.canonical(new_pid)

        best_score = 0.0
        best_old   = None

        for old_pid, rec in self.lost_registry.items():
            age = frame_no - rec["frame"]
            if age > self.reid_memory_frames:
                continue
            pred_cx = rec["last_cx"] + rec["vx"] * age
            pred_cy = rec["last_cy"] + rec["vy"] * age
            dist    = np.hypot(cx - pred_cx, cy - pred_cy)
            if dist > self.max_dist * 1.5:
                continue
            pos_score = max(0.0, 1.0 - dist / (self.max_dist * 1.5))

            old_speed  = np.hypot(rec["vx"], rec["vy"]) + 1e-6
            new_speed  = np.hypot(vx, vy) + 1e-6
            speed_ratio = min(old_speed, new_speed) / max(old_speed, new_speed)
            if old_speed > 0.5 and new_speed > 0.5:
                cos_a = ((rec["vx"]*vx + rec["vy"]*vy)
                         / (old_speed * new_speed))
                vel_score = 0.5 * speed_ratio + 0.5 * max(0.0, cos_a)
            else:
                vel_score = speed_ratio

            old_area   = rec["bbox_area"] + 1
            new_area   = bbox_area + 1
            bbox_score = min(old_area, new_area) / max(old_area, new_area)

            app_score = DeepEmbedder.cosine(embedding, rec["embedding"])

            # Appearance weight 0.50 — dominant
            score = (self.w_pos  * pos_score
                   + self.w_vel  * vel_score
                   + self.w_bbox * bbox_score
                   + self.w_app  * app_score)

            if score > best_score:
                best_score = score
                best_old   = old_pid

        if best_old is not None and best_score >= self.threshold:
            self.id_map[new_pid] = best_old
            del self.lost_registry[best_old]
            msg = (f"Re-ID: P#{new_pid} → P#{best_old}  "
                   f"score={best_score:.3f}  frame={frame_no}")
            logger.info(msg)
            print(f"  [RE-ID] {msg}")
            self.reid_events.append({
                "frame":  frame_no,
                "new_id": new_pid,
                "old_id": best_old,
                "score":  round(best_score, 4),
            })
            return best_old

        return new_pid

    def expire_lost(self, frame_no: int):
        for old_pid in list(self.lost_registry):
            if frame_no - self.lost_registry[old_pid]["frame"] > self.reid_memory_frames:
                del self.lost_registry[old_pid]


# ─────────────────────────────────────────────────────────
# OwnershipMemory
# ─────────────────────────────────────────────────────────

class OwnershipMemory:
    """
    Stores (owner_pid, score) per sack.
    Switch only when new_score > current_score + margin.
    """
    def __init__(self, switch_margin: float):
        self.margin = switch_margin
        self._owner: dict[int, tuple[int, float]] = {}

    def update(self, sid: int, new_pid: int, new_score: float) -> int:
        if sid not in self._owner:
            self._owner[sid] = (new_pid, new_score)
            return new_pid
        cur_pid, cur_score = self._owner[sid]
        if new_pid == cur_pid:
            self._owner[sid] = (new_pid, new_score)
            return new_pid
        if new_score >= cur_score + self.margin:
            self._owner[sid] = (new_pid, new_score)
            return new_pid
        return cur_pid

    def get(self, sid: int):
        rec = self._owner.get(sid)
        return rec[0] if rec else None

    def get_score(self, sid: int) -> float:
        """Expose current ownership confidence score."""
        rec = self._owner.get(sid)
        return rec[1] if rec else 0.0

    def iter_by_owner(self):
        """
        Iterate over (sack_id, owner_pid) pairs.

        Replaces direct access to the private ``_owner`` dict from outside
        this class.  Returns a snapshot list so callers can safely mutate
        the registry during iteration.
        """
        return [(sid, rec[0]) for sid, rec in list(self._owner.items())]

    def clear(self, sid: int):
        self._owner.pop(sid, None)

    def cleanup(self, active_sids: set):
        for sid in list(self._owner):
            if sid not in active_sids:
                del self._owner[sid]


# ─────────────────────────────────────────────────────────
# GhostSacks
# ─────────────────────────────────────────────────────────

class GhostSacks:
    """
    Extrapolates a sack's position for a short window after it is lost.

    A ghost expires after ``max_frames`` and must NOT come back.  It
    previously did: ``update()`` recreated a ghost for any sid present in
    ``sack_positions`` but absent from ``active_sids``, and that dict was
    not pruned, so every sack ID ever seen got a fresh ghost the frame
    after its previous one aged out — forever.  ``_spent`` records the IDs
    whose occlusion window has already been used up.
    """

    def __init__(self, max_frames: int):
        self.max_frames = max_frames
        self.ghosts:    dict = {}
        self._prev_pos: dict = {}
        self._velocity: dict = {}
        self._spent:    set  = set()

    def record_position(self, sid: int, cx: int, cy: int):
        if sid in self._prev_pos:
            px, py = self._prev_pos[sid]
            self._velocity[sid] = (float(cx - px), float(cy - py))
        self._prev_pos[sid] = (cx, cy)

    def update(self, active_sids: set, sack_owner: dict,
               sack_positions: dict, sack_boxes: dict):
        # A re-detected sack gets a fresh occlusion budget.
        self._spent -= active_sids

        for sid in list(self.ghosts):
            if sid in active_sids:
                del self.ghosts[sid]
            else:
                g = self.ghosts[sid]
                g["frames"] += 1
                g["cx"] = int(g["cx"] + g["vx"])
                g["cy"] = int(g["cy"] + g["vy"])
                g["x1"] = int(g["x1"] + g["vx"])
                g["x2"] = int(g["x2"] + g["vx"])
                g["y1"] = int(g["y1"] + g["vy"])
                g["y2"] = int(g["y2"] + g["vy"])
                if g["frames"] > self.max_frames:
                    del self.ghosts[sid]
                    self._spent.add(sid)

        for sid, (cx, cy) in sack_positions.items():
            if (sid not in active_sids
                    and sid not in self.ghosts
                    and sid not in self._spent):
                vx, vy = self._velocity.get(sid, (0.0, 0.0))
                x1, y1, x2, y2 = sack_boxes.get(sid, (cx, cy, cx, cy))
                self.ghosts[sid] = {
                    "frames": 1,
                    "cx": cx, "cy": cy,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "vx": vx, "vy": vy,
                    "owner": sack_owner.get(sid),
                }

    def cleanup(self, known_sids: set):
        """Drop position/velocity history for IDs no longer tracked at all."""
        for sid in list(self._prev_pos):
            if sid not in known_sids:
                self._prev_pos.pop(sid, None)
                self._velocity.pop(sid, None)
                self._spent.discard(sid)

    def iter_ghosts(self):
        yield from self.ghosts.items()
