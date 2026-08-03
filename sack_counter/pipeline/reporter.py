"""
reporter.py — End-of-session report printing for Sack Counter v20.

Functions
---------
print_report : Pretty-print the full delivery summary to stdout.
"""

from __future__ import annotations

import time

from ..version import VERSION_TAG as _VERSION_TAG
from ..box_pipeline import BoxCountingPipeline
from ..trackers import IdentityRelinker


def _hr(char: str = "─", width: int = 62) -> str:
    return char * width


def _int_keyed(workers: dict) -> dict:
    """
    Return *workers* re-keyed by int person ID.

    Accepts both a live ``AnalyticsEngine.summary()`` (int keys) and one
    that has been through a JSON round-trip (str keys), so the report reads
    the same either way.  Keys that are not integer-like are dropped.
    """
    out = {}
    for key, value in workers.items():
        try:
            out[int(key)] = value
        except (TypeError, ValueError):
            continue
    return out


def print_report(
    delivery_log: list[dict],
    box_pipeline: BoxCountingPipeline,
    person_peak_count: dict[int, int],
    person_sack_delivered: dict[int, set],
    total_sacks_counted: int,
    relinker: IdentityRelinker,
    analytics_summary: dict,
) -> None:
    """
    Pretty-print the final delivery report to stdout.

    Sections printed:
    - Per-person delivery summary (sacks, boxes, avg load, avg confidence).
    - Throughput (items/min, peak/min, elapsed, minute trend).
    - Confidence breakdown (HIGH / MEDIUM / LOW counts).
    - Grand totals.

    Args:
        delivery_log:          List of sack delivery event dicts.
        box_pipeline:          BoxCountingPipeline instance.
        person_peak_count:     Committed peak count per person ID.
        person_sack_delivered: Set of delivered sack IDs per person.
        total_sacks_counted:   Grand total sacks counted.
        relinker:              IdentityRelinker (unused directly; reserved).
        analytics_summary:     Dict from :meth:`AnalyticsEngine.summary`.
    """
    box_del_counts = box_pipeline.delivery_counts()
    # Use analytics workers as the authoritative source — person_peak_delivery
    # may be missing workers whose deliveries were committed via the orphan path
    # or whose peak was tracked in person_peak_count but not flushed into
    # person_peak_delivery (e.g. P#4, P#457 were absent from the summary table).
    #
    # AnalyticsEngine.summary() keys workers by int pid, but this report was
    # written against a JSON round-trip of that dict and looked every worker
    # up as str(pid).  Every lookup missed: the fallback never fired and Avg
    # Load / Avg Conf printed 0.00 for everyone.  Normalise to int once here
    # so the report works on both a live summary and a reloaded JSON one.
    analytics_workers = _int_keyed(analytics_summary.get("workers", {}))
    carriers_from_peak = set(pid for pid, pk in person_peak_count.items() if pk > 0)
    carriers_from_analytics = set(pid for pid, w in analytics_workers.items()
                                  if w.get("total_sacks", 0) > 0)
    carriers     = sorted(carriers_from_peak | carriers_from_analytics)
    box_carriers = sorted(pid for pid, cnt in box_del_counts.items() if cnt > 0)
    all_carriers = sorted(set(carriers) | set(box_carriers))
    now_str      = time.strftime("%Y-%m-%d  %H:%M:%S")

    print()
    print(_hr("═"))
    print(f"  SACK / BOX DELIVERY REPORT  —  Sack Counter {_VERSION_TAG}")
    print(f"  Generated : {now_str}")
    print(_hr("═"))

    # ── Per-person table ──────────────────────────────────────
    if all_carriers:
        print()
        print("  PER-PERSON DELIVERY SUMMARY")
        print("  " + _hr())
        print(
            f"  {'Person':<10} {'Sacks (peak)':<16} {'Boxes':<10} "
            f"{'Total':<8} {'Avg Load':<10} {'Avg Conf':<10}"
        )
        print("  " + _hr())
        grand_sacks = grand_boxes = 0
        for pid in all_carriers:
            peak_sacks  = person_peak_count.get(pid, 0)
            # Fall back to analytics for workers missing from person_peak_delivery
            if peak_sacks == 0:
                peak_sacks = analytics_workers.get(pid, {}).get("total_sacks", 0)
            boxes       = box_del_counts.get(pid, 0)
            grand_sacks += peak_sacks
            grand_boxes += boxes
            gate_sacks  = len(person_sack_delivered.get(pid, set()))
            flag        = (
                f"  <- peak rescued +{peak_sacks - gate_sacks}"
                if peak_sacks > gate_sacks else ""
            )
            ws       = analytics_workers.get(pid, {})
            avg_load = ws.get("avg_load_size", 0.0)
            avg_conf = ws.get("avg_confidence", 0.0)
            print(
                f"  P#{pid:<8} {peak_sacks:<16} {boxes:<10} "
                f"{peak_sacks + boxes:<8} {avg_load:<10.2f} {avg_conf:<10.3f}{flag}"
            )
        print("  " + _hr())
        print(
            f"  {'TOTAL':<10} {grand_sacks:<16} {grand_boxes:<10} "
            f"{grand_sacks + grand_boxes:<8}"
        )
        print("  " + _hr())
    else:
        print()
        print("  No deliveries recorded.")

    # ── Throughput ────────────────────────────────────────────
    tp    = analytics_summary.get("throughput", {})
    trend = analytics_summary.get("trend_by_minute", {})
    if tp:
        print()
        print("  THROUGHPUT")
        print("  " + _hr())
        print(f"  Items / minute (avg)  : {tp.get('items_per_minute', 0):.2f}")
        print(f"  Peak / minute         : {tp.get('peak_per_minute', 0):.0f}")
        print(f"  Elapsed               : {tp.get('elapsed_minutes', 0):.1f} min")
        if trend:
            print(f"  Trend by minute       : {trend}")
        print("  " + _hr())

    # ── Confidence breakdown ──────────────────────────────────
    all_events = delivery_log + box_pipeline.delivery_log
    if all_events:
        high = sum(1 for d in all_events if d.get("confidence_class") == "HIGH")
        med  = sum(1 for d in all_events if d.get("confidence_class") == "MEDIUM")
        low  = sum(1 for d in all_events if d.get("confidence_class") == "LOW")
        lo2  = sum(1 for d in all_events if not d.get("high_conf", True))
        print()
        print("  CONFIDENCE BREAKDOWN")
        print("  " + _hr())
        print(f"  HIGH         : {high}")
        print(f"  MEDIUM       : {med}")
        print(f"  LOW          : {low}")
        print(f"  Low pipeline : {lo2}")
        print("  " + _hr())

    # ── Grand totals ──────────────────────────────────────────
    # Prefer person_peak_count; fall back to analytics for workers missing from it
    def _peak(pid: int) -> int:
        if pid in person_peak_count and person_peak_count[pid] > 0:
            return person_peak_count[pid]
        return analytics_workers.get(pid, {}).get("total_sacks", 0)

    grand_sacks_final = sum(_peak(pid) for pid in carriers)
    grand_boxes_final = sum(box_del_counts.values())
    print()
    print(_hr("="))
    print(f"  GRAND TOTAL SACKS (peak-based)  : {grand_sacks_final}")
    print(f"  GRAND TOTAL BOXES               : {grand_boxes_final}")
    print(f"  COMBINED DELIVERIES             : {grand_sacks_final + grand_boxes_final}")
    print(_hr("="))
    print()
