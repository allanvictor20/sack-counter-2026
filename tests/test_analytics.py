"""
test_analytics.py — Unit tests for the AnalyticsEngine.
"""

import unittest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sack_counter.pipeline.analytics import AnalyticsEngine


class TestAnalyticsEngine(unittest.TestCase):
    def setUp(self):
        self.engine = AnalyticsEngine(src_fps=20.0)

    def test_record_and_summarise_single_worker(self):
        self.engine.record_sack_delivery(worker_id=1, peak_count=5, confidence=0.85, frame_no=100)
        self.engine.record_sack_delivery(worker_id=1, peak_count=3, confidence=0.70, frame_no=200)
        summary = self.engine.summary()
        w = summary["workers"][1]
        self.assertEqual(w["total_sacks"], 8)
        self.assertAlmostEqual(w["avg_load_size"], 4.0)
        self.assertAlmostEqual(w["avg_confidence"], 0.775)
        self.assertEqual(w["num_passes"], 2)

    def test_totals(self):
        self.engine.record_sack_delivery(1, 4, 0.9, 10)
        self.engine.record_sack_delivery(2, 6, 0.8, 20)
        self.engine.record_box_delivery(1, 2, 30)
        summary = self.engine.summary()
        self.assertEqual(summary["totals"]["sacks"], 10)
        self.assertEqual(summary["totals"]["boxes"], 2)
        self.assertEqual(summary["totals"]["combined"], 12)
        self.assertEqual(summary["totals"]["active_workers"], 2)

    def test_anomaly_counter(self):
        self.engine.record_anomaly()
        self.engine.record_anomaly()
        summary = self.engine.summary()
        self.assertEqual(summary["confidence"]["anomalies"], 2)

    def test_reid_counter(self):
        self.engine.record_reid_event()
        summary = self.engine.summary()
        self.assertEqual(summary["pipeline"]["reid_events"], 1)

    def test_trend_buckets(self):
        fps = 20.0
        bucket_size = int(fps * 60)
        self.engine.record_sack_delivery(1, 3, 0.9, frame_no=1)
        self.engine.record_sack_delivery(1, 2, 0.8, frame_no=bucket_size + 1)
        summary = self.engine.summary()
        self.assertIn("minute_1", summary["trend_by_minute"])
        self.assertIn("minute_2", summary["trend_by_minute"])

    def test_empty_summary_no_crash(self):
        summary = self.engine.summary()
        self.assertEqual(summary["totals"]["sacks"], 0)
        self.assertEqual(summary["totals"]["combined"], 0)

    def test_worker_stats_direct(self):
        self.engine.record_sack_delivery(7, 5, 0.9, 50)
        ws = self.engine.worker_stats(7)
        self.assertIsNotNone(ws)
        self.assertEqual(ws.total_sacks, 5)
        self.assertEqual(ws.total_items, 5)

    def test_unknown_worker_stats_returns_none(self):
        self.assertIsNone(self.engine.worker_stats(999))


if __name__ == "__main__":
    unittest.main()
