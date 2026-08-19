"""Unit tests for live prediction metrics."""
from __future__ import annotations

import unittest
from datetime import datetime, timezone

import pandas as pd

from technical_analysis_trainer.storage.prediction_metrics import (
    PredictionPair,
    compute_stats_from_pairs,
)


def _utc(y, m, d, h: int) -> datetime:
    return datetime(y, m, d, h, tzinfo=timezone.utc)


class ComputeStatsFromPairsTest(unittest.TestCase):
    def test_quantile_coverage_and_trend_accuracy(self) -> None:
        t0 = _utc(2026, 6, 1, 10)
        t1 = _utc(2026, 6, 1, 11)
        t2 = _utc(2026, 6, 1, 12)

        pairs = [
            PredictionPair(
                candle_ts=t0,
                target_ts=t1,
                last_close=100.0,
                q10=98.0,
                q50=101.0,
                q90=105.0,
                predicted_trend="UP",
                prob_up=0.7,
                prob_down=0.3,
            ),
            PredictionPair(
                candle_ts=t1,
                target_ts=t2,
                last_close=102.0,
                q10=100.0,
                q50=103.0,
                q90=104.0,
                predicted_trend="UP",
                prob_up=0.4,
                prob_down=0.6,
            ),
        ]

        candles = pd.DataFrame(
            {
                "ts": [t0, t1, t2],
                "close": [100.0, 103.0, 101.5],
            }
        )

        result = compute_stats_from_pairs(
            pairs,
            candles,
            symbol="BTCUSDT",
            timeframe="1h",
            trend_threshold=0.001,
        )

        self.assertEqual(result["meta"]["evaluated_count"], 2)
        self.assertEqual(result["quantile"]["coverage_rate"], 1.0)
        self.assertAlmostEqual(result["trend"]["accuracy"], 0.5)
        self.assertEqual(result["trend"]["confusion"]["tp"], 1)
        self.assertEqual(result["trend"]["confusion"]["fp"], 1)
        self.assertEqual(len(result["series"]), 2)
        self.assertTrue(result["series"][0]["in_band"])
        self.assertTrue(result["series"][0]["trend_correct"])

    def test_neutral_trend_skipped(self) -> None:
        t0 = _utc(2026, 6, 2, 8)
        t1 = _utc(2026, 6, 2, 9)

        pairs = [
            PredictionPair(
                candle_ts=t0,
                target_ts=t1,
                last_close=100.0,
                q10=99.0,
                q50=100.0,
                q90=101.0,
                predicted_trend="UP",
                prob_up=0.55,
                prob_down=0.45,
            ),
        ]

        candles = pd.DataFrame({"ts": [t0, t1], "close": [100.0, 100.05]})

        result = compute_stats_from_pairs(
            pairs,
            candles,
            symbol="BTCUSDT",
            timeframe="1h",
            trend_threshold=0.001,
        )

        self.assertEqual(result["meta"]["evaluated_count"], 1)
        self.assertEqual(result["meta"]["neutral_skipped"], 1)
        self.assertIsNone(result["trend"]["accuracy"])
        self.assertIsNone(result["series"][0]["actual_trend"])
        self.assertIsNone(result["series"][0]["trend_correct"])

    def test_quantile_breach_outside_band(self) -> None:
        t0 = _utc(2026, 6, 3, 14)
        t1 = _utc(2026, 6, 3, 15)

        pairs = [
            PredictionPair(
                candle_ts=t0,
                target_ts=t1,
                last_close=100.0,
                q10=99.0,
                q50=100.0,
                q90=101.0,
                predicted_trend="DOWN",
                prob_up=0.2,
                prob_down=0.8,
            ),
        ]

        candles = pd.DataFrame({"ts": [t0, t1], "close": [100.0, 110.0]})

        result = compute_stats_from_pairs(
            pairs,
            candles,
            symbol="BTCUSDT",
            timeframe="1h",
        )

        self.assertEqual(result["quantile"]["coverage_rate"], 0.0)
        self.assertGreater(result["quantile"]["avg_breach_pct"], 0.0)
        self.assertFalse(result["series"][0]["in_band"])
        self.assertGreater(result["series"][0]["breach_pct"], 0.0)


if __name__ == "__main__":
    unittest.main()
