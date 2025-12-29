//! Post-event drift analysis (PEAD analog for crypto)
//!
//! Analyzes how prices drift after significant events.

use crate::data::types::Candle;
use crate::events::CryptoEvent;
use chrono::{DateTime, TimeZone, Utc};
use serde::{Deserialize, Serialize};

/// Analyzer for post-event drift
pub struct PostEventAnalyzer {
    /// Number of periods to analyze after event
    window: usize,
}

impl Default for PostEventAnalyzer {
    fn default() -> Self {
        Self { window: 24 } // 24 hours for hourly data
    }
}

impl PostEventAnalyzer {
    /// Create a new analyzer with specified window
    pub fn new(window: usize) -> Self {
        Self { window }
    }

    /// Analyze drift after detected events
    pub fn analyze_drift(&self, candles: &[Candle], events: &[CryptoEvent]) -> Vec<DriftResult> {
        let mut results = Vec::new();

        for event in events {
            if let Some(result) = self.analyze_single_event(candles, event) {
                results.push(result);
            }
        }

        results
    }

    /// Analyze drift for a single event
    fn analyze_single_event(&self, candles: &[Candle], event: &CryptoEvent) -> Option<DriftResult> {
        let idx = event.candle_index;

        // Need enough candles after the event
        if idx + self.window >= candles.len() {
            return None;
        }

        let event_candle = &candles[idx];
        let event_price = event_candle.close;

        // Calculate returns at different horizons
        let day0_return = event_candle.return_pct();

        let day1_idx = (idx + 1).min(candles.len() - 1);
        let day1_return = (candles[day1_idx].close - event_price) / event_price;

        let day3_idx = (idx + 3).min(candles.len() - 1);
        let day3_return = (candles[day3_idx].close - event_price) / event_price;

        let day5_idx = (idx + 5).min(candles.len() - 1);
        let day5_return = (candles[day5_idx].close - event_price) / event_price;

        let end_idx = (idx + self.window).min(candles.len() - 1);
        let total_return = (candles[end_idx].close - event_price) / event_price;

        // Calculate maximum adverse excursion (MAE) and maximum favorable excursion (MFE)
        let post_candles = &candles[idx..=end_idx];
        let is_bullish = event.direction > 0.0;

        let (mae, mfe) = self.calculate_excursions(post_candles, event_price, is_bullish);

        // Check if drift continues in event direction
        let drift_continues = (event.direction > 0.0 && total_return > 0.0)
            || (event.direction < 0.0 && total_return < 0.0);

        Some(DriftResult {
            event_time: event.datetime(),
            event_type: event.event_type,
            event_magnitude: event.magnitude,
            event_direction: event.direction,
            event_price,
            day0_return,
            day1_return,
            day3_return,
            day5_return,
            total_return,
            mae,
            mfe,
            drift_continues,
        })
    }

    /// Calculate maximum adverse and favorable excursions
    fn calculate_excursions(
        &self,
        candles: &[Candle],
        entry_price: f64,
        is_long: bool,
    ) -> (f64, f64) {
        let mut mae = 0.0; // Maximum Adverse Excursion
        let mut mfe = 0.0; // Maximum Favorable Excursion

        for candle in candles {
            if is_long {
                // For long: low is adverse, high is favorable
                let adverse = (candle.low - entry_price) / entry_price;
                let favorable = (candle.high - entry_price) / entry_price;
                mae = mae.min(adverse);
                mfe = mfe.max(favorable);
            } else {
                // For short: high is adverse, low is favorable
                let adverse = (candle.high - entry_price) / entry_price;
                let favorable = (entry_price - candle.low) / entry_price;
                mae = mae.max(adverse);
                mfe = mfe.max(favorable);
            }
        }

        (mae, mfe)
    }

    /// Calculate aggregate drift statistics
    pub fn aggregate_stats(&self, results: &[DriftResult]) -> DriftStats {
        if results.is_empty() {
            return DriftStats::default();
        }

        let n = results.len() as f64;

        let avg_day0: f64 = results.iter().map(|r| r.day0_return).sum::<f64>() / n;
        let avg_day1: f64 = results.iter().map(|r| r.day1_return).sum::<f64>() / n;
        let avg_day3: f64 = results.iter().map(|r| r.day3_return).sum::<f64>() / n;
        let avg_day5: f64 = results.iter().map(|r| r.day5_return).sum::<f64>() / n;
        let avg_total: f64 = results.iter().map(|r| r.total_return).sum::<f64>() / n;

        let drift_rate = results.iter().filter(|r| r.drift_continues).count() as f64 / n;

        // Separate bullish and bearish events
        let bullish: Vec<_> = results.iter().filter(|r| r.event_direction > 0.0).collect();
        let bearish: Vec<_> = results.iter().filter(|r| r.event_direction < 0.0).collect();

        let bullish_drift = if !bullish.is_empty() {
            bullish.iter().map(|r| r.total_return).sum::<f64>() / bullish.len() as f64
        } else {
            0.0
        };

        let bearish_drift = if !bearish.is_empty() {
            bearish.iter().map(|r| r.total_return).sum::<f64>() / bearish.len() as f64
        } else {
            0.0
        };

        DriftStats {
            total_events: results.len(),
            avg_day0_return: avg_day0,
            avg_day1_return: avg_day1,
            avg_day3_return: avg_day3,
            avg_day5_return: avg_day5,
            avg_total_return: avg_total,
            drift_continuation_rate: drift_rate,
            bullish_avg_drift: bullish_drift,
            bearish_avg_drift: bearish_drift,
            bullish_events: bullish.len(),
            bearish_events: bearish.len(),
        }
    }

    /// Analyze drift by event magnitude bins
    pub fn analyze_by_magnitude(&self, results: &[DriftResult]) -> Vec<(String, DriftStats)> {
        let mut weak: Vec<_> = Vec::new();
        let mut moderate: Vec<_> = Vec::new();
        let mut strong: Vec<_> = Vec::new();

        for result in results {
            if result.event_magnitude < 2.0 {
                weak.push(result.clone());
            } else if result.event_magnitude < 3.0 {
                moderate.push(result.clone());
            } else {
                strong.push(result.clone());
            }
        }

        vec![
            ("Weak (<2.0)".to_string(), self.aggregate_stats(&weak)),
            ("Moderate (2.0-3.0)".to_string(), self.aggregate_stats(&moderate)),
            ("Strong (>3.0)".to_string(), self.aggregate_stats(&strong)),
        ]
    }
}

/// Result of drift analysis for a single event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftResult {
    /// Time of the event
    pub event_time: DateTime<Utc>,
    /// Type of event
    pub event_type: crate::events::EventType,
    /// Magnitude of the event
    pub event_magnitude: f64,
    /// Direction of the event
    pub event_direction: f64,
    /// Price at event
    pub event_price: f64,
    /// Return on day 0 (event day)
    pub day0_return: f64,
    /// Return from event to day 1
    pub day1_return: f64,
    /// Return from event to day 3
    pub day3_return: f64,
    /// Return from event to day 5
    pub day5_return: f64,
    /// Total return over analysis window
    pub total_return: f64,
    /// Maximum adverse excursion
    pub mae: f64,
    /// Maximum favorable excursion
    pub mfe: f64,
    /// Whether drift continues in event direction
    pub drift_continues: bool,
}

/// Aggregate drift statistics
#[derive(Debug, Clone, Default)]
pub struct DriftStats {
    pub total_events: usize,
    pub avg_day0_return: f64,
    pub avg_day1_return: f64,
    pub avg_day3_return: f64,
    pub avg_day5_return: f64,
    pub avg_total_return: f64,
    pub drift_continuation_rate: f64,
    pub bullish_avg_drift: f64,
    pub bearish_avg_drift: f64,
    pub bullish_events: usize,
    pub bearish_events: usize,
}

impl std::fmt::Display for DriftStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Post-Event Drift Statistics:")?;
        writeln!(f, "  Total Events: {}", self.total_events)?;
        writeln!(f, "  Day 0 Return: {:.2}%", self.avg_day0_return * 100.0)?;
        writeln!(f, "  Day 1 Return: {:.2}%", self.avg_day1_return * 100.0)?;
        writeln!(f, "  Day 3 Return: {:.2}%", self.avg_day3_return * 100.0)?;
        writeln!(f, "  Day 5 Return: {:.2}%", self.avg_day5_return * 100.0)?;
        writeln!(f, "  Total Return: {:.2}%", self.avg_total_return * 100.0)?;
        writeln!(
            f,
            "  Drift Continuation: {:.1}%",
            self.drift_continuation_rate * 100.0
        )?;
        writeln!(
            f,
            "  Bullish Drift: {:.2}% ({} events)",
            self.bullish_avg_drift * 100.0,
            self.bullish_events
        )?;
        writeln!(
            f,
            "  Bearish Drift: {:.2}% ({} events)",
            self.bearish_avg_drift * 100.0,
            self.bearish_events
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::EventType;

    fn make_candles() -> Vec<Candle> {
        (0..100)
            .map(|i| {
                let price = 100.0 + (i as f64 * 0.1);
                Candle {
                    timestamp: i * 3600000, // hourly
                    open: price,
                    high: price + 1.0,
                    low: price - 1.0,
                    close: price + 0.5,
                    volume: 1000.0,
                    turnover: 100000.0,
                }
            })
            .collect()
    }

    fn make_event(idx: usize) -> CryptoEvent {
        CryptoEvent {
            timestamp: idx as u64 * 3600000,
            event_type: EventType::VolumeSpike,
            magnitude: 2.5,
            direction: 1.0,
            price: 100.0 + (idx as f64 * 0.1),
            volume: 3000.0,
            candle_index: idx,
            metadata: Default::default(),
        }
    }

    #[test]
    fn test_drift_analysis() {
        let analyzer = PostEventAnalyzer::new(24);
        let candles = make_candles();
        let events = vec![make_event(10), make_event(40)];

        let results = analyzer.analyze_drift(&candles, &events);

        assert_eq!(results.len(), 2);
        assert!(results[0].total_return.is_finite());
    }

    #[test]
    fn test_aggregate_stats() {
        let analyzer = PostEventAnalyzer::new(24);
        let candles = make_candles();
        let events = vec![make_event(10), make_event(40)];

        let results = analyzer.analyze_drift(&candles, &events);
        let stats = analyzer.aggregate_stats(&results);

        assert_eq!(stats.total_events, 2);
    }
}
