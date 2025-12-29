//! Event detection algorithms

use super::types::{CryptoEvent, EventMetadata, EventType};
use crate::data::types::Candle;

/// Event detector for crypto market data
///
/// Detects various market events that may be analogous to earnings announcements:
/// - Volume spikes
/// - Price gaps
/// - Volatility expansion
///
/// # Example
///
/// ```rust,ignore
/// let detector = EventDetector::new(2.0, 0.03);
/// let events = detector.detect_all_events(&candles);
/// ```
pub struct EventDetector {
    /// Threshold for volume spike (multiple of average)
    volume_threshold: f64,
    /// Threshold for price gap (percentage)
    gap_threshold: f64,
    /// Lookback period for calculating averages
    lookback: usize,
    /// Minimum magnitude to report
    min_magnitude: f64,
}

impl Default for EventDetector {
    fn default() -> Self {
        Self {
            volume_threshold: 2.0,
            gap_threshold: 0.03,
            lookback: 20,
            min_magnitude: 1.5,
        }
    }
}

impl EventDetector {
    /// Create a new event detector
    pub fn new(volume_threshold: f64, gap_threshold: f64) -> Self {
        Self {
            volume_threshold,
            gap_threshold,
            ..Default::default()
        }
    }

    /// Set lookback period
    pub fn with_lookback(mut self, lookback: usize) -> Self {
        self.lookback = lookback;
        self
    }

    /// Set minimum magnitude
    pub fn with_min_magnitude(mut self, min_magnitude: f64) -> Self {
        self.min_magnitude = min_magnitude;
        self
    }

    /// Detect all types of events
    pub fn detect_all_events(&self, candles: &[Candle]) -> Vec<CryptoEvent> {
        let mut events = Vec::new();

        // Detect different event types
        events.extend(self.detect_volume_events(candles));
        events.extend(self.detect_gap_events(candles));
        events.extend(self.detect_volatility_events(candles));
        events.extend(self.detect_large_moves(candles));

        // Sort by timestamp
        events.sort_by_key(|e| e.timestamp);

        // Merge overlapping events
        self.merge_events(events)
    }

    /// Detect volume spike events
    pub fn detect_volume_events(&self, candles: &[Candle]) -> Vec<CryptoEvent> {
        if candles.len() < self.lookback + 1 {
            return vec![];
        }

        let mut events = Vec::new();

        for i in self.lookback..candles.len() {
            let current = &candles[i];

            // Calculate average volume over lookback period
            let avg_volume: f64 = candles[(i - self.lookback)..i]
                .iter()
                .map(|c| c.volume)
                .sum::<f64>()
                / self.lookback as f64;

            if avg_volume == 0.0 {
                continue;
            }

            let volume_multiple = current.volume / avg_volume;

            if volume_multiple >= self.volume_threshold {
                let direction = if current.close > current.open {
                    1.0
                } else {
                    -1.0
                };

                events.push(CryptoEvent {
                    timestamp: current.timestamp,
                    event_type: EventType::VolumeSpike,
                    magnitude: volume_multiple,
                    direction: direction * volume_multiple,
                    price: current.close,
                    volume: current.volume,
                    candle_index: i,
                    metadata: EventMetadata {
                        volume_multiple: Some(volume_multiple),
                        return_pct: Some(current.return_pct()),
                        ..Default::default()
                    },
                });
            }
        }

        events
    }

    /// Detect price gap events
    pub fn detect_gap_events(&self, candles: &[Candle]) -> Vec<CryptoEvent> {
        if candles.len() < 2 {
            return vec![];
        }

        let mut events = Vec::new();

        for i in 1..candles.len() {
            let prev = &candles[i - 1];
            let current = &candles[i];

            // Gap is open vs previous close
            let gap = (current.open - prev.close) / prev.close;

            if gap.abs() >= self.gap_threshold {
                let magnitude = gap.abs() / self.gap_threshold;

                events.push(CryptoEvent {
                    timestamp: current.timestamp,
                    event_type: EventType::PriceGap,
                    magnitude,
                    direction: gap.signum() * magnitude,
                    price: current.open,
                    volume: current.volume,
                    candle_index: i,
                    metadata: EventMetadata {
                        gap_size: Some(gap * 100.0),
                        return_pct: Some(current.return_pct()),
                        ..Default::default()
                    },
                });
            }
        }

        events
    }

    /// Detect volatility expansion events
    pub fn detect_volatility_events(&self, candles: &[Candle]) -> Vec<CryptoEvent> {
        if candles.len() < self.lookback + 1 {
            return vec![];
        }

        let mut events = Vec::new();

        // Calculate true ranges
        let mut true_ranges: Vec<f64> = vec![candles[0].high - candles[0].low];
        for i in 1..candles.len() {
            true_ranges.push(candles[i].true_range(Some(candles[i - 1].close)));
        }

        for i in self.lookback..candles.len() {
            let current = &candles[i];
            let current_tr = true_ranges[i];

            // Calculate average true range
            let avg_tr: f64 = true_ranges[(i - self.lookback)..i].iter().sum::<f64>()
                / self.lookback as f64;

            if avg_tr == 0.0 {
                continue;
            }

            let tr_multiple = current_tr / avg_tr;

            if tr_multiple >= self.volume_threshold {
                let direction = if current.close > current.open {
                    1.0
                } else {
                    -1.0
                };

                events.push(CryptoEvent {
                    timestamp: current.timestamp,
                    event_type: EventType::VolatilityExpansion,
                    magnitude: tr_multiple,
                    direction: direction * tr_multiple,
                    price: current.close,
                    volume: current.volume,
                    candle_index: i,
                    metadata: EventMetadata {
                        volatility_multiple: Some(tr_multiple),
                        return_pct: Some(current.return_pct()),
                        ..Default::default()
                    },
                });
            }
        }

        events
    }

    /// Detect large single-candle moves
    pub fn detect_large_moves(&self, candles: &[Candle]) -> Vec<CryptoEvent> {
        if candles.len() < self.lookback + 1 {
            return vec![];
        }

        let mut events = Vec::new();

        // Calculate returns
        let returns: Vec<f64> = candles.iter().map(|c| c.return_pct()).collect();

        for i in self.lookback..candles.len() {
            let current = &candles[i];
            let current_return = returns[i];

            // Calculate mean and std of returns
            let slice = &returns[(i - self.lookback)..i];
            let mean: f64 = slice.iter().sum::<f64>() / self.lookback as f64;
            let variance: f64 = slice.iter().map(|r| (r - mean).powi(2)).sum::<f64>()
                / (self.lookback - 1) as f64;
            let std = variance.sqrt();

            if std == 0.0 {
                continue;
            }

            let zscore = (current_return - mean) / std;

            if zscore.abs() >= 2.0 {
                events.push(CryptoEvent {
                    timestamp: current.timestamp,
                    event_type: EventType::LargeMove,
                    magnitude: zscore.abs(),
                    direction: zscore,
                    price: current.close,
                    volume: current.volume,
                    candle_index: i,
                    metadata: EventMetadata {
                        return_pct: Some(current_return * 100.0),
                        ..Default::default()
                    },
                });
            }
        }

        events
    }

    /// Merge overlapping events at the same timestamp
    fn merge_events(&self, events: Vec<CryptoEvent>) -> Vec<CryptoEvent> {
        use std::collections::HashMap;

        let mut by_timestamp: HashMap<u64, Vec<CryptoEvent>> = HashMap::new();

        for event in events {
            by_timestamp
                .entry(event.timestamp)
                .or_default()
                .push(event);
        }

        let mut result = Vec::new();

        for (timestamp, events) in by_timestamp {
            if events.len() == 1 {
                result.push(events.into_iter().next().unwrap());
            } else {
                // Multiple events at same timestamp - create composite
                let total_magnitude: f64 = events.iter().map(|e| e.magnitude).sum();
                let avg_direction: f64 =
                    events.iter().map(|e| e.direction).sum::<f64>() / events.len() as f64;

                let first = &events[0];
                result.push(CryptoEvent {
                    timestamp,
                    event_type: EventType::CompositeEvent,
                    magnitude: total_magnitude / events.len() as f64,
                    direction: avg_direction,
                    price: first.price,
                    volume: first.volume,
                    candle_index: first.candle_index,
                    metadata: EventMetadata {
                        volume_multiple: events.iter().filter_map(|e| e.metadata.volume_multiple).next(),
                        gap_size: events.iter().filter_map(|e| e.metadata.gap_size).next(),
                        volatility_multiple: events.iter().filter_map(|e| e.metadata.volatility_multiple).next(),
                        return_pct: first.metadata.return_pct,
                    },
                });
            }
        }

        result.sort_by_key(|e| e.timestamp);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_candles() -> Vec<Candle> {
        vec![
            Candle { timestamp: 0, open: 100.0, high: 101.0, low: 99.0, close: 100.5, volume: 1000.0, turnover: 100000.0 },
            Candle { timestamp: 1000, open: 100.5, high: 101.5, low: 100.0, close: 101.0, volume: 1100.0, turnover: 111000.0 },
            Candle { timestamp: 2000, open: 101.0, high: 102.0, low: 100.5, close: 101.5, volume: 1200.0, turnover: 121800.0 },
            Candle { timestamp: 3000, open: 101.5, high: 102.5, low: 101.0, close: 102.0, volume: 5000.0, turnover: 510000.0 }, // Volume spike
        ]
    }

    #[test]
    fn test_detector_creation() {
        let detector = EventDetector::new(2.0, 0.03);
        assert_eq!(detector.volume_threshold, 2.0);
        assert_eq!(detector.gap_threshold, 0.03);
    }

    #[test]
    fn test_volume_detection() {
        let detector = EventDetector::new(2.0, 0.03).with_lookback(3);
        let candles = make_test_candles();
        let events = detector.detect_volume_events(&candles);

        assert!(!events.is_empty());
        assert!(events.iter().any(|e| e.event_type == EventType::VolumeSpike));
    }
}
