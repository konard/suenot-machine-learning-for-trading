//! Machine Learning Signal Strategy.
//!
//! This strategy uses pre-computed ML signals (predictions) from an external model.
//! Signals can be loaded from a file or provided programmatically.

use crate::models::Candle;
use crate::strategies::{Signal, Strategy};
use chrono::{DateTime, Utc};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// ML Signal Strategy.
///
/// Uses pre-computed signals from a machine learning model.
/// Signals are mapped by timestamp and should be in range [-1, 1]:
/// - Positive values indicate buy signals
/// - Negative values indicate sell signals
/// - Values close to 0 indicate no action
#[derive(Debug, Clone)]
pub struct MlSignalStrategy {
    /// Pre-computed signals indexed by timestamp
    signals: HashMap<i64, f64>,
    /// Threshold for generating buy signal
    buy_threshold: f64,
    /// Threshold for generating sell signal
    sell_threshold: f64,
    /// Name of the strategy
    name: String,
}

impl MlSignalStrategy {
    /// Create a new ML signal strategy.
    ///
    /// # Arguments
    /// * `signals` - Map of Unix timestamp (seconds) to signal value (-1 to 1)
    /// * `buy_threshold` - Minimum signal value to generate buy (e.g., 0.5)
    /// * `sell_threshold` - Maximum signal value to generate sell (e.g., -0.5)
    pub fn new(
        signals: HashMap<i64, f64>,
        buy_threshold: f64,
        sell_threshold: f64,
    ) -> Self {
        Self {
            signals,
            buy_threshold,
            sell_threshold,
            name: "ML Signal Strategy".to_string(),
        }
    }

    /// Create from a vector of (timestamp, signal) pairs.
    pub fn from_vec(
        signals: Vec<(DateTime<Utc>, f64)>,
        buy_threshold: f64,
        sell_threshold: f64,
    ) -> Self {
        let signal_map: HashMap<i64, f64> = signals
            .into_iter()
            .map(|(ts, sig)| (ts.timestamp(), sig))
            .collect();

        Self::new(signal_map, buy_threshold, sell_threshold)
    }

    /// Load signals from a JSON file.
    ///
    /// Expected format: `{"signals": [{"timestamp": 1234567890, "value": 0.75}, ...]}`
    pub fn from_json_file(
        path: &Path,
        buy_threshold: f64,
        sell_threshold: f64,
    ) -> anyhow::Result<Self> {
        let content = fs::read_to_string(path)?;
        let data: serde_json::Value = serde_json::from_str(&content)?;

        let signals: HashMap<i64, f64> = data["signals"]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("Invalid format: missing 'signals' array"))?
            .iter()
            .filter_map(|item| {
                let ts = item["timestamp"].as_i64()?;
                let val = item["value"].as_f64()?;
                Some((ts, val))
            })
            .collect();

        Ok(Self::new(signals, buy_threshold, sell_threshold))
    }

    /// Load signals from a CSV file.
    ///
    /// Expected format: `timestamp,signal` with header row
    pub fn from_csv_file(
        path: &Path,
        buy_threshold: f64,
        sell_threshold: f64,
    ) -> anyhow::Result<Self> {
        let content = fs::read_to_string(path)?;
        let mut signals = HashMap::new();

        for (i, line) in content.lines().enumerate() {
            if i == 0 {
                continue; // Skip header
            }

            let parts: Vec<&str> = line.split(',').collect();
            if parts.len() >= 2 {
                if let (Ok(ts), Ok(val)) = (parts[0].parse::<i64>(), parts[1].parse::<f64>()) {
                    signals.insert(ts, val);
                }
            }
        }

        Ok(Self::new(signals, buy_threshold, sell_threshold))
    }

    /// Set custom name for the strategy.
    pub fn with_name(mut self, name: &str) -> Self {
        self.name = name.to_string();
        self
    }

    /// Get the signal for a specific timestamp.
    pub fn get_signal(&self, timestamp: DateTime<Utc>) -> Option<f64> {
        self.signals.get(&timestamp.timestamp()).copied()
    }

    /// Get number of signals loaded.
    pub fn signal_count(&self) -> usize {
        self.signals.len()
    }
}

impl Strategy for MlSignalStrategy {
    fn name(&self) -> &str {
        &self.name
    }

    fn on_candle(&mut self, candle: &Candle, _historical: &[Candle]) -> Signal {
        let timestamp = candle.timestamp.timestamp();

        match self.signals.get(&timestamp) {
            Some(&signal_value) => {
                if signal_value >= self.buy_threshold {
                    // Normalize signal to 0-1 range for position sizing
                    let strength = ((signal_value - self.buy_threshold)
                        / (1.0 - self.buy_threshold))
                        .min(1.0)
                        .max(0.0);
                    Signal::Buy(0.5 + strength * 0.5) // 0.5 to 1.0
                } else if signal_value <= self.sell_threshold {
                    let strength = ((self.sell_threshold - signal_value)
                        / (1.0 + self.sell_threshold))
                        .min(1.0)
                        .max(0.0);
                    Signal::Sell(0.5 + strength * 0.5)
                } else {
                    Signal::Hold
                }
            }
            None => Signal::Hold,
        }
    }

    fn reset(&mut self) {
        // Signals are static, nothing to reset
    }
}

/// Generate mock ML signals for testing.
///
/// Creates signals based on a simple momentum indicator.
pub fn generate_mock_signals(candles: &[Candle], lookback: usize) -> Vec<(DateTime<Utc>, f64)> {
    let mut signals = Vec::new();

    for (i, candle) in candles.iter().enumerate() {
        if i < lookback {
            signals.push((candle.timestamp, 0.0));
            continue;
        }

        // Simple momentum: compare current price to lookback price
        let past_price = candles[i - lookback].close;
        let current_price = candle.close;

        // Calculate return and normalize to -1 to 1
        let returns = (current_price - past_price) / past_price;
        let signal = (returns * 10.0).tanh(); // Squash to -1 to 1

        signals.push((candle.timestamp, signal));
    }

    signals
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    #[test]
    fn test_ml_signal_strategy() {
        let mut signals = HashMap::new();
        let ts = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        signals.insert(ts.timestamp(), 0.8);

        let mut strategy = MlSignalStrategy::new(signals, 0.5, -0.5);

        let candle = Candle::new(ts, "BTCUSDT".to_string(), 100.0, 101.0, 99.0, 100.5, 1000.0);

        let signal = strategy.on_candle(&candle, &[candle.clone()]);
        assert!(signal.is_buy());
    }

    #[test]
    fn test_generate_mock_signals() {
        let candles: Vec<Candle> = (0..50)
            .map(|i| {
                let ts = Utc::now();
                Candle::new(
                    ts,
                    "BTCUSDT".to_string(),
                    100.0 + i as f64,
                    101.0 + i as f64,
                    99.0 + i as f64,
                    100.5 + i as f64,
                    1000.0,
                )
            })
            .collect();

        let signals = generate_mock_signals(&candles, 10);
        assert_eq!(signals.len(), candles.len());

        // Uptrending prices should generate positive signals
        let last_signal = signals.last().unwrap().1;
        assert!(last_signal > 0.0);
    }
}
