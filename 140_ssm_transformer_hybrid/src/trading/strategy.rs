//! Trading strategy using SSM-Transformer Hybrid model.
//!
//! Combines model predictions with risk management to generate
//! position-sized trading signals.

use crate::model::hybrid::SSMTransformerModel;
use crate::data::bybit::Kline;
use crate::data::features::FeatureGenerator;
use crate::trading::signals::TradingSignal;

/// Configuration for the trading strategy.
#[derive(Debug, Clone)]
pub struct StrategyConfig {
    /// Threshold for direction probability to take a position
    pub direction_threshold: f64,
    /// Maximum position size (fraction of capital)
    pub max_position: f64,
    /// Base position size scaling factor
    pub base_size: f64,
    /// Lookback window (sequence length for model input)
    pub lookback: usize,
}

impl Default for StrategyConfig {
    fn default() -> Self {
        Self {
            direction_threshold: 0.55,
            max_position: 1.0,
            base_size: 0.1,
            lookback: 100,
        }
    }
}

/// Trading strategy that uses the SSM-Transformer Hybrid model
/// to generate trading signals from kline data.
#[derive(Debug)]
pub struct TradingStrategy {
    pub model: SSMTransformerModel,
    pub feature_gen: FeatureGenerator,
    pub config: StrategyConfig,
}

impl TradingStrategy {
    pub fn new(model: SSMTransformerModel, config: StrategyConfig) -> Self {
        Self {
            model,
            feature_gen: FeatureGenerator::new(),
            config,
        }
    }

    /// Generate a trading signal from the latest kline data.
    ///
    /// Requires at least `lookback + 50` klines (50 for feature warmup).
    pub fn generate_signal(&self, klines: &[Kline]) -> Option<TradingSignal> {
        let min_required = self.config.lookback + 50;
        if klines.len() < min_required {
            tracing::warn!(
                "Not enough data: {} klines, need at least {}",
                klines.len(),
                min_required
            );
            return None;
        }

        // Generate features
        let features = self.feature_gen.generate(klines);

        // Extract the last `lookback` valid feature rows
        let valid_features: Vec<Vec<f64>> = features
            .iter()
            .filter_map(|f| f.as_ref().map(|r| r.to_vec()))
            .collect();

        if valid_features.len() < self.config.lookback {
            return None;
        }

        let start = valid_features.len() - self.config.lookback;
        let sequence = &valid_features[start..];

        // Run model
        let preds = self.model.forward(sequence);

        Some(TradingSignal::from_predictions(
            &preds,
            self.config.direction_threshold,
            self.config.max_position,
            self.config.base_size,
        ))
    }

    /// Generate signals for all valid positions in the data.
    /// Returns (signal, actual_return) pairs.
    pub fn generate_all_signals(&self, klines: &[Kline]) -> Vec<(TradingSignal, f64)> {
        let features = self.feature_gen.generate(klines);
        let valid_features: Vec<Vec<f64>> = features
            .iter()
            .filter_map(|f| f.as_ref().map(|r| r.to_vec()))
            .collect();

        if valid_features.len() < self.config.lookback + 1 {
            return Vec::new();
        }

        let mut signals = Vec::new();
        let closes: Vec<f64> = klines.iter().map(|k| k.close).collect();
        let offset = klines.len() - valid_features.len();

        for i in self.config.lookback..valid_features.len() - 1 {
            let sequence = &valid_features[i - self.config.lookback..i];
            let preds = self.model.forward(sequence);

            let signal = TradingSignal::from_predictions(
                &preds,
                self.config.direction_threshold,
                self.config.max_position,
                self.config.base_size,
            );

            let actual_idx = offset + i;
            let actual_return = if actual_idx + 1 < closes.len() && closes[actual_idx] > 0.0 {
                (closes[actual_idx + 1] / closes[actual_idx]).ln()
            } else {
                0.0
            };

            signals.push((signal, actual_return));
        }

        signals
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::bybit::generate_synthetic_klines;

    #[test]
    fn test_strategy_signal_generation() {
        let model = SSMTransformerModel::new(15, 16, 2, 1, 8, 4);
        let config = StrategyConfig {
            lookback: 20,
            ..Default::default()
        };
        let strategy = TradingStrategy::new(model, config);

        let klines = generate_synthetic_klines(200);
        let signal = strategy.generate_signal(&klines);
        assert!(signal.is_some());
    }

    #[test]
    fn test_strategy_all_signals() {
        let model = SSMTransformerModel::new(15, 16, 2, 1, 8, 4);
        let config = StrategyConfig {
            lookback: 20,
            ..Default::default()
        };
        let strategy = TradingStrategy::new(model, config);

        let klines = generate_synthetic_klines(200);
        let signals = strategy.generate_all_signals(&klines);
        assert!(!signals.is_empty());
    }
}
