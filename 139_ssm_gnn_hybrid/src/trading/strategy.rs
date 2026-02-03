//! SSM-GNN trading strategy implementation.

use crate::data::bybit::Kline;
use crate::data::features::{build_correlation_graph, compute_technical_features};
use crate::model::SsmGnnHybrid;
use crate::trading::signals::{compute_weights, generate_signals};

/// Configuration for the SSM-GNN trading strategy.
pub struct StrategyConfig {
    /// SSM model dimension
    pub d_model: usize,
    /// SSM state dimension
    pub d_state: usize,
    /// Lookback window for features
    pub feature_window: usize,
    /// Lookback window for model input
    pub model_window: usize,
    /// Correlation threshold for graph construction
    pub correlation_threshold: f64,
}

impl Default for StrategyConfig {
    fn default() -> Self {
        Self {
            d_model: 64,
            d_state: 16,
            feature_window: 14,
            model_window: 50,
            correlation_threshold: 0.3,
        }
    }
}

/// SSM-GNN trading strategy.
///
/// Processes multi-asset kline data through the SSM-GNN hybrid model
/// to generate portfolio weights.
pub struct SsmGnnStrategy {
    config: StrategyConfig,
    model: SsmGnnHybrid,
}

impl SsmGnnStrategy {
    /// Create a new strategy with given config.
    pub fn new(config: StrategyConfig, n_features: usize) -> Self {
        let model = SsmGnnHybrid::new(n_features, config.d_model, config.d_state, 3);
        Self { config, model }
    }

    /// Generate portfolio weights from multi-asset kline data.
    ///
    /// # Arguments
    /// * `multi_klines` - Vector of kline data per asset
    ///
    /// # Returns
    /// Portfolio weight vector (one weight per asset)
    pub fn generate_weights(&self, multi_klines: &[Vec<Kline>]) -> Vec<f64> {
        let n_assets = multi_klines.len();
        if n_assets == 0 {
            return vec![];
        }

        // Compute features per asset
        let all_features: Vec<Vec<Vec<f64>>> = multi_klines
            .iter()
            .map(|klines| compute_technical_features(klines, self.config.feature_window))
            .collect();

        // Get close prices for graph construction
        let close_prices: Vec<Vec<f64>> = multi_klines
            .iter()
            .map(|klines| klines.iter().map(|k| k.close).collect())
            .collect();

        // Build correlation graph
        let (src, dst, _weights) =
            build_correlation_graph(&close_prices, self.config.correlation_threshold);
        let edge_index = vec![src, dst];

        // Window the features for model input
        let min_len = all_features.iter().map(|f| f.len()).min().unwrap_or(0);
        if min_len < self.config.model_window {
            return vec![0.0; n_assets];
        }

        let windowed_features: Vec<Vec<Vec<f64>>> = all_features
            .iter()
            .map(|feats| feats[min_len - self.config.model_window..min_len].to_vec())
            .collect();

        // Generate signals
        let signals = generate_signals(&self.model, &windowed_features, &edge_index);

        // Compute weights
        compute_weights(&signals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::bybit::generate_simulated_klines;

    #[test]
    fn test_strategy_generate_weights() {
        let config = StrategyConfig::default();
        let strategy = SsmGnnStrategy::new(config, 5);

        let klines = vec![
            generate_simulated_klines("BTCUSDT", 100),
            generate_simulated_klines("ETHUSDT", 100),
            generate_simulated_klines("SOLUSDT", 100),
        ];

        let weights = strategy.generate_weights(&klines);
        assert_eq!(weights.len(), 3);
    }
}
