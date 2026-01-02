//! # Trading Signals
//!
//! Generate trading signals using score matching.

use crate::score::{LangevinConfig, LangevinDynamics, MultiScaleScoreNetwork};
use crate::utils::{Candle, MarketState, compute_market_state};
use ndarray::Array1;

/// Trading signal with metadata
#[derive(Debug, Clone)]
pub struct TradingSignal {
    /// Signal direction: positive = long, negative = short, zero = neutral
    pub signal: f64,
    /// Confidence level (0 to 1)
    pub confidence: f64,
    /// Whether current state is within learned distribution
    pub in_distribution: bool,
    /// Raw score values (for debugging)
    pub raw_score: Vec<f64>,
    /// Denoised market state
    pub denoised_state: Option<Vec<f64>>,
    /// Timestamp of signal
    pub timestamp: Option<chrono::DateTime<chrono::Utc>>,
}

impl TradingSignal {
    /// Get position size based on signal and confidence
    pub fn position_size(&self, base_size: f64) -> f64 {
        if !self.in_distribution {
            return 0.0;
        }
        self.signal * self.confidence * base_size
    }

    /// Check if should trade (confidence above threshold)
    pub fn should_trade(&self, threshold: f64) -> bool {
        self.in_distribution && self.confidence >= threshold
    }

    /// Get signal direction as string
    pub fn direction(&self) -> &'static str {
        if self.signal > 0.0 {
            "LONG"
        } else if self.signal < 0.0 {
            "SHORT"
        } else {
            "NEUTRAL"
        }
    }
}

/// Trader configuration
#[derive(Debug, Clone)]
pub struct TraderConfig {
    /// Index of return feature in state vector
    pub return_feature_idx: usize,
    /// Confidence threshold for trading
    pub confidence_threshold: f64,
    /// Log density threshold for in-distribution check
    pub density_threshold: f64,
    /// Number of denoising steps
    pub denoise_steps: usize,
    /// Lookback period for feature computation
    pub lookback: usize,
}

impl Default for TraderConfig {
    fn default() -> Self {
        Self {
            return_feature_idx: 0,
            confidence_threshold: 0.3,
            density_threshold: -5.0,
            denoise_steps: 20,
            lookback: 20,
        }
    }
}

/// Score Matching based trader
pub struct ScoreMatchingTrader {
    /// Score network
    network: MultiScaleScoreNetwork,
    /// Trader configuration
    config: TraderConfig,
    /// Langevin configuration
    langevin_config: LangevinConfig,
}

impl ScoreMatchingTrader {
    /// Create a new trader
    pub fn new(network: MultiScaleScoreNetwork) -> Self {
        Self {
            network,
            config: TraderConfig::default(),
            langevin_config: LangevinConfig::default(),
        }
    }

    /// Create trader with custom configuration
    pub fn with_config(network: MultiScaleScoreNetwork, config: TraderConfig) -> Self {
        Self {
            network,
            config,
            langevin_config: LangevinConfig::default(),
        }
    }

    /// Generate trading signal from market state
    ///
    /// # Arguments
    ///
    /// * `state` - Market state vector
    ///
    /// # Returns
    ///
    /// Trading signal with confidence and metadata
    pub fn generate_signal_from_state(&self, state: &Array1<f64>) -> TradingSignal {
        let langevin = LangevinDynamics::new(&self.network, self.langevin_config.clone());

        // Get signal from score
        let (signal, confidence) = langevin.get_trading_signal(state, self.config.return_feature_idx);

        // Check if in distribution
        let in_distribution = langevin.is_in_distribution(state, self.config.density_threshold);

        // Denoise state
        let denoised = langevin.denoise(state, self.config.denoise_steps);

        // Get raw score
        let level = self.network.num_noise_levels - 1;
        let score = self.network.forward(state, level);

        // Apply confidence threshold
        let final_signal = if in_distribution && confidence >= self.config.confidence_threshold {
            signal * confidence
        } else {
            0.0
        };

        TradingSignal {
            signal: final_signal,
            confidence,
            in_distribution,
            raw_score: score.to_vec(),
            denoised_state: Some(denoised.to_vec()),
            timestamp: Some(chrono::Utc::now()),
        }
    }

    /// Generate trading signal from candles
    ///
    /// # Arguments
    ///
    /// * `candles` - Recent candle data (at least lookback + 1 candles)
    ///
    /// # Returns
    ///
    /// Trading signal with confidence and metadata
    pub fn generate_signal(&self, candles: &[Candle]) -> Option<TradingSignal> {
        if candles.len() < self.config.lookback + 1 {
            return None;
        }

        // Compute market state
        let state = compute_market_state(candles, self.config.lookback);

        // Generate signal
        let mut signal = self.generate_signal_from_state(&state);

        // Set timestamp from last candle
        if let Some(last) = candles.last() {
            signal.timestamp = Some(last.timestamp);
        }

        Some(signal)
    }

    /// Get network reference
    pub fn network(&self) -> &MultiScaleScoreNetwork {
        &self.network
    }

    /// Get configuration reference
    pub fn config(&self) -> &TraderConfig {
        &self.config
    }

    /// Update configuration
    pub fn set_config(&mut self, config: TraderConfig) {
        self.config = config;
    }

    /// Update confidence threshold
    pub fn set_confidence_threshold(&mut self, threshold: f64) {
        self.config.confidence_threshold = threshold;
    }
}

/// Ensemble trader using multiple networks
pub struct EnsembleTrader {
    /// Individual traders
    traders: Vec<ScoreMatchingTrader>,
    /// Aggregation method
    aggregation: AggregationMethod,
}

/// Method for aggregating signals from multiple models
#[derive(Debug, Clone, Copy)]
pub enum AggregationMethod {
    /// Average all signals
    Mean,
    /// Take median signal
    Median,
    /// Weighted average by confidence
    ConfidenceWeighted,
    /// Only trade if majority agree on direction
    Majority,
}

impl EnsembleTrader {
    /// Create new ensemble trader
    pub fn new(traders: Vec<ScoreMatchingTrader>, aggregation: AggregationMethod) -> Self {
        Self { traders, aggregation }
    }

    /// Generate aggregated signal from all traders
    pub fn generate_signal(&self, candles: &[Candle]) -> Option<TradingSignal> {
        let signals: Vec<TradingSignal> = self.traders
            .iter()
            .filter_map(|t| t.generate_signal(candles))
            .collect();

        if signals.is_empty() {
            return None;
        }

        let (signal, confidence) = match self.aggregation {
            AggregationMethod::Mean => {
                let avg_signal: f64 = signals.iter().map(|s| s.signal).sum::<f64>() / signals.len() as f64;
                let avg_conf: f64 = signals.iter().map(|s| s.confidence).sum::<f64>() / signals.len() as f64;
                (avg_signal, avg_conf)
            }
            AggregationMethod::Median => {
                let mut sorted_signals: Vec<f64> = signals.iter().map(|s| s.signal).collect();
                sorted_signals.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let median = sorted_signals[sorted_signals.len() / 2];

                let mut sorted_conf: Vec<f64> = signals.iter().map(|s| s.confidence).collect();
                sorted_conf.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let median_conf = sorted_conf[sorted_conf.len() / 2];

                (median, median_conf)
            }
            AggregationMethod::ConfidenceWeighted => {
                let total_conf: f64 = signals.iter().map(|s| s.confidence).sum();
                if total_conf == 0.0 {
                    (0.0, 0.0)
                } else {
                    let weighted: f64 = signals.iter()
                        .map(|s| s.signal * s.confidence)
                        .sum::<f64>() / total_conf;
                    (weighted, total_conf / signals.len() as f64)
                }
            }
            AggregationMethod::Majority => {
                let n_long = signals.iter().filter(|s| s.signal > 0.0).count();
                let n_short = signals.iter().filter(|s| s.signal < 0.0).count();
                let n_total = signals.len();

                if n_long > n_total / 2 {
                    let avg_conf: f64 = signals.iter()
                        .filter(|s| s.signal > 0.0)
                        .map(|s| s.confidence)
                        .sum::<f64>() / n_long as f64;
                    (1.0, avg_conf)
                } else if n_short > n_total / 2 {
                    let avg_conf: f64 = signals.iter()
                        .filter(|s| s.signal < 0.0)
                        .map(|s| s.confidence)
                        .sum::<f64>() / n_short as f64;
                    (-1.0, avg_conf)
                } else {
                    (0.0, 0.0)
                }
            }
        };

        let in_dist = signals.iter().any(|s| s.in_distribution);

        Some(TradingSignal {
            signal,
            confidence,
            in_distribution: in_dist,
            raw_score: vec![],
            denoised_state: None,
            timestamp: signals.first().and_then(|s| s.timestamp),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trading_signal() {
        let signal = TradingSignal {
            signal: 0.5,
            confidence: 0.8,
            in_distribution: true,
            raw_score: vec![],
            denoised_state: None,
            timestamp: None,
        };

        assert_eq!(signal.direction(), "LONG");
        assert!(signal.should_trade(0.5));
        assert_eq!(signal.position_size(1.0), 0.4); // 0.5 * 0.8 * 1.0
    }

    #[test]
    fn test_trader_config_default() {
        let config = TraderConfig::default();
        assert_eq!(config.confidence_threshold, 0.3);
        assert_eq!(config.lookback, 20);
    }

    #[test]
    fn test_trader_creation() {
        let network = MultiScaleScoreNetwork::new(10, 32, 2, 5, 0.01, 1.0);
        let trader = ScoreMatchingTrader::new(network);

        assert_eq!(trader.config().lookback, 20);
    }
}
