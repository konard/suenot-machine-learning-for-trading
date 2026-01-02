//! Trading signal generation

use crate::graph::market_graph::MarketGraph;
use crate::transformer::model::{get_predictions, GraphTransformer, NodePrediction};
use ndarray::Array2;
use serde::{Deserialize, Serialize};

/// Signal type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SignalType {
    Long,
    Short,
    Hold,
}

/// Trading signal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Signal {
    /// Asset symbol
    pub symbol: String,
    /// Signal type
    pub signal_type: SignalType,
    /// Confidence (0-1)
    pub confidence: f64,
    /// Expected return
    pub expected_return: f64,
    /// Timestamp
    pub timestamp: i64,
}

impl Signal {
    /// Create a new signal
    pub fn new(symbol: &str, signal_type: SignalType, confidence: f64, expected_return: f64) -> Self {
        Self {
            symbol: symbol.to_string(),
            signal_type,
            confidence,
            expected_return,
            timestamp: chrono::Utc::now().timestamp_millis(),
        }
    }

    /// Check if signal is actionable (not hold)
    pub fn is_actionable(&self) -> bool {
        self.signal_type != SignalType::Hold
    }

    /// Get signal direction as multiplier (-1, 0, 1)
    pub fn direction(&self) -> i32 {
        match self.signal_type {
            SignalType::Long => 1,
            SignalType::Short => -1,
            SignalType::Hold => 0,
        }
    }
}

/// Signal generator configuration
#[derive(Debug, Clone)]
pub struct SignalGeneratorConfig {
    /// Minimum probability to generate a signal
    pub min_probability: f64,
    /// Minimum expected return to generate a signal
    pub min_expected_return: f64,
    /// Maximum signals to generate
    pub max_signals: usize,
    /// Whether to allow short signals
    pub allow_shorts: bool,
}

impl Default for SignalGeneratorConfig {
    fn default() -> Self {
        Self {
            min_probability: 0.55,
            min_expected_return: 0.001,
            max_signals: 10,
            allow_shorts: true,
        }
    }
}

/// Signal generator
pub struct SignalGenerator {
    config: SignalGeneratorConfig,
}

impl Default for SignalGenerator {
    fn default() -> Self {
        Self::new(0.55)
    }
}

impl SignalGenerator {
    /// Create with default config and custom threshold
    pub fn new(min_probability: f64) -> Self {
        Self {
            config: SignalGeneratorConfig {
                min_probability,
                ..Default::default()
            },
        }
    }

    /// Create with full config
    pub fn with_config(config: SignalGeneratorConfig) -> Self {
        Self { config }
    }

    /// Generate signals from model predictions
    pub fn generate(
        &self,
        model: &GraphTransformer,
        features: &Array2<f64>,
        graph: &MarketGraph,
    ) -> Vec<Signal> {
        let predictions = get_predictions(model, features, graph);
        self.from_predictions(&predictions)
    }

    /// Generate signals from pre-computed predictions
    pub fn from_predictions(&self, predictions: &[NodePrediction]) -> Vec<Signal> {
        let mut signals: Vec<Signal> = predictions
            .iter()
            .filter_map(|pred| self.prediction_to_signal(pred))
            .collect();

        // Sort by confidence and take top signals
        signals.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
        signals.truncate(self.config.max_signals);

        signals
    }

    /// Convert a single prediction to a signal
    fn prediction_to_signal(&self, pred: &NodePrediction) -> Option<Signal> {
        // Check for long signal
        if pred.prob_up >= self.config.min_probability
            && pred.predicted_return >= self.config.min_expected_return
        {
            return Some(Signal::new(
                &pred.symbol,
                SignalType::Long,
                pred.prob_up,
                pred.predicted_return,
            ));
        }

        // Check for short signal
        if self.config.allow_shorts
            && pred.prob_down >= self.config.min_probability
            && pred.predicted_return <= -self.config.min_expected_return
        {
            return Some(Signal::new(
                &pred.symbol,
                SignalType::Short,
                pred.prob_down,
                pred.predicted_return,
            ));
        }

        None
    }

    /// Generate hold signals for assets without strong predictions
    pub fn generate_with_holds(
        &self,
        model: &GraphTransformer,
        features: &Array2<f64>,
        graph: &MarketGraph,
    ) -> Vec<Signal> {
        let predictions = get_predictions(model, features, graph);

        predictions
            .iter()
            .map(|pred| {
                self.prediction_to_signal(pred).unwrap_or_else(|| {
                    Signal::new(&pred.symbol, SignalType::Hold, pred.confidence(), 0.0)
                })
            })
            .collect()
    }
}

/// Simple rule-based signal generator (no ML)
pub struct RuleBasedSignalGenerator {
    /// RSI overbought threshold
    pub rsi_overbought: f64,
    /// RSI oversold threshold
    pub rsi_oversold: f64,
    /// Minimum volume ratio
    pub min_volume_ratio: f64,
}

impl Default for RuleBasedSignalGenerator {
    fn default() -> Self {
        Self {
            rsi_overbought: 70.0,
            rsi_oversold: 30.0,
            min_volume_ratio: 1.5,
        }
    }
}

impl RuleBasedSignalGenerator {
    /// Generate signals based on simple rules
    pub fn generate(&self, graph: &MarketGraph) -> Vec<Signal> {
        graph
            .nodes
            .iter()
            .filter_map(|node| {
                // Simple momentum-based signal
                if node.price_change_24h > 5.0 {
                    Some(Signal::new(
                        &node.symbol,
                        SignalType::Long,
                        0.6,
                        node.price_change_24h / 100.0,
                    ))
                } else if node.price_change_24h < -5.0 {
                    Some(Signal::new(
                        &node.symbol,
                        SignalType::Short,
                        0.6,
                        node.price_change_24h / 100.0,
                    ))
                } else {
                    None
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transformer::model::NodePrediction;

    #[test]
    fn test_signal_generation() {
        let predictions = vec![
            NodePrediction {
                symbol: "BTCUSDT".to_string(),
                prob_up: 0.7,
                prob_neutral: 0.2,
                prob_down: 0.1,
                predicted_return: 0.02,
            },
            NodePrediction {
                symbol: "ETHUSDT".to_string(),
                prob_up: 0.3,
                prob_neutral: 0.4,
                prob_down: 0.3,
                predicted_return: 0.0,
            },
            NodePrediction {
                symbol: "DOGEUSDT".to_string(),
                prob_up: 0.1,
                prob_neutral: 0.2,
                prob_down: 0.7,
                predicted_return: -0.03,
            },
        ];

        let generator = SignalGenerator::new(0.6);
        let signals = generator.from_predictions(&predictions);

        assert_eq!(signals.len(), 2); // BTC long, DOGE short
        assert_eq!(signals[0].symbol, "BTCUSDT");
        assert_eq!(signals[0].signal_type, SignalType::Long);
    }
}
