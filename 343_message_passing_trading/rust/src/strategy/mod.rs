//! Trading strategy module using MPNN signals.

mod signals;

pub use signals::*;

use crate::data::Candle;
use crate::graph::MarketGraph;
use crate::mpnn::MPNN;
use ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Signal types for trading decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SignalType {
    /// Strong buy signal
    StrongBuy,
    /// Buy signal
    Buy,
    /// Neutral / hold
    Neutral,
    /// Sell signal
    Sell,
    /// Strong sell signal
    StrongSell,
}

impl SignalType {
    /// Convert signal to position sizing factor.
    pub fn to_position_size(&self) -> f64 {
        match self {
            SignalType::StrongBuy => 1.0,
            SignalType::Buy => 0.5,
            SignalType::Neutral => 0.0,
            SignalType::Sell => -0.5,
            SignalType::StrongSell => -1.0,
        }
    }

    /// Create from a continuous score.
    pub fn from_score(score: f64) -> Self {
        if score > 0.5 {
            SignalType::StrongBuy
        } else if score > 0.2 {
            SignalType::Buy
        } else if score > -0.2 {
            SignalType::Neutral
        } else if score > -0.5 {
            SignalType::Sell
        } else {
            SignalType::StrongSell
        }
    }
}

/// A trading signal for a specific asset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Signal {
    /// Symbol
    pub symbol: String,
    /// Signal type
    pub signal_type: SignalType,
    /// Continuous score
    pub score: f64,
    /// Confidence level
    pub confidence: f64,
    /// Timestamp
    pub timestamp: u64,
    /// Additional metadata
    pub metadata: HashMap<String, f64>,
}

impl Signal {
    /// Create a new signal.
    pub fn new(symbol: impl Into<String>, score: f64, confidence: f64, timestamp: u64) -> Self {
        Self {
            symbol: symbol.into(),
            signal_type: SignalType::from_score(score),
            score,
            confidence,
            timestamp,
            metadata: HashMap::new(),
        }
    }

    /// Add metadata to the signal.
    pub fn with_metadata(mut self, key: impl Into<String>, value: f64) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Check if this is a buy signal.
    pub fn is_buy(&self) -> bool {
        matches!(self.signal_type, SignalType::Buy | SignalType::StrongBuy)
    }

    /// Check if this is a sell signal.
    pub fn is_sell(&self) -> bool {
        matches!(self.signal_type, SignalType::Sell | SignalType::StrongSell)
    }
}

/// MPNN-based trading strategy.
pub struct MPNNStrategy {
    /// MPNN model
    mpnn: MPNN,
    /// Thresholds for signal generation
    buy_threshold: f64,
    sell_threshold: f64,
    /// Minimum confidence for signals
    min_confidence: f64,
    /// Position size limits
    max_position_size: f64,
}

impl MPNNStrategy {
    /// Create a new MPNN strategy.
    pub fn new(mpnn: MPNN) -> Self {
        Self {
            mpnn,
            buy_threshold: 0.2,
            sell_threshold: -0.2,
            min_confidence: 0.5,
            max_position_size: 0.2,
        }
    }

    /// Set buy/sell thresholds.
    pub fn with_thresholds(mut self, buy: f64, sell: f64) -> Self {
        self.buy_threshold = buy;
        self.sell_threshold = sell;
        self
    }

    /// Set minimum confidence.
    pub fn with_min_confidence(mut self, conf: f64) -> Self {
        self.min_confidence = conf;
        self
    }

    /// Set maximum position size.
    pub fn with_max_position_size(mut self, size: f64) -> Self {
        self.max_position_size = size;
        self
    }

    /// Generate signals from market graph.
    pub fn generate_signals(
        &self,
        graph: &mut MarketGraph,
        timestamp: u64,
    ) -> Result<Vec<Signal>, crate::mpnn::MPNNError> {
        // Get MPNN output
        let output = self.mpnn.forward(graph)?;

        let mut signals = Vec::new();

        for (i, node) in graph.nodes.iter().enumerate() {
            let row = output.row(i);

            // Compute score (mean of output dimensions)
            let score: f64 = row.iter().sum::<f64>() / row.len() as f64;

            // Compute confidence (based on output variance)
            let mean = score;
            let variance: f64 = row.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / row.len() as f64;
            let confidence = 1.0 / (1.0 + variance.sqrt());

            // Apply tanh to bound score
            let bounded_score = score.tanh();

            let signal = Signal::new(
                node.symbol.clone(),
                bounded_score,
                confidence,
                timestamp,
            )
            .with_metadata("raw_score", score)
            .with_metadata("variance", variance);

            signals.push(signal);
        }

        // Filter by confidence
        let signals: Vec<Signal> = signals
            .into_iter()
            .filter(|s| s.confidence >= self.min_confidence)
            .collect();

        Ok(signals)
    }

    /// Generate portfolio weights from signals.
    pub fn generate_weights(&self, signals: &[Signal]) -> HashMap<String, f64> {
        let mut weights = HashMap::new();

        // Filter actionable signals
        let actionable: Vec<&Signal> = signals
            .iter()
            .filter(|s| s.score.abs() > self.buy_threshold.abs())
            .collect();

        if actionable.is_empty() {
            return weights;
        }

        // Compute raw weights
        let total_score: f64 = actionable.iter().map(|s| s.score.abs()).sum();

        for signal in actionable {
            let weight = (signal.score / total_score) * self.max_position_size;
            weights.insert(signal.symbol.clone(), weight);
        }

        weights
    }

    /// Get top N signals by absolute score.
    pub fn top_signals(&self, signals: &[Signal], n: usize) -> Vec<Signal> {
        let mut sorted = signals.to_vec();
        sorted.sort_by(|a, b| {
            b.score
                .abs()
                .partial_cmp(&a.score.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted.into_iter().take(n).collect()
    }

    /// Get buy signals sorted by score.
    pub fn buy_signals(&self, signals: &[Signal]) -> Vec<Signal> {
        let mut buys: Vec<Signal> = signals
            .iter()
            .filter(|s| s.score > self.buy_threshold)
            .cloned()
            .collect();
        buys.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        buys
    }

    /// Get sell signals sorted by score.
    pub fn sell_signals(&self, signals: &[Signal]) -> Vec<Signal> {
        let mut sells: Vec<Signal> = signals
            .iter()
            .filter(|s| s.score < self.sell_threshold)
            .cloned()
            .collect();
        sells.sort_by(|a, b| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sells
    }
}

/// Cross-asset momentum strategy using MPNN.
pub struct CrossAssetMomentum {
    /// Base strategy
    strategy: MPNNStrategy,
    /// Lookback period for momentum
    lookback: usize,
    /// Momentum threshold
    momentum_threshold: f64,
}

impl CrossAssetMomentum {
    /// Create a new cross-asset momentum strategy.
    pub fn new(mpnn: MPNN, lookback: usize) -> Self {
        Self {
            strategy: MPNNStrategy::new(mpnn),
            lookback,
            momentum_threshold: 0.05,
        }
    }

    /// Generate momentum-adjusted signals.
    pub fn generate_signals(
        &self,
        graph: &mut MarketGraph,
        candles: &HashMap<String, Vec<Candle>>,
        timestamp: u64,
    ) -> Result<Vec<Signal>, crate::mpnn::MPNNError> {
        // Get base MPNN signals
        let mut signals = self.strategy.generate_signals(graph, timestamp)?;

        // Adjust by momentum
        for signal in &mut signals {
            if let Some(symbol_candles) = candles.get(&signal.symbol) {
                let momentum = self.compute_momentum(symbol_candles);
                let momentum_factor = if momentum > self.momentum_threshold {
                    1.2
                } else if momentum < -self.momentum_threshold {
                    0.8
                } else {
                    1.0
                };

                signal.score *= momentum_factor;
                signal.signal_type = SignalType::from_score(signal.score);
                signal.metadata.insert("momentum".to_string(), momentum);
            }
        }

        Ok(signals)
    }

    /// Compute momentum from candle data.
    fn compute_momentum(&self, candles: &[Candle]) -> f64 {
        if candles.len() < self.lookback + 1 {
            return 0.0;
        }

        let recent = &candles[candles.len() - self.lookback..];
        let returns: Vec<f64> = recent
            .windows(2)
            .map(|w| (w[1].close - w[0].close) / w[0].close)
            .collect();

        if returns.is_empty() {
            return 0.0;
        }

        // Compute weighted average return (more recent = higher weight)
        let n = returns.len() as f64;
        let mut weighted_sum = 0.0;
        let mut weight_sum = 0.0;

        for (i, &r) in returns.iter().enumerate() {
            let weight = (i + 1) as f64;
            weighted_sum += r * weight;
            weight_sum += weight;
        }

        weighted_sum / weight_sum
    }
}

/// Market regime detector using graph structure.
pub struct RegimeDetector {
    /// Correlation threshold for high correlation regime
    high_corr_threshold: f64,
    /// Correlation threshold for low correlation regime
    low_corr_threshold: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarketRegime {
    /// High correlation - trending market
    HighCorrelation,
    /// Low correlation - mean reverting
    LowCorrelation,
    /// Mixed - normal market
    Normal,
    /// Crisis - very high correlation, negative trend
    Crisis,
}

impl RegimeDetector {
    /// Create a new regime detector.
    pub fn new() -> Self {
        Self {
            high_corr_threshold: 0.7,
            low_corr_threshold: 0.3,
        }
    }

    /// Detect current market regime from graph.
    pub fn detect(&self, graph: &MarketGraph) -> MarketRegime {
        if graph.edges.is_empty() {
            return MarketRegime::Normal;
        }

        // Calculate average edge weight (correlation)
        let avg_weight: f64 = graph.edges.iter().map(|e| e.weight.abs()).sum::<f64>()
            / graph.edges.len() as f64;

        // Calculate average node momentum (from features)
        let avg_momentum: f64 = graph
            .nodes
            .iter()
            .filter_map(|n| n.features.get(6).copied()) // Momentum is typically index 6
            .sum::<f64>()
            / graph.nodes.len().max(1) as f64;

        // Determine regime
        if avg_weight > self.high_corr_threshold {
            if avg_momentum < -0.02 {
                MarketRegime::Crisis
            } else {
                MarketRegime::HighCorrelation
            }
        } else if avg_weight < self.low_corr_threshold {
            MarketRegime::LowCorrelation
        } else {
            MarketRegime::Normal
        }
    }
}

impl Default for RegimeDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_signal_type_from_score() {
        assert_eq!(SignalType::from_score(0.7), SignalType::StrongBuy);
        assert_eq!(SignalType::from_score(0.3), SignalType::Buy);
        assert_eq!(SignalType::from_score(0.0), SignalType::Neutral);
        assert_eq!(SignalType::from_score(-0.3), SignalType::Sell);
        assert_eq!(SignalType::from_score(-0.7), SignalType::StrongSell);
    }

    #[test]
    fn test_signal_creation() {
        let signal = Signal::new("BTCUSDT", 0.5, 0.8, 1234567890);
        assert_eq!(signal.signal_type, SignalType::StrongBuy);
        assert!(signal.is_buy());
        assert!(!signal.is_sell());
    }

    #[test]
    fn test_regime_detector() {
        let detector = RegimeDetector::new();
        let mut graph = MarketGraph::new();

        // Add nodes
        let features = array![0.1, 0.2, 0.0, 0.0, 0.0, 0.0, 0.05, 0.0];
        graph.add_node("BTCUSDT", features.clone());
        graph.add_node("ETHUSDT", features);

        // High correlation edge
        graph.add_edge(0, 1, 0.9);

        let regime = detector.detect(&graph);
        assert_eq!(regime, MarketRegime::HighCorrelation);
    }
}
