//! Trading signal generation from sentiment analysis.
//!
//! This module provides:
//! - `TradingSignalGenerator`: Convert sentiment to trading signals
//! - `CryptoSignalGenerator`: Crypto-specific signal generation
//! - Signal aggregation methods

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Signal type (BUY, SELL, HOLD).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SignalType {
    Buy,
    Sell,
    Hold,
}

impl std::fmt::Display for SignalType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SignalType::Buy => write!(f, "BUY"),
            SignalType::Sell => write!(f, "SELL"),
            SignalType::Hold => write!(f, "HOLD"),
        }
    }
}

/// Sentiment type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Sentiment {
    Bullish,
    Neutral,
    Bearish,
}

impl Sentiment {
    /// Parse sentiment from string.
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "bullish" | "positive" | "buy" => Some(Sentiment::Bullish),
            "neutral" | "hold" => Some(Sentiment::Neutral),
            "bearish" | "negative" | "sell" => Some(Sentiment::Bearish),
            _ => None,
        }
    }

    /// Convert to class index.
    pub fn to_class(&self) -> usize {
        match self {
            Sentiment::Bullish => 0,
            Sentiment::Neutral => 1,
            Sentiment::Bearish => 2,
        }
    }
}

/// A trading signal with metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingSignal {
    /// Signal type
    pub signal: SignalType,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,
    /// Source sentiment
    pub sentiment: Sentiment,
    /// Optional price target
    pub price_target: Option<f64>,
    /// Optional stop loss
    pub stop_loss: Option<f64>,
    /// Optional take profit
    pub take_profit: Option<f64>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl TradingSignal {
    /// Create a new trading signal.
    pub fn new(signal: SignalType, confidence: f32, sentiment: Sentiment) -> Self {
        Self {
            signal,
            confidence,
            sentiment,
            price_target: None,
            stop_loss: None,
            take_profit: None,
            metadata: HashMap::new(),
        }
    }

    /// Set price target.
    pub fn with_price_target(mut self, target: f64) -> Self {
        self.price_target = Some(target);
        self
    }

    /// Set stop loss.
    pub fn with_stop_loss(mut self, stop: f64) -> Self {
        self.stop_loss = Some(stop);
        self
    }

    /// Set take profit.
    pub fn with_take_profit(mut self, tp: f64) -> Self {
        self.take_profit = Some(tp);
        self
    }

    /// Add metadata.
    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }
}

/// Trading signal generator configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalConfig {
    /// Threshold for bullish sentiment to generate BUY
    pub bullish_threshold: f32,
    /// Threshold for bearish sentiment to generate SELL
    pub bearish_threshold: f32,
    /// Minimum confidence to generate non-HOLD signal
    pub confidence_threshold: f32,
    /// Whether to use asymmetric thresholds (tighter for SELL)
    pub asymmetric: bool,
}

impl Default for SignalConfig {
    fn default() -> Self {
        Self {
            bullish_threshold: 0.6,
            bearish_threshold: 0.6,
            confidence_threshold: 0.5,
            asymmetric: false,
        }
    }
}

/// Trading signal generator.
///
/// Converts sentiment predictions to actionable trading signals.
pub struct TradingSignalGenerator {
    /// Configuration
    config: SignalConfig,
}

impl TradingSignalGenerator {
    /// Create a new signal generator with default configuration.
    pub fn new() -> Self {
        Self {
            config: SignalConfig::default(),
        }
    }

    /// Create with custom configuration.
    pub fn with_config(config: SignalConfig) -> Self {
        Self { config }
    }

    /// Generate a trading signal from sentiment.
    ///
    /// # Arguments
    ///
    /// * `sentiment` - The sentiment (Bullish, Neutral, Bearish)
    /// * `confidence` - Confidence score (0.0 to 1.0)
    ///
    /// # Returns
    ///
    /// A `TradingSignal` with the appropriate action.
    pub fn generate_signal(&self, sentiment: Sentiment, confidence: f32) -> TradingSignal {
        let signal = if confidence < self.config.confidence_threshold {
            SignalType::Hold
        } else {
            match sentiment {
                Sentiment::Bullish if confidence >= self.config.bullish_threshold => SignalType::Buy,
                Sentiment::Bearish if confidence >= self.config.bearish_threshold => SignalType::Sell,
                _ => SignalType::Hold,
            }
        };

        TradingSignal::new(signal, confidence, sentiment)
    }

    /// Generate signal from probability distribution.
    ///
    /// # Arguments
    ///
    /// * `probs` - Probabilities for [bullish, neutral, bearish]
    pub fn generate_from_probs(&self, probs: &[f32; 3]) -> TradingSignal {
        let max_idx = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(1);

        let sentiment = match max_idx {
            0 => Sentiment::Bullish,
            2 => Sentiment::Bearish,
            _ => Sentiment::Neutral,
        };

        let confidence = probs[max_idx];

        self.generate_signal(sentiment, confidence)
    }

    /// Aggregate multiple signals.
    ///
    /// # Arguments
    ///
    /// * `signals` - Vector of signals to aggregate
    /// * `method` - Aggregation method ("majority", "weighted", "unanimous")
    pub fn aggregate_signals(&self, signals: &[TradingSignal], method: &str) -> TradingSignal {
        if signals.is_empty() {
            return TradingSignal::new(SignalType::Hold, 0.0, Sentiment::Neutral);
        }

        match method {
            "majority" => self.majority_vote(signals),
            "weighted" => self.confidence_weighted(signals),
            "unanimous" => self.unanimous(signals),
            _ => self.majority_vote(signals),
        }
    }

    /// Majority vote aggregation.
    fn majority_vote(&self, signals: &[TradingSignal]) -> TradingSignal {
        let mut counts: HashMap<SignalType, usize> = HashMap::new();

        for signal in signals {
            *counts.entry(signal.signal).or_insert(0) += 1;
        }

        let (best_signal, count) = counts
            .into_iter()
            .max_by_key(|(_, c)| *c)
            .unwrap_or((SignalType::Hold, 0));

        let confidence = count as f32 / signals.len() as f32;

        // Find corresponding sentiment
        let sentiment = signals
            .iter()
            .find(|s| s.signal == best_signal)
            .map(|s| s.sentiment)
            .unwrap_or(Sentiment::Neutral);

        TradingSignal::new(best_signal, confidence, sentiment)
    }

    /// Confidence-weighted aggregation.
    fn confidence_weighted(&self, signals: &[TradingSignal]) -> TradingSignal {
        let mut weights: HashMap<SignalType, f32> = HashMap::new();

        for signal in signals {
            *weights.entry(signal.signal).or_insert(0.0) += signal.confidence;
        }

        let total: f32 = weights.values().sum();

        let (best_signal, weight) = weights
            .into_iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap_or((SignalType::Hold, 0.0));

        let confidence = if total > 0.0 { weight / total } else { 0.0 };

        let sentiment = signals
            .iter()
            .find(|s| s.signal == best_signal)
            .map(|s| s.sentiment)
            .unwrap_or(Sentiment::Neutral);

        TradingSignal::new(best_signal, confidence, sentiment)
    }

    /// Unanimous aggregation (all must agree).
    fn unanimous(&self, signals: &[TradingSignal]) -> TradingSignal {
        if signals.is_empty() {
            return TradingSignal::new(SignalType::Hold, 0.0, Sentiment::Neutral);
        }

        let first = signals[0].signal;
        let unanimous = signals.iter().all(|s| s.signal == first);

        if unanimous {
            let avg_confidence: f32 =
                signals.iter().map(|s| s.confidence).sum::<f32>() / signals.len() as f32;
            TradingSignal::new(first, avg_confidence, signals[0].sentiment)
        } else {
            TradingSignal::new(SignalType::Hold, 0.5, Sentiment::Neutral)
        }
    }
}

impl Default for TradingSignalGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// Crypto-specific signal generator.
///
/// Includes additional factors like funding rate and volatility.
pub struct CryptoSignalGenerator {
    /// Base signal generator
    base_generator: TradingSignalGenerator,
    /// Impact of funding rate on signals
    funding_rate_impact: f32,
    /// Whether to adjust for volatility
    volatility_adjustment: bool,
}

impl CryptoSignalGenerator {
    /// Create a new crypto signal generator.
    pub fn new() -> Self {
        Self {
            base_generator: TradingSignalGenerator::new(),
            funding_rate_impact: 0.1,
            volatility_adjustment: true,
        }
    }

    /// Set funding rate impact.
    pub fn with_funding_impact(mut self, impact: f32) -> Self {
        self.funding_rate_impact = impact.clamp(0.0, 1.0);
        self
    }

    /// Enable/disable volatility adjustment.
    pub fn with_volatility_adjustment(mut self, enabled: bool) -> Self {
        self.volatility_adjustment = enabled;
        self
    }

    /// Generate signal with crypto-specific factors.
    ///
    /// # Arguments
    ///
    /// * `sentiment` - Base sentiment
    /// * `confidence` - Confidence score
    /// * `funding_rate` - Current funding rate (positive = longs pay shorts)
    /// * `volatility` - Current volatility (annualized)
    pub fn generate_signal(
        &self,
        sentiment: Sentiment,
        confidence: f32,
        funding_rate: f64,
        volatility: f64,
    ) -> TradingSignal {
        // Get base signal
        let mut signal = self.base_generator.generate_signal(sentiment, confidence);

        // Adjust for funding rate
        // High positive funding = expensive to hold longs
        // High negative funding = expensive to hold shorts
        let funding_adjustment = -funding_rate as f32 * self.funding_rate_impact * 100.0;

        let adjusted_confidence = match signal.signal {
            SignalType::Buy => (signal.confidence + funding_adjustment).clamp(0.0, 1.0),
            SignalType::Sell => (signal.confidence - funding_adjustment).clamp(0.0, 1.0),
            SignalType::Hold => signal.confidence,
        };

        // Adjust for volatility
        let volatility_factor = if self.volatility_adjustment {
            // Reduce confidence in high volatility environments
            let base_vol = 0.5; // 50% annualized as baseline
            let vol_ratio = (volatility as f32 / base_vol).clamp(0.5, 2.0);
            1.0 / vol_ratio
        } else {
            1.0
        };

        signal.confidence = (adjusted_confidence * volatility_factor).clamp(0.0, 1.0);

        // Add metadata
        signal.metadata.insert("funding_rate".to_string(), format!("{:.4}", funding_rate));
        signal.metadata.insert("volatility".to_string(), format!("{:.2}", volatility));
        signal.metadata.insert("funding_adjustment".to_string(), format!("{:.4}", funding_adjustment));

        signal
    }
}

impl Default for CryptoSignalGenerator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_generation() {
        let generator = TradingSignalGenerator::new();

        let signal = generator.generate_signal(Sentiment::Bullish, 0.8);
        assert_eq!(signal.signal, SignalType::Buy);

        let signal = generator.generate_signal(Sentiment::Bearish, 0.7);
        assert_eq!(signal.signal, SignalType::Sell);

        let signal = generator.generate_signal(Sentiment::Neutral, 0.9);
        assert_eq!(signal.signal, SignalType::Hold);
    }

    #[test]
    fn test_low_confidence_hold() {
        let generator = TradingSignalGenerator::new();

        let signal = generator.generate_signal(Sentiment::Bullish, 0.3);
        assert_eq!(signal.signal, SignalType::Hold);
    }

    #[test]
    fn test_prob_generation() {
        let generator = TradingSignalGenerator::new();

        let signal = generator.generate_from_probs(&[0.8, 0.1, 0.1]);
        assert_eq!(signal.signal, SignalType::Buy);

        let signal = generator.generate_from_probs(&[0.1, 0.1, 0.8]);
        assert_eq!(signal.signal, SignalType::Sell);
    }

    #[test]
    fn test_aggregation() {
        let generator = TradingSignalGenerator::new();

        let signals = vec![
            TradingSignal::new(SignalType::Buy, 0.8, Sentiment::Bullish),
            TradingSignal::new(SignalType::Buy, 0.7, Sentiment::Bullish),
            TradingSignal::new(SignalType::Sell, 0.6, Sentiment::Bearish),
        ];

        let aggregated = generator.aggregate_signals(&signals, "majority");
        assert_eq!(aggregated.signal, SignalType::Buy);
    }

    #[test]
    fn test_crypto_signal() {
        let generator = CryptoSignalGenerator::new();

        // High positive funding should reduce BUY confidence
        let signal = generator.generate_signal(Sentiment::Bullish, 0.8, 0.001, 0.5);
        assert!(signal.confidence < 0.8);

        // High negative funding should increase BUY confidence
        let signal = generator.generate_signal(Sentiment::Bullish, 0.8, -0.001, 0.5);
        assert!(signal.confidence > 0.75);
    }

    #[test]
    fn test_sentiment_parsing() {
        assert_eq!(Sentiment::from_str("bullish"), Some(Sentiment::Bullish));
        assert_eq!(Sentiment::from_str("BEARISH"), Some(Sentiment::Bearish));
        assert_eq!(Sentiment::from_str("neutral"), Some(Sentiment::Neutral));
        assert_eq!(Sentiment::from_str("invalid"), None);
    }
}
