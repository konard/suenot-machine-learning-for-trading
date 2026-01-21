//! Trading signal generation based on few-shot predictions.
//!
//! This module converts few-shot learning predictions into actionable
//! trading signals with confidence-weighted position sizing.

use crate::strategy::regime::{MarketRegime, RegimeClassification};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// Type of trading signal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SignalType {
    /// Open or add to long position
    Long,
    /// Open or add to short position
    Short,
    /// Close existing position
    Close,
    /// Reduce position size
    ReducePosition,
    /// No action recommended
    Hold,
}

impl SignalType {
    /// Get the direction multiplier for this signal.
    pub fn direction(&self) -> f64 {
        match self {
            SignalType::Long => 1.0,
            SignalType::Short => -1.0,
            SignalType::Close => 0.0,
            SignalType::ReducePosition => 0.0,
            SignalType::Hold => 0.0,
        }
    }

    /// Check if this signal requires action.
    pub fn requires_action(&self) -> bool {
        !matches!(self, SignalType::Hold)
    }
}

/// A trading signal with associated metadata.
#[derive(Debug, Clone)]
pub struct TradingSignal {
    /// Type of signal
    pub signal_type: SignalType,
    /// Confidence in the signal (0-1)
    pub confidence: f64,
    /// Recommended position size as fraction of capital (0-1)
    pub position_size: f64,
    /// Take profit level (percentage)
    pub take_profit: Option<f64>,
    /// Stop loss level (percentage)
    pub stop_loss: Option<f64>,
    /// Asset symbol this signal is for
    pub symbol: String,
    /// Market regime that generated this signal
    pub regime: Option<MarketRegime>,
    /// Signal generation timestamp
    pub timestamp: Instant,
    /// Time-to-live for this signal
    pub ttl: Duration,
    /// Additional metadata
    pub metadata: SignalMetadata,
}

impl TradingSignal {
    /// Create a new trading signal.
    pub fn new(signal_type: SignalType, confidence: f64, symbol: &str) -> Self {
        Self {
            signal_type,
            confidence: confidence.clamp(0.0, 1.0),
            position_size: confidence.clamp(0.0, 1.0) * 0.5, // Max 50% of capital
            take_profit: None,
            stop_loss: None,
            symbol: symbol.to_string(),
            regime: None,
            timestamp: Instant::now(),
            ttl: Duration::from_secs(300), // 5 minute default TTL
            metadata: SignalMetadata::default(),
        }
    }

    /// Create a hold signal.
    pub fn hold(symbol: &str) -> Self {
        Self::new(SignalType::Hold, 0.0, symbol)
    }

    /// Set take profit level.
    pub fn with_take_profit(mut self, tp_percent: f64) -> Self {
        self.take_profit = Some(tp_percent);
        self
    }

    /// Set stop loss level.
    pub fn with_stop_loss(mut self, sl_percent: f64) -> Self {
        self.stop_loss = Some(sl_percent);
        self
    }

    /// Set the regime.
    pub fn with_regime(mut self, regime: MarketRegime) -> Self {
        self.regime = Some(regime);
        self
    }

    /// Set time-to-live.
    pub fn with_ttl(mut self, ttl: Duration) -> Self {
        self.ttl = ttl;
        self
    }

    /// Check if the signal has expired.
    pub fn is_expired(&self) -> bool {
        self.timestamp.elapsed() > self.ttl
    }

    /// Check if the signal is strong enough to act on.
    pub fn is_actionable(&self) -> bool {
        self.signal_type.requires_action()
            && self.confidence >= 0.5
            && !self.is_expired()
    }

    /// Calculate expected value based on confidence and risk/reward.
    pub fn expected_value(&self) -> f64 {
        let tp = self.take_profit.unwrap_or(2.0);
        let sl = self.stop_loss.unwrap_or(1.0);

        // Expected value = (win_prob * win_amount) - (lose_prob * lose_amount)
        let win_prob = self.confidence;
        let lose_prob = 1.0 - self.confidence;

        (win_prob * tp) - (lose_prob * sl)
    }
}

/// Additional metadata for trading signals.
#[derive(Debug, Clone, Default)]
pub struct SignalMetadata {
    /// Number of support examples used
    pub support_count: usize,
    /// Method used for prediction
    pub method: String,
    /// Feature importance scores
    pub feature_importance: Vec<(String, f64)>,
    /// Raw prediction scores
    pub raw_scores: Vec<f64>,
}

/// Configuration for signal generation.
#[derive(Debug, Clone)]
pub struct SignalConfig {
    /// Minimum confidence to generate a signal
    pub min_confidence: f64,
    /// Maximum position size (fraction of capital)
    pub max_position_size: f64,
    /// Default take profit percentage
    pub default_take_profit: f64,
    /// Default stop loss percentage
    pub default_stop_loss: f64,
    /// Enable regime-based adjustments
    pub use_regime_adjustment: bool,
    /// Signal cooldown period
    pub cooldown: Duration,
}

impl Default for SignalConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.6,
            max_position_size: 0.25,
            default_take_profit: 2.0,
            default_stop_loss: 1.0,
            use_regime_adjustment: true,
            cooldown: Duration::from_secs(60),
        }
    }
}

/// Generator for trading signals based on few-shot predictions.
pub struct SignalGenerator {
    /// Configuration
    config: SignalConfig,
    /// Last signal timestamp per symbol
    last_signals: std::collections::HashMap<String, Instant>,
}

impl SignalGenerator {
    /// Create a new signal generator.
    pub fn new(config: SignalConfig) -> Self {
        Self {
            config,
            last_signals: std::collections::HashMap::new(),
        }
    }

    /// Generate a trading signal from regime classification.
    pub fn generate(&mut self, symbol: &str, classification: &RegimeClassification) -> TradingSignal {
        // Check cooldown
        if let Some(&last) = self.last_signals.get(symbol) {
            if last.elapsed() < self.config.cooldown {
                return TradingSignal::hold(symbol);
            }
        }

        // Check confidence threshold
        if classification.confidence < self.config.min_confidence {
            return TradingSignal::hold(symbol);
        }

        // Determine signal type based on regime
        let signal_type = self.regime_to_signal(classification.regime);

        if signal_type == SignalType::Hold {
            return TradingSignal::hold(symbol);
        }

        // Calculate position size based on confidence
        let base_size = classification.confidence * self.config.max_position_size;
        let position_size = if self.config.use_regime_adjustment {
            self.adjust_for_regime(base_size, classification.regime)
        } else {
            base_size
        };

        // Calculate risk parameters
        let (take_profit, stop_loss) = self.calculate_risk_params(classification);

        // Update last signal time
        self.last_signals.insert(symbol.to_string(), Instant::now());

        let mut signal = TradingSignal::new(signal_type, classification.confidence, symbol);
        signal.position_size = position_size;
        signal.take_profit = Some(take_profit);
        signal.stop_loss = Some(stop_loss);
        signal.regime = Some(classification.regime);
        signal.metadata.method = "few_shot".to_string();

        signal
    }

    /// Generate signal from raw class probabilities.
    pub fn generate_from_probs(
        &mut self,
        symbol: &str,
        class_probs: &[f64],
        class_labels: &[MarketRegime],
    ) -> TradingSignal {
        if class_probs.len() != class_labels.len() || class_probs.is_empty() {
            return TradingSignal::hold(symbol);
        }

        // Find best class
        let (best_idx, &best_prob) = class_probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        let best_regime = class_labels[best_idx];

        // Check confidence
        if best_prob < self.config.min_confidence {
            return TradingSignal::hold(symbol);
        }

        let signal_type = self.regime_to_signal(best_regime);

        if signal_type == SignalType::Hold {
            return TradingSignal::hold(symbol);
        }

        let position_size = best_prob * self.config.max_position_size;

        let mut signal = TradingSignal::new(signal_type, best_prob, symbol);
        signal.position_size = position_size;
        signal.take_profit = Some(self.config.default_take_profit);
        signal.stop_loss = Some(self.config.default_stop_loss);
        signal.regime = Some(best_regime);
        signal.metadata.raw_scores = class_probs.to_vec();

        signal
    }

    /// Convert market regime to signal type.
    fn regime_to_signal(&self, regime: MarketRegime) -> SignalType {
        match regime {
            MarketRegime::StrongUptrend => SignalType::Long,
            MarketRegime::WeakUptrend => SignalType::Long,
            MarketRegime::Recovery => SignalType::Long,
            MarketRegime::StrongDowntrend => SignalType::Short,
            MarketRegime::WeakDowntrend => SignalType::Short,
            MarketRegime::Crash => SignalType::Close, // Close positions during crash
            MarketRegime::Volatile => SignalType::ReducePosition,
            MarketRegime::Sideways => SignalType::Hold,
        }
    }

    /// Adjust position size based on regime characteristics.
    fn adjust_for_regime(&self, base_size: f64, regime: MarketRegime) -> f64 {
        let multiplier = match regime {
            MarketRegime::StrongUptrend => 1.0,
            MarketRegime::WeakUptrend => 0.7,
            MarketRegime::Sideways => 0.3,
            MarketRegime::WeakDowntrend => 0.7,
            MarketRegime::StrongDowntrend => 1.0,
            MarketRegime::Volatile => 0.3, // Reduce in volatile markets
            MarketRegime::Crash => 0.2,
            MarketRegime::Recovery => 0.8,
        };

        (base_size * multiplier).min(self.config.max_position_size)
    }

    /// Calculate take profit and stop loss based on regime.
    fn calculate_risk_params(&self, classification: &RegimeClassification) -> (f64, f64) {
        let base_tp = self.config.default_take_profit;
        let base_sl = self.config.default_stop_loss;

        // Adjust based on regime volatility expectations
        let (tp_mult, sl_mult) = match classification.regime {
            MarketRegime::StrongUptrend | MarketRegime::StrongDowntrend => (1.5, 0.8),
            MarketRegime::WeakUptrend | MarketRegime::WeakDowntrend => (1.0, 1.0),
            MarketRegime::Volatile => (2.0, 1.5), // Wider stops in volatile markets
            MarketRegime::Crash => (1.0, 2.0), // Tight TP, wide SL during crash
            MarketRegime::Recovery => (1.5, 1.0),
            MarketRegime::Sideways => (0.8, 0.8), // Tighter in sideways
        };

        // Adjust for uncertainty
        let uncertainty_factor = 1.0 + classification.uncertainty * 0.5;

        (base_tp * tp_mult, base_sl * sl_mult * uncertainty_factor)
    }

    /// Reset cooldown for a symbol.
    pub fn reset_cooldown(&mut self, symbol: &str) {
        self.last_signals.remove(symbol);
    }

    /// Clear all cooldowns.
    pub fn clear_cooldowns(&mut self) {
        self.last_signals.clear();
    }
}

impl Default for SignalGenerator {
    fn default() -> Self {
        Self::new(SignalConfig::default())
    }
}

/// Aggregates multiple signals into a final decision.
pub struct SignalAggregator {
    /// Minimum number of signals to aggregate
    min_signals: usize,
    /// Weight for each signal source
    weights: Vec<f64>,
}

impl SignalAggregator {
    /// Create a new aggregator.
    pub fn new(min_signals: usize) -> Self {
        Self {
            min_signals,
            weights: Vec::new(),
        }
    }

    /// Set weights for signal sources.
    pub fn with_weights(mut self, weights: Vec<f64>) -> Self {
        self.weights = weights;
        self
    }

    /// Aggregate multiple signals.
    pub fn aggregate(&self, signals: &[TradingSignal]) -> Option<TradingSignal> {
        if signals.len() < self.min_signals {
            return None;
        }

        // Filter to actionable signals
        let actionable: Vec<_> = signals.iter().filter(|s| s.is_actionable()).collect();

        if actionable.is_empty() {
            return signals.first().map(|s| TradingSignal::hold(&s.symbol));
        }

        // Weighted vote for signal type
        let mut long_score = 0.0;
        let mut short_score = 0.0;
        let mut close_score = 0.0;

        for (i, signal) in actionable.iter().enumerate() {
            let weight = self.weights.get(i).copied().unwrap_or(1.0);
            let score = signal.confidence * weight;

            match signal.signal_type {
                SignalType::Long => long_score += score,
                SignalType::Short => short_score += score,
                SignalType::Close | SignalType::ReducePosition => close_score += score,
                SignalType::Hold => {}
            }
        }

        let total = long_score + short_score + close_score;
        if total < 0.01 {
            return signals.first().map(|s| TradingSignal::hold(&s.symbol));
        }

        // Determine winning signal type
        let (signal_type, confidence) = if long_score > short_score && long_score > close_score {
            (SignalType::Long, long_score / total)
        } else if short_score > long_score && short_score > close_score {
            (SignalType::Short, short_score / total)
        } else {
            (SignalType::Close, close_score / total)
        };

        let symbol = &actionable[0].symbol;
        let mut result = TradingSignal::new(signal_type, confidence, symbol);

        // Average position size from agreeing signals
        let agreeing: Vec<_> = actionable
            .iter()
            .filter(|s| s.signal_type == signal_type)
            .collect();

        if !agreeing.is_empty() {
            result.position_size = agreeing.iter().map(|s| s.position_size).sum::<f64>() / agreeing.len() as f64;
            result.take_profit = agreeing.iter().filter_map(|s| s.take_profit).next();
            result.stop_loss = agreeing.iter().filter_map(|s| s.stop_loss).next();
            result.regime = agreeing.iter().filter_map(|s| s.regime).next();
        }

        Some(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_creation() {
        let signal = TradingSignal::new(SignalType::Long, 0.8, "BTCUSDT");

        assert_eq!(signal.signal_type, SignalType::Long);
        assert!((signal.confidence - 0.8).abs() < 0.001);
        assert!(signal.is_actionable());
    }

    #[test]
    fn test_signal_expected_value() {
        let signal = TradingSignal::new(SignalType::Long, 0.7, "BTCUSDT")
            .with_take_profit(3.0)
            .with_stop_loss(1.0);

        // EV = 0.7 * 3.0 - 0.3 * 1.0 = 2.1 - 0.3 = 1.8
        let ev = signal.expected_value();
        assert!((ev - 1.8).abs() < 0.001);
    }

    #[test]
    fn test_signal_generator() {
        let mut generator = SignalGenerator::default();

        let classification = RegimeClassification {
            regime: MarketRegime::StrongUptrend,
            confidence: 0.85,
            uncertainty: 0.1,
            second_best: Some(MarketRegime::WeakUptrend),
            is_confident: true,
        };

        let signal = generator.generate("BTCUSDT", &classification);

        assert_eq!(signal.signal_type, SignalType::Long);
        assert!(signal.confidence > 0.8);
    }

    #[test]
    fn test_hold_on_low_confidence() {
        let mut generator = SignalGenerator::default();

        let classification = RegimeClassification {
            regime: MarketRegime::StrongUptrend,
            confidence: 0.3, // Below threshold
            uncertainty: 0.5,
            second_best: None,
            is_confident: false,
        };

        let signal = generator.generate("BTCUSDT", &classification);

        assert_eq!(signal.signal_type, SignalType::Hold);
    }
}
