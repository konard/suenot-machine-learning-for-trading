//! # Signal Generator
//!
//! Trading signal generation from model predictions and order flow.

use crate::data::snapshot::FeatureVector;
use crate::models::gradient_boosting::GradientBoostingModel;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Trading signal
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Signal {
    /// Buy signal
    Long,
    /// Sell signal
    Short,
    /// No signal
    Hold,
    /// Close existing position
    Exit,
}

/// Signal with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingSignal {
    /// Signal type
    pub signal: Signal,
    /// Confidence (0 to 1)
    pub confidence: f64,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Expected return
    pub expected_return: f64,
    /// Stop loss price
    pub stop_loss: Option<f64>,
    /// Take profit price
    pub take_profit: Option<f64>,
}

/// Signal generator configuration
#[derive(Debug, Clone)]
pub struct SignalConfig {
    /// Probability threshold for long
    pub prob_threshold_long: f64,
    /// Probability threshold for short
    pub prob_threshold_short: f64,
    /// OFI threshold (z-score)
    pub ofi_threshold: f64,
    /// Maximum spread (bps)
    pub max_spread_bps: f64,
    /// Minimum confidence
    pub min_confidence: f64,
    /// Stop loss (% of price)
    pub stop_loss_pct: f64,
    /// Take profit (% of price)
    pub take_profit_pct: f64,
}

impl Default for SignalConfig {
    fn default() -> Self {
        Self {
            prob_threshold_long: 0.55,
            prob_threshold_short: 0.45,
            ofi_threshold: 1.5,
            max_spread_bps: 10.0,
            min_confidence: 0.1,
            stop_loss_pct: 0.15,
            take_profit_pct: 0.25,
        }
    }
}

/// Signal generator
pub struct SignalGenerator {
    /// Model for predictions
    model: Option<GradientBoostingModel>,
    /// Configuration
    config: SignalConfig,
    /// Last signal
    last_signal: Option<TradingSignal>,
    /// Cooldown (ms since last signal)
    cooldown_ms: u64,
    /// Last signal timestamp
    last_signal_time: Option<DateTime<Utc>>,
}

impl SignalGenerator {
    /// Create new generator
    pub fn new(config: SignalConfig) -> Self {
        Self {
            model: None,
            config,
            last_signal: None,
            cooldown_ms: 1000, // 1 second cooldown
            last_signal_time: None,
        }
    }

    /// Set the model
    pub fn set_model(&mut self, model: GradientBoostingModel) {
        self.model = Some(model);
    }

    /// Generate signal from features
    pub fn generate(
        &mut self,
        features: &FeatureVector,
        current_price: f64,
        spread_bps: f64,
    ) -> TradingSignal {
        let timestamp = features.timestamp;

        // Check cooldown
        if let Some(last_time) = self.last_signal_time {
            let elapsed = (timestamp - last_time).num_milliseconds() as u64;
            if elapsed < self.cooldown_ms {
                return TradingSignal {
                    signal: Signal::Hold,
                    confidence: 0.0,
                    timestamp,
                    expected_return: 0.0,
                    stop_loss: None,
                    take_profit: None,
                };
            }
        }

        // Check spread
        if spread_bps > self.config.max_spread_bps {
            return TradingSignal {
                signal: Signal::Hold,
                confidence: 0.0,
                timestamp,
                expected_return: 0.0,
                stop_loss: None,
                take_profit: None,
            };
        }

        // Get prediction
        let probability = if let Some(model) = &self.model {
            model.predict_proba(features)
        } else {
            0.5
        };

        // Get OFI z-score
        let ofi_zscore = features.get("ofi_zscore").unwrap_or(0.0);

        // Generate signal
        let (signal, confidence) = self.determine_signal(probability, ofi_zscore);

        // Skip low confidence signals
        if confidence < self.config.min_confidence {
            return TradingSignal {
                signal: Signal::Hold,
                confidence: 0.0,
                timestamp,
                expected_return: 0.0,
                stop_loss: None,
                take_profit: None,
            };
        }

        // Calculate expected return
        let expected_return = match signal {
            Signal::Long => (probability - 0.5) * 2.0 * 0.1, // Scale to rough %
            Signal::Short => (0.5 - probability) * 2.0 * 0.1,
            _ => 0.0,
        };

        // Calculate stop/take profit
        let (stop_loss, take_profit) = match signal {
            Signal::Long => (
                Some(current_price * (1.0 - self.config.stop_loss_pct / 100.0)),
                Some(current_price * (1.0 + self.config.take_profit_pct / 100.0)),
            ),
            Signal::Short => (
                Some(current_price * (1.0 + self.config.stop_loss_pct / 100.0)),
                Some(current_price * (1.0 - self.config.take_profit_pct / 100.0)),
            ),
            _ => (None, None),
        };

        let trading_signal = TradingSignal {
            signal,
            confidence,
            timestamp,
            expected_return,
            stop_loss,
            take_profit,
        };

        if signal != Signal::Hold {
            self.last_signal = Some(trading_signal.clone());
            self.last_signal_time = Some(timestamp);
        }

        trading_signal
    }

    /// Determine signal from probability and OFI
    fn determine_signal(&self, probability: f64, ofi_zscore: f64) -> (Signal, f64) {
        // Long signal
        if probability > self.config.prob_threshold_long && ofi_zscore > self.config.ofi_threshold {
            let confidence = (probability - 0.5) * 2.0;
            return (Signal::Long, confidence);
        }

        // Short signal
        if probability < self.config.prob_threshold_short && ofi_zscore < -self.config.ofi_threshold
        {
            let confidence = (0.5 - probability) * 2.0;
            return (Signal::Short, confidence);
        }

        (Signal::Hold, 0.0)
    }

    /// Check if should exit position
    pub fn should_exit(
        &self,
        current_price: f64,
        entry_price: f64,
        is_long: bool,
        stop_loss: f64,
        take_profit: f64,
    ) -> bool {
        if is_long {
            current_price <= stop_loss || current_price >= take_profit
        } else {
            current_price >= stop_loss || current_price <= take_profit
        }
    }

    /// Get last signal
    pub fn last_signal(&self) -> Option<&TradingSignal> {
        self.last_signal.as_ref()
    }

    /// Set cooldown
    pub fn set_cooldown(&mut self, ms: u64) {
        self.cooldown_ms = ms;
    }
}

impl Default for SignalGenerator {
    fn default() -> Self {
        Self::new(SignalConfig::default())
    }
}

impl std::fmt::Display for Signal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Signal::Long => write!(f, "LONG"),
            Signal::Short => write!(f, "SHORT"),
            Signal::Hold => write!(f, "HOLD"),
            Signal::Exit => write!(f, "EXIT"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_generation() {
        let mut generator = SignalGenerator::new(SignalConfig::default());

        let mut features = FeatureVector::new(Utc::now());
        features.add("ofi_zscore", 2.0); // Strong buy pressure

        // Without model, should return hold
        let signal = generator.generate(&features, 100.0, 5.0);
        assert_eq!(signal.signal, Signal::Hold);
    }

    #[test]
    fn test_spread_filter() {
        let mut generator = SignalGenerator::new(SignalConfig {
            max_spread_bps: 5.0,
            ..Default::default()
        });

        let features = FeatureVector::new(Utc::now());

        // High spread should result in Hold
        let signal = generator.generate(&features, 100.0, 10.0);
        assert_eq!(signal.signal, Signal::Hold);
    }
}
