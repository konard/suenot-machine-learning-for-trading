//! Trading signal generation from multi-task model outputs

use crate::model::FusedPrediction;
use serde::{Deserialize, Serialize};

/// Type of trading signal
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SignalType {
    /// Strong buy recommendation
    StrongBuy,
    /// Moderate buy recommendation
    Buy,
    /// Hold / no action
    Hold,
    /// Moderate sell recommendation
    Sell,
    /// Strong sell recommendation
    StrongSell,
}

impl SignalType {
    /// Convert to position direction (-1, 0, +1)
    pub fn direction(&self) -> f64 {
        match self {
            SignalType::StrongBuy => 1.0,
            SignalType::Buy => 1.0,
            SignalType::Hold => 0.0,
            SignalType::Sell => -1.0,
            SignalType::StrongSell => -1.0,
        }
    }

    /// Convert to position sizing (0 to 1)
    pub fn sizing(&self) -> f64 {
        match self {
            SignalType::StrongBuy => 1.0,
            SignalType::Buy => 0.5,
            SignalType::Hold => 0.0,
            SignalType::Sell => 0.5,
            SignalType::StrongSell => 1.0,
        }
    }

    /// Human-readable label
    pub fn label(&self) -> &str {
        match self {
            SignalType::StrongBuy => "Strong Buy",
            SignalType::Buy => "Buy",
            SignalType::Hold => "Hold",
            SignalType::Sell => "Sell",
            SignalType::StrongSell => "Strong Sell",
        }
    }
}

/// A trading signal with metadata
#[derive(Debug, Clone)]
pub struct TradingSignal {
    /// Signal type
    pub signal_type: SignalType,
    /// Raw signal value (-1.0 to 1.0)
    pub raw_signal: f64,
    /// Confidence (0.0 to 1.0)
    pub confidence: f64,
    /// Detected regime
    pub regime: String,
    /// Risk level
    pub risk_level: f64,
    /// Recommended position size (0.0 to 1.0)
    pub position_size: f64,
    /// Task contribution breakdown
    pub contributions: Vec<(String, f64)>,
}

/// Configuration for signal generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalConfig {
    /// Threshold for buy signal
    pub buy_threshold: f64,
    /// Threshold for strong buy signal
    pub strong_buy_threshold: f64,
    /// Threshold for sell signal
    pub sell_threshold: f64,
    /// Threshold for strong sell signal
    pub strong_sell_threshold: f64,
    /// Maximum position size (0.0 to 1.0)
    pub max_position_size: f64,
    /// Minimum confidence for signal
    pub min_confidence: f64,
    /// Risk scaling factor
    pub risk_scaling: f64,
}

impl Default for SignalConfig {
    fn default() -> Self {
        Self {
            buy_threshold: 0.15,
            strong_buy_threshold: 0.40,
            sell_threshold: -0.15,
            strong_sell_threshold: -0.40,
            max_position_size: 1.0,
            min_confidence: 0.5,
            risk_scaling: 0.5,
        }
    }
}

/// Signal generator that converts model outputs to trading signals
pub struct SignalGenerator {
    config: SignalConfig,
}

impl SignalGenerator {
    /// Create a new signal generator
    pub fn new(config: SignalConfig) -> Self {
        Self { config }
    }

    /// Generate trading signals from fused predictions
    pub fn generate(&self, prediction: &FusedPrediction) -> Vec<TradingSignal> {
        prediction
            .signal
            .iter()
            .enumerate()
            .map(|(i, &raw_signal)| {
                let confidence = prediction.confidence.get(i).copied().unwrap_or(0.0);
                let regime = prediction
                    .regime
                    .get(i)
                    .cloned()
                    .unwrap_or_else(|| "Unknown".to_string());
                let risk_level = prediction.risk_level.get(i).copied().unwrap_or(0.5);

                // Determine signal type
                let signal_type = if confidence < self.config.min_confidence {
                    SignalType::Hold
                } else if raw_signal >= self.config.strong_buy_threshold {
                    SignalType::StrongBuy
                } else if raw_signal >= self.config.buy_threshold {
                    SignalType::Buy
                } else if raw_signal <= self.config.strong_sell_threshold {
                    SignalType::StrongSell
                } else if raw_signal <= self.config.sell_threshold {
                    SignalType::Sell
                } else {
                    SignalType::Hold
                };

                // Calculate position size
                let base_size = signal_type.sizing();
                let risk_adjustment = 1.0 - risk_level * self.config.risk_scaling;
                let confidence_adjustment = confidence;
                let position_size = (base_size * risk_adjustment * confidence_adjustment)
                    .clamp(0.0, self.config.max_position_size);

                TradingSignal {
                    signal_type,
                    raw_signal,
                    confidence,
                    regime,
                    risk_level,
                    position_size,
                    contributions: prediction.task_contributions.clone(),
                }
            })
            .collect()
    }

    /// Generate a single signal from raw values
    pub fn generate_single(
        &self,
        raw_signal: f64,
        confidence: f64,
        regime: &str,
        risk_level: f64,
    ) -> TradingSignal {
        let signal_type = if confidence < self.config.min_confidence {
            SignalType::Hold
        } else if raw_signal >= self.config.strong_buy_threshold {
            SignalType::StrongBuy
        } else if raw_signal >= self.config.buy_threshold {
            SignalType::Buy
        } else if raw_signal <= self.config.strong_sell_threshold {
            SignalType::StrongSell
        } else if raw_signal <= self.config.sell_threshold {
            SignalType::Sell
        } else {
            SignalType::Hold
        };

        let base_size = signal_type.sizing();
        let risk_adjustment = 1.0 - risk_level * self.config.risk_scaling;
        let position_size = (base_size * risk_adjustment * confidence)
            .clamp(0.0, self.config.max_position_size);

        TradingSignal {
            signal_type,
            raw_signal,
            confidence,
            regime: regime.to_string(),
            risk_level,
            position_size,
            contributions: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_generation() {
        let gen = SignalGenerator::new(SignalConfig::default());

        // Strong buy signal
        let signal = gen.generate_single(0.5, 0.8, "Trending", 0.2);
        assert_eq!(signal.signal_type, SignalType::StrongBuy);
        assert!(signal.position_size > 0.0);

        // Hold signal (low confidence)
        let signal = gen.generate_single(0.5, 0.3, "Calm", 0.1);
        assert_eq!(signal.signal_type, SignalType::Hold);
        assert_eq!(signal.position_size, 0.0);
    }

    #[test]
    fn test_risk_scaling() {
        let gen = SignalGenerator::new(SignalConfig::default());

        // Same signal, different risk levels
        let low_risk = gen.generate_single(0.5, 0.8, "Trending", 0.1);
        let high_risk = gen.generate_single(0.5, 0.8, "Volatile", 0.9);

        assert!(low_risk.position_size > high_risk.position_size);
    }
}
