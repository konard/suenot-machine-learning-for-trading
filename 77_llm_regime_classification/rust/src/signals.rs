//! Trading signal generation from regime classification.
//!
//! Converts regime classifications into actionable trading signals
//! with position sizing and risk management.

use crate::classifier::{MarketRegime, RegimeResult};
use std::collections::HashMap;

/// Trading signal types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignalType {
    StrongBuy,
    Buy,
    Hold,
    Sell,
    StrongSell,
}

impl SignalType {
    /// Get string representation.
    pub fn as_str(&self) -> &'static str {
        match self {
            SignalType::StrongBuy => "strong_buy",
            SignalType::Buy => "buy",
            SignalType::Hold => "hold",
            SignalType::Sell => "sell",
            SignalType::StrongSell => "strong_sell",
        }
    }
}

/// Trading signal with position sizing.
#[derive(Debug, Clone)]
pub struct TradingSignal {
    /// Type of signal
    pub signal_type: SignalType,
    /// Position size (-1 to 1, negative = short)
    pub position_size: f64,
    /// Stop loss as decimal (e.g., 0.05 = 5%)
    pub stop_loss: Option<f64>,
    /// Take profit as decimal
    pub take_profit: Option<f64>,
    /// Confidence in signal
    pub confidence: f64,
    /// Underlying regime
    pub regime: MarketRegime,
    /// Reasoning for the signal
    pub reasoning: String,
}

impl TradingSignal {
    /// Create a new trading signal.
    pub fn new(
        signal_type: SignalType,
        position_size: f64,
        regime: MarketRegime,
        confidence: f64,
    ) -> Self {
        Self {
            signal_type,
            position_size,
            stop_loss: None,
            take_profit: None,
            confidence,
            regime,
            reasoning: String::new(),
        }
    }

    /// Set stop loss.
    pub fn with_stop_loss(mut self, stop_loss: f64) -> Self {
        self.stop_loss = Some(stop_loss);
        self
    }

    /// Set take profit.
    pub fn with_take_profit(mut self, take_profit: f64) -> Self {
        self.take_profit = Some(take_profit);
        self
    }

    /// Set reasoning.
    pub fn with_reasoning(mut self, reasoning: &str) -> Self {
        self.reasoning = reasoning.to_string();
        self
    }
}

/// Regime position configuration.
#[derive(Debug, Clone)]
pub struct RegimePosition {
    pub signal: SignalType,
    pub position: f64,
    pub stop_loss: f64,
    pub take_profit: f64,
}

/// Signal generator based on regime classification.
pub struct SignalGenerator {
    /// Regime to position mapping
    regime_positions: HashMap<MarketRegime, RegimePosition>,
    /// Minimum confidence threshold
    confidence_threshold: f64,
}

impl SignalGenerator {
    /// Create a new signal generator with custom positions.
    pub fn new(confidence_threshold: f64) -> Self {
        let mut positions = HashMap::new();

        positions.insert(
            MarketRegime::Bull,
            RegimePosition {
                signal: SignalType::Buy,
                position: 1.0,
                stop_loss: 0.05,
                take_profit: 0.15,
            },
        );

        positions.insert(
            MarketRegime::Bear,
            RegimePosition {
                signal: SignalType::Sell,
                position: -0.5,
                stop_loss: 0.03,
                take_profit: 0.10,
            },
        );

        positions.insert(
            MarketRegime::Sideways,
            RegimePosition {
                signal: SignalType::Hold,
                position: 0.3,
                stop_loss: 0.03,
                take_profit: 0.05,
            },
        );

        positions.insert(
            MarketRegime::HighVolatility,
            RegimePosition {
                signal: SignalType::Hold,
                position: 0.2,
                stop_loss: 0.02,
                take_profit: 0.04,
            },
        );

        positions.insert(
            MarketRegime::Crisis,
            RegimePosition {
                signal: SignalType::StrongSell,
                position: 0.0,
                stop_loss: 0.0,
                take_profit: 0.0,
            },
        );

        Self {
            regime_positions: positions,
            confidence_threshold,
        }
    }

    /// Generate trading signal from regime result.
    pub fn generate_signal(&self, regime_result: &RegimeResult) -> TradingSignal {
        let regime = regime_result.regime;
        let confidence = regime_result.confidence;

        // Get base parameters
        let params = self
            .regime_positions
            .get(&regime)
            .cloned()
            .unwrap_or(RegimePosition {
                signal: SignalType::Hold,
                position: 0.0,
                stop_loss: 0.03,
                take_profit: 0.05,
            });

        // Adjust position by confidence
        let (adjusted_position, signal_type) = if confidence < self.confidence_threshold {
            (params.position * 0.5, SignalType::Hold)
        } else {
            (params.position * confidence, params.signal)
        };

        let reasoning = format!(
            "Regime: {}, Confidence: {:.1}%, Position: {:.1}%",
            regime.as_str(),
            confidence * 100.0,
            adjusted_position * 100.0
        );

        TradingSignal::new(signal_type, adjusted_position, regime, confidence)
            .with_stop_loss(params.stop_loss)
            .with_take_profit(params.take_profit)
            .with_reasoning(&reasoning)
    }

    /// Generate signals for a series of regime results.
    pub fn generate_signals(&self, results: &[RegimeResult]) -> Vec<TradingSignal> {
        results.iter().map(|r| self.generate_signal(r)).collect()
    }
}

impl Default for SignalGenerator {
    fn default() -> Self {
        Self::new(0.6)
    }
}

/// Position sizer based on risk management.
#[allow(dead_code)]
pub struct PositionSizer {
    /// Maximum position as fraction of capital
    max_position: f64,
    /// Target risk per trade (reserved for future risk-based sizing)
    risk_per_trade: f64,
    /// Whether to scale by volatility
    volatility_scaling: bool,
}

impl PositionSizer {
    /// Create a new position sizer.
    pub fn new(max_position: f64, risk_per_trade: f64, volatility_scaling: bool) -> Self {
        Self {
            max_position,
            risk_per_trade,
            volatility_scaling,
        }
    }

    /// Calculate position size.
    pub fn calculate_position(
        &self,
        signal: &TradingSignal,
        capital: f64,
        current_price: f64,
        volatility: f64,
    ) -> PositionResult {
        let mut base_position = signal.position_size * capital;

        // Volatility scaling
        if self.volatility_scaling && volatility > 0.0 {
            let target_vol = 0.15; // 15% portfolio vol
            let vol_scalar = (target_vol / volatility).clamp(0.5, 2.0);
            base_position *= vol_scalar;
        }

        // Apply maximum position constraint
        let max_dollars = self.max_position * capital;
        let position_dollars = base_position.clamp(-max_dollars, max_dollars);

        // Calculate units
        let position_units = if current_price > 0.0 {
            position_dollars / current_price
        } else {
            0.0
        };

        // Stop loss price
        let stop_loss_price = signal.stop_loss.map(|sl| {
            if position_dollars > 0.0 {
                current_price * (1.0 - sl)
            } else {
                current_price * (1.0 + sl)
            }
        });

        PositionResult {
            position_dollars,
            position_units,
            position_pct: if capital > 0.0 {
                position_dollars / capital
            } else {
                0.0
            },
            entry_price: current_price,
            stop_loss_price,
            stop_loss_pct: signal.stop_loss,
            take_profit_pct: signal.take_profit,
            regime: signal.regime,
            confidence: signal.confidence,
        }
    }
}

impl Default for PositionSizer {
    fn default() -> Self {
        Self::new(1.0, 0.02, true)
    }
}

/// Result of position sizing calculation.
#[derive(Debug, Clone)]
pub struct PositionResult {
    pub position_dollars: f64,
    pub position_units: f64,
    pub position_pct: f64,
    pub entry_price: f64,
    pub stop_loss_price: Option<f64>,
    pub stop_loss_pct: Option<f64>,
    pub take_profit_pct: Option<f64>,
    pub regime: MarketRegime,
    pub confidence: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_generation() {
        let generator = SignalGenerator::default();
        let result = RegimeResult::new(
            MarketRegime::Bull,
            0.8,
            0.8,
            "Bull market",
        );

        let signal = generator.generate_signal(&result);
        assert_eq!(signal.signal_type, SignalType::Buy);
        assert!(signal.position_size > 0.0);
    }

    #[test]
    fn test_position_sizing() {
        let sizer = PositionSizer::default();
        let signal = TradingSignal::new(SignalType::Buy, 0.5, MarketRegime::Bull, 0.8)
            .with_stop_loss(0.05);

        let result = sizer.calculate_position(&signal, 100000.0, 100.0, 0.2);

        assert!(result.position_dollars > 0.0);
        assert!(result.position_units > 0.0);
        assert!(result.stop_loss_price.is_some());
    }

    #[test]
    fn test_crisis_signal() {
        let generator = SignalGenerator::default();
        let result = RegimeResult::new(
            MarketRegime::Crisis,
            0.9,
            0.9,
            "Crisis",
        );

        let signal = generator.generate_signal(&result);
        assert_eq!(signal.signal_type, SignalType::StrongSell);
        assert_eq!(signal.position_size, 0.0);
    }
}
