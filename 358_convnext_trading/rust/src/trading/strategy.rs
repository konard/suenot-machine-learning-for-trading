//! Trading strategy implementation

use anyhow::Result;
use ndarray::Array3;

use crate::convnext::ConvNeXt;
use crate::data::{Candle, FeatureBuilder};

use super::Signal;

/// Position in the market
#[derive(Clone, Debug, PartialEq)]
pub enum Position {
    /// Long position with entry price and size
    Long { entry_price: f64, size: f64 },
    /// Short position with entry price and size
    Short { entry_price: f64, size: f64 },
    /// No position
    Flat,
}

impl Position {
    /// Calculate unrealized PnL
    pub fn unrealized_pnl(&self, current_price: f64) -> f64 {
        match self {
            Position::Long { entry_price, size } => (current_price - entry_price) * size,
            Position::Short { entry_price, size } => (entry_price - current_price) * size,
            Position::Flat => 0.0,
        }
    }

    /// Get position size
    pub fn size(&self) -> f64 {
        match self {
            Position::Long { size, .. } => *size,
            Position::Short { size, .. } => *size,
            Position::Flat => 0.0,
        }
    }

    /// Check if position is open
    pub fn is_open(&self) -> bool {
        !matches!(self, Position::Flat)
    }
}

/// Trading strategy using ConvNeXt model
pub struct Strategy {
    /// ConvNeXt model
    model: ConvNeXt,
    /// Feature builder
    feature_builder: FeatureBuilder,
    /// Maximum risk per trade (fraction of portfolio)
    max_risk: f64,
    /// Sequence length for model input
    seq_length: usize,
    /// Stop loss percentage
    stop_loss: f64,
    /// Take profit percentage
    take_profit: f64,
}

impl Strategy {
    /// Create a new strategy
    pub fn new(model: ConvNeXt, max_risk: f64) -> Self {
        Self {
            model,
            feature_builder: FeatureBuilder::new(),
            max_risk,
            seq_length: 256,
            stop_loss: 0.02, // 2%
            take_profit: 0.04, // 4%
        }
    }

    /// Create with custom parameters
    pub fn with_params(
        model: ConvNeXt,
        max_risk: f64,
        seq_length: usize,
        stop_loss: f64,
        take_profit: f64,
    ) -> Self {
        Self {
            model,
            feature_builder: FeatureBuilder::new(),
            max_risk,
            seq_length,
            stop_loss,
            take_profit,
        }
    }

    /// Generate signal from recent candles
    pub fn generate_signal(&self, candles: &[Candle]) -> Result<Signal> {
        if candles.len() < self.seq_length {
            return Ok(Signal::Hold);
        }

        // Build features
        let features = self.feature_builder.build(candles)?;
        let (n_candles, n_features) = features.dim();

        // Get last sequence
        let start_idx = n_candles.saturating_sub(self.seq_length);
        let seq_features = features
            .slice(ndarray::s![start_idx.., ..])
            .to_owned();

        // Reshape to [1, seq_length, features] then transpose to [1, features, seq_length]
        let input = seq_features
            .into_shape((1, self.seq_length, n_features))?
            .permuted_axes([0, 2, 1])
            .to_owned();

        // Run model
        let output = self.model.forward(&input);

        Ok(Signal::from_output(&output))
    }

    /// Calculate position size using Kelly criterion
    pub fn calculate_position_size(
        &self,
        signal: &Signal,
        portfolio_value: f64,
        current_price: f64,
    ) -> f64 {
        if signal.is_hold() {
            return 0.0;
        }

        let confidence = signal.confidence();
        let edge = confidence - 0.5;
        let win_rate = confidence;
        let win_loss_ratio = self.take_profit / self.stop_loss;

        // Kelly fraction
        let kelly_f = if win_loss_ratio > 0.0 {
            (win_rate * win_loss_ratio - (1.0 - win_rate)) / win_loss_ratio
        } else {
            0.0
        };

        // Half-Kelly for safety
        let position_fraction = (kelly_f * 0.5).max(0.0).min(self.max_risk);

        // Calculate position size in units
        let position_value = portfolio_value * position_fraction;
        position_value / current_price
    }

    /// Check if stop loss or take profit hit
    pub fn check_exit(&self, position: &Position, current_price: f64) -> bool {
        match position {
            Position::Long { entry_price, .. } => {
                let pnl_pct = (current_price - entry_price) / entry_price;
                pnl_pct <= -self.stop_loss || pnl_pct >= self.take_profit
            }
            Position::Short { entry_price, .. } => {
                let pnl_pct = (entry_price - current_price) / entry_price;
                pnl_pct <= -self.stop_loss || pnl_pct >= self.take_profit
            }
            Position::Flat => false,
        }
    }

    /// Get stop loss percentage
    pub fn stop_loss(&self) -> f64 {
        self.stop_loss
    }

    /// Get take profit percentage
    pub fn take_profit(&self) -> f64 {
        self.take_profit
    }

    /// Get max risk per trade
    pub fn max_risk(&self) -> f64 {
        self.max_risk
    }

    /// Get sequence length
    pub fn seq_length(&self) -> usize {
        self.seq_length
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::convnext::ConvNeXtConfig;

    #[test]
    fn test_position_pnl() {
        let pos = Position::Long {
            entry_price: 100.0,
            size: 10.0,
        };
        assert!((pos.unrealized_pnl(110.0) - 100.0).abs() < 1e-6);
        assert!((pos.unrealized_pnl(90.0) - (-100.0)).abs() < 1e-6);

        let pos = Position::Short {
            entry_price: 100.0,
            size: 10.0,
        };
        assert!((pos.unrealized_pnl(90.0) - 100.0).abs() < 1e-6);
        assert!((pos.unrealized_pnl(110.0) - (-100.0)).abs() < 1e-6);
    }

    #[test]
    fn test_strategy_position_size() {
        let config = ConvNeXtConfig::tiny();
        let model = ConvNeXt::new(config);
        let strategy = Strategy::new(model, 0.02);

        let signal = Signal::Long { confidence: 0.7 };
        let size = strategy.calculate_position_size(&signal, 10000.0, 100.0);

        assert!(size > 0.0);
        assert!(size * 100.0 <= 10000.0 * strategy.max_risk());
    }

    #[test]
    fn test_check_exit() {
        let config = ConvNeXtConfig::tiny();
        let model = ConvNeXt::new(config);
        let strategy = Strategy::new(model, 0.02);

        let pos = Position::Long {
            entry_price: 100.0,
            size: 1.0,
        };

        // Stop loss at -2%
        assert!(strategy.check_exit(&pos, 97.0));
        // Take profit at +4%
        assert!(strategy.check_exit(&pos, 105.0));
        // In between - no exit
        assert!(!strategy.check_exit(&pos, 101.0));
    }
}
