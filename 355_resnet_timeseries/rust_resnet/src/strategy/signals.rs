//! Trading signal generation

use ndarray::Array2;
use serde::{Deserialize, Serialize};

/// Trading signal types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TradingSignal {
    /// Long position (buy)
    Long,
    /// Short position (sell)
    Short,
    /// No position
    Neutral,
}

impl TradingSignal {
    /// Convert from class prediction
    pub fn from_class(class: u8) -> Self {
        match class {
            0 => TradingSignal::Short,  // Down prediction -> Short
            2 => TradingSignal::Long,   // Up prediction -> Long
            _ => TradingSignal::Neutral,
        }
    }

    /// Get the position direction
    pub fn direction(&self) -> f32 {
        match self {
            TradingSignal::Long => 1.0,
            TradingSignal::Short => -1.0,
            TradingSignal::Neutral => 0.0,
        }
    }
}

/// Trading strategy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingStrategy {
    /// Confidence threshold for trading
    pub confidence_threshold: f32,
    /// Position sizing method
    pub position_sizing: PositionSizing,
    /// Maximum position size as fraction of portfolio
    pub max_position: f32,
    /// Base position size
    pub base_position: f32,
}

/// Position sizing methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PositionSizing {
    /// Fixed position size
    Fixed,
    /// Kelly criterion
    Kelly,
    /// Confidence-based scaling
    Confidence,
}

impl Default for TradingStrategy {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.6,
            position_sizing: PositionSizing::Confidence,
            max_position: 0.25,
            base_position: 0.1,
        }
    }
}

impl TradingStrategy {
    /// Create a new trading strategy
    pub fn new(confidence_threshold: f32, max_position: f32) -> Self {
        Self {
            confidence_threshold,
            max_position,
            ..Default::default()
        }
    }

    /// Generate trading signal from model probabilities
    ///
    /// # Arguments
    ///
    /// * `probs` - Probability distribution over classes [down, neutral, up]
    ///
    /// # Returns
    ///
    /// Tuple of (signal, confidence)
    pub fn generate_signal(&self, probs: &[f32; 3]) -> (TradingSignal, f32) {
        let prob_down = probs[0];
        let prob_neutral = probs[1];
        let prob_up = probs[2];

        // Find the most confident prediction
        let max_prob = prob_down.max(prob_neutral).max(prob_up);

        if max_prob < self.confidence_threshold {
            return (TradingSignal::Neutral, prob_neutral);
        }

        if prob_up >= prob_down && prob_up >= prob_neutral {
            (TradingSignal::Long, prob_up)
        } else if prob_down >= prob_up && prob_down >= prob_neutral {
            (TradingSignal::Short, prob_down)
        } else {
            (TradingSignal::Neutral, prob_neutral)
        }
    }

    /// Calculate position size based on confidence
    ///
    /// # Arguments
    ///
    /// * `confidence` - Model confidence (0.0 to 1.0)
    /// * `portfolio_value` - Current portfolio value
    ///
    /// # Returns
    ///
    /// Position size in portfolio currency
    pub fn calculate_position_size(&self, confidence: f32, portfolio_value: f32) -> f32 {
        let size = match self.position_sizing {
            PositionSizing::Fixed => self.base_position * portfolio_value,
            PositionSizing::Kelly => {
                // Simplified Kelly: scale by edge
                let edge = (confidence - 0.5).max(0.0);
                let kelly_fraction = edge * 2.0; // Convert to fraction
                kelly_fraction.min(self.max_position) * portfolio_value
            }
            PositionSizing::Confidence => {
                // Linear scaling with confidence
                let scale = (confidence - self.confidence_threshold)
                    / (1.0 - self.confidence_threshold);
                let fraction = self.base_position + scale * (self.max_position - self.base_position);
                fraction.min(self.max_position) * portfolio_value
            }
        };

        size.min(self.max_position * portfolio_value)
    }

    /// Generate signals for a batch of predictions
    pub fn generate_signals_batch(&self, probs: &Array2<f32>) -> Vec<(TradingSignal, f32)> {
        let n = probs.shape()[0];
        let mut signals = Vec::with_capacity(n);

        for i in 0..n {
            let p = [probs[[i, 0]], probs[[i, 1]], probs[[i, 2]]];
            signals.push(self.generate_signal(&p));
        }

        signals
    }
}

/// Trade record for backtesting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    /// Entry timestamp
    pub entry_time: i64,
    /// Exit timestamp
    pub exit_time: Option<i64>,
    /// Entry price
    pub entry_price: f32,
    /// Exit price
    pub exit_price: Option<f32>,
    /// Position direction (1.0 for long, -1.0 for short)
    pub direction: f32,
    /// Position size
    pub size: f32,
    /// Trade PnL
    pub pnl: Option<f32>,
    /// Trade signal
    pub signal: TradingSignal,
    /// Confidence at entry
    pub confidence: f32,
}

impl Trade {
    /// Create a new open trade
    pub fn open(
        entry_time: i64,
        entry_price: f32,
        direction: f32,
        size: f32,
        signal: TradingSignal,
        confidence: f32,
    ) -> Self {
        Self {
            entry_time,
            exit_time: None,
            entry_price,
            exit_price: None,
            direction,
            size,
            pnl: None,
            signal,
            confidence,
        }
    }

    /// Close the trade
    pub fn close(&mut self, exit_time: i64, exit_price: f32) {
        self.exit_time = Some(exit_time);
        self.exit_price = Some(exit_price);
        self.pnl = Some(self.calculate_pnl(exit_price));
    }

    /// Calculate PnL for a given exit price
    pub fn calculate_pnl(&self, current_price: f32) -> f32 {
        let price_change = current_price - self.entry_price;
        let return_pct = price_change / self.entry_price;
        self.size * return_pct * self.direction
    }

    /// Check if trade is open
    pub fn is_open(&self) -> bool {
        self.exit_time.is_none()
    }

    /// Get return percentage
    pub fn return_pct(&self) -> Option<f32> {
        self.exit_price.map(|exit| {
            let price_change = exit - self.entry_price;
            (price_change / self.entry_price) * self.direction
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_from_class() {
        assert_eq!(TradingSignal::from_class(0), TradingSignal::Short);
        assert_eq!(TradingSignal::from_class(1), TradingSignal::Neutral);
        assert_eq!(TradingSignal::from_class(2), TradingSignal::Long);
    }

    #[test]
    fn test_generate_signal() {
        let strategy = TradingStrategy::default();

        // High confidence up
        let (signal, conf) = strategy.generate_signal(&[0.1, 0.2, 0.7]);
        assert_eq!(signal, TradingSignal::Long);
        assert!((conf - 0.7).abs() < 0.001);

        // High confidence down
        let (signal, _) = strategy.generate_signal(&[0.7, 0.2, 0.1]);
        assert_eq!(signal, TradingSignal::Short);

        // Low confidence
        let (signal, _) = strategy.generate_signal(&[0.3, 0.4, 0.3]);
        assert_eq!(signal, TradingSignal::Neutral);
    }

    #[test]
    fn test_position_sizing() {
        let strategy = TradingStrategy::default();

        // High confidence should give larger position
        let size_high = strategy.calculate_position_size(0.9, 100000.0);
        let size_low = strategy.calculate_position_size(0.65, 100000.0);

        assert!(size_high > size_low);
        assert!(size_high <= 25000.0); // Max 25%
    }

    #[test]
    fn test_trade_pnl() {
        let mut trade = Trade::open(
            1000,
            50000.0,
            1.0, // Long
            10000.0,
            TradingSignal::Long,
            0.7,
        );

        // Price goes up 2%
        trade.close(2000, 51000.0);

        let pnl = trade.pnl.unwrap();
        assert!((pnl - 200.0).abs() < 0.1); // 2% of 10000
    }
}
