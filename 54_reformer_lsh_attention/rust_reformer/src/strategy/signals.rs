//! Signal generation for trading strategies
//!
//! Converts model predictions to actionable trading signals.

use ndarray::Array1;
use serde::{Deserialize, Serialize};

/// Trading signal type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TradingSignal {
    /// Strong buy signal
    StrongBuy,
    /// Buy signal
    Buy,
    /// Hold/neutral signal
    Hold,
    /// Sell signal
    Sell,
    /// Strong sell signal
    StrongSell,
}

impl TradingSignal {
    /// Convert to position multiplier (-1.0 to 1.0)
    pub fn to_position(&self) -> f64 {
        match self {
            TradingSignal::StrongBuy => 1.0,
            TradingSignal::Buy => 0.5,
            TradingSignal::Hold => 0.0,
            TradingSignal::Sell => -0.5,
            TradingSignal::StrongSell => -1.0,
        }
    }

    /// Convert from predicted return
    pub fn from_return(ret: f64, thresholds: &SignalThresholds) -> Self {
        if ret > thresholds.strong_buy {
            TradingSignal::StrongBuy
        } else if ret > thresholds.buy {
            TradingSignal::Buy
        } else if ret < thresholds.strong_sell {
            TradingSignal::StrongSell
        } else if ret < thresholds.sell {
            TradingSignal::Sell
        } else {
            TradingSignal::Hold
        }
    }
}

/// Thresholds for signal generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalThresholds {
    /// Threshold for strong buy signal
    pub strong_buy: f64,
    /// Threshold for buy signal
    pub buy: f64,
    /// Threshold for sell signal
    pub sell: f64,
    /// Threshold for strong sell signal
    pub strong_sell: f64,
}

impl Default for SignalThresholds {
    fn default() -> Self {
        Self {
            strong_buy: 0.02,   // 2%
            buy: 0.005,         // 0.5%
            sell: -0.005,       // -0.5%
            strong_sell: -0.02, // -2%
        }
    }
}

/// Signal generator from model predictions
pub struct SignalGenerator {
    /// Signal thresholds
    thresholds: SignalThresholds,
    /// Minimum confidence for signal generation
    min_confidence: f64,
    /// Use prediction ensemble
    use_ensemble: bool,
}

impl SignalGenerator {
    /// Create a new signal generator
    pub fn new(thresholds: SignalThresholds) -> Self {
        Self {
            thresholds,
            min_confidence: 0.0,
            use_ensemble: false,
        }
    }

    /// Create with default thresholds
    pub fn default_thresholds() -> Self {
        Self::new(SignalThresholds::default())
    }

    /// Set minimum confidence
    pub fn with_min_confidence(mut self, confidence: f64) -> Self {
        self.min_confidence = confidence;
        self
    }

    /// Generate signal from single prediction
    pub fn generate(&self, prediction: f64) -> TradingSignal {
        TradingSignal::from_return(prediction, &self.thresholds)
    }

    /// Generate signal from prediction array (uses mean)
    pub fn generate_from_array(&self, predictions: &Array1<f64>) -> TradingSignal {
        let mean = predictions.mean().unwrap_or(0.0);
        self.generate(mean)
    }

    /// Generate signal with confidence score
    pub fn generate_with_confidence(&self, prediction: f64) -> (TradingSignal, f64) {
        let signal = self.generate(prediction);

        // Calculate confidence based on distance from thresholds
        let confidence = self.calculate_confidence(prediction);

        (signal, confidence)
    }

    /// Calculate confidence score for a prediction
    fn calculate_confidence(&self, prediction: f64) -> f64 {
        // Distance from hold region normalized
        if prediction > self.thresholds.buy {
            let dist = prediction - self.thresholds.buy;
            let range = self.thresholds.strong_buy - self.thresholds.buy;
            (dist / range).min(1.0)
        } else if prediction < self.thresholds.sell {
            let dist = self.thresholds.sell - prediction;
            let range = self.thresholds.sell - self.thresholds.strong_sell;
            (dist / range).min(1.0)
        } else {
            // In hold region - low confidence
            0.0
        }
    }

    /// Generate position size based on prediction and confidence
    pub fn generate_position_size(&self, prediction: f64, max_position: f64) -> f64 {
        let (signal, confidence) = self.generate_with_confidence(prediction);

        if confidence < self.min_confidence {
            return 0.0;
        }

        let base_position = signal.to_position() * max_position;
        base_position * confidence
    }
}

/// Trading strategy combining signals with risk management
#[derive(Debug, Clone)]
pub struct TradingStrategy {
    /// Signal generator
    signal_generator: SignalGenerator,
    /// Maximum position size (fraction of capital)
    max_position: f64,
    /// Stop loss percentage
    stop_loss: f64,
    /// Take profit percentage
    take_profit: f64,
    /// Trailing stop percentage (0 to disable)
    trailing_stop: f64,
    /// Current position
    current_position: f64,
    /// Entry price
    entry_price: Option<f64>,
    /// Highest price since entry (for trailing stop)
    highest_since_entry: f64,
    /// Lowest price since entry
    lowest_since_entry: f64,
}

impl TradingStrategy {
    /// Create a new trading strategy
    pub fn new(
        thresholds: SignalThresholds,
        max_position: f64,
        stop_loss: f64,
        take_profit: f64,
    ) -> Self {
        Self {
            signal_generator: SignalGenerator::new(thresholds),
            max_position,
            stop_loss,
            take_profit,
            trailing_stop: 0.0,
            current_position: 0.0,
            entry_price: None,
            highest_since_entry: 0.0,
            lowest_since_entry: f64::MAX,
        }
    }

    /// Set trailing stop
    pub fn with_trailing_stop(mut self, trailing_stop: f64) -> Self {
        self.trailing_stop = trailing_stop;
        self
    }

    /// Process a new prediction and current price
    pub fn process(&mut self, prediction: f64, current_price: f64) -> StrategyAction {
        // Update tracking prices
        if self.entry_price.is_some() {
            self.highest_since_entry = self.highest_since_entry.max(current_price);
            self.lowest_since_entry = self.lowest_since_entry.min(current_price);
        }

        // Check stop loss / take profit first
        if let Some(action) = self.check_exit_conditions(current_price) {
            return action;
        }

        // Generate new signal
        let position_size = self
            .signal_generator
            .generate_position_size(prediction, self.max_position);

        // Determine action
        self.determine_action(position_size, current_price)
    }

    /// Check stop loss and take profit conditions
    fn check_exit_conditions(&mut self, current_price: f64) -> Option<StrategyAction> {
        let entry_price = self.entry_price?;

        if self.current_position > 0.0 {
            // Long position
            let pnl_pct = (current_price - entry_price) / entry_price;

            // Stop loss
            if pnl_pct < -self.stop_loss {
                return Some(StrategyAction::Close {
                    reason: "Stop Loss".to_string(),
                });
            }

            // Take profit
            if pnl_pct > self.take_profit {
                return Some(StrategyAction::Close {
                    reason: "Take Profit".to_string(),
                });
            }

            // Trailing stop
            if self.trailing_stop > 0.0 {
                let drawdown = (self.highest_since_entry - current_price) / self.highest_since_entry;
                if drawdown > self.trailing_stop {
                    return Some(StrategyAction::Close {
                        reason: "Trailing Stop".to_string(),
                    });
                }
            }
        } else if self.current_position < 0.0 {
            // Short position
            let pnl_pct = (entry_price - current_price) / entry_price;

            // Stop loss
            if pnl_pct < -self.stop_loss {
                return Some(StrategyAction::Close {
                    reason: "Stop Loss".to_string(),
                });
            }

            // Take profit
            if pnl_pct > self.take_profit {
                return Some(StrategyAction::Close {
                    reason: "Take Profit".to_string(),
                });
            }

            // Trailing stop
            if self.trailing_stop > 0.0 {
                let drawup = (current_price - self.lowest_since_entry) / self.lowest_since_entry;
                if drawup > self.trailing_stop {
                    return Some(StrategyAction::Close {
                        reason: "Trailing Stop".to_string(),
                    });
                }
            }
        }

        None
    }

    /// Determine action based on target position
    fn determine_action(&mut self, target_position: f64, current_price: f64) -> StrategyAction {
        let position_diff = target_position - self.current_position;

        if position_diff.abs() < 0.01 {
            return StrategyAction::Hold;
        }

        if position_diff > 0.0 {
            // Need to buy
            StrategyAction::Buy {
                size: position_diff,
                price: current_price,
            }
        } else {
            // Need to sell
            StrategyAction::Sell {
                size: -position_diff,
                price: current_price,
            }
        }
    }

    /// Update position after action
    pub fn update_position(&mut self, action: &StrategyAction, price: f64) {
        match action {
            StrategyAction::Buy { size, .. } => {
                if self.current_position == 0.0 {
                    self.entry_price = Some(price);
                    self.highest_since_entry = price;
                    self.lowest_since_entry = price;
                }
                self.current_position += size;
            }
            StrategyAction::Sell { size, .. } => {
                self.current_position -= size;
                if self.current_position == 0.0 {
                    self.entry_price = None;
                } else if self.entry_price.is_none() {
                    self.entry_price = Some(price);
                    self.highest_since_entry = price;
                    self.lowest_since_entry = price;
                }
            }
            StrategyAction::Close { .. } => {
                self.current_position = 0.0;
                self.entry_price = None;
            }
            StrategyAction::Hold => {}
        }
    }

    /// Get current position
    pub fn position(&self) -> f64 {
        self.current_position
    }

    /// Reset strategy state
    pub fn reset(&mut self) {
        self.current_position = 0.0;
        self.entry_price = None;
        self.highest_since_entry = 0.0;
        self.lowest_since_entry = f64::MAX;
    }
}

/// Strategy action
#[derive(Debug, Clone)]
pub enum StrategyAction {
    /// Buy order
    Buy { size: f64, price: f64 },
    /// Sell order
    Sell { size: f64, price: f64 },
    /// Close position
    Close { reason: String },
    /// Hold current position
    Hold,
}

impl StrategyAction {
    /// Check if action requires execution
    pub fn requires_execution(&self) -> bool {
        !matches!(self, StrategyAction::Hold)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_from_return() {
        let thresholds = SignalThresholds::default();

        assert_eq!(
            TradingSignal::from_return(0.03, &thresholds),
            TradingSignal::StrongBuy
        );
        assert_eq!(
            TradingSignal::from_return(0.01, &thresholds),
            TradingSignal::Buy
        );
        assert_eq!(
            TradingSignal::from_return(0.0, &thresholds),
            TradingSignal::Hold
        );
        assert_eq!(
            TradingSignal::from_return(-0.01, &thresholds),
            TradingSignal::Sell
        );
        assert_eq!(
            TradingSignal::from_return(-0.03, &thresholds),
            TradingSignal::StrongSell
        );
    }

    #[test]
    fn test_signal_to_position() {
        assert_eq!(TradingSignal::StrongBuy.to_position(), 1.0);
        assert_eq!(TradingSignal::Hold.to_position(), 0.0);
        assert_eq!(TradingSignal::StrongSell.to_position(), -1.0);
    }

    #[test]
    fn test_signal_generator() {
        let generator = SignalGenerator::default_thresholds();

        let signal = generator.generate(0.03);
        assert_eq!(signal, TradingSignal::StrongBuy);
    }

    #[test]
    fn test_position_size() {
        let generator = SignalGenerator::default_thresholds().with_min_confidence(0.0);

        let size = generator.generate_position_size(0.03, 1.0);
        assert!(size > 0.0);

        let size_hold = generator.generate_position_size(0.0, 1.0);
        assert_eq!(size_hold, 0.0);
    }

    #[test]
    fn test_strategy_stop_loss() {
        let mut strategy = TradingStrategy::new(
            SignalThresholds::default(),
            1.0,
            0.02, // 2% stop loss
            0.05, // 5% take profit
        );

        // Open long position
        strategy.current_position = 1.0;
        strategy.entry_price = Some(100.0);
        strategy.highest_since_entry = 100.0;
        strategy.lowest_since_entry = 100.0;

        // Price drops 3% - should trigger stop loss
        let action = strategy.process(0.01, 97.0);

        match action {
            StrategyAction::Close { reason } => assert!(reason.contains("Stop Loss")),
            _ => panic!("Expected Close action"),
        }
    }

    #[test]
    fn test_strategy_take_profit() {
        let mut strategy = TradingStrategy::new(
            SignalThresholds::default(),
            1.0,
            0.02, // 2% stop loss
            0.05, // 5% take profit
        );

        // Open long position
        strategy.current_position = 1.0;
        strategy.entry_price = Some(100.0);
        strategy.highest_since_entry = 105.5;
        strategy.lowest_since_entry = 100.0;

        // Price up 6% - should trigger take profit
        let action = strategy.process(0.01, 106.0);

        match action {
            StrategyAction::Close { reason } => assert!(reason.contains("Take Profit")),
            _ => panic!("Expected Close action"),
        }
    }

    #[test]
    fn test_strategy_trailing_stop() {
        let mut strategy = TradingStrategy::new(
            SignalThresholds::default(),
            1.0,
            0.10, // 10% stop loss
            0.20, // 20% take profit
        )
        .with_trailing_stop(0.03); // 3% trailing stop

        // Open long position
        strategy.current_position = 1.0;
        strategy.entry_price = Some(100.0);
        strategy.highest_since_entry = 110.0; // Rose 10%
        strategy.lowest_since_entry = 100.0;

        // Price drops 4% from high - should trigger trailing stop
        let action = strategy.process(0.01, 105.0);

        match action {
            StrategyAction::Close { reason } => assert!(reason.contains("Trailing Stop")),
            _ => panic!("Expected Close action"),
        }
    }
}
