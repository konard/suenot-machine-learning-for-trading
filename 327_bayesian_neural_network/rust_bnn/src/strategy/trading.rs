//! Uncertainty-aware trading strategy.

use chrono::{DateTime, Utc};

use crate::bnn::model::BayesianNN;
use crate::bnn::uncertainty::UncertaintyEstimate;
use super::signal::{Signal, SignalGenerator, SignalType};

/// Trading strategy using Bayesian Neural Networks.
#[derive(Debug)]
pub struct TradingStrategy {
    /// Signal generator
    pub signal_generator: SignalGenerator,
    /// Stop loss percentage
    pub stop_loss: f64,
    /// Take profit percentage
    pub take_profit: f64,
    /// Maximum drawdown allowed
    pub max_drawdown: f64,
}

impl Default for TradingStrategy {
    fn default() -> Self {
        Self {
            signal_generator: SignalGenerator::default(),
            stop_loss: 0.02,
            take_profit: 0.04,
            max_drawdown: 0.10,
        }
    }
}

impl TradingStrategy {
    /// Create a new trading strategy.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set signal generator parameters.
    pub fn signal_params(mut self, prob_threshold: f64, conf_threshold: f64) -> Self {
        self.signal_generator.prob_threshold = prob_threshold;
        self.signal_generator.conf_threshold = conf_threshold;
        self
    }

    /// Set risk parameters.
    pub fn risk_params(mut self, stop_loss: f64, take_profit: f64) -> Self {
        self.stop_loss = stop_loss;
        self.take_profit = take_profit;
        self
    }

    /// Generate signal from model prediction.
    pub fn generate_signal(
        &self,
        estimate: &UncertaintyEstimate,
        timestamp: DateTime<Utc>,
    ) -> Signal {
        self.signal_generator.generate(estimate, timestamp)
    }

    /// Check if position should be closed.
    pub fn should_close_position(
        &self,
        entry_price: f64,
        current_price: f64,
        position_type: SignalType,
        current_signal: &Signal,
    ) -> (bool, &'static str) {
        let pnl_pct = match position_type {
            SignalType::Long => (current_price - entry_price) / entry_price,
            SignalType::Short => (entry_price - current_price) / entry_price,
            SignalType::Neutral => return (false, "no_position"),
        };

        // Take profit
        if pnl_pct >= self.take_profit {
            return (true, "take_profit");
        }

        // Stop loss
        if pnl_pct <= -self.stop_loss {
            return (true, "stop_loss");
        }

        // Signal reversal
        match (position_type, current_signal.signal_type) {
            (SignalType::Long, SignalType::Short) => return (true, "signal_reversal"),
            (SignalType::Short, SignalType::Long) => return (true, "signal_reversal"),
            _ => {}
        }

        (false, "hold")
    }

    /// Calculate dynamic stop loss based on uncertainty.
    ///
    /// Higher uncertainty → wider stop loss to avoid noise exits.
    pub fn dynamic_stop_loss(&self, base_stop: f64, uncertainty: f64) -> f64 {
        // Scale stop loss by uncertainty (more uncertainty = wider stop)
        let uncertainty_factor = 1.0 + uncertainty.sqrt();
        (base_stop * uncertainty_factor).min(base_stop * 2.0) // Cap at 2x base
    }

    /// Calculate position size accounting for portfolio risk.
    pub fn calculate_position_value(
        &self,
        portfolio_value: f64,
        signal: &Signal,
        current_drawdown: f64,
    ) -> f64 {
        // Base position from signal
        let base_position = portfolio_value * signal.position_size;

        // Reduce position if in drawdown
        let drawdown_factor = if current_drawdown > 0.0 {
            let remaining_buffer = (self.max_drawdown - current_drawdown) / self.max_drawdown;
            remaining_buffer.max(0.25) // Always allow at least 25% of normal size
        } else {
            1.0
        };

        base_position * drawdown_factor
    }
}

/// Portfolio state for tracking.
#[derive(Debug, Clone)]
pub struct Portfolio {
    /// Current cash
    pub cash: f64,
    /// Current position size (positive = long, negative = short)
    pub position: f64,
    /// Average entry price
    pub entry_price: f64,
    /// Position type
    pub position_type: SignalType,
    /// Total portfolio value
    pub total_value: f64,
    /// Peak portfolio value (for drawdown calculation)
    pub peak_value: f64,
    /// Unrealized PnL
    pub unrealized_pnl: f64,
}

impl Default for Portfolio {
    fn default() -> Self {
        Self::new(10000.0)
    }
}

impl Portfolio {
    /// Create a new portfolio with initial capital.
    pub fn new(initial_capital: f64) -> Self {
        Self {
            cash: initial_capital,
            position: 0.0,
            entry_price: 0.0,
            position_type: SignalType::Neutral,
            total_value: initial_capital,
            peak_value: initial_capital,
            unrealized_pnl: 0.0,
        }
    }

    /// Update portfolio value at current price.
    pub fn update_value(&mut self, current_price: f64) {
        if self.position != 0.0 {
            let position_value = self.position.abs() * current_price;
            let cost_basis = self.position.abs() * self.entry_price;

            self.unrealized_pnl = if self.position > 0.0 {
                position_value - cost_basis
            } else {
                cost_basis - position_value
            };

            self.total_value = self.cash + position_value + self.unrealized_pnl;
        } else {
            self.total_value = self.cash;
            self.unrealized_pnl = 0.0;
        }

        if self.total_value > self.peak_value {
            self.peak_value = self.total_value;
        }
    }

    /// Get current drawdown.
    pub fn current_drawdown(&self) -> f64 {
        if self.peak_value > 0.0 {
            (self.peak_value - self.total_value) / self.peak_value
        } else {
            0.0
        }
    }

    /// Open a position.
    pub fn open_position(
        &mut self,
        position_type: SignalType,
        position_value: f64,
        price: f64,
        transaction_cost: f64,
    ) {
        let cost = position_value * (1.0 + transaction_cost);
        let position_size = position_value / price;

        self.position = if position_type == SignalType::Long {
            position_size
        } else {
            -position_size
        };
        self.entry_price = price;
        self.position_type = position_type;
        self.cash -= cost;
    }

    /// Close current position.
    pub fn close_position(&mut self, price: f64, transaction_cost: f64) -> f64 {
        if self.position == 0.0 {
            return 0.0;
        }

        let position_value = self.position.abs() * price;
        let cost = position_value * transaction_cost;

        let pnl = if self.position > 0.0 {
            (price - self.entry_price) * self.position.abs() - cost
        } else {
            (self.entry_price - price) * self.position.abs() - cost
        };

        self.cash += position_value - cost + self.unrealized_pnl;
        self.position = 0.0;
        self.entry_price = 0.0;
        self.position_type = SignalType::Neutral;
        self.unrealized_pnl = 0.0;

        pnl
    }
}
