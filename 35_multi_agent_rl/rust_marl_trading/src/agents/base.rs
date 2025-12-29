//! Base agent traits and types

use serde::{Deserialize, Serialize};

use crate::environment::Observation;

/// Unique identifier for an agent
pub type AgentId = usize;

/// Action that an agent can take
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentAction {
    /// Do nothing
    Hold,
    /// Market buy with quantity
    MarketBuy { quantity: f64 },
    /// Market sell with quantity
    MarketSell { quantity: f64 },
    /// Limit buy at price with quantity
    LimitBuy { price: f64, quantity: f64 },
    /// Limit sell at price with quantity
    LimitSell { price: f64, quantity: f64 },
    /// Quote both sides (market making)
    Quote {
        bid_price: f64,
        ask_price: f64,
        quantity: f64,
    },
    /// Cancel all orders
    CancelAll,
}

impl AgentAction {
    /// Check if this is a buy action
    pub fn is_buy(&self) -> bool {
        matches!(self, AgentAction::MarketBuy { .. } | AgentAction::LimitBuy { .. })
    }

    /// Check if this is a sell action
    pub fn is_sell(&self) -> bool {
        matches!(self, AgentAction::MarketSell { .. } | AgentAction::LimitSell { .. })
    }

    /// Check if this is a hold/no-op action
    pub fn is_hold(&self) -> bool {
        matches!(self, AgentAction::Hold | AgentAction::CancelAll)
    }

    /// Get the quantity involved in the action
    pub fn quantity(&self) -> f64 {
        match self {
            AgentAction::Hold | AgentAction::CancelAll => 0.0,
            AgentAction::MarketBuy { quantity }
            | AgentAction::MarketSell { quantity }
            | AgentAction::LimitBuy { quantity, .. }
            | AgentAction::LimitSell { quantity, .. }
            | AgentAction::Quote { quantity, .. } => *quantity,
        }
    }
}

/// Agent's internal state
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AgentState {
    /// Cash balance
    pub cash: f64,
    /// Asset inventory
    pub inventory: f64,
    /// Unrealized PnL
    pub unrealized_pnl: f64,
    /// Realized PnL
    pub realized_pnl: f64,
    /// Average entry price
    pub avg_entry_price: f64,
    /// Number of trades made
    pub trade_count: usize,
    /// Historical price observations
    pub price_history: Vec<f64>,
    /// Historical return observations
    pub return_history: Vec<f64>,
}

impl AgentState {
    /// Create a new agent state with initial cash
    pub fn new(initial_cash: f64) -> Self {
        Self {
            cash: initial_cash,
            inventory: 0.0,
            unrealized_pnl: 0.0,
            realized_pnl: 0.0,
            avg_entry_price: 0.0,
            trade_count: 0,
            price_history: Vec::new(),
            return_history: Vec::new(),
        }
    }

    /// Get total portfolio value at current price
    pub fn portfolio_value(&self, current_price: f64) -> f64 {
        self.cash + self.inventory * current_price
    }

    /// Get total PnL
    pub fn total_pnl(&self) -> f64 {
        self.realized_pnl + self.unrealized_pnl
    }

    /// Update unrealized PnL based on current price
    pub fn update_unrealized_pnl(&mut self, current_price: f64) {
        if self.inventory.abs() > 1e-10 {
            self.unrealized_pnl = (current_price - self.avg_entry_price) * self.inventory;
        } else {
            self.unrealized_pnl = 0.0;
        }
    }

    /// Record a buy trade
    pub fn record_buy(&mut self, price: f64, quantity: f64) {
        let cost = price * quantity;
        self.cash -= cost;

        // Update average entry price
        if self.inventory >= 0.0 {
            let total_cost = self.avg_entry_price * self.inventory + cost;
            self.inventory += quantity;
            self.avg_entry_price = total_cost / self.inventory;
        } else {
            // Covering short position
            let covered = quantity.min(-self.inventory);
            self.realized_pnl += (self.avg_entry_price - price) * covered;
            self.inventory += quantity;

            if self.inventory > 0.0 {
                self.avg_entry_price = price;
            }
        }

        self.trade_count += 1;
    }

    /// Record a sell trade
    pub fn record_sell(&mut self, price: f64, quantity: f64) {
        let revenue = price * quantity;
        self.cash += revenue;

        if self.inventory > 0.0 {
            // Selling long position
            let sold = quantity.min(self.inventory);
            self.realized_pnl += (price - self.avg_entry_price) * sold;
            self.inventory -= quantity;

            if self.inventory < 0.0 {
                self.avg_entry_price = price;
            }
        } else {
            // Opening/extending short position
            let total_value = self.avg_entry_price * (-self.inventory) + revenue;
            self.inventory -= quantity;
            self.avg_entry_price = total_value / (-self.inventory);
        }

        self.trade_count += 1;
    }

    /// Add a price observation
    pub fn observe_price(&mut self, price: f64) {
        if let Some(&last_price) = self.price_history.last() {
            if last_price > 0.0 {
                let ret = (price - last_price) / last_price;
                self.return_history.push(ret);
            }
        }
        self.price_history.push(price);

        // Keep only recent history
        if self.price_history.len() > 1000 {
            self.price_history.remove(0);
        }
        if self.return_history.len() > 1000 {
            self.return_history.remove(0);
        }
    }

    /// Get momentum (average return over period)
    pub fn momentum(&self, period: usize) -> f64 {
        if self.return_history.len() < period {
            return 0.0;
        }
        let recent: Vec<_> = self.return_history.iter().rev().take(period).collect();
        recent.iter().copied().sum::<f64>() / period as f64
    }

    /// Get volatility (standard deviation of returns)
    pub fn volatility(&self, period: usize) -> f64 {
        if self.return_history.len() < period {
            return 0.0;
        }
        let recent: Vec<_> = self.return_history.iter().rev().take(period).copied().collect();
        let mean = recent.iter().sum::<f64>() / recent.len() as f64;
        let variance = recent.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / recent.len() as f64;
        variance.sqrt()
    }

    /// Get SMA (simple moving average of prices)
    pub fn sma(&self, period: usize) -> f64 {
        if self.price_history.len() < period {
            return self.price_history.last().copied().unwrap_or(0.0);
        }
        let recent: Vec<_> = self.price_history.iter().rev().take(period).collect();
        recent.iter().copied().sum::<f64>() / period as f64
    }

    /// Reset the state
    pub fn reset(&mut self, initial_cash: f64) {
        self.cash = initial_cash;
        self.inventory = 0.0;
        self.unrealized_pnl = 0.0;
        self.realized_pnl = 0.0;
        self.avg_entry_price = 0.0;
        self.trade_count = 0;
        self.price_history.clear();
        self.return_history.clear();
    }
}

/// Trait for all trading agents
pub trait Agent: Send + Sync {
    /// Get the agent's unique ID
    fn id(&self) -> AgentId;

    /// Set the agent's ID
    fn set_id(&mut self, id: AgentId);

    /// Get the agent's name/type
    fn name(&self) -> &str;

    /// Choose an action based on the current observation
    fn act(&mut self, observation: &Observation, state: &AgentState) -> AgentAction;

    /// Update the agent after receiving a reward
    fn learn(&mut self, _reward: f64, _next_observation: &Observation, _done: bool) {
        // Default: no learning (rule-based agents)
    }

    /// Reset the agent's internal state
    fn reset(&mut self) {
        // Default: nothing to reset
    }

    /// Clone the agent as a boxed trait object
    fn clone_box(&self) -> Box<dyn Agent>;

    /// Get agent's parameters as a string (for logging)
    fn params_string(&self) -> String {
        "{}".to_string()
    }
}

impl Clone for Box<dyn Agent> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_state_creation() {
        let state = AgentState::new(100_000.0);
        assert_eq!(state.cash, 100_000.0);
        assert_eq!(state.inventory, 0.0);
        assert_eq!(state.portfolio_value(50_000.0), 100_000.0);
    }

    #[test]
    fn test_agent_state_buy() {
        let mut state = AgentState::new(100_000.0);
        state.record_buy(50_000.0, 1.0);

        assert_eq!(state.cash, 50_000.0);
        assert_eq!(state.inventory, 1.0);
        assert_eq!(state.avg_entry_price, 50_000.0);
        assert_eq!(state.trade_count, 1);
    }

    #[test]
    fn test_agent_state_sell() {
        let mut state = AgentState::new(100_000.0);
        state.record_buy(50_000.0, 1.0);
        state.record_sell(55_000.0, 1.0);

        assert_eq!(state.cash, 105_000.0);
        assert_eq!(state.inventory, 0.0);
        assert_eq!(state.realized_pnl, 5_000.0);
    }

    #[test]
    fn test_momentum_calculation() {
        let mut state = AgentState::new(100_000.0);

        // Add price history
        for i in 0..20 {
            state.observe_price(100.0 + i as f64);
        }

        // Positive momentum
        assert!(state.momentum(10) > 0.0);
    }

    #[test]
    fn test_agent_action() {
        let buy = AgentAction::MarketBuy { quantity: 10.0 };
        let sell = AgentAction::MarketSell { quantity: 5.0 };
        let hold = AgentAction::Hold;

        assert!(buy.is_buy());
        assert!(!buy.is_sell());
        assert!(sell.is_sell());
        assert!(hold.is_hold());
        assert_eq!(buy.quantity(), 10.0);
    }
}
