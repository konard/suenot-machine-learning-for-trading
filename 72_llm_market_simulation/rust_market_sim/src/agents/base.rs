//! Base Agent Trait
//!
//! Defines the common interface for all trading agents.

use crate::market::{OrderType, Side};
use serde::{Deserialize, Serialize};

/// Trading action
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Action {
    /// Buy shares
    Buy,
    /// Sell shares
    Sell,
    /// Do nothing
    Hold,
}

/// Market state information provided to agents
#[derive(Debug, Clone)]
pub struct MarketState {
    /// Current price
    pub current_price: f64,
    /// Best bid price
    pub best_bid: Option<f64>,
    /// Best ask price
    pub best_ask: Option<f64>,
    /// Price history
    pub price_history: Vec<f64>,
    /// Fundamental value (if known)
    pub fundamental_value: f64,
    /// Current step number
    pub step: u64,
}

/// Agent's decision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentDecision {
    /// Action to take
    pub action: Action,
    /// Quantity
    pub quantity: u64,
    /// Order type
    pub order_type: OrderType,
    /// Limit price (for limit orders)
    pub limit_price: Option<f64>,
    /// Reasoning (for explainability)
    pub reasoning: String,
}

impl AgentDecision {
    /// Create a hold decision
    pub fn hold(reason: &str) -> Self {
        Self {
            action: Action::Hold,
            quantity: 0,
            order_type: OrderType::Market,
            limit_price: None,
            reasoning: reason.to_string(),
        }
    }

    /// Create a buy decision
    pub fn buy(quantity: u64, order_type: OrderType, price: Option<f64>, reason: &str) -> Self {
        Self {
            action: Action::Buy,
            quantity,
            order_type,
            limit_price: price,
            reasoning: reason.to_string(),
        }
    }

    /// Create a sell decision
    pub fn sell(quantity: u64, order_type: OrderType, price: Option<f64>, reason: &str) -> Self {
        Self {
            action: Action::Sell,
            quantity,
            order_type,
            limit_price: price,
            reasoning: reason.to_string(),
        }
    }

    /// Get order side
    pub fn side(&self) -> Option<Side> {
        match self.action {
            Action::Buy => Some(Side::Buy),
            Action::Sell => Some(Side::Sell),
            Action::Hold => None,
        }
    }
}

/// Trait for trading agents
pub trait Agent: Send + Sync {
    /// Get agent's unique ID
    fn id(&self) -> &str;

    /// Get agent's current cash balance
    fn cash(&self) -> f64;

    /// Get agent's current share holdings
    fn shares(&self) -> i64;

    /// Make a trading decision based on market state
    fn make_decision(&mut self, state: &MarketState) -> AgentDecision;

    /// Update agent's state after a trade
    fn update_position(&mut self, cash_delta: f64, shares_delta: i64);

    /// Get agent's total portfolio value
    fn portfolio_value(&self, current_price: f64) -> f64 {
        self.cash() + (self.shares() as f64 * current_price)
    }

    /// Get agent type name
    fn agent_type(&self) -> &str;
}
