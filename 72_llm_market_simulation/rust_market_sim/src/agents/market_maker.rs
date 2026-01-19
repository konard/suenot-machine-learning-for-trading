//! Market Maker Agent
//!
//! Provides liquidity by continuously quoting bid and ask prices.
//! Makes money from the spread while managing inventory risk.

use super::base::{Agent, AgentDecision, MarketState, Action};
use crate::market::OrderType;

/// Market Maker agent
#[derive(Debug, Clone)]
pub struct MarketMaker {
    /// Agent ID
    id: String,
    /// Current cash balance
    cash: f64,
    /// Current share holdings
    shares: i64,
    /// Base spread (e.g., 0.002 = 0.2%)
    base_spread: f64,
    /// Target inventory
    target_inventory: i64,
    /// Maximum inventory deviation
    max_inventory_deviation: i64,
    /// Inventory adjustment factor
    inventory_skew: f64,
    /// Quote size
    quote_size: u64,
    /// Last quote side (alternates)
    last_side_was_bid: bool,
    /// Trade count
    trade_count: u64,
}

impl MarketMaker {
    /// Create a new Market Maker agent
    pub fn new(
        id: String,
        initial_cash: f64,
        initial_shares: i64,
    ) -> Self {
        Self {
            id,
            cash: initial_cash,
            shares: initial_shares,
            base_spread: 0.002,
            target_inventory: initial_shares,
            max_inventory_deviation: 200,
            inventory_skew: 0.0005,
            quote_size: 10,
            last_side_was_bid: false,
            trade_count: 0,
        }
    }

    /// Set base spread
    pub fn with_spread(mut self, spread: f64) -> Self {
        self.base_spread = spread;
        self
    }

    /// Set quote size
    pub fn with_quote_size(mut self, size: u64) -> Self {
        self.quote_size = size;
        self
    }

    /// Calculate bid and ask prices based on inventory
    fn calculate_quotes(&self, mid_price: f64) -> (f64, f64) {
        let half_spread = self.base_spread / 2.0;

        // Adjust quotes based on inventory
        let inventory_deviation = self.shares - self.target_inventory;
        let skew = inventory_deviation as f64 * self.inventory_skew;

        // If we have too much inventory, lower bid and ask to encourage selling
        // If we have too little, raise them to encourage buying
        let bid = mid_price * (1.0 - half_spread - skew);
        let ask = mid_price * (1.0 + half_spread - skew);

        (bid, ask)
    }

    /// Check if inventory allows quoting
    fn can_quote_bid(&self) -> bool {
        self.cash > 0.0 &&
        (self.shares - self.target_inventory) < self.max_inventory_deviation
    }

    fn can_quote_ask(&self) -> bool {
        self.shares > 0 &&
        (self.target_inventory - self.shares) < self.max_inventory_deviation
    }
}

impl Agent for MarketMaker {
    fn id(&self) -> &str {
        &self.id
    }

    fn cash(&self) -> f64 {
        self.cash
    }

    fn shares(&self) -> i64 {
        self.shares
    }

    fn make_decision(&mut self, state: &MarketState) -> AgentDecision {
        let mid_price = match (state.best_bid, state.best_ask) {
            (Some(bid), Some(ask)) => (bid + ask) / 2.0,
            _ => state.current_price,
        };

        let (bid_price, ask_price) = self.calculate_quotes(mid_price);

        // Alternate between posting bid and ask to simulate continuous quoting
        self.last_side_was_bid = !self.last_side_was_bid;

        if self.last_side_was_bid {
            // Post bid (buy order)
            if self.can_quote_bid() {
                let qty = self.quote_size.min((self.cash / bid_price) as u64);
                if qty > 0 {
                    self.trade_count += 1;
                    return AgentDecision {
                        action: Action::Buy,
                        quantity: qty,
                        order_type: OrderType::Limit,
                        limit_price: Some(bid_price),
                        reasoning: format!(
                            "Posting bid at ${:.2} (spread: {:.2}%, inventory: {})",
                            bid_price,
                            self.base_spread * 100.0,
                            self.shares
                        ),
                    };
                }
            }
        } else {
            // Post ask (sell order)
            if self.can_quote_ask() {
                let qty = self.quote_size.min(self.shares as u64);
                if qty > 0 {
                    self.trade_count += 1;
                    return AgentDecision {
                        action: Action::Sell,
                        quantity: qty,
                        order_type: OrderType::Limit,
                        limit_price: Some(ask_price),
                        reasoning: format!(
                            "Posting ask at ${:.2} (spread: {:.2}%, inventory: {})",
                            ask_price,
                            self.base_spread * 100.0,
                            self.shares
                        ),
                    };
                }
            }
        }

        // If we can't quote, just hold
        AgentDecision::hold(&format!(
            "Cannot quote: cash={:.2}, shares={}, inventory_deviation={}",
            self.cash,
            self.shares,
            self.shares - self.target_inventory
        ))
    }

    fn update_position(&mut self, cash_delta: f64, shares_delta: i64) {
        self.cash += cash_delta;
        self.shares += shares_delta;
    }

    fn agent_type(&self) -> &str {
        "MarketMaker"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_market_state(price: f64) -> MarketState {
        MarketState {
            current_price: price,
            best_bid: Some(price - 0.5),
            best_ask: Some(price + 0.5),
            price_history: vec![100.0, price],
            fundamental_value: 100.0,
            step: 1,
        }
    }

    #[test]
    fn test_quote_spread() {
        let agent = MarketMaker::new("mm".to_string(), 10000.0, 100)
            .with_spread(0.01);

        let (bid, ask) = agent.calculate_quotes(100.0);

        // Spread should be approximately 1%
        assert!(ask > bid);
        assert!((ask - bid - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_inventory_skew() {
        // High inventory should skew prices down
        let agent_high = MarketMaker::new("mm".to_string(), 10000.0, 300)
            .with_spread(0.01);

        // Low inventory should skew prices up
        let agent_low = MarketMaker::new("mm".to_string(), 10000.0, 0)
            .with_spread(0.01);

        let (bid_high, ask_high) = agent_high.calculate_quotes(100.0);
        let (bid_low, ask_low) = agent_low.calculate_quotes(100.0);

        // High inventory agent should have lower prices
        assert!(bid_high < bid_low);
        assert!(ask_high < ask_low);
    }

    #[test]
    fn test_posts_quotes() {
        let mut agent = MarketMaker::new("mm".to_string(), 10000.0, 100);
        let state = create_market_state(100.0);

        // First decision
        let decision1 = agent.make_decision(&state);
        // Second decision (alternates side)
        let decision2 = agent.make_decision(&state);

        // Should post on both sides
        let actions = vec![decision1.action, decision2.action];
        assert!(actions.contains(&Action::Buy) || actions.contains(&Action::Sell));
    }
}
