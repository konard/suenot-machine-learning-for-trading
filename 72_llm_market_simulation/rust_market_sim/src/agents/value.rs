//! Value Investor Agent
//!
//! Buys when price is below fundamental value, sells when above.
//! Patient, long-term oriented trading style.

use super::base::{Agent, AgentDecision, MarketState};
use crate::market::OrderType;

/// Value Investor agent configuration
#[derive(Debug, Clone)]
pub struct ValueInvestor {
    /// Agent ID
    id: String,
    /// Current cash balance
    cash: f64,
    /// Current share holdings
    shares: i64,
    /// Estimated fundamental value
    fundamental_value: f64,
    /// Discount threshold to trigger buy (e.g., 0.05 = 5%)
    discount_threshold: f64,
    /// Premium threshold to trigger sell
    premium_threshold: f64,
    /// Maximum position as fraction of portfolio
    max_position_pct: f64,
    /// Trade history
    trade_count: u64,
}

impl ValueInvestor {
    /// Create a new Value Investor agent
    pub fn new(
        id: String,
        initial_cash: f64,
        initial_shares: i64,
        fundamental_value: f64,
    ) -> Self {
        Self {
            id,
            cash: initial_cash,
            shares: initial_shares,
            fundamental_value,
            discount_threshold: 0.05,
            premium_threshold: 0.05,
            max_position_pct: 0.3,
            trade_count: 0,
        }
    }

    /// Set custom thresholds
    pub fn with_thresholds(mut self, discount: f64, premium: f64) -> Self {
        self.discount_threshold = discount;
        self.premium_threshold = premium;
        self
    }

    /// Set maximum position percentage
    pub fn with_max_position(mut self, max_pct: f64) -> Self {
        self.max_position_pct = max_pct;
        self
    }

    /// Update fundamental value estimate
    pub fn set_fundamental_value(&mut self, value: f64) {
        self.fundamental_value = value;
    }

    fn calculate_trade_size(&self, current_price: f64, is_buy: bool) -> u64 {
        let portfolio_value = self.cash + (self.shares as f64 * current_price);
        let max_position_value = portfolio_value * self.max_position_pct;

        if is_buy {
            // Calculate how many shares we can buy
            let available_cash = self.cash * 0.9; // Keep some cash buffer
            let max_shares_by_cash = (available_cash / current_price) as u64;

            // Calculate how many more shares we can hold
            let current_position_value = self.shares as f64 * current_price;
            let room_for_more = ((max_position_value - current_position_value) / current_price).max(0.0) as u64;

            max_shares_by_cash.min(room_for_more).min(10) // Cap at 10 per trade
        } else {
            // Sell up to 10 shares at a time
            (self.shares as u64).min(10)
        }
    }
}

impl Agent for ValueInvestor {
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
        let current_price = state.current_price;
        let deviation = (current_price - self.fundamental_value) / self.fundamental_value;

        // Buy when price is significantly below fundamental
        if deviation < -self.discount_threshold {
            let quantity = self.calculate_trade_size(current_price, true);
            if quantity > 0 {
                self.trade_count += 1;
                return AgentDecision::buy(
                    quantity,
                    OrderType::Limit,
                    Some(current_price * 1.001), // Slightly above to ensure fill
                    &format!(
                        "Price ${:.2} is {:.1}% below fundamental ${:.2}. Good value opportunity.",
                        current_price,
                        -deviation * 100.0,
                        self.fundamental_value
                    ),
                );
            }
        }

        // Sell when price is significantly above fundamental
        if deviation > self.premium_threshold && self.shares > 0 {
            let quantity = self.calculate_trade_size(current_price, false);
            if quantity > 0 {
                self.trade_count += 1;
                return AgentDecision::sell(
                    quantity,
                    OrderType::Limit,
                    Some(current_price * 0.999), // Slightly below to ensure fill
                    &format!(
                        "Price ${:.2} is {:.1}% above fundamental ${:.2}. Taking profits.",
                        current_price,
                        deviation * 100.0,
                        self.fundamental_value
                    ),
                );
            }
        }

        AgentDecision::hold(&format!(
            "Price ${:.2} within {:.1}% of fundamental ${:.2}. Waiting for better opportunity.",
            current_price,
            deviation.abs() * 100.0,
            self.fundamental_value
        ))
    }

    fn update_position(&mut self, cash_delta: f64, shares_delta: i64) {
        self.cash += cash_delta;
        self.shares += shares_delta;
    }

    fn agent_type(&self) -> &str {
        "ValueInvestor"
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
    fn test_buy_below_fundamental() {
        let mut agent = ValueInvestor::new("test".to_string(), 10000.0, 0, 100.0);

        // Price 10% below fundamental
        let state = create_market_state(90.0);
        let decision = agent.make_decision(&state);

        assert!(matches!(decision.action, super::super::base::Action::Buy));
        assert!(decision.quantity > 0);
    }

    #[test]
    fn test_sell_above_fundamental() {
        let mut agent = ValueInvestor::new("test".to_string(), 10000.0, 50, 100.0);

        // Price 10% above fundamental
        let state = create_market_state(110.0);
        let decision = agent.make_decision(&state);

        assert!(matches!(decision.action, super::super::base::Action::Sell));
        assert!(decision.quantity > 0);
    }

    #[test]
    fn test_hold_at_fair_value() {
        let mut agent = ValueInvestor::new("test".to_string(), 10000.0, 50, 100.0);

        // Price at fundamental
        let state = create_market_state(100.0);
        let decision = agent.make_decision(&state);

        assert!(matches!(decision.action, super::super::base::Action::Hold));
    }
}
