//! Momentum Trader Agent
//!
//! Follows price trends using moving average crossover strategy.
//! Active, trend-following trading style.

use super::base::{Agent, AgentDecision, MarketState};
use crate::market::OrderType;

/// Momentum Trader agent
#[derive(Debug, Clone)]
pub struct MomentumTrader {
    /// Agent ID
    id: String,
    /// Current cash balance
    cash: f64,
    /// Current share holdings
    shares: i64,
    /// Short-term MA window
    short_window: usize,
    /// Long-term MA window
    long_window: usize,
    /// Entry threshold (MA difference to trigger trade)
    entry_threshold: f64,
    /// Current position (1 = long, -1 = short, 0 = flat)
    position: i8,
    /// Entry price for current position
    entry_price: f64,
    /// Stop loss percentage
    stop_loss_pct: f64,
    /// Take profit percentage
    take_profit_pct: f64,
    /// Trade count
    trade_count: u64,
}

impl MomentumTrader {
    /// Create a new Momentum Trader agent
    pub fn new(
        id: String,
        initial_cash: f64,
        initial_shares: i64,
    ) -> Self {
        Self {
            id,
            cash: initial_cash,
            shares: initial_shares,
            short_window: 5,
            long_window: 20,
            entry_threshold: 0.02,
            position: 0,
            entry_price: 0.0,
            stop_loss_pct: 0.05,
            take_profit_pct: 0.10,
            trade_count: 0,
        }
    }

    /// Set MA windows
    pub fn with_windows(mut self, short: usize, long: usize) -> Self {
        self.short_window = short;
        self.long_window = long;
        self
    }

    /// Set entry threshold
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.entry_threshold = threshold;
        self
    }

    /// Calculate simple moving average
    fn calculate_sma(prices: &[f64], window: usize) -> Option<f64> {
        if prices.len() < window {
            return None;
        }
        let sum: f64 = prices.iter().rev().take(window).sum();
        Some(sum / window as f64)
    }

    /// Check if should exit position
    fn should_exit(&self, current_price: f64) -> bool {
        if self.entry_price <= 0.0 {
            return false;
        }

        let pnl_pct = (current_price - self.entry_price) / self.entry_price;

        if self.position > 0 {
            // Long position
            pnl_pct <= -self.stop_loss_pct || pnl_pct >= self.take_profit_pct
        } else if self.position < 0 {
            // Short position (inverted)
            pnl_pct >= self.stop_loss_pct || pnl_pct <= -self.take_profit_pct
        } else {
            false
        }
    }
}

impl Agent for MomentumTrader {
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
        let prices = &state.price_history;

        // Need enough history for MAs
        if prices.len() < self.long_window {
            return AgentDecision::hold("Not enough price history for MA calculation");
        }

        // Calculate MAs
        let short_ma = Self::calculate_sma(prices, self.short_window);
        let long_ma = Self::calculate_sma(prices, self.long_window);

        let (short_ma, long_ma) = match (short_ma, long_ma) {
            (Some(s), Some(l)) => (s, l),
            _ => return AgentDecision::hold("Unable to calculate moving averages"),
        };

        // Check for stop loss / take profit
        if self.position != 0 && self.should_exit(current_price) {
            let pnl_pct = (current_price - self.entry_price) / self.entry_price * 100.0;
            if self.shares > 0 {
                self.trade_count += 1;
                self.position = 0;
                self.entry_price = 0.0;
                return AgentDecision::sell(
                    self.shares as u64,
                    OrderType::Market,
                    None,
                    &format!("Exiting long position. PnL: {:.1}%", pnl_pct),
                );
            }
        }

        // Calculate momentum signal
        let ma_diff = (short_ma - long_ma) / long_ma;

        // Bullish crossover
        if ma_diff > self.entry_threshold && self.position <= 0 {
            let quantity = ((self.cash * 0.5) / current_price) as u64;
            if quantity > 0 {
                self.trade_count += 1;
                self.position = 1;
                self.entry_price = current_price;
                return AgentDecision::buy(
                    quantity,
                    OrderType::Market,
                    None,
                    &format!(
                        "Bullish MA crossover. Short MA ({:.2}) > Long MA ({:.2}) by {:.1}%",
                        short_ma, long_ma, ma_diff * 100.0
                    ),
                );
            }
        }

        // Bearish crossover
        if ma_diff < -self.entry_threshold && self.position >= 0 && self.shares > 0 {
            self.trade_count += 1;
            self.position = -1;
            self.entry_price = current_price;
            return AgentDecision::sell(
                self.shares as u64,
                OrderType::Market,
                None,
                &format!(
                    "Bearish MA crossover. Short MA ({:.2}) < Long MA ({:.2}) by {:.1}%",
                    short_ma, long_ma, -ma_diff * 100.0
                ),
            );
        }

        AgentDecision::hold(&format!(
            "MA diff {:.2}% within threshold. Short: {:.2}, Long: {:.2}",
            ma_diff * 100.0, short_ma, long_ma
        ))
    }

    fn update_position(&mut self, cash_delta: f64, shares_delta: i64) {
        self.cash += cash_delta;
        self.shares += shares_delta;
    }

    fn agent_type(&self) -> &str {
        "MomentumTrader"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_uptrend_state() -> MarketState {
        // Create price history with uptrend
        let prices: Vec<f64> = (0..30).map(|i| 100.0 + i as f64 * 0.5).collect();
        MarketState {
            current_price: *prices.last().unwrap(),
            best_bid: Some(prices.last().unwrap() - 0.5),
            best_ask: Some(prices.last().unwrap() + 0.5),
            price_history: prices,
            fundamental_value: 100.0,
            step: 30,
        }
    }

    fn create_downtrend_state() -> MarketState {
        // Create price history with downtrend
        let prices: Vec<f64> = (0..30).map(|i| 115.0 - i as f64 * 0.5).collect();
        MarketState {
            current_price: *prices.last().unwrap(),
            best_bid: Some(prices.last().unwrap() - 0.5),
            best_ask: Some(prices.last().unwrap() + 0.5),
            price_history: prices,
            fundamental_value: 100.0,
            step: 30,
        }
    }

    #[test]
    fn test_buy_on_uptrend() {
        let mut agent = MomentumTrader::new("test".to_string(), 10000.0, 0)
            .with_windows(5, 20)
            .with_threshold(0.01);

        let state = create_uptrend_state();
        let decision = agent.make_decision(&state);

        // Should buy on strong uptrend
        assert!(matches!(decision.action, super::super::base::Action::Buy));
    }

    #[test]
    fn test_sell_on_downtrend() {
        let mut agent = MomentumTrader::new("test".to_string(), 5000.0, 50)
            .with_windows(5, 20)
            .with_threshold(0.01);

        let state = create_downtrend_state();
        let decision = agent.make_decision(&state);

        // Should sell on strong downtrend
        assert!(matches!(decision.action, super::super::base::Action::Sell));
    }
}
