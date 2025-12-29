//! Trend-following agent implementation

use super::{Agent, AgentAction, AgentId, AgentState};
use crate::environment::Observation;

/// Trend-following agent
///
/// Buys on uptrend, sells on downtrend.
/// Uses momentum indicator to determine trend direction.
#[derive(Debug, Clone)]
pub struct TrendFollowingAgent {
    id: AgentId,
    /// Momentum threshold to trigger trades
    threshold: f64,
    /// Lookback period for momentum calculation
    lookback: usize,
    /// Trade size as fraction of available capital
    size_fraction: f64,
    /// Maximum position size as fraction of capital
    max_position: f64,
}

impl Default for TrendFollowingAgent {
    fn default() -> Self {
        Self {
            id: 0,
            threshold: 0.02,
            lookback: 10,
            size_fraction: 0.1,
            max_position: 0.5,
        }
    }
}

impl TrendFollowingAgent {
    /// Create a new trend-following agent
    pub fn new(threshold: f64, lookback: usize, size_fraction: f64) -> Self {
        Self {
            id: 0,
            threshold,
            lookback,
            size_fraction,
            max_position: 0.5,
        }
    }

    /// Set maximum position size
    pub fn with_max_position(mut self, max_position: f64) -> Self {
        self.max_position = max_position;
        self
    }

    /// Calculate trade size based on state
    fn calculate_size(&self, state: &AgentState, price: f64) -> f64 {
        let available = state.cash * self.size_fraction;
        let max_size = state.portfolio_value(price) * self.max_position / price;
        (available / price).min(max_size).max(0.001)
    }
}

impl Agent for TrendFollowingAgent {
    fn id(&self) -> AgentId {
        self.id
    }

    fn set_id(&mut self, id: AgentId) {
        self.id = id;
    }

    fn name(&self) -> &str {
        "TrendFollowing"
    }

    fn act(&mut self, observation: &Observation, state: &AgentState) -> AgentAction {
        let momentum = state.momentum(self.lookback);
        let price = observation.mid_price;

        if momentum > self.threshold {
            // Strong uptrend - buy
            let size = self.calculate_size(state, price);
            if state.cash >= size * price {
                return AgentAction::MarketBuy { quantity: size };
            }
        } else if momentum < -self.threshold {
            // Strong downtrend - sell
            if state.inventory > 0.0 {
                let size = (state.inventory * self.size_fraction).max(0.001);
                return AgentAction::MarketSell { quantity: size };
            }
        }

        AgentAction::Hold
    }

    fn reset(&mut self) {
        // No internal state to reset
    }

    fn clone_box(&self) -> Box<dyn Agent> {
        Box::new(self.clone())
    }

    fn params_string(&self) -> String {
        format!(
            "threshold={:.4}, lookback={}, size={:.2}",
            self.threshold, self.lookback, self.size_fraction
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_observation(price: f64) -> Observation {
        Observation {
            mid_price: price,
            spread: 1.0,
            bid_depth: 100.0,
            ask_depth: 100.0,
            imbalance: 0.0,
            last_price: price,
            volume: 1000.0,
            volatility: 0.02,
            timestamp: 0,
        }
    }

    #[test]
    fn test_trend_following_buy() {
        let mut agent = TrendFollowingAgent::default();
        let mut state = AgentState::new(100_000.0);

        // Create uptrend
        for i in 0..20 {
            state.observe_price(100.0 + i as f64 * 2.0);
        }

        let obs = create_test_observation(140.0);
        let action = agent.act(&obs, &state);

        assert!(matches!(action, AgentAction::MarketBuy { .. }));
    }

    #[test]
    fn test_trend_following_sell() {
        let mut agent = TrendFollowingAgent::default();
        let mut state = AgentState::new(100_000.0);
        state.inventory = 10.0;

        // Create downtrend
        for i in 0..20 {
            state.observe_price(200.0 - i as f64 * 5.0);
        }

        let obs = create_test_observation(100.0);
        let action = agent.act(&obs, &state);

        assert!(matches!(action, AgentAction::MarketSell { .. }));
    }

    #[test]
    fn test_trend_following_hold() {
        let mut agent = TrendFollowingAgent::default();
        let mut state = AgentState::new(100_000.0);

        // Flat market
        for _ in 0..20 {
            state.observe_price(100.0);
        }

        let obs = create_test_observation(100.0);
        let action = agent.act(&obs, &state);

        assert!(matches!(action, AgentAction::Hold));
    }
}
