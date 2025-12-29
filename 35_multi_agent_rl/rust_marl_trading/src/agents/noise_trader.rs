//! Noise trader agent implementation

use rand::Rng;

use super::{Agent, AgentAction, AgentId, AgentState};
use crate::environment::Observation;

/// Noise trader agent
///
/// Trades randomly, providing liquidity and noise to the market.
/// Simulates uninformed retail traders.
#[derive(Debug, Clone)]
pub struct NoiseTrader {
    id: AgentId,
    /// Probability of taking action (vs holding)
    activity_rate: f64,
    /// Probability of buy vs sell when active
    buy_probability: f64,
    /// Base trade size
    base_size: f64,
    /// Size variability (uniform random in [1-var, 1+var] * base_size)
    size_variability: f64,
}

impl Default for NoiseTrader {
    fn default() -> Self {
        Self {
            id: 0,
            activity_rate: 0.3,
            buy_probability: 0.5,
            base_size: 0.5,
            size_variability: 0.5,
        }
    }
}

impl NoiseTrader {
    /// Create a new noise trader
    pub fn new(activity_rate: f64, buy_probability: f64, base_size: f64) -> Self {
        Self {
            id: 0,
            activity_rate,
            buy_probability,
            base_size,
            size_variability: 0.5,
        }
    }

    /// Create a bullish noise trader (more likely to buy)
    pub fn bullish() -> Self {
        Self {
            buy_probability: 0.7,
            ..Default::default()
        }
    }

    /// Create a bearish noise trader (more likely to sell)
    pub fn bearish() -> Self {
        Self {
            buy_probability: 0.3,
            ..Default::default()
        }
    }

    /// Create a very active noise trader
    pub fn hyperactive() -> Self {
        Self {
            activity_rate: 0.6,
            ..Default::default()
        }
    }

    fn random_size(&self) -> f64 {
        let mut rng = rand::thread_rng();
        let multiplier = 1.0 + rng.gen_range(-self.size_variability..self.size_variability);
        (self.base_size * multiplier).max(0.01)
    }
}

impl Agent for NoiseTrader {
    fn id(&self) -> AgentId {
        self.id
    }

    fn set_id(&mut self, id: AgentId) {
        self.id = id;
    }

    fn name(&self) -> &str {
        "NoiseTrader"
    }

    fn act(&mut self, observation: &Observation, state: &AgentState) -> AgentAction {
        let mut rng = rand::thread_rng();

        // Decide whether to trade
        if rng.gen::<f64>() > self.activity_rate {
            return AgentAction::Hold;
        }

        let price = observation.mid_price;
        let size = self.random_size();

        // Decide buy or sell
        if rng.gen::<f64>() < self.buy_probability {
            // Buy
            if state.cash >= size * price {
                AgentAction::MarketBuy { quantity: size }
            } else {
                AgentAction::Hold
            }
        } else {
            // Sell
            if state.inventory >= size {
                AgentAction::MarketSell { quantity: size }
            } else if state.inventory > 0.0 {
                AgentAction::MarketSell {
                    quantity: state.inventory,
                }
            } else {
                AgentAction::Hold
            }
        }
    }

    fn reset(&mut self) {
        // No internal state to reset
    }

    fn clone_box(&self) -> Box<dyn Agent> {
        Box::new(self.clone())
    }

    fn params_string(&self) -> String {
        format!(
            "activity={:.2}, buy_prob={:.2}, size={:.2}",
            self.activity_rate, self.buy_probability, self.base_size
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
    fn test_noise_trader_randomness() {
        let mut agent = NoiseTrader::default();
        let state = AgentState::new(100_000.0);
        let obs = create_test_observation(100.0);

        // Run many times to verify randomness
        let mut actions = Vec::new();
        for _ in 0..100 {
            actions.push(agent.act(&obs, &state));
        }

        // Should have a mix of actions
        let hold_count = actions
            .iter()
            .filter(|a| matches!(a, AgentAction::Hold))
            .count();
        let buy_count = actions
            .iter()
            .filter(|a| matches!(a, AgentAction::MarketBuy { .. }))
            .count();

        // With default settings, expect roughly 70% hold, 15% buy, 15% sell
        assert!(hold_count > 50); // Most should be hold
        assert!(buy_count > 5); // Some buys
    }

    #[test]
    fn test_bullish_trader() {
        let mut agent = NoiseTrader::bullish();
        let state = AgentState::new(100_000.0);
        let obs = create_test_observation(100.0);

        let mut buy_count = 0;
        let mut sell_count = 0;

        for _ in 0..1000 {
            match agent.act(&obs, &state) {
                AgentAction::MarketBuy { .. } => buy_count += 1,
                AgentAction::MarketSell { .. } => sell_count += 1,
                _ => {}
            }
        }

        // Bullish trader should buy more than sell
        assert!(buy_count > sell_count);
    }
}
