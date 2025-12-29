//! Market maker agent implementation

use super::{Agent, AgentAction, AgentId, AgentState};
use crate::environment::Observation;

/// Market maker agent
///
/// Provides liquidity by quoting both sides of the market.
/// Earns the bid-ask spread while managing inventory risk.
#[derive(Debug, Clone)]
pub struct MarketMakerAgent {
    id: AgentId,
    /// Base half-spread (distance from mid to quote)
    base_spread: f64,
    /// Inventory skew coefficient (adjust quotes based on inventory)
    skew_coefficient: f64,
    /// Maximum inventory before reducing quotes
    max_inventory: f64,
    /// Quote size
    quote_size: f64,
    /// Volatility adjustment factor
    volatility_factor: f64,
}

impl Default for MarketMakerAgent {
    fn default() -> Self {
        Self {
            id: 0,
            base_spread: 0.001, // 0.1% half-spread
            skew_coefficient: 0.0001, // Inventory adjustment
            max_inventory: 10.0,
            quote_size: 1.0,
            volatility_factor: 2.0,
        }
    }
}

impl MarketMakerAgent {
    /// Create a new market maker
    pub fn new(base_spread: f64, quote_size: f64) -> Self {
        Self {
            id: 0,
            base_spread,
            skew_coefficient: 0.0001,
            max_inventory: 10.0,
            quote_size,
            volatility_factor: 2.0,
        }
    }

    /// Set maximum inventory
    pub fn with_max_inventory(mut self, max_inventory: f64) -> Self {
        self.max_inventory = max_inventory;
        self
    }

    /// Set skew coefficient
    pub fn with_skew(mut self, skew_coefficient: f64) -> Self {
        self.skew_coefficient = skew_coefficient;
        self
    }

    /// Calculate quotes based on mid price, inventory, and volatility
    fn calculate_quotes(
        &self,
        mid_price: f64,
        inventory: f64,
        volatility: f64,
    ) -> (f64, f64) {
        // Adjust spread based on volatility
        let vol_adjusted_spread = self.base_spread * (1.0 + self.volatility_factor * volatility);

        // Skew quotes based on inventory
        // Positive inventory -> lower bid (less willing to buy), higher ask (more willing to sell)
        let inventory_skew = inventory * self.skew_coefficient;

        let bid_price = mid_price * (1.0 - vol_adjusted_spread - inventory_skew);
        let ask_price = mid_price * (1.0 + vol_adjusted_spread - inventory_skew);

        (bid_price, ask_price)
    }

    /// Determine quote size based on inventory
    fn calculate_quote_size(&self, inventory: f64) -> f64 {
        // Reduce size when approaching max inventory
        let inventory_ratio = inventory.abs() / self.max_inventory;
        let size_multiplier = (1.0 - inventory_ratio).max(0.1);
        self.quote_size * size_multiplier
    }
}

impl Agent for MarketMakerAgent {
    fn id(&self) -> AgentId {
        self.id
    }

    fn set_id(&mut self, id: AgentId) {
        self.id = id;
    }

    fn name(&self) -> &str {
        "MarketMaker"
    }

    fn act(&mut self, observation: &Observation, state: &AgentState) -> AgentAction {
        let mid_price = observation.mid_price;
        let volatility = observation.volatility;
        let inventory = state.inventory;

        // Check if we're too exposed
        if inventory.abs() > self.max_inventory {
            // Reduce exposure first
            if inventory > 0.0 {
                return AgentAction::MarketSell {
                    quantity: (inventory - self.max_inventory * 0.8).max(0.1),
                };
            } else {
                return AgentAction::MarketBuy {
                    quantity: (-inventory - self.max_inventory * 0.8).max(0.1),
                };
            }
        }

        // Calculate quotes
        let (bid_price, ask_price) = self.calculate_quotes(mid_price, inventory, volatility);
        let quote_size = self.calculate_quote_size(inventory);

        // Check if we have enough cash to support the bid
        let bid_value = bid_price * quote_size;
        if state.cash < bid_value {
            // Can't afford to quote, just quote ask side
            return AgentAction::LimitSell {
                price: ask_price,
                quantity: quote_size,
            };
        }

        AgentAction::Quote {
            bid_price,
            ask_price,
            quantity: quote_size,
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
            "spread={:.4}, size={:.2}, max_inv={:.1}",
            self.base_spread, self.quote_size, self.max_inventory
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_observation(price: f64, volatility: f64) -> Observation {
        Observation {
            mid_price: price,
            spread: 1.0,
            bid_depth: 100.0,
            ask_depth: 100.0,
            imbalance: 0.0,
            last_price: price,
            volume: 1000.0,
            volatility,
            timestamp: 0,
        }
    }

    #[test]
    fn test_market_maker_quotes() {
        let mut agent = MarketMakerAgent::default();
        let state = AgentState::new(100_000.0);
        let obs = create_test_observation(50000.0, 0.02);

        let action = agent.act(&obs, &state);

        match action {
            AgentAction::Quote {
                bid_price,
                ask_price,
                quantity,
            } => {
                assert!(bid_price < 50000.0);
                assert!(ask_price > 50000.0);
                assert!(quantity > 0.0);
                println!(
                    "Bid: {:.2}, Ask: {:.2}, Spread: {:.2}",
                    bid_price,
                    ask_price,
                    ask_price - bid_price
                );
            }
            _ => panic!("Expected Quote action"),
        }
    }

    #[test]
    fn test_inventory_skew() {
        let mut agent = MarketMakerAgent::default();
        let obs = create_test_observation(50000.0, 0.02);

        // With positive inventory
        let mut state = AgentState::new(100_000.0);
        state.inventory = 5.0;
        let action_long = agent.act(&obs, &state);

        // With negative inventory
        state.inventory = -5.0;
        let action_short = agent.act(&obs, &state);

        // Long position should have lower bid (less willing to buy more)
        // Short position should have higher bid (more willing to buy)
        match (action_long, action_short) {
            (
                AgentAction::Quote {
                    bid_price: bid_long,
                    ..
                },
                AgentAction::Quote {
                    bid_price: bid_short,
                    ..
                },
            ) => {
                assert!(bid_long < bid_short);
            }
            _ => {} // May hit inventory limits
        }
    }

    #[test]
    fn test_max_inventory_reduction() {
        let mut agent = MarketMakerAgent::default().with_max_inventory(5.0);
        let obs = create_test_observation(50000.0, 0.02);

        let mut state = AgentState::new(100_000.0);
        state.inventory = 10.0; // Over max

        let action = agent.act(&obs, &state);

        // Should try to reduce inventory
        assert!(matches!(action, AgentAction::MarketSell { .. }));
    }
}
