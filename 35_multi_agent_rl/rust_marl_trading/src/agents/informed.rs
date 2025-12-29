//! Informed trader agent implementation

use rand::Rng;

use super::{Agent, AgentAction, AgentId, AgentState};
use crate::environment::Observation;

/// Informed trader agent
///
/// Has access to information about future price movements.
/// Used for testing robustness of other strategies.
#[derive(Debug, Clone)]
pub struct InformedTrader {
    id: AgentId,
    /// Accuracy of information (0.5 = random, 1.0 = perfect)
    accuracy: f64,
    /// Trade size
    trade_size: f64,
    /// Minimum expected return to trade
    min_edge: f64,
    /// Future return signal (set externally)
    future_signal: f64,
}

impl Default for InformedTrader {
    fn default() -> Self {
        Self {
            id: 0,
            accuracy: 0.8,
            trade_size: 1.0,
            min_edge: 0.005,
            future_signal: 0.0,
        }
    }
}

impl InformedTrader {
    /// Create a new informed trader
    pub fn new(accuracy: f64, trade_size: f64) -> Self {
        Self {
            id: 0,
            accuracy: accuracy.clamp(0.5, 1.0),
            trade_size,
            min_edge: 0.005,
            future_signal: 0.0,
        }
    }

    /// Set the future signal (expected return)
    pub fn set_signal(&mut self, signal: f64) {
        self.future_signal = signal;
    }

    /// Get the expected return based on accuracy
    fn get_expected_return(&self) -> f64 {
        let mut rng = rand::thread_rng();

        if rng.gen::<f64>() < self.accuracy {
            // Correct signal
            self.future_signal
        } else {
            // Noise
            rng.gen_range(-0.02..0.02)
        }
    }
}

impl Agent for InformedTrader {
    fn id(&self) -> AgentId {
        self.id
    }

    fn set_id(&mut self, id: AgentId) {
        self.id = id;
    }

    fn name(&self) -> &str {
        "InformedTrader"
    }

    fn act(&mut self, observation: &Observation, state: &AgentState) -> AgentAction {
        let expected_return = self.get_expected_return();
        let price = observation.mid_price;

        if expected_return > self.min_edge {
            // Expect price to go up - buy
            let size = self.trade_size;
            if state.cash >= size * price {
                return AgentAction::MarketBuy { quantity: size };
            }
        } else if expected_return < -self.min_edge {
            // Expect price to go down - sell
            if state.inventory >= self.trade_size {
                return AgentAction::MarketSell {
                    quantity: self.trade_size,
                };
            } else if state.inventory > 0.0 {
                return AgentAction::MarketSell {
                    quantity: state.inventory,
                };
            }
        }

        AgentAction::Hold
    }

    fn reset(&mut self) {
        self.future_signal = 0.0;
    }

    fn clone_box(&self) -> Box<dyn Agent> {
        Box::new(self.clone())
    }

    fn params_string(&self) -> String {
        format!(
            "accuracy={:.2}, size={:.2}, edge={:.4}",
            self.accuracy, self.trade_size, self.min_edge
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
    fn test_informed_trader_buy() {
        let mut agent = InformedTrader::new(1.0, 1.0); // Perfect information
        agent.set_signal(0.02); // 2% expected return

        let state = AgentState::new(100_000.0);
        let obs = create_test_observation(50000.0);

        let action = agent.act(&obs, &state);
        assert!(matches!(action, AgentAction::MarketBuy { .. }));
    }

    #[test]
    fn test_informed_trader_sell() {
        let mut agent = InformedTrader::new(1.0, 1.0);
        agent.set_signal(-0.02); // -2% expected return

        let mut state = AgentState::new(100_000.0);
        state.inventory = 5.0;
        let obs = create_test_observation(50000.0);

        let action = agent.act(&obs, &state);
        assert!(matches!(action, AgentAction::MarketSell { .. }));
    }

    #[test]
    fn test_informed_trader_hold() {
        let mut agent = InformedTrader::new(1.0, 1.0);
        agent.set_signal(0.001); // Too small edge

        let state = AgentState::new(100_000.0);
        let obs = create_test_observation(50000.0);

        let action = agent.act(&obs, &state);
        assert!(matches!(action, AgentAction::Hold));
    }
}
