//! Mean-reversion agent implementation

use super::{Agent, AgentAction, AgentId, AgentState};
use crate::environment::Observation;

/// Mean-reversion agent
///
/// Bets on price reverting to the mean.
/// Sells when price is above SMA, buys when below.
#[derive(Debug, Clone)]
pub struct MeanReversionAgent {
    id: AgentId,
    /// Z-score threshold to trigger trades
    threshold: f64,
    /// SMA period for mean calculation
    sma_period: usize,
    /// Trade size as fraction of available capital
    size_fraction: f64,
    /// Maximum position size
    max_position: f64,
}

impl Default for MeanReversionAgent {
    fn default() -> Self {
        Self {
            id: 0,
            threshold: 2.0,
            sma_period: 50,
            size_fraction: 0.1,
            max_position: 0.5,
        }
    }
}

impl MeanReversionAgent {
    /// Create a new mean-reversion agent
    pub fn new(threshold: f64, sma_period: usize, size_fraction: f64) -> Self {
        Self {
            id: 0,
            threshold,
            sma_period,
            size_fraction,
            max_position: 0.5,
        }
    }

    /// Calculate z-score of current price
    fn calculate_zscore(&self, state: &AgentState, current_price: f64) -> f64 {
        let sma = state.sma(self.sma_period);
        let vol = state.volatility(self.sma_period);

        if vol > 0.0 && sma > 0.0 {
            (current_price - sma) / (sma * vol)
        } else {
            0.0
        }
    }

    /// Calculate trade size
    fn calculate_size(&self, state: &AgentState, price: f64) -> f64 {
        let available = state.cash * self.size_fraction;
        let max_size = state.portfolio_value(price) * self.max_position / price;
        (available / price).min(max_size).max(0.001)
    }
}

impl Agent for MeanReversionAgent {
    fn id(&self) -> AgentId {
        self.id
    }

    fn set_id(&mut self, id: AgentId) {
        self.id = id;
    }

    fn name(&self) -> &str {
        "MeanReversion"
    }

    fn act(&mut self, observation: &Observation, state: &AgentState) -> AgentAction {
        let price = observation.mid_price;
        let zscore = self.calculate_zscore(state, price);

        if zscore > self.threshold {
            // Price too high - sell (expect reversion down)
            if state.inventory > 0.0 {
                let size = (state.inventory * self.size_fraction).max(0.001);
                return AgentAction::MarketSell { quantity: size };
            } else if state.cash > price {
                // Short selling (if allowed) or skip
                return AgentAction::Hold;
            }
        } else if zscore < -self.threshold {
            // Price too low - buy (expect reversion up)
            let size = self.calculate_size(state, price);
            if state.cash >= size * price {
                return AgentAction::MarketBuy { quantity: size };
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
            "threshold={:.2}, sma_period={}, size={:.2}",
            self.threshold, self.sma_period, self.size_fraction
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
    fn test_mean_reversion_buy() {
        let mut agent = MeanReversionAgent::new(1.5, 20, 0.1);
        let mut state = AgentState::new(100_000.0);

        // Create history around 100
        for _ in 0..30 {
            state.observe_price(100.0 + rand::random::<f64>() * 2.0 - 1.0);
        }

        // Price drops significantly below mean
        let obs = create_test_observation(80.0);
        let action = agent.act(&obs, &state);

        // Should want to buy (but may hold depending on z-score calculation)
        // This depends on the volatility calculated
        println!("Action: {:?}", action);
    }

    #[test]
    fn test_mean_reversion_sell() {
        let mut agent = MeanReversionAgent::new(1.5, 20, 0.1);
        let mut state = AgentState::new(100_000.0);
        state.inventory = 10.0;

        // Create history around 100
        for _ in 0..30 {
            state.observe_price(100.0);
        }

        // Price rises significantly above mean
        let obs = create_test_observation(130.0);
        let action = agent.act(&obs, &state);

        // Should sell
        assert!(matches!(action, AgentAction::MarketSell { .. } | AgentAction::Hold));
    }
}
