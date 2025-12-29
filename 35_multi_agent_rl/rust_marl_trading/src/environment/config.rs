//! Environment configuration

use serde::{Deserialize, Serialize};

/// Configuration for multi-agent environment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiAgentConfig {
    /// Number of agents
    pub n_agents: usize,
    /// Initial cash for each agent
    pub initial_cash: f64,
    /// Initial asset price
    pub initial_price: f64,
    /// Maximum steps per episode
    pub max_steps: usize,
    /// Price volatility (std of returns)
    pub volatility: f64,
    /// Mean reversion strength
    pub mean_reversion: f64,
    /// Fundamental value
    pub fundamental_value: f64,
    /// Transaction cost (as fraction)
    pub transaction_cost: f64,
    /// Price impact coefficient
    pub price_impact: f64,
    /// Reward type
    pub reward_type: RewardType,
    /// Whether to use historical data
    pub use_historical: bool,
}

impl Default for MultiAgentConfig {
    fn default() -> Self {
        Self {
            n_agents: 4,
            initial_cash: 100_000.0,
            initial_price: 50_000.0,
            max_steps: 1000,
            volatility: 0.02,
            mean_reversion: 0.1,
            fundamental_value: 50_000.0,
            transaction_cost: 0.001,
            price_impact: 0.0001,
            reward_type: RewardType::PnL,
            use_historical: false,
        }
    }
}

impl MultiAgentConfig {
    /// Create a config for high-frequency trading simulation
    pub fn hft() -> Self {
        Self {
            volatility: 0.001,
            transaction_cost: 0.0001,
            price_impact: 0.00001,
            max_steps: 10_000,
            ..Default::default()
        }
    }

    /// Create a config for swing trading simulation
    pub fn swing() -> Self {
        Self {
            volatility: 0.03,
            transaction_cost: 0.002,
            price_impact: 0.001,
            max_steps: 252, // ~1 trading year in days
            ..Default::default()
        }
    }

    /// Create a config with specific number of agents
    pub fn with_agents(mut self, n_agents: usize) -> Self {
        self.n_agents = n_agents;
        self
    }

    /// Set volatility
    pub fn with_volatility(mut self, volatility: f64) -> Self {
        self.volatility = volatility;
        self
    }

    /// Set transaction costs
    pub fn with_transaction_cost(mut self, cost: f64) -> Self {
        self.transaction_cost = cost;
        self
    }

    /// Set reward type
    pub fn with_reward_type(mut self, reward_type: RewardType) -> Self {
        self.reward_type = reward_type;
        self
    }
}

/// Type of reward to use
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RewardType {
    /// Raw PnL change
    PnL,
    /// Sharpe-like reward (PnL / volatility)
    SharpeAdjusted,
    /// Log returns
    LogReturns,
    /// Relative ranking among agents
    Ranking,
    /// Zero-sum (relative to mean performance)
    ZeroSum,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = MultiAgentConfig::default();
        assert_eq!(config.n_agents, 4);
        assert_eq!(config.initial_cash, 100_000.0);
    }

    #[test]
    fn test_hft_config() {
        let config = MultiAgentConfig::hft();
        assert!(config.volatility < 0.01);
        assert!(config.max_steps > 1000);
    }

    #[test]
    fn test_builder() {
        let config = MultiAgentConfig::default()
            .with_agents(10)
            .with_volatility(0.05)
            .with_transaction_cost(0.002);

        assert_eq!(config.n_agents, 10);
        assert_eq!(config.volatility, 0.05);
        assert_eq!(config.transaction_cost, 0.002);
    }
}
