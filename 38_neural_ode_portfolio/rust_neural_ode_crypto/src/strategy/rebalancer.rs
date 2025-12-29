//! # Continuous Rebalancer
//!
//! Portfolio rebalancing strategy based on Neural ODE trajectories.

use serde::{Deserialize, Serialize};

use crate::data::Features;
use crate::model::NeuralODEPortfolio;

/// Decision whether to rebalance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RebalanceDecision {
    /// Whether rebalancing is needed
    pub should_rebalance: bool,
    /// Maximum deviation from target
    pub max_deviation: f64,
    /// Current weights
    pub current_weights: Vec<f64>,
    /// Target weights
    pub target_weights: Vec<f64>,
}

/// Single trade
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    /// Asset index
    pub asset_idx: usize,
    /// Asset name
    pub asset_name: String,
    /// Dollar amount (positive = buy, negative = sell)
    pub dollar_amount: f64,
    /// Weight change
    pub weight_change: f64,
}

/// Continuous portfolio rebalancer using Neural ODE
pub struct ContinuousRebalancer {
    /// Neural ODE model
    model: NeuralODEPortfolio,
    /// Threshold for triggering rebalance
    threshold: f64,
    /// Minimum trade size in dollars
    min_trade_size: f64,
    /// Transaction cost (percentage)
    transaction_cost: f64,
    /// Asset names
    asset_names: Vec<String>,
}

impl ContinuousRebalancer {
    /// Create a new rebalancer
    pub fn new(model: NeuralODEPortfolio, threshold: f64) -> Self {
        let n_assets = model.n_assets;
        Self {
            model,
            threshold,
            min_trade_size: 10.0,
            transaction_cost: 0.001, // 0.1%
            asset_names: (0..n_assets).map(|i| format!("Asset_{}", i)).collect(),
        }
    }

    /// Set asset names
    pub fn with_asset_names(mut self, names: Vec<String>) -> Self {
        self.asset_names = names;
        self
    }

    /// Set minimum trade size
    pub fn with_min_trade_size(mut self, size: f64) -> Self {
        self.min_trade_size = size;
        self
    }

    /// Set transaction cost
    pub fn with_transaction_cost(mut self, cost: f64) -> Self {
        self.transaction_cost = cost;
        self
    }

    /// Check if rebalancing is needed
    pub fn check_rebalance(
        &self,
        current_weights: &[f64],
        features: &Features,
    ) -> RebalanceDecision {
        // Get target weights from model
        let target_weights = self.model.get_target_weights(
            current_weights,
            features,
            0.1,  // Short horizon for immediate target
        );

        // Calculate maximum deviation
        let max_deviation = current_weights
            .iter()
            .zip(target_weights.iter())
            .map(|(c, t)| (c - t).abs())
            .fold(0.0_f64, f64::max);

        RebalanceDecision {
            should_rebalance: max_deviation > self.threshold,
            max_deviation,
            current_weights: current_weights.to_vec(),
            target_weights,
        }
    }

    /// Compute trades needed to rebalance
    pub fn compute_trades(
        &self,
        current_weights: &[f64],
        features: &Features,
        portfolio_value: f64,
    ) -> (Vec<Trade>, Vec<f64>) {
        // Get intermediate target (not final, for gradual rebalancing)
        let trajectory = self.model.solve_trajectory(
            current_weights,
            features,
            (0.0, 0.5),  // Half time horizon
            10,
        );

        // Use 20% progress point for gradual rebalancing
        let target_weights = &trajectory[2].1;

        let mut trades = Vec::new();

        for i in 0..current_weights.len() {
            let weight_change = target_weights[i] - current_weights[i];
            let dollar_amount = weight_change * portfolio_value;

            // Only include significant trades
            if dollar_amount.abs() >= self.min_trade_size {
                trades.push(Trade {
                    asset_idx: i,
                    asset_name: self.asset_names.get(i)
                        .cloned()
                        .unwrap_or_else(|| format!("Asset_{}", i)),
                    dollar_amount,
                    weight_change,
                });
            }
        }

        // Sort by absolute size (largest first)
        trades.sort_by(|a, b| {
            b.dollar_amount.abs().partial_cmp(&a.dollar_amount.abs()).unwrap()
        });

        (trades, target_weights.clone())
    }

    /// Calculate total transaction cost for trades
    pub fn calculate_cost(&self, trades: &[Trade]) -> f64 {
        let total_volume: f64 = trades.iter().map(|t| t.dollar_amount.abs()).sum();
        total_volume * self.transaction_cost
    }

    /// Execute rebalancing (simulation)
    pub fn execute_rebalance(
        &self,
        current_weights: &[f64],
        features: &Features,
        portfolio_value: f64,
    ) -> RebalanceResult {
        let decision = self.check_rebalance(current_weights, features);

        if !decision.should_rebalance {
            return RebalanceResult {
                executed: false,
                trades: Vec::new(),
                new_weights: current_weights.to_vec(),
                transaction_cost: 0.0,
                turnover: 0.0,
            };
        }

        let (trades, new_weights) = self.compute_trades(
            current_weights,
            features,
            portfolio_value,
        );

        let transaction_cost = self.calculate_cost(&trades);

        let turnover: f64 = trades.iter().map(|t| t.weight_change.abs()).sum();

        RebalanceResult {
            executed: true,
            trades,
            new_weights,
            transaction_cost,
            turnover,
        }
    }

    /// Get the underlying model
    pub fn model(&self) -> &NeuralODEPortfolio {
        &self.model
    }

    /// Get mutable reference to model
    pub fn model_mut(&mut self) -> &mut NeuralODEPortfolio {
        &mut self.model
    }
}

/// Result of rebalancing operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RebalanceResult {
    /// Whether rebalancing was executed
    pub executed: bool,
    /// List of trades
    pub trades: Vec<Trade>,
    /// New portfolio weights
    pub new_weights: Vec<f64>,
    /// Total transaction cost
    pub transaction_cost: f64,
    /// Total turnover (sum of absolute weight changes)
    pub turnover: f64,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_features() -> Features {
        Features {
            n_assets: 3,
            n_features: 5,
            data: vec![
                vec![0.1, 0.2, 0.3, 0.4, 0.5],
                vec![0.2, 0.3, 0.4, 0.5, 0.6],
                vec![0.3, 0.4, 0.5, 0.6, 0.7],
            ],
            names: vec!["a", "b", "c", "d", "e"]
                .into_iter()
                .map(String::from)
                .collect(),
        }
    }

    #[test]
    fn test_rebalancer_creation() {
        let model = NeuralODEPortfolio::new(3, 5, 8);
        let rebalancer = ContinuousRebalancer::new(model, 0.05);
        assert!((rebalancer.threshold - 0.05).abs() < 1e-10);
    }

    #[test]
    fn test_check_rebalance() {
        let model = NeuralODEPortfolio::new(3, 5, 8);
        let rebalancer = ContinuousRebalancer::new(model, 0.05);

        let features = create_test_features();
        let weights = vec![0.4, 0.35, 0.25];

        let decision = rebalancer.check_rebalance(&weights, &features);
        assert_eq!(decision.current_weights.len(), 3);
        assert_eq!(decision.target_weights.len(), 3);
    }

    #[test]
    fn test_compute_trades() {
        let model = NeuralODEPortfolio::new(3, 5, 8);
        let rebalancer = ContinuousRebalancer::new(model, 0.01)
            .with_min_trade_size(1.0);

        let features = create_test_features();
        let weights = vec![0.5, 0.3, 0.2];

        let (trades, new_weights) = rebalancer.compute_trades(
            &weights,
            &features,
            10000.0,
        );

        // New weights should sum to 1
        let sum: f64 = new_weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_calculate_cost() {
        let model = NeuralODEPortfolio::new(3, 5, 8);
        let rebalancer = ContinuousRebalancer::new(model, 0.05)
            .with_transaction_cost(0.001);

        let trades = vec![
            Trade {
                asset_idx: 0,
                asset_name: "BTC".into(),
                dollar_amount: 1000.0,
                weight_change: 0.1,
            },
            Trade {
                asset_idx: 1,
                asset_name: "ETH".into(),
                dollar_amount: -500.0,
                weight_change: -0.05,
            },
        ];

        let cost = rebalancer.calculate_cost(&trades);
        assert!((cost - 1.5).abs() < 1e-10); // (1000 + 500) * 0.001
    }
}
