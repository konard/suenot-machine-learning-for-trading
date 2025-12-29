//! Observation types for agents

use serde::{Deserialize, Serialize};

/// Observation visible to an agent
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Observation {
    /// Current mid price
    pub mid_price: f64,
    /// Current bid-ask spread
    pub spread: f64,
    /// Total bid depth
    pub bid_depth: f64,
    /// Total ask depth
    pub ask_depth: f64,
    /// Order book imbalance (-1 to 1)
    pub imbalance: f64,
    /// Last traded price
    pub last_price: f64,
    /// Recent trading volume
    pub volume: f64,
    /// Recent volatility
    pub volatility: f64,
    /// Current timestamp (step number)
    pub timestamp: usize,
}

impl Observation {
    /// Create a new observation
    pub fn new(
        mid_price: f64,
        spread: f64,
        bid_depth: f64,
        ask_depth: f64,
        volume: f64,
        volatility: f64,
        timestamp: usize,
    ) -> Self {
        let total_depth = bid_depth + ask_depth;
        let imbalance = if total_depth > 0.0 {
            (bid_depth - ask_depth) / total_depth
        } else {
            0.0
        };

        Self {
            mid_price,
            spread,
            bid_depth,
            ask_depth,
            imbalance,
            last_price: mid_price,
            volume,
            volatility,
            timestamp,
        }
    }

    /// Convert to feature vector for ML models
    pub fn to_features(&self) -> Vec<f64> {
        vec![
            self.mid_price / 50000.0, // Normalized price assumption
            self.spread / self.mid_price,
            self.bid_depth / 100.0,
            self.ask_depth / 100.0,
            self.imbalance,
            self.volume / 1000.0,
            self.volatility,
        ]
    }

    /// Get normalized features with custom normalization
    pub fn to_normalized_features(
        &self,
        price_norm: f64,
        depth_norm: f64,
        volume_norm: f64,
    ) -> Vec<f64> {
        vec![
            self.mid_price / price_norm,
            self.spread / self.mid_price,
            self.bid_depth / depth_norm,
            self.ask_depth / depth_norm,
            self.imbalance,
            self.volume / volume_norm,
            self.volatility,
        ]
    }

    /// Best bid price
    pub fn best_bid(&self) -> f64 {
        self.mid_price - self.spread / 2.0
    }

    /// Best ask price
    pub fn best_ask(&self) -> f64 {
        self.mid_price + self.spread / 2.0
    }
}

/// Extended observation with additional information
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExtendedObservation {
    /// Base observation
    pub base: Observation,
    /// Agent's current inventory
    pub inventory: f64,
    /// Agent's current cash
    pub cash: f64,
    /// Agent's unrealized PnL
    pub unrealized_pnl: f64,
    /// Agent's realized PnL
    pub realized_pnl: f64,
    /// Number of other agents (for context)
    pub n_agents: usize,
    /// Recent price changes
    pub price_changes: Vec<f64>,
}

impl ExtendedObservation {
    /// Create extended observation
    pub fn new(base: Observation, inventory: f64, cash: f64, n_agents: usize) -> Self {
        Self {
            base,
            inventory,
            cash,
            unrealized_pnl: 0.0,
            realized_pnl: 0.0,
            n_agents,
            price_changes: Vec::new(),
        }
    }

    /// Get portfolio value at current price
    pub fn portfolio_value(&self) -> f64 {
        self.cash + self.inventory * self.base.mid_price
    }

    /// Convert to feature vector
    pub fn to_features(&self) -> Vec<f64> {
        let mut features = self.base.to_features();

        // Add agent-specific features
        features.push(self.inventory / 10.0);
        features.push(self.cash / 100_000.0);
        features.push(self.unrealized_pnl / 10_000.0);

        // Add price momentum features
        if !self.price_changes.is_empty() {
            let momentum: f64 = self.price_changes.iter().sum::<f64>() / self.price_changes.len() as f64;
            features.push(momentum);
        } else {
            features.push(0.0);
        }

        features
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_observation_creation() {
        let obs = Observation::new(
            50000.0, // mid_price
            10.0,    // spread
            100.0,   // bid_depth
            100.0,   // ask_depth
            1000.0,  // volume
            0.02,    // volatility
            0,       // timestamp
        );

        assert_eq!(obs.mid_price, 50000.0);
        assert_eq!(obs.imbalance, 0.0); // Equal depth
        assert_eq!(obs.best_bid(), 49995.0);
        assert_eq!(obs.best_ask(), 50005.0);
    }

    #[test]
    fn test_imbalance_calculation() {
        let obs = Observation::new(
            50000.0,
            10.0,
            150.0, // More bids
            50.0,  // Fewer asks
            1000.0,
            0.02,
            0,
        );

        assert!(obs.imbalance > 0.0); // Positive = more buying pressure
        assert!((obs.imbalance - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_to_features() {
        let obs = Observation::new(50000.0, 10.0, 100.0, 100.0, 1000.0, 0.02, 0);

        let features = obs.to_features();
        assert_eq!(features.len(), 7);
    }

    #[test]
    fn test_extended_observation() {
        let base = Observation::new(50000.0, 10.0, 100.0, 100.0, 1000.0, 0.02, 0);
        let ext = ExtendedObservation::new(base, 5.0, 75_000.0, 4);

        assert_eq!(ext.portfolio_value(), 75_000.0 + 5.0 * 50_000.0);

        let features = ext.to_features();
        assert!(features.len() > 7); // More features than base
    }
}
