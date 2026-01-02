//! Node definitions for the dynamic graph

use ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Unique identifier for a node
pub type NodeId = String;

/// Features associated with a graph node (cryptocurrency asset)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeFeatures {
    /// Current price
    pub price: f64,
    /// Price change percentage (24h)
    pub price_change_24h: f64,
    /// Trading volume (24h)
    pub volume_24h: f64,
    /// Volatility (rolling std dev of returns)
    pub volatility: f64,
    /// RSI indicator
    pub rsi: f64,
    /// MACD signal
    pub macd: f64,
    /// Order book imbalance (-1 to 1)
    pub order_book_imbalance: f64,
    /// Bid-ask spread
    pub spread: f64,
    /// Open interest (for perpetuals)
    pub open_interest: f64,
    /// Funding rate (for perpetuals)
    pub funding_rate: f64,
    /// Last update timestamp (Unix ms)
    pub timestamp: u64,
    /// Raw feature vector for GNN
    #[serde(skip)]
    pub embedding: Option<Array1<f64>>,
}

impl Default for NodeFeatures {
    fn default() -> Self {
        Self {
            price: 0.0,
            price_change_24h: 0.0,
            volume_24h: 0.0,
            volatility: 0.0,
            rsi: 50.0,
            macd: 0.0,
            order_book_imbalance: 0.0,
            spread: 0.0,
            open_interest: 0.0,
            funding_rate: 0.0,
            timestamp: 0,
            embedding: None,
        }
    }
}

impl NodeFeatures {
    /// Create new node features with basic price data
    pub fn new(price: f64, volume: f64, timestamp: u64) -> Self {
        Self {
            price,
            volume_24h: volume,
            timestamp,
            ..Default::default()
        }
    }

    /// Convert features to a fixed-size vector for GNN input
    pub fn to_vector(&self) -> Array1<f64> {
        Array1::from_vec(vec![
            self.normalize_price(),
            self.price_change_24h / 100.0, // Normalize percentage
            self.normalize_volume(),
            self.volatility,
            (self.rsi - 50.0) / 50.0, // Normalize RSI to [-1, 1]
            self.macd.tanh(),          // Bound MACD
            self.order_book_imbalance,
            self.spread.min(0.01) / 0.01, // Cap spread at 1%
            self.normalize_oi(),
            self.funding_rate * 100.0, // Scale funding rate
        ])
    }

    /// Get feature dimension
    pub fn feature_dim() -> usize {
        10
    }

    fn normalize_price(&self) -> f64 {
        // Log-normalize price
        if self.price > 0.0 {
            self.price.ln() / 15.0 // Assuming max ~$1M assets
        } else {
            0.0
        }
    }

    fn normalize_volume(&self) -> f64 {
        // Log-normalize volume
        if self.volume_24h > 0.0 {
            self.volume_24h.ln() / 25.0 // Normalize to ~0-1 range
        } else {
            0.0
        }
    }

    fn normalize_oi(&self) -> f64 {
        if self.open_interest > 0.0 {
            self.open_interest.ln() / 25.0
        } else {
            0.0
        }
    }

    /// Update features from new market data
    pub fn update(&mut self, price: f64, volume: f64, timestamp: u64) {
        let old_price = self.price;
        self.price = price;
        self.volume_24h = volume;
        self.timestamp = timestamp;

        // Update price change
        if old_price > 0.0 {
            self.price_change_24h = (price - old_price) / old_price * 100.0;
        }
    }
}

/// A node in the dynamic graph representing a cryptocurrency
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    /// Unique identifier (e.g., "BTCUSDT")
    pub id: NodeId,
    /// Node features
    pub features: NodeFeatures,
    /// Historical embeddings for temporal analysis
    #[serde(skip)]
    pub history: Vec<Array1<f64>>,
    /// Maximum history length
    pub max_history: usize,
}

impl Node {
    /// Create a new node with given ID and features
    pub fn new(id: impl Into<NodeId>, features: NodeFeatures) -> Self {
        Self {
            id: id.into(),
            features,
            history: Vec::new(),
            max_history: 100,
        }
    }

    /// Update node with new features
    pub fn update_features(&mut self, features: NodeFeatures) {
        // Store current embedding in history
        if let Some(embedding) = &self.features.embedding {
            self.history.push(embedding.clone());
            if self.history.len() > self.max_history {
                self.history.remove(0);
            }
        }
        self.features = features;
    }

    /// Get the current feature vector
    pub fn get_feature_vector(&self) -> Array1<f64> {
        self.features.to_vector()
    }

    /// Set the node embedding
    pub fn set_embedding(&mut self, embedding: Array1<f64>) {
        self.features.embedding = Some(embedding);
    }

    /// Get the current embedding if available
    pub fn get_embedding(&self) -> Option<&Array1<f64>> {
        self.features.embedding.as_ref()
    }
}

impl fmt::Display for Node {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Node({}: ${:.2}, vol={:.0})",
            self.id, self.features.price, self.features.volume_24h
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_creation() {
        let features = NodeFeatures::new(50000.0, 1_000_000.0, 1234567890);
        let node = Node::new("BTCUSDT", features);

        assert_eq!(node.id, "BTCUSDT");
        assert_eq!(node.features.price, 50000.0);
    }

    #[test]
    fn test_feature_vector() {
        let features = NodeFeatures {
            price: 50000.0,
            volume_24h: 1_000_000.0,
            rsi: 70.0,
            ..Default::default()
        };

        let vec = features.to_vector();
        assert_eq!(vec.len(), NodeFeatures::feature_dim());
    }

    #[test]
    fn test_node_update() {
        let mut node = Node::new("ETHUSDT", NodeFeatures::new(3000.0, 500_000.0, 1000));

        node.features.update(3100.0, 600_000.0, 2000);

        assert_eq!(node.features.price, 3100.0);
        assert!(node.features.price_change_24h > 0.0);
    }
}
