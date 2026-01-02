//! Edge definitions for the dynamic graph

use ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::fmt;

use super::NodeId;

/// Unique identifier for an edge
pub type EdgeId = (NodeId, NodeId);

/// Features associated with a graph edge (relationship between assets)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeFeatures {
    /// Correlation coefficient between assets
    pub correlation: f64,
    /// Rolling correlation (shorter window)
    pub short_correlation: f64,
    /// Cointegration score
    pub cointegration: f64,
    /// Granger causality score (source -> target)
    pub granger_causality: f64,
    /// Lead-lag relationship (positive = source leads)
    pub lead_lag: f64,
    /// Mutual information
    pub mutual_information: f64,
    /// Edge age (time since creation)
    pub age: u64,
    /// Last update timestamp
    pub timestamp: u64,
    /// Edge weight (computed importance)
    pub weight: f64,
}

impl Default for EdgeFeatures {
    fn default() -> Self {
        Self {
            correlation: 0.0,
            short_correlation: 0.0,
            cointegration: 0.0,
            granger_causality: 0.0,
            lead_lag: 0.0,
            mutual_information: 0.0,
            age: 0,
            timestamp: 0,
            weight: 1.0,
        }
    }
}

impl EdgeFeatures {
    /// Create new edge features with correlation
    pub fn with_correlation(correlation: f64, timestamp: u64) -> Self {
        Self {
            correlation,
            weight: correlation.abs(),
            timestamp,
            ..Default::default()
        }
    }

    /// Convert features to a fixed-size vector for GNN input
    pub fn to_vector(&self) -> Array1<f64> {
        Array1::from_vec(vec![
            self.correlation,
            self.short_correlation,
            self.cointegration,
            self.granger_causality,
            self.lead_lag.tanh(), // Bound to [-1, 1]
            self.mutual_information.min(1.0), // Cap at 1
            self.time_decay(),
            self.weight,
        ])
    }

    /// Get feature dimension
    pub fn feature_dim() -> usize {
        8
    }

    /// Compute time decay factor
    fn time_decay(&self) -> f64 {
        // Exponential decay based on age
        let decay_rate = 0.0001; // Decay constant
        (-decay_rate * self.age as f64).exp()
    }

    /// Update edge features
    pub fn update(&mut self, correlation: f64, timestamp: u64) {
        self.correlation = correlation;
        self.age = timestamp.saturating_sub(self.timestamp);
        self.timestamp = timestamp;
        self.weight = self.compute_weight();
    }

    /// Compute edge weight based on features
    fn compute_weight(&self) -> f64 {
        let base = self.correlation.abs();
        let recency = self.time_decay();
        let significance = self.mutual_information.min(1.0);

        (base * 0.5 + significance * 0.3 + recency * 0.2).clamp(0.0, 1.0)
    }

    /// Check if edge should be pruned
    pub fn should_prune(&self, threshold: f64) -> bool {
        self.weight < threshold || self.correlation.abs() < 0.1
    }
}

/// An edge in the dynamic graph representing a relationship between assets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    /// Source node ID
    pub source: NodeId,
    /// Target node ID
    pub target: NodeId,
    /// Edge features
    pub features: EdgeFeatures,
    /// Whether edge is bidirectional
    pub bidirectional: bool,
}

impl Edge {
    /// Create a new edge between two nodes
    pub fn new(source: impl Into<NodeId>, target: impl Into<NodeId>, features: EdgeFeatures) -> Self {
        Self {
            source: source.into(),
            target: target.into(),
            features,
            bidirectional: true,
        }
    }

    /// Create a directed edge
    pub fn directed(source: impl Into<NodeId>, target: impl Into<NodeId>, features: EdgeFeatures) -> Self {
        Self {
            source: source.into(),
            target: target.into(),
            features,
            bidirectional: false,
        }
    }

    /// Get edge ID
    pub fn id(&self) -> EdgeId {
        (self.source.clone(), self.target.clone())
    }

    /// Get reverse edge ID
    pub fn reverse_id(&self) -> EdgeId {
        (self.target.clone(), self.source.clone())
    }

    /// Get the feature vector
    pub fn get_feature_vector(&self) -> Array1<f64> {
        self.features.to_vector()
    }

    /// Update edge with new features
    pub fn update(&mut self, correlation: f64, timestamp: u64) {
        self.features.update(correlation, timestamp);
    }
}

impl fmt::Display for Edge {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let arrow = if self.bidirectional { "<->" } else { "->" };
        write!(
            f,
            "Edge({} {} {}: corr={:.3}, w={:.3})",
            self.source, arrow, self.target, self.features.correlation, self.features.weight
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_edge_creation() {
        let features = EdgeFeatures::with_correlation(0.85, 1234567890);
        let edge = Edge::new("BTCUSDT", "ETHUSDT", features);

        assert_eq!(edge.source, "BTCUSDT");
        assert_eq!(edge.target, "ETHUSDT");
        assert!(edge.bidirectional);
    }

    #[test]
    fn test_edge_features_vector() {
        let features = EdgeFeatures {
            correlation: 0.9,
            weight: 0.8,
            ..Default::default()
        };

        let vec = features.to_vector();
        assert_eq!(vec.len(), EdgeFeatures::feature_dim());
    }

    #[test]
    fn test_edge_pruning() {
        let mut features = EdgeFeatures::default();
        features.correlation = 0.05;
        features.weight = 0.05;

        assert!(features.should_prune(0.1));

        features.correlation = 0.9;
        features.weight = 0.9;
        assert!(!features.should_prune(0.1));
    }

    #[test]
    fn test_directed_edge() {
        let features = EdgeFeatures::with_correlation(0.7, 1000);
        let edge = Edge::directed("BTCUSDT", "ETHUSDT", features);

        assert!(!edge.bidirectional);
    }
}
