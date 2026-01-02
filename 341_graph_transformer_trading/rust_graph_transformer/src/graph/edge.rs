//! Graph edge representation for asset relationships
//!
//! Edges represent connections between cryptocurrency assets.

use ndarray::Array1;
use serde::{Deserialize, Serialize};

/// Type of edge connection between assets
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EdgeType {
    /// Correlation-based edge (assets move together)
    Correlation,
    /// Same sector classification
    SameSector,
    /// Cross-exchange relationship
    CrossExchange,
    /// On-chain relationship (token transfers, DEX pairs)
    OnChain,
    /// Lead-lag relationship (one leads, other follows)
    LeadLag,
    /// Inverse correlation (assets move opposite)
    InverseCorrelation,
    /// Custom edge type
    Custom,
}

impl EdgeType {
    /// Get a weight multiplier for this edge type
    pub fn weight_multiplier(&self) -> f64 {
        match self {
            EdgeType::Correlation => 1.0,
            EdgeType::SameSector => 0.8,
            EdgeType::LeadLag => 0.9,
            EdgeType::InverseCorrelation => -1.0,
            EdgeType::OnChain => 0.7,
            EdgeType::CrossExchange => 0.5,
            EdgeType::Custom => 1.0,
        }
    }

    /// Check if this edge type indicates positive relationship
    pub fn is_positive(&self) -> bool {
        !matches!(self, EdgeType::InverseCorrelation)
    }
}

/// Edge in the market graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketEdge {
    /// Source node index
    pub source: usize,
    /// Target node index
    pub target: usize,
    /// Edge type
    pub edge_type: EdgeType,
    /// Edge weight (e.g., correlation coefficient)
    pub weight: f64,
    /// Edge features
    #[serde(skip)]
    pub features: Option<Array1<f64>>,
}

impl MarketEdge {
    /// Create a new edge
    pub fn new(source: usize, target: usize, edge_type: EdgeType, weight: f64) -> Self {
        Self {
            source,
            target,
            edge_type,
            weight,
            features: None,
        }
    }

    /// Create a correlation edge
    pub fn correlation(source: usize, target: usize, correlation: f64) -> Self {
        let edge_type = if correlation < 0.0 {
            EdgeType::InverseCorrelation
        } else {
            EdgeType::Correlation
        };
        Self::new(source, target, edge_type, correlation.abs())
    }

    /// Create a sector edge
    pub fn same_sector(source: usize, target: usize) -> Self {
        Self::new(source, target, EdgeType::SameSector, 1.0)
    }

    /// Create a lead-lag edge
    pub fn lead_lag(leader: usize, follower: usize, lag_correlation: f64) -> Self {
        Self::new(leader, follower, EdgeType::LeadLag, lag_correlation)
    }

    /// Set edge features
    pub fn with_features(mut self, features: Array1<f64>) -> Self {
        self.features = Some(features);
        self
    }

    /// Get the effective weight (weight * type multiplier)
    pub fn effective_weight(&self) -> f64 {
        self.weight * self.edge_type.weight_multiplier()
    }

    /// Check if edge connects two specific nodes (undirected)
    pub fn connects(&self, node_a: usize, node_b: usize) -> bool {
        (self.source == node_a && self.target == node_b)
            || (self.source == node_b && self.target == node_a)
    }

    /// Check if edge is from a specific node
    pub fn is_from(&self, node: usize) -> bool {
        self.source == node
    }

    /// Check if edge is to a specific node
    pub fn is_to(&self, node: usize) -> bool {
        self.target == node
    }

    /// Get the other node in the edge
    pub fn other_node(&self, node: usize) -> Option<usize> {
        if self.source == node {
            Some(self.target)
        } else if self.target == node {
            Some(self.source)
        } else {
            None
        }
    }

    /// Create edge features from weight and type
    pub fn compute_features(&self, feature_dim: usize) -> Array1<f64> {
        let mut features = Array1::zeros(feature_dim);

        // Weight feature
        if feature_dim > 0 {
            features[0] = self.weight;
        }

        // Effective weight
        if feature_dim > 1 {
            features[1] = self.effective_weight();
        }

        // Edge type one-hot encoding
        let type_offset = 2;
        let type_idx = match self.edge_type {
            EdgeType::Correlation => 0,
            EdgeType::SameSector => 1,
            EdgeType::LeadLag => 2,
            EdgeType::InverseCorrelation => 3,
            EdgeType::OnChain => 4,
            EdgeType::CrossExchange => 5,
            EdgeType::Custom => 6,
        };
        if type_offset + type_idx < feature_dim {
            features[type_offset + type_idx] = 1.0;
        }

        features
    }
}

/// Edge list representation for efficient graph operations
#[derive(Debug, Clone, Default)]
pub struct EdgeList {
    /// Source indices
    pub sources: Vec<usize>,
    /// Target indices
    pub targets: Vec<usize>,
    /// Edge weights
    pub weights: Vec<f64>,
    /// Edge types
    pub types: Vec<EdgeType>,
}

impl EdgeList {
    /// Create a new empty edge list
    pub fn new() -> Self {
        Self::default()
    }

    /// Create from a vector of edges
    pub fn from_edges(edges: &[MarketEdge]) -> Self {
        let n = edges.len();
        let mut list = Self {
            sources: Vec::with_capacity(n),
            targets: Vec::with_capacity(n),
            weights: Vec::with_capacity(n),
            types: Vec::with_capacity(n),
        };

        for edge in edges {
            list.sources.push(edge.source);
            list.targets.push(edge.target);
            list.weights.push(edge.weight);
            list.types.push(edge.edge_type);
        }

        list
    }

    /// Add an edge
    pub fn add(&mut self, source: usize, target: usize, weight: f64, edge_type: EdgeType) {
        self.sources.push(source);
        self.targets.push(target);
        self.weights.push(weight);
        self.types.push(edge_type);
    }

    /// Number of edges
    pub fn len(&self) -> usize {
        self.sources.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.sources.is_empty()
    }

    /// Get edge at index
    pub fn get(&self, idx: usize) -> Option<(usize, usize, f64, EdgeType)> {
        if idx < self.len() {
            Some((
                self.sources[idx],
                self.targets[idx],
                self.weights[idx],
                self.types[idx],
            ))
        } else {
            None
        }
    }

    /// Get edges from a specific node
    pub fn edges_from(&self, node: usize) -> Vec<(usize, f64)> {
        self.sources
            .iter()
            .enumerate()
            .filter(|(_, &s)| s == node)
            .map(|(i, _)| (self.targets[i], self.weights[i]))
            .collect()
    }

    /// Get edges to a specific node
    pub fn edges_to(&self, node: usize) -> Vec<(usize, f64)> {
        self.targets
            .iter()
            .enumerate()
            .filter(|(_, &t)| t == node)
            .map(|(i, _)| (self.sources[i], self.weights[i]))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_edge_creation() {
        let edge = MarketEdge::new(0, 1, EdgeType::Correlation, 0.85);
        assert_eq!(edge.source, 0);
        assert_eq!(edge.target, 1);
        assert_eq!(edge.weight, 0.85);
    }

    #[test]
    fn test_correlation_edge() {
        let pos_edge = MarketEdge::correlation(0, 1, 0.8);
        assert_eq!(pos_edge.edge_type, EdgeType::Correlation);

        let neg_edge = MarketEdge::correlation(0, 1, -0.7);
        assert_eq!(neg_edge.edge_type, EdgeType::InverseCorrelation);
    }

    #[test]
    fn test_edge_list() {
        let mut list = EdgeList::new();
        list.add(0, 1, 0.8, EdgeType::Correlation);
        list.add(0, 2, 0.7, EdgeType::SameSector);
        list.add(1, 2, 0.6, EdgeType::Correlation);

        assert_eq!(list.len(), 3);

        let from_0 = list.edges_from(0);
        assert_eq!(from_0.len(), 2);
    }
}
