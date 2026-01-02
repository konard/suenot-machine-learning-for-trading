//! Graph pooling algorithms
//!
//! This module provides implementations of various graph pooling methods
//! for creating hierarchical representations of markets.

mod diffpool;
mod sagpool;
mod mincut;

pub use diffpool::DiffPool;
pub use sagpool::SAGPool;
pub use mincut::MinCutPool;

use crate::graph::MarketGraph;
use ndarray::{Array1, Array2};

/// Result of graph pooling operation
#[derive(Debug, Clone)]
pub struct PoolingResult {
    /// Pooled node features [n_clusters, n_features]
    pub pooled_features: Array2<f64>,

    /// Pooled adjacency matrix [n_clusters, n_clusters]
    pub pooled_adjacency: Array2<f64>,

    /// Soft cluster assignments [n_nodes, n_clusters]
    pub cluster_assignments: Array2<f64>,

    /// Pooling loss (for training)
    pub loss: f64,
}

impl PoolingResult {
    /// Get hard cluster assignments (argmax)
    pub fn hard_assignments(&self) -> Vec<usize> {
        let n = self.cluster_assignments.nrows();
        (0..n)
            .map(|i| {
                let row = self.cluster_assignments.row(i);
                row.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap_or(0)
            })
            .collect()
    }

    /// Get cluster sizes
    pub fn cluster_sizes(&self) -> Vec<f64> {
        let n_clusters = self.cluster_assignments.ncols();
        (0..n_clusters)
            .map(|c| self.cluster_assignments.column(c).sum())
            .collect()
    }

    /// Calculate cluster entropy (measure of assignment uncertainty)
    pub fn entropy(&self) -> f64 {
        let eps = 1e-10;
        -self
            .cluster_assignments
            .iter()
            .map(|&p| {
                let p_safe = p.max(eps);
                p_safe * p_safe.ln()
            })
            .sum::<f64>()
            / self.cluster_assignments.nrows() as f64
    }
}

/// Hierarchical pooling with multiple levels
pub struct HierarchicalPooling {
    /// Number of clusters at level 1
    n_clusters_l1: usize,

    /// Number of clusters at level 2
    n_clusters_l2: usize,

    /// Pooling method for level 1
    pool_l1: DiffPool,

    /// Pooling method for level 2
    pool_l2: DiffPool,
}

impl HierarchicalPooling {
    /// Create a new hierarchical pooling module
    pub fn new(n_clusters_l1: usize, n_clusters_l2: usize) -> Self {
        Self {
            n_clusters_l1,
            n_clusters_l2,
            pool_l1: DiffPool::new(n_clusters_l1),
            pool_l2: DiffPool::new(n_clusters_l2),
        }
    }

    /// Forward pass through hierarchical pooling
    pub fn forward(&self, graph: &MarketGraph) -> anyhow::Result<HierarchicalResult> {
        // Level 1: Asset -> Clusters
        let result_l1 = self.pool_l1.forward(&graph.features, &graph.adjacency)?;

        // Level 2: Clusters -> Super-clusters
        let result_l2 = self
            .pool_l2
            .forward(&result_l1.pooled_features, &result_l1.pooled_adjacency)?;

        // Market-level representation (mean of super-clusters)
        let market_embedding = result_l2
            .pooled_features
            .mean_axis(ndarray::Axis(0))
            .unwrap();

        Ok(HierarchicalResult {
            level1: result_l1,
            level2: result_l2,
            market_embedding,
        })
    }

    /// Get cluster assignments for original assets
    pub fn get_asset_clusters(&self, graph: &MarketGraph) -> anyhow::Result<Vec<usize>> {
        let result = self.forward(graph)?;
        Ok(result.level1.hard_assignments())
    }
}

/// Result of hierarchical pooling
#[derive(Debug, Clone)]
pub struct HierarchicalResult {
    /// Level 1 pooling result (assets -> clusters)
    pub level1: PoolingResult,

    /// Level 2 pooling result (clusters -> super-clusters)
    pub level2: PoolingResult,

    /// Market-level embedding
    pub market_embedding: Array1<f64>,
}

impl HierarchicalResult {
    /// Get total pooling loss
    pub fn total_loss(&self) -> f64 {
        self.level1.loss + self.level2.loss
    }

    /// Get asset to cluster mapping
    pub fn asset_clusters(&self) -> Vec<usize> {
        self.level1.hard_assignments()
    }

    /// Get cluster to super-cluster mapping
    pub fn cluster_superclusters(&self) -> Vec<usize> {
        self.level2.hard_assignments()
    }

    /// Get asset to super-cluster mapping (transitive)
    pub fn asset_superclusters(&self) -> Vec<usize> {
        let asset_to_cluster = self.asset_clusters();
        let cluster_to_super = self.cluster_superclusters();

        asset_to_cluster
            .iter()
            .map(|&c| cluster_to_super.get(c).copied().unwrap_or(0))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use crate::api::Kline;

    fn create_test_klines(symbol: &str, n: usize) -> Vec<Kline> {
        (0..n)
            .map(|i| Kline {
                symbol: symbol.to_string(),
                open_time: 1000 * i as i64,
                open: 100.0 + i as f64,
                high: 101.0 + i as f64,
                low: 99.0 + i as f64,
                close: 100.5 + i as f64,
                volume: 1000.0,
                turnover: 100000.0,
            })
            .collect()
    }

    #[test]
    fn test_hierarchical_pooling() {
        let mut klines = HashMap::new();
        for i in 0..10 {
            klines.insert(format!("ASSET{}", i), create_test_klines(&format!("ASSET{}", i), 50));
        }

        let graph = MarketGraph::from_klines(&klines).unwrap();
        let pooling = HierarchicalPooling::new(3, 2);
        let result = pooling.forward(&graph);

        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.level1.pooled_features.nrows(), 3);
        assert_eq!(result.level2.pooled_features.nrows(), 2);
    }
}
