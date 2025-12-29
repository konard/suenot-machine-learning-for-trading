//! Graph construction module for cryptocurrency networks.

mod correlation;
mod knn;
mod temporal;

pub use correlation::CorrelationGraph;
pub use knn::KNNGraph;
pub use temporal::TemporalGraph;

use petgraph::graph::{Graph, NodeIndex};
use std::collections::HashMap;

/// Cryptocurrency graph structure.
#[derive(Debug, Clone)]
pub struct CryptoGraph {
    /// Internal petgraph representation
    pub graph: Graph<String, f64>,
    /// Mapping from symbol to node index
    pub symbol_to_node: HashMap<String, NodeIndex>,
    /// Mapping from node index to symbol
    pub node_to_symbol: HashMap<NodeIndex, String>,
}

impl CryptoGraph {
    /// Create a new empty crypto graph.
    pub fn new() -> Self {
        Self {
            graph: Graph::new(),
            symbol_to_node: HashMap::new(),
            node_to_symbol: HashMap::new(),
        }
    }

    /// Add a node (cryptocurrency) to the graph.
    pub fn add_node(&mut self, symbol: &str) -> NodeIndex {
        if let Some(&idx) = self.symbol_to_node.get(symbol) {
            return idx;
        }
        let idx = self.graph.add_node(symbol.to_string());
        self.symbol_to_node.insert(symbol.to_string(), idx);
        self.node_to_symbol.insert(idx, symbol.to_string());
        idx
    }

    /// Add an edge (relationship) between two cryptocurrencies.
    pub fn add_edge(&mut self, from: &str, to: &str, weight: f64) {
        let from_idx = self.add_node(from);
        let to_idx = self.add_node(to);
        self.graph.add_edge(from_idx, to_idx, weight);
    }

    /// Get the number of nodes.
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Get the number of edges.
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// Get neighbors of a node.
    pub fn neighbors(&self, symbol: &str) -> Vec<String> {
        if let Some(&idx) = self.symbol_to_node.get(symbol) {
            self.graph
                .neighbors(idx)
                .filter_map(|n| self.node_to_symbol.get(&n).cloned())
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Get edge weight between two nodes.
    pub fn edge_weight(&self, from: &str, to: &str) -> Option<f64> {
        let from_idx = self.symbol_to_node.get(from)?;
        let to_idx = self.symbol_to_node.get(to)?;
        self.graph.find_edge(*from_idx, *to_idx).map(|e| {
            *self.graph.edge_weight(e).unwrap()
        })
    }

    /// Get degree (number of connections) of a node.
    pub fn degree(&self, symbol: &str) -> usize {
        if let Some(&idx) = self.symbol_to_node.get(symbol) {
            self.graph.neighbors(idx).count()
        } else {
            0
        }
    }

    /// Convert to edge index format for GNN.
    /// Returns (source_indices, target_indices) as Vec<i64>.
    pub fn to_edge_index(&self) -> (Vec<i64>, Vec<i64>) {
        let mut sources = Vec::new();
        let mut targets = Vec::new();

        for edge in self.graph.edge_indices() {
            if let Some((source, target)) = self.graph.edge_endpoints(edge) {
                sources.push(source.index() as i64);
                targets.push(target.index() as i64);
                // Add reverse edge for undirected graph
                sources.push(target.index() as i64);
                targets.push(source.index() as i64);
            }
        }

        (sources, targets)
    }

    /// Get edge weights as a vector.
    pub fn get_edge_weights(&self) -> Vec<f64> {
        self.graph
            .edge_indices()
            .flat_map(|e| {
                let w = *self.graph.edge_weight(e).unwrap();
                vec![w, w] // Duplicate for both directions
            })
            .collect()
    }

    /// Get all symbols in the graph.
    pub fn symbols(&self) -> Vec<String> {
        self.symbol_to_node.keys().cloned().collect()
    }

    /// Calculate density of the graph.
    pub fn density(&self) -> f64 {
        let n = self.node_count() as f64;
        let e = self.edge_count() as f64;
        if n <= 1.0 {
            0.0
        } else {
            (2.0 * e) / (n * (n - 1.0))
        }
    }

    /// Find hub nodes (high degree centrality).
    pub fn find_hubs(&self, top_k: usize) -> Vec<(String, usize)> {
        let mut degrees: Vec<(String, usize)> = self
            .symbols()
            .iter()
            .map(|s| (s.clone(), self.degree(s)))
            .collect();
        degrees.sort_by(|a, b| b.1.cmp(&a.1));
        degrees.into_iter().take(top_k).collect()
    }
}

impl Default for CryptoGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait for graph builders.
pub trait GraphBuilder {
    /// Build a graph from returns data.
    fn build(&self, returns: &[Vec<f64>], symbols: &[String]) -> CryptoGraph;
}

/// Lead-lag pair information.
#[derive(Debug, Clone)]
pub struct LeadLagPair {
    /// Leading symbol
    pub leader: String,
    /// Lagging symbol
    pub lagger: String,
    /// Lag in periods
    pub lag: i32,
    /// P-value from Granger causality test
    pub pvalue: f64,
    /// Correlation at optimal lag
    pub correlation: f64,
}

/// Detect lead-lag relationships using cross-correlation.
pub fn detect_lead_lag(
    returns: &[Vec<f64>],
    symbols: &[String],
    max_lag: usize,
) -> Vec<LeadLagPair> {
    let mut pairs = Vec::new();

    for i in 0..symbols.len() {
        for j in 0..symbols.len() {
            if i == j {
                continue;
            }

            // Find optimal lag
            let (best_lag, best_corr) = find_optimal_lag(&returns[i], &returns[j], max_lag);

            // If significant lag found (i leads j)
            if best_lag > 0 && best_corr.abs() > 0.3 {
                pairs.push(LeadLagPair {
                    leader: symbols[i].clone(),
                    lagger: symbols[j].clone(),
                    lag: best_lag as i32,
                    pvalue: 0.01, // Placeholder - implement proper test
                    correlation: best_corr,
                });
            }
        }
    }

    // Sort by correlation strength
    pairs.sort_by(|a, b| b.correlation.abs().partial_cmp(&a.correlation.abs()).unwrap());
    pairs
}

/// Find optimal lag using cross-correlation.
fn find_optimal_lag(x: &[f64], y: &[f64], max_lag: usize) -> (usize, f64) {
    let mut best_lag = 0;
    let mut best_corr = 0.0;

    for lag in 0..=max_lag {
        if lag >= x.len() || lag >= y.len() {
            break;
        }

        let x_lagged = &x[..x.len() - lag];
        let y_current = &y[lag..];

        let min_len = x_lagged.len().min(y_current.len());
        if min_len < 10 {
            continue;
        }

        let corr = crate::data::features::pearson_correlation(
            &x_lagged[..min_len],
            &y_current[..min_len],
        );

        if corr.abs() > best_corr.abs() {
            best_corr = corr;
            best_lag = lag;
        }
    }

    (best_lag, best_corr)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crypto_graph() {
        let mut graph = CryptoGraph::new();
        graph.add_edge("BTC", "ETH", 0.8);
        graph.add_edge("ETH", "SOL", 0.6);

        assert_eq!(graph.node_count(), 3);
        assert_eq!(graph.edge_count(), 2);
        assert_eq!(graph.neighbors("ETH").len(), 2);
    }

    #[test]
    fn test_to_edge_index() {
        let mut graph = CryptoGraph::new();
        graph.add_edge("A", "B", 1.0);

        let (sources, targets) = graph.to_edge_index();
        assert_eq!(sources.len(), 2); // Both directions
        assert_eq!(targets.len(), 2);
    }
}
