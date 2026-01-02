//! Graph Structures for Market Representation
//!
//! Defines the graph structure used to represent cryptocurrency markets
//! where nodes are assets and edges represent correlations/relationships.

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::data::Candle;

/// A node in the market graph representing a cryptocurrency asset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNode {
    /// Node index
    pub idx: usize,

    /// Asset symbol (e.g., "BTCUSDT")
    pub symbol: String,

    /// Node features (technical indicators)
    pub features: Vec<f64>,

    /// Coordinates in embedding space (for equivariance)
    pub coordinates: Vec<f64>,
}

impl GraphNode {
    /// Create a new graph node
    pub fn new(idx: usize, symbol: String, features: Vec<f64>, coordinates: Vec<f64>) -> Self {
        Self {
            idx,
            symbol,
            features,
            coordinates,
        }
    }

    /// Get feature dimension
    pub fn feature_dim(&self) -> usize {
        self.features.len()
    }

    /// Get coordinate dimension
    pub fn coord_dim(&self) -> usize {
        self.coordinates.len()
    }
}

/// An edge in the market graph representing relationship between assets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEdge {
    /// Source node index
    pub source: usize,

    /// Target node index
    pub target: usize,

    /// Edge features (correlation, etc.)
    pub features: Vec<f64>,
}

impl GraphEdge {
    /// Create a new graph edge
    pub fn new(source: usize, target: usize, features: Vec<f64>) -> Self {
        Self {
            source,
            target,
            features,
        }
    }

    /// Get feature dimension
    pub fn feature_dim(&self) -> usize {
        self.features.len()
    }
}

/// A complete graph structure
#[derive(Debug, Clone)]
pub struct Graph {
    /// Nodes in the graph
    pub nodes: Vec<GraphNode>,

    /// Edges in the graph
    pub edges: Vec<GraphEdge>,

    /// Node features as matrix [num_nodes, feature_dim]
    pub node_features: Array2<f64>,

    /// Node coordinates as matrix [num_nodes, coord_dim]
    pub coordinates: Array2<f64>,

    /// Edge index as [2, num_edges] (source, target pairs)
    pub edge_index: Array2<usize>,

    /// Edge features as matrix [num_edges, edge_feature_dim]
    pub edge_features: Array2<f64>,
}

impl Graph {
    /// Create a graph from nodes and edges
    pub fn from_nodes_edges(nodes: Vec<GraphNode>, edges: Vec<GraphEdge>) -> Self {
        let num_nodes = nodes.len();
        let num_edges = edges.len();

        // Get dimensions from first node/edge (assume all same)
        let feature_dim = nodes.first().map(|n| n.feature_dim()).unwrap_or(0);
        let coord_dim = nodes.first().map(|n| n.coord_dim()).unwrap_or(3);
        let edge_feature_dim = edges.first().map(|e| e.feature_dim()).unwrap_or(0);

        // Build matrices
        let mut node_features = Array2::zeros((num_nodes, feature_dim));
        let mut coordinates = Array2::zeros((num_nodes, coord_dim));

        for (i, node) in nodes.iter().enumerate() {
            for (j, &f) in node.features.iter().enumerate() {
                if j < feature_dim {
                    node_features[[i, j]] = f;
                }
            }
            for (j, &c) in node.coordinates.iter().enumerate() {
                if j < coord_dim {
                    coordinates[[i, j]] = c;
                }
            }
        }

        let mut edge_index = Array2::zeros((2, num_edges));
        let mut edge_features = Array2::zeros((num_edges, edge_feature_dim.max(1)));

        for (i, edge) in edges.iter().enumerate() {
            edge_index[[0, i]] = edge.source;
            edge_index[[1, i]] = edge.target;
            for (j, &f) in edge.features.iter().enumerate() {
                if j < edge_feature_dim {
                    edge_features[[i, j]] = f;
                }
            }
        }

        Self {
            nodes,
            edges,
            node_features,
            coordinates,
            edge_index,
            edge_features,
        }
    }

    /// Get number of nodes
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Get number of edges
    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }

    /// Get node feature dimension
    pub fn node_feature_dim(&self) -> usize {
        self.node_features.ncols()
    }

    /// Get coordinate dimension
    pub fn coord_dim(&self) -> usize {
        self.coordinates.ncols()
    }

    /// Get edge feature dimension
    pub fn edge_feature_dim(&self) -> usize {
        self.edge_features.ncols()
    }

    /// Get neighbors of a node
    pub fn neighbors(&self, node_idx: usize) -> Vec<usize> {
        self.edges
            .iter()
            .filter_map(|e| {
                if e.source == node_idx {
                    Some(e.target)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get node by symbol
    pub fn get_node_by_symbol(&self, symbol: &str) -> Option<&GraphNode> {
        self.nodes.iter().find(|n| n.symbol == symbol)
    }

    /// Update node features
    pub fn update_node_features(&mut self, new_features: Array2<f64>) {
        self.node_features = new_features;
    }

    /// Update coordinates
    pub fn update_coordinates(&mut self, new_coords: Array2<f64>) {
        self.coordinates = new_coords;
    }
}

/// Market graph builder
pub struct MarketGraph {
    /// Correlation threshold for edge creation
    correlation_threshold: f64,

    /// Window size for correlation calculation
    window_size: usize,

    /// Coordinate dimension for embedding
    coord_dim: usize,
}

impl MarketGraph {
    /// Create a new market graph builder
    pub fn new(correlation_threshold: f64) -> Self {
        Self {
            correlation_threshold,
            window_size: 168, // 1 week of hourly data
            coord_dim: 3,
        }
    }

    /// Set window size
    pub fn with_window_size(mut self, size: usize) -> Self {
        self.window_size = size;
        self
    }

    /// Set coordinate dimension
    pub fn with_coord_dim(mut self, dim: usize) -> Self {
        self.coord_dim = dim;
        self
    }

    /// Build graph from multi-asset candle data
    pub fn from_candles(&self, candles_map: &HashMap<String, Vec<Candle>>) -> Graph {
        let symbols: Vec<String> = candles_map.keys().cloned().collect();
        let n = symbols.len();

        // Calculate returns for each asset
        let returns: Vec<Vec<f64>> = symbols
            .iter()
            .map(|s| {
                candles_map
                    .get(s)
                    .map(|candles| {
                        candles
                            .windows(2)
                            .map(|w| (w[1].close - w[0].close) / w[0].close)
                            .collect()
                    })
                    .unwrap_or_default()
            })
            .collect();

        // Calculate correlation matrix
        let corr_matrix = self.calculate_correlation_matrix(&returns);

        // Calculate initial coordinates using simple PCA-like embedding
        let coordinates = self.calculate_coordinates(&corr_matrix);

        // Build nodes
        let nodes: Vec<GraphNode> = symbols
            .iter()
            .enumerate()
            .map(|(i, symbol)| {
                let features = self.extract_node_features(candles_map.get(symbol).unwrap());
                let coords = coordinates[i].clone();
                GraphNode::new(i, symbol.clone(), features, coords)
            })
            .collect();

        // Build edges based on correlation
        let mut edges = Vec::new();
        for i in 0..n {
            for j in 0..n {
                if i != j && corr_matrix[[i, j]].abs() > self.correlation_threshold {
                    edges.push(GraphEdge::new(
                        i,
                        j,
                        vec![
                            corr_matrix[[i, j]],
                            corr_matrix[[i, j]].abs(),
                            if corr_matrix[[i, j]] > 0.0 { 1.0 } else { -1.0 },
                        ],
                    ));
                }
            }
        }

        Graph::from_nodes_edges(nodes, edges)
    }

    /// Calculate correlation matrix from returns
    fn calculate_correlation_matrix(&self, returns: &[Vec<f64>]) -> Array2<f64> {
        let n = returns.len();
        let mut corr = Array2::zeros((n, n));

        for i in 0..n {
            for j in 0..n {
                if i == j {
                    corr[[i, j]] = 1.0;
                } else {
                    corr[[i, j]] = self.pearson_correlation(&returns[i], &returns[j]);
                }
            }
        }

        corr
    }

    /// Calculate Pearson correlation
    fn pearson_correlation(&self, x: &[f64], y: &[f64]) -> f64 {
        let n = x.len().min(y.len());
        if n < 2 {
            return 0.0;
        }

        let mean_x: f64 = x.iter().take(n).sum::<f64>() / n as f64;
        let mean_y: f64 = y.iter().take(n).sum::<f64>() / n as f64;

        let mut cov = 0.0;
        let mut var_x = 0.0;
        let mut var_y = 0.0;

        for i in 0..n {
            let dx = x[i] - mean_x;
            let dy = y[i] - mean_y;
            cov += dx * dy;
            var_x += dx * dx;
            var_y += dy * dy;
        }

        let denom = (var_x * var_y).sqrt();
        if denom.abs() < 1e-10 {
            0.0
        } else {
            cov / denom
        }
    }

    /// Calculate initial coordinates from correlation matrix (simple PCA-like)
    fn calculate_coordinates(&self, corr: &Array2<f64>) -> Vec<Vec<f64>> {
        let n = corr.nrows();
        let mut coords = Vec::with_capacity(n);

        // Simple MDS-like embedding using first few eigenvectors
        // For simplicity, we use a heuristic based on correlations
        for i in 0..n {
            let mut coord = vec![0.0; self.coord_dim];

            // Use correlations with first few assets as coordinates
            for k in 0..self.coord_dim.min(n) {
                coord[k] = corr[[i, k]];
            }

            // Normalize
            let norm: f64 = coord.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 1e-10 {
                for c in coord.iter_mut() {
                    *c /= norm;
                }
            }

            coords.push(coord);
        }

        coords
    }

    /// Extract node features from candle data
    fn extract_node_features(&self, candles: &[Candle]) -> Vec<f64> {
        if candles.is_empty() {
            return vec![0.0; 10];
        }

        let returns: Vec<f64> = candles
            .windows(2)
            .map(|w| (w[1].close - w[0].close) / w[0].close)
            .collect();

        let last = candles.last().unwrap();

        // Technical features
        vec![
            // Recent returns
            *returns.last().unwrap_or(&0.0),
            returns.iter().rev().take(24).sum::<f64>(),
            returns.iter().sum::<f64>(),
            // Volatility
            self.std_dev(&returns) * (24.0 * 365.0_f64).sqrt(),
            // Skewness
            self.skewness(&returns),
            // Kurtosis
            self.kurtosis(&returns),
            // Momentum (EMA-like)
            self.exponential_mean(&returns, 12),
            // Volume ratio
            last.volume / candles.iter().map(|c| c.volume).sum::<f64>() * candles.len() as f64,
            // RSI-like
            self.rsi_like(&returns, 14),
            // Close position
            last.close_position(),
        ]
    }

    /// Calculate standard deviation
    fn std_dev(&self, values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }
        let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
        let variance: f64 = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
            / (values.len() - 1) as f64;
        variance.sqrt()
    }

    /// Calculate skewness
    fn skewness(&self, values: &[f64]) -> f64 {
        if values.len() < 3 {
            return 0.0;
        }
        let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
        let std = self.std_dev(values);
        if std.abs() < 1e-10 {
            return 0.0;
        }
        let m3: f64 = values.iter().map(|x| ((x - mean) / std).powi(3)).sum::<f64>()
            / values.len() as f64;
        m3
    }

    /// Calculate kurtosis
    fn kurtosis(&self, values: &[f64]) -> f64 {
        if values.len() < 4 {
            return 0.0;
        }
        let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
        let std = self.std_dev(values);
        if std.abs() < 1e-10 {
            return 0.0;
        }
        let m4: f64 = values.iter().map(|x| ((x - mean) / std).powi(4)).sum::<f64>()
            / values.len() as f64;
        m4 - 3.0 // Excess kurtosis
    }

    /// Calculate exponential weighted mean
    fn exponential_mean(&self, values: &[f64], span: usize) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        let alpha = 2.0 / (span as f64 + 1.0);
        let mut ema = values[0];
        for &v in values.iter().skip(1) {
            ema = alpha * v + (1.0 - alpha) * ema;
        }
        ema
    }

    /// Calculate RSI-like indicator
    fn rsi_like(&self, returns: &[f64], period: usize) -> f64 {
        if returns.len() < period {
            return 50.0;
        }

        let recent: Vec<f64> = returns.iter().rev().take(period).cloned().collect();
        let gains: f64 = recent.iter().filter(|&&r| r > 0.0).sum();
        let losses: f64 = recent.iter().filter(|&&r| r < 0.0).map(|r| r.abs()).sum();

        if losses.abs() < 1e-10 {
            100.0
        } else {
            100.0 - 100.0 / (1.0 + gains / losses)
        }
    }
}

impl Default for MarketGraph {
    fn default() -> Self {
        Self::new(0.3)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_node() {
        let node = GraphNode::new(
            0,
            "BTCUSDT".to_string(),
            vec![0.01, 0.02, 0.5],
            vec![1.0, 0.0, 0.0],
        );
        assert_eq!(node.feature_dim(), 3);
        assert_eq!(node.coord_dim(), 3);
    }

    #[test]
    fn test_graph_construction() {
        let nodes = vec![
            GraphNode::new(0, "BTC".to_string(), vec![0.01], vec![1.0, 0.0, 0.0]),
            GraphNode::new(1, "ETH".to_string(), vec![0.02], vec![0.0, 1.0, 0.0]),
        ];

        let edges = vec![
            GraphEdge::new(0, 1, vec![0.8]),
            GraphEdge::new(1, 0, vec![0.8]),
        ];

        let graph = Graph::from_nodes_edges(nodes, edges);

        assert_eq!(graph.num_nodes(), 2);
        assert_eq!(graph.num_edges(), 2);
    }

    #[test]
    fn test_market_graph_builder() {
        let mut candles_map = HashMap::new();

        candles_map.insert(
            "BTCUSDT".to_string(),
            vec![
                Candle::new(1000, 100.0, 105.0, 98.0, 103.0, 1000.0, 100000.0),
                Candle::new(2000, 103.0, 108.0, 101.0, 106.0, 1100.0, 115000.0),
                Candle::new(3000, 106.0, 110.0, 104.0, 108.0, 1050.0, 112000.0),
            ],
        );

        candles_map.insert(
            "ETHUSDT".to_string(),
            vec![
                Candle::new(1000, 2000.0, 2100.0, 1950.0, 2050.0, 500.0, 1000000.0),
                Candle::new(2000, 2050.0, 2150.0, 2000.0, 2100.0, 550.0, 1100000.0),
                Candle::new(3000, 2100.0, 2200.0, 2050.0, 2150.0, 520.0, 1120000.0),
            ],
        );

        let builder = MarketGraph::new(0.1);
        let graph = builder.from_candles(&candles_map);

        assert_eq!(graph.num_nodes(), 2);
        // Both assets going up together should have correlation > 0.1
        assert!(graph.num_edges() > 0);
    }
}
