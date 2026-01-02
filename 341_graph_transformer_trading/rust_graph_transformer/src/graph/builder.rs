//! Market graph builder
//!
//! Constructs market graphs from cryptocurrency data.

use super::edge::{EdgeType, MarketEdge};
use super::market_graph::MarketGraph;
use super::node::{CryptoNode, CryptoSector};
use crate::api::types::{Kline, Ticker};
use crate::utils::math::correlation;
use ndarray::Array1;
use std::collections::HashMap;

/// Configuration for graph building
#[derive(Debug, Clone)]
pub struct GraphBuilderConfig {
    /// Minimum correlation threshold for creating edges
    pub correlation_threshold: f64,
    /// Window size for correlation calculation
    pub correlation_window: usize,
    /// Whether to include sector edges
    pub include_sector_edges: bool,
    /// Whether to include negative correlations
    pub include_negative_correlations: bool,
    /// Maximum number of edges per node (None = unlimited)
    pub max_edges_per_node: Option<usize>,
}

impl Default for GraphBuilderConfig {
    fn default() -> Self {
        Self {
            correlation_threshold: 0.5,
            correlation_window: 30,
            include_sector_edges: true,
            include_negative_correlations: true,
            max_edges_per_node: None,
        }
    }
}

/// Builder for constructing market graphs
pub struct MarketGraphBuilder {
    config: GraphBuilderConfig,
}

impl Default for MarketGraphBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl MarketGraphBuilder {
    /// Create a new graph builder with default config
    pub fn new() -> Self {
        Self {
            config: GraphBuilderConfig::default(),
        }
    }

    /// Create with custom config
    pub fn with_config(config: GraphBuilderConfig) -> Self {
        Self { config }
    }

    /// Set correlation threshold
    pub fn correlation_threshold(mut self, threshold: f64) -> Self {
        self.config.correlation_threshold = threshold;
        self
    }

    /// Set correlation window
    pub fn correlation_window(mut self, window: usize) -> Self {
        self.config.correlation_window = window;
        self
    }

    /// Enable/disable sector edges
    pub fn include_sector_edges(mut self, include: bool) -> Self {
        self.config.include_sector_edges = include;
        self
    }

    /// Enable/disable negative correlations
    pub fn include_negative_correlations(mut self, include: bool) -> Self {
        self.config.include_negative_correlations = include;
        self
    }

    /// Set max edges per node
    pub fn max_edges_per_node(mut self, max: usize) -> Self {
        self.config.max_edges_per_node = Some(max);
        self
    }

    /// Build graph from tickers (snapshot-based, no historical data)
    pub fn build_from_tickers(&self, tickers: &[Ticker]) -> MarketGraph {
        let mut graph = MarketGraph::new();

        // Create nodes
        for (i, ticker) in tickers.iter().enumerate() {
            let node = CryptoNode::with_data(
                i,
                &ticker.symbol,
                ticker.last_price,
                ticker.price_change_24h * 100.0,  // Convert to percentage
                ticker.volume_24h,
            )
            .with_sector(&format!("{:?}", CryptoSector::from_symbol(&ticker.symbol)));

            graph.add_node(node);
        }

        // Create sector edges if enabled
        if self.config.include_sector_edges {
            self.add_sector_edges(&mut graph);
        }

        graph
    }

    /// Build graph from historical kline data
    pub fn build_from_klines(&self, klines_map: &HashMap<String, Vec<Kline>>) -> MarketGraph {
        let mut graph = MarketGraph::new();
        let symbols: Vec<&String> = klines_map.keys().collect();

        // Create nodes
        for (i, symbol) in symbols.iter().enumerate() {
            if let Some(klines) = klines_map.get(*symbol) {
                if let Some(latest) = klines.last() {
                    let node = CryptoNode::new(i, symbol, latest.close);
                    graph.add_node(node);
                }
            }
        }

        // Calculate returns for correlation
        let returns_map: HashMap<String, Vec<f64>> = klines_map
            .iter()
            .filter_map(|(symbol, klines)| {
                if klines.len() < 2 {
                    return None;
                }
                let returns: Vec<f64> = klines
                    .windows(2)
                    .map(|w| (w[1].close - w[0].close) / w[0].close)
                    .collect();
                Some((symbol.clone(), returns))
            })
            .collect();

        // Create correlation edges
        self.add_correlation_edges(&mut graph, &returns_map, &symbols);

        // Add sector edges if enabled
        if self.config.include_sector_edges {
            self.add_sector_edges(&mut graph);
        }

        graph
    }

    /// Add correlation-based edges
    fn add_correlation_edges(
        &self,
        graph: &mut MarketGraph,
        returns_map: &HashMap<String, Vec<f64>>,
        symbols: &[&String],
    ) {
        for i in 0..symbols.len() {
            for j in (i + 1)..symbols.len() {
                let symbol_i = symbols[i];
                let symbol_j = symbols[j];

                if let (Some(returns_i), Some(returns_j)) =
                    (returns_map.get(symbol_i), returns_map.get(symbol_j))
                {
                    // Use the shorter window
                    let window = self.config.correlation_window.min(returns_i.len().min(returns_j.len()));
                    if window < 5 {
                        continue;  // Need at least 5 data points
                    }

                    let corr = correlation(
                        &returns_i[returns_i.len() - window..],
                        &returns_j[returns_j.len() - window..],
                    );

                    // Check if correlation meets threshold
                    let should_add = if self.config.include_negative_correlations {
                        corr.abs() >= self.config.correlation_threshold
                    } else {
                        corr >= self.config.correlation_threshold
                    };

                    if should_add {
                        let edge = MarketEdge::correlation(i, j, corr);
                        graph.add_edge(edge);
                    }
                }
            }
        }
    }

    /// Add sector-based edges
    fn add_sector_edges(&self, graph: &mut MarketGraph) {
        let n = graph.num_nodes();

        for i in 0..n {
            for j in (i + 1)..n {
                let sector_i = CryptoSector::from_symbol(&graph.nodes[i].symbol);
                let sector_j = CryptoSector::from_symbol(&graph.nodes[j].symbol);

                if sector_i == sector_j && sector_i != CryptoSector::Other {
                    // Check if edge already exists
                    if graph.get_edge(i, j).is_none() {
                        let edge = MarketEdge::same_sector(i, j);
                        graph.add_edge(edge);
                    }
                }
            }
        }
    }

    /// Build a complete graph (all nodes connected)
    pub fn build_complete_graph(&self, tickers: &[Ticker]) -> MarketGraph {
        let mut graph = self.build_from_tickers(tickers);

        let n = graph.num_nodes();
        for i in 0..n {
            for j in (i + 1)..n {
                if graph.get_edge(i, j).is_none() {
                    // Add default edge with weight based on volume similarity
                    let vol_i = graph.nodes[i].volume_24h;
                    let vol_j = graph.nodes[j].volume_24h;
                    let vol_sim = 1.0 - (vol_i - vol_j).abs() / (vol_i + vol_j).max(1.0);
                    let edge = MarketEdge::new(i, j, EdgeType::Custom, vol_sim);
                    graph.add_edge(edge);
                }
            }
        }

        graph
    }

    /// Prune edges to keep only top-k per node
    pub fn prune_edges(&self, graph: &mut MarketGraph, k: usize) {
        let n = graph.num_nodes();
        let mut edges_to_keep = std::collections::HashSet::new();

        for node_idx in 0..n {
            // Get all edges for this node
            let mut node_edges: Vec<(usize, f64)> = graph
                .edges
                .iter()
                .enumerate()
                .filter(|(_, e)| e.source == node_idx || e.target == node_idx)
                .map(|(i, e)| (i, e.weight))
                .collect();

            // Sort by weight (descending) and keep top k
            node_edges.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            for (edge_idx, _) in node_edges.into_iter().take(k) {
                edges_to_keep.insert(edge_idx);
            }
        }

        // Keep only marked edges
        let mut new_edges = Vec::new();
        for (i, edge) in graph.edges.drain(..).enumerate() {
            if edges_to_keep.contains(&i) {
                new_edges.push(edge);
            }
        }
        graph.edges = new_edges;
    }
}

/// Utility function to create a simple test graph
pub fn create_test_graph() -> MarketGraph {
    let nodes = vec![
        CryptoNode::with_data(0, "BTCUSDT", 45000.0, 2.5, 1000000.0),
        CryptoNode::with_data(1, "ETHUSDT", 2500.0, 3.0, 500000.0),
        CryptoNode::with_data(2, "SOLUSDT", 100.0, 5.0, 200000.0),
        CryptoNode::with_data(3, "BNBUSDT", 300.0, 1.5, 150000.0),
        CryptoNode::with_data(4, "DOGEUSDT", 0.08, 10.0, 100000.0),
    ];

    let edges = vec![
        MarketEdge::correlation(0, 1, 0.85),
        MarketEdge::correlation(0, 2, 0.70),
        MarketEdge::correlation(0, 3, 0.65),
        MarketEdge::correlation(1, 2, 0.75),
        MarketEdge::correlation(1, 3, 0.60),
        MarketEdge::correlation(2, 3, 0.55),
        MarketEdge::same_sector(0, 1),  // Both Layer 1
        MarketEdge::correlation(4, 0, 0.40),  // DOGE-BTC
    ];

    MarketGraph::from_nodes_edges(nodes, edges)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_config() {
        let config = GraphBuilderConfig::default();
        assert_eq!(config.correlation_threshold, 0.5);
        assert!(config.include_sector_edges);
    }

    #[test]
    fn test_build_from_tickers() {
        let tickers = vec![
            Ticker {
                symbol: "BTCUSDT".to_string(),
                last_price: 45000.0,
                high_24h: 46000.0,
                low_24h: 44000.0,
                price_change_24h: 0.02,
                volume_24h: 1000.0,
                turnover_24h: 45000000.0,
                bid_price: 44990.0,
                ask_price: 45010.0,
            },
            Ticker {
                symbol: "ETHUSDT".to_string(),
                last_price: 2500.0,
                high_24h: 2600.0,
                low_24h: 2400.0,
                price_change_24h: 0.03,
                volume_24h: 5000.0,
                turnover_24h: 12500000.0,
                bid_price: 2499.0,
                ask_price: 2501.0,
            },
        ];

        let builder = MarketGraphBuilder::new();
        let graph = builder.build_from_tickers(&tickers);

        assert_eq!(graph.num_nodes(), 2);
    }

    #[test]
    fn test_create_test_graph() {
        let graph = create_test_graph();
        assert_eq!(graph.num_nodes(), 5);
        assert!(graph.num_edges() > 0);
    }
}
