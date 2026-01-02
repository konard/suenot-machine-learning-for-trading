//! Graph builder for constructing market graphs from price data.

use super::{correlation::CorrelationMatrix, MarketGraph, GraphParams};
use crate::data::MarketData;
use anyhow::Result;

/// Method for calculating correlations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CorrelationMethod {
    /// Pearson correlation (linear)
    Pearson,
    /// Spearman correlation (rank-based)
    Spearman,
    /// Kendall tau correlation
    Kendall,
}

impl Default for CorrelationMethod {
    fn default() -> Self {
        Self::Pearson
    }
}

/// Graph construction method
#[derive(Debug, Clone, Copy)]
pub enum GraphType {
    /// Threshold-based: include edge if correlation > threshold
    Threshold,
    /// K-nearest neighbors: each node connects to K most correlated
    KNN,
    /// Minimum spanning tree
    MST,
    /// Planar maximally filtered graph
    PMFG,
    /// Full graph with all correlations as weights
    Full,
}

impl Default for GraphType {
    fn default() -> Self {
        Self::Threshold
    }
}

/// Builder for constructing market graphs
#[derive(Debug, Clone)]
pub struct GraphBuilder {
    /// Correlation method to use
    method: CorrelationMethod,
    /// Graph construction type
    graph_type: GraphType,
    /// Threshold for edge inclusion
    threshold: f64,
    /// K for KNN
    k: usize,
    /// Rolling window size for correlation
    window: Option<usize>,
    /// Whether to use absolute correlations
    use_absolute: bool,
}

impl Default for GraphBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphBuilder {
    /// Create a new GraphBuilder with default settings
    pub fn new() -> Self {
        Self {
            method: CorrelationMethod::Pearson,
            graph_type: GraphType::Threshold,
            threshold: 0.5,
            k: 3,
            window: None,
            use_absolute: false,
        }
    }

    /// Set the correlation method
    pub fn with_method(mut self, method: CorrelationMethod) -> Self {
        self.method = method;
        self
    }

    /// Set the graph construction type
    pub fn with_graph_type(mut self, graph_type: GraphType) -> Self {
        self.graph_type = graph_type;
        self
    }

    /// Set the correlation threshold
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold;
        self
    }

    /// Set K for KNN
    pub fn with_k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Set rolling window size
    pub fn with_window(mut self, window: usize) -> Self {
        self.window = Some(window);
        self
    }

    /// Set whether to use absolute correlations
    pub fn with_absolute(mut self, use_absolute: bool) -> Self {
        self.use_absolute = use_absolute;
        self
    }

    /// Build the market graph from market data
    pub fn build(&self, data: &MarketData) -> Result<MarketGraph> {
        // Calculate returns
        let returns = data.returns();

        // Build correlation matrix
        let corr_matrix = CorrelationMatrix::from_returns(&returns, self.method);

        // Build graph based on type
        let mut graph = match self.graph_type {
            GraphType::Threshold => self.build_threshold_graph(&data.symbols, &corr_matrix),
            GraphType::KNN => self.build_knn_graph(&data.symbols, &corr_matrix),
            GraphType::MST => self.build_mst_graph(&data.symbols, &corr_matrix),
            GraphType::PMFG => self.build_pmfg_graph(&data.symbols, &corr_matrix),
            GraphType::Full => self.build_full_graph(&data.symbols, &corr_matrix),
        };

        // Set parameters
        graph.params = GraphParams {
            correlation_method: self.method,
            threshold: Some(self.threshold),
            k: Some(self.k),
            window: self.window,
        };

        Ok(graph)
    }

    /// Build from pre-computed correlation matrix
    pub fn build_from_correlations(
        &self,
        symbols: &[String],
        correlations: &CorrelationMatrix,
    ) -> MarketGraph {
        match self.graph_type {
            GraphType::Threshold => self.build_threshold_graph(symbols, correlations),
            GraphType::KNN => self.build_knn_graph(symbols, correlations),
            GraphType::MST => self.build_mst_graph(symbols, correlations),
            GraphType::PMFG => self.build_pmfg_graph(symbols, correlations),
            GraphType::Full => self.build_full_graph(symbols, correlations),
        }
    }

    /// Build threshold-based graph
    fn build_threshold_graph(
        &self,
        symbols: &[String],
        correlations: &CorrelationMatrix,
    ) -> MarketGraph {
        let mut graph = MarketGraph::with_symbols(symbols);

        for i in 0..symbols.len() {
            for j in (i + 1)..symbols.len() {
                let corr = correlations.get(i, j);
                let weight = if self.use_absolute { corr.abs() } else { corr };

                if weight.abs() >= self.threshold {
                    graph.add_edge(&symbols[i], &symbols[j], weight);
                }
            }
        }

        graph
    }

    /// Build K-nearest neighbors graph
    fn build_knn_graph(
        &self,
        symbols: &[String],
        correlations: &CorrelationMatrix,
    ) -> MarketGraph {
        let mut graph = MarketGraph::with_symbols(symbols);

        for i in 0..symbols.len() {
            // Get correlations with all other nodes
            let mut neighbors: Vec<(usize, f64)> = (0..symbols.len())
                .filter(|&j| j != i)
                .map(|j| {
                    let corr = correlations.get(i, j);
                    (j, if self.use_absolute { corr.abs() } else { corr })
                })
                .collect();

            // Sort by correlation (descending)
            neighbors.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // Take top K
            for (j, weight) in neighbors.into_iter().take(self.k) {
                graph.add_edge(&symbols[i], &symbols[j], weight);
            }
        }

        graph
    }

    /// Build minimum spanning tree
    fn build_mst_graph(
        &self,
        symbols: &[String],
        correlations: &CorrelationMatrix,
    ) -> MarketGraph {
        let n = symbols.len();
        if n == 0 {
            return MarketGraph::new();
        }

        // Kruskal's algorithm
        // Convert correlations to distances (1 - |correlation|)
        let mut edges: Vec<(usize, usize, f64, f64)> = Vec::new();

        for i in 0..n {
            for j in (i + 1)..n {
                let corr = correlations.get(i, j);
                let distance = 1.0 - corr.abs(); // Lower distance = higher correlation
                edges.push((i, j, distance, corr));
            }
        }

        // Sort by distance (ascending)
        edges.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

        // Union-Find for cycle detection
        let mut parent: Vec<usize> = (0..n).collect();
        let mut rank: Vec<usize> = vec![0; n];

        fn find(parent: &mut [usize], i: usize) -> usize {
            if parent[i] != i {
                parent[i] = find(parent, parent[i]);
            }
            parent[i]
        }

        fn union(parent: &mut [usize], rank: &mut [usize], x: usize, y: usize) {
            let px = find(parent, x);
            let py = find(parent, y);

            if rank[px] < rank[py] {
                parent[px] = py;
            } else if rank[px] > rank[py] {
                parent[py] = px;
            } else {
                parent[py] = px;
                rank[px] += 1;
            }
        }

        let mut graph = MarketGraph::with_symbols(symbols);
        let mut edge_count = 0;

        for (i, j, _distance, corr) in edges {
            if edge_count >= n - 1 {
                break;
            }

            let pi = find(&mut parent, i);
            let pj = find(&mut parent, j);

            if pi != pj {
                union(&mut parent, &mut rank, pi, pj);
                graph.add_edge(&symbols[i], &symbols[j], corr);
                edge_count += 1;
            }
        }

        graph
    }

    /// Build planar maximally filtered graph
    /// Simplified implementation - adds 3(n-2) edges maintaining planarity heuristically
    fn build_pmfg_graph(
        &self,
        symbols: &[String],
        correlations: &CorrelationMatrix,
    ) -> MarketGraph {
        let n = symbols.len();
        if n < 4 {
            return self.build_full_graph(symbols, correlations);
        }

        // Start with MST
        let mut graph = self.build_mst_graph(symbols, correlations);
        let mst_edges = graph.edge_count();

        // Target edges for PMFG: 3(n-2)
        let target_edges = 3 * (n - 2);
        let edges_to_add = target_edges.saturating_sub(mst_edges);

        // Get all non-MST edges sorted by correlation
        let mut candidate_edges: Vec<(usize, usize, f64)> = Vec::new();

        for i in 0..n {
            for j in (i + 1)..n {
                if graph.edge_weight(&symbols[i], &symbols[j]).is_none() {
                    let corr = correlations.get(i, j);
                    candidate_edges.push((i, j, corr));
                }
            }
        }

        // Sort by absolute correlation (descending)
        candidate_edges.sort_by(|a, b| {
            b.2.abs()
                .partial_cmp(&a.2.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Add edges (simplified - doesn't check true planarity)
        for (i, j, corr) in candidate_edges.into_iter().take(edges_to_add) {
            graph.add_edge(&symbols[i], &symbols[j], corr);
        }

        graph
    }

    /// Build full graph with all edges
    fn build_full_graph(
        &self,
        symbols: &[String],
        correlations: &CorrelationMatrix,
    ) -> MarketGraph {
        let mut graph = MarketGraph::with_symbols(symbols);

        for i in 0..symbols.len() {
            for j in (i + 1)..symbols.len() {
                let corr = correlations.get(i, j);
                let weight = if self.use_absolute { corr.abs() } else { corr };
                graph.add_edge(&symbols[i], &symbols[j], weight);
            }
        }

        graph
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_correlations() -> (Vec<String>, CorrelationMatrix) {
        let symbols = vec![
            "BTC".to_string(),
            "ETH".to_string(),
            "SOL".to_string(),
            "AVAX".to_string(),
        ];

        // Create correlation matrix
        let corr_data = vec![
            vec![1.0, 0.85, 0.72, 0.65],
            vec![0.85, 1.0, 0.78, 0.70],
            vec![0.72, 0.78, 1.0, 0.82],
            vec![0.65, 0.70, 0.82, 1.0],
        ];

        let matrix = CorrelationMatrix::from_matrix(corr_data);
        (symbols, matrix)
    }

    #[test]
    fn test_threshold_graph() {
        let (symbols, correlations) = create_test_correlations();

        let graph = GraphBuilder::new()
            .with_threshold(0.75)
            .build_from_correlations(&symbols, &correlations);

        // Should include: BTC-ETH (0.85), ETH-SOL (0.78), SOL-AVAX (0.82)
        assert_eq!(graph.edge_count(), 3);
        assert!(graph.edge_weight("BTC", "ETH").is_some());
    }

    #[test]
    fn test_knn_graph() {
        let (symbols, correlations) = create_test_correlations();

        let graph = GraphBuilder::new()
            .with_graph_type(GraphType::KNN)
            .with_k(2)
            .build_from_correlations(&symbols, &correlations);

        // Each node connects to its 2 nearest neighbors
        // Note: KNN is not symmetric, so edge count varies
        assert!(graph.edge_count() >= 4);
    }

    #[test]
    fn test_mst_graph() {
        let (symbols, correlations) = create_test_correlations();

        let graph = GraphBuilder::new()
            .with_graph_type(GraphType::MST)
            .build_from_correlations(&symbols, &correlations);

        // MST has n-1 edges
        assert_eq!(graph.edge_count(), 3);
    }

    #[test]
    fn test_full_graph() {
        let (symbols, correlations) = create_test_correlations();

        let graph = GraphBuilder::new()
            .with_graph_type(GraphType::Full)
            .build_from_correlations(&symbols, &correlations);

        // Full graph has n*(n-1)/2 edges
        assert_eq!(graph.edge_count(), 6);
    }
}
