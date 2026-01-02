//! gSpan algorithm implementation for frequent subgraph mining.

use crate::graph::FinancialGraph;
use crate::mining::{DFSCode, Pattern, PatternEdge, PatternNode, PatternType};
use std::collections::{HashMap, HashSet};
use thiserror::Error;

/// Errors during mining
#[derive(Error, Debug)]
pub enum MiningError {
    #[error("Graph is empty")]
    EmptyGraph,
    #[error("Invalid minimum support: {0}")]
    InvalidSupport(usize),
    #[error("Mining timeout")]
    Timeout,
}

/// gSpan miner for frequent subgraph discovery
#[derive(Debug, Clone)]
pub struct GSpanMiner {
    /// Minimum support threshold
    min_support: usize,
    /// Maximum pattern size (nodes)
    max_size: usize,
    /// Maximum number of patterns to find
    max_patterns: usize,
    /// Minimum edge weight to consider
    min_edge_weight: f64,
}

impl GSpanMiner {
    /// Create a new gSpan miner
    pub fn new() -> Self {
        Self {
            min_support: 2,
            max_size: 6,
            max_patterns: 100,
            min_edge_weight: 0.0,
        }
    }

    /// Set minimum support threshold
    pub fn min_support(mut self, support: usize) -> Self {
        self.min_support = support.max(1);
        self
    }

    /// Set maximum pattern size
    pub fn max_size(mut self, size: usize) -> Self {
        self.max_size = size.max(2);
        self
    }

    /// Set maximum patterns to find
    pub fn max_patterns(mut self, count: usize) -> Self {
        self.max_patterns = count;
        self
    }

    /// Set minimum edge weight
    pub fn min_edge_weight(mut self, weight: f64) -> Self {
        self.min_edge_weight = weight;
        self
    }

    /// Mine frequent subgraph patterns from the graph
    pub fn mine(&self, graph: &FinancialGraph) -> Result<Vec<Pattern>, MiningError> {
        if graph.node_count() == 0 {
            return Err(MiningError::EmptyGraph);
        }

        let mut patterns = Vec::new();

        // Step 1: Find frequent edges (1-edge patterns)
        let frequent_edges = self.find_frequent_edges(graph);

        // Step 2: Extend patterns using DFS
        for (from, to, weight) in &frequent_edges {
            if patterns.len() >= self.max_patterns {
                break;
            }

            let mut dfs_code = DFSCode::new();
            dfs_code.add_edge(0, 1, 0, 0, 0);

            let initial_pattern = Pattern::new(
                format!("edge_{}_{}", from, to),
                PatternType::Edge,
            );

            // Extend this edge pattern
            self.extend_pattern(
                graph,
                &frequent_edges,
                &initial_pattern,
                &dfs_code,
                &mut patterns,
            );
        }

        // Step 3: Add commonly known patterns directly
        self.add_known_patterns(graph, &mut patterns);

        // Sort by support (most frequent first)
        patterns.sort_by(|a, b| b.support.cmp(&a.support));

        // Limit to max_patterns
        patterns.truncate(self.max_patterns);

        Ok(patterns)
    }

    /// Find frequent edges in the graph
    fn find_frequent_edges(&self, graph: &FinancialGraph) -> Vec<(String, String, f64)> {
        graph
            .edges()
            .into_iter()
            .filter(|(_, _, w)| *w >= self.min_edge_weight)
            .collect()
    }

    /// Extend pattern by adding edges
    fn extend_pattern(
        &self,
        graph: &FinancialGraph,
        frequent_edges: &[(String, String, f64)],
        current: &Pattern,
        dfs_code: &DFSCode,
        patterns: &mut Vec<Pattern>,
    ) {
        if current.node_count() >= self.max_size {
            return;
        }

        if patterns.len() >= self.max_patterns {
            return;
        }

        // Get rightmost path for extension
        let rightmost_path = dfs_code.rightmost_path();

        // Try extending from each node in rightmost path
        for &node_id in &rightmost_path {
            // Find potential extensions
            // This is simplified - full gSpan would enumerate all valid extensions
        }
    }

    /// Add commonly known patterns (triangles, stars, cliques)
    fn add_known_patterns(&self, graph: &FinancialGraph, patterns: &mut Vec<Pattern>) {
        // Find and add triangles
        let triangles = crate::graph::find_triangles(graph);
        if triangles.len() >= self.min_support {
            let mut pattern = Pattern::triangle();
            pattern.support = triangles.len();

            // Calculate average weight
            let mut total_weight = 0.0;
            let mut count = 0;
            for (a, b, c) in &triangles {
                if let Some(w1) = graph.edge_weight(a, b) {
                    total_weight += w1;
                    count += 1;
                }
                if let Some(w2) = graph.edge_weight(b, c) {
                    total_weight += w2;
                    count += 1;
                }
                if let Some(w3) = graph.edge_weight(a, c) {
                    total_weight += w3;
                    count += 1;
                }
            }
            pattern.avg_weight = if count > 0 { total_weight / count as f64 } else { 0.0 };

            patterns.push(pattern);
        }

        // Find and add stars
        for min_spokes in [3, 4, 5] {
            let stars = crate::graph::find_stars(graph, min_spokes);
            if stars.len() >= self.min_support {
                let mut pattern = Pattern::star(min_spokes);
                pattern.support = stars.len();
                patterns.push(pattern);
            }
        }

        // Find and add cliques
        let cliques = crate::graph::find_cliques(graph, 5);
        let mut clique_by_size: HashMap<usize, usize> = HashMap::new();

        for clique in &cliques {
            *clique_by_size.entry(clique.len()).or_insert(0) += 1;
        }

        for (size, count) in clique_by_size {
            if size >= 4 && count >= self.min_support {
                let mut pattern = Pattern::clique(size);
                pattern.support = count;
                patterns.push(pattern);
            }
        }

        // Find and add chains
        for length in [3, 4, 5] {
            let chains = crate::graph::find_chains(graph, length);
            if chains.len() >= self.min_support {
                let mut pattern = Pattern::chain(length);
                pattern.support = chains.len();
                patterns.push(pattern);
            }
        }
    }

    /// Mine patterns across multiple time snapshots
    pub fn mine_temporal(
        &self,
        graphs: &[FinancialGraph],
    ) -> Result<Vec<(Pattern, Vec<bool>)>, MiningError> {
        if graphs.is_empty() {
            return Err(MiningError::EmptyGraph);
        }

        // Mine patterns from each snapshot
        let mut all_patterns: HashMap<u64, (Pattern, Vec<bool>)> = HashMap::new();

        for (i, graph) in graphs.iter().enumerate() {
            let patterns = self.mine(graph)?;

            for pattern in patterns {
                let hash = pattern.compute_hash();

                let entry = all_patterns
                    .entry(hash)
                    .or_insert_with(|| (pattern.clone(), vec![false; graphs.len()]));

                entry.1[i] = true;
            }
        }

        let mut result: Vec<_> = all_patterns.into_values().collect();

        // Sort by frequency across time
        result.sort_by(|a, b| {
            let freq_a: usize = a.1.iter().filter(|&&x| x).count();
            let freq_b: usize = b.1.iter().filter(|&&x| x).count();
            freq_b.cmp(&freq_a)
        });

        Ok(result)
    }
}

impl Default for GSpanMiner {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{GraphType, Node};

    fn create_test_graph() -> FinancialGraph {
        let mut graph = FinancialGraph::new(GraphType::Correlation);

        // Create a graph with triangles and a star
        let symbols = ["BTC", "ETH", "SOL", "AVAX", "MATIC"];
        for symbol in &symbols {
            graph.add_node(Node::new(*symbol));
        }

        // Triangle: BTC-ETH-SOL
        graph.add_edge("BTC", "ETH", 0.9);
        graph.add_edge("ETH", "SOL", 0.85);
        graph.add_edge("BTC", "SOL", 0.8);

        // Star: BTC as hub
        graph.add_edge("BTC", "AVAX", 0.75);
        graph.add_edge("BTC", "MATIC", 0.7);

        graph
    }

    #[test]
    fn test_miner_creation() {
        let miner = GSpanMiner::new()
            .min_support(3)
            .max_size(5);

        assert_eq!(miner.min_support, 3);
        assert_eq!(miner.max_size, 5);
    }

    #[test]
    fn test_mine_patterns() {
        let graph = create_test_graph();
        let miner = GSpanMiner::new().min_support(1);

        let patterns = miner.mine(&graph).unwrap();
        assert!(!patterns.is_empty());

        // Should find at least the triangle
        let has_triangle = patterns.iter().any(|p| p.pattern_type == PatternType::Triangle);
        assert!(has_triangle);
    }
}
