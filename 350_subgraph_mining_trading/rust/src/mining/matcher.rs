//! Pattern matching for finding patterns in graphs.

use crate::graph::FinancialGraph;
use crate::mining::{Pattern, PatternType};
use std::collections::HashMap;

/// Result of pattern matching
#[derive(Debug, Clone)]
pub struct MatchResult {
    /// Pattern that was matched
    pub pattern_id: String,
    /// Matched node mappings (pattern node -> graph symbol)
    pub node_mapping: HashMap<usize, String>,
    /// Total weight of matched edges
    pub total_weight: f64,
    /// Average edge weight
    pub avg_weight: f64,
    /// Match quality score (0-1)
    pub quality: f64,
}

/// A pattern match instance
#[derive(Debug, Clone)]
pub struct PatternMatch {
    /// The pattern that was matched
    pub pattern: Pattern,
    /// Symbols involved in the match
    pub symbols: Vec<String>,
    /// Edge weights in the match
    pub weights: Vec<f64>,
    /// Match timestamp
    pub timestamp: i64,
}

impl PatternMatch {
    /// Create a new pattern match
    pub fn new(pattern: Pattern, symbols: Vec<String>, weights: Vec<f64>) -> Self {
        Self {
            pattern,
            symbols,
            weights,
            timestamp: chrono::Utc::now().timestamp(),
        }
    }

    /// Get average edge weight
    pub fn avg_weight(&self) -> f64 {
        if self.weights.is_empty() {
            0.0
        } else {
            self.weights.iter().sum::<f64>() / self.weights.len() as f64
        }
    }

    /// Get minimum edge weight
    pub fn min_weight(&self) -> f64 {
        self.weights.iter().cloned().fold(f64::INFINITY, f64::min)
    }

    /// Get maximum edge weight
    pub fn max_weight(&self) -> f64 {
        self.weights.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    }
}

/// Pattern matcher for finding patterns in graphs
#[derive(Debug, Clone)]
pub struct PatternMatcher {
    /// Patterns to match against
    patterns: Vec<Pattern>,
    /// Minimum match quality threshold
    min_quality: f64,
    /// Whether to find all matches or just first
    find_all: bool,
}

impl PatternMatcher {
    /// Create a new pattern matcher
    pub fn new(patterns: Vec<Pattern>) -> Self {
        Self {
            patterns,
            min_quality: 0.0,
            find_all: true,
        }
    }

    /// Set minimum quality threshold
    pub fn min_quality(mut self, quality: f64) -> Self {
        self.min_quality = quality.clamp(0.0, 1.0);
        self
    }

    /// Set whether to find all matches
    pub fn find_all(mut self, find_all: bool) -> Self {
        self.find_all = find_all;
        self
    }

    /// Find all pattern matches in the graph
    pub fn find_matches(&self, graph: &FinancialGraph) -> Vec<PatternMatch> {
        let mut all_matches = Vec::new();

        for pattern in &self.patterns {
            let matches = self.match_pattern(graph, pattern);
            all_matches.extend(matches);

            if !self.find_all && !all_matches.is_empty() {
                break;
            }
        }

        all_matches
    }

    /// Match a single pattern against the graph
    fn match_pattern(&self, graph: &FinancialGraph, pattern: &Pattern) -> Vec<PatternMatch> {
        match pattern.pattern_type {
            PatternType::Triangle => self.match_triangles(graph, pattern),
            PatternType::Star => self.match_stars(graph, pattern),
            PatternType::Chain => self.match_chains(graph, pattern),
            PatternType::Clique => self.match_cliques(graph, pattern),
            PatternType::Edge => self.match_edges(graph, pattern),
            PatternType::General => Vec::new(), // General patterns need VF2 algorithm
        }
    }

    /// Match triangle patterns
    fn match_triangles(&self, graph: &FinancialGraph, pattern: &Pattern) -> Vec<PatternMatch> {
        let triangles = crate::graph::find_triangles(graph);
        let mut matches = Vec::new();

        for (a, b, c) in triangles {
            let mut weights = Vec::new();

            if let Some(w) = graph.edge_weight(&a, &b) {
                weights.push(w);
            }
            if let Some(w) = graph.edge_weight(&b, &c) {
                weights.push(w);
            }
            if let Some(w) = graph.edge_weight(&a, &c) {
                weights.push(w);
            }

            let pm = PatternMatch::new(
                pattern.clone(),
                vec![a, b, c],
                weights,
            );

            if pm.avg_weight() >= self.min_quality {
                matches.push(pm);
            }
        }

        matches
    }

    /// Match star patterns
    fn match_stars(&self, graph: &FinancialGraph, pattern: &Pattern) -> Vec<PatternMatch> {
        let num_spokes = pattern.node_count() - 1; // Minus the hub
        let stars = crate::graph::find_stars(graph, num_spokes);
        let mut matches = Vec::new();

        for (hub, spokes) in stars {
            let mut symbols = vec![hub.clone()];
            let mut weights = Vec::new();

            for spoke in &spokes {
                symbols.push(spoke.clone());
                if let Some(w) = graph.edge_weight(&hub, spoke) {
                    weights.push(w);
                }
            }

            // Only take exact number of spokes
            symbols.truncate(pattern.node_count());
            weights.truncate(pattern.edge_count());

            let pm = PatternMatch::new(pattern.clone(), symbols, weights);
            if pm.avg_weight() >= self.min_quality {
                matches.push(pm);
            }
        }

        matches
    }

    /// Match chain patterns
    fn match_chains(&self, graph: &FinancialGraph, pattern: &Pattern) -> Vec<PatternMatch> {
        let length = pattern.node_count();
        let chains = crate::graph::find_chains(graph, length);
        let mut matches = Vec::new();

        for chain in chains {
            let mut weights = Vec::new();

            for i in 0..chain.len() - 1 {
                if let Some(w) = graph.edge_weight(&chain[i], &chain[i + 1]) {
                    weights.push(w);
                }
            }

            let pm = PatternMatch::new(pattern.clone(), chain, weights);
            if pm.avg_weight() >= self.min_quality {
                matches.push(pm);
            }
        }

        matches
    }

    /// Match clique patterns
    fn match_cliques(&self, graph: &FinancialGraph, pattern: &Pattern) -> Vec<PatternMatch> {
        let size = pattern.node_count();
        let cliques = crate::graph::find_cliques(graph, size);
        let mut matches = Vec::new();

        for clique in cliques {
            if clique.len() != size {
                continue;
            }

            let mut weights = Vec::new();
            for i in 0..clique.len() {
                for j in (i + 1)..clique.len() {
                    if let Some(w) = graph.edge_weight(&clique[i], &clique[j]) {
                        weights.push(w);
                    }
                }
            }

            let pm = PatternMatch::new(pattern.clone(), clique, weights);
            if pm.avg_weight() >= self.min_quality {
                matches.push(pm);
            }
        }

        matches
    }

    /// Match edge patterns
    fn match_edges(&self, graph: &FinancialGraph, pattern: &Pattern) -> Vec<PatternMatch> {
        let mut matches = Vec::new();

        for (from, to, weight) in graph.edges() {
            if weight >= self.min_quality {
                let pm = PatternMatch::new(
                    pattern.clone(),
                    vec![from, to],
                    vec![weight],
                );
                matches.push(pm);
            }
        }

        matches
    }

    /// Compare patterns between two graphs (for change detection)
    pub fn compare_graphs(
        &self,
        old_graph: &FinancialGraph,
        new_graph: &FinancialGraph,
    ) -> PatternChanges {
        let old_matches = self.find_matches(old_graph);
        let new_matches = self.find_matches(new_graph);

        // Create sets of matched symbol combinations
        let old_set: std::collections::HashSet<_> = old_matches
            .iter()
            .map(|m| {
                let mut symbols = m.symbols.clone();
                symbols.sort();
                (m.pattern.pattern_type, symbols)
            })
            .collect();

        let new_set: std::collections::HashSet<_> = new_matches
            .iter()
            .map(|m| {
                let mut symbols = m.symbols.clone();
                symbols.sort();
                (m.pattern.pattern_type, symbols)
            })
            .collect();

        // Find additions and removals
        let appeared: Vec<_> = new_matches
            .into_iter()
            .filter(|m| {
                let mut symbols = m.symbols.clone();
                symbols.sort();
                !old_set.contains(&(m.pattern.pattern_type, symbols))
            })
            .collect();

        let disappeared: Vec<_> = old_matches
            .into_iter()
            .filter(|m| {
                let mut symbols = m.symbols.clone();
                symbols.sort();
                !new_set.contains(&(m.pattern.pattern_type, symbols))
            })
            .collect();

        PatternChanges {
            appeared,
            disappeared,
        }
    }
}

/// Changes in patterns between two graph snapshots
#[derive(Debug, Clone)]
pub struct PatternChanges {
    /// Patterns that appeared (didn't exist before)
    pub appeared: Vec<PatternMatch>,
    /// Patterns that disappeared (existed before)
    pub disappeared: Vec<PatternMatch>,
}

impl PatternChanges {
    /// Check if there are any changes
    pub fn has_changes(&self) -> bool {
        !self.appeared.is_empty() || !self.disappeared.is_empty()
    }

    /// Get count of new patterns
    pub fn new_count(&self) -> usize {
        self.appeared.len()
    }

    /// Get count of lost patterns
    pub fn lost_count(&self) -> usize {
        self.disappeared.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{GraphType, Node};

    fn create_test_graph() -> FinancialGraph {
        let mut graph = FinancialGraph::new(GraphType::Correlation);

        for symbol in ["A", "B", "C", "D"] {
            graph.add_node(Node::new(symbol));
        }

        // Triangle A-B-C
        graph.add_edge("A", "B", 0.9);
        graph.add_edge("B", "C", 0.85);
        graph.add_edge("A", "C", 0.8);

        // Extra edge
        graph.add_edge("C", "D", 0.75);

        graph
    }

    #[test]
    fn test_match_triangles() {
        let graph = create_test_graph();
        let patterns = vec![Pattern::triangle()];
        let matcher = PatternMatcher::new(patterns);

        let matches = matcher.find_matches(&graph);
        assert!(!matches.is_empty());

        let triangle_match = &matches[0];
        assert_eq!(triangle_match.symbols.len(), 3);
    }

    #[test]
    fn test_pattern_changes() {
        let mut old_graph = create_test_graph();
        let mut new_graph = old_graph.clone();

        // Add new triangle in new graph
        new_graph.add_node(Node::new("E"));
        new_graph.add_edge("C", "E", 0.7);
        new_graph.add_edge("D", "E", 0.7);

        let patterns = vec![Pattern::triangle()];
        let matcher = PatternMatcher::new(patterns);

        let changes = matcher.compare_graphs(&old_graph, &new_graph);

        // New triangle C-D-E should appear
        assert!(changes.has_changes());
    }
}
