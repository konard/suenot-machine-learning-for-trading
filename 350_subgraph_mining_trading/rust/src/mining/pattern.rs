//! Pattern representation for subgraph mining.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

/// Type of pattern structure
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PatternType {
    /// Single edge (2 nodes)
    Edge,
    /// Triangle (3 nodes, all connected)
    Triangle,
    /// Star pattern (1 hub, multiple spokes)
    Star,
    /// Linear chain
    Chain,
    /// Clique (fully connected)
    Clique,
    /// General subgraph
    General,
}

impl PatternType {
    /// Get pattern type name
    pub fn name(&self) -> &str {
        match self {
            PatternType::Edge => "edge",
            PatternType::Triangle => "triangle",
            PatternType::Star => "star",
            PatternType::Chain => "chain",
            PatternType::Clique => "clique",
            PatternType::General => "general",
        }
    }

    /// Check if pattern is bullish
    pub fn is_bullish_signal(&self) -> bool {
        matches!(self, PatternType::Triangle | PatternType::Clique)
    }
}

/// Node in a pattern (abstract, not tied to specific symbol)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternNode {
    /// Node ID within pattern
    pub id: usize,
    /// Label (optional, for typed patterns)
    pub label: Option<String>,
}

impl PatternNode {
    pub fn new(id: usize) -> Self {
        Self { id, label: None }
    }

    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }
}

/// Edge in a pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternEdge {
    /// Source node ID
    pub from: usize,
    /// Target node ID
    pub to: usize,
    /// Edge label/weight constraint (optional)
    pub label: Option<String>,
    /// Minimum weight threshold
    pub min_weight: Option<f64>,
}

impl PatternEdge {
    pub fn new(from: usize, to: usize) -> Self {
        Self {
            from,
            to,
            label: None,
            min_weight: None,
        }
    }

    pub fn with_min_weight(mut self, weight: f64) -> Self {
        self.min_weight = Some(weight);
        self
    }
}

/// DFS code for canonical pattern representation (gSpan)
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DFSCode {
    /// List of edges in DFS order
    pub edges: Vec<(usize, usize, i32, i32, i32)>,
}

impl DFSCode {
    pub fn new() -> Self {
        Self { edges: Vec::new() }
    }

    /// Add edge to DFS code
    pub fn add_edge(&mut self, from: usize, to: usize, from_label: i32, edge_label: i32, to_label: i32) {
        self.edges.push((from, to, from_label, edge_label, to_label));
    }

    /// Check if this code is minimum (canonical)
    pub fn is_minimum(&self) -> bool {
        // Simplified check - proper implementation would compare with all rotations
        !self.edges.is_empty()
    }

    /// Get rightmost path for extension
    pub fn rightmost_path(&self) -> Vec<usize> {
        if self.edges.is_empty() {
            return Vec::new();
        }

        let mut path = Vec::new();
        let mut current = self.edges.last().unwrap().1;
        path.push(current);

        for edge in self.edges.iter().rev() {
            if edge.1 == current {
                current = edge.0;
                path.push(current);
            }
        }

        path.reverse();
        path
    }
}

impl Default for DFSCode {
    fn default() -> Self {
        Self::new()
    }
}

/// A mined subgraph pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pattern {
    /// Unique pattern identifier
    pub id: String,
    /// Pattern type classification
    pub pattern_type: PatternType,
    /// Nodes in the pattern
    pub nodes: Vec<PatternNode>,
    /// Edges in the pattern
    pub edges: Vec<PatternEdge>,
    /// Canonical DFS code
    pub dfs_code: DFSCode,
    /// Support count (frequency)
    pub support: usize,
    /// Average weight of edges when matched
    pub avg_weight: f64,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

impl Pattern {
    /// Create a new pattern
    pub fn new(id: impl Into<String>, pattern_type: PatternType) -> Self {
        Self {
            id: id.into(),
            pattern_type,
            nodes: Vec::new(),
            edges: Vec::new(),
            dfs_code: DFSCode::new(),
            support: 0,
            avg_weight: 0.0,
            metadata: HashMap::new(),
        }
    }

    /// Create a triangle pattern
    pub fn triangle() -> Self {
        let mut pattern = Self::new("triangle", PatternType::Triangle);
        pattern.nodes = vec![PatternNode::new(0), PatternNode::new(1), PatternNode::new(2)];
        pattern.edges = vec![
            PatternEdge::new(0, 1),
            PatternEdge::new(1, 2),
            PatternEdge::new(0, 2),
        ];
        pattern
    }

    /// Create a star pattern with specified number of spokes
    pub fn star(num_spokes: usize) -> Self {
        let mut pattern = Self::new(format!("star_{}", num_spokes), PatternType::Star);

        // Hub is node 0
        pattern.nodes.push(PatternNode::new(0));

        // Spokes are nodes 1..=num_spokes
        for i in 1..=num_spokes {
            pattern.nodes.push(PatternNode::new(i));
            pattern.edges.push(PatternEdge::new(0, i));
        }

        pattern
    }

    /// Create a chain pattern of specified length
    pub fn chain(length: usize) -> Self {
        let mut pattern = Self::new(format!("chain_{}", length), PatternType::Chain);

        for i in 0..length {
            pattern.nodes.push(PatternNode::new(i));
            if i > 0 {
                pattern.edges.push(PatternEdge::new(i - 1, i));
            }
        }

        pattern
    }

    /// Create a clique pattern of specified size
    pub fn clique(size: usize) -> Self {
        let mut pattern = Self::new(format!("clique_{}", size), PatternType::Clique);

        for i in 0..size {
            pattern.nodes.push(PatternNode::new(i));
        }

        for i in 0..size {
            for j in (i + 1)..size {
                pattern.edges.push(PatternEdge::new(i, j));
            }
        }

        pattern
    }

    /// Number of nodes in pattern
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of edges in pattern
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Add a node to the pattern
    pub fn add_node(&mut self, node: PatternNode) {
        self.nodes.push(node);
    }

    /// Add an edge to the pattern
    pub fn add_edge(&mut self, edge: PatternEdge) {
        self.edges.push(edge);
    }

    /// Set support count
    pub fn with_support(mut self, support: usize) -> Self {
        self.support = support;
        self
    }

    /// Get trading signal interpretation
    pub fn trading_signal(&self) -> PatternSignal {
        match self.pattern_type {
            PatternType::Triangle => PatternSignal::Bullish,
            PatternType::Clique => PatternSignal::Bullish,
            PatternType::Star => PatternSignal::Neutral,
            PatternType::Chain => PatternSignal::Neutral,
            PatternType::Edge => PatternSignal::Neutral,
            PatternType::General => PatternSignal::Neutral,
        }
    }

    /// Compute pattern hash for comparison
    pub fn compute_hash(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        let mut hasher = DefaultHasher::new();
        self.dfs_code.hash(&mut hasher);
        hasher.finish()
    }
}

/// Trading signal derived from pattern
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PatternSignal {
    Bullish,
    Bearish,
    Neutral,
}

impl PartialEq for Pattern {
    fn eq(&self, other: &Self) -> bool {
        self.dfs_code == other.dfs_code
    }
}

impl Eq for Pattern {}

impl Hash for Pattern {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.dfs_code.hash(state);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pattern_creation() {
        let triangle = Pattern::triangle();
        assert_eq!(triangle.node_count(), 3);
        assert_eq!(triangle.edge_count(), 3);
        assert_eq!(triangle.pattern_type, PatternType::Triangle);
    }

    #[test]
    fn test_star_pattern() {
        let star = Pattern::star(4);
        assert_eq!(star.node_count(), 5); // 1 hub + 4 spokes
        assert_eq!(star.edge_count(), 4);
    }

    #[test]
    fn test_clique_pattern() {
        let clique = Pattern::clique(4);
        assert_eq!(clique.node_count(), 4);
        assert_eq!(clique.edge_count(), 6); // 4*3/2 = 6
    }

    #[test]
    fn test_chain_pattern() {
        let chain = Pattern::chain(5);
        assert_eq!(chain.node_count(), 5);
        assert_eq!(chain.edge_count(), 4);
    }
}
