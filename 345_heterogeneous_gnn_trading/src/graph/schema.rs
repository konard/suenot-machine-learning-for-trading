//! Graph schema and metapath definitions
//!
//! Defines the valid node and edge types, and metapaths for the trading graph.

use std::collections::{HashMap, HashSet};
use serde::{Deserialize, Serialize};

use super::{NodeType, EdgeType};
use super::edge::Relation;

/// Graph schema defining valid relations
#[derive(Debug, Clone)]
pub struct GraphSchema {
    /// Set of valid relations (source_type, edge_type, target_type)
    valid_relations: HashSet<Relation>,
    /// Predefined metapaths for trading analysis
    metapaths: Vec<Metapath>,
}

impl GraphSchema {
    /// Create a new empty schema
    pub fn new() -> Self {
        Self {
            valid_relations: HashSet::new(),
            metapaths: Vec::new(),
        }
    }

    /// Create the default trading schema
    pub fn trading_schema() -> Self {
        let mut schema = Self::new();

        // Asset-Asset relations
        schema.add_relation(NodeType::Asset, EdgeType::Correlation, NodeType::Asset);
        schema.add_relation(NodeType::Asset, EdgeType::Cointegration, NodeType::Asset);
        schema.add_relation(NodeType::Asset, EdgeType::LeadLag, NodeType::Asset);
        schema.add_relation(NodeType::Asset, EdgeType::SameSector, NodeType::Asset);

        // Asset-Exchange relations
        schema.add_relation(NodeType::Asset, EdgeType::TradesOn, NodeType::Exchange);
        schema.add_relation(NodeType::Asset, EdgeType::Liquidity, NodeType::Exchange);

        // Wallet-Asset relations
        schema.add_relation(NodeType::Wallet, EdgeType::Holds, NodeType::Asset);
        schema.add_relation(NodeType::Wallet, EdgeType::Accumulating, NodeType::Asset);
        schema.add_relation(NodeType::Wallet, EdgeType::Distributing, NodeType::Asset);

        // Asset-MarketCondition relations
        schema.add_relation(NodeType::Asset, EdgeType::Sensitivity, NodeType::MarketCondition);
        schema.add_relation(NodeType::Asset, EdgeType::Beta, NodeType::MarketCondition);

        // Add predefined trading metapaths
        schema.add_trading_metapaths();

        schema
    }

    /// Add a valid relation to the schema
    pub fn add_relation(&mut self, source: NodeType, edge: EdgeType, target: NodeType) {
        self.valid_relations.insert(Relation::new(source, edge, target));

        // If bidirectional, add reverse relation
        if edge.is_bidirectional() {
            self.valid_relations.insert(Relation::new(target, edge, source));
        }
    }

    /// Check if a relation is valid
    pub fn is_valid_relation(&self, source: NodeType, edge: EdgeType, target: NodeType) -> bool {
        self.valid_relations.contains(&Relation::new(source, edge, target))
    }

    /// Get all valid relations
    pub fn relations(&self) -> &HashSet<Relation> {
        &self.valid_relations
    }

    /// Get all relations for a given source type
    pub fn relations_from(&self, source_type: NodeType) -> Vec<&Relation> {
        self.valid_relations
            .iter()
            .filter(|r| r.source_type == source_type)
            .collect()
    }

    /// Get all relations for a given edge type
    pub fn relations_with_edge(&self, edge_type: EdgeType) -> Vec<&Relation> {
        self.valid_relations
            .iter()
            .filter(|r| r.edge_type == edge_type)
            .collect()
    }

    /// Add predefined trading metapaths
    fn add_trading_metapaths(&mut self) {
        // Metapath 1: Asset-Correlation-Asset (A-c-A)
        self.metapaths.push(Metapath::new(
            "asset_correlation",
            "Direct asset correlation",
            vec![
                (NodeType::Asset, Some(EdgeType::Correlation)),
                (NodeType::Asset, None),
            ],
        ));

        // Metapath 2: Asset-Exchange-Asset (A-E-A)
        self.metapaths.push(Metapath::new(
            "same_exchange",
            "Assets on same exchange",
            vec![
                (NodeType::Asset, Some(EdgeType::TradesOn)),
                (NodeType::Exchange, Some(EdgeType::TradesOn)),
                (NodeType::Asset, None),
            ],
        ));

        // Metapath 3: Wallet-Asset-Asset (W-A-A)
        self.metapaths.push(Metapath::new(
            "whale_correlation",
            "Whale portfolio correlation",
            vec![
                (NodeType::Wallet, Some(EdgeType::Holds)),
                (NodeType::Asset, Some(EdgeType::Correlation)),
                (NodeType::Asset, None),
            ],
        ));

        // Metapath 4: Asset-MarketCondition-Asset (A-M-A)
        self.metapaths.push(Metapath::new(
            "market_sensitivity",
            "Assets with similar market sensitivity",
            vec![
                (NodeType::Asset, Some(EdgeType::Sensitivity)),
                (NodeType::MarketCondition, Some(EdgeType::Sensitivity)),
                (NodeType::Asset, None),
            ],
        ));

        // Metapath 5: Asset-Sector-Asset-Correlation-Asset (A-s-A-c-A)
        self.metapaths.push(Metapath::new(
            "sector_correlation",
            "Sector-based correlation chain",
            vec![
                (NodeType::Asset, Some(EdgeType::SameSector)),
                (NodeType::Asset, Some(EdgeType::Correlation)),
                (NodeType::Asset, None),
            ],
        ));

        // Metapath 6: Asset-LeadLag-Asset (A-ll-A)
        self.metapaths.push(Metapath::new(
            "lead_lag",
            "Lead-lag relationship",
            vec![
                (NodeType::Asset, Some(EdgeType::LeadLag)),
                (NodeType::Asset, None),
            ],
        ));

        // Metapath 7: Wallet-Accumulating-Asset-TradesOn-Exchange (W-a-A-t-E)
        self.metapaths.push(Metapath::new(
            "whale_accumulation_venue",
            "Whale accumulation on specific exchanges",
            vec![
                (NodeType::Wallet, Some(EdgeType::Accumulating)),
                (NodeType::Asset, Some(EdgeType::TradesOn)),
                (NodeType::Exchange, None),
            ],
        ));
    }

    /// Get all predefined metapaths
    pub fn metapaths(&self) -> &[Metapath] {
        &self.metapaths
    }

    /// Get metapath by name
    pub fn get_metapath(&self, name: &str) -> Option<&Metapath> {
        self.metapaths.iter().find(|m| m.name == name)
    }

    /// Add a custom metapath
    pub fn add_metapath(&mut self, metapath: Metapath) {
        self.metapaths.push(metapath);
    }
}

impl Default for GraphSchema {
    fn default() -> Self {
        Self::trading_schema()
    }
}

/// A metapath definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metapath {
    /// Unique name for this metapath
    pub name: String,
    /// Human-readable description
    pub description: String,
    /// Sequence of (node_type, edge_type) pairs
    /// The last entry has None for edge_type as it's the end
    pub path: Vec<(NodeType, Option<EdgeType>)>,
}

impl Metapath {
    /// Create a new metapath
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        path: Vec<(NodeType, Option<EdgeType>)>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            path,
        }
    }

    /// Get the length of the metapath (number of edges)
    pub fn len(&self) -> usize {
        self.path.len().saturating_sub(1)
    }

    /// Check if metapath is empty
    pub fn is_empty(&self) -> bool {
        self.path.is_empty()
    }

    /// Get the start node type
    pub fn start_type(&self) -> Option<NodeType> {
        self.path.first().map(|(t, _)| *t)
    }

    /// Get the end node type
    pub fn end_type(&self) -> Option<NodeType> {
        self.path.last().map(|(t, _)| *t)
    }

    /// Check if this metapath is symmetric (can be reversed)
    pub fn is_symmetric(&self) -> bool {
        if self.path.len() < 2 {
            return true;
        }

        // Check if all edges are bidirectional
        self.path
            .iter()
            .filter_map(|(_, e)| *e)
            .all(|e| e.is_bidirectional())
    }

    /// Get the sequence of edge types
    pub fn edge_sequence(&self) -> Vec<EdgeType> {
        self.path
            .iter()
            .filter_map(|(_, e)| *e)
            .collect()
    }

    /// Get the sequence of node types
    pub fn node_sequence(&self) -> Vec<NodeType> {
        self.path.iter().map(|(t, _)| *t).collect()
    }

    /// Create a short string representation
    pub fn short_repr(&self) -> String {
        let mut parts = Vec::new();
        for (i, (node_type, edge_type)) in self.path.iter().enumerate() {
            let node_char = match node_type {
                NodeType::Asset => "A",
                NodeType::Exchange => "E",
                NodeType::Wallet => "W",
                NodeType::MarketCondition => "M",
                NodeType::Timeframe => "T",
            };
            parts.push(node_char.to_string());

            if let Some(edge) = edge_type {
                let edge_char = match edge {
                    EdgeType::Correlation => "c",
                    EdgeType::Cointegration => "ci",
                    EdgeType::LeadLag => "ll",
                    EdgeType::SameSector => "s",
                    EdgeType::TradesOn => "t",
                    EdgeType::Liquidity => "l",
                    EdgeType::Holds => "h",
                    EdgeType::Accumulating => "a",
                    EdgeType::Distributing => "d",
                    EdgeType::Sensitivity => "sn",
                    EdgeType::Beta => "b",
                    EdgeType::Generic => "g",
                };
                parts.push(format!("-{}-", edge_char));
            }
        }
        parts.join("")
    }
}

/// An instance of a metapath (actual nodes following the path)
#[derive(Debug, Clone)]
pub struct MetapathInstance {
    /// The metapath definition
    pub metapath: Metapath,
    /// Sequence of node IDs following the path
    pub node_ids: Vec<String>,
    /// Aggregate weight/score for this instance
    pub weight: f64,
}

impl MetapathInstance {
    /// Create a new metapath instance
    pub fn new(metapath: Metapath, node_ids: Vec<String>) -> Self {
        Self {
            metapath,
            node_ids,
            weight: 1.0,
        }
    }

    /// Create with weight
    pub fn with_weight(metapath: Metapath, node_ids: Vec<String>, weight: f64) -> Self {
        Self {
            metapath,
            node_ids,
            weight,
        }
    }

    /// Get the start node ID
    pub fn start_node(&self) -> Option<&str> {
        self.node_ids.first().map(|s| s.as_str())
    }

    /// Get the end node ID
    pub fn end_node(&self) -> Option<&str> {
        self.node_ids.last().map(|s| s.as_str())
    }

    /// Check if this instance contains a specific node
    pub fn contains_node(&self, node_id: &str) -> bool {
        self.node_ids.iter().any(|n| n == node_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schema_creation() {
        let schema = GraphSchema::trading_schema();
        assert!(schema.is_valid_relation(
            NodeType::Asset,
            EdgeType::Correlation,
            NodeType::Asset
        ));
        assert!(schema.is_valid_relation(
            NodeType::Asset,
            EdgeType::TradesOn,
            NodeType::Exchange
        ));
        assert!(!schema.is_valid_relation(
            NodeType::Exchange,
            EdgeType::Holds,
            NodeType::Asset
        ));
    }

    #[test]
    fn test_metapath_creation() {
        let metapath = Metapath::new(
            "test",
            "Test metapath",
            vec![
                (NodeType::Asset, Some(EdgeType::Correlation)),
                (NodeType::Asset, None),
            ],
        );
        assert_eq!(metapath.len(), 1);
        assert_eq!(metapath.start_type(), Some(NodeType::Asset));
        assert_eq!(metapath.end_type(), Some(NodeType::Asset));
        assert!(metapath.is_symmetric());
    }

    #[test]
    fn test_metapath_short_repr() {
        let metapath = Metapath::new(
            "test",
            "Test",
            vec![
                (NodeType::Asset, Some(EdgeType::Correlation)),
                (NodeType::Asset, None),
            ],
        );
        assert_eq!(metapath.short_repr(), "A-c-A");
    }

    #[test]
    fn test_trading_metapaths() {
        let schema = GraphSchema::trading_schema();
        assert!(!schema.metapaths().is_empty());
        assert!(schema.get_metapath("asset_correlation").is_some());
        assert!(schema.get_metapath("same_exchange").is_some());
    }

    #[test]
    fn test_metapath_instance() {
        let metapath = Metapath::new(
            "test",
            "Test",
            vec![
                (NodeType::Asset, Some(EdgeType::Correlation)),
                (NodeType::Asset, None),
            ],
        );
        let instance = MetapathInstance::new(
            metapath,
            vec!["BTCUSDT".to_string(), "ETHUSDT".to_string()],
        );
        assert_eq!(instance.start_node(), Some("BTCUSDT"));
        assert_eq!(instance.end_node(), Some("ETHUSDT"));
        assert!(instance.contains_node("BTCUSDT"));
    }
}
