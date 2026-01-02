//! Edge definitions for heterogeneous graphs
//!
//! Provides typed edges for different relationships in the trading graph.

use ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::fmt;

use super::NodeId;

/// Unique identifier for an edge
pub type EdgeId = (NodeId, NodeId, EdgeType);

/// Edge type enumeration for heterogeneous graphs
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EdgeType {
    // Asset-Asset edges
    /// Price correlation between assets
    Correlation,
    /// Statistical cointegration
    Cointegration,
    /// Lead-lag relationship (Granger causality)
    LeadLag,
    /// Same sector/category
    SameSector,

    // Asset-Exchange edges
    /// Asset is traded on exchange
    TradesOn,
    /// Liquidity relationship
    Liquidity,

    // Wallet-Asset edges
    /// Wallet holds asset
    Holds,
    /// Wallet is accumulating asset
    Accumulating,
    /// Wallet is distributing asset
    Distributing,

    // Asset-MarketCondition edges
    /// Asset sensitivity to market condition
    Sensitivity,
    /// Market beta
    Beta,

    // Generic
    /// Generic/unknown relationship
    Generic,
}

impl fmt::Display for EdgeType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EdgeType::Correlation => write!(f, "correlation"),
            EdgeType::Cointegration => write!(f, "cointegration"),
            EdgeType::LeadLag => write!(f, "lead_lag"),
            EdgeType::SameSector => write!(f, "same_sector"),
            EdgeType::TradesOn => write!(f, "trades_on"),
            EdgeType::Liquidity => write!(f, "liquidity"),
            EdgeType::Holds => write!(f, "holds"),
            EdgeType::Accumulating => write!(f, "accumulating"),
            EdgeType::Distributing => write!(f, "distributing"),
            EdgeType::Sensitivity => write!(f, "sensitivity"),
            EdgeType::Beta => write!(f, "beta"),
            EdgeType::Generic => write!(f, "generic"),
        }
    }
}

impl EdgeType {
    /// Get all edge types
    pub fn all() -> Vec<EdgeType> {
        vec![
            EdgeType::Correlation,
            EdgeType::Cointegration,
            EdgeType::LeadLag,
            EdgeType::SameSector,
            EdgeType::TradesOn,
            EdgeType::Liquidity,
            EdgeType::Holds,
            EdgeType::Accumulating,
            EdgeType::Distributing,
            EdgeType::Sensitivity,
            EdgeType::Beta,
            EdgeType::Generic,
        ]
    }

    /// Check if edge is bidirectional
    pub fn is_bidirectional(&self) -> bool {
        matches!(
            self,
            EdgeType::Correlation
                | EdgeType::Cointegration
                | EdgeType::SameSector
                | EdgeType::Generic
        )
    }

    /// Get the feature dimension for this edge type
    pub fn feature_dim(&self) -> usize {
        match self {
            EdgeType::Correlation => 4,
            EdgeType::Cointegration => 3,
            EdgeType::LeadLag => 5,
            EdgeType::SameSector => 2,
            EdgeType::TradesOn => 6,
            EdgeType::Liquidity => 4,
            EdgeType::Holds => 5,
            EdgeType::Accumulating => 4,
            EdgeType::Distributing => 4,
            EdgeType::Sensitivity => 3,
            EdgeType::Beta => 2,
            EdgeType::Generic => 1,
        }
    }

    /// Get edge type from string
    pub fn from_str(s: &str) -> Option<EdgeType> {
        match s.to_lowercase().as_str() {
            "correlation" | "corr" => Some(EdgeType::Correlation),
            "cointegration" | "coint" => Some(EdgeType::Cointegration),
            "lead_lag" | "leadlag" => Some(EdgeType::LeadLag),
            "same_sector" | "sector" => Some(EdgeType::SameSector),
            "trades_on" | "trades" => Some(EdgeType::TradesOn),
            "liquidity" | "liq" => Some(EdgeType::Liquidity),
            "holds" | "hold" => Some(EdgeType::Holds),
            "accumulating" | "accum" => Some(EdgeType::Accumulating),
            "distributing" | "dist" => Some(EdgeType::Distributing),
            "sensitivity" | "sens" => Some(EdgeType::Sensitivity),
            "beta" => Some(EdgeType::Beta),
            _ => Some(EdgeType::Generic),
        }
    }
}

/// Edge features for different edge types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeFeatures {
    /// Edge weight (importance)
    pub weight: f64,
    /// Correlation value (for correlation edges)
    pub correlation: Option<f64>,
    /// Lead-lag in seconds (for lead-lag edges)
    pub lead_lag_seconds: Option<i64>,
    /// Volume (for liquidity/trading edges)
    pub volume: Option<f64>,
    /// Amount (for holding edges)
    pub amount: Option<f64>,
    /// Confidence score (0-1)
    pub confidence: f64,
    /// Timestamp of last update
    pub timestamp: u64,
    /// Additional features as vector
    pub extra_features: Option<Vec<f64>>,
}

impl EdgeFeatures {
    /// Create new edge features with just weight
    pub fn new(weight: f64) -> Self {
        Self {
            weight,
            correlation: None,
            lead_lag_seconds: None,
            volume: None,
            amount: None,
            confidence: 1.0,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0),
            extra_features: None,
        }
    }

    /// Create correlation edge features
    pub fn with_correlation(correlation: f64, timestamp: u64) -> Self {
        Self {
            weight: correlation.abs(),
            correlation: Some(correlation),
            lead_lag_seconds: None,
            volume: None,
            amount: None,
            confidence: 1.0,
            timestamp,
            extra_features: None,
        }
    }

    /// Create lead-lag edge features
    pub fn with_lead_lag(correlation: f64, lag_seconds: i64, timestamp: u64) -> Self {
        Self {
            weight: correlation.abs(),
            correlation: Some(correlation),
            lead_lag_seconds: Some(lag_seconds),
            volume: None,
            amount: None,
            confidence: 1.0,
            timestamp,
            extra_features: None,
        }
    }

    /// Create holding edge features
    pub fn with_holding(amount: f64, timestamp: u64) -> Self {
        Self {
            weight: amount.ln().max(0.0) / 10.0,  // Normalize weight
            correlation: None,
            lead_lag_seconds: None,
            volume: None,
            amount: Some(amount),
            confidence: 1.0,
            timestamp,
            extra_features: None,
        }
    }

    /// Create trading/liquidity edge features
    pub fn with_volume(volume: f64, timestamp: u64) -> Self {
        Self {
            weight: volume.ln().max(0.0) / 10.0,  // Normalize weight
            correlation: None,
            lead_lag_seconds: None,
            volume: Some(volume),
            amount: None,
            confidence: 1.0,
            timestamp,
            extra_features: None,
        }
    }

    /// Convert to feature vector for the given edge type
    pub fn to_vector(&self, edge_type: EdgeType) -> Array1<f64> {
        let mut features = vec![self.weight, self.confidence];

        match edge_type {
            EdgeType::Correlation => {
                features.push(self.correlation.unwrap_or(0.0));
                features.push((self.timestamp as f64) / 1e12);
            }
            EdgeType::Cointegration => {
                features.push(self.correlation.unwrap_or(0.0));
            }
            EdgeType::LeadLag => {
                features.push(self.correlation.unwrap_or(0.0));
                features.push(self.lead_lag_seconds.unwrap_or(0) as f64 / 3600.0);
                features.push((self.timestamp as f64) / 1e12);
            }
            EdgeType::SameSector => {
                // Just weight and confidence
            }
            EdgeType::TradesOn | EdgeType::Liquidity => {
                features.push(self.volume.unwrap_or(0.0).ln().max(0.0));
                features.push((self.timestamp as f64) / 1e12);
            }
            EdgeType::Holds | EdgeType::Accumulating | EdgeType::Distributing => {
                features.push(self.amount.unwrap_or(0.0).ln().max(0.0));
                features.push((self.timestamp as f64) / 1e12);
            }
            EdgeType::Sensitivity | EdgeType::Beta => {
                features.push(self.correlation.unwrap_or(0.0));
            }
            EdgeType::Generic => {
                // Just weight and confidence
            }
        }

        // Add extra features if present
        if let Some(ref extra) = self.extra_features {
            features.extend(extra.iter());
        }

        Array1::from(features)
    }
}

impl Default for EdgeFeatures {
    fn default() -> Self {
        Self::new(1.0)
    }
}

/// An edge in the heterogeneous graph
#[derive(Debug, Clone)]
pub struct Edge {
    /// Source node ID
    pub source: NodeId,
    /// Target node ID
    pub target: NodeId,
    /// Edge type
    pub edge_type: EdgeType,
    /// Edge features
    pub features: EdgeFeatures,
}

impl Edge {
    /// Create a new edge
    pub fn new(
        source: impl Into<String>,
        target: impl Into<String>,
        edge_type: EdgeType,
        features: EdgeFeatures,
    ) -> Self {
        Self {
            source: source.into(),
            target: target.into(),
            edge_type,
            features,
        }
    }

    /// Get edge ID (tuple of source, target, type)
    pub fn id(&self) -> EdgeId {
        (self.source.clone(), self.target.clone(), self.edge_type)
    }

    /// Get feature vector
    pub fn feature_vector(&self) -> Array1<f64> {
        self.features.to_vector(self.edge_type)
    }

    /// Check if edge connects given nodes
    pub fn connects(&self, node1: &str, node2: &str) -> bool {
        (self.source == node1 && self.target == node2)
            || (self.edge_type.is_bidirectional() && self.source == node2 && self.target == node1)
    }
}

/// Relation triplet (source_type, edge_type, target_type)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Relation {
    pub source_type: super::NodeType,
    pub edge_type: EdgeType,
    pub target_type: super::NodeType,
}

impl Relation {
    pub fn new(
        source_type: super::NodeType,
        edge_type: EdgeType,
        target_type: super::NodeType,
    ) -> Self {
        Self {
            source_type,
            edge_type,
            target_type,
        }
    }
}

impl fmt::Display for Relation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "({}, {}, {})",
            self.source_type, self.edge_type, self.target_type
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_edge_types() {
        assert!(EdgeType::Correlation.is_bidirectional());
        assert!(!EdgeType::LeadLag.is_bidirectional());
        assert!(!EdgeType::Holds.is_bidirectional());
    }

    #[test]
    fn test_edge_features_correlation() {
        let features = EdgeFeatures::with_correlation(0.85, 1000);
        assert_eq!(features.correlation, Some(0.85));
        assert_eq!(features.weight, 0.85);
    }

    #[test]
    fn test_edge_creation() {
        let features = EdgeFeatures::with_correlation(0.75, 1000);
        let edge = Edge::new("BTCUSDT", "ETHUSDT", EdgeType::Correlation, features);
        assert_eq!(edge.source, "BTCUSDT");
        assert_eq!(edge.target, "ETHUSDT");
        assert!(edge.connects("BTCUSDT", "ETHUSDT"));
        assert!(edge.connects("ETHUSDT", "BTCUSDT"));  // Bidirectional
    }

    #[test]
    fn test_edge_feature_vector() {
        let features = EdgeFeatures::with_correlation(0.85, 1000);
        let vector = features.to_vector(EdgeType::Correlation);
        assert!(vector.len() >= 2);
    }
}
