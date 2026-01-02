//! Node definitions for heterogeneous graphs
//!
//! Provides typed nodes for different entities in the trading graph.

use ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Unique identifier for a node
pub type NodeId = String;

/// Node type enumeration for heterogeneous graphs
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NodeType {
    /// Cryptocurrency asset (BTC, ETH, SOL, etc.)
    Asset,
    /// Trading exchange (Bybit, Binance, etc.)
    Exchange,
    /// Wallet address (whale wallets, exchange wallets)
    Wallet,
    /// Market condition (bull, bear, sideways)
    MarketCondition,
    /// Timeframe node for multi-resolution analysis
    Timeframe,
}

impl fmt::Display for NodeType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NodeType::Asset => write!(f, "Asset"),
            NodeType::Exchange => write!(f, "Exchange"),
            NodeType::Wallet => write!(f, "Wallet"),
            NodeType::MarketCondition => write!(f, "MarketCondition"),
            NodeType::Timeframe => write!(f, "Timeframe"),
        }
    }
}

impl NodeType {
    /// Get the feature dimension for this node type
    pub fn feature_dim(&self) -> usize {
        match self {
            NodeType::Asset => 32,
            NodeType::Exchange => 16,
            NodeType::Wallet => 12,
            NodeType::MarketCondition => 8,
            NodeType::Timeframe => 10,
        }
    }

    /// Get all node types
    pub fn all() -> Vec<NodeType> {
        vec![
            NodeType::Asset,
            NodeType::Exchange,
            NodeType::Wallet,
            NodeType::MarketCondition,
            NodeType::Timeframe,
        ]
    }
}

/// Asset-specific features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetFeatures {
    /// Current price
    pub price: f64,
    /// 24h trading volume
    pub volume_24h: f64,
    /// Volatility (standard deviation of returns)
    pub volatility: f64,
    /// Market capitalization (if available)
    pub market_cap: f64,
    /// Funding rate (for perpetuals)
    pub funding_rate: f64,
    /// Open interest
    pub open_interest: f64,
    /// 1-hour return
    pub return_1h: f64,
    /// 4-hour return
    pub return_4h: f64,
    /// 24-hour return
    pub return_24h: f64,
    /// Bid-ask spread
    pub spread: f64,
    /// Order book imbalance
    pub imbalance: f64,
    /// Timestamp
    pub timestamp: u64,
}

impl AssetFeatures {
    /// Create new asset features
    pub fn new(price: f64, volume_24h: f64, volatility: f64) -> Self {
        Self {
            price,
            volume_24h,
            volatility,
            market_cap: 0.0,
            funding_rate: 0.0,
            open_interest: 0.0,
            return_1h: 0.0,
            return_4h: 0.0,
            return_24h: 0.0,
            spread: 0.0,
            imbalance: 0.0,
            timestamp: 0,
        }
    }

    /// Convert to feature vector
    pub fn to_vector(&self) -> Array1<f64> {
        Array1::from(vec![
            self.price.ln(),  // Log price for scale invariance
            self.volume_24h.ln().max(0.0),
            self.volatility,
            self.market_cap.ln().max(0.0),
            self.funding_rate * 100.0,  // Scale funding rate
            self.open_interest.ln().max(0.0),
            self.return_1h,
            self.return_4h,
            self.return_24h,
            self.spread * 10000.0,  // Spread in bps
            self.imbalance,
            (self.timestamp as f64) / 1e12,  // Normalized timestamp
        ])
    }
}

impl Default for AssetFeatures {
    fn default() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }
}

/// Exchange-specific features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExchangeFeatures {
    /// Total 24h trading volume
    pub total_volume: f64,
    /// Number of trading pairs
    pub num_pairs: u32,
    /// Liquidity score (0-1)
    pub liquidity_score: f64,
    /// Average spread across pairs
    pub avg_spread: f64,
    /// Reliability score (0-1)
    pub reliability_score: f64,
    /// Number of active users estimate
    pub active_users: u64,
}

impl ExchangeFeatures {
    /// Create new exchange features
    pub fn new(total_volume: f64, num_pairs: u32, liquidity_score: f64) -> Self {
        Self {
            total_volume,
            num_pairs,
            liquidity_score,
            avg_spread: 0.0,
            reliability_score: 1.0,
            active_users: 0,
        }
    }

    /// Convert to feature vector
    pub fn to_vector(&self) -> Array1<f64> {
        Array1::from(vec![
            self.total_volume.ln().max(0.0),
            (self.num_pairs as f64).ln(),
            self.liquidity_score,
            self.avg_spread * 10000.0,
            self.reliability_score,
            (self.active_users as f64).ln().max(0.0),
        ])
    }
}

impl Default for ExchangeFeatures {
    fn default() -> Self {
        Self::new(0.0, 0, 0.5)
    }
}

/// Wallet-specific features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalletFeatures {
    /// Total balance in USD
    pub balance_usd: f64,
    /// Number of transactions
    pub tx_count: u64,
    /// Average transaction size
    pub avg_tx_size: f64,
    /// Wallet age in days
    pub age_days: u32,
    /// Estimated PnL
    pub pnl_estimate: f64,
    /// Activity score (0-1)
    pub activity_score: f64,
    /// Is whale (balance > threshold)
    pub is_whale: bool,
}

impl WalletFeatures {
    /// Create new wallet features
    pub fn new(balance_usd: f64, tx_count: u64, is_whale: bool) -> Self {
        Self {
            balance_usd,
            tx_count,
            avg_tx_size: 0.0,
            age_days: 0,
            pnl_estimate: 0.0,
            activity_score: 0.0,
            is_whale,
        }
    }

    /// Convert to feature vector
    pub fn to_vector(&self) -> Array1<f64> {
        Array1::from(vec![
            self.balance_usd.ln().max(0.0),
            (self.tx_count as f64).ln().max(0.0),
            self.avg_tx_size.ln().max(0.0),
            (self.age_days as f64).ln().max(0.0),
            self.pnl_estimate / 1000.0,  // Normalize PnL
            self.activity_score,
            if self.is_whale { 1.0 } else { 0.0 },
        ])
    }
}

impl Default for WalletFeatures {
    fn default() -> Self {
        Self::new(0.0, 0, false)
    }
}

/// Market condition features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketConditionFeatures {
    /// BTC dominance (0-100)
    pub btc_dominance: f64,
    /// Total market cap in billions
    pub total_market_cap: f64,
    /// Fear & Greed index (0-100)
    pub fear_greed_index: f64,
    /// Average funding rate across top pairs
    pub avg_funding_rate: f64,
    /// Long/short ratio
    pub long_short_ratio: f64,
    /// Market trend (-1 bear, 0 neutral, 1 bull)
    pub trend: f64,
    /// Volatility regime (0 low, 1 medium, 2 high)
    pub volatility_regime: u8,
}

impl MarketConditionFeatures {
    /// Create new market condition features
    pub fn new(btc_dominance: f64, fear_greed_index: f64, trend: f64) -> Self {
        Self {
            btc_dominance,
            total_market_cap: 0.0,
            fear_greed_index,
            avg_funding_rate: 0.0,
            long_short_ratio: 1.0,
            trend,
            volatility_regime: 1,
        }
    }

    /// Convert to feature vector
    pub fn to_vector(&self) -> Array1<f64> {
        Array1::from(vec![
            self.btc_dominance / 100.0,
            self.total_market_cap.ln().max(0.0),
            self.fear_greed_index / 100.0,
            self.avg_funding_rate * 100.0,
            self.long_short_ratio.ln(),
            self.trend,
            self.volatility_regime as f64 / 2.0,
        ])
    }
}

impl Default for MarketConditionFeatures {
    fn default() -> Self {
        Self::new(50.0, 50.0, 0.0)
    }
}

/// Generic node features that can hold any type-specific features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeFeatures {
    Asset(AssetFeatures),
    Exchange(ExchangeFeatures),
    Wallet(WalletFeatures),
    MarketCondition(MarketConditionFeatures),
    Generic(Array1<f64>),
}

impl NodeFeatures {
    /// Get the node type for these features
    pub fn node_type(&self) -> NodeType {
        match self {
            NodeFeatures::Asset(_) => NodeType::Asset,
            NodeFeatures::Exchange(_) => NodeType::Exchange,
            NodeFeatures::Wallet(_) => NodeType::Wallet,
            NodeFeatures::MarketCondition(_) => NodeType::MarketCondition,
            NodeFeatures::Generic(_) => NodeType::Asset,  // Default
        }
    }

    /// Convert to feature vector
    pub fn to_vector(&self) -> Array1<f64> {
        match self {
            NodeFeatures::Asset(f) => f.to_vector(),
            NodeFeatures::Exchange(f) => f.to_vector(),
            NodeFeatures::Wallet(f) => f.to_vector(),
            NodeFeatures::MarketCondition(f) => f.to_vector(),
            NodeFeatures::Generic(v) => v.clone(),
        }
    }

    /// Get feature dimension
    pub fn dim(&self) -> usize {
        self.to_vector().len()
    }
}

impl From<AssetFeatures> for NodeFeatures {
    fn from(f: AssetFeatures) -> Self {
        NodeFeatures::Asset(f)
    }
}

impl From<ExchangeFeatures> for NodeFeatures {
    fn from(f: ExchangeFeatures) -> Self {
        NodeFeatures::Exchange(f)
    }
}

impl From<WalletFeatures> for NodeFeatures {
    fn from(f: WalletFeatures) -> Self {
        NodeFeatures::Wallet(f)
    }
}

impl From<MarketConditionFeatures> for NodeFeatures {
    fn from(f: MarketConditionFeatures) -> Self {
        NodeFeatures::MarketCondition(f)
    }
}

impl Default for NodeFeatures {
    fn default() -> Self {
        NodeFeatures::Generic(Array1::zeros(8))
    }
}

/// A node in the heterogeneous graph
#[derive(Debug, Clone)]
pub struct Node {
    /// Unique identifier
    pub id: NodeId,
    /// Node type
    pub node_type: NodeType,
    /// Node features
    pub features: NodeFeatures,
    /// Last update timestamp
    pub updated_at: u64,
}

impl Node {
    /// Create a new node
    pub fn new(id: impl Into<String>, node_type: NodeType, features: NodeFeatures) -> Self {
        Self {
            id: id.into(),
            node_type,
            features,
            updated_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0),
        }
    }

    /// Update node features
    pub fn update_features(&mut self, features: NodeFeatures) {
        self.features = features;
        self.updated_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
    }

    /// Get feature vector
    pub fn feature_vector(&self) -> Array1<f64> {
        self.features.to_vector()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_types() {
        assert_eq!(NodeType::Asset.feature_dim(), 32);
        assert_eq!(NodeType::Exchange.feature_dim(), 16);
        assert_eq!(NodeType::all().len(), 5);
    }

    #[test]
    fn test_asset_features() {
        let features = AssetFeatures::new(50000.0, 1_000_000.0, 0.02);
        let vector = features.to_vector();
        assert!(vector.len() > 0);
    }

    #[test]
    fn test_node_creation() {
        let features = AssetFeatures::new(50000.0, 1_000_000.0, 0.02);
        let node = Node::new("BTCUSDT", NodeType::Asset, features.into());
        assert_eq!(node.id, "BTCUSDT");
        assert_eq!(node.node_type, NodeType::Asset);
    }
}
