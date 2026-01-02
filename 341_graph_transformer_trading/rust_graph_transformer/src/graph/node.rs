//! Graph node representation for cryptocurrency assets
//!
//! Each node represents a cryptocurrency with its features.

use ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Cryptocurrency node in the market graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CryptoNode {
    /// Unique node ID
    pub id: usize,
    /// Trading symbol (e.g., "BTCUSDT")
    pub symbol: String,
    /// Current price
    pub price: f64,
    /// 24h price change percentage
    pub price_change_24h: f64,
    /// 24h trading volume
    pub volume_24h: f64,
    /// Market capitalization rank
    pub market_cap_rank: Option<u32>,
    /// Sector classification
    pub sector: Option<String>,
    /// Feature vector for the node
    #[serde(skip)]
    pub features: Option<Array1<f64>>,
    /// Embedding vector (from transformer)
    #[serde(skip)]
    pub embedding: Option<Array1<f64>>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl CryptoNode {
    /// Create a new crypto node
    pub fn new(id: usize, symbol: &str, price: f64) -> Self {
        Self {
            id,
            symbol: symbol.to_string(),
            price,
            price_change_24h: 0.0,
            volume_24h: 0.0,
            market_cap_rank: None,
            sector: None,
            features: None,
            embedding: None,
            metadata: HashMap::new(),
        }
    }

    /// Create with full data
    pub fn with_data(
        id: usize,
        symbol: &str,
        price: f64,
        price_change_24h: f64,
        volume_24h: f64,
    ) -> Self {
        Self {
            id,
            symbol: symbol.to_string(),
            price,
            price_change_24h,
            volume_24h,
            market_cap_rank: None,
            sector: None,
            features: None,
            embedding: None,
            metadata: HashMap::new(),
        }
    }

    /// Set the sector classification
    pub fn with_sector(mut self, sector: &str) -> Self {
        self.sector = Some(sector.to_string());
        self
    }

    /// Set the market cap rank
    pub fn with_rank(mut self, rank: u32) -> Self {
        self.market_cap_rank = Some(rank);
        self
    }

    /// Set the feature vector
    pub fn with_features(mut self, features: Array1<f64>) -> Self {
        self.features = Some(features);
        self
    }

    /// Update the embedding from transformer output
    pub fn set_embedding(&mut self, embedding: Array1<f64>) {
        self.embedding = Some(embedding);
    }

    /// Get feature dimension
    pub fn feature_dim(&self) -> Option<usize> {
        self.features.as_ref().map(|f| f.len())
    }

    /// Get embedding dimension
    pub fn embedding_dim(&self) -> Option<usize> {
        self.embedding.as_ref().map(|e| e.len())
    }

    /// Check if node has features
    pub fn has_features(&self) -> bool {
        self.features.is_some()
    }

    /// Check if node has embedding
    pub fn has_embedding(&self) -> bool {
        self.embedding.is_some()
    }

    /// Get normalized price change (for quick comparisons)
    pub fn is_bullish(&self) -> bool {
        self.price_change_24h > 0.0
    }

    /// Get volatility indicator based on 24h change
    pub fn volatility_indicator(&self) -> f64 {
        self.price_change_24h.abs()
    }
}

/// Builder for CryptoNode with fluent interface
#[derive(Default)]
pub struct CryptoNodeBuilder {
    id: usize,
    symbol: String,
    price: f64,
    price_change_24h: f64,
    volume_24h: f64,
    market_cap_rank: Option<u32>,
    sector: Option<String>,
    features: Option<Array1<f64>>,
}

impl CryptoNodeBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn id(mut self, id: usize) -> Self {
        self.id = id;
        self
    }

    pub fn symbol(mut self, symbol: &str) -> Self {
        self.symbol = symbol.to_string();
        self
    }

    pub fn price(mut self, price: f64) -> Self {
        self.price = price;
        self
    }

    pub fn price_change(mut self, change: f64) -> Self {
        self.price_change_24h = change;
        self
    }

    pub fn volume(mut self, volume: f64) -> Self {
        self.volume_24h = volume;
        self
    }

    pub fn rank(mut self, rank: u32) -> Self {
        self.market_cap_rank = Some(rank);
        self
    }

    pub fn sector(mut self, sector: &str) -> Self {
        self.sector = Some(sector.to_string());
        self
    }

    pub fn features(mut self, features: Array1<f64>) -> Self {
        self.features = Some(features);
        self
    }

    pub fn build(self) -> CryptoNode {
        CryptoNode {
            id: self.id,
            symbol: self.symbol,
            price: self.price,
            price_change_24h: self.price_change_24h,
            volume_24h: self.volume_24h,
            market_cap_rank: self.market_cap_rank,
            sector: self.sector,
            features: self.features,
            embedding: None,
            metadata: HashMap::new(),
        }
    }
}

/// Sector classifications for cryptocurrencies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CryptoSector {
    /// Layer 1 blockchains (BTC, ETH, SOL, etc.)
    Layer1,
    /// Layer 2 solutions (MATIC, ARB, OP, etc.)
    Layer2,
    /// DeFi protocols (UNI, AAVE, COMP, etc.)
    DeFi,
    /// Exchange tokens (BNB, FTT, etc.)
    Exchange,
    /// Meme coins (DOGE, SHIB, PEPE, etc.)
    Meme,
    /// Stablecoins (USDT, USDC, DAI, etc.)
    Stablecoin,
    /// Gaming/Metaverse (SAND, MANA, AXS, etc.)
    Gaming,
    /// Infrastructure (LINK, GRT, FIL, etc.)
    Infrastructure,
    /// Privacy coins (XMR, ZEC, etc.)
    Privacy,
    /// Other
    Other,
}

impl CryptoSector {
    /// Get sector from symbol (simplified classification)
    pub fn from_symbol(symbol: &str) -> Self {
        let s = symbol.to_uppercase();
        match s.as_str() {
            "BTCUSDT" | "ETHUSDT" | "SOLUSDT" | "AVAXUSDT" | "ADAUSDT" | "DOTUSDT" => Self::Layer1,
            "MATICUSDT" | "ARBUSDT" | "OPUSDT" => Self::Layer2,
            "UNIUSDT" | "AAVEUSDT" | "COMPUSDT" | "MKRUSDT" | "CRVUSDT" => Self::DeFi,
            "BNBUSDT" | "OKBUSDT" => Self::Exchange,
            "DOGEUSDT" | "SHIBUSDT" | "PEPEUSDT" | "FLOKIUSDT" => Self::Meme,
            "USDTUSDC" | "DAIUSDT" => Self::Stablecoin,
            "SANDUSDT" | "MANAUSDT" | "AXSUSDT" | "ENJUSDT" => Self::Gaming,
            "LINKUSDT" | "GRTUSDT" | "FILUSDT" | "ATOMUSDT" => Self::Infrastructure,
            "XMRUSDT" | "ZECUSDT" => Self::Privacy,
            _ => Self::Other,
        }
    }

    /// Check if two sectors are related
    pub fn is_related(&self, other: &Self) -> bool {
        if self == other {
            return true;
        }
        // Some sectors are related
        matches!(
            (self, other),
            (Self::Layer1, Self::Layer2)
                | (Self::Layer2, Self::Layer1)
                | (Self::DeFi, Self::Layer1)
                | (Self::Layer1, Self::DeFi)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_creation() {
        let node = CryptoNode::new(0, "BTCUSDT", 45000.0);
        assert_eq!(node.symbol, "BTCUSDT");
        assert_eq!(node.price, 45000.0);
        assert_eq!(node.id, 0);
    }

    #[test]
    fn test_node_builder() {
        let node = CryptoNodeBuilder::new()
            .id(0)
            .symbol("BTCUSDT")
            .price(45000.0)
            .price_change(5.5)
            .volume(1000000.0)
            .sector("Layer1")
            .build();

        assert_eq!(node.price_change_24h, 5.5);
        assert!(node.is_bullish());
    }

    #[test]
    fn test_sector_classification() {
        assert_eq!(CryptoSector::from_symbol("BTCUSDT"), CryptoSector::Layer1);
        assert_eq!(CryptoSector::from_symbol("DOGEUSDT"), CryptoSector::Meme);
        assert_eq!(CryptoSector::from_symbol("MATICUSDT"), CryptoSector::Layer2);
    }
}
