//! Exchange Integration Module
//!
//! Provides connectivity to cryptocurrency exchanges for market data and trading.

pub mod bybit;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Order book level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookLevel {
    /// Price
    pub price: f64,
    /// Quantity
    pub quantity: f64,
}

/// Order book snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBook {
    /// Symbol
    pub symbol: String,
    /// Bid levels (best first)
    pub bids: Vec<OrderBookLevel>,
    /// Ask levels (best first)
    pub asks: Vec<OrderBookLevel>,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Sequence number
    pub sequence: u64,
}

impl OrderBook {
    /// Get best bid price
    pub fn best_bid(&self) -> Option<f64> {
        self.bids.first().map(|l| l.price)
    }

    /// Get best ask price
    pub fn best_ask(&self) -> Option<f64> {
        self.asks.first().map(|l| l.price)
    }

    /// Get mid price
    pub fn mid_price(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some((bid + ask) / 2.0),
            _ => None,
        }
    }

    /// Get spread
    pub fn spread(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some(ask - bid),
            _ => None,
        }
    }

    /// Get spread in basis points
    pub fn spread_bps(&self) -> Option<f64> {
        match (self.spread(), self.mid_price()) {
            (Some(spread), Some(mid)) if mid > 0.0 => Some(spread / mid * 10000.0),
            _ => None,
        }
    }
}

/// Trade data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    /// Symbol
    pub symbol: String,
    /// Trade ID
    pub trade_id: String,
    /// Price
    pub price: f64,
    /// Quantity
    pub quantity: f64,
    /// Side (buy/sell)
    pub side: TradeSide,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

/// Trade side
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TradeSide {
    Buy,
    Sell,
}

/// Ticker data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ticker {
    /// Symbol
    pub symbol: String,
    /// Last price
    pub last_price: f64,
    /// 24h high
    pub high_24h: f64,
    /// 24h low
    pub low_24h: f64,
    /// 24h volume
    pub volume_24h: f64,
    /// 24h price change percentage
    pub price_change_pct: f64,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

/// Exchange client trait
#[async_trait::async_trait]
pub trait ExchangeClient: Send + Sync {
    /// Get order book for a symbol
    async fn get_orderbook(&self, symbol: &str, depth: usize) -> anyhow::Result<OrderBook>;

    /// Get recent trades for a symbol
    async fn get_trades(&self, symbol: &str, limit: usize) -> anyhow::Result<Vec<Trade>>;

    /// Get ticker for a symbol
    async fn get_ticker(&self, symbol: &str) -> anyhow::Result<Ticker>;
}

// Re-export async_trait for external use
pub use async_trait::async_trait;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_orderbook_mid_price() {
        let ob = OrderBook {
            symbol: "BTCUSDT".to_string(),
            bids: vec![OrderBookLevel { price: 50000.0, quantity: 1.0 }],
            asks: vec![OrderBookLevel { price: 50010.0, quantity: 1.0 }],
            timestamp: Utc::now(),
            sequence: 1,
        };

        assert_eq!(ob.mid_price(), Some(50005.0));
        assert_eq!(ob.spread(), Some(10.0));
    }

    #[test]
    fn test_orderbook_spread_bps() {
        let ob = OrderBook {
            symbol: "BTCUSDT".to_string(),
            bids: vec![OrderBookLevel { price: 50000.0, quantity: 1.0 }],
            asks: vec![OrderBookLevel { price: 50050.0, quantity: 1.0 }],
            timestamp: Utc::now(),
            sequence: 1,
        };

        let spread_bps = ob.spread_bps().unwrap();
        assert!((spread_bps - 10.0).abs() < 0.1);  // ~10 bps
    }
}
