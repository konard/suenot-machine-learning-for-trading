//! Data module for cryptocurrency market data
//!
//! Provides integration with Bybit exchange for fetching real-time
//! and historical market data.

mod bybit;
mod candle;

pub use bybit::BybitClient;
pub use candle::{Candle, CandleBuilder, Timeframe};

use chrono::{DateTime, Utc};
use thiserror::Error;

/// Data fetching errors
#[derive(Error, Debug)]
pub enum DataError {
    #[error("HTTP request failed: {0}")]
    HttpError(#[from] reqwest::Error),

    #[error("JSON parsing failed: {0}")]
    ParseError(#[from] serde_json::Error),

    #[error("API error: {code} - {message}")]
    ApiError { code: i32, message: String },

    #[error("Invalid symbol: {0}")]
    InvalidSymbol(String),

    #[error("Invalid timeframe: {0}")]
    InvalidTimeframe(String),

    #[error("No data available for the specified period")]
    NoData,

    #[error("Rate limit exceeded")]
    RateLimitExceeded,
}

/// Trading pair information
#[derive(Debug, Clone)]
pub struct TradingPair {
    /// Symbol (e.g., "BTCUSDT")
    pub symbol: String,
    /// Base currency (e.g., "BTC")
    pub base: String,
    /// Quote currency (e.g., "USDT")
    pub quote: String,
    /// Minimum order quantity
    pub min_qty: f64,
    /// Quantity step size
    pub qty_step: f64,
    /// Price tick size
    pub tick_size: f64,
}

/// Order book entry
#[derive(Debug, Clone, Copy)]
pub struct OrderBookLevel {
    /// Price level
    pub price: f64,
    /// Quantity at this level
    pub quantity: f64,
}

/// Order book snapshot
#[derive(Debug, Clone)]
pub struct OrderBook {
    /// Symbol
    pub symbol: String,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Bid levels (sorted by price descending)
    pub bids: Vec<OrderBookLevel>,
    /// Ask levels (sorted by price ascending)
    pub asks: Vec<OrderBookLevel>,
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

    /// Calculate mid price
    pub fn mid_price(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some((bid + ask) / 2.0),
            _ => None,
        }
    }

    /// Calculate spread
    pub fn spread(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some(ask - bid),
            _ => None,
        }
    }

    /// Calculate spread in basis points
    pub fn spread_bps(&self) -> Option<f64> {
        match (self.spread(), self.mid_price()) {
            (Some(spread), Some(mid)) => Some(spread / mid * 10000.0),
            _ => None,
        }
    }

    /// Calculate total bid volume
    pub fn total_bid_volume(&self) -> f64 {
        self.bids.iter().map(|l| l.quantity).sum()
    }

    /// Calculate total ask volume
    pub fn total_ask_volume(&self) -> f64 {
        self.asks.iter().map(|l| l.quantity).sum()
    }

    /// Calculate order book imbalance
    pub fn imbalance(&self) -> f64 {
        let bid_vol = self.total_bid_volume();
        let ask_vol = self.total_ask_volume();
        if bid_vol + ask_vol == 0.0 {
            0.0
        } else {
            (bid_vol - ask_vol) / (bid_vol + ask_vol)
        }
    }
}

/// Recent trade
#[derive(Debug, Clone)]
pub struct Trade {
    /// Trade ID
    pub id: String,
    /// Symbol
    pub symbol: String,
    /// Price
    pub price: f64,
    /// Quantity
    pub quantity: f64,
    /// Trade timestamp
    pub timestamp: DateTime<Utc>,
    /// Was the buyer the maker?
    pub is_buyer_maker: bool,
}

/// Ticker statistics
#[derive(Debug, Clone)]
pub struct Ticker {
    /// Symbol
    pub symbol: String,
    /// Last price
    pub last_price: f64,
    /// 24h price change percentage
    pub price_change_pct: f64,
    /// 24h high
    pub high_24h: f64,
    /// 24h low
    pub low_24h: f64,
    /// 24h volume
    pub volume_24h: f64,
    /// 24h turnover (in quote currency)
    pub turnover_24h: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_order_book_calculations() {
        let ob = OrderBook {
            symbol: "BTCUSDT".to_string(),
            timestamp: Utc::now(),
            bids: vec![
                OrderBookLevel {
                    price: 50000.0,
                    quantity: 1.0,
                },
                OrderBookLevel {
                    price: 49999.0,
                    quantity: 2.0,
                },
            ],
            asks: vec![
                OrderBookLevel {
                    price: 50001.0,
                    quantity: 1.5,
                },
                OrderBookLevel {
                    price: 50002.0,
                    quantity: 2.5,
                },
            ],
        };

        assert_eq!(ob.best_bid(), Some(50000.0));
        assert_eq!(ob.best_ask(), Some(50001.0));
        assert_eq!(ob.mid_price(), Some(50000.5));
        assert_eq!(ob.spread(), Some(1.0));
        assert_eq!(ob.total_bid_volume(), 3.0);
        assert_eq!(ob.total_ask_volume(), 4.0);

        // Imbalance = (3 - 4) / (3 + 4) = -1/7
        let imbalance = ob.imbalance();
        assert!((imbalance - (-1.0 / 7.0)).abs() < 1e-10);
    }
}
