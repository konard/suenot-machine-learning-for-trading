//! Data module for Bybit API and market data processing
//!
//! This module provides functionality to fetch and process
//! cryptocurrency market data from Bybit exchange.

mod bybit;
mod orderbook;
mod features;

pub use bybit::{BybitClient, BybitConfig, BybitError};
pub use orderbook::{OrderBook, OrderBookLevel, OrderBookSnapshot};
pub use features::{FeatureEngine, MarketFeatures};

use serde::{Deserialize, Serialize};

/// OHLCV Kline/Candlestick data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Kline {
    /// Open timestamp (milliseconds)
    pub timestamp: u64,
    /// Open price
    pub open: f64,
    /// High price
    pub high: f64,
    /// Low price
    pub low: f64,
    /// Close price
    pub close: f64,
    /// Volume
    pub volume: f64,
    /// Turnover (quote volume)
    pub turnover: f64,
}

impl Kline {
    /// Calculate return from open to close
    pub fn return_oc(&self) -> f64 {
        if self.open > 0.0 {
            (self.close - self.open) / self.open
        } else {
            0.0
        }
    }

    /// Calculate range
    pub fn range(&self) -> f64 {
        self.high - self.low
    }

    /// Calculate range as percentage of close
    pub fn range_pct(&self) -> f64 {
        if self.close > 0.0 {
            self.range() / self.close
        } else {
            0.0
        }
    }

    /// Check if bullish candle
    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }
}

/// Real-time ticker data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ticker {
    /// Symbol
    pub symbol: String,
    /// Last traded price
    pub last_price: f64,
    /// 24h high
    pub high_24h: f64,
    /// 24h low
    pub low_24h: f64,
    /// 24h volume
    pub volume_24h: f64,
    /// 24h turnover
    pub turnover_24h: f64,
    /// Price change 24h percentage
    pub price_change_24h: f64,
    /// Best bid price
    pub bid_price: f64,
    /// Best ask price
    pub ask_price: f64,
    /// Timestamp
    pub timestamp: u64,
}

impl Ticker {
    /// Calculate bid-ask spread
    pub fn spread(&self) -> f64 {
        self.ask_price - self.bid_price
    }

    /// Calculate spread as percentage
    pub fn spread_pct(&self) -> f64 {
        if self.bid_price > 0.0 {
            self.spread() / self.bid_price
        } else {
            0.0
        }
    }

    /// Calculate mid price
    pub fn mid_price(&self) -> f64 {
        (self.bid_price + self.ask_price) / 2.0
    }
}

/// Funding rate data for perpetual contracts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FundingRate {
    /// Symbol
    pub symbol: String,
    /// Funding rate
    pub funding_rate: f64,
    /// Funding rate timestamp
    pub funding_rate_timestamp: u64,
    /// Next funding time
    pub next_funding_time: u64,
}

/// Open interest data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenInterest {
    /// Symbol
    pub symbol: String,
    /// Open interest value
    pub open_interest: f64,
    /// Timestamp
    pub timestamp: u64,
}

/// Trade data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    /// Trade ID
    pub id: String,
    /// Symbol
    pub symbol: String,
    /// Price
    pub price: f64,
    /// Size/Quantity
    pub size: f64,
    /// Side: "Buy" or "Sell"
    pub side: String,
    /// Timestamp
    pub timestamp: u64,
}

impl Trade {
    /// Check if buy trade
    pub fn is_buy(&self) -> bool {
        self.side == "Buy"
    }

    /// Get signed size (positive for buy, negative for sell)
    pub fn signed_size(&self) -> f64 {
        if self.is_buy() {
            self.size
        } else {
            -self.size
        }
    }
}

/// Market data snapshot for a symbol
#[derive(Debug, Clone)]
pub struct MarketSnapshot {
    /// Symbol
    pub symbol: String,
    /// Current ticker
    pub ticker: Option<Ticker>,
    /// Current order book
    pub orderbook: Option<OrderBookSnapshot>,
    /// Recent klines
    pub klines: Vec<Kline>,
    /// Recent trades
    pub trades: Vec<Trade>,
    /// Funding rate (for perps)
    pub funding: Option<FundingRate>,
    /// Open interest (for perps)
    pub open_interest: Option<OpenInterest>,
    /// Timestamp
    pub timestamp: u64,
}

impl MarketSnapshot {
    /// Create new empty snapshot
    pub fn new(symbol: impl Into<String>) -> Self {
        Self {
            symbol: symbol.into(),
            ticker: None,
            orderbook: None,
            klines: Vec::new(),
            trades: Vec::new(),
            funding: None,
            open_interest: None,
            timestamp: 0,
        }
    }

    /// Get current price
    pub fn price(&self) -> Option<f64> {
        self.ticker.as_ref().map(|t| t.last_price)
    }

    /// Calculate VWAP from recent trades
    pub fn vwap(&self) -> Option<f64> {
        if self.trades.is_empty() {
            return None;
        }

        let mut total_value = 0.0;
        let mut total_volume = 0.0;

        for trade in &self.trades {
            total_value += trade.price * trade.size;
            total_volume += trade.size;
        }

        if total_volume > 0.0 {
            Some(total_value / total_volume)
        } else {
            None
        }
    }

    /// Calculate buy/sell imbalance from recent trades
    pub fn trade_imbalance(&self) -> f64 {
        if self.trades.is_empty() {
            return 0.0;
        }

        let mut buy_volume = 0.0;
        let mut sell_volume = 0.0;

        for trade in &self.trades {
            if trade.is_buy() {
                buy_volume += trade.size;
            } else {
                sell_volume += trade.size;
            }
        }

        let total = buy_volume + sell_volume;
        if total > 0.0 {
            (buy_volume - sell_volume) / total
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kline() {
        let kline = Kline {
            timestamp: 1000,
            open: 100.0,
            high: 110.0,
            low: 95.0,
            close: 105.0,
            volume: 1000.0,
            turnover: 100000.0,
        };

        assert_eq!(kline.return_oc(), 0.05);
        assert_eq!(kline.range(), 15.0);
        assert!(kline.is_bullish());
    }

    #[test]
    fn test_ticker() {
        let ticker = Ticker {
            symbol: "BTCUSDT".to_string(),
            last_price: 50000.0,
            high_24h: 51000.0,
            low_24h: 49000.0,
            volume_24h: 10000.0,
            turnover_24h: 500000000.0,
            price_change_24h: 2.0,
            bid_price: 49995.0,
            ask_price: 50005.0,
            timestamp: 1000,
        };

        assert_eq!(ticker.spread(), 10.0);
        assert_eq!(ticker.mid_price(), 50000.0);
    }

    #[test]
    fn test_trade() {
        let buy_trade = Trade {
            id: "1".to_string(),
            symbol: "BTCUSDT".to_string(),
            price: 50000.0,
            size: 1.0,
            side: "Buy".to_string(),
            timestamp: 1000,
        };

        assert!(buy_trade.is_buy());
        assert_eq!(buy_trade.signed_size(), 1.0);

        let sell_trade = Trade {
            id: "2".to_string(),
            symbol: "BTCUSDT".to_string(),
            price: 50000.0,
            size: 1.0,
            side: "Sell".to_string(),
            timestamp: 1000,
        };

        assert!(!sell_trade.is_buy());
        assert_eq!(sell_trade.signed_size(), -1.0);
    }
}
