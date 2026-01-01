//! Data models for Bybit API responses

use serde::{Deserialize, Serialize};

/// Kline (candlestick) data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Kline {
    /// Start timestamp (milliseconds)
    pub start_time: i64,
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
    /// Calculate typical price (HLC/3)
    pub fn typical_price(&self) -> f64 {
        (self.high + self.low + self.close) / 3.0
    }

    /// Calculate OHLC average
    pub fn ohlc_average(&self) -> f64 {
        (self.open + self.high + self.low + self.close) / 4.0
    }

    /// Calculate true range
    pub fn true_range(&self, prev_close: Option<f64>) -> f64 {
        let hl = self.high - self.low;
        match prev_close {
            Some(pc) => {
                let hc = (self.high - pc).abs();
                let lc = (self.low - pc).abs();
                hl.max(hc).max(lc)
            }
            None => hl,
        }
    }

    /// Calculate log return from previous close
    pub fn log_return(&self, prev_close: f64) -> f64 {
        (self.close / prev_close).ln()
    }

    /// Check if bullish candle
    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }

    /// Calculate body size
    pub fn body_size(&self) -> f64 {
        (self.close - self.open).abs()
    }

    /// Calculate upper shadow
    pub fn upper_shadow(&self) -> f64 {
        self.high - self.close.max(self.open)
    }

    /// Calculate lower shadow
    pub fn lower_shadow(&self) -> f64 {
        self.close.min(self.open) - self.low
    }
}

/// Order book data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBook {
    /// Symbol
    pub symbol: String,
    /// Bid prices and quantities
    pub bids: Vec<(f64, f64)>,
    /// Ask prices and quantities
    pub asks: Vec<(f64, f64)>,
    /// Timestamp
    pub timestamp: i64,
}

impl OrderBook {
    /// Get best bid price
    pub fn best_bid(&self) -> Option<f64> {
        self.bids.first().map(|(p, _)| *p)
    }

    /// Get best ask price
    pub fn best_ask(&self) -> Option<f64> {
        self.asks.first().map(|(p, _)| *p)
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

    /// Calculate bid-ask imbalance
    pub fn imbalance(&self, depth: usize) -> f64 {
        let bid_volume: f64 = self.bids.iter()
            .take(depth)
            .map(|(_, q)| q)
            .sum();
        let ask_volume: f64 = self.asks.iter()
            .take(depth)
            .map(|(_, q)| q)
            .sum();

        let total = bid_volume + ask_volume;
        if total > 0.0 {
            (bid_volume - ask_volume) / total
        } else {
            0.0
        }
    }

    /// Calculate weighted mid price (microprice)
    pub fn microprice(&self) -> Option<f64> {
        match (self.bids.first(), self.asks.first()) {
            (Some((bid_p, bid_q)), Some((ask_p, ask_q))) => {
                let total = bid_q + ask_q;
                if total > 0.0 {
                    Some((bid_p * ask_q + ask_p * bid_q) / total)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Get volume at price level
    pub fn volume_at_price(&self, price: f64, tolerance: f64) -> f64 {
        let mut volume = 0.0;

        for (p, q) in &self.bids {
            if (*p - price).abs() <= tolerance {
                volume += q;
            }
        }
        for (p, q) in &self.asks {
            if (*p - price).abs() <= tolerance {
                volume += q;
            }
        }

        volume
    }
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
    /// Quantity
    pub quantity: f64,
    /// Side (Buy/Sell)
    pub side: TradeSide,
    /// Timestamp
    pub timestamp: i64,
}

/// Trade side
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
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
    /// 24h turnover
    pub turnover_24h: f64,
    /// Price change percentage
    pub price_change_percent: f64,
    /// Timestamp
    pub timestamp: i64,
}

/// API response wrapper
#[derive(Debug, Clone, Deserialize)]
pub struct ApiResponse<T> {
    /// Return code (0 = success)
    #[serde(rename = "retCode")]
    pub ret_code: i32,
    /// Return message
    #[serde(rename = "retMsg")]
    pub ret_msg: String,
    /// Result data
    pub result: T,
    /// Server time
    pub time: i64,
}

/// Kline list response
#[derive(Debug, Clone, Deserialize)]
pub struct KlineListResult {
    /// Symbol
    pub symbol: String,
    /// Category
    pub category: String,
    /// List of klines (as arrays)
    pub list: Vec<Vec<String>>,
}

/// Order book response
#[derive(Debug, Clone, Deserialize)]
pub struct OrderBookResult {
    /// Symbol
    #[serde(rename = "s")]
    pub symbol: String,
    /// Bids
    #[serde(rename = "b")]
    pub bids: Vec<Vec<String>>,
    /// Asks
    #[serde(rename = "a")]
    pub asks: Vec<Vec<String>>,
    /// Timestamp
    #[serde(rename = "ts")]
    pub timestamp: i64,
}

/// Symbol info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolInfo {
    /// Symbol name
    pub symbol: String,
    /// Base currency
    pub base_coin: String,
    /// Quote currency
    pub quote_coin: String,
    /// Contract type
    pub contract_type: String,
    /// Status
    pub status: String,
    /// Price filter
    pub price_filter: PriceFilter,
    /// Lot size filter
    pub lot_size_filter: LotSizeFilter,
}

/// Price filter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceFilter {
    /// Minimum price
    pub min_price: f64,
    /// Maximum price
    pub max_price: f64,
    /// Tick size
    pub tick_size: f64,
}

/// Lot size filter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LotSizeFilter {
    /// Minimum quantity
    pub min_qty: f64,
    /// Maximum quantity
    pub max_qty: f64,
    /// Quantity step
    pub qty_step: f64,
}

/// Funding rate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FundingRate {
    /// Symbol
    pub symbol: String,
    /// Funding rate
    pub funding_rate: f64,
    /// Next funding time
    pub funding_rate_timestamp: i64,
}

/// Market depth snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DepthSnapshot {
    /// Update ID
    pub update_id: i64,
    /// Bids
    pub bids: Vec<(f64, f64)>,
    /// Asks
    pub asks: Vec<(f64, f64)>,
}

impl DepthSnapshot {
    /// Convert to OrderBook
    pub fn to_orderbook(&self, symbol: &str) -> OrderBook {
        OrderBook {
            symbol: symbol.to_string(),
            bids: self.bids.clone(),
            asks: self.asks.clone(),
            timestamp: chrono::Utc::now().timestamp_millis(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kline_metrics() {
        let kline = Kline {
            start_time: 0,
            open: 100.0,
            high: 110.0,
            low: 95.0,
            close: 105.0,
            volume: 1000.0,
            turnover: 100000.0,
        };

        assert!((kline.typical_price() - 103.333).abs() < 0.01);
        assert!(kline.is_bullish());
        assert_eq!(kline.body_size(), 5.0);
        assert_eq!(kline.upper_shadow(), 5.0);
        assert_eq!(kline.lower_shadow(), 5.0);
    }

    #[test]
    fn test_orderbook_metrics() {
        let orderbook = OrderBook {
            symbol: "BTCUSDT".to_string(),
            bids: vec![(100.0, 10.0), (99.0, 20.0)],
            asks: vec![(101.0, 15.0), (102.0, 25.0)],
            timestamp: 0,
        };

        assert_eq!(orderbook.best_bid(), Some(100.0));
        assert_eq!(orderbook.best_ask(), Some(101.0));
        assert_eq!(orderbook.mid_price(), Some(100.5));
        assert_eq!(orderbook.spread(), Some(1.0));

        // Imbalance: (10-15)/(10+15) = -0.2
        let imb = orderbook.imbalance(1);
        assert!((imb - (-0.2)).abs() < 0.01);
    }

    #[test]
    fn test_microprice() {
        let orderbook = OrderBook {
            symbol: "BTCUSDT".to_string(),
            bids: vec![(100.0, 10.0)],
            asks: vec![(101.0, 10.0)],
            timestamp: 0,
        };

        // Equal volumes should give mid price
        assert_eq!(orderbook.microprice(), Some(100.5));

        let orderbook2 = OrderBook {
            symbol: "BTCUSDT".to_string(),
            bids: vec![(100.0, 20.0)],
            asks: vec![(101.0, 10.0)],
            timestamp: 0,
        };

        // More bid volume should pull price toward ask
        let mp = orderbook2.microprice().unwrap();
        assert!(mp > 100.5);
    }
}
