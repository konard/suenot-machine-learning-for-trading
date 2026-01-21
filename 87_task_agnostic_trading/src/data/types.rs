//! Data types for market information

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// OHLCV candlestick data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Kline {
    /// Start timestamp
    pub timestamp: DateTime<Utc>,
    /// Opening price
    pub open: f64,
    /// High price
    pub high: f64,
    /// Low price
    pub low: f64,
    /// Close price
    pub close: f64,
    /// Trading volume
    pub volume: f64,
    /// Turnover (quote volume)
    pub turnover: f64,
}

impl Kline {
    /// Calculate the candle body size
    pub fn body_size(&self) -> f64 {
        (self.close - self.open).abs()
    }

    /// Calculate the full range (high - low)
    pub fn range(&self) -> f64 {
        self.high - self.low
    }

    /// Check if bullish candle
    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }

    /// Calculate return
    pub fn return_pct(&self) -> f64 {
        if self.open > 0.0 {
            (self.close - self.open) / self.open
        } else {
            0.0
        }
    }

    /// Calculate upper wick size
    pub fn upper_wick(&self) -> f64 {
        self.high - self.open.max(self.close)
    }

    /// Calculate lower wick size
    pub fn lower_wick(&self) -> f64 {
        self.open.min(self.close) - self.low
    }
}

/// Individual trade data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    /// Trade timestamp
    pub timestamp: DateTime<Utc>,
    /// Trade price
    pub price: f64,
    /// Trade quantity
    pub quantity: f64,
    /// Trade side (true = buy, false = sell)
    pub is_buyer_maker: bool,
}

/// Order book level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookLevel {
    /// Price level
    pub price: f64,
    /// Quantity at this level
    pub quantity: f64,
}

/// Order book snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBook {
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

    /// Calculate spread as percentage
    pub fn spread_pct(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) if bid > 0.0 => Some((ask - bid) / bid),
            _ => None,
        }
    }

    /// Calculate order book imbalance
    pub fn imbalance(&self, depth: usize) -> f64 {
        let bid_vol: f64 = self.bids.iter().take(depth).map(|l| l.quantity).sum();
        let ask_vol: f64 = self.asks.iter().take(depth).map(|l| l.quantity).sum();

        if bid_vol + ask_vol > 0.0 {
            (bid_vol - ask_vol) / (bid_vol + ask_vol)
        } else {
            0.0
        }
    }

    /// Calculate volume-weighted average price for bids
    pub fn vwap_bid(&self, depth: usize) -> Option<f64> {
        let levels: Vec<_> = self.bids.iter().take(depth).collect();
        if levels.is_empty() {
            return None;
        }

        let total_vol: f64 = levels.iter().map(|l| l.quantity).sum();
        if total_vol == 0.0 {
            return None;
        }

        let weighted_price: f64 = levels.iter().map(|l| l.price * l.quantity).sum();
        Some(weighted_price / total_vol)
    }

    /// Calculate volume-weighted average price for asks
    pub fn vwap_ask(&self, depth: usize) -> Option<f64> {
        let levels: Vec<_> = self.asks.iter().take(depth).collect();
        if levels.is_empty() {
            return None;
        }

        let total_vol: f64 = levels.iter().map(|l| l.quantity).sum();
        if total_vol == 0.0 {
            return None;
        }

        let weighted_price: f64 = levels.iter().map(|l| l.price * l.quantity).sum();
        Some(weighted_price / total_vol)
    }
}

/// Funding rate data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FundingRate {
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Symbol
    pub symbol: String,
    /// Funding rate value
    pub funding_rate: f64,
    /// Funding interval in hours
    pub funding_interval: u32,
}

/// Ticker data
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
    /// Price change percentage
    pub price_change_pct: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kline_calculations() {
        let kline = Kline {
            timestamp: Utc::now(),
            open: 100.0,
            high: 110.0,
            low: 95.0,
            close: 105.0,
            volume: 1000.0,
            turnover: 100000.0,
        };

        assert_eq!(kline.body_size(), 5.0);
        assert_eq!(kline.range(), 15.0);
        assert!(kline.is_bullish());
        assert!((kline.return_pct() - 0.05).abs() < 1e-10);
        assert_eq!(kline.upper_wick(), 5.0);  // 110 - 105
        assert_eq!(kline.lower_wick(), 5.0);  // 100 - 95
    }

    #[test]
    fn test_order_book() {
        let ob = OrderBook {
            timestamp: Utc::now(),
            bids: vec![
                OrderBookLevel { price: 99.0, quantity: 10.0 },
                OrderBookLevel { price: 98.0, quantity: 20.0 },
            ],
            asks: vec![
                OrderBookLevel { price: 101.0, quantity: 15.0 },
                OrderBookLevel { price: 102.0, quantity: 25.0 },
            ],
        };

        assert_eq!(ob.best_bid(), Some(99.0));
        assert_eq!(ob.best_ask(), Some(101.0));
        assert_eq!(ob.mid_price(), Some(100.0));
        assert_eq!(ob.spread(), Some(2.0));
    }

    #[test]
    fn test_order_book_imbalance() {
        let ob = OrderBook {
            timestamp: Utc::now(),
            bids: vec![
                OrderBookLevel { price: 99.0, quantity: 30.0 },
            ],
            asks: vec![
                OrderBookLevel { price: 101.0, quantity: 10.0 },
            ],
        };

        // (30 - 10) / (30 + 10) = 0.5
        assert!((ob.imbalance(1) - 0.5).abs() < 1e-10);
    }
}
