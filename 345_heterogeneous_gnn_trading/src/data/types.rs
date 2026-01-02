//! Market data types

use serde::{Deserialize, Serialize};

/// Candlestick/Kline data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Kline {
    /// Start time of the kline (timestamp in ms)
    pub timestamp: u64,
    /// Opening price
    pub open: f64,
    /// Highest price
    pub high: f64,
    /// Lowest price
    pub low: f64,
    /// Closing price
    pub close: f64,
    /// Trading volume
    pub volume: f64,
    /// Turnover (quote volume)
    pub turnover: f64,
}

impl Kline {
    /// Get the typical price (HLC average)
    pub fn typical_price(&self) -> f64 {
        (self.high + self.low + self.close) / 3.0
    }

    /// Get the range (high - low)
    pub fn range(&self) -> f64 {
        self.high - self.low
    }

    /// Check if bullish candle
    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }

    /// Get body size
    pub fn body(&self) -> f64 {
        (self.close - self.open).abs()
    }

    /// Get return
    pub fn returns(&self) -> f64 {
        if self.open > 0.0 {
            (self.close - self.open) / self.open
        } else {
            0.0
        }
    }
}

/// Order book level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookLevel {
    /// Price level
    pub price: f64,
    /// Size at this level
    pub size: f64,
}

/// Order book snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBook {
    /// Symbol
    pub symbol: String,
    /// Bid levels (highest first)
    pub bids: Vec<OrderBookLevel>,
    /// Ask levels (lowest first)
    pub asks: Vec<OrderBookLevel>,
    /// Timestamp
    pub timestamp: u64,
}

impl OrderBook {
    /// Create a new order book
    pub fn new(symbol: impl Into<String>, bids: Vec<OrderBookLevel>, asks: Vec<OrderBookLevel>) -> Self {
        Self {
            symbol: symbol.into(),
            bids,
            asks,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0),
        }
    }

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

    /// Get order book imbalance
    pub fn imbalance(&self, depth: usize) -> f64 {
        let bid_volume: f64 = self.bids.iter().take(depth).map(|l| l.size).sum();
        let ask_volume: f64 = self.asks.iter().take(depth).map(|l| l.size).sum();
        let total = bid_volume + ask_volume;

        if total > 0.0 {
            (bid_volume - ask_volume) / total
        } else {
            0.0
        }
    }

    /// Get total bid depth to a price level
    pub fn bid_depth(&self, levels: usize) -> f64 {
        self.bids.iter().take(levels).map(|l| l.price * l.size).sum()
    }

    /// Get total ask depth to a price level
    pub fn ask_depth(&self, levels: usize) -> f64 {
        self.asks.iter().take(levels).map(|l| l.price * l.size).sum()
    }
}

/// Ticker/24h statistics
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
    /// 24h volume (in base currency)
    pub volume_24h: f64,
    /// 24h turnover (in quote currency)
    pub turnover_24h: f64,
    /// 24h price change percentage
    pub price_change_24h: f64,
    /// Best bid price
    pub bid_price: f64,
    /// Best ask price
    pub ask_price: f64,
    /// Timestamp
    pub timestamp: u64,
}

impl Ticker {
    /// Get spread
    pub fn spread(&self) -> f64 {
        self.ask_price - self.bid_price
    }

    /// Get spread in basis points
    pub fn spread_bps(&self) -> f64 {
        if self.last_price > 0.0 {
            self.spread() / self.last_price * 10000.0
        } else {
            0.0
        }
    }

    /// Get 24h range
    pub fn range_24h(&self) -> f64 {
        self.high_24h - self.low_24h
    }

    /// Get 24h range as percentage
    pub fn range_24h_pct(&self) -> f64 {
        if self.low_24h > 0.0 {
            self.range_24h() / self.low_24h * 100.0
        } else {
            0.0
        }
    }
}

/// Individual trade
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    /// Trade ID
    pub id: String,
    /// Symbol
    pub symbol: String,
    /// Trade price
    pub price: f64,
    /// Trade size
    pub size: f64,
    /// Trade side (Buy/Sell)
    pub side: String,
    /// Timestamp
    pub timestamp: u64,
}

impl Trade {
    /// Check if this is a buy trade
    pub fn is_buy(&self) -> bool {
        self.side.eq_ignore_ascii_case("buy")
    }

    /// Get trade value
    pub fn value(&self) -> f64 {
        self.price * self.size
    }
}

/// Funding rate data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FundingRate {
    /// Symbol
    pub symbol: String,
    /// Current funding rate
    pub funding_rate: f64,
    /// Timestamp of funding rate
    pub funding_rate_timestamp: u64,
    /// Next funding time
    pub next_funding_time: u64,
}

impl FundingRate {
    /// Get annualized funding rate (assuming 8-hour intervals)
    pub fn annualized(&self) -> f64 {
        self.funding_rate * 3.0 * 365.0 * 100.0  // 3 times per day, 365 days
    }

    /// Check if funding is positive (longs pay shorts)
    pub fn is_positive(&self) -> bool {
        self.funding_rate > 0.0
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kline() {
        let kline = Kline {
            timestamp: 0,
            open: 100.0,
            high: 110.0,
            low: 95.0,
            close: 105.0,
            volume: 1000.0,
            turnover: 100000.0,
        };

        assert!(kline.is_bullish());
        assert_eq!(kline.range(), 15.0);
        assert_eq!(kline.body(), 5.0);
        assert!((kline.returns() - 0.05).abs() < 1e-10);
    }

    #[test]
    fn test_order_book() {
        let book = OrderBook::new(
            "BTCUSDT",
            vec![
                OrderBookLevel { price: 50000.0, size: 1.0 },
                OrderBookLevel { price: 49990.0, size: 2.0 },
            ],
            vec![
                OrderBookLevel { price: 50010.0, size: 1.5 },
                OrderBookLevel { price: 50020.0, size: 1.0 },
            ],
        );

        assert_eq!(book.best_bid(), Some(50000.0));
        assert_eq!(book.best_ask(), Some(50010.0));
        assert_eq!(book.spread(), Some(10.0));
        assert_eq!(book.mid_price(), Some(50005.0));
    }

    #[test]
    fn test_imbalance() {
        let book = OrderBook::new(
            "BTCUSDT",
            vec![
                OrderBookLevel { price: 100.0, size: 10.0 },
            ],
            vec![
                OrderBookLevel { price: 101.0, size: 5.0 },
            ],
        );

        let imbalance = book.imbalance(1);
        assert!((imbalance - 0.333).abs() < 0.01);  // (10-5)/(10+5) = 0.333
    }

    #[test]
    fn test_funding_rate() {
        let fr = FundingRate {
            symbol: "BTCUSDT".to_string(),
            funding_rate: 0.0001,
            funding_rate_timestamp: 0,
            next_funding_time: 0,
        };

        assert!(fr.is_positive());
        assert!(fr.annualized() > 10.0);  // Should be around 10.95%
    }
}
