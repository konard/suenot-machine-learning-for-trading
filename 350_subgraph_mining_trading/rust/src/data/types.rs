//! Market data types.

use serde::{Deserialize, Serialize};

/// Time frame for candles
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TimeFrame {
    #[serde(rename = "1")]
    M1,
    #[serde(rename = "3")]
    M3,
    #[serde(rename = "5")]
    M5,
    #[serde(rename = "15")]
    M15,
    #[serde(rename = "30")]
    M30,
    #[serde(rename = "60")]
    H1,
    #[serde(rename = "120")]
    H2,
    #[serde(rename = "240")]
    H4,
    #[serde(rename = "360")]
    H6,
    #[serde(rename = "720")]
    H12,
    #[serde(rename = "D")]
    D1,
    #[serde(rename = "W")]
    W1,
    #[serde(rename = "M")]
    M1Month,
}

impl TimeFrame {
    /// Convert to Bybit API interval string
    pub fn to_bybit_interval(&self) -> &'static str {
        match self {
            TimeFrame::M1 => "1",
            TimeFrame::M3 => "3",
            TimeFrame::M5 => "5",
            TimeFrame::M15 => "15",
            TimeFrame::M30 => "30",
            TimeFrame::H1 => "60",
            TimeFrame::H2 => "120",
            TimeFrame::H4 => "240",
            TimeFrame::H6 => "360",
            TimeFrame::H12 => "720",
            TimeFrame::D1 => "D",
            TimeFrame::W1 => "W",
            TimeFrame::M1Month => "M",
        }
    }

    /// Get duration in seconds
    pub fn seconds(&self) -> i64 {
        match self {
            TimeFrame::M1 => 60,
            TimeFrame::M3 => 180,
            TimeFrame::M5 => 300,
            TimeFrame::M15 => 900,
            TimeFrame::M30 => 1800,
            TimeFrame::H1 => 3600,
            TimeFrame::H2 => 7200,
            TimeFrame::H4 => 14400,
            TimeFrame::H6 => 21600,
            TimeFrame::H12 => 43200,
            TimeFrame::D1 => 86400,
            TimeFrame::W1 => 604800,
            TimeFrame::M1Month => 2592000,
        }
    }

    /// Parse from string
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "1m" | "1" => Some(TimeFrame::M1),
            "3m" | "3" => Some(TimeFrame::M3),
            "5m" | "5" => Some(TimeFrame::M5),
            "15m" | "15" => Some(TimeFrame::M15),
            "30m" | "30" => Some(TimeFrame::M30),
            "1h" | "60" => Some(TimeFrame::H1),
            "2h" | "120" => Some(TimeFrame::H2),
            "4h" | "240" => Some(TimeFrame::H4),
            "6h" | "360" => Some(TimeFrame::H6),
            "12h" | "720" => Some(TimeFrame::H12),
            "1d" | "d" => Some(TimeFrame::D1),
            "1w" | "w" => Some(TimeFrame::W1),
            "1m" | "m" => Some(TimeFrame::M1Month),
            _ => None,
        }
    }
}

/// OHLCV candle data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candle {
    /// Timestamp in seconds
    pub timestamp: i64,
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

impl Candle {
    /// Create a new candle
    pub fn new(
        timestamp: i64,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
    ) -> Self {
        Self {
            timestamp,
            open,
            high,
            low,
            close,
            volume,
            turnover: 0.0,
        }
    }

    /// Get typical price (HLC average)
    pub fn typical_price(&self) -> f64 {
        (self.high + self.low + self.close) / 3.0
    }

    /// Get candle range
    pub fn range(&self) -> f64 {
        self.high - self.low
    }

    /// Get candle body
    pub fn body(&self) -> f64 {
        (self.close - self.open).abs()
    }

    /// Check if candle is bullish
    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }

    /// Check if candle is bearish
    pub fn is_bearish(&self) -> bool {
        self.close < self.open
    }

    /// Get return from open to close
    pub fn return_pct(&self) -> f64 {
        if self.open == 0.0 {
            return 0.0;
        }
        (self.close - self.open) / self.open
    }
}

/// Ticker data (current price info)
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
    /// 24h price change percentage
    pub price_change_24h: f64,
    /// Bid price
    pub bid_price: f64,
    /// Ask price
    pub ask_price: f64,
    /// Timestamp
    pub timestamp: i64,
}

impl Ticker {
    /// Get spread
    pub fn spread(&self) -> f64 {
        self.ask_price - self.bid_price
    }

    /// Get spread percentage
    pub fn spread_pct(&self) -> f64 {
        if self.bid_price == 0.0 {
            return 0.0;
        }
        self.spread() / self.bid_price
    }

    /// Get mid price
    pub fn mid_price(&self) -> f64 {
        (self.bid_price + self.ask_price) / 2.0
    }
}

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
    /// Bid levels (sorted by price descending)
    pub bids: Vec<OrderBookLevel>,
    /// Ask levels (sorted by price ascending)
    pub asks: Vec<OrderBookLevel>,
    /// Timestamp
    pub timestamp: i64,
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

    /// Get total bid volume up to specified depth
    pub fn bid_volume(&self, depth: usize) -> f64 {
        self.bids.iter().take(depth).map(|l| l.quantity).sum()
    }

    /// Get total ask volume up to specified depth
    pub fn ask_volume(&self, depth: usize) -> f64 {
        self.asks.iter().take(depth).map(|l| l.quantity).sum()
    }

    /// Get order imbalance
    pub fn imbalance(&self, depth: usize) -> f64 {
        let bid_vol = self.bid_volume(depth);
        let ask_vol = self.ask_volume(depth);
        let total = bid_vol + ask_vol;

        if total == 0.0 {
            return 0.0;
        }

        (bid_vol - ask_vol) / total
    }
}

/// Aggregated market data for multiple symbols
#[derive(Debug, Clone, Default)]
pub struct MarketData {
    /// Candles by symbol
    pub candles: std::collections::HashMap<String, Vec<Candle>>,
    /// Latest tickers by symbol
    pub tickers: std::collections::HashMap<String, Ticker>,
    /// Order books by symbol
    pub order_books: std::collections::HashMap<String, OrderBook>,
    /// Last update timestamp
    pub last_update: i64,
}

impl MarketData {
    /// Create new empty market data
    pub fn new() -> Self {
        Self::default()
    }

    /// Add candles for a symbol
    pub fn add_candles(&mut self, symbol: impl Into<String>, candles: Vec<Candle>) {
        self.candles.insert(symbol.into(), candles);
        self.last_update = chrono::Utc::now().timestamp();
    }

    /// Add ticker for a symbol
    pub fn add_ticker(&mut self, ticker: Ticker) {
        let symbol = ticker.symbol.clone();
        self.tickers.insert(symbol, ticker);
        self.last_update = chrono::Utc::now().timestamp();
    }

    /// Get latest prices
    pub fn latest_prices(&self) -> std::collections::HashMap<String, f64> {
        self.tickers
            .iter()
            .map(|(s, t)| (s.clone(), t.last_price))
            .collect()
    }

    /// Get symbols
    pub fn symbols(&self) -> Vec<&String> {
        self.candles.keys().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_candle() {
        let candle = Candle::new(1609459200, 100.0, 110.0, 95.0, 105.0, 1000.0);

        assert!(candle.is_bullish());
        assert_eq!(candle.range(), 15.0);
        assert_eq!(candle.body(), 5.0);
        assert!((candle.return_pct() - 0.05).abs() < 0.001);
    }

    #[test]
    fn test_order_book() {
        let ob = OrderBook {
            symbol: "BTCUSDT".to_string(),
            bids: vec![
                OrderBookLevel { price: 50000.0, quantity: 1.0 },
                OrderBookLevel { price: 49990.0, quantity: 2.0 },
            ],
            asks: vec![
                OrderBookLevel { price: 50010.0, quantity: 1.5 },
                OrderBookLevel { price: 50020.0, quantity: 2.5 },
            ],
            timestamp: 0,
        };

        assert_eq!(ob.best_bid(), Some(50000.0));
        assert_eq!(ob.best_ask(), Some(50010.0));
        assert_eq!(ob.spread(), Some(10.0));
        assert_eq!(ob.mid_price(), Some(50005.0));
    }

    #[test]
    fn test_timeframe() {
        assert_eq!(TimeFrame::H1.seconds(), 3600);
        assert_eq!(TimeFrame::D1.to_bybit_interval(), "D");
    }
}
