//! API data types

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Timeframe for candlestick data
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TimeFrame {
    #[serde(rename = "1")]
    Min1,
    #[serde(rename = "3")]
    Min3,
    #[serde(rename = "5")]
    Min5,
    #[serde(rename = "15")]
    Min15,
    #[serde(rename = "30")]
    Min30,
    #[serde(rename = "60")]
    Hour1,
    #[serde(rename = "120")]
    Hour2,
    #[serde(rename = "240")]
    Hour4,
    #[serde(rename = "360")]
    Hour6,
    #[serde(rename = "720")]
    Hour12,
    #[serde(rename = "D")]
    Day1,
    #[serde(rename = "W")]
    Week1,
    #[serde(rename = "M")]
    Month1,
}

impl TimeFrame {
    pub fn as_str(&self) -> &'static str {
        match self {
            TimeFrame::Min1 => "1",
            TimeFrame::Min3 => "3",
            TimeFrame::Min5 => "5",
            TimeFrame::Min15 => "15",
            TimeFrame::Min30 => "30",
            TimeFrame::Hour1 => "60",
            TimeFrame::Hour2 => "120",
            TimeFrame::Hour4 => "240",
            TimeFrame::Hour6 => "360",
            TimeFrame::Hour12 => "720",
            TimeFrame::Day1 => "D",
            TimeFrame::Week1 => "W",
            TimeFrame::Month1 => "M",
        }
    }

    /// Get duration in milliseconds
    pub fn duration_ms(&self) -> i64 {
        match self {
            TimeFrame::Min1 => 60_000,
            TimeFrame::Min3 => 180_000,
            TimeFrame::Min5 => 300_000,
            TimeFrame::Min15 => 900_000,
            TimeFrame::Min30 => 1_800_000,
            TimeFrame::Hour1 => 3_600_000,
            TimeFrame::Hour2 => 7_200_000,
            TimeFrame::Hour4 => 14_400_000,
            TimeFrame::Hour6 => 21_600_000,
            TimeFrame::Hour12 => 43_200_000,
            TimeFrame::Day1 => 86_400_000,
            TimeFrame::Week1 => 604_800_000,
            TimeFrame::Month1 => 2_592_000_000,
        }
    }
}

/// OHLCV candlestick data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candle {
    /// Opening timestamp
    pub timestamp: DateTime<Utc>,
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
        timestamp: DateTime<Utc>,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
        turnover: f64,
    ) -> Self {
        Self {
            timestamp,
            open,
            high,
            low,
            close,
            volume,
            turnover,
        }
    }

    /// Calculate typical price (HLC average)
    pub fn typical_price(&self) -> f64 {
        (self.high + self.low + self.close) / 3.0
    }

    /// Calculate price range
    pub fn range(&self) -> f64 {
        self.high - self.low
    }

    /// Calculate return from open to close
    pub fn return_pct(&self) -> f64 {
        (self.close - self.open) / self.open
    }

    /// Check if bullish candle
    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }

    /// Check if bearish candle
    pub fn is_bearish(&self) -> bool {
        self.close < self.open
    }
}

/// Collection of market data with computed features
#[derive(Debug, Clone, Default)]
pub struct MarketData {
    pub candles: Vec<Candle>,
    pub symbol: String,
    pub timeframe: Option<TimeFrame>,
}

impl MarketData {
    /// Create new MarketData
    pub fn new(symbol: impl Into<String>, candles: Vec<Candle>) -> Self {
        Self {
            candles,
            symbol: symbol.into(),
            timeframe: None,
        }
    }

    /// Get the latest candle
    pub fn latest(&self) -> Option<&Candle> {
        self.candles.last()
    }

    /// Get latest price
    pub fn latest_price(&self) -> Option<f64> {
        self.latest().map(|c| c.close)
    }

    /// Get closing prices
    pub fn closes(&self) -> Vec<f64> {
        self.candles.iter().map(|c| c.close).collect()
    }

    /// Get volumes
    pub fn volumes(&self) -> Vec<f64> {
        self.candles.iter().map(|c| c.volume).collect()
    }

    /// Calculate simple moving average
    pub fn sma(&self, period: usize) -> Vec<f64> {
        if self.candles.len() < period {
            return vec![];
        }

        let closes = self.closes();
        closes
            .windows(period)
            .map(|w| w.iter().sum::<f64>() / period as f64)
            .collect()
    }

    /// Calculate exponential moving average
    pub fn ema(&self, period: usize) -> Vec<f64> {
        if self.candles.is_empty() {
            return vec![];
        }

        let closes = self.closes();
        let multiplier = 2.0 / (period as f64 + 1.0);
        let mut ema = vec![closes[0]];

        for close in closes.iter().skip(1) {
            let prev_ema = *ema.last().unwrap();
            ema.push((close - prev_ema) * multiplier + prev_ema);
        }

        ema
    }

    /// Calculate returns
    pub fn returns(&self) -> Vec<f64> {
        let closes = self.closes();
        closes
            .windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect()
    }

    /// Calculate volatility (standard deviation of returns)
    pub fn volatility(&self, period: usize) -> Vec<f64> {
        let returns = self.returns();
        if returns.len() < period {
            return vec![];
        }

        returns
            .windows(period)
            .map(|w| {
                let mean = w.iter().sum::<f64>() / period as f64;
                let variance = w.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / period as f64;
                variance.sqrt()
            })
            .collect()
    }

    /// Calculate momentum (rate of change)
    pub fn momentum(&self, period: usize) -> Vec<f64> {
        let closes = self.closes();
        if closes.len() <= period {
            return vec![];
        }

        closes
            .iter()
            .skip(period)
            .zip(closes.iter())
            .map(|(current, past)| (current - past) / past)
            .collect()
    }
}

/// Order book snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookSnapshot {
    pub timestamp: DateTime<Utc>,
    pub symbol: String,
    pub bids: Vec<(f64, f64)>, // (price, quantity)
    pub asks: Vec<(f64, f64)>, // (price, quantity)
}

impl OrderBookSnapshot {
    /// Get best bid price
    pub fn best_bid(&self) -> Option<f64> {
        self.bids.first().map(|(p, _)| *p)
    }

    /// Get best ask price
    pub fn best_ask(&self) -> Option<f64> {
        self.asks.first().map(|(p, _)| *p)
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

    /// Get spread as percentage of mid price
    pub fn spread_pct(&self) -> Option<f64> {
        match (self.spread(), self.mid_price()) {
            (Some(spread), Some(mid)) if mid > 0.0 => Some(spread / mid * 100.0),
            _ => None,
        }
    }

    /// Get total bid depth (sum of bid quantities)
    pub fn bid_depth(&self) -> f64 {
        self.bids.iter().map(|(_, q)| q).sum()
    }

    /// Get total ask depth (sum of ask quantities)
    pub fn ask_depth(&self) -> f64 {
        self.asks.iter().map(|(_, q)| q).sum()
    }

    /// Get order book imbalance (-1 to 1, positive = more bids)
    pub fn imbalance(&self) -> f64 {
        let bid_depth = self.bid_depth();
        let ask_depth = self.ask_depth();
        let total = bid_depth + ask_depth;

        if total > 0.0 {
            (bid_depth - ask_depth) / total
        } else {
            0.0
        }
    }
}

/// Ticker information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ticker {
    pub symbol: String,
    pub last_price: f64,
    pub bid_price: f64,
    pub ask_price: f64,
    pub volume_24h: f64,
    pub turnover_24h: f64,
    pub high_24h: f64,
    pub low_24h: f64,
    pub price_change_24h: f64,
    pub price_change_pct_24h: f64,
}

// Bybit API response types
#[derive(Debug, Deserialize)]
pub(crate) struct BybitResponse<T> {
    #[serde(rename = "retCode")]
    pub ret_code: i32,
    #[serde(rename = "retMsg")]
    pub ret_msg: String,
    pub result: T,
}

#[derive(Debug, Deserialize)]
pub(crate) struct KlineResult {
    pub symbol: String,
    pub category: String,
    pub list: Vec<Vec<String>>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct OrderBookResult {
    pub s: String, // symbol
    pub b: Vec<Vec<String>>, // bids
    pub a: Vec<Vec<String>>, // asks
    pub ts: i64, // timestamp
}

#[derive(Debug, Deserialize)]
pub(crate) struct TickerResult {
    pub list: Vec<TickerItem>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct TickerItem {
    pub symbol: String,
    #[serde(rename = "lastPrice")]
    pub last_price: String,
    #[serde(rename = "bid1Price")]
    pub bid1_price: String,
    #[serde(rename = "ask1Price")]
    pub ask1_price: String,
    #[serde(rename = "volume24h")]
    pub volume_24h: String,
    #[serde(rename = "turnover24h")]
    pub turnover_24h: String,
    #[serde(rename = "highPrice24h")]
    pub high_price_24h: String,
    #[serde(rename = "lowPrice24h")]
    pub low_price_24h: String,
    #[serde(rename = "price24hPcnt")]
    pub price_24h_pcnt: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_candle_calculations() {
        let candle = Candle::new(
            Utc::now(),
            100.0, // open
            110.0, // high
            95.0,  // low
            105.0, // close
            1000.0, // volume
            100000.0, // turnover
        );

        assert!((candle.typical_price() - 103.333).abs() < 0.01);
        assert!((candle.range() - 15.0).abs() < 0.001);
        assert!((candle.return_pct() - 0.05).abs() < 0.001);
        assert!(candle.is_bullish());
        assert!(!candle.is_bearish());
    }

    #[test]
    fn test_orderbook_snapshot() {
        let snapshot = OrderBookSnapshot {
            timestamp: Utc::now(),
            symbol: "BTCUSDT".to_string(),
            bids: vec![(50000.0, 1.0), (49990.0, 2.0)],
            asks: vec![(50010.0, 1.5), (50020.0, 2.5)],
        };

        assert_eq!(snapshot.best_bid(), Some(50000.0));
        assert_eq!(snapshot.best_ask(), Some(50010.0));
        assert_eq!(snapshot.mid_price(), Some(50005.0));
        assert_eq!(snapshot.spread(), Some(10.0));
        assert_eq!(snapshot.bid_depth(), 3.0);
        assert_eq!(snapshot.ask_depth(), 4.0);
    }
}
