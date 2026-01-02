//! Bybit exchange API client
//!
//! Provides methods to fetch market data from Bybit:
//! - Kline (candlestick) data
//! - Order book data
//! - Recent trades

use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use reqwest::Client;
use serde::{Deserialize, Serialize};

/// Bybit API base URL
const BASE_URL: &str = "https://api.bybit.com";

/// Candlestick data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candle {
    pub timestamp: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub turnover: f64,
}

impl Candle {
    /// Calculate typical price (HLC average)
    pub fn typical_price(&self) -> f64 {
        (self.high + self.low + self.close) / 3.0
    }

    /// Calculate return from open to close
    pub fn return_pct(&self) -> f64 {
        (self.close - self.open) / self.open
    }

    /// Check if bullish
    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }

    /// Get body size as percentage
    pub fn body_pct(&self) -> f64 {
        (self.close - self.open).abs() / self.open
    }
}

/// Order book level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookLevel {
    pub price: f64,
    pub quantity: f64,
}

/// Order book data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBook {
    pub symbol: String,
    pub timestamp: u64,
    pub bids: Vec<OrderBookLevel>,
    pub asks: Vec<OrderBookLevel>,
}

impl OrderBook {
    /// Calculate mid price
    pub fn mid_price(&self) -> Option<f64> {
        let best_bid = self.bids.first()?.price;
        let best_ask = self.asks.first()?.price;
        Some((best_bid + best_ask) / 2.0)
    }

    /// Calculate spread
    pub fn spread(&self) -> Option<f64> {
        let best_bid = self.bids.first()?.price;
        let best_ask = self.asks.first()?.price;
        Some(best_ask - best_bid)
    }

    /// Calculate spread as percentage
    pub fn spread_pct(&self) -> Option<f64> {
        let mid = self.mid_price()?;
        let spread = self.spread()?;
        Some(spread / mid)
    }

    /// Calculate order book imbalance
    pub fn imbalance(&self, depth: usize) -> f64 {
        let bid_vol: f64 = self.bids.iter().take(depth).map(|l| l.quantity).sum();
        let ask_vol: f64 = self.asks.iter().take(depth).map(|l| l.quantity).sum();
        (bid_vol - ask_vol) / (bid_vol + ask_vol)
    }
}

/// Trade data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    pub id: String,
    pub symbol: String,
    pub price: f64,
    pub quantity: f64,
    pub side: TradeSide,
    pub timestamp: i64,
}

/// Trade side
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TradeSide {
    Buy,
    Sell,
}

/// Kline intervals
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Interval {
    Min1,
    Min3,
    Min5,
    Min15,
    Min30,
    Hour1,
    Hour2,
    Hour4,
    Hour6,
    Hour12,
    Day1,
    Week1,
    Month1,
}

impl Interval {
    /// Convert to API string
    pub fn as_str(&self) -> &'static str {
        match self {
            Interval::Min1 => "1",
            Interval::Min3 => "3",
            Interval::Min5 => "5",
            Interval::Min15 => "15",
            Interval::Min30 => "30",
            Interval::Hour1 => "60",
            Interval::Hour2 => "120",
            Interval::Hour4 => "240",
            Interval::Hour6 => "360",
            Interval::Hour12 => "720",
            Interval::Day1 => "D",
            Interval::Week1 => "W",
            Interval::Month1 => "M",
        }
    }

    /// Parse from string
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "1" | "1m" | "1min" => Some(Interval::Min1),
            "3" | "3m" | "3min" => Some(Interval::Min3),
            "5" | "5m" | "5min" => Some(Interval::Min5),
            "15" | "15m" | "15min" => Some(Interval::Min15),
            "30" | "30m" | "30min" => Some(Interval::Min30),
            "60" | "1h" | "1hour" => Some(Interval::Hour1),
            "120" | "2h" | "2hour" => Some(Interval::Hour2),
            "240" | "4h" | "4hour" => Some(Interval::Hour4),
            "360" | "6h" | "6hour" => Some(Interval::Hour6),
            "720" | "12h" | "12hour" => Some(Interval::Hour12),
            "d" | "1d" | "day" => Some(Interval::Day1),
            "w" | "1w" | "week" => Some(Interval::Week1),
            "m" | "1M" | "month" => Some(Interval::Month1),
            _ => None,
        }
    }

    /// Get interval in milliseconds
    pub fn to_millis(&self) -> i64 {
        match self {
            Interval::Min1 => 60_000,
            Interval::Min3 => 3 * 60_000,
            Interval::Min5 => 5 * 60_000,
            Interval::Min15 => 15 * 60_000,
            Interval::Min30 => 30 * 60_000,
            Interval::Hour1 => 60 * 60_000,
            Interval::Hour2 => 2 * 60 * 60_000,
            Interval::Hour4 => 4 * 60 * 60_000,
            Interval::Hour6 => 6 * 60 * 60_000,
            Interval::Hour12 => 12 * 60 * 60_000,
            Interval::Day1 => 24 * 60 * 60_000,
            Interval::Week1 => 7 * 24 * 60 * 60_000,
            Interval::Month1 => 30 * 24 * 60 * 60_000,
        }
    }
}

/// Bybit API response wrapper
#[derive(Debug, Deserialize)]
struct BybitResponse<T> {
    #[serde(rename = "retCode")]
    ret_code: i32,
    #[serde(rename = "retMsg")]
    ret_msg: String,
    result: T,
}

/// Kline result from API
#[derive(Debug, Deserialize)]
struct KlineResult {
    #[allow(dead_code)]
    symbol: String,
    #[allow(dead_code)]
    category: String,
    list: Vec<Vec<String>>,
}

/// Order book result from API
#[derive(Debug, Deserialize)]
struct OrderBookResult {
    s: String,
    b: Vec<Vec<String>>,
    a: Vec<Vec<String>>,
    ts: u64,
}

/// Trade result from API
#[derive(Debug, Deserialize)]
struct TradeResult {
    #[allow(dead_code)]
    category: String,
    list: Vec<TradeItem>,
}

#[derive(Debug, Deserialize)]
struct TradeItem {
    #[serde(rename = "execId")]
    exec_id: String,
    symbol: String,
    price: String,
    size: String,
    side: String,
    time: String,
}

/// Bybit API client
#[derive(Debug, Clone)]
pub struct BybitClient {
    client: Client,
    base_url: String,
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

impl BybitClient {
    /// Create a new client with default settings
    pub fn new() -> Self {
        Self {
            client: Client::new(),
            base_url: BASE_URL.to_string(),
        }
    }

    /// Create a testnet client
    pub fn testnet() -> Self {
        Self {
            client: Client::new(),
            base_url: "https://api-testnet.bybit.com".to_string(),
        }
    }

    /// Fetch kline (candlestick) data
    ///
    /// # Arguments
    /// * `symbol` - Trading pair (e.g., "BTCUSDT")
    /// * `interval` - Kline interval (e.g., "1h")
    /// * `limit` - Number of candles (max 1000)
    pub async fn get_klines(&self, symbol: &str, interval: &str, limit: usize) -> Result<Vec<Candle>> {
        let interval_enum = Interval::from_str(interval)
            .ok_or_else(|| anyhow!("Invalid interval: {}", interval))?;

        let url = format!(
            "{}/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
            self.base_url,
            symbol.to_uppercase(),
            interval_enum.as_str(),
            limit.min(1000)
        );

        let response: BybitResponse<KlineResult> = self.client
            .get(&url)
            .send()
            .await?
            .json()
            .await?;

        if response.ret_code != 0 {
            return Err(anyhow!(
                "API error {}: {}",
                response.ret_code,
                response.ret_msg
            ));
        }

        let mut candles: Vec<Candle> = response
            .result
            .list
            .iter()
            .filter_map(|item| {
                if item.len() >= 7 {
                    Some(Candle {
                        timestamp: item[0].parse().ok()?,
                        open: item[1].parse().ok()?,
                        high: item[2].parse().ok()?,
                        low: item[3].parse().ok()?,
                        close: item[4].parse().ok()?,
                        volume: item[5].parse().ok()?,
                        turnover: item[6].parse().ok()?,
                    })
                } else {
                    None
                }
            })
            .collect();

        // Sort by timestamp (oldest first)
        candles.sort_by_key(|c| c.timestamp);

        Ok(candles)
    }

    /// Fetch klines with time range
    pub async fn get_klines_range(
        &self,
        symbol: &str,
        interval: &str,
        start_time: DateTime<Utc>,
        end_time: DateTime<Utc>,
    ) -> Result<Vec<Candle>> {
        let interval_enum = Interval::from_str(interval)
            .ok_or_else(|| anyhow!("Invalid interval: {}", interval))?;

        let url = format!(
            "{}/v5/market/kline?category=spot&symbol={}&interval={}&start={}&end={}&limit=1000",
            self.base_url,
            symbol.to_uppercase(),
            interval_enum.as_str(),
            start_time.timestamp_millis(),
            end_time.timestamp_millis()
        );

        let response: BybitResponse<KlineResult> = self.client
            .get(&url)
            .send()
            .await?
            .json()
            .await?;

        if response.ret_code != 0 {
            return Err(anyhow!(
                "API error {}: {}",
                response.ret_code,
                response.ret_msg
            ));
        }

        let mut candles: Vec<Candle> = response
            .result
            .list
            .iter()
            .filter_map(|item| {
                if item.len() >= 7 {
                    Some(Candle {
                        timestamp: item[0].parse().ok()?,
                        open: item[1].parse().ok()?,
                        high: item[2].parse().ok()?,
                        low: item[3].parse().ok()?,
                        close: item[4].parse().ok()?,
                        volume: item[5].parse().ok()?,
                        turnover: item[6].parse().ok()?,
                    })
                } else {
                    None
                }
            })
            .collect();

        candles.sort_by_key(|c| c.timestamp);
        Ok(candles)
    }

    /// Fetch order book
    pub async fn get_orderbook(&self, symbol: &str, limit: usize) -> Result<OrderBook> {
        let url = format!(
            "{}/v5/market/orderbook?category=spot&symbol={}&limit={}",
            self.base_url,
            symbol.to_uppercase(),
            limit.min(200)
        );

        let response: BybitResponse<OrderBookResult> = self.client
            .get(&url)
            .send()
            .await?
            .json()
            .await?;

        if response.ret_code != 0 {
            return Err(anyhow!(
                "API error {}: {}",
                response.ret_code,
                response.ret_msg
            ));
        }

        let parse_levels = |levels: &[Vec<String>]| -> Vec<OrderBookLevel> {
            levels
                .iter()
                .filter_map(|level| {
                    if level.len() >= 2 {
                        Some(OrderBookLevel {
                            price: level[0].parse().ok()?,
                            quantity: level[1].parse().ok()?,
                        })
                    } else {
                        None
                    }
                })
                .collect()
        };

        Ok(OrderBook {
            symbol: response.result.s,
            timestamp: response.result.ts,
            bids: parse_levels(&response.result.b),
            asks: parse_levels(&response.result.a),
        })
    }

    /// Fetch recent trades
    pub async fn get_recent_trades(&self, symbol: &str, limit: usize) -> Result<Vec<Trade>> {
        let url = format!(
            "{}/v5/market/recent-trade?category=spot&symbol={}&limit={}",
            self.base_url,
            symbol.to_uppercase(),
            limit.min(1000)
        );

        let response: BybitResponse<TradeResult> = self.client
            .get(&url)
            .send()
            .await?
            .json()
            .await?;

        if response.ret_code != 0 {
            return Err(anyhow!(
                "API error {}: {}",
                response.ret_code,
                response.ret_msg
            ));
        }

        let trades: Vec<Trade> = response
            .result
            .list
            .iter()
            .filter_map(|item| {
                Some(Trade {
                    id: item.exec_id.clone(),
                    symbol: item.symbol.clone(),
                    price: item.price.parse().ok()?,
                    quantity: item.size.parse().ok()?,
                    side: if item.side == "Buy" {
                        TradeSide::Buy
                    } else {
                        TradeSide::Sell
                    },
                    timestamp: item.time.parse().ok()?,
                })
            })
            .collect();

        Ok(trades)
    }

    /// Fetch multiple symbols concurrently
    pub async fn get_klines_multi(
        &self,
        symbols: &[&str],
        interval: &str,
        limit: usize,
    ) -> Result<Vec<(String, Vec<Candle>)>> {
        use futures::future::join_all;

        let futures: Vec<_> = symbols
            .iter()
            .map(|symbol| async move {
                let candles = self.get_klines(symbol, interval, limit).await?;
                Ok::<_, anyhow::Error>((symbol.to_string(), candles))
            })
            .collect();

        let results = join_all(futures).await;

        results.into_iter().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interval_parsing() {
        assert_eq!(Interval::from_str("1h"), Some(Interval::Hour1));
        assert_eq!(Interval::from_str("4h"), Some(Interval::Hour4));
        assert_eq!(Interval::from_str("1d"), Some(Interval::Day1));
        assert_eq!(Interval::from_str("invalid"), None);
    }

    #[test]
    fn test_candle_methods() {
        let candle = Candle {
            timestamp: 1000,
            open: 100.0,
            high: 110.0,
            low: 95.0,
            close: 105.0,
            volume: 1000.0,
            turnover: 100000.0,
        };

        assert!(candle.is_bullish());
        assert!((candle.return_pct() - 0.05).abs() < 0.001);
        assert!((candle.typical_price() - 103.333).abs() < 0.1);
    }
}
