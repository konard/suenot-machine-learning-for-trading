//! Bybit Exchange API Client
//!
//! Fetches cryptocurrency market data from Bybit exchange.

use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use crate::{Error, Result};

/// Bybit API base URL
const BYBIT_API_URL: &str = "https://api.bybit.com";

/// Candle/Kline data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candle {
    /// Start time of the candle
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
    /// Calculate price return
    pub fn return_pct(&self) -> f64 {
        if self.open > 0.0 {
            (self.close - self.open) / self.open
        } else {
            0.0
        }
    }

    /// Calculate price range
    pub fn range(&self) -> f64 {
        self.high - self.low
    }

    /// Calculate body size (absolute)
    pub fn body_size(&self) -> f64 {
        (self.close - self.open).abs()
    }

    /// Check if bullish candle
    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }

    /// Get typical price
    pub fn typical_price(&self) -> f64 {
        (self.high + self.low + self.close) / 3.0
    }
}

/// Single trade
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    /// Trade ID
    pub id: String,
    /// Trade price
    pub price: f64,
    /// Trade size
    pub size: f64,
    /// Trade side (Buy/Sell)
    pub side: String,
    /// Timestamp
    pub timestamp: i64,
}

impl Trade {
    /// Check if buy order
    pub fn is_buy(&self) -> bool {
        self.side.to_lowercase() == "buy"
    }

    /// Get signed size (positive for buys, negative for sells)
    pub fn signed_size(&self) -> f64 {
        if self.is_buy() {
            self.size
        } else {
            -self.size
        }
    }
}

/// Order book level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookLevel {
    /// Price
    pub price: f64,
    /// Size
    pub size: f64,
}

/// Order book snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBook {
    /// Symbol
    pub symbol: String,
    /// Timestamp
    pub timestamp: i64,
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

    /// Calculate spread
    pub fn spread(&self) -> Option<f64> {
        match (self.best_ask(), self.best_bid()) {
            (Some(ask), Some(bid)) => Some(ask - bid),
            _ => None,
        }
    }

    /// Calculate mid price
    pub fn mid_price(&self) -> Option<f64> {
        match (self.best_ask(), self.best_bid()) {
            (Some(ask), Some(bid)) => Some((ask + bid) / 2.0),
            _ => None,
        }
    }

    /// Calculate order book imbalance
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

    /// Get volume at price level
    pub fn volume_at_price(&self, price: f64, tolerance: f64) -> f64 {
        let bid_vol: f64 = self.bids
            .iter()
            .filter(|l| (l.price - price).abs() <= tolerance)
            .map(|l| l.size)
            .sum();

        let ask_vol: f64 = self.asks
            .iter()
            .filter(|l| (l.price - price).abs() <= tolerance)
            .map(|l| l.size)
            .sum();

        bid_vol + ask_vol
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

/// Kline response result
#[derive(Debug, Deserialize)]
struct KlineResult {
    list: Vec<Vec<String>>,
}

/// Order book response result
#[derive(Debug, Deserialize)]
struct OrderBookResult {
    s: String,
    b: Vec<Vec<String>>,
    a: Vec<Vec<String>>,
    ts: i64,
}

/// Recent trades response result
#[derive(Debug, Deserialize)]
struct TradesResult {
    list: Vec<TradeItem>,
}

#[derive(Debug, Deserialize)]
struct TradeItem {
    #[serde(rename = "execId")]
    exec_id: String,
    price: String,
    size: String,
    side: String,
    time: String,
}

/// Bybit API client
#[derive(Debug, Clone)]
pub struct BybitClient {
    client: reqwest::Client,
    base_url: String,
}

impl BybitClient {
    /// Create a new Bybit client
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: BYBIT_API_URL.to_string(),
        }
    }

    /// Create client with custom base URL (for testing)
    pub fn with_base_url(base_url: &str) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: base_url.to_string(),
        }
    }

    /// Fetch kline/candle data
    ///
    /// # Arguments
    /// * `symbol` - Trading pair (e.g., "BTCUSDT")
    /// * `interval` - Candle interval ("1", "5", "15", "60", "240", "D", "W")
    /// * `limit` - Number of candles (max 1000)
    pub async fn get_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> Result<Vec<Candle>> {
        let url = format!(
            "{}/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
            self.base_url, symbol, interval, limit
        );

        let response: BybitResponse<KlineResult> = self.client
            .get(&url)
            .send()
            .await?
            .json()
            .await?;

        if response.ret_code != 0 {
            return Err(Error::Data(response.ret_msg));
        }

        let candles = response.result.list
            .into_iter()
            .map(|item| {
                Candle {
                    timestamp: item[0].parse().unwrap_or(0),
                    open: item[1].parse().unwrap_or(0.0),
                    high: item[2].parse().unwrap_or(0.0),
                    low: item[3].parse().unwrap_or(0.0),
                    close: item[4].parse().unwrap_or(0.0),
                    volume: item[5].parse().unwrap_or(0.0),
                    turnover: item[6].parse().unwrap_or(0.0),
                }
            })
            .collect();

        Ok(candles)
    }

    /// Fetch order book
    ///
    /// # Arguments
    /// * `symbol` - Trading pair (e.g., "BTCUSDT")
    /// * `depth` - Order book depth (5, 10, 20, 50, 100, 200)
    pub async fn get_orderbook(&self, symbol: &str, depth: usize) -> Result<OrderBook> {
        let url = format!(
            "{}/v5/market/orderbook?category=spot&symbol={}&limit={}",
            self.base_url, symbol, depth
        );

        let response: BybitResponse<OrderBookResult> = self.client
            .get(&url)
            .send()
            .await?
            .json()
            .await?;

        if response.ret_code != 0 {
            return Err(Error::Data(response.ret_msg));
        }

        let bids = response.result.b
            .into_iter()
            .map(|item| OrderBookLevel {
                price: item[0].parse().unwrap_or(0.0),
                size: item[1].parse().unwrap_or(0.0),
            })
            .collect();

        let asks = response.result.a
            .into_iter()
            .map(|item| OrderBookLevel {
                price: item[0].parse().unwrap_or(0.0),
                size: item[1].parse().unwrap_or(0.0),
            })
            .collect();

        Ok(OrderBook {
            symbol: response.result.s,
            timestamp: response.result.ts,
            bids,
            asks,
        })
    }

    /// Fetch recent trades
    ///
    /// # Arguments
    /// * `symbol` - Trading pair (e.g., "BTCUSDT")
    /// * `limit` - Number of trades (max 1000)
    pub async fn get_recent_trades(&self, symbol: &str, limit: usize) -> Result<Vec<Trade>> {
        let url = format!(
            "{}/v5/market/recent-trade?category=spot&symbol={}&limit={}",
            self.base_url, symbol, limit
        );

        let response: BybitResponse<TradesResult> = self.client
            .get(&url)
            .send()
            .await?
            .json()
            .await?;

        if response.ret_code != 0 {
            return Err(Error::Data(response.ret_msg));
        }

        let trades = response.result.list
            .into_iter()
            .map(|item| Trade {
                id: item.exec_id,
                price: item.price.parse().unwrap_or(0.0),
                size: item.size.parse().unwrap_or(0.0),
                side: item.side,
                timestamp: item.time.parse().unwrap_or(0),
            })
            .collect();

        Ok(trades)
    }

    /// Fetch ticker information
    pub async fn get_ticker(&self, symbol: &str) -> Result<(f64, f64)> {
        let url = format!(
            "{}/v5/market/tickers?category=spot&symbol={}",
            self.base_url, symbol
        );

        #[derive(Deserialize)]
        struct TickerResult {
            list: Vec<TickerItem>,
        }

        #[derive(Deserialize)]
        struct TickerItem {
            #[serde(rename = "lastPrice")]
            last_price: String,
            #[serde(rename = "volume24h")]
            volume_24h: String,
        }

        let response: BybitResponse<TickerResult> = self.client
            .get(&url)
            .send()
            .await?
            .json()
            .await?;

        if response.ret_code != 0 {
            return Err(Error::Data(response.ret_msg));
        }

        if let Some(ticker) = response.result.list.first() {
            Ok((
                ticker.last_price.parse().unwrap_or(0.0),
                ticker.volume_24h.parse().unwrap_or(0.0),
            ))
        } else {
            Err(Error::Data("No ticker data".to_string()))
        }
    }
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

/// Generate simulated candle data for testing
pub fn generate_simulated_candles(num_candles: usize, start_price: f64) -> Vec<Candle> {
    use rand::Rng;
    use rand_distr::{Normal, Distribution};

    let mut rng = rand::thread_rng();
    let returns_dist = Normal::new(0.0, 0.02).unwrap(); // 2% daily volatility
    let mut candles = Vec::with_capacity(num_candles);
    let mut price = start_price;
    let mut timestamp = chrono::Utc::now().timestamp_millis();

    for _ in 0..num_candles {
        let ret = returns_dist.sample(&mut rng);
        let open = price;
        let close = price * (1.0 + ret);

        let high = open.max(close) * (1.0 + rng.gen::<f64>() * 0.01);
        let low = open.min(close) * (1.0 - rng.gen::<f64>() * 0.01);

        let volume = rng.gen_range(100.0..10000.0);

        candles.push(Candle {
            timestamp,
            open,
            high,
            low,
            close,
            volume,
            turnover: volume * (high + low) / 2.0,
        });

        price = close;
        timestamp += 60000; // 1 minute intervals
    }

    candles
}

/// Generate simulated order book
pub fn generate_simulated_orderbook(mid_price: f64, depth: usize) -> OrderBook {
    use rand::Rng;

    let mut rng = rand::thread_rng();
    let mut bids = Vec::with_capacity(depth);
    let mut asks = Vec::with_capacity(depth);

    for i in 0..depth {
        let spread = 0.0001 * (i + 1) as f64;
        let bid_price = mid_price * (1.0 - spread);
        let ask_price = mid_price * (1.0 + spread);

        bids.push(OrderBookLevel {
            price: bid_price,
            size: rng.gen_range(0.1..10.0),
        });

        asks.push(OrderBookLevel {
            price: ask_price,
            size: rng.gen_range(0.1..10.0),
        });
    }

    OrderBook {
        symbol: "BTCUSDT".to_string(),
        timestamp: chrono::Utc::now().timestamp_millis(),
        bids,
        asks,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_candle_calculations() {
        let candle = Candle {
            timestamp: 0,
            open: 100.0,
            high: 110.0,
            low: 95.0,
            close: 105.0,
            volume: 1000.0,
            turnover: 102500.0,
        };

        assert!((candle.return_pct() - 0.05).abs() < 1e-10);
        assert!((candle.range() - 15.0).abs() < 1e-10);
        assert!((candle.body_size() - 5.0).abs() < 1e-10);
        assert!(candle.is_bullish());
    }

    #[test]
    fn test_orderbook_calculations() {
        let orderbook = OrderBook {
            symbol: "BTCUSDT".to_string(),
            timestamp: 0,
            bids: vec![
                OrderBookLevel { price: 99.0, size: 10.0 },
                OrderBookLevel { price: 98.0, size: 20.0 },
            ],
            asks: vec![
                OrderBookLevel { price: 101.0, size: 15.0 },
                OrderBookLevel { price: 102.0, size: 25.0 },
            ],
        };

        assert_eq!(orderbook.best_bid(), Some(99.0));
        assert_eq!(orderbook.best_ask(), Some(101.0));
        assert_eq!(orderbook.spread(), Some(2.0));
        assert_eq!(orderbook.mid_price(), Some(100.0));

        // Imbalance: (10-15)/(10+15) = -0.2
        let imbalance = orderbook.imbalance(1);
        assert!((imbalance - (-0.2)).abs() < 1e-10);
    }

    #[test]
    fn test_simulated_data() {
        let candles = generate_simulated_candles(100, 50000.0);
        assert_eq!(candles.len(), 100);

        let orderbook = generate_simulated_orderbook(50000.0, 10);
        assert_eq!(orderbook.bids.len(), 10);
        assert_eq!(orderbook.asks.len(), 10);
    }
}
