//! # Bybit API Client
//!
//! Async client for fetching market data from the Bybit exchange.

use crate::{DeepLiftError, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};

/// Base URL for Bybit API
const BYBIT_API_BASE: &str = "https://api.bybit.com";

/// Kline/candlestick data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Kline {
    /// Start timestamp in milliseconds
    pub start_time: u64,
    /// Open price
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

/// Ticker data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ticker {
    /// Symbol
    pub symbol: String,
    /// Last traded price
    pub last_price: f64,
    /// Index price
    pub index_price: f64,
    /// Mark price
    pub mark_price: f64,
    /// 24h price change percentage
    pub price_24h_pcnt: f64,
    /// 24h high price
    pub high_price_24h: f64,
    /// 24h low price
    pub low_price_24h: f64,
    /// 24h turnover
    pub turnover_24h: f64,
    /// 24h volume
    pub volume_24h: f64,
    /// Funding rate
    pub funding_rate: f64,
    /// Next funding time
    pub next_funding_time: u64,
    /// Bid price
    pub bid_price: f64,
    /// Ask price
    pub ask_price: f64,
}

/// Raw API response structure
#[derive(Debug, Deserialize)]
struct ApiResponse<T> {
    #[serde(rename = "retCode")]
    ret_code: i32,
    #[serde(rename = "retMsg")]
    ret_msg: String,
    result: T,
}

/// Kline API result
#[derive(Debug, Deserialize)]
struct KlineResult {
    symbol: String,
    category: String,
    list: Vec<Vec<String>>,
}

/// Ticker API result
#[derive(Debug, Deserialize)]
struct TickerResult {
    category: String,
    list: Vec<RawTicker>,
}

/// Raw ticker from API
#[derive(Debug, Deserialize)]
struct RawTicker {
    symbol: String,
    #[serde(rename = "lastPrice")]
    last_price: String,
    #[serde(rename = "indexPrice")]
    index_price: String,
    #[serde(rename = "markPrice")]
    mark_price: String,
    #[serde(rename = "price24hPcnt")]
    price_24h_pcnt: String,
    #[serde(rename = "highPrice24h")]
    high_price_24h: String,
    #[serde(rename = "lowPrice24h")]
    low_price_24h: String,
    #[serde(rename = "turnover24h")]
    turnover_24h: String,
    #[serde(rename = "volume24h")]
    volume_24h: String,
    #[serde(rename = "fundingRate")]
    funding_rate: String,
    #[serde(rename = "nextFundingTime")]
    next_funding_time: String,
    #[serde(rename = "bid1Price")]
    bid_price: String,
    #[serde(rename = "ask1Price")]
    ask_price: String,
}

/// Bybit API client for fetching market data
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
    /// Create a new Bybit client with default settings
    pub fn new() -> Self {
        Self {
            client: Client::new(),
            base_url: BYBIT_API_BASE.to_string(),
        }
    }

    /// Create a client with custom base URL (for testing)
    pub fn with_base_url(base_url: &str) -> Self {
        Self {
            client: Client::new(),
            base_url: base_url.to_string(),
        }
    }

    /// Fetch kline/candlestick data
    ///
    /// # Arguments
    /// * `symbol` - Trading pair symbol (e.g., "BTCUSDT")
    /// * `interval` - Kline interval (e.g., "1", "5", "15", "60", "240", "D")
    /// * `limit` - Number of klines to fetch (max 200)
    ///
    /// # Returns
    /// Vector of Kline data sorted by time (oldest first)
    pub async fn fetch_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> Result<Vec<Kline>> {
        let url = format!(
            "{}/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
            self.base_url, symbol, interval, limit.min(200)
        );

        let response = self.client
            .get(&url)
            .send()
            .await?
            .json::<ApiResponse<KlineResult>>()
            .await?;

        if response.ret_code != 0 {
            return Err(DeepLiftError::DataError(format!(
                "API error: {} - {}",
                response.ret_code, response.ret_msg
            )));
        }

        let mut klines: Vec<Kline> = response.result.list
            .into_iter()
            .filter_map(|row| {
                if row.len() >= 7 {
                    Some(Kline {
                        start_time: row[0].parse().unwrap_or(0),
                        open: row[1].parse().unwrap_or(0.0),
                        high: row[2].parse().unwrap_or(0.0),
                        low: row[3].parse().unwrap_or(0.0),
                        close: row[4].parse().unwrap_or(0.0),
                        volume: row[5].parse().unwrap_or(0.0),
                        turnover: row[6].parse().unwrap_or(0.0),
                    })
                } else {
                    None
                }
            })
            .collect();

        // Sort by time (API returns newest first)
        klines.sort_by_key(|k| k.start_time);

        Ok(klines)
    }

    /// Fetch ticker data for a symbol
    ///
    /// # Arguments
    /// * `symbol` - Trading pair symbol (e.g., "BTCUSDT")
    ///
    /// # Returns
    /// Current ticker data
    pub async fn fetch_ticker(&self, symbol: &str) -> Result<Ticker> {
        let url = format!(
            "{}/v5/market/tickers?category=linear&symbol={}",
            self.base_url, symbol
        );

        let response = self.client
            .get(&url)
            .send()
            .await?
            .json::<ApiResponse<TickerResult>>()
            .await?;

        if response.ret_code != 0 {
            return Err(DeepLiftError::DataError(format!(
                "API error: {} - {}",
                response.ret_code, response.ret_msg
            )));
        }

        let raw = response.result.list
            .into_iter()
            .next()
            .ok_or_else(|| DeepLiftError::DataError("No ticker data found".to_string()))?;

        Ok(Ticker {
            symbol: raw.symbol,
            last_price: raw.last_price.parse().unwrap_or(0.0),
            index_price: raw.index_price.parse().unwrap_or(0.0),
            mark_price: raw.mark_price.parse().unwrap_or(0.0),
            price_24h_pcnt: raw.price_24h_pcnt.parse().unwrap_or(0.0),
            high_price_24h: raw.high_price_24h.parse().unwrap_or(0.0),
            low_price_24h: raw.low_price_24h.parse().unwrap_or(0.0),
            turnover_24h: raw.turnover_24h.parse().unwrap_or(0.0),
            volume_24h: raw.volume_24h.parse().unwrap_or(0.0),
            funding_rate: raw.funding_rate.parse().unwrap_or(0.0),
            next_funding_time: raw.next_funding_time.parse().unwrap_or(0),
            bid_price: raw.bid_price.parse().unwrap_or(0.0),
            ask_price: raw.ask_price.parse().unwrap_or(0.0),
        })
    }

    /// Get historical data as feature vectors for the model
    ///
    /// # Arguments
    /// * `symbol` - Trading pair symbol
    /// * `interval` - Kline interval
    /// * `lookback` - Number of periods to look back
    ///
    /// # Returns
    /// Vector of feature vectors, each containing normalized OHLCV data
    pub async fn get_historical_data(
        &self,
        symbol: &str,
        interval: &str,
        lookback: usize,
    ) -> Result<Vec<Vec<f64>>> {
        let klines = self.fetch_klines(symbol, interval, lookback + 50).await?;

        if klines.len() < lookback + 1 {
            return Err(DeepLiftError::InsufficientData(format!(
                "Need at least {} klines, got {}",
                lookback + 1,
                klines.len()
            )));
        }

        let mut features = Vec::new();

        for i in lookback..klines.len() {
            let window = &klines[i - lookback..=i];
            let feature_vec = self.compute_features(window);
            features.push(feature_vec);
        }

        Ok(features)
    }

    /// Compute feature vector from kline window
    fn compute_features(&self, klines: &[Kline]) -> Vec<f64> {
        let n = klines.len();
        if n == 0 {
            return vec![];
        }

        let latest = &klines[n - 1];
        let first = &klines[0];

        // Price features
        let price_change = (latest.close - first.close) / first.close;
        let high_low_range = (latest.high - latest.low) / latest.close;
        let open_close_range = (latest.close - latest.open) / latest.open;

        // Moving averages
        let closes: Vec<f64> = klines.iter().map(|k| k.close).collect();
        let sma = closes.iter().sum::<f64>() / n as f64;
        let price_vs_sma = (latest.close - sma) / sma;

        // Volatility
        let mean = sma;
        let variance = closes.iter()
            .map(|c| (c - mean).powi(2))
            .sum::<f64>() / n as f64;
        let volatility = variance.sqrt() / mean;

        // Volume features
        let volumes: Vec<f64> = klines.iter().map(|k| k.volume).collect();
        let avg_volume = volumes.iter().sum::<f64>() / n as f64;
        let volume_ratio = if avg_volume > 0.0 {
            latest.volume / avg_volume
        } else {
            1.0
        };

        // Momentum (Rate of Change)
        let roc = if n > 1 {
            let prev_close = klines[n - 2].close;
            (latest.close - prev_close) / prev_close
        } else {
            0.0
        };

        // RSI-like indicator (simplified)
        let mut gains = 0.0;
        let mut losses = 0.0;
        for i in 1..n {
            let change = klines[i].close - klines[i - 1].close;
            if change > 0.0 {
                gains += change;
            } else {
                losses -= change;
            }
        }
        let rsi = if gains + losses > 0.0 {
            gains / (gains + losses)
        } else {
            0.5
        };

        // MACD-like indicator (simplified)
        let short_period = (n / 2).max(1);
        let short_sma: f64 = closes[n - short_period..].iter().sum::<f64>() / short_period as f64;
        let macd = (short_sma - sma) / sma;

        // Bollinger Band position
        let std_dev = volatility * mean;
        let upper_band = sma + 2.0 * std_dev;
        let lower_band = sma - 2.0 * std_dev;
        let bb_position = if upper_band > lower_band {
            (latest.close - lower_band) / (upper_band - lower_band)
        } else {
            0.5
        };

        vec![
            price_change,
            high_low_range,
            open_close_range,
            price_vs_sma,
            volatility,
            volume_ratio - 1.0, // Center around 0
            roc,
            rsi - 0.5, // Center around 0
            macd,
            bb_position - 0.5, // Center around 0
        ]
    }
}

/// Feature names for interpretation
pub const FEATURE_NAMES: [&str; 10] = [
    "price_change",
    "high_low_range",
    "open_close_range",
    "price_vs_sma",
    "volatility",
    "volume_ratio",
    "momentum",
    "rsi",
    "macd",
    "bb_position",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_creation() {
        let client = BybitClient::new();
        assert_eq!(client.base_url, BYBIT_API_BASE);
    }

    #[test]
    fn test_custom_base_url() {
        let client = BybitClient::with_base_url("https://custom.api.com");
        assert_eq!(client.base_url, "https://custom.api.com");
    }

    #[test]
    fn test_compute_features() {
        let client = BybitClient::new();
        let klines = vec![
            Kline {
                start_time: 1000,
                open: 100.0,
                high: 110.0,
                low: 95.0,
                close: 105.0,
                volume: 1000.0,
                turnover: 105000.0,
            },
            Kline {
                start_time: 2000,
                open: 105.0,
                high: 115.0,
                low: 100.0,
                close: 110.0,
                volume: 1200.0,
                turnover: 132000.0,
            },
        ];

        let features = client.compute_features(&klines);
        assert_eq!(features.len(), 10);
    }

    #[test]
    fn test_feature_names() {
        assert_eq!(FEATURE_NAMES.len(), 10);
        assert_eq!(FEATURE_NAMES[0], "price_change");
    }
}
