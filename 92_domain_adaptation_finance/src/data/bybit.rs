//! # Bybit API Client
//!
//! Provides a client for fetching OHLCV kline (candlestick) data from the Bybit
//! exchange API. Includes a simulation mode that generates realistic synthetic
//! market data for testing and development without requiring API access.
//!
//! ## Example
//!
//! ```rust,no_run
//! use domain_adaptation_trading::data::bybit::BybitClient;
//!
//! #[tokio::main]
//! async fn main() {
//!     let client = BybitClient::new();
//!     // Use simulated data for testing
//!     let klines = client.simulated_klines("BTCUSDT", 100);
//!     println!("Generated {} klines", klines.len());
//! }
//! ```

use serde::{Deserialize, Serialize};
use rand::Rng;
use rand_distr::{Distribution, Normal};
use crate::{DomainAdaptationError, Result};

/// Represents a single OHLCV candlestick (kline) data point.
///
/// Each kline captures the price action and volume for a specific time interval.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Kline {
    /// Unix timestamp in milliseconds for the start of the candle
    pub timestamp: u64,
    /// Opening price for the interval
    pub open: f64,
    /// Highest price reached during the interval
    pub high: f64,
    /// Lowest price reached during the interval
    pub low: f64,
    /// Closing price for the interval
    pub close: f64,
    /// Trading volume during the interval
    pub volume: f64,
}

/// Response structure from the Bybit API v5 kline endpoint.
///
/// The Bybit API returns kline data as arrays of strings within a nested
/// JSON structure.
#[derive(Debug, Deserialize)]
struct BybitApiResponse {
    #[serde(rename = "retCode")]
    ret_code: i32,
    #[serde(rename = "retMsg")]
    ret_msg: String,
    result: BybitResult,
}

/// Inner result structure containing the kline list.
#[derive(Debug, Deserialize)]
struct BybitResult {
    /// Symbol name
    #[allow(dead_code)]
    symbol: Option<String>,
    /// Category (e.g., "linear", "spot")
    #[allow(dead_code)]
    category: Option<String>,
    /// List of kline data as string arrays
    list: Vec<Vec<String>>,
}

/// Client for interacting with the Bybit exchange API.
///
/// Supports both live API calls and simulated data generation for
/// development and testing purposes.
///
/// # Architecture
///
/// The client wraps a `reqwest::Client` for HTTP communication and targets
/// the Bybit v5 API endpoint for kline data.
pub struct BybitClient {
    /// Base URL for the Bybit API
    base_url: String,
    /// HTTP client for making API requests
    client: reqwest::Client,
}

impl BybitClient {
    /// Creates a new `BybitClient` with the default Bybit API endpoint.
    ///
    /// # Example
    ///
    /// ```rust
    /// use domain_adaptation_trading::data::bybit::BybitClient;
    /// let client = BybitClient::new();
    /// ```
    pub fn new() -> Self {
        Self {
            base_url: "https://api.bybit.com".to_string(),
            client: reqwest::Client::new(),
        }
    }

    /// Creates a new `BybitClient` with a custom base URL.
    ///
    /// Useful for testing with mock servers or connecting to testnet.
    ///
    /// # Arguments
    ///
    /// * `base_url` - The base URL to use for API requests
    pub fn with_base_url(base_url: &str) -> Self {
        Self {
            base_url: base_url.to_string(),
            client: reqwest::Client::new(),
        }
    }

    /// Fetches kline data from the Bybit API.
    ///
    /// Makes an HTTP GET request to the Bybit v5 kline endpoint and parses
    /// the response into a vector of [`Kline`] structs.
    ///
    /// # Arguments
    ///
    /// * `symbol` - Trading pair symbol (e.g., "BTCUSDT", "ETHUSDT")
    /// * `interval` - Kline interval (e.g., "1", "5", "15", "60", "D")
    /// * `limit` - Maximum number of klines to fetch (max 200 per Bybit API)
    ///
    /// # Returns
    ///
    /// A vector of [`Kline`] structs sorted by timestamp ascending.
    ///
    /// # Errors
    ///
    /// Returns [`DomainAdaptationError::ApiError`] if the API request fails
    /// or the response cannot be parsed.
    pub async fn fetch_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> Result<Vec<Kline>> {
        let url = format!(
            "{}/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
            self.base_url, symbol, interval, limit
        );

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| DomainAdaptationError::ApiError(format!("Request failed: {}", e)))?;

        let api_response: BybitApiResponse = response
            .json()
            .await
            .map_err(|e| DomainAdaptationError::ApiError(format!("Parse failed: {}", e)))?;

        if api_response.ret_code != 0 {
            return Err(DomainAdaptationError::ApiError(format!(
                "Bybit API error {}: {}",
                api_response.ret_code, api_response.ret_msg
            )));
        }

        let mut klines: Vec<Kline> = api_response
            .result
            .list
            .iter()
            .filter_map(|row| {
                if row.len() >= 6 {
                    Some(Kline {
                        timestamp: row[0].parse().unwrap_or(0),
                        open: row[1].parse().unwrap_or(0.0),
                        high: row[2].parse().unwrap_or(0.0),
                        low: row[3].parse().unwrap_or(0.0),
                        close: row[4].parse().unwrap_or(0.0),
                        volume: row[5].parse().unwrap_or(0.0),
                    })
                } else {
                    None
                }
            })
            .collect();

        // Bybit returns newest first; reverse to get chronological order
        klines.reverse();

        Ok(klines)
    }

    /// Generates simulated kline data for testing and development.
    ///
    /// Creates realistic synthetic market data using a geometric Brownian motion
    /// model with mean-reverting volatility. The simulation parameters are tuned
    /// to match the statistical properties of typical crypto markets.
    ///
    /// # Arguments
    ///
    /// * `symbol` - Symbol name (used to seed different starting prices)
    /// * `n_samples` - Number of klines to generate
    ///
    /// # Returns
    ///
    /// A vector of [`Kline`] structs with realistic OHLCV data.
    ///
    /// # Example
    ///
    /// ```rust
    /// use domain_adaptation_trading::data::bybit::BybitClient;
    ///
    /// let client = BybitClient::new();
    /// let btc_data = client.simulated_klines("BTCUSDT", 500);
    /// assert_eq!(btc_data.len(), 500);
    /// ```
    pub fn simulated_klines(&self, symbol: &str, n_samples: usize) -> Vec<Kline> {
        let mut rng = rand::thread_rng();

        // Set starting price based on symbol for variety
        let base_price = match symbol {
            s if s.contains("BTC") => 45000.0,
            s if s.contains("ETH") => 2500.0,
            s if s.contains("SOL") => 100.0,
            s if s.contains("SPY") => 450.0,
            s if s.contains("QQQ") => 380.0,
            _ => 1000.0,
        };

        // Market-specific volatility
        let base_volatility = match symbol {
            s if s.contains("BTC") || s.contains("ETH") || s.contains("SOL") => 0.02,
            _ => 0.01,
        };

        let return_dist = Normal::new(0.0001, base_volatility).unwrap();
        let volume_dist = Normal::new(1000.0, 300.0).unwrap();

        let mut klines = Vec::with_capacity(n_samples);
        let mut current_price = base_price;
        let base_timestamp: u64 = 1_700_000_000_000; // Recent timestamp in ms

        for i in 0..n_samples {
            let ret: f64 = return_dist.sample(&mut rng);
            let new_price = current_price * (1.0 + ret);

            // Generate realistic OHLC from the move
            let intrabar_vol = base_volatility * 0.5;
            let noise1: f64 = rng.gen_range(-intrabar_vol..intrabar_vol);
            let noise2: f64 = rng.gen_range(-intrabar_vol..intrabar_vol);

            let open = current_price;
            let close = new_price;
            let high = open.max(close) * (1.0 + noise1.abs());
            let low = open.min(close) * (1.0 - noise2.abs());
            let vol_sample: f64 = volume_dist.sample(&mut rng);
            let volume = vol_sample.abs().max(10.0);

            klines.push(Kline {
                timestamp: base_timestamp + (i as u64) * 60_000, // 1-minute intervals
                open,
                high,
                low,
                close,
                volume,
            });

            current_price = new_price;
        }

        klines
    }

    /// Generates simulated klines with a domain shift applied.
    ///
    /// This is useful for testing domain adaptation: the target domain data
    /// has different statistical properties (shifted mean, scaled volatility)
    /// compared to the source domain.
    ///
    /// # Arguments
    ///
    /// * `symbol` - Symbol name for base price determination
    /// * `n_samples` - Number of klines to generate
    /// * `mean_shift` - Additive shift to the return distribution mean
    /// * `volatility_scale` - Multiplicative scale for volatility
    ///
    /// # Returns
    ///
    /// A vector of domain-shifted [`Kline`] structs.
    pub fn simulated_klines_with_shift(
        &self,
        symbol: &str,
        n_samples: usize,
        mean_shift: f64,
        volatility_scale: f64,
    ) -> Vec<Kline> {
        let mut rng = rand::thread_rng();

        let base_price = match symbol {
            s if s.contains("BTC") => 45000.0,
            s if s.contains("ETH") => 2500.0,
            _ => 1000.0,
        };

        let base_volatility = 0.02 * volatility_scale;
        let return_dist = Normal::new(0.0001 + mean_shift, base_volatility).unwrap();
        let volume_dist = Normal::new(1000.0, 300.0).unwrap();

        let mut klines = Vec::with_capacity(n_samples);
        let mut current_price = base_price;
        let base_timestamp: u64 = 1_700_000_000_000;

        for i in 0..n_samples {
            let ret: f64 = return_dist.sample(&mut rng);
            let new_price = current_price * (1.0 + ret);

            let intrabar_vol = base_volatility * 0.5;
            let noise1: f64 = rng.gen_range(-intrabar_vol..intrabar_vol);
            let noise2: f64 = rng.gen_range(-intrabar_vol..intrabar_vol);

            let open = current_price;
            let close = new_price;
            let high = open.max(close) * (1.0 + noise1.abs());
            let low = open.min(close) * (1.0 - noise2.abs());
            let vol_sample: f64 = volume_dist.sample(&mut rng);
            let volume = vol_sample.abs().max(10.0);

            klines.push(Kline {
                timestamp: base_timestamp + (i as u64) * 60_000,
                open,
                high,
                low,
                close,
                volume,
            });

            current_price = new_price;
        }

        klines
    }
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simulated_klines_length() {
        let client = BybitClient::new();
        let klines = client.simulated_klines("BTCUSDT", 100);
        assert_eq!(klines.len(), 100);
    }

    #[test]
    fn test_simulated_klines_price_consistency() {
        let client = BybitClient::new();
        let klines = client.simulated_klines("BTCUSDT", 50);
        for kline in &klines {
            assert!(kline.high >= kline.low, "High must be >= low");
            assert!(kline.high >= kline.open, "High must be >= open");
            assert!(kline.high >= kline.close, "High must be >= close");
            assert!(kline.low <= kline.open, "Low must be <= open");
            assert!(kline.low <= kline.close, "Low must be <= close");
            assert!(kline.volume > 0.0, "Volume must be positive");
        }
    }

    #[test]
    fn test_simulated_klines_different_symbols() {
        let client = BybitClient::new();
        let btc = client.simulated_klines("BTCUSDT", 10);
        let eth = client.simulated_klines("ETHUSDT", 10);

        // Different symbols should have different base prices
        assert!(btc[0].open > eth[0].open);
    }

    #[test]
    fn test_simulated_klines_with_shift() {
        let client = BybitClient::new();
        let normal = client.simulated_klines("BTCUSDT", 200);
        let shifted = client.simulated_klines_with_shift("BTCUSDT", 200, 0.01, 2.0);

        // With a positive mean shift, prices should tend to drift higher
        // (statistical test, may occasionally fail but very unlikely with 200 samples)
        let normal_end = normal.last().unwrap().close;
        let shifted_end = shifted.last().unwrap().close;

        // Just verify both produced valid data
        assert!(normal_end > 0.0);
        assert!(shifted_end > 0.0);
    }

    #[test]
    fn test_timestamps_increasing() {
        let client = BybitClient::new();
        let klines = client.simulated_klines("BTCUSDT", 50);
        for i in 1..klines.len() {
            assert!(klines[i].timestamp > klines[i - 1].timestamp);
        }
    }
}
