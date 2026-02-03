//! Bybit exchange data client.
//!
//! Fetches OHLCV (kline) data from the Bybit public API v5.
//! No API key required for public market data.

use crate::RDDError;
use serde::Deserialize;

/// OHLCV candle data.
#[derive(Debug, Clone)]
pub struct Candle {
    pub timestamp: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// Market data container.
#[derive(Debug, Clone)]
pub struct MarketData {
    pub symbol: String,
    pub interval: String,
    pub candles: Vec<Candle>,
}

impl MarketData {
    /// Get close prices as a vector.
    pub fn close_prices(&self) -> Vec<f64> {
        self.candles.iter().map(|c| c.close).collect()
    }

    /// Get open prices as a vector.
    pub fn open_prices(&self) -> Vec<f64> {
        self.candles.iter().map(|c| c.open).collect()
    }

    /// Get high prices as a vector.
    pub fn high_prices(&self) -> Vec<f64> {
        self.candles.iter().map(|c| c.high).collect()
    }

    /// Get low prices as a vector.
    pub fn low_prices(&self) -> Vec<f64> {
        self.candles.iter().map(|c| c.low).collect()
    }

    /// Get returns (percentage change) as a vector.
    pub fn returns(&self) -> Vec<f64> {
        let closes = self.close_prices();
        let mut rets = vec![0.0];
        for i in 1..closes.len() {
            if closes[i - 1] > 0.0 {
                rets.push((closes[i] - closes[i - 1]) / closes[i - 1]);
            } else {
                rets.push(0.0);
            }
        }
        rets
    }

    /// Get forward returns (returns shifted by n periods).
    pub fn forward_returns(&self, periods: usize) -> Vec<f64> {
        let closes = self.close_prices();
        let n = closes.len();
        let mut rets = vec![f64::NAN; n];

        for i in 0..(n.saturating_sub(periods)) {
            if closes[i] > 0.0 {
                rets[i] = (closes[i + periods] - closes[i]) / closes[i];
            }
        }
        rets
    }

    /// Get volumes as a vector.
    pub fn volumes(&self) -> Vec<f64> {
        self.candles.iter().map(|c| c.volume).collect()
    }

    /// Get timestamps as a vector.
    pub fn timestamps(&self) -> Vec<u64> {
        self.candles.iter().map(|c| c.timestamp).collect()
    }

    /// Get number of candles.
    pub fn len(&self) -> usize {
        self.candles.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.candles.is_empty()
    }
}

#[derive(Deserialize)]
struct BybitResponse {
    #[serde(rename = "retCode")]
    ret_code: i32,
    #[serde(rename = "retMsg")]
    ret_msg: String,
    result: BybitResult,
}

#[derive(Deserialize)]
struct BybitResult {
    list: Vec<Vec<String>>,
}

/// Funding rate data.
#[derive(Debug, Clone)]
pub struct FundingRate {
    pub timestamp: u64,
    pub symbol: String,
    pub funding_rate: f64,
    pub funding_rate_annualized: f64,
}

#[derive(Deserialize)]
struct FundingResponse {
    #[serde(rename = "retCode")]
    ret_code: i32,
    #[serde(rename = "retMsg")]
    ret_msg: String,
    result: FundingResult,
}

#[derive(Deserialize)]
struct FundingResult {
    list: Vec<FundingItem>,
}

#[derive(Deserialize)]
struct FundingItem {
    symbol: String,
    #[serde(rename = "fundingRate")]
    funding_rate: String,
    #[serde(rename = "fundingRateTimestamp")]
    funding_rate_timestamp: String,
}

/// Bybit API client for fetching market data.
#[derive(Debug, Clone)]
pub struct BybitClient {
    base_url: String,
    client: reqwest::Client,
}

impl BybitClient {
    /// Create a new Bybit client.
    pub fn new() -> Self {
        Self {
            base_url: "https://api.bybit.com".to_string(),
            client: reqwest::Client::new(),
        }
    }

    /// Fetch kline (OHLCV) data from Bybit.
    ///
    /// # Arguments
    /// * `symbol` - Trading pair (e.g., "BTCUSDT")
    /// * `interval` - Kline interval (1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, W, M)
    /// * `limit` - Number of candles (max 1000)
    pub async fn fetch_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> Result<MarketData, RDDError> {
        let url = format!("{}/v5/market/kline", self.base_url);
        let response = self
            .client
            .get(&url)
            .query(&[
                ("category", "linear"),
                ("symbol", symbol),
                ("interval", interval),
                ("limit", &limit.min(1000).to_string()),
            ])
            .send()
            .await?;

        let body: BybitResponse = response.json().await?;

        if body.ret_code != 0 {
            return Err(RDDError::InvalidParameter(format!(
                "Bybit API error: {}",
                body.ret_msg
            )));
        }

        let mut candles: Vec<Candle> = body
            .result
            .list
            .iter()
            .filter_map(|row| {
                if row.len() >= 6 {
                    Some(Candle {
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

        // Bybit returns newest first, reverse to chronological order
        candles.reverse();

        Ok(MarketData {
            symbol: symbol.to_string(),
            interval: interval.to_string(),
            candles,
        })
    }

    /// Fetch historical funding rates.
    ///
    /// # Arguments
    /// * `symbol` - Trading pair (e.g., "BTCUSDT")
    /// * `limit` - Number of records (max 200)
    pub async fn fetch_funding_rates(
        &self,
        symbol: &str,
        limit: usize,
    ) -> Result<Vec<FundingRate>, RDDError> {
        let url = format!("{}/v5/market/funding/history", self.base_url);
        let response = self
            .client
            .get(&url)
            .query(&[
                ("category", "linear"),
                ("symbol", symbol),
                ("limit", &limit.min(200).to_string()),
            ])
            .send()
            .await?;

        let body: FundingResponse = response.json().await?;

        if body.ret_code != 0 {
            return Err(RDDError::InvalidParameter(format!(
                "Bybit API error: {}",
                body.ret_msg
            )));
        }

        let mut rates: Vec<FundingRate> = body
            .result
            .list
            .iter()
            .map(|item| {
                let rate: f64 = item.funding_rate.parse().unwrap_or(0.0);
                FundingRate {
                    timestamp: item.funding_rate_timestamp.parse().unwrap_or(0),
                    symbol: item.symbol.clone(),
                    funding_rate: rate,
                    // Annualize: 3 payments per day * 365 days
                    funding_rate_annualized: rate * 3.0 * 365.0 * 100.0,
                }
            })
            .collect();

        // Reverse to chronological order
        rates.reverse();

        Ok(rates)
    }
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

/// Generate synthetic OHLCV data for testing.
pub fn generate_synthetic_data(
    symbol: &str,
    n_points: usize,
    base_price: f64,
    volatility: f64,
    seed: u64,
) -> MarketData {
    use rand::prelude::*;
    use rand_distr::Normal;

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, volatility).unwrap();

    let mut candles = Vec::with_capacity(n_points);
    let mut price = base_price;

    for i in 0..n_points {
        let ret: f64 = normal.sample(&mut rng);
        let new_price = price * (1.0 + ret);
        let high = new_price * (1.0 + rng.gen::<f64>() * volatility);
        let low = new_price * (1.0 - rng.gen::<f64>() * volatility);
        let volume = (rng.gen::<f64>() * 1000.0 + 100.0).abs();

        candles.push(Candle {
            timestamp: (1672531200000 + i as u64 * 3600000), // hourly from 2023-01-01
            open: price,
            high,
            low,
            close: new_price,
            volume,
        });

        price = new_price;
    }

    MarketData {
        symbol: symbol.to_string(),
        interval: "60".to_string(),
        candles,
    }
}

/// Generate synthetic funding rate data for testing.
pub fn generate_synthetic_funding_rates(
    symbol: &str,
    n_points: usize,
    seed: u64,
) -> Vec<FundingRate> {
    use rand::prelude::*;
    use rand_distr::Normal;

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0001, 0.0005).unwrap();

    let mut rates = Vec::with_capacity(n_points);

    for i in 0..n_points {
        let rate: f64 = normal.sample(&mut rng);
        rates.push(FundingRate {
            timestamp: 1672531200000 + i as u64 * 8 * 3600000, // 8-hourly
            symbol: symbol.to_string(),
            funding_rate: rate,
            funding_rate_annualized: rate * 3.0 * 365.0 * 100.0,
        });
    }

    rates
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synthetic_data() {
        let data = generate_synthetic_data("BTCUSDT", 100, 50000.0, 0.02, 42);
        assert_eq!(data.len(), 100);
        assert_eq!(data.symbol, "BTCUSDT");

        let closes = data.close_prices();
        assert_eq!(closes.len(), 100);

        let returns = data.returns();
        assert_eq!(returns.len(), 100);
        assert_eq!(returns[0], 0.0); // First return is 0
    }

    #[test]
    fn test_forward_returns() {
        let data = generate_synthetic_data("BTCUSDT", 100, 50000.0, 0.02, 42);
        let fwd_returns = data.forward_returns(5);

        assert_eq!(fwd_returns.len(), 100);
        // Last 5 should be NaN
        assert!(fwd_returns[95].is_nan());
        // First ones should be valid
        assert!(!fwd_returns[0].is_nan());
    }
}
