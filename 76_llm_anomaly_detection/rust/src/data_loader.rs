//! Data loading utilities for financial data.
//!
//! Supports loading OHLCV data from:
//! - Bybit (cryptocurrency)
//! - Yahoo Finance (stocks)

use crate::types::{Candle, Features};
use anyhow::{anyhow, Result};
use chrono::{DateTime, TimeZone, Utc};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Bybit API response for klines.
#[derive(Debug, Deserialize)]
struct BybitKlineResponse {
    #[serde(rename = "retCode")]
    ret_code: i32,
    #[serde(rename = "retMsg")]
    ret_msg: String,
    result: BybitKlineResult,
}

#[derive(Debug, Deserialize)]
struct BybitKlineResult {
    symbol: String,
    list: Vec<Vec<String>>,
}

/// Bybit API response for ticker.
#[derive(Debug, Deserialize)]
struct BybitTickerResponse {
    #[serde(rename = "retCode")]
    ret_code: i32,
    result: BybitTickerResult,
}

#[derive(Debug, Deserialize)]
struct BybitTickerResult {
    list: Vec<BybitTicker>,
}

#[derive(Debug, Deserialize)]
struct BybitTicker {
    symbol: String,
    #[serde(rename = "lastPrice")]
    last_price: String,
    #[serde(rename = "highPrice24h")]
    high_price_24h: String,
    #[serde(rename = "lowPrice24h")]
    low_price_24h: String,
    #[serde(rename = "volume24h")]
    volume_24h: String,
    #[serde(rename = "price24hPcnt")]
    price_24h_pcnt: String,
}

/// Ticker data.
#[derive(Debug, Clone, Serialize)]
pub struct Ticker {
    pub symbol: String,
    pub last_price: f64,
    pub high_24h: f64,
    pub low_24h: f64,
    pub volume_24h: f64,
    pub price_change_pct: f64,
}

/// Bybit data loader.
pub struct BybitLoader {
    client: Client,
    base_url: String,
}

impl BybitLoader {
    /// Create a new Bybit loader.
    pub fn new() -> Self {
        Self {
            client: Client::new(),
            base_url: "https://api.bybit.com".to_string(),
        }
    }

    /// Convert interval string to Bybit format.
    fn convert_interval(interval: &str) -> &str {
        match interval {
            "1m" => "1",
            "3m" => "3",
            "5m" => "5",
            "15m" => "15",
            "30m" => "30",
            "1h" => "60",
            "2h" => "120",
            "4h" => "240",
            "6h" => "360",
            "12h" => "720",
            "1d" | "1D" => "D",
            "1w" | "1W" => "W",
            "1M" => "M",
            _ => interval,
        }
    }

    /// Get kline (candlestick) data.
    pub async fn get_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> Result<Vec<Candle>> {
        let bybit_interval = Self::convert_interval(interval);

        let url = format!("{}/v5/market/kline", self.base_url);

        let response = self
            .client
            .get(&url)
            .query(&[
                ("category", "linear"),
                ("symbol", symbol),
                ("interval", bybit_interval),
                ("limit", &limit.to_string()),
            ])
            .send()
            .await?;

        let data: BybitKlineResponse = response.json().await?;

        if data.ret_code != 0 {
            return Err(anyhow!("Bybit API error: {}", data.ret_msg));
        }

        let mut candles: Vec<Candle> = data
            .result
            .list
            .iter()
            .filter_map(|k| {
                if k.len() < 6 {
                    return None;
                }

                let timestamp_ms: i64 = k[0].parse().ok()?;
                let timestamp = Utc.timestamp_millis_opt(timestamp_ms).single()?;

                Some(Candle {
                    timestamp,
                    open: k[1].parse().ok()?,
                    high: k[2].parse().ok()?,
                    low: k[3].parse().ok()?,
                    close: k[4].parse().ok()?,
                    volume: k[5].parse().ok()?,
                })
            })
            .collect();

        // Sort by timestamp ascending
        candles.sort_by_key(|c| c.timestamp);

        Ok(candles)
    }

    /// Get 24-hour ticker data.
    pub async fn get_ticker(&self, symbol: &str) -> Result<Ticker> {
        let url = format!("{}/v5/market/tickers", self.base_url);

        let response = self
            .client
            .get(&url)
            .query(&[("category", "linear"), ("symbol", symbol)])
            .send()
            .await?;

        let data: BybitTickerResponse = response.json().await?;

        if data.ret_code != 0 {
            return Err(anyhow!("Bybit API error"));
        }

        let ticker = data
            .result
            .list
            .first()
            .ok_or_else(|| anyhow!("No ticker data for {}", symbol))?;

        Ok(Ticker {
            symbol: ticker.symbol.clone(),
            last_price: ticker.last_price.parse()?,
            high_24h: ticker.high_price_24h.parse()?,
            low_24h: ticker.low_price_24h.parse()?,
            volume_24h: ticker.volume_24h.parse()?,
            price_change_pct: ticker.price_24h_pcnt.parse::<f64>()? * 100.0,
        })
    }
}

impl Default for BybitLoader {
    fn default() -> Self {
        Self::new()
    }
}

/// Feature calculator for OHLCV data.
pub struct FeatureCalculator {
    window_size: usize,
}

impl FeatureCalculator {
    /// Create a new feature calculator.
    pub fn new(window_size: usize) -> Self {
        Self { window_size }
    }

    /// Calculate rolling mean.
    fn rolling_mean(data: &[f64], window: usize) -> Vec<f64> {
        if data.len() < window {
            return vec![f64::NAN; data.len()];
        }

        let mut result = vec![f64::NAN; window - 1];

        for i in (window - 1)..data.len() {
            let sum: f64 = data[(i + 1 - window)..=i].iter().sum();
            result.push(sum / window as f64);
        }

        result
    }

    /// Calculate rolling standard deviation.
    fn rolling_std(data: &[f64], window: usize) -> Vec<f64> {
        if data.len() < window {
            return vec![f64::NAN; data.len()];
        }

        let mut result = vec![f64::NAN; window - 1];

        for i in (window - 1)..data.len() {
            let slice = &data[(i + 1 - window)..=i];
            let mean: f64 = slice.iter().sum::<f64>() / window as f64;
            let variance: f64 = slice.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / window as f64;
            result.push(variance.sqrt());
        }

        result
    }

    /// Calculate features for all candles.
    pub fn calculate_features(&self, candles: &[Candle]) -> Vec<Features> {
        if candles.is_empty() {
            return vec![];
        }

        let n = candles.len();

        // Extract close prices and volumes
        let closes: Vec<f64> = candles.iter().map(|c| c.close).collect();
        let volumes: Vec<f64> = candles.iter().map(|c| c.volume).collect();
        let ranges: Vec<f64> = candles.iter().map(|c| c.range_pct()).collect();

        // Calculate returns
        let mut returns = vec![0.0];
        let mut log_returns = vec![0.0];
        for i in 1..n {
            returns.push((closes[i] - closes[i - 1]) / closes[i - 1]);
            log_returns.push((closes[i] / closes[i - 1]).ln());
        }

        // Calculate rolling statistics
        let vol_ma = Self::rolling_mean(&volumes, self.window_size);
        let returns_ma = Self::rolling_mean(&returns, self.window_size);
        let returns_std = Self::rolling_std(&returns, self.window_size);
        let vol_std = Self::rolling_std(&volumes, self.window_size);
        let range_ma = Self::rolling_mean(&ranges, self.window_size);

        // Build features
        let mut features = Vec::with_capacity(n);

        for i in 0..n {
            let volatility = if i >= self.window_size {
                returns_std[i]
            } else {
                f64::NAN
            };

            let volume_ratio = if vol_ma[i].is_finite() && vol_ma[i] > 0.0 {
                volumes[i] / vol_ma[i]
            } else {
                1.0
            };

            let range_ratio = if range_ma[i].is_finite() && range_ma[i] > 0.0 {
                ranges[i] / range_ma[i]
            } else {
                1.0
            };

            let returns_zscore = if returns_std[i].is_finite() && returns_std[i] > 0.0 {
                (returns[i] - returns_ma[i]) / returns_std[i]
            } else {
                0.0
            };

            let volume_zscore = if vol_std[i].is_finite() && vol_std[i] > 0.0 {
                (volumes[i] - vol_ma[i]) / vol_std[i]
            } else {
                0.0
            };

            features.push(Features {
                returns: returns[i],
                log_returns: log_returns[i],
                volatility,
                volume_ratio,
                range_ratio,
                returns_zscore,
                volume_zscore,
            });
        }

        features
    }
}

impl Default for FeatureCalculator {
    fn default() -> Self {
        Self::new(20)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rolling_mean() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = FeatureCalculator::rolling_mean(&data, 3);

        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert!((result[2] - 2.0).abs() < 0.001);
        assert!((result[3] - 3.0).abs() < 0.001);
        assert!((result[4] - 4.0).abs() < 0.001);
    }

    #[test]
    fn test_feature_calculator() {
        let candles: Vec<Candle> = (0..30)
            .map(|i| Candle {
                timestamp: Utc::now(),
                open: 100.0 + i as f64,
                high: 102.0 + i as f64,
                low: 99.0 + i as f64,
                close: 101.0 + i as f64,
                volume: 1000.0 + (i * 10) as f64,
            })
            .collect();

        let calc = FeatureCalculator::new(10);
        let features = calc.calculate_features(&candles);

        assert_eq!(features.len(), candles.len());
        assert!(features[20].volatility.is_finite());
        assert!(features[20].volume_ratio.is_finite());
    }
}
