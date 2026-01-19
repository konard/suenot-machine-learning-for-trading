//! Data Loading Utilities
//!
//! This module provides utilities for loading financial data from
//! Bybit and Yahoo Finance APIs.

use anyhow::{Context, Result};
use ndarray::Array2;
use serde::{Deserialize, Serialize};

/// OHLCV data structure
#[derive(Debug, Clone)]
pub struct OHLCVData {
    pub data: Array2<f32>,
    pub symbol: String,
    pub interval: String,
}

impl OHLCVData {
    /// Create new OHLCV data from array
    pub fn new(data: Array2<f32>, symbol: &str, interval: &str) -> Self {
        Self {
            data,
            symbol: symbol.to_string(),
            interval: interval.to_string(),
        }
    }

    /// Get number of candles
    pub fn len(&self) -> usize {
        self.data.shape()[0]
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get close prices
    pub fn close_prices(&self) -> Vec<f32> {
        self.data.column(3).to_vec()
    }

    /// Get latest close price
    pub fn latest_close(&self) -> f32 {
        self.data[[self.len() - 1, 3]]
    }
}

/// Bybit API response structure
#[derive(Debug, Deserialize)]
struct BybitResponse {
    #[serde(rename = "retCode")]
    ret_code: i32,
    #[serde(rename = "retMsg")]
    ret_msg: String,
    result: BybitResult,
}

#[derive(Debug, Deserialize)]
struct BybitResult {
    list: Vec<Vec<String>>,
}

/// Load OHLCV data from Bybit exchange.
///
/// # Arguments
///
/// * `symbol` - Trading pair symbol (e.g., "BTCUSDT")
/// * `interval` - Candle interval ("1m", "5m", "15m", "1h", "4h", "1d")
/// * `limit` - Number of candles to fetch (max 1000)
///
/// # Returns
///
/// Result containing OHLCV data or error
///
/// # Example
///
/// ```rust,no_run
/// use gqa_trading::load_bybit_data;
///
/// let data = load_bybit_data("BTCUSDT", "1h", 500).unwrap();
/// println!("Loaded {} candles", data.len());
/// ```
pub fn load_bybit_data(symbol: &str, interval: &str, limit: usize) -> Result<OHLCVData> {
    let interval_map = [
        ("1m", "1"),
        ("5m", "5"),
        ("15m", "15"),
        ("30m", "30"),
        ("1h", "60"),
        ("4h", "240"),
        ("1d", "D"),
    ];

    let bybit_interval = interval_map
        .iter()
        .find(|(k, _)| *k == interval)
        .map(|(_, v)| *v)
        .unwrap_or(interval);

    let url = format!(
        "https://api.bybit.com/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
        symbol,
        bybit_interval,
        limit.min(1000)
    );

    log::info!("Loading {} data from Bybit ({} interval)...", symbol, interval);

    let client = reqwest::blocking::Client::new();
    let response: BybitResponse = client
        .get(&url)
        .timeout(std::time::Duration::from_secs(10))
        .send()
        .context("Failed to fetch from Bybit")?
        .json()
        .context("Failed to parse Bybit response")?;

    if response.ret_code != 0 {
        anyhow::bail!("Bybit API error: {}", response.ret_msg);
    }

    let klines = &response.result.list;
    if klines.is_empty() {
        anyhow::bail!("No data returned for {}", symbol);
    }

    // Parse klines: [timestamp, open, high, low, close, volume, turnover]
    let mut data = Array2::zeros((klines.len(), 5));

    for (i, kline) in klines.iter().rev().enumerate() {
        if kline.len() >= 6 {
            data[[i, 0]] = kline[1].parse().unwrap_or(0.0); // Open
            data[[i, 1]] = kline[2].parse().unwrap_or(0.0); // High
            data[[i, 2]] = kline[3].parse().unwrap_or(0.0); // Low
            data[[i, 3]] = kline[4].parse().unwrap_or(0.0); // Close
            data[[i, 4]] = kline[5].parse().unwrap_or(0.0); // Volume
        }
    }

    log::info!(
        "Loaded {} candles, price range: ${:.2} - ${:.2}",
        data.shape()[0],
        data.column(2).fold(f32::MAX, |a, &b| a.min(b)),
        data.column(1).fold(f32::MIN, |a, &b| a.max(b))
    );

    Ok(OHLCVData::new(data, symbol, interval))
}

/// Yahoo Finance API response structures
#[derive(Debug, Deserialize)]
struct YahooResponse {
    chart: YahooChart,
}

#[derive(Debug, Deserialize)]
struct YahooChart {
    result: Option<Vec<YahooResult>>,
    error: Option<YahooError>,
}

#[derive(Debug, Deserialize)]
struct YahooResult {
    indicators: YahooIndicators,
}

#[derive(Debug, Deserialize)]
struct YahooIndicators {
    quote: Vec<YahooQuote>,
}

#[derive(Debug, Deserialize)]
struct YahooQuote {
    open: Vec<Option<f32>>,
    high: Vec<Option<f32>>,
    low: Vec<Option<f32>>,
    close: Vec<Option<f32>>,
    volume: Vec<Option<i64>>,
}

#[derive(Debug, Deserialize)]
struct YahooError {
    code: String,
    description: String,
}

/// Load OHLCV data from Yahoo Finance.
///
/// # Arguments
///
/// * `symbol` - Stock ticker symbol (e.g., "AAPL", "GOOGL")
/// * `period` - Data period ("1mo", "3mo", "6mo", "1y", "2y")
/// * `interval` - Data interval ("1d", "1wk", "1mo")
///
/// # Returns
///
/// Result containing OHLCV data or error
pub fn load_yahoo_data(symbol: &str, period: &str, interval: &str) -> Result<OHLCVData> {
    let url = format!(
        "https://query1.finance.yahoo.com/v8/finance/chart/{}?range={}&interval={}",
        symbol, period, interval
    );

    log::info!(
        "Loading {} data from Yahoo Finance ({}, {})...",
        symbol,
        period,
        interval
    );

    let client = reqwest::blocking::Client::new();
    let response: YahooResponse = client
        .get(&url)
        .timeout(std::time::Duration::from_secs(10))
        .send()
        .context("Failed to fetch from Yahoo Finance")?
        .json()
        .context("Failed to parse Yahoo response")?;

    if let Some(error) = response.chart.error {
        anyhow::bail!("Yahoo Finance error: {} - {}", error.code, error.description);
    }

    let result = response
        .chart
        .result
        .and_then(|r| r.into_iter().next())
        .context("No data in Yahoo response")?;

    let quote = result
        .indicators
        .quote
        .into_iter()
        .next()
        .context("No quote data")?;

    let len = quote.close.len();
    let mut data = Array2::zeros((len, 5));

    for i in 0..len {
        data[[i, 0]] = quote.open[i].unwrap_or(0.0);
        data[[i, 1]] = quote.high[i].unwrap_or(0.0);
        data[[i, 2]] = quote.low[i].unwrap_or(0.0);
        data[[i, 3]] = quote.close[i].unwrap_or(0.0);
        data[[i, 4]] = quote.volume[i].map(|v| v as f32).unwrap_or(0.0);
    }

    // Filter out rows with zero close price
    let valid_rows: Vec<_> = (0..len).filter(|&i| data[[i, 3]] > 0.0).collect();
    let mut filtered = Array2::zeros((valid_rows.len(), 5));
    for (new_i, &old_i) in valid_rows.iter().enumerate() {
        for j in 0..5 {
            filtered[[new_i, j]] = data[[old_i, j]];
        }
    }

    log::info!("Loaded {} candles from Yahoo Finance", filtered.shape()[0]);

    Ok(OHLCVData::new(filtered, symbol, interval))
}

/// Generate synthetic OHLCV data for testing.
///
/// # Arguments
///
/// * `length` - Number of candles to generate
/// * `base_price` - Starting price
/// * `volatility` - Daily volatility (e.g., 0.02 for 2%)
pub fn generate_synthetic_data(length: usize, base_price: f32, volatility: f32) -> OHLCVData {
    use rand::Rng;
    use rand_distr::{Distribution, Normal};

    let mut rng = rand::thread_rng();
    let normal = Normal::new(0.0, volatility as f64).unwrap();

    let mut data = Array2::zeros((length, 5));
    let mut price = base_price;

    for i in 0..length {
        // Generate return
        let ret = normal.sample(&mut rng) as f32;
        price *= 1.0 + ret;

        // Generate OHLCV
        let open = price * (1.0 + rng.gen_range(-0.005..0.005));
        let close = price;
        let high = open.max(close) * (1.0 + rng.gen_range(0.0..0.01));
        let low = open.min(close) * (1.0 - rng.gen_range(0.0..0.01));
        let volume = rng.gen_range(100000.0..1000000.0);

        data[[i, 0]] = open;
        data[[i, 1]] = high;
        data[[i, 2]] = low;
        data[[i, 3]] = close;
        data[[i, 4]] = volume;
    }

    OHLCVData::new(data, "SYNTHETIC", "1h")
}

/// Normalize OHLCV data.
///
/// # Arguments
///
/// * `data` - OHLCV data array
/// * `method` - Normalization method ("zscore" or "minmax")
///
/// # Returns
///
/// Tuple of (normalized data, normalization parameters)
pub fn normalize_data(
    data: &Array2<f32>,
    method: &str,
) -> Result<(Array2<f32>, NormParams)> {
    match method {
        "zscore" => {
            let mean = data.mean_axis(ndarray::Axis(0)).unwrap();
            let std = data.std_axis(ndarray::Axis(0), 0.0);

            let mut normalized = data.clone();
            for (j, mut col) in normalized.columns_mut().into_iter().enumerate() {
                let s = if std[j] > 0.0 { std[j] } else { 1.0 };
                col.mapv_inplace(|v| (v - mean[j]) / s);
            }

            Ok((
                normalized,
                NormParams::ZScore {
                    mean: mean.to_vec(),
                    std: std.to_vec(),
                },
            ))
        }
        "minmax" => {
            let min = data.fold_axis(ndarray::Axis(0), f32::MAX, |&a, &b| a.min(b));
            let max = data.fold_axis(ndarray::Axis(0), f32::MIN, |&a, &b| a.max(b));

            let mut normalized = data.clone();
            for (j, mut col) in normalized.columns_mut().into_iter().enumerate() {
                let range = max[j] - min[j];
                let r = if range > 0.0 { range } else { 1.0 };
                col.mapv_inplace(|v| (v - min[j]) / r);
            }

            Ok((
                normalized,
                NormParams::MinMax {
                    min: min.to_vec(),
                    max: max.to_vec(),
                },
            ))
        }
        _ => anyhow::bail!("Unknown normalization method: {}", method),
    }
}

/// Normalization parameters
#[derive(Debug, Clone)]
pub enum NormParams {
    ZScore { mean: Vec<f32>, std: Vec<f32> },
    MinMax { min: Vec<f32>, max: Vec<f32> },
}

/// Prepare sequences for model input.
///
/// # Arguments
///
/// * `data` - OHLCV data array
/// * `seq_len` - Sequence length
/// * `pred_horizon` - Prediction horizon
/// * `threshold` - Classification threshold
///
/// # Returns
///
/// Tuple of (sequences, labels)
pub fn prepare_sequences(
    data: &Array2<f32>,
    seq_len: usize,
    pred_horizon: usize,
    threshold: f32,
) -> Result<(Vec<Array2<f32>>, Vec<usize>)> {
    let n_samples = data.shape()[0];

    if n_samples < seq_len + pred_horizon {
        anyhow::bail!(
            "Not enough data: {} samples for seq_len={}, pred_horizon={}",
            n_samples,
            seq_len,
            pred_horizon
        );
    }

    let n_sequences = n_samples - seq_len - pred_horizon + 1;
    let mut sequences = Vec::with_capacity(n_sequences);
    let mut labels = Vec::with_capacity(n_sequences);

    for i in 0..n_sequences {
        // Extract sequence
        let seq = data.slice(ndarray::s![i..i + seq_len, ..]).to_owned();
        sequences.push(seq);

        // Calculate label
        let current_close = data[[i + seq_len - 1, 3]];
        let future_close = data[[i + seq_len - 1 + pred_horizon, 3]];
        let pct_change = (future_close - current_close) / current_close;

        let label = if pct_change > threshold {
            2 // Up
        } else if pct_change < -threshold {
            0 // Down
        } else {
            1 // Neutral
        };

        labels.push(label);
    }

    Ok((sequences, labels))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synthetic_data() {
        let data = generate_synthetic_data(100, 50000.0, 0.02);
        assert_eq!(data.len(), 100);
        assert!(data.latest_close() > 0.0);
    }

    #[test]
    fn test_normalize() {
        let data = generate_synthetic_data(100, 100.0, 0.02);
        let (normalized, _params) = normalize_data(&data.data, "zscore").unwrap();

        // Mean should be close to 0
        let mean = normalized.mean().unwrap();
        assert!(mean.abs() < 0.1);
    }

    #[test]
    fn test_prepare_sequences() {
        let data = generate_synthetic_data(200, 100.0, 0.02);
        let (sequences, labels) = prepare_sequences(&data.data, 60, 1, 0.0).unwrap();

        assert_eq!(sequences.len(), 140);
        assert_eq!(labels.len(), 140);
        assert!(labels.iter().all(|&l| l < 3));
    }
}
