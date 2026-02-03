//! Data Loading Module
//!
//! This module provides utilities for loading market data from various sources,
//! including the Bybit API for cryptocurrency data.

use chrono::{DateTime, Utc};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

/// Market data (OHLCV)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketData {
    pub timestamp: Vec<DateTime<Utc>>,
    pub open: Vec<f64>,
    pub high: Vec<f64>,
    pub low: Vec<f64>,
    pub close: Vec<f64>,
    pub volume: Vec<f64>,
}

impl MarketData {
    /// Create new empty market data
    pub fn new() -> Self {
        Self {
            timestamp: Vec::new(),
            open: Vec::new(),
            high: Vec::new(),
            low: Vec::new(),
            close: Vec::new(),
            volume: Vec::new(),
        }
    }

    /// Get the number of data points
    pub fn len(&self) -> usize {
        self.close.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.close.is_empty()
    }
}

impl Default for MarketData {
    fn default() -> Self {
        Self::new()
    }
}

/// Technical features computed from market data
#[derive(Debug, Clone)]
pub struct TechnicalFeatures {
    pub feature_names: Vec<String>,
    pub features: Array2<f64>,
    pub labels: Option<Array1<f64>>,
}

impl TechnicalFeatures {
    /// Compute technical features from market data
    pub fn compute(data: &MarketData, target_periods: usize) -> crate::Result<Self> {
        if data.len() < 200 {
            return Err(crate::DistillationError::DataError(
                "Need at least 200 data points".to_string(),
            ));
        }

        let n = data.len();
        let close = &data.close;
        let high = &data.high;
        let low = &data.low;
        let volume = &data.volume;

        // Feature names
        let feature_names = vec![
            "rsi_14".to_string(),
            "macd".to_string(),
            "macd_signal".to_string(),
            "macd_histogram".to_string(),
            "bb_position_20".to_string(),
            "atr_14".to_string(),
            "adx_14".to_string(),
            "volume_ratio".to_string(),
            "roc_10".to_string(),
            "volatility_20".to_string(),
        ];

        let n_features = feature_names.len();
        let valid_start = 200; // Skip initial rows due to lookback
        let valid_len = n - valid_start - target_periods;

        let mut features = Array2::<f64>::zeros((valid_len, n_features));

        // Compute RSI
        let rsi = compute_rsi(close, 14);

        // Compute MACD
        let ema_12 = compute_ema(close, 12);
        let ema_26 = compute_ema(close, 26);
        let macd: Vec<f64> = ema_12.iter().zip(&ema_26).map(|(a, b)| a - b).collect();
        let macd_signal = compute_ema(&macd, 9);
        let macd_histogram: Vec<f64> = macd
            .iter()
            .zip(&macd_signal)
            .map(|(a, b)| a - b)
            .collect();

        // Compute Bollinger Bands position
        let bb_position = compute_bb_position(close, 20);

        // Compute ATR
        let atr = compute_atr(high, low, close, 14);

        // Compute ADX
        let adx = compute_adx(high, low, close, 14);

        // Compute volume ratio
        let volume_sma = compute_sma(volume, 20);
        let volume_ratio: Vec<f64> = volume
            .iter()
            .zip(&volume_sma)
            .map(|(v, s)| if *s > 0.0 { v / s } else { 1.0 })
            .collect();

        // Compute ROC
        let roc = compute_roc(close, 10);

        // Compute volatility
        let volatility = compute_volatility(close, 20);

        // Fill feature matrix
        for i in 0..valid_len {
            let idx = valid_start + i;
            features[[i, 0]] = rsi[idx];
            features[[i, 1]] = macd[idx];
            features[[i, 2]] = macd_signal[idx];
            features[[i, 3]] = macd_histogram[idx];
            features[[i, 4]] = bb_position[idx];
            features[[i, 5]] = atr[idx];
            features[[i, 6]] = adx[idx];
            features[[i, 7]] = volume_ratio[idx];
            features[[i, 8]] = roc[idx];
            features[[i, 9]] = volatility[idx];
        }

        // Compute labels (future returns)
        let mut labels = Array1::<f64>::zeros(valid_len);
        for i in 0..valid_len {
            let idx = valid_start + i;
            let future_return = (close[idx + target_periods] - close[idx]) / close[idx];
            labels[i] = if future_return > 0.0 { 1.0 } else { 0.0 };
        }

        Ok(Self {
            feature_names,
            features,
            labels: Some(labels),
        })
    }
}

/// Bybit API client
pub struct BybitClient {
    base_url: String,
    client: reqwest::Client,
}

impl BybitClient {
    /// Create a new Bybit client
    pub fn new() -> Self {
        Self {
            base_url: "https://api.bybit.com".to_string(),
            client: reqwest::Client::new(),
        }
    }

    /// Fetch kline (candlestick) data
    pub async fn get_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> crate::Result<MarketData> {
        let url = format!("{}/v5/market/kline", self.base_url);

        let params = [
            ("category", "spot"),
            ("symbol", symbol),
            ("interval", interval),
            ("limit", &limit.to_string()),
        ];

        let response = self
            .client
            .get(&url)
            .query(&params)
            .send()
            .await?
            .json::<BybitResponse>()
            .await?;

        if response.ret_code != 0 {
            return Err(crate::DistillationError::ApiError(response.ret_msg));
        }

        let mut data = MarketData::new();

        for kline in response.result.list.into_iter().rev() {
            // Reverse to get chronological order
            let timestamp = DateTime::from_timestamp_millis(kline.0.parse::<i64>().unwrap_or(0))
                .unwrap_or_else(|| Utc::now());

            data.timestamp.push(timestamp);
            data.open.push(kline.1.parse().unwrap_or(0.0));
            data.high.push(kline.2.parse().unwrap_or(0.0));
            data.low.push(kline.3.parse().unwrap_or(0.0));
            data.close.push(kline.4.parse().unwrap_or(0.0));
            data.volume.push(kline.5.parse().unwrap_or(0.0));
        }

        Ok(data)
    }
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

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
    list: Vec<(String, String, String, String, String, String, String)>,
}

// Technical indicator helper functions

fn compute_sma(data: &[f64], period: usize) -> Vec<f64> {
    let mut result = vec![0.0; data.len()];
    for i in period - 1..data.len() {
        let sum: f64 = data[i + 1 - period..=i].iter().sum();
        result[i] = sum / period as f64;
    }
    result
}

fn compute_ema(data: &[f64], period: usize) -> Vec<f64> {
    let mut result = vec![0.0; data.len()];
    let multiplier = 2.0 / (period as f64 + 1.0);

    // Initialize with SMA
    if data.len() >= period {
        result[period - 1] = data[..period].iter().sum::<f64>() / period as f64;

        for i in period..data.len() {
            result[i] = (data[i] - result[i - 1]) * multiplier + result[i - 1];
        }
    }

    result
}

fn compute_rsi(close: &[f64], period: usize) -> Vec<f64> {
    let mut result = vec![50.0; close.len()];

    if close.len() < period + 1 {
        return result;
    }

    let mut gains = Vec::with_capacity(close.len());
    let mut losses = Vec::with_capacity(close.len());

    gains.push(0.0);
    losses.push(0.0);

    for i in 1..close.len() {
        let change = close[i] - close[i - 1];
        gains.push(if change > 0.0 { change } else { 0.0 });
        losses.push(if change < 0.0 { -change } else { 0.0 });
    }

    let avg_gain = compute_ema(&gains, period);
    let avg_loss = compute_ema(&losses, period);

    for i in period..close.len() {
        if avg_loss[i] > 0.0 {
            let rs = avg_gain[i] / avg_loss[i];
            result[i] = 100.0 - (100.0 / (1.0 + rs));
        } else {
            result[i] = 100.0;
        }
    }

    result
}

fn compute_bb_position(close: &[f64], period: usize) -> Vec<f64> {
    let mut result = vec![0.5; close.len()];
    let sma = compute_sma(close, period);

    for i in period - 1..close.len() {
        let slice = &close[i + 1 - period..=i];
        let mean = sma[i];
        let variance: f64 = slice.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / period as f64;
        let std = variance.sqrt();

        let upper = mean + 2.0 * std;
        let lower = mean - 2.0 * std;

        if upper > lower {
            result[i] = (close[i] - lower) / (upper - lower);
        }
    }

    result
}

fn compute_atr(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<f64> {
    let mut tr = vec![0.0; close.len()];

    for i in 1..close.len() {
        let hl = high[i] - low[i];
        let hc = (high[i] - close[i - 1]).abs();
        let lc = (low[i] - close[i - 1]).abs();
        tr[i] = hl.max(hc).max(lc);
    }

    compute_ema(&tr, period)
}

fn compute_adx(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<f64> {
    let n = close.len();
    let mut result = vec![25.0; n];

    if n < period * 2 {
        return result;
    }

    let mut plus_dm = vec![0.0; n];
    let mut minus_dm = vec![0.0; n];

    for i in 1..n {
        let up = high[i] - high[i - 1];
        let down = low[i - 1] - low[i];

        plus_dm[i] = if up > down && up > 0.0 { up } else { 0.0 };
        minus_dm[i] = if down > up && down > 0.0 { down } else { 0.0 };
    }

    let atr = compute_atr(high, low, close, period);
    let plus_di_smooth = compute_ema(&plus_dm, period);
    let minus_di_smooth = compute_ema(&minus_dm, period);

    let mut dx = vec![0.0; n];

    for i in period..n {
        if atr[i] > 0.0 {
            let plus_di = 100.0 * plus_di_smooth[i] / atr[i];
            let minus_di = 100.0 * minus_di_smooth[i] / atr[i];
            let di_sum = plus_di + minus_di;
            if di_sum > 0.0 {
                dx[i] = 100.0 * (plus_di - minus_di).abs() / di_sum;
            }
        }
    }

    let adx = compute_ema(&dx, period);

    for i in period * 2..n {
        result[i] = adx[i];
    }

    result
}

fn compute_roc(close: &[f64], period: usize) -> Vec<f64> {
    let mut result = vec![0.0; close.len()];

    for i in period..close.len() {
        if close[i - period] > 0.0 {
            result[i] = ((close[i] - close[i - period]) / close[i - period]) * 100.0;
        }
    }

    result
}

fn compute_volatility(close: &[f64], period: usize) -> Vec<f64> {
    let mut returns = vec![0.0; close.len()];

    for i in 1..close.len() {
        if close[i - 1] > 0.0 {
            returns[i] = (close[i] / close[i - 1]).ln();
        }
    }

    let mut result = vec![0.0; close.len()];

    for i in period..close.len() {
        let slice = &returns[i + 1 - period..=i];
        let mean: f64 = slice.iter().sum::<f64>() / period as f64;
        let variance: f64 = slice.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / period as f64;
        result[i] = variance.sqrt() * (252.0_f64).sqrt(); // Annualized
    }

    result
}

/// Generate synthetic market data for testing
pub fn generate_synthetic_data(n_points: usize, base_price: f64) -> MarketData {
    use rand::Rng;
    use rand_distr::{Distribution, Normal};

    let mut rng = rand::thread_rng();
    let normal = Normal::new(0.0001, 0.02).unwrap();

    let mut data = MarketData::new();
    let mut price = base_price;

    let start_time = Utc::now() - chrono::Duration::hours(n_points as i64);

    for i in 0..n_points {
        let ret = normal.sample(&mut rng);
        price *= 1.0 + ret;

        let high = price * (1.0 + rng.gen_range(0.0..0.015));
        let low = price * (1.0 - rng.gen_range(0.0..0.015));
        let open = price * (1.0 + rng.gen_range(-0.005..0.005));

        data.timestamp
            .push(start_time + chrono::Duration::hours(i as i64));
        data.open.push(open);
        data.high.push(high);
        data.low.push(low);
        data.close.push(price);
        data.volume.push(rng.gen_range(100.0..10000.0));
    }

    data
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synthetic_data() {
        let data = generate_synthetic_data(500, 50000.0);
        assert_eq!(data.len(), 500);
    }

    #[test]
    fn test_technical_features() {
        let data = generate_synthetic_data(500, 50000.0);
        let features = TechnicalFeatures::compute(&data, 1).unwrap();

        assert!(!features.feature_names.is_empty());
        assert!(features.features.nrows() > 0);
        assert!(features.labels.is_some());
    }

    #[test]
    fn test_rsi() {
        let close: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let rsi = compute_rsi(&close, 14);
        assert_eq!(rsi.len(), 100);
    }
}
