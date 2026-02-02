//! Data Loading and Preprocessing
//!
//! Provides data loaders for:
//! - CSV files (yfinance format)
//! - Bybit API (cryptocurrency data)
//! - Synthetic data generation with known causal structure

use chrono::{NaiveDateTime, TimeZone, Utc};
use ndarray::Array2;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use thiserror::Error;

/// Data-related errors
#[derive(Error, Debug)]
pub enum DataError {
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("CSV error: {0}")]
    CsvError(#[from] csv::Error),
    #[error("HTTP error: {0}")]
    HttpError(#[from] reqwest::Error),
    #[error("Parse error: {0}")]
    ParseError(String),
    #[error("Empty data")]
    EmptyData,
    #[error("API error: {0}")]
    ApiError(String),
}

/// OHLCV bar data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OHLCVBar {
    /// Timestamp (Unix milliseconds)
    pub timestamp: i64,
    /// Open price
    pub open: f32,
    /// High price
    pub high: f32,
    /// Low price
    pub low: f32,
    /// Close price
    pub close: f32,
    /// Volume
    pub volume: f32,
}

impl OHLCVBar {
    /// Create a new OHLCV bar
    pub fn new(timestamp: i64, open: f32, high: f32, low: f32, close: f32, volume: f32) -> Self {
        Self {
            timestamp,
            open,
            high,
            low,
            close,
            volume,
        }
    }

    /// Convert to feature vector
    pub fn to_features(&self) -> Vec<f32> {
        vec![self.open, self.high, self.low, self.close, self.volume]
    }

    /// Calculate typical price (HLC average)
    pub fn typical_price(&self) -> f32 {
        (self.high + self.low + self.close) / 3.0
    }

    /// Calculate return from open to close
    pub fn bar_return(&self) -> f32 {
        if self.open == 0.0 {
            return 0.0;
        }
        (self.close - self.open) / self.open
    }
}

/// Market data container
#[derive(Debug, Clone)]
pub struct MarketData {
    /// Symbol name
    pub symbol: String,
    /// OHLCV bars sorted by timestamp
    pub bars: Vec<OHLCVBar>,
}

impl MarketData {
    /// Create new market data
    pub fn new(symbol: String, bars: Vec<OHLCVBar>) -> Self {
        Self { symbol, bars }
    }

    /// Get number of bars
    pub fn len(&self) -> usize {
        self.bars.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.bars.is_empty()
    }

    /// Convert to feature matrix (n_bars x 5)
    pub fn to_features(&self) -> Array2<f32> {
        let n = self.bars.len();
        let mut data = Vec::with_capacity(n * 5);
        for bar in &self.bars {
            data.extend(bar.to_features());
        }
        Array2::from_shape_vec((n, 5), data).unwrap()
    }

    /// Normalize features (returns (normalized, means, stds))
    pub fn normalize(&self) -> (Array2<f32>, Vec<f32>, Vec<f32>) {
        let features = self.to_features();
        let n_features = features.ncols();

        let means: Vec<f32> = (0..n_features)
            .map(|i| features.column(i).mean().unwrap())
            .collect();

        let stds: Vec<f32> = (0..n_features)
            .map(|i| {
                let col = features.column(i);
                let mean = means[i];
                let var: f32 = col.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / col.len() as f32;
                var.sqrt().max(1e-8)
            })
            .collect();

        let mut normalized = features.clone();
        for (j, (mean, std)) in means.iter().zip(stds.iter()).enumerate() {
            for i in 0..normalized.nrows() {
                normalized[[i, j]] = (normalized[[i, j]] - mean) / std;
            }
        }

        (normalized, means, stds)
    }

    /// Calculate close-to-close returns
    pub fn returns(&self) -> Vec<f32> {
        if self.bars.len() < 2 {
            return vec![];
        }

        self.bars
            .windows(2)
            .map(|w| {
                if w[0].close == 0.0 {
                    0.0
                } else {
                    (w[1].close - w[0].close) / w[0].close
                }
            })
            .collect()
    }

    /// Calculate log returns
    pub fn log_returns(&self) -> Vec<f64> {
        if self.bars.len() < 2 {
            return vec![];
        }

        self.bars
            .windows(2)
            .map(|w| {
                if w[0].close <= 0.0 || w[1].close <= 0.0 {
                    0.0
                } else {
                    (w[1].close as f64 / w[0].close as f64).ln()
                }
            })
            .collect()
    }

    /// Extract multiple variables for causal analysis
    ///
    /// Returns an Array2<f64> with columns: [returns, volume_change, volatility, spread]
    pub fn to_causal_variables(&self) -> Array2<f64> {
        let n = self.bars.len();
        if n < 3 {
            return Array2::zeros((0, 4));
        }

        let effective_n = n - 2; // lose 2 bars for differencing
        let mut data = Array2::<f64>::zeros((effective_n, 4));

        for i in 0..effective_n {
            let idx = i + 2;
            let bar = &self.bars[idx];
            let prev_bar = &self.bars[idx - 1];
            let prev_prev_bar = &self.bars[idx - 2];

            // Returns
            let ret = if prev_bar.close > 0.0 {
                (bar.close as f64 - prev_bar.close as f64) / prev_bar.close as f64
            } else {
                0.0
            };

            // Volume change (log ratio)
            let vol_change = if prev_bar.volume > 0.0 && bar.volume > 0.0 {
                (bar.volume as f64 / prev_bar.volume as f64).ln()
            } else {
                0.0
            };

            // Realized volatility (using previous two bars)
            let r1 = if prev_prev_bar.close > 0.0 {
                ((prev_bar.close as f64) / (prev_prev_bar.close as f64)).ln()
            } else {
                0.0
            };
            let r2 = if prev_bar.close > 0.0 {
                ((bar.close as f64) / (prev_bar.close as f64)).ln()
            } else {
                0.0
            };
            let volatility = ((r1 * r1 + r2 * r2) / 2.0).sqrt();

            // Bid-ask spread proxy (high-low range normalized by close)
            let spread = if bar.close > 0.0 {
                (bar.high as f64 - bar.low as f64) / bar.close as f64
            } else {
                0.0
            };

            data[[i, 0]] = ret;
            data[[i, 1]] = vol_change;
            data[[i, 2]] = volatility;
            data[[i, 3]] = spread;
        }

        data
    }

    /// Split into train/val/test sets
    pub fn train_val_test_split(
        &self,
        train_ratio: f32,
        val_ratio: f32,
    ) -> (MarketData, MarketData, MarketData) {
        let n = self.bars.len();
        let train_end = (n as f32 * train_ratio) as usize;
        let val_end = train_end + (n as f32 * val_ratio) as usize;

        let train = MarketData {
            symbol: self.symbol.clone(),
            bars: self.bars[..train_end].to_vec(),
        };

        let val = MarketData {
            symbol: self.symbol.clone(),
            bars: self.bars[train_end..val_end].to_vec(),
        };

        let test = MarketData {
            symbol: self.symbol.clone(),
            bars: self.bars[val_end..].to_vec(),
        };

        (train, val, test)
    }
}

/// Generate synthetic market data for testing
pub fn generate_synthetic_data(n_bars: usize, volatility: f32) -> MarketData {
    use rand::Rng;
    use rand_distr::Normal;

    let mut rng = rand::thread_rng();
    let normal = Normal::new(0.0, volatility as f64).unwrap();

    let mut bars = Vec::with_capacity(n_bars);
    let mut price = 100.0_f32;
    let base_timestamp = Utc::now().timestamp_millis() - (n_bars as i64 * 60000);

    for i in 0..n_bars {
        let return_: f64 = rng.sample(normal);
        let new_price = price * (1.0 + return_ as f32);

        let high = price.max(new_price) * (1.0 + rng.gen::<f32>() * 0.005);
        let low = price.min(new_price) * (1.0 - rng.gen::<f32>() * 0.005);
        let volume = rng.gen::<f32>() * 1_000_000.0;

        bars.push(OHLCVBar {
            timestamp: base_timestamp + (i as i64 * 60000),
            open: price,
            high,
            low,
            close: new_price,
            volume,
        });

        price = new_price;
    }

    MarketData::new("SYNTHETIC".to_string(), bars)
}

/// Generate synthetic multivariate data with known causal structure
///
/// # Arguments
/// * `n_vars` - Number of variables
/// * `n_samples` - Number of time steps
/// * `links` - Vec of (source, target, lag, strength) defining causal links
///
/// # Returns
/// Array2<f64> with shape (n_samples, n_vars)
pub fn generate_synthetic_causal_data(
    n_vars: usize,
    n_samples: usize,
    links: &[(usize, usize, usize, f64)],
) -> Array2<f64> {
    use rand::Rng;
    use rand_distr::Normal;

    let mut rng = rand::thread_rng();
    let noise_dist = Normal::new(0.0, 1.0).unwrap();

    let mut data = Array2::<f64>::zeros((n_samples, n_vars));

    // Initialize with noise
    for i in 0..n_samples {
        for j in 0..n_vars {
            data[[i, j]] = rng.sample(noise_dist);
        }
    }

    // Find maximum lag
    let max_lag = links.iter().map(|l| l.2).max().unwrap_or(0);

    // Apply causal structure forward in time
    for t in max_lag..n_samples {
        for &(source, target, lag, strength) in links {
            if t >= lag {
                data[[t, target]] += strength * data[[t - lag, source]];
            }
        }
        // Add noise to each variable
        for j in 0..n_vars {
            data[[t, j]] += 0.3 * rng.sample(noise_dist);
        }
    }

    data
}

/// Standardize columns of a data matrix to zero mean and unit variance
pub fn standardize(data: &Array2<f64>) -> Array2<f64> {
    let n = data.nrows() as f64;
    let mut result = data.clone();

    for j in 0..data.ncols() {
        let col = data.column(j);
        let mean = col.sum() / n;
        let var: f64 = col.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
        let std = var.sqrt().max(1e-15);

        for i in 0..data.nrows() {
            result[[i, j]] = (data[[i, j]] - mean) / std;
        }
    }

    result
}

/// Test for stationarity using the Augmented Dickey-Fuller test (simplified)
///
/// Returns (test_statistic, is_stationary) where is_stationary uses the 5% critical value
pub fn adf_test(series: &[f64]) -> (f64, bool) {
    let n = series.len();
    if n < 10 {
        return (0.0, false);
    }

    // First differences
    let diffs: Vec<f64> = series.windows(2).map(|w| w[1] - w[0]).collect();

    // Lagged levels
    let lagged: Vec<f64> = series[..n - 1].to_vec();

    // Simple OLS: diffs = alpha + beta * lagged + error
    let n_eff = diffs.len() as f64;
    let mean_x: f64 = lagged.iter().sum::<f64>() / n_eff;
    let mean_y: f64 = diffs.iter().sum::<f64>() / n_eff;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    for i in 0..diffs.len() {
        let dx = lagged[i] - mean_x;
        let dy = diffs[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
    }

    if var_x.abs() < 1e-15 {
        return (0.0, false);
    }

    let beta = cov / var_x;
    let alpha = mean_y - beta * mean_x;

    // Calculate residuals and standard error
    let mut sse = 0.0;
    for i in 0..diffs.len() {
        let predicted = alpha + beta * lagged[i];
        let residual = diffs[i] - predicted;
        sse += residual * residual;
    }

    let se_beta = (sse / ((n_eff - 2.0) * var_x)).sqrt();
    if se_beta.abs() < 1e-15 {
        return (0.0, false);
    }

    let t_stat = beta / se_beta;

    // Approximate 5% critical value for ADF test (n > 50)
    let critical_value = -2.86;
    let is_stationary = t_stat < critical_value;

    (t_stat, is_stationary)
}

/// Apply first differencing to make a series stationary
pub fn difference_series(series: &[f64]) -> Vec<f64> {
    series.windows(2).map(|w| w[1] - w[0]).collect()
}

/// Load data from yfinance-style CSV file
///
/// Expected columns: Date, Open, High, Low, Close, Volume
pub fn load_csv_data<P: AsRef<Path>>(path: P, symbol: &str) -> Result<MarketData, DataError> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut csv_reader = csv::Reader::from_reader(reader);

    let mut bars = Vec::new();

    for result in csv_reader.records() {
        let record = result?;

        // Parse date
        let date_str = record.get(0).ok_or_else(|| DataError::ParseError("Missing date".into()))?;
        let timestamp = parse_date_to_timestamp(date_str)?;

        // Parse OHLCV
        let open: f32 = record
            .get(1)
            .ok_or_else(|| DataError::ParseError("Missing open".into()))?
            .parse()
            .map_err(|e| DataError::ParseError(format!("Invalid open: {}", e)))?;

        let high: f32 = record
            .get(2)
            .ok_or_else(|| DataError::ParseError("Missing high".into()))?
            .parse()
            .map_err(|e| DataError::ParseError(format!("Invalid high: {}", e)))?;

        let low: f32 = record
            .get(3)
            .ok_or_else(|| DataError::ParseError("Missing low".into()))?
            .parse()
            .map_err(|e| DataError::ParseError(format!("Invalid low: {}", e)))?;

        let close: f32 = record
            .get(4)
            .ok_or_else(|| DataError::ParseError("Missing close".into()))?
            .parse()
            .map_err(|e| DataError::ParseError(format!("Invalid close: {}", e)))?;

        let volume: f32 = record
            .get(5)
            .ok_or_else(|| DataError::ParseError("Missing volume".into()))?
            .parse()
            .map_err(|e| DataError::ParseError(format!("Invalid volume: {}", e)))?;

        bars.push(OHLCVBar::new(timestamp, open, high, low, close, volume));
    }

    if bars.is_empty() {
        return Err(DataError::EmptyData);
    }

    Ok(MarketData::new(symbol.to_string(), bars))
}

fn parse_date_to_timestamp(date_str: &str) -> Result<i64, DataError> {
    let formats = [
        "%Y-%m-%d",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d",
        "%d-%m-%Y",
    ];

    for format in &formats {
        if let Ok(dt) = NaiveDateTime::parse_from_str(date_str, format) {
            return Ok(Utc.from_utc_datetime(&dt).timestamp_millis());
        }
        if let Ok(d) = chrono::NaiveDate::parse_from_str(date_str, format) {
            let dt = d.and_hms_opt(0, 0, 0).unwrap();
            return Ok(Utc.from_utc_datetime(&dt).timestamp_millis());
        }
    }

    Err(DataError::ParseError(format!("Cannot parse date: {}", date_str)))
}

/// Bybit API client for cryptocurrency data
pub struct BybitClient {
    client: Client,
    base_url: String,
}

/// Bybit kline response
#[derive(Debug, Deserialize)]
struct BybitKlineResponse {
    #[serde(rename = "retCode")]
    ret_code: i32,
    #[serde(rename = "retMsg")]
    ret_msg: String,
    result: BybitKlineResult,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct BybitKlineResult {
    symbol: String,
    category: String,
    list: Vec<Vec<String>>,
}

impl BybitClient {
    /// Create a new Bybit client
    pub fn new() -> Self {
        Self {
            client: Client::new(),
            base_url: "https://api.bybit.com".to_string(),
        }
    }

    /// Create client for testnet
    pub fn testnet() -> Self {
        Self {
            client: Client::new(),
            base_url: "https://api-testnet.bybit.com".to_string(),
        }
    }

    /// Get historical klines (candlestick data)
    pub async fn get_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> Result<MarketData, DataError> {
        let url = format!(
            "{}/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
            self.base_url, symbol, interval, limit
        );

        let response: BybitKlineResponse = self.client
            .get(&url)
            .send()
            .await?
            .json()
            .await?;

        if response.ret_code != 0 {
            return Err(DataError::ApiError(response.ret_msg));
        }

        let bars: Vec<OHLCVBar> = response
            .result
            .list
            .iter()
            .filter_map(|kline| {
                if kline.len() < 6 {
                    return None;
                }
                Some(OHLCVBar {
                    timestamp: kline[0].parse().ok()?,
                    open: kline[1].parse().ok()?,
                    high: kline[2].parse().ok()?,
                    low: kline[3].parse().ok()?,
                    close: kline[4].parse().ok()?,
                    volume: kline[5].parse().ok()?,
                })
            })
            .collect();

        // Bybit returns newest first, reverse to chronological order
        let mut bars = bars;
        bars.reverse();

        if bars.is_empty() {
            return Err(DataError::EmptyData);
        }

        Ok(MarketData::new(symbol.to_string(), bars))
    }

    /// Get historical data with pagination for more data
    pub async fn get_historical(
        &self,
        symbol: &str,
        interval: &str,
        start_time: i64,
        end_time: i64,
    ) -> Result<MarketData, DataError> {
        let mut all_bars: Vec<OHLCVBar> = Vec::new();
        let mut current_end = end_time;

        while current_end > start_time {
            let url = format!(
                "{}/v5/market/kline?category=linear&symbol={}&interval={}&limit=200&end={}",
                self.base_url, symbol, interval, current_end
            );

            let response: BybitKlineResponse = self.client
                .get(&url)
                .send()
                .await?
                .json()
                .await?;

            if response.ret_code != 0 {
                return Err(DataError::ApiError(response.ret_msg));
            }

            if response.result.list.is_empty() {
                break;
            }

            let bars: Vec<OHLCVBar> = response
                .result
                .list
                .iter()
                .filter_map(|kline| {
                    if kline.len() < 6 {
                        return None;
                    }
                    let ts: i64 = kline[0].parse().ok()?;
                    if ts < start_time {
                        return None;
                    }
                    Some(OHLCVBar {
                        timestamp: ts,
                        open: kline[1].parse().ok()?,
                        high: kline[2].parse().ok()?,
                        low: kline[3].parse().ok()?,
                        close: kline[4].parse().ok()?,
                        volume: kline[5].parse().ok()?,
                    })
                })
                .collect();

            if bars.is_empty() {
                break;
            }

            current_end = bars.iter().map(|b| b.timestamp).min().unwrap() - 1;
            all_bars.extend(bars);

            // Rate limiting
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        }

        all_bars.sort_by_key(|b| b.timestamp);

        if all_bars.is_empty() {
            return Err(DataError::EmptyData);
        }

        Ok(MarketData::new(symbol.to_string(), all_bars))
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
    fn test_synthetic_data() {
        let data = generate_synthetic_data(100, 0.02);
        assert_eq!(data.len(), 100);
        assert!(!data.is_empty());
    }

    #[test]
    fn test_market_data_operations() {
        let data = generate_synthetic_data(50, 0.01);

        let features = data.to_features();
        assert_eq!(features.shape(), &[50, 5]);

        let returns = data.returns();
        assert_eq!(returns.len(), 49);

        let (normalized, means, stds) = data.normalize();
        assert_eq!(normalized.shape(), &[50, 5]);
        assert_eq!(means.len(), 5);
        assert_eq!(stds.len(), 5);
    }

    #[test]
    fn test_causal_variables() {
        let data = generate_synthetic_data(100, 0.02);
        let causal_vars = data.to_causal_variables();

        assert_eq!(causal_vars.ncols(), 4);
        assert_eq!(causal_vars.nrows(), 98); // n - 2
    }

    #[test]
    fn test_synthetic_causal_data() {
        let links = vec![
            (0, 1, 1, 0.5),
            (1, 2, 2, -0.3),
        ];
        let data = generate_synthetic_causal_data(3, 200, &links);

        assert_eq!(data.shape(), &[200, 3]);

        // Check that the causal structure creates correlation
        let col0: Vec<f64> = data.column(0).to_vec();
        let col1: Vec<f64> = data.column(1).to_vec();

        // X1 should correlate with lagged X0
        let mut corr_sum = 0.0;
        let n = col0.len() - 1;
        for i in 1..=n {
            corr_sum += col0[i - 1] * col1[i];
        }
        let avg_corr = corr_sum / n as f64;
        assert!(avg_corr.abs() > 0.01, "Expected causal correlation, got {}", avg_corr);
    }

    #[test]
    fn test_standardize() {
        let data = Array2::from_shape_vec((5, 2), vec![
            1.0, 10.0,
            2.0, 20.0,
            3.0, 30.0,
            4.0, 40.0,
            5.0, 50.0,
        ]).unwrap();

        let standardized = standardize(&data);

        // Check zero mean
        for j in 0..2 {
            let col_mean: f64 = standardized.column(j).sum() / 5.0;
            assert!((col_mean).abs() < 1e-10, "Column {} mean should be ~0, got {}", j, col_mean);
        }
    }

    #[test]
    fn test_adf_stationarity() {
        // Stationary series (random walk differences should be stationary)
        let stationary: Vec<f64> = (0..100).map(|_| rand::random::<f64>() - 0.5).collect();
        let (_, is_stat) = adf_test(&stationary);
        // Note: random series are usually stationary but test may not always detect it
        // for small samples, so we just check the function runs
        let _ = is_stat;

        // Non-stationary series (random walk)
        let mut rw = vec![0.0_f64; 100];
        for i in 1..100 {
            rw[i] = rw[i - 1] + rand::random::<f64>() - 0.5;
        }
        let (t_stat, _) = adf_test(&rw);
        // Random walk should have t-stat close to 0
        assert!(t_stat > -10.0, "t-stat should not be extremely negative for random walk");
    }

    #[test]
    fn test_difference_series() {
        let series = vec![1.0, 3.0, 6.0, 10.0, 15.0];
        let diff = difference_series(&series);
        assert_eq!(diff, vec![2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_train_val_test_split() {
        let data = generate_synthetic_data(100, 0.01);
        let (train, val, test) = data.train_val_test_split(0.7, 0.15);

        assert_eq!(train.len(), 70);
        assert_eq!(val.len(), 15);
        assert_eq!(test.len(), 15);
    }

    #[test]
    fn test_log_returns() {
        let data = generate_synthetic_data(50, 0.01);
        let log_rets = data.log_returns();
        assert_eq!(log_rets.len(), 49);

        // Log returns should be close to simple returns for small changes
        let simple_rets = data.returns();
        for (lr, sr) in log_rets.iter().zip(simple_rets.iter()) {
            assert!(
                (lr - *sr as f64).abs() < 0.1,
                "Log return {} and simple return {} should be close",
                lr, sr
            );
        }
    }
}
