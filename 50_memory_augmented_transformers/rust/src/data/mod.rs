//! Data Loading and Preprocessing
//!
//! Provides data loaders for:
//! - CSV files (yfinance format)
//! - Bybit API (cryptocurrency data)

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

    /// Convert to feature matrix
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

    /// Calculate returns
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

    /// Create sequences for training
    pub fn create_sequences(
        &self,
        seq_len: usize,
        horizon: usize,
    ) -> (Vec<Array2<f32>>, Vec<f32>) {
        let (normalized, _, _) = self.normalize();
        let returns = self.returns();

        let mut sequences = Vec::new();
        let mut targets = Vec::new();

        // Need seq_len history + horizon future
        if normalized.nrows() < seq_len + horizon {
            return (sequences, targets);
        }

        for i in 0..(normalized.nrows() - seq_len - horizon + 1) {
            let seq = normalized.slice(ndarray::s![i..i + seq_len, ..]).to_owned();
            sequences.push(seq);

            // Target: sum of future returns
            let future_return: f32 = returns[i + seq_len - 1..i + seq_len - 1 + horizon]
                .iter()
                .sum();
            targets.push(future_return);
        }

        (sequences, targets)
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
    // Try different date formats
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
        // Try date-only format
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

            // Get the earliest timestamp from this batch
            current_end = bars.iter().map(|b| b.timestamp).min().unwrap() - 1;
            all_bars.extend(bars);

            // Rate limiting
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        }

        // Sort by timestamp
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
    fn test_create_sequences() {
        let data = generate_synthetic_data(100, 0.01);
        let (sequences, targets) = data.create_sequences(10, 5);

        assert!(!sequences.is_empty());
        assert_eq!(sequences.len(), targets.len());
        assert_eq!(sequences[0].shape(), &[10, 5]);
    }

    #[test]
    fn test_train_val_test_split() {
        let data = generate_synthetic_data(100, 0.01);
        let (train, val, test) = data.train_val_test_split(0.7, 0.15);

        assert_eq!(train.len(), 70);
        assert_eq!(val.len(), 15);
        assert_eq!(test.len(), 15);
    }
}
