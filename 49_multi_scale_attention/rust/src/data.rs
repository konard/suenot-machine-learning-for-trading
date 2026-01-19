//! Data loading and preprocessing for multi-scale attention.
//!
//! This module provides utilities for:
//! - Fetching data from Bybit API
//! - Creating multi-scale features
//! - Generating sequences for training

use anyhow::{Context, Result};
use ndarray::{Array1, Array2, Array3, s};
use ndarray_rand::RandomExt;
use rand_distr::Normal;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Bybit kline (candlestick) data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BybitKline {
    pub timestamp: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub turnover: f64,
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

/// Multi-scale data container
#[derive(Debug, Clone)]
pub struct MultiScaleData {
    /// Features for each scale: scale_name -> [n_samples, n_features]
    pub features: HashMap<String, Array2<f64>>,
    /// Timestamps aligned to the base scale
    pub timestamps: Vec<i64>,
    /// Raw OHLCV data
    pub ohlcv: Option<Vec<BybitKline>>,
}

/// Fetch klines from Bybit API
///
/// # Arguments
/// * `symbol` - Trading pair (e.g., "BTCUSDT")
/// * `interval` - Kline interval ("1", "3", "5", "15", "30", "60", "120", "240", "360", "720", "D", "W", "M")
/// * `limit` - Number of klines to fetch (max 1000)
///
/// # Returns
/// Vector of BybitKline data
pub fn fetch_bybit_klines(
    symbol: &str,
    interval: &str,
    limit: usize,
) -> Result<Vec<BybitKline>> {
    let url = format!(
        "https://api.bybit.com/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
        symbol, interval, limit.min(1000)
    );

    let client = reqwest::blocking::Client::new();
    let response: BybitResponse = client
        .get(&url)
        .send()
        .context("Failed to send request to Bybit API")?
        .json()
        .context("Failed to parse Bybit API response")?;

    if response.ret_code != 0 {
        anyhow::bail!("Bybit API error: {}", response.ret_msg);
    }

    let klines: Vec<BybitKline> = response
        .result
        .list
        .into_iter()
        .map(|row| BybitKline {
            timestamp: row[0].parse().unwrap_or(0),
            open: row[1].parse().unwrap_or(0.0),
            high: row[2].parse().unwrap_or(0.0),
            low: row[3].parse().unwrap_or(0.0),
            close: row[4].parse().unwrap_or(0.0),
            volume: row[5].parse().unwrap_or(0.0),
            turnover: row[6].parse().unwrap_or(0.0),
        })
        .collect();

    // Bybit returns data in descending order, reverse to get ascending
    let mut klines = klines;
    klines.reverse();

    Ok(klines)
}

/// Fetch extended kline data by making multiple API calls
pub fn fetch_bybit_klines_extended(
    symbol: &str,
    interval: &str,
    total_limit: usize,
    verbose: bool,
) -> Result<Vec<BybitKline>> {
    let mut all_klines = Vec::new();
    let batch_size = 1000;
    let mut end_time: Option<i64> = None;

    while all_klines.len() < total_limit {
        let remaining = total_limit - all_klines.len();
        let limit = remaining.min(batch_size);

        let url = if let Some(end) = end_time {
            format!(
                "https://api.bybit.com/v5/market/kline?category=spot&symbol={}&interval={}&limit={}&end={}",
                symbol, interval, limit, end
            )
        } else {
            format!(
                "https://api.bybit.com/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
                symbol, interval, limit
            )
        };

        let client = reqwest::blocking::Client::new();
        let response: BybitResponse = client
            .get(&url)
            .send()
            .context("Failed to send request to Bybit API")?
            .json()
            .context("Failed to parse Bybit API response")?;

        if response.ret_code != 0 {
            anyhow::bail!("Bybit API error: {}", response.ret_msg);
        }

        if response.result.list.is_empty() {
            break;
        }

        let klines: Vec<BybitKline> = response
            .result
            .list
            .into_iter()
            .map(|row| BybitKline {
                timestamp: row[0].parse().unwrap_or(0),
                open: row[1].parse().unwrap_or(0.0),
                high: row[2].parse().unwrap_or(0.0),
                low: row[3].parse().unwrap_or(0.0),
                close: row[4].parse().unwrap_or(0.0),
                volume: row[5].parse().unwrap_or(0.0),
                turnover: row[6].parse().unwrap_or(0.0),
            })
            .collect();

        // Update end_time for next batch (earliest timestamp - 1)
        if let Some(last) = klines.last() {
            end_time = Some(last.timestamp - 1);
        }

        // Reverse and prepend (since we're fetching backwards in time)
        let mut klines = klines;
        klines.reverse();

        for k in klines.into_iter().rev() {
            all_klines.insert(0, k);
        }

        if verbose {
            println!("Fetched {} klines, total: {}", limit, all_klines.len());
        }

        if all_klines.len() >= total_limit {
            break;
        }

        // Rate limiting
        std::thread::sleep(std::time::Duration::from_millis(100));
    }

    Ok(all_klines)
}

/// Calculate technical indicators from OHLCV data
fn calculate_indicators(klines: &[BybitKline]) -> Array2<f64> {
    let n = klines.len();
    let n_features = 15;
    let mut features = Array2::zeros((n, n_features));

    // Pre-calculate price arrays for efficiency
    let closes: Vec<f64> = klines.iter().map(|k| k.close).collect();
    let volumes: Vec<f64> = klines.iter().map(|k| k.volume).collect();

    for i in 0..n {
        let k = &klines[i];

        // 0: Returns
        let returns = if i > 0 {
            (k.close - closes[i - 1]) / closes[i - 1]
        } else {
            0.0
        };
        features[[i, 0]] = returns;

        // 1: Log returns
        features[[i, 1]] = if i > 0 {
            (k.close / closes[i - 1]).ln()
        } else {
            0.0
        };

        // 2: High-Low range (normalized)
        let range = if k.close > 0.0 {
            (k.high - k.low) / k.close
        } else {
            0.0
        };
        features[[i, 2]] = range;

        // 3: Close-Open change (normalized)
        let body = if k.open > 0.0 {
            (k.close - k.open) / k.open
        } else {
            0.0
        };
        features[[i, 3]] = body;

        // 4: Upper shadow (normalized)
        let upper_shadow = if k.close > 0.0 {
            (k.high - k.close.max(k.open)) / k.close
        } else {
            0.0
        };
        features[[i, 4]] = upper_shadow;

        // 5: Lower shadow (normalized)
        let lower_shadow = if k.close > 0.0 {
            (k.close.min(k.open) - k.low) / k.close
        } else {
            0.0
        };
        features[[i, 5]] = lower_shadow;

        // 6-8: SMA ratios (5, 10, 20 periods)
        for (j, period) in [5, 10, 20].iter().enumerate() {
            if i >= *period {
                let sma: f64 = closes[i - period + 1..=i].iter().sum::<f64>() / *period as f64;
                features[[i, 6 + j]] = if sma > 0.0 { k.close / sma - 1.0 } else { 0.0 };
            }
        }

        // 9: Volume change
        if i > 0 && volumes[i - 1] > 0.0 {
            features[[i, 9]] = (k.volume / volumes[i - 1]).ln();
        }

        // 10: Volume SMA ratio
        if i >= 20 {
            let vol_sma: f64 = volumes[i - 19..=i].iter().sum::<f64>() / 20.0;
            features[[i, 10]] = if vol_sma > 0.0 { k.volume / vol_sma } else { 1.0 };
        }

        // 11: RSI-like momentum (14 periods)
        if i >= 14 {
            let mut gains = 0.0;
            let mut losses = 0.0;
            for j in (i - 13)..=i {
                let change = closes[j] - closes[j - 1];
                if change > 0.0 {
                    gains += change;
                } else {
                    losses -= change;
                }
            }
            let rs = if losses > 0.0 { gains / losses } else { 100.0 };
            features[[i, 11]] = 100.0 - 100.0 / (1.0 + rs);
            // Normalize to [-1, 1]
            features[[i, 11]] = (features[[i, 11]] - 50.0) / 50.0;
        }

        // 12: Price position in range (where close is between high and low)
        let hl_range = k.high - k.low;
        features[[i, 12]] = if hl_range > 0.0 {
            2.0 * (k.close - k.low) / hl_range - 1.0
        } else {
            0.0
        };

        // 13: Volatility (20-period rolling std of returns)
        if i >= 20 {
            let returns_slice: Vec<f64> = (i - 19..=i)
                .filter(|&j| j > 0)
                .map(|j| (closes[j] - closes[j - 1]) / closes[j - 1])
                .collect();
            if !returns_slice.is_empty() {
                let mean: f64 = returns_slice.iter().sum::<f64>() / returns_slice.len() as f64;
                let variance: f64 = returns_slice.iter().map(|r| (r - mean).powi(2)).sum::<f64>()
                    / returns_slice.len() as f64;
                features[[i, 13]] = variance.sqrt() * 100.0; // Scale up for visibility
            }
        }

        // 14: Trend (linear regression slope of last 10 periods, normalized)
        if i >= 10 {
            let x_mean = 4.5; // Mean of 0..9
            let y_vals: Vec<f64> = closes[i - 9..=i].to_vec();
            let y_mean: f64 = y_vals.iter().sum::<f64>() / 10.0;

            let mut numerator = 0.0;
            let mut denominator = 0.0;
            for (j, &y) in y_vals.iter().enumerate() {
                let x = j as f64;
                numerator += (x - x_mean) * (y - y_mean);
                denominator += (x - x_mean).powi(2);
            }
            let slope = if denominator > 0.0 {
                numerator / denominator
            } else {
                0.0
            };
            // Normalize by price level
            features[[i, 14]] = if y_mean > 0.0 {
                slope / y_mean * 100.0
            } else {
                0.0
            };
        }
    }

    // Handle NaN and Inf values
    for val in features.iter_mut() {
        if val.is_nan() || val.is_infinite() {
            *val = 0.0;
        }
    }

    features
}

/// Resample klines to a larger timeframe
fn resample_klines(klines: &[BybitKline], factor: usize) -> Vec<BybitKline> {
    klines
        .chunks(factor)
        .filter(|chunk| !chunk.is_empty())
        .map(|chunk| {
            let first = &chunk[0];
            let last = chunk.last().unwrap();
            BybitKline {
                timestamp: first.timestamp,
                open: first.open,
                high: chunk.iter().map(|k| k.high).fold(f64::NEG_INFINITY, f64::max),
                low: chunk.iter().map(|k| k.low).fold(f64::INFINITY, f64::min),
                close: last.close,
                volume: chunk.iter().map(|k| k.volume).sum(),
                turnover: chunk.iter().map(|k| k.turnover).sum(),
            }
        })
        .collect()
}

/// Create multi-scale features from base kline data
///
/// # Arguments
/// * `klines` - Base kline data (should be the shortest timeframe)
/// * `scales` - List of scale names (e.g., ["1min", "5min", "1H"])
///
/// # Returns
/// MultiScaleData containing features for each scale
pub fn create_multi_scale_features(
    klines: &[BybitKline],
    scales: &[&str],
) -> MultiScaleData {
    let mut features = HashMap::new();

    // Define resampling factors relative to base (assumes base is 1min)
    let resample_factors: HashMap<&str, usize> = [
        ("1min", 1),
        ("5min", 5),
        ("15min", 15),
        ("30min", 30),
        ("1H", 60),
        ("4H", 240),
        ("1D", 1440),
    ]
    .into_iter()
    .collect();

    for scale in scales {
        let factor = resample_factors.get(scale).copied().unwrap_or(1);
        let resampled = if factor == 1 {
            klines.to_vec()
        } else {
            resample_klines(klines, factor)
        };

        let scale_features = calculate_indicators(&resampled);
        features.insert(scale.to_string(), scale_features);
    }

    let timestamps: Vec<i64> = klines.iter().map(|k| k.timestamp).collect();

    MultiScaleData {
        features,
        timestamps,
        ohlcv: Some(klines.to_vec()),
    }
}

/// Create sequences for training/inference
///
/// # Arguments
/// * `multi_scale_data` - Multi-scale feature data
/// * `lookback_periods` - Lookback periods for each scale
/// * `prediction_horizon` - How many steps ahead to predict
///
/// # Returns
/// Tuple of (scale_sequences, targets, timestamps)
pub fn create_sequences(
    multi_scale_data: &MultiScaleData,
    lookback_periods: &HashMap<String, usize>,
    prediction_horizon: usize,
) -> (HashMap<String, Array3<f64>>, Array1<f64>, Vec<i64>) {
    // Find the scale with minimum samples to align
    let min_samples = multi_scale_data
        .features
        .values()
        .map(|f| f.nrows())
        .min()
        .unwrap_or(0);

    // Calculate max lookback to determine start index
    let max_lookback = lookback_periods.values().copied().max().unwrap_or(1);
    let n_sequences = min_samples.saturating_sub(max_lookback + prediction_horizon);

    if n_sequences == 0 {
        return (
            HashMap::new(),
            Array1::zeros(0),
            Vec::new(),
        );
    }

    let mut sequences: HashMap<String, Array3<f64>> = HashMap::new();
    let mut targets = Vec::with_capacity(n_sequences);
    let mut timestamps = Vec::with_capacity(n_sequences);

    // Initialize sequence arrays
    for (scale, lookback) in lookback_periods {
        if let Some(features) = multi_scale_data.features.get(scale) {
            let n_features = features.ncols();
            sequences.insert(
                scale.clone(),
                Array3::zeros((n_sequences, *lookback, n_features)),
            );
        }
    }

    // Fill sequences
    for i in 0..n_sequences {
        let idx = max_lookback + i;

        for (scale, lookback) in lookback_periods {
            if let Some(features) = multi_scale_data.features.get(scale) {
                // Calculate aligned index for this scale
                let scale_ratio = features.nrows() as f64 / min_samples as f64;
                let scale_idx = ((idx as f64) * scale_ratio) as usize;

                if scale_idx >= *lookback && scale_idx < features.nrows() {
                    if let Some(seq) = sequences.get_mut(scale) {
                        for j in 0..*lookback {
                            let src_idx = scale_idx - lookback + 1 + j;
                            if src_idx < features.nrows() {
                                for k in 0..features.ncols() {
                                    seq[[i, j, k]] = features[[src_idx, k]];
                                }
                            }
                        }
                    }
                }
            }
        }

        // Target: next period return (using the primary/shortest scale)
        if let Some(primary_features) = multi_scale_data.features.values().next() {
            let target_idx = idx + prediction_horizon;
            if target_idx < primary_features.nrows() {
                targets.push(primary_features[[target_idx, 0]]); // Return feature
            } else {
                targets.push(0.0);
            }
        }

        if idx < multi_scale_data.timestamps.len() {
            timestamps.push(multi_scale_data.timestamps[idx]);
        }
    }

    (
        sequences,
        Array1::from_vec(targets),
        timestamps,
    )
}

/// Generate synthetic multi-scale data for testing
///
/// # Arguments
/// * `n_samples` - Number of samples to generate
/// * `scales` - Scale names
/// * `n_features` - Number of features per scale
/// * `lookbacks` - Lookback periods for each scale
///
/// # Returns
/// Tuple of (sequences, targets)
pub fn generate_synthetic_data(
    n_samples: usize,
    scales: &[&str],
    n_features: usize,
    lookbacks: &HashMap<String, usize>,
) -> (HashMap<String, Array3<f64>>, Array1<f64>) {
    let mut sequences = HashMap::new();

    for scale in scales {
        let lookback = lookbacks.get(*scale).copied().unwrap_or(10);
        let data = Array3::random(
            (n_samples, lookback, n_features),
            Normal::new(0.0, 1.0).unwrap(),
        );
        sequences.insert(scale.to_string(), data);
    }

    // Generate synthetic targets (random returns)
    let targets = Array1::random(n_samples, Normal::new(0.0, 0.02).unwrap());

    (sequences, targets)
}

/// Split data into train/validation/test sets
///
/// # Arguments
/// * `sequences` - Multi-scale sequences
/// * `targets` - Target values
/// * `train_ratio` - Fraction for training (default 0.7)
/// * `val_ratio` - Fraction for validation (default 0.15)
///
/// # Returns
/// Tuple of (train, val, test) data
pub fn train_val_test_split(
    sequences: &HashMap<String, Array3<f64>>,
    targets: &Array1<f64>,
    train_ratio: f64,
    val_ratio: f64,
) -> (
    (HashMap<String, Array3<f64>>, Array1<f64>),
    (HashMap<String, Array3<f64>>, Array1<f64>),
    (HashMap<String, Array3<f64>>, Array1<f64>),
) {
    let n_samples = targets.len();
    let train_end = (n_samples as f64 * train_ratio) as usize;
    let val_end = train_end + (n_samples as f64 * val_ratio) as usize;

    let split_sequences = |start: usize, end: usize| -> HashMap<String, Array3<f64>> {
        sequences
            .iter()
            .map(|(k, v)| {
                (k.clone(), v.slice(s![start..end, .., ..]).to_owned())
            })
            .collect()
    };

    let train_seq = split_sequences(0, train_end);
    let val_seq = split_sequences(train_end, val_end);
    let test_seq = split_sequences(val_end, n_samples);

    let train_targets = targets.slice(s![0..train_end]).to_owned();
    let val_targets = targets.slice(s![train_end..val_end]).to_owned();
    let test_targets = targets.slice(s![val_end..]).to_owned();

    (
        (train_seq, train_targets),
        (val_seq, val_targets),
        (test_seq, test_targets),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_synthetic_data() {
        let scales = ["1min", "5min", "1H"];
        let lookbacks: HashMap<String, usize> = [
            ("1min".to_string(), 60),
            ("5min".to_string(), 24),
            ("1H".to_string(), 12),
        ]
        .into_iter()
        .collect();

        let (sequences, targets) = generate_synthetic_data(100, &scales, 15, &lookbacks);

        assert_eq!(sequences.len(), 3);
        assert_eq!(targets.len(), 100);
        assert_eq!(sequences["1min"].dim(), (100, 60, 15));
        assert_eq!(sequences["5min"].dim(), (100, 24, 15));
        assert_eq!(sequences["1H"].dim(), (100, 12, 15));
    }

    #[test]
    fn test_train_val_test_split() {
        let scales = ["1min", "5min"];
        let lookbacks: HashMap<String, usize> = [
            ("1min".to_string(), 10),
            ("5min".to_string(), 5),
        ]
        .into_iter()
        .collect();

        let (sequences, targets) = generate_synthetic_data(100, &scales, 10, &lookbacks);
        let ((train_seq, train_y), (val_seq, val_y), (test_seq, test_y)) =
            train_val_test_split(&sequences, &targets, 0.7, 0.15);

        assert_eq!(train_y.len(), 70);
        assert_eq!(val_y.len(), 15);
        assert_eq!(test_y.len(), 15);

        assert_eq!(train_seq["1min"].dim().0, 70);
        assert_eq!(val_seq["1min"].dim().0, 15);
        assert_eq!(test_seq["1min"].dim().0, 15);
    }
}
