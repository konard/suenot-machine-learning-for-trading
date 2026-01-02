//! Feature engineering for trading data
//!
//! This module provides technical indicator calculations and
//! feature normalization for machine learning.

use anyhow::Result;
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

use super::ohlcv::OHLCVDataset;

/// Normalization parameters for features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizationParams {
    pub means: Vec<f64>,
    pub stds: Vec<f64>,
    pub feature_names: Vec<String>,
}

impl NormalizationParams {
    /// Create new normalization params
    pub fn new(means: Vec<f64>, stds: Vec<f64>, feature_names: Vec<String>) -> Self {
        Self {
            means,
            stds,
            feature_names,
        }
    }

    /// Normalize a feature matrix
    pub fn normalize(&self, data: &Array2<f64>) -> Array2<f64> {
        let mut normalized = data.clone();
        for (i, (mean, std)) in self.means.iter().zip(self.stds.iter()).enumerate() {
            let std = if *std < 1e-8 { 1.0 } else { *std };
            for j in 0..normalized.nrows() {
                normalized[[j, i]] = (normalized[[j, i]] - mean) / std;
            }
        }
        normalized
    }

    /// Denormalize a feature matrix
    pub fn denormalize(&self, data: &Array2<f64>) -> Array2<f64> {
        let mut denormalized = data.clone();
        for (i, (mean, std)) in self.means.iter().zip(self.stds.iter()).enumerate() {
            for j in 0..denormalized.nrows() {
                denormalized[[j, i]] = denormalized[[j, i]] * std + mean;
            }
        }
        denormalized
    }

    /// Save parameters to file
    pub fn save(&self, path: &str) -> Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load parameters from file
    pub fn load(path: &str) -> Result<Self> {
        let json = std::fs::read_to_string(path)?;
        let params = serde_json::from_str(&json)?;
        Ok(params)
    }
}

/// Feature builder for creating technical indicators
pub struct FeatureBuilder {
    window_size: usize,
    features: Vec<String>,
}

impl FeatureBuilder {
    /// Create a new feature builder
    pub fn new(window_size: usize, features: Vec<String>) -> Self {
        Self {
            window_size,
            features,
        }
    }

    /// Build features from OHLCV dataset
    pub fn build(&self, dataset: &OHLCVDataset) -> Result<(Array2<f64>, NormalizationParams)> {
        let n = dataset.len();
        if n < self.window_size + 50 {
            anyhow::bail!(
                "Dataset too small: {} candles, need at least {}",
                n,
                self.window_size + 50
            );
        }

        // Calculate all features
        let closes = dataset.closes();
        let volumes = dataset.volumes();
        let highs: Vec<f64> = dataset.data.iter().map(|d| d.high).collect();
        let lows: Vec<f64> = dataset.data.iter().map(|d| d.low).collect();
        let opens: Vec<f64> = dataset.data.iter().map(|d| d.open).collect();

        let mut feature_cols: Vec<Array1<f64>> = Vec::new();
        let mut feature_names: Vec<String> = Vec::new();

        // Core OHLCV features (normalized returns)
        if self.features.contains(&"close".to_string()) {
            feature_cols.push(Array1::from(self.returns(&closes)));
            feature_names.push("close_return".to_string());
        }

        if self.features.contains(&"volume".to_string()) {
            feature_cols.push(Array1::from(self.log_returns(&volumes)));
            feature_names.push("volume_change".to_string());
        }

        // RSI
        if self.features.contains(&"rsi".to_string()) {
            feature_cols.push(Array1::from(self.rsi(&closes, 14)));
            feature_names.push("rsi_14".to_string());
        }

        // MACD
        if self.features.contains(&"macd".to_string()) {
            let (macd, signal, hist) = self.macd(&closes, 12, 26, 9);
            feature_cols.push(Array1::from(macd));
            feature_cols.push(Array1::from(signal));
            feature_cols.push(Array1::from(hist));
            feature_names.push("macd".to_string());
            feature_names.push("macd_signal".to_string());
            feature_names.push("macd_hist".to_string());
        }

        // Bollinger Bands
        if self.features.contains(&"bb_upper".to_string())
            || self.features.contains(&"bb_lower".to_string())
        {
            let (upper, middle, lower) = self.bollinger_bands(&closes, 20, 2.0);
            let bb_position: Vec<f64> = closes
                .iter()
                .zip(upper.iter().zip(lower.iter()))
                .map(|(c, (u, l))| {
                    if (u - l).abs() > 1e-8 {
                        (c - l) / (u - l)
                    } else {
                        0.5
                    }
                })
                .collect();
            feature_cols.push(Array1::from(bb_position));
            feature_names.push("bb_position".to_string());

            let bb_width: Vec<f64> = upper
                .iter()
                .zip(middle.iter().zip(lower.iter()))
                .map(|(u, (m, l))| {
                    if *m > 1e-8 {
                        (u - l) / m
                    } else {
                        0.0
                    }
                })
                .collect();
            feature_cols.push(Array1::from(bb_width));
            feature_names.push("bb_width".to_string());
        }

        // ATR (Average True Range)
        let atr = self.atr(&highs, &lows, &closes, 14);
        feature_cols.push(Array1::from(atr));
        feature_names.push("atr_14".to_string());

        // OBV (On-Balance Volume)
        let obv = self.obv(&closes, &volumes);
        feature_cols.push(Array1::from(self.returns(&obv)));
        feature_names.push("obv_change".to_string());

        // Price momentum (multiple periods)
        for period in [5, 10, 20] {
            let mom = self.momentum(&closes, period);
            feature_cols.push(Array1::from(mom));
            feature_names.push(format!("momentum_{}", period));
        }

        // Volatility
        let vol = self.volatility(&closes, 20);
        feature_cols.push(Array1::from(vol));
        feature_names.push("volatility_20".to_string());

        // Candlestick patterns
        let body_ratio: Vec<f64> = dataset
            .data
            .iter()
            .map(|d| {
                let range = d.range();
                if range > 1e-8 {
                    d.body_size() / range
                } else {
                    0.5
                }
            })
            .collect();
        feature_cols.push(Array1::from(body_ratio));
        feature_names.push("body_ratio".to_string());

        // Upper shadow ratio
        let upper_shadow: Vec<f64> = dataset
            .data
            .iter()
            .map(|d| {
                let range = d.range();
                if range > 1e-8 {
                    d.upper_shadow() / range
                } else {
                    0.0
                }
            })
            .collect();
        feature_cols.push(Array1::from(upper_shadow));
        feature_names.push("upper_shadow".to_string());

        // Lower shadow ratio
        let lower_shadow: Vec<f64> = dataset
            .data
            .iter()
            .map(|d| {
                let range = d.range();
                if range > 1e-8 {
                    d.lower_shadow() / range
                } else {
                    0.0
                }
            })
            .collect();
        feature_cols.push(Array1::from(lower_shadow));
        feature_names.push("lower_shadow".to_string());

        // Combine all features into matrix
        let min_len = feature_cols.iter().map(|c| c.len()).min().unwrap_or(0);
        let num_features = feature_cols.len();

        let mut matrix = Array2::zeros((min_len, num_features));
        for (i, col) in feature_cols.iter().enumerate() {
            let offset = col.len() - min_len;
            for j in 0..min_len {
                matrix[[j, i]] = col[offset + j];
            }
        }

        // Calculate normalization parameters
        let mut means = Vec::with_capacity(num_features);
        let mut stds = Vec::with_capacity(num_features);

        for i in 0..num_features {
            let col = matrix.column(i);
            let mean = col.mean().unwrap_or(0.0);
            let variance = col.mapv(|x| (x - mean).powi(2)).mean().unwrap_or(1.0);
            let std = variance.sqrt();

            means.push(mean);
            stds.push(if std < 1e-8 { 1.0 } else { std });
        }

        let norm_params = NormalizationParams::new(means, stds, feature_names);

        // Normalize the data
        let normalized = norm_params.normalize(&matrix);

        Ok((normalized, norm_params))
    }

    // Helper functions for technical indicators

    fn returns(&self, prices: &[f64]) -> Vec<f64> {
        if prices.len() < 2 {
            return vec![0.0; prices.len()];
        }

        let mut returns = vec![0.0];
        for i in 1..prices.len() {
            if prices[i - 1] > 1e-8 {
                returns.push((prices[i] - prices[i - 1]) / prices[i - 1]);
            } else {
                returns.push(0.0);
            }
        }
        returns
    }

    fn log_returns(&self, values: &[f64]) -> Vec<f64> {
        if values.len() < 2 {
            return vec![0.0; values.len()];
        }

        let mut returns = vec![0.0];
        for i in 1..values.len() {
            if values[i] > 1e-8 && values[i - 1] > 1e-8 {
                returns.push((values[i] / values[i - 1]).ln());
            } else {
                returns.push(0.0);
            }
        }
        returns
    }

    fn sma(&self, prices: &[f64], period: usize) -> Vec<f64> {
        if prices.len() < period {
            return vec![prices.iter().sum::<f64>() / prices.len() as f64; prices.len()];
        }

        let mut sma = vec![0.0; period - 1];
        for i in (period - 1)..prices.len() {
            let sum: f64 = prices[(i + 1 - period)..=i].iter().sum();
            sma.push(sum / period as f64);
        }
        sma
    }

    fn ema(&self, prices: &[f64], period: usize) -> Vec<f64> {
        if prices.is_empty() {
            return vec![];
        }

        let multiplier = 2.0 / (period as f64 + 1.0);
        let mut ema = vec![prices[0]];

        for i in 1..prices.len() {
            let new_ema = (prices[i] - ema[i - 1]) * multiplier + ema[i - 1];
            ema.push(new_ema);
        }
        ema
    }

    fn rsi(&self, prices: &[f64], period: usize) -> Vec<f64> {
        if prices.len() < 2 {
            return vec![50.0; prices.len()];
        }

        let mut gains = vec![0.0];
        let mut losses = vec![0.0];

        for i in 1..prices.len() {
            let change = prices[i] - prices[i - 1];
            if change > 0.0 {
                gains.push(change);
                losses.push(0.0);
            } else {
                gains.push(0.0);
                losses.push(-change);
            }
        }

        let avg_gain = self.ema(&gains, period);
        let avg_loss = self.ema(&losses, period);

        avg_gain
            .iter()
            .zip(avg_loss.iter())
            .map(|(g, l)| {
                if *l < 1e-8 {
                    100.0
                } else {
                    100.0 - (100.0 / (1.0 + g / l))
                }
            })
            .collect()
    }

    fn macd(
        &self,
        prices: &[f64],
        fast: usize,
        slow: usize,
        signal: usize,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let ema_fast = self.ema(prices, fast);
        let ema_slow = self.ema(prices, slow);

        let macd_line: Vec<f64> = ema_fast
            .iter()
            .zip(ema_slow.iter())
            .map(|(f, s)| f - s)
            .collect();

        let signal_line = self.ema(&macd_line, signal);

        let histogram: Vec<f64> = macd_line
            .iter()
            .zip(signal_line.iter())
            .map(|(m, s)| m - s)
            .collect();

        (macd_line, signal_line, histogram)
    }

    fn bollinger_bands(
        &self,
        prices: &[f64],
        period: usize,
        num_std: f64,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let middle = self.sma(prices, period);

        let mut upper = Vec::with_capacity(prices.len());
        let mut lower = Vec::with_capacity(prices.len());

        for i in 0..prices.len() {
            if i < period - 1 {
                upper.push(middle[i] + num_std * 0.01 * middle[i]);
                lower.push(middle[i] - num_std * 0.01 * middle[i]);
            } else {
                let slice = &prices[(i + 1 - period)..=i];
                let mean = middle[i];
                let variance: f64 =
                    slice.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / period as f64;
                let std = variance.sqrt();

                upper.push(mean + num_std * std);
                lower.push(mean - num_std * std);
            }
        }

        (upper, middle, lower)
    }

    fn atr(&self, highs: &[f64], lows: &[f64], closes: &[f64], period: usize) -> Vec<f64> {
        if highs.is_empty() {
            return vec![];
        }

        let mut true_ranges = vec![highs[0] - lows[0]];

        for i in 1..highs.len() {
            let tr = (highs[i] - lows[i])
                .max((highs[i] - closes[i - 1]).abs())
                .max((lows[i] - closes[i - 1]).abs());
            true_ranges.push(tr);
        }

        self.ema(&true_ranges, period)
    }

    fn obv(&self, closes: &[f64], volumes: &[f64]) -> Vec<f64> {
        if closes.is_empty() {
            return vec![];
        }

        let mut obv = vec![volumes[0]];

        for i in 1..closes.len() {
            let delta = if closes[i] > closes[i - 1] {
                volumes[i]
            } else if closes[i] < closes[i - 1] {
                -volumes[i]
            } else {
                0.0
            };
            obv.push(obv[i - 1] + delta);
        }

        obv
    }

    fn momentum(&self, prices: &[f64], period: usize) -> Vec<f64> {
        let mut mom = vec![0.0; period];

        for i in period..prices.len() {
            if prices[i - period] > 1e-8 {
                mom.push((prices[i] - prices[i - period]) / prices[i - period]);
            } else {
                mom.push(0.0);
            }
        }

        mom
    }

    fn volatility(&self, prices: &[f64], period: usize) -> Vec<f64> {
        let returns = self.returns(prices);
        let mut vol = vec![0.0; period - 1];

        for i in (period - 1)..returns.len() {
            let slice = &returns[(i + 1 - period)..=i];
            let mean = slice.iter().sum::<f64>() / period as f64;
            let variance: f64 =
                slice.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / period as f64;
            vol.push(variance.sqrt());
        }

        vol
    }
}

/// Generate classification labels based on forward returns
pub fn generate_labels(
    closes: &[f64],
    horizon: usize,
    threshold_pct: f64,
) -> Vec<i64> {
    let mut labels = Vec::with_capacity(closes.len());

    for i in 0..closes.len() {
        if i + horizon >= closes.len() {
            labels.push(1); // Neutral for end of data
        } else {
            let current = closes[i];
            let future = closes[i + horizon];
            let return_pct = if current > 1e-8 {
                (future - current) / current * 100.0
            } else {
                0.0
            };

            if return_pct > threshold_pct {
                labels.push(2); // Bullish
            } else if return_pct < -threshold_pct {
                labels.push(0); // Bearish
            } else {
                labels.push(1); // Neutral
            }
        }
    }

    labels
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sma() {
        let builder = FeatureBuilder::new(64, vec![]);
        let prices = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sma = builder.sma(&prices, 3);

        assert_eq!(sma.len(), 5);
        assert!((sma[2] - 2.0).abs() < 1e-8);
        assert!((sma[3] - 3.0).abs() < 1e-8);
        assert!((sma[4] - 4.0).abs() < 1e-8);
    }

    #[test]
    fn test_rsi() {
        let builder = FeatureBuilder::new(64, vec![]);
        let prices: Vec<f64> = (0..30).map(|i| 100.0 + i as f64).collect();
        let rsi = builder.rsi(&prices, 14);

        // All prices increasing, RSI should be high
        assert!(rsi.last().unwrap() > &70.0);
    }

    #[test]
    fn test_generate_labels() {
        let closes = vec![100.0, 101.0, 99.0, 100.5, 102.0];
        let labels = generate_labels(&closes, 1, 0.5);

        assert_eq!(labels.len(), 5);
        assert_eq!(labels[0], 2); // 100 -> 101 = +1% (bullish)
        assert_eq!(labels[1], 0); // 101 -> 99 = -2% (bearish)
    }
}
