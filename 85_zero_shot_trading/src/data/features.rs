//! Feature engineering for market data.

use crate::data::bybit::Kline;
use crate::{Result, ZeroShotError};

/// Feature generator for market time series.
#[derive(Debug, Clone)]
pub struct FeatureGenerator {
    /// Sequence length for features
    pub sequence_length: usize,
}

impl Default for FeatureGenerator {
    fn default() -> Self {
        Self::new(50)
    }
}

impl FeatureGenerator {
    /// Create a new feature generator.
    pub fn new(sequence_length: usize) -> Self {
        Self { sequence_length }
    }

    /// Generate features from kline data.
    pub fn generate(&self, klines: &[Kline]) -> Result<Vec<Vec<f64>>> {
        if klines.len() < self.sequence_length + 50 {
            return Err(ZeroShotError::NotEnoughData {
                needed: self.sequence_length + 50,
                got: klines.len(),
            });
        }

        let prices: Vec<f64> = klines.iter().map(|k| k.close).collect();
        let highs: Vec<f64> = klines.iter().map(|k| k.high).collect();
        let lows: Vec<f64> = klines.iter().map(|k| k.low).collect();
        let volumes: Vec<f64> = klines.iter().map(|k| k.volume).collect();

        // Calculate features for each timestep
        let mut features = Vec::new();

        for i in 50..klines.len() {
            let mut row = Vec::new();

            // Returns
            let returns = (prices[i] - prices[i - 1]) / prices[i - 1];
            row.push(returns);

            // Log returns
            let log_returns = (prices[i] / prices[i - 1]).ln();
            row.push(log_returns);

            // High-low range
            let range = (highs[i] - lows[i]) / prices[i];
            row.push(range);

            // Close position in range
            let close_pos = if (highs[i] - lows[i]).abs() > 1e-8 {
                (prices[i] - lows[i]) / (highs[i] - lows[i])
            } else {
                0.5
            };
            row.push(close_pos);

            // Volume ratio (vs 20-period MA)
            let vol_ma: f64 = volumes[i - 20..i].iter().sum::<f64>() / 20.0;
            let vol_ratio = if vol_ma > 1e-8 {
                volumes[i] / vol_ma
            } else {
                1.0
            };
            row.push(vol_ratio);

            // Volatility (20-period)
            let returns_20: Vec<f64> = (1..=20)
                .map(|j| (prices[i - j + 1] - prices[i - j]) / prices[i - j])
                .collect();
            let vol_20 = std_dev(&returns_20);
            row.push(vol_20);

            // Volatility (5-period)
            let returns_5: Vec<f64> = (1..=5)
                .map(|j| (prices[i - j + 1] - prices[i - j]) / prices[i - j])
                .collect();
            let vol_5 = std_dev(&returns_5);
            row.push(vol_5);

            // SMA ratio (10/20)
            let sma_10: f64 = prices[i - 10..i].iter().sum::<f64>() / 10.0;
            let sma_20: f64 = prices[i - 20..i].iter().sum::<f64>() / 20.0;
            let sma_ratio = sma_10 / sma_20;
            row.push(sma_ratio);

            // RSI (14-period)
            let rsi = calculate_rsi(&prices[i - 14..=i]);
            row.push(rsi / 100.0); // Normalize to [0, 1]

            // MACD
            let ema_12 = ema(&prices[..=i], 12);
            let ema_26 = ema(&prices[..=i], 26);
            let macd = (ema_12 - ema_26) / prices[i];
            row.push(macd);

            // Bollinger Band position
            let bb_middle = sma_20;
            let bb_std = std_dev(&prices[i - 20..i].to_vec());
            let bb_upper = bb_middle + 2.0 * bb_std;
            let bb_lower = bb_middle - 2.0 * bb_std;
            let bb_pos = if (bb_upper - bb_lower).abs() > 1e-8 {
                (prices[i] - bb_lower) / (bb_upper - bb_lower)
            } else {
                0.5
            };
            row.push(bb_pos);

            features.push(row);
        }

        // Take last sequence_length rows
        if features.len() < self.sequence_length {
            return Err(ZeroShotError::NotEnoughData {
                needed: self.sequence_length,
                got: features.len(),
            });
        }

        let start = features.len() - self.sequence_length;
        let features = features[start..].to_vec();

        // Normalize features (z-score per column)
        let normalized = z_score_normalize(&features)?;

        Ok(normalized)
    }
}

/// Prepare features from kline data (convenience function).
pub fn prepare_features(klines: &[Kline]) -> Result<Vec<Vec<f64>>> {
    let generator = FeatureGenerator::default();
    generator.generate(klines)
}

/// Calculate standard deviation.
fn std_dev(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
    let variance: f64 = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
    variance.sqrt()
}

/// Calculate RSI.
fn calculate_rsi(prices: &[f64]) -> f64 {
    if prices.len() < 2 {
        return 50.0;
    }

    let mut gains = Vec::new();
    let mut losses = Vec::new();

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

    let avg_gain: f64 = gains.iter().sum::<f64>() / gains.len() as f64;
    let avg_loss: f64 = losses.iter().sum::<f64>() / losses.len() as f64;

    if avg_loss.abs() < 1e-8 {
        return 100.0;
    }

    let rs = avg_gain / avg_loss;
    100.0 - (100.0 / (1.0 + rs))
}

/// Calculate EMA.
fn ema(prices: &[f64], period: usize) -> f64 {
    if prices.is_empty() {
        return 0.0;
    }
    if prices.len() < period {
        return prices.iter().sum::<f64>() / prices.len() as f64;
    }

    let multiplier = 2.0 / (period as f64 + 1.0);
    let mut ema_val = prices[..period].iter().sum::<f64>() / period as f64;

    for price in &prices[period..] {
        ema_val = (price - ema_val) * multiplier + ema_val;
    }

    ema_val
}

/// Z-score normalization per column.
fn z_score_normalize(features: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
    if features.is_empty() {
        return Ok(Vec::new());
    }

    let num_features = features[0].len();
    let num_rows = features.len();

    // Calculate mean and std for each column
    let mut means = vec![0.0; num_features];
    let mut stds = vec![0.0; num_features];

    for row in features {
        for (i, val) in row.iter().enumerate() {
            means[i] += val / num_rows as f64;
        }
    }

    for row in features {
        for (i, val) in row.iter().enumerate() {
            stds[i] += (val - means[i]).powi(2) / num_rows as f64;
        }
    }

    for std in &mut stds {
        *std = std.sqrt().max(1e-8);
    }

    // Normalize
    let normalized: Vec<Vec<f64>> = features
        .iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .map(|(i, val)| (val - means[i]) / stds[i])
                .collect()
        })
        .collect();

    Ok(normalized)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn create_dummy_klines(n: usize) -> Vec<Kline> {
        let mut klines = Vec::with_capacity(n);
        let mut price = 100.0;

        for i in 0..n {
            price *= 1.0 + (rand::random::<f64>() - 0.5) * 0.02;
            klines.push(Kline {
                timestamp: Utc::now(),
                open: price * 0.99,
                high: price * 1.01,
                low: price * 0.98,
                close: price,
                volume: 1000.0 * (1.0 + rand::random::<f64>()),
                turnover: price * 1000.0,
            });
        }

        klines
    }

    #[test]
    fn test_feature_generation() {
        let klines = create_dummy_klines(200);
        let generator = FeatureGenerator::new(50);

        let features = generator.generate(&klines).unwrap();
        assert_eq!(features.len(), 50);
        assert_eq!(features[0].len(), 11);
    }

    #[test]
    fn test_not_enough_data() {
        let klines = create_dummy_klines(50);
        let generator = FeatureGenerator::new(50);

        let result = generator.generate(&klines);
        assert!(result.is_err());
    }
}
