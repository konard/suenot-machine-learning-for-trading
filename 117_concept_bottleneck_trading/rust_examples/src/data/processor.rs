//! Data processing utilities for preparing market data.

use crate::api::bybit::Kline;
use ndarray::Array1;

/// Data processor for normalizing and transforming market data.
pub struct DataProcessor {
    pub lookback_window: usize,
}

impl DataProcessor {
    /// Create a new data processor.
    pub fn new(lookback_window: usize) -> Self {
        Self { lookback_window }
    }

    /// Extract close prices from klines.
    pub fn extract_closes(&self, klines: &[Kline]) -> Vec<f64> {
        klines.iter().map(|k| k.close).collect()
    }

    /// Extract high prices from klines.
    pub fn extract_highs(&self, klines: &[Kline]) -> Vec<f64> {
        klines.iter().map(|k| k.high).collect()
    }

    /// Extract low prices from klines.
    pub fn extract_lows(&self, klines: &[Kline]) -> Vec<f64> {
        klines.iter().map(|k| k.low).collect()
    }

    /// Extract volumes from klines.
    pub fn extract_volumes(&self, klines: &[Kline]) -> Vec<f64> {
        klines.iter().map(|k| k.volume).collect()
    }

    /// Calculate returns from price series.
    pub fn calculate_returns(&self, prices: &[f64]) -> Vec<f64> {
        if prices.len() < 2 {
            return vec![];
        }

        prices
            .windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect()
    }

    /// Normalize data using z-score.
    pub fn normalize_zscore(&self, data: &[f64]) -> Vec<f64> {
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
        let std = variance.sqrt();

        if std == 0.0 {
            return vec![0.0; data.len()];
        }

        data.iter().map(|x| (x - mean) / std).collect()
    }

    /// Normalize data to [0, 1] range.
    pub fn normalize_minmax(&self, data: &[f64]) -> Vec<f64> {
        let min = data.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let range = max - min;
        if range == 0.0 {
            return vec![0.5; data.len()];
        }

        data.iter().map(|x| (x - min) / range).collect()
    }

    /// Calculate rolling mean.
    pub fn rolling_mean(&self, data: &[f64], window: usize) -> Vec<f64> {
        if data.len() < window {
            return vec![];
        }

        data.windows(window)
            .map(|w| w.iter().sum::<f64>() / window as f64)
            .collect()
    }

    /// Calculate rolling standard deviation.
    pub fn rolling_std(&self, data: &[f64], window: usize) -> Vec<f64> {
        if data.len() < window {
            return vec![];
        }

        data.windows(window)
            .map(|w| {
                let mean = w.iter().sum::<f64>() / window as f64;
                let variance = w.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / window as f64;
                variance.sqrt()
            })
            .collect()
    }

    /// Convert Vec to ndarray Array1.
    pub fn to_array(&self, data: Vec<f64>) -> Array1<f64> {
        Array1::from_vec(data)
    }

    /// Create feature matrix from klines.
    pub fn create_features(&self, klines: &[Kline]) -> Vec<Vec<f64>> {
        let closes = self.extract_closes(klines);
        let highs = self.extract_highs(klines);
        let lows = self.extract_lows(klines);
        let volumes = self.extract_volumes(klines);

        let returns = self.calculate_returns(&closes);
        let vol_ratio = self.calculate_volume_ratio(&volumes);

        let n = klines.len();
        let mut features = Vec::with_capacity(n);

        for i in 0..n {
            let mut row = vec![
                closes[i],
                highs[i],
                lows[i],
                volumes[i],
            ];

            if i > 0 {
                row.push(returns[i - 1]);
            } else {
                row.push(0.0);
            }

            if i < vol_ratio.len() {
                row.push(vol_ratio[i]);
            } else {
                row.push(1.0);
            }

            features.push(row);
        }

        features
    }

    /// Calculate volume ratio (current / average).
    fn calculate_volume_ratio(&self, volumes: &[f64]) -> Vec<f64> {
        if volumes.is_empty() {
            return vec![];
        }

        let window = self.lookback_window.min(volumes.len());
        let mut ratios = Vec::with_capacity(volumes.len());

        for i in 0..volumes.len() {
            let start = i.saturating_sub(window);
            let avg: f64 = volumes[start..=i].iter().sum::<f64>()
                / (i - start + 1) as f64;

            if avg > 0.0 {
                ratios.push(volumes[i] / avg);
            } else {
                ratios.push(1.0);
            }
        }

        ratios
    }
}

impl Default for DataProcessor {
    fn default() -> Self {
        Self::new(20)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_returns() {
        let processor = DataProcessor::new(20);
        let prices = vec![100.0, 105.0, 103.0];
        let returns = processor.calculate_returns(&prices);

        assert_eq!(returns.len(), 2);
        assert!((returns[0] - 0.05).abs() < 1e-10);
    }

    #[test]
    fn test_normalize_zscore() {
        let processor = DataProcessor::new(20);
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let normalized = processor.normalize_zscore(&data);

        // Mean should be ~0
        let mean: f64 = normalized.iter().sum::<f64>() / normalized.len() as f64;
        assert!(mean.abs() < 1e-10);
    }

    #[test]
    fn test_rolling_mean() {
        let processor = DataProcessor::new(20);
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let rolling = processor.rolling_mean(&data, 3);

        assert_eq!(rolling.len(), 3);
        assert!((rolling[0] - 2.0).abs() < 1e-10);
        assert!((rolling[1] - 3.0).abs() < 1e-10);
        assert!((rolling[2] - 4.0).abs() < 1e-10);
    }
}
