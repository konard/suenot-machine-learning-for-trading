//! Technical feature engineering for trading.
//!
//! This module provides functions to compute various technical indicators
//! from price data for use in machine learning models.

use crate::data::bybit::Kline;

/// Feature generator for trading data
pub struct FeatureGenerator {
    /// Lookback window for indicators
    window: usize,
}

impl FeatureGenerator {
    /// Create a new feature generator
    pub fn new(window: usize) -> Self {
        Self { window }
    }

    /// Create with default window of 20
    pub fn default_window() -> Self {
        Self { window: 20 }
    }

    /// Compute all features from kline data
    ///
    /// Returns a vector of feature vectors, one for each valid data point
    pub fn compute_features(&self, klines: &[Kline]) -> Vec<Vec<f64>> {
        if klines.len() < self.window + 10 {
            return Vec::new();
        }

        let closes: Vec<f64> = klines.iter().map(|k| k.close).collect();
        let _highs: Vec<f64> = klines.iter().map(|k| k.high).collect();
        let _lows: Vec<f64> = klines.iter().map(|k| k.low).collect();
        let volumes: Vec<f64> = klines.iter().map(|k| k.volume).collect();

        // Compute individual features
        let returns_1 = self.compute_returns(&closes, 1);
        let returns_5 = self.compute_returns(&closes, 5);
        let returns_10 = self.compute_returns(&closes, 10);
        let sma_ratio = self.compute_sma_ratio(&closes);
        let ema_ratio = self.compute_ema_ratio(&closes);
        let volatility = self.compute_volatility(&closes);
        let momentum = self.compute_momentum(&closes);
        let rsi = self.compute_rsi(&closes);
        let macd = self.compute_macd(&closes);
        let bb_position = self.compute_bollinger_position(&closes);
        let volume_sma_ratio = self.compute_volume_sma_ratio(&volumes);

        // Combine features
        let start_idx = self.window + 10;
        let mut features = Vec::new();

        for i in start_idx..closes.len() {
            let idx = i - start_idx;
            if idx < returns_1.len()
                && idx < returns_5.len()
                && idx < returns_10.len()
                && idx < sma_ratio.len()
                && idx < ema_ratio.len()
                && idx < volatility.len()
                && idx < momentum.len()
                && idx < rsi.len()
                && idx < macd.len()
                && idx < bb_position.len()
                && idx < volume_sma_ratio.len()
            {
                features.push(vec![
                    returns_1[idx],
                    returns_5[idx],
                    returns_10[idx],
                    sma_ratio[idx],
                    ema_ratio[idx],
                    volatility[idx],
                    momentum[idx],
                    rsi[idx],
                    macd[idx],
                    bb_position[idx],
                    volume_sma_ratio[idx],
                ]);
            }
        }

        features
    }

    /// Compute returns over n periods
    fn compute_returns(&self, prices: &[f64], n: usize) -> Vec<f64> {
        if prices.len() <= n {
            return Vec::new();
        }

        let start = self.window + 10 - n;
        prices[start..]
            .windows(n + 1)
            .map(|w| (w[n] / w[0]) - 1.0)
            .collect()
    }

    /// Compute SMA ratio (price / SMA)
    fn compute_sma_ratio(&self, prices: &[f64]) -> Vec<f64> {
        if prices.len() < self.window {
            return Vec::new();
        }

        let sma = self.compute_sma(prices, self.window);
        let start = self.window - 1;

        prices[start + 10..]
            .iter()
            .zip(sma[10..].iter())
            .map(|(p, s)| if *s > 0.0 { p / s } else { 1.0 })
            .collect()
    }

    /// Compute EMA ratio (price / EMA)
    fn compute_ema_ratio(&self, prices: &[f64]) -> Vec<f64> {
        let ema = self.compute_ema(prices, self.window);
        if ema.len() < 10 {
            return Vec::new();
        }

        let start = self.window + 10;
        prices[start..]
            .iter()
            .zip(ema[start - self.window..].iter())
            .map(|(p, e)| if *e > 0.0 { p / e } else { 1.0 })
            .collect()
    }

    /// Compute rolling volatility
    fn compute_volatility(&self, prices: &[f64]) -> Vec<f64> {
        if prices.len() < self.window + 1 {
            return Vec::new();
        }

        // First compute returns
        let returns: Vec<f64> = prices
            .windows(2)
            .map(|w| (w[1] / w[0]) - 1.0)
            .collect();

        // Then compute rolling std
        let start = self.window + 9;
        returns[start..]
            .windows(self.window)
            .map(|w| {
                let mean = w.iter().sum::<f64>() / w.len() as f64;
                let variance = w.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / w.len() as f64;
                variance.sqrt()
            })
            .collect()
    }

    /// Compute momentum (current price / price n periods ago - 1)
    fn compute_momentum(&self, prices: &[f64]) -> Vec<f64> {
        if prices.len() < self.window {
            return Vec::new();
        }

        let start = self.window + 10;
        (start..prices.len())
            .map(|i| (prices[i] / prices[i - self.window]) - 1.0)
            .collect()
    }

    /// Compute RSI (Relative Strength Index)
    fn compute_rsi(&self, prices: &[f64]) -> Vec<f64> {
        if prices.len() < self.window + 1 {
            return Vec::new();
        }

        let deltas: Vec<f64> = prices.windows(2).map(|w| w[1] - w[0]).collect();

        let gains: Vec<f64> = deltas.iter().map(|d| d.max(0.0)).collect();
        let losses: Vec<f64> = deltas.iter().map(|d| (-d).max(0.0)).collect();

        let start = self.window + 9;
        let mut rsi_values = Vec::new();

        for i in start..deltas.len() {
            let window_start = i - self.window + 1;
            let avg_gain: f64 = gains[window_start..=i].iter().sum::<f64>() / self.window as f64;
            let avg_loss: f64 = losses[window_start..=i].iter().sum::<f64>() / self.window as f64;

            let rsi = if avg_loss > 0.0 {
                100.0 - (100.0 / (1.0 + avg_gain / avg_loss))
            } else if avg_gain > 0.0 {
                100.0
            } else {
                50.0
            };

            rsi_values.push(rsi / 100.0); // Normalize to [0, 1]
        }

        rsi_values
    }

    /// Compute MACD (normalized by price)
    fn compute_macd(&self, prices: &[f64]) -> Vec<f64> {
        let ema12 = self.compute_ema(prices, 12);
        let ema26 = self.compute_ema(prices, 26);

        if ema12.len() < 26 || ema26.is_empty() {
            return Vec::new();
        }

        let start = self.window + 10;
        (start..prices.len().min(ema12.len() + 12).min(ema26.len() + 26))
            .filter_map(|i| {
                let idx12 = i.checked_sub(12)?;
                let idx26 = i.checked_sub(26)?;
                if idx12 < ema12.len() && idx26 < ema26.len() && prices[i] > 0.0 {
                    Some((ema12[idx12] - ema26[idx26]) / prices[i])
                } else {
                    None
                }
            })
            .collect()
    }

    /// Compute Bollinger Band position
    fn compute_bollinger_position(&self, prices: &[f64]) -> Vec<f64> {
        if prices.len() < self.window {
            return Vec::new();
        }

        let start = self.window + 10;
        (start..prices.len())
            .map(|i| {
                let window = &prices[i - self.window..i];
                let mean = window.iter().sum::<f64>() / self.window as f64;
                let variance = window.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / self.window as f64;
                let std = variance.sqrt();

                if std > 0.0 {
                    (prices[i] - mean) / (2.0 * std)
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// Compute volume SMA ratio
    fn compute_volume_sma_ratio(&self, volumes: &[f64]) -> Vec<f64> {
        if volumes.len() < self.window {
            return Vec::new();
        }

        let sma = self.compute_sma(volumes, self.window);
        let start = self.window + 9;

        volumes[start + 1..]
            .iter()
            .zip(sma[10..].iter())
            .map(|(v, s)| if *s > 0.0 { v / s } else { 1.0 })
            .collect()
    }

    /// Compute Simple Moving Average
    fn compute_sma(&self, data: &[f64], window: usize) -> Vec<f64> {
        if data.len() < window {
            return Vec::new();
        }

        data.windows(window)
            .map(|w| w.iter().sum::<f64>() / window as f64)
            .collect()
    }

    /// Compute Exponential Moving Average
    fn compute_ema(&self, data: &[f64], span: usize) -> Vec<f64> {
        if data.is_empty() {
            return Vec::new();
        }

        let alpha = 2.0 / (span as f64 + 1.0);
        let mut ema = Vec::with_capacity(data.len());
        ema.push(data[0]);

        for i in 1..data.len() {
            let value = alpha * data[i] + (1.0 - alpha) * ema[i - 1];
            ema.push(value);
        }

        ema
    }

    /// Get the number of features generated
    pub fn num_features(&self) -> usize {
        11 // returns_1, returns_5, returns_10, sma_ratio, ema_ratio, volatility, momentum, rsi, macd, bb_position, volume_sma_ratio
    }

    /// Get the lookback window
    pub fn window(&self) -> usize {
        self.window
    }
}

impl Default for FeatureGenerator {
    fn default() -> Self {
        Self::default_window()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::bybit::SimulatedDataGenerator;

    #[test]
    fn test_feature_generation() {
        let klines = SimulatedDataGenerator::generate_klines(100, 50000.0, 0.02);
        let generator = FeatureGenerator::new(20);
        let features = generator.compute_features(&klines);

        assert!(!features.is_empty());
        for feature_vec in &features {
            assert_eq!(feature_vec.len(), generator.num_features());
            for f in feature_vec {
                assert!(f.is_finite(), "Feature should be finite: {}", f);
            }
        }
    }

    #[test]
    fn test_sma_computation() {
        let generator = FeatureGenerator::new(5);
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let sma = generator.compute_sma(&data, 3);

        assert_eq!(sma.len(), 5);
        assert!((sma[0] - 2.0).abs() < 1e-10); // (1+2+3)/3
        assert!((sma[1] - 3.0).abs() < 1e-10); // (2+3+4)/3
    }

    #[test]
    fn test_ema_computation() {
        let generator = FeatureGenerator::new(5);
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ema = generator.compute_ema(&data, 3);

        assert_eq!(ema.len(), 5);
        assert!((ema[0] - 1.0).abs() < 1e-10);
        assert!(ema[4] > ema[0]); // Should trend upward with increasing data
    }

    #[test]
    fn test_insufficient_data() {
        let klines = SimulatedDataGenerator::generate_klines(10, 50000.0, 0.02);
        let generator = FeatureGenerator::new(20);
        let features = generator.compute_features(&klines);

        assert!(features.is_empty());
    }
}
