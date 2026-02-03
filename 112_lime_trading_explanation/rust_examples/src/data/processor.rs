//! Data processing utilities for trading data.

use crate::api::bybit::Candle;
use ndarray::{Array1, Array2};

/// Data processor for converting candle data to feature matrices
pub struct DataProcessor {
    /// Feature names
    pub feature_names: Vec<String>,
}

impl DataProcessor {
    /// Create a new data processor with default features
    pub fn new() -> Self {
        Self {
            feature_names: vec![
                "return_1d".to_string(),
                "return_5d".to_string(),
                "rsi_14".to_string(),
                "macd".to_string(),
                "macd_signal".to_string(),
                "bb_position".to_string(),
                "volatility_20d".to_string(),
                "volume_ratio_20".to_string(),
                "price_sma_20_ratio".to_string(),
                "price_sma_50_ratio".to_string(),
            ],
        }
    }

    /// Convert candles to a feature matrix
    ///
    /// # Arguments
    /// * `candles` - Vector of candle data
    ///
    /// # Returns
    /// Tuple of (feature matrix, target vector, valid indices)
    pub fn prepare_features(&self, candles: &[Candle]) -> (Array2<f64>, Array1<f64>, Vec<usize>) {
        let n = candles.len();
        if n < 60 {
            return (
                Array2::zeros((0, self.feature_names.len())),
                Array1::zeros(0),
                vec![],
            );
        }

        let closes: Vec<f64> = candles.iter().map(|c| c.close).collect();
        let volumes: Vec<f64> = candles.iter().map(|c| c.volume).collect();

        // Compute features
        let returns_1d = self.compute_returns(&closes, 1);
        let returns_5d = self.compute_returns(&closes, 5);
        let rsi_14 = self.compute_rsi(&closes, 14);
        let (macd, macd_signal) = self.compute_macd(&closes, 12, 26, 9);
        let bb_position = self.compute_bollinger_position(&closes, 20, 2.0);
        let volatility_20d = self.compute_volatility(&closes, 20);
        let volume_ratio_20 = self.compute_volume_ratio(&volumes, 20);
        let price_sma_20_ratio = self.compute_price_sma_ratio(&closes, 20);
        let price_sma_50_ratio = self.compute_price_sma_ratio(&closes, 50);

        // Find valid indices (where all features are computed)
        let start_idx = 60; // Skip first 60 rows for feature computation
        let end_idx = n - 1; // Skip last row for target computation

        let mut valid_indices = Vec::new();
        let mut features = Vec::new();
        let mut targets = Vec::new();

        for i in start_idx..end_idx {
            if returns_1d[i].is_finite()
                && returns_5d[i].is_finite()
                && rsi_14[i].is_finite()
                && macd[i].is_finite()
                && volatility_20d[i].is_finite()
            {
                features.push(vec![
                    returns_1d[i],
                    returns_5d[i],
                    rsi_14[i],
                    macd[i],
                    macd_signal[i],
                    bb_position[i],
                    volatility_20d[i],
                    volume_ratio_20[i],
                    price_sma_20_ratio[i],
                    price_sma_50_ratio[i],
                ]);

                // Target: 1 if price goes up tomorrow, 0 otherwise
                let target = if closes[i + 1] > closes[i] { 1.0 } else { 0.0 };
                targets.push(target);
                valid_indices.push(i);
            }
        }

        let n_samples = features.len();
        let n_features = self.feature_names.len();

        let feature_matrix = if n_samples > 0 {
            Array2::from_shape_vec(
                (n_samples, n_features),
                features.into_iter().flatten().collect(),
            )
            .unwrap()
        } else {
            Array2::zeros((0, n_features))
        };

        let target_vector = Array1::from_vec(targets);

        (feature_matrix, target_vector, valid_indices)
    }

    /// Compute returns over a given period
    fn compute_returns(&self, prices: &[f64], period: usize) -> Vec<f64> {
        let n = prices.len();
        let mut returns = vec![f64::NAN; n];

        for i in period..n {
            returns[i] = (prices[i] / prices[i - period]) - 1.0;
        }

        returns
    }

    /// Compute RSI indicator
    fn compute_rsi(&self, prices: &[f64], period: usize) -> Vec<f64> {
        let n = prices.len();
        let mut rsi = vec![f64::NAN; n];

        if n <= period {
            return rsi;
        }

        let mut gains = vec![0.0; n];
        let mut losses = vec![0.0; n];

        for i in 1..n {
            let change = prices[i] - prices[i - 1];
            if change > 0.0 {
                gains[i] = change;
            } else {
                losses[i] = -change;
            }
        }

        // Initial averages
        let mut avg_gain: f64 = gains[1..=period].iter().sum::<f64>() / period as f64;
        let mut avg_loss: f64 = losses[1..=period].iter().sum::<f64>() / period as f64;

        for i in period..n {
            if i > period {
                avg_gain = (avg_gain * (period - 1) as f64 + gains[i]) / period as f64;
                avg_loss = (avg_loss * (period - 1) as f64 + losses[i]) / period as f64;
            }

            if avg_loss == 0.0 {
                rsi[i] = 100.0;
            } else {
                let rs = avg_gain / avg_loss;
                rsi[i] = 100.0 - (100.0 / (1.0 + rs));
            }
        }

        rsi
    }

    /// Compute MACD indicator
    fn compute_macd(
        &self,
        prices: &[f64],
        fast: usize,
        slow: usize,
        signal: usize,
    ) -> (Vec<f64>, Vec<f64>) {
        let n = prices.len();
        let mut macd = vec![f64::NAN; n];
        let mut macd_signal = vec![f64::NAN; n];

        let ema_fast = self.compute_ema(prices, fast);
        let ema_slow = self.compute_ema(prices, slow);

        for i in 0..n {
            if ema_fast[i].is_finite() && ema_slow[i].is_finite() {
                macd[i] = ema_fast[i] - ema_slow[i];
            }
        }

        let macd_valid: Vec<f64> = macd.iter().filter(|x| x.is_finite()).copied().collect();
        if !macd_valid.is_empty() {
            let signal_line = self.compute_ema(&macd_valid, signal);
            let mut j = 0;
            for i in 0..n {
                if macd[i].is_finite() && j < signal_line.len() {
                    macd_signal[i] = signal_line[j];
                    j += 1;
                }
            }
        }

        (macd, macd_signal)
    }

    /// Compute EMA
    fn compute_ema(&self, prices: &[f64], period: usize) -> Vec<f64> {
        let n = prices.len();
        let mut ema = vec![f64::NAN; n];

        if n < period {
            return ema;
        }

        let multiplier = 2.0 / (period as f64 + 1.0);

        // Initial SMA
        let sma: f64 = prices[..period].iter().sum::<f64>() / period as f64;
        ema[period - 1] = sma;

        for i in period..n {
            ema[i] = (prices[i] - ema[i - 1]) * multiplier + ema[i - 1];
        }

        ema
    }

    /// Compute Bollinger Band position
    fn compute_bollinger_position(&self, prices: &[f64], period: usize, std_dev: f64) -> Vec<f64> {
        let n = prices.len();
        let mut position = vec![f64::NAN; n];

        for i in period..n {
            let window = &prices[i - period..i];
            let sma: f64 = window.iter().sum::<f64>() / period as f64;
            let variance: f64 =
                window.iter().map(|x| (x - sma).powi(2)).sum::<f64>() / period as f64;
            let std = variance.sqrt();

            let upper = sma + std_dev * std;
            let lower = sma - std_dev * std;

            if upper != lower {
                position[i] = (prices[i] - lower) / (upper - lower);
            } else {
                position[i] = 0.5;
            }
        }

        position
    }

    /// Compute volatility
    fn compute_volatility(&self, prices: &[f64], period: usize) -> Vec<f64> {
        let returns = self.compute_returns(prices, 1);
        let n = returns.len();
        let mut volatility = vec![f64::NAN; n];

        for i in period..n {
            let window: Vec<f64> = returns[i - period..i]
                .iter()
                .filter(|x| x.is_finite())
                .copied()
                .collect();

            if !window.is_empty() {
                let mean: f64 = window.iter().sum::<f64>() / window.len() as f64;
                let variance: f64 =
                    window.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / window.len() as f64;
                volatility[i] = variance.sqrt() * (252.0_f64).sqrt(); // Annualized
            }
        }

        volatility
    }

    /// Compute volume ratio
    fn compute_volume_ratio(&self, volumes: &[f64], period: usize) -> Vec<f64> {
        let n = volumes.len();
        let mut ratio = vec![f64::NAN; n];

        for i in period..n {
            let avg_volume: f64 = volumes[i - period..i].iter().sum::<f64>() / period as f64;
            if avg_volume > 0.0 {
                ratio[i] = volumes[i] / avg_volume;
            }
        }

        ratio
    }

    /// Compute price to SMA ratio
    fn compute_price_sma_ratio(&self, prices: &[f64], period: usize) -> Vec<f64> {
        let n = prices.len();
        let mut ratio = vec![f64::NAN; n];

        for i in period..n {
            let sma: f64 = prices[i - period..i].iter().sum::<f64>() / period as f64;
            if sma > 0.0 {
                ratio[i] = prices[i] / sma;
            }
        }

        ratio
    }
}

impl Default for DataProcessor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_returns() {
        let processor = DataProcessor::new();
        let prices = vec![100.0, 102.0, 101.0, 105.0, 103.0];
        let returns = processor.compute_returns(&prices, 1);

        assert!(returns[0].is_nan());
        assert!((returns[1] - 0.02).abs() < 1e-10);
    }

    #[test]
    fn test_compute_rsi() {
        let processor = DataProcessor::new();
        let prices: Vec<f64> = (0..20).map(|i| 100.0 + i as f64).collect();
        let rsi = processor.compute_rsi(&prices, 14);

        // In an uptrend, RSI should be high
        assert!(rsi[19] > 70.0);
    }
}
