//! Data loading and sequence preparation for Nyströmformer

use ndarray::{Array2, Array3};

use crate::api::Kline;
use crate::data::Features;

/// Trading dataset for model training and inference
#[derive(Debug, Clone)]
pub struct TradingDataset {
    /// Feature sequences [num_samples, seq_len, num_features]
    pub x: Array3<f64>,
    /// Target values [num_samples, horizon]
    pub y: Array2<f64>,
    /// Corresponding prices for backtesting
    pub prices: Vec<f64>,
    /// Feature names
    pub feature_names: Vec<String>,
}

impl TradingDataset {
    /// Returns the number of samples
    pub fn len(&self) -> usize {
        self.x.dim().0
    }

    /// Returns true if empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns sequence length
    pub fn seq_len(&self) -> usize {
        self.x.dim().1
    }

    /// Returns number of features
    pub fn num_features(&self) -> usize {
        self.x.dim().2
    }

    /// Splits dataset into train/val/test
    pub fn split(
        &self,
        train_ratio: f64,
        val_ratio: f64,
    ) -> (TradingDataset, TradingDataset, TradingDataset) {
        let n = self.len();
        let train_end = (n as f64 * train_ratio) as usize;
        let val_end = train_end + (n as f64 * val_ratio) as usize;

        let train = self.slice(0, train_end);
        let val = self.slice(train_end, val_end);
        let test = self.slice(val_end, n);

        (train, val, test)
    }

    /// Extracts a slice of the dataset
    fn slice(&self, start: usize, end: usize) -> TradingDataset {
        let x = self.x.slice(ndarray::s![start..end, .., ..]).to_owned();
        let y = self.y.slice(ndarray::s![start..end, ..]).to_owned();
        let prices = self.prices[start..end].to_vec();

        TradingDataset {
            x,
            y,
            prices,
            feature_names: self.feature_names.clone(),
        }
    }

    /// Gets a batch of samples
    pub fn get_batch(&self, indices: &[usize]) -> (Array3<f64>, Array2<f64>) {
        let batch_size = indices.len();
        let seq_len = self.seq_len();
        let num_features = self.num_features();
        let horizon = self.y.dim().1;

        let mut x_batch = Array3::zeros((batch_size, seq_len, num_features));
        let mut y_batch = Array2::zeros((batch_size, horizon));

        for (i, &idx) in indices.iter().enumerate() {
            for t in 0..seq_len {
                for f in 0..num_features {
                    x_batch[[i, t, f]] = self.x[[idx, t, f]];
                }
            }
            for h in 0..horizon {
                y_batch[[i, h]] = self.y[[idx, h]];
            }
        }

        (x_batch, y_batch)
    }
}

/// Sequence data loader for long sequences
#[derive(Debug, Clone)]
pub struct SequenceLoader {
    /// RSI period
    pub rsi_period: usize,
    /// ATR period
    pub atr_period: usize,
    /// Volatility period
    pub volatility_period: usize,
    /// Bollinger Bands period
    pub bb_period: usize,
    /// MACD periods (fast, slow, signal)
    pub macd_periods: (usize, usize, usize),
}

impl Default for SequenceLoader {
    fn default() -> Self {
        Self::new()
    }
}

impl SequenceLoader {
    /// Creates a new sequence loader with default parameters
    pub fn new() -> Self {
        Self {
            rsi_period: 14,
            atr_period: 14,
            volatility_period: 20,
            bb_period: 20,
            macd_periods: (12, 26, 9),
        }
    }

    /// Creates loader with custom parameters
    pub fn with_params(
        rsi_period: usize,
        atr_period: usize,
        volatility_period: usize,
        bb_period: usize,
    ) -> Self {
        Self {
            rsi_period,
            atr_period,
            volatility_period,
            bb_period,
            macd_periods: (12, 26, 9),
        }
    }

    /// Prepares dataset from klines
    ///
    /// # Arguments
    /// * `klines` - Raw OHLCV data
    /// * `seq_len` - Sequence length for the model
    /// * `horizon` - Prediction horizon
    ///
    /// # Returns
    /// TradingDataset ready for model training
    pub fn prepare_dataset(
        &self,
        klines: &[Kline],
        seq_len: usize,
        horizon: usize,
    ) -> Result<TradingDataset, String> {
        if klines.len() < seq_len + horizon + 100 {
            return Err(format!(
                "Not enough data: need at least {} candles, got {}",
                seq_len + horizon + 100,
                klines.len()
            ));
        }

        // Extract price data
        let closes: Vec<f64> = klines.iter().map(|k| k.close).collect();
        let volumes: Vec<f64> = klines.iter().map(|k| k.volume).collect();

        // Calculate features
        let log_returns = Features::log_returns(&closes);
        let rsi = Features::rsi(&closes, self.rsi_period);
        let atr = Features::atr(klines, self.atr_period);
        let volatility = Features::rolling_volatility(&closes, self.volatility_period);
        let bb_pct = Features::bollinger_pct_b(&closes, self.bb_period, 2.0);
        let vol_ratio = Features::volume_ratio(&volumes, 20);
        let (_macd, _signal, hist) = Features::macd(
            &closes,
            self.macd_periods.0,
            self.macd_periods.1,
            self.macd_periods.2,
        );

        // Normalize features
        let log_returns_norm = Features::clip(&log_returns, -0.1, 0.1);
        let rsi_norm = rsi; // Already 0-1
        let atr_norm = self.normalize_by_price(&atr, &closes);
        let volatility_norm = Features::clip(&volatility, 0.0, 0.1);
        let bb_norm = Features::clip(&bb_pct, 0.0, 1.0);
        let vol_ratio_norm = Features::clip(
            &vol_ratio.iter().map(|&v| (v - 1.0) / 2.0).collect::<Vec<_>>(),
            -1.0,
            1.0,
        );

        // MACD normalization using z-score
        let macd_norm = Features::zscore(&hist, 20);
        let macd_norm = Features::clip(&macd_norm, -3.0, 3.0)
            .iter()
            .map(|&v| v / 6.0 + 0.5) // Scale to [0, 1]
            .collect::<Vec<_>>();

        let feature_names = vec![
            "log_return".to_string(),
            "rsi".to_string(),
            "atr_norm".to_string(),
            "volatility".to_string(),
            "bb_pct".to_string(),
            "vol_ratio".to_string(),
            "macd_hist".to_string(),
        ];

        let num_features = feature_names.len();

        // Skip initial period where indicators are undefined
        let start_idx = self.bb_period.max(self.atr_period).max(self.macd_periods.1) + 10;
        let n_samples = klines.len() - start_idx - seq_len - horizon;

        if n_samples == 0 {
            return Err("No valid samples after preprocessing".to_string());
        }

        let mut x = Array3::zeros((n_samples, seq_len, num_features));
        let mut y = Array2::zeros((n_samples, horizon));
        let mut prices = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            let base_idx = start_idx + i;

            // Fill feature sequence
            for t in 0..seq_len {
                let idx = base_idx + t;
                x[[i, t, 0]] = log_returns_norm[idx];
                x[[i, t, 1]] = rsi_norm[idx];
                x[[i, t, 2]] = atr_norm[idx];
                x[[i, t, 3]] = volatility_norm[idx];
                x[[i, t, 4]] = bb_norm[idx];
                x[[i, t, 5]] = vol_ratio_norm[idx];
                x[[i, t, 6]] = macd_norm[idx];
            }

            // Fill target (future returns)
            let pred_start = base_idx + seq_len;
            for h in 0..horizon {
                let future_idx = pred_start + h;
                if future_idx < closes.len() {
                    y[[i, h]] = log_returns[future_idx];
                }
            }

            // Store corresponding price
            prices.push(closes[base_idx + seq_len - 1]);
        }

        Ok(TradingDataset {
            x,
            y,
            prices,
            feature_names,
        })
    }

    /// Normalizes values by price
    fn normalize_by_price(&self, values: &[f64], prices: &[f64]) -> Vec<f64> {
        values
            .iter()
            .zip(prices.iter())
            .map(|(&v, &p)| if p > 0.0 { v / p } else { 0.0 })
            .collect()
    }

    /// Generates synthetic data for testing
    pub fn generate_synthetic(
        &self,
        n_samples: usize,
        seq_len: usize,
        num_features: usize,
        horizon: usize,
    ) -> TradingDataset {
        use std::f64::consts::PI;

        let mut x = Array3::zeros((n_samples, seq_len, num_features));
        let mut y = Array2::zeros((n_samples, horizon));
        let mut prices = Vec::with_capacity(n_samples);

        // Generate synthetic price series
        let total_len = n_samples + seq_len + horizon;
        let mut price = 100.0;
        let mut all_prices = Vec::with_capacity(total_len);

        for i in 0..total_len {
            let trend = 0.0001 * (i as f64 / total_len as f64);
            let cycle = 0.02 * (2.0 * PI * i as f64 / 50.0).sin();
            let noise = rand_normal() * 0.01;

            let return_ = trend + cycle * 0.1 + noise;
            price *= 1.0 + return_;
            all_prices.push(price);
        }

        for i in 0..n_samples {
            // Generate feature sequence
            for t in 0..seq_len {
                let idx = i + t;
                // Feature 0: normalized return
                x[[i, t, 0]] = if idx > 0 {
                    (all_prices[idx] / all_prices[idx - 1] - 1.0).max(-0.1).min(0.1)
                } else {
                    0.0
                };

                // Feature 1: RSI-like (normalized momentum)
                x[[i, t, 1]] = (x[[i, t, 0]] * 10.0).tanh() * 0.5 + 0.5;

                // Feature 2: Volatility proxy
                x[[i, t, 2]] = (rand_normal() * 0.01).abs();

                // Feature 3: Random feature
                x[[i, t, 3]] = rand::random::<f64>();

                // Fill remaining features with noise
                for f in 4..num_features {
                    x[[i, t, f]] = rand::random::<f64>() * 0.1;
                }
            }

            // Generate target returns
            let pred_start = i + seq_len;
            for h in 0..horizon {
                let future_idx = pred_start + h;
                if future_idx < all_prices.len() && pred_start + h > 0 {
                    y[[i, h]] = (all_prices[future_idx] / all_prices[future_idx - 1] - 1.0)
                        .max(-0.1)
                        .min(0.1);
                }
            }

            prices.push(all_prices[i + seq_len - 1]);
        }

        let feature_names = (0..num_features)
            .map(|i| format!("feature_{}", i))
            .collect();

        TradingDataset {
            x,
            y,
            prices,
            feature_names,
        }
    }
}

/// Generates a random number from standard normal distribution
fn rand_normal() -> f64 {
    use std::f64::consts::PI;
    let u1: f64 = rand::random::<f64>().max(1e-10);
    let u2: f64 = rand::random();
    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_synthetic() {
        let loader = SequenceLoader::new();
        let dataset = loader.generate_synthetic(100, 64, 6, 12);

        assert_eq!(dataset.len(), 100);
        assert_eq!(dataset.seq_len(), 64);
        assert_eq!(dataset.num_features(), 6);
        assert_eq!(dataset.y.dim().1, 12);
    }

    #[test]
    fn test_dataset_split() {
        let loader = SequenceLoader::new();
        let dataset = loader.generate_synthetic(100, 64, 6, 12);

        let (train, val, test) = dataset.split(0.7, 0.15);

        assert_eq!(train.len(), 70);
        assert_eq!(val.len(), 15);
        assert_eq!(test.len(), 15);
    }

    #[test]
    fn test_get_batch() {
        let loader = SequenceLoader::new();
        let dataset = loader.generate_synthetic(100, 64, 6, 12);

        let indices = vec![0, 10, 20, 30];
        let (x_batch, y_batch) = dataset.get_batch(&indices);

        assert_eq!(x_batch.dim(), (4, 64, 6));
        assert_eq!(y_batch.dim(), (4, 12));
    }

    #[test]
    fn test_prepare_from_klines() {
        let loader = SequenceLoader::new();

        // Generate synthetic klines
        let mut klines = Vec::new();
        let mut price = 100.0;

        for i in 0..1000 {
            let change = rand_normal() * 0.01;
            price *= 1.0 + change;

            klines.push(Kline {
                open_time: i as i64 * 60000,
                open: price * (1.0 - 0.001),
                high: price * (1.0 + 0.005),
                low: price * (1.0 - 0.005),
                close: price,
                volume: 1000.0 + rand_normal().abs() * 500.0,
                quote_volume: price * 1000.0,
            });
        }

        let result = loader.prepare_dataset(&klines, 128, 24);
        assert!(result.is_ok());

        let dataset = result.unwrap();
        assert!(dataset.len() > 0);
        assert_eq!(dataset.seq_len(), 128);
        assert_eq!(dataset.num_features(), 7);
    }
}
