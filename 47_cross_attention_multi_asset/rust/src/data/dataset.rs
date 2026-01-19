//! Dataset preparation for Cross-Attention Multi-Asset Trading
//!
//! Prepares data for training and inference.

use super::{Candle, DataConfig, Features, FeatureConfig, compute_features};
use std::collections::HashMap;

/// Data split structure
#[derive(Debug, Clone)]
pub struct DataSplit {
    /// Input features [n_samples, n_assets, seq_len, n_features]
    pub x: Vec<Vec<Vec<Vec<f64>>>>,
    /// Target returns [n_samples, n_assets]
    pub y: Vec<Vec<f64>>,
    /// Timestamps for each sample
    pub timestamps: Vec<i64>,
}

/// Cross-Attention Dataset
///
/// Prepares multi-asset data for the cross-attention model.
pub struct CrossAttentionDataset {
    /// Feature data per asset
    features: HashMap<String, Vec<Features>>,
    /// Asset symbols
    symbols: Vec<String>,
    /// Data configuration
    config: DataConfig,
    /// Feature configuration
    feature_config: FeatureConfig,
}

impl CrossAttentionDataset {
    /// Create a new dataset from candle data
    pub fn new(
        candle_data: HashMap<String, Vec<Candle>>,
        config: DataConfig,
        feature_config: FeatureConfig,
    ) -> Self {
        let symbols: Vec<String> = candle_data.keys().cloned().collect();

        let features: HashMap<String, Vec<Features>> = candle_data
            .iter()
            .map(|(symbol, candles)| {
                (symbol.clone(), compute_features(candles, &feature_config))
            })
            .collect();

        Self {
            features,
            symbols,
            config,
            feature_config,
        }
    }

    /// Get asset symbols
    pub fn symbols(&self) -> &[String] {
        &self.symbols
    }

    /// Get number of assets
    pub fn n_assets(&self) -> usize {
        self.symbols.len()
    }

    /// Get number of features
    pub fn n_features(&self) -> usize {
        6 // Fixed: log_return, volume_ratio, volatility, rsi, macd, momentum
    }

    /// Prepare data for training
    ///
    /// Returns (X, y) where:
    /// - X: [n_samples, n_assets, lookback, n_features]
    /// - y: [n_samples, n_assets] (future returns)
    pub fn prepare(&self) -> Option<(Vec<Vec<Vec<Vec<f64>>>>, Vec<Vec<f64>>, Vec<i64>)> {
        // Validate config to avoid panics and divide-by-zero
        if self.config.lookback == 0 || self.config.horizon == 0 {
            return None;
        }

        // Find minimum length across all assets
        let min_len = self.features.values().map(|f| f.len()).min()?;

        if min_len < self.config.lookback + self.config.horizon {
            return None;
        }

        let n_samples = min_len - self.config.lookback - self.config.horizon + 1;
        let n_assets = self.symbols.len();
        let lookback = self.config.lookback;
        let horizon = self.config.horizon;

        let mut x = Vec::with_capacity(n_samples);
        let mut y = Vec::with_capacity(n_samples);
        let mut timestamps = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            let start_idx = i;
            let end_idx = i + lookback;

            // Build input sequence for all assets
            let mut sample_x = Vec::with_capacity(n_assets);
            let mut sample_y = Vec::with_capacity(n_assets);

            for symbol in &self.symbols {
                let features = &self.features[symbol];

                // Input: lookback window of features
                let asset_x: Vec<Vec<f64>> = (start_idx..end_idx)
                    .map(|j| features[j].to_array().to_vec())
                    .collect();
                sample_x.push(asset_x);

                // Target: average return over horizon
                let future_returns: f64 = (end_idx..end_idx + horizon)
                    .map(|j| features[j].log_return)
                    .sum::<f64>()
                    / horizon as f64;
                sample_y.push(future_returns);
            }

            x.push(sample_x);
            y.push(sample_y);
            timestamps.push(i as i64 * 3600000); // Placeholder timestamp
        }

        Some((x, y, timestamps))
    }

    /// Split data into train/val/test sets
    pub fn split(&self) -> Option<(DataSplit, DataSplit, DataSplit)> {
        // Validate ratios to avoid panics and out-of-bounds
        if !(0.0..=1.0).contains(&self.config.train_ratio)
            || !(0.0..=1.0).contains(&self.config.val_ratio)
            || self.config.train_ratio + self.config.val_ratio > 1.0
        {
            return None;
        }

        let (x, y, timestamps) = self.prepare()?;

        let n = x.len();
        if n == 0 {
            return None;
        }

        let train_end = (n as f64 * self.config.train_ratio) as usize;
        let val_end = (train_end + (n as f64 * self.config.val_ratio) as usize).min(n);

        let train = DataSplit {
            x: x[..train_end].to_vec(),
            y: y[..train_end].to_vec(),
            timestamps: timestamps[..train_end].to_vec(),
        };

        let val = DataSplit {
            x: x[train_end..val_end].to_vec(),
            y: y[train_end..val_end].to_vec(),
            timestamps: timestamps[train_end..val_end].to_vec(),
        };

        let test = DataSplit {
            x: x[val_end..].to_vec(),
            y: y[val_end..].to_vec(),
            timestamps: timestamps[val_end..].to_vec(),
        };

        Some((train, val, test))
    }
}

/// Convert nested vectors to flat array for tensor creation
pub fn flatten_to_tensor_data(data: &[Vec<Vec<Vec<f64>>>]) -> Vec<f32> {
    data.iter()
        .flat_map(|sample| {
            sample.iter().flat_map(|asset| {
                asset
                    .iter()
                    .flat_map(|timestep| timestep.iter().map(|&v| v as f32))
            })
        })
        .collect()
}

/// Get tensor shape from data
pub fn get_tensor_shape(data: &[Vec<Vec<Vec<f64>>>]) -> (usize, usize, usize, usize) {
    let n_samples = data.len();
    let n_assets = data.get(0).map(|s| s.len()).unwrap_or(0);
    let seq_len = data.get(0).and_then(|s| s.get(0)).map(|a| a.len()).unwrap_or(0);
    let n_features = data
        .get(0)
        .and_then(|s| s.get(0))
        .and_then(|a| a.get(0))
        .map(|t| t.len())
        .unwrap_or(0);

    (n_samples, n_assets, seq_len, n_features)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_candles(n: usize) -> HashMap<String, Vec<Candle>> {
        let symbols = vec!["BTCUSDT", "ETHUSDT", "SOLUSDT"];
        let mut result = HashMap::new();

        for symbol in symbols {
            let candles: Vec<Candle> = (0..n)
                .map(|i| {
                    let base = 100.0 + (i as f64 * 0.1).sin() * 10.0;
                    Candle {
                        timestamp: i as i64 * 3600000,
                        open: base,
                        high: base + 1.0,
                        low: base - 1.0,
                        close: base + 0.5,
                        volume: 1000.0 + (i as f64 * 100.0),
                        turnover: 100000.0,
                    }
                })
                .collect();
            result.insert(symbol.to_string(), candles);
        }

        result
    }

    #[test]
    fn test_dataset_creation() {
        let candles = create_test_candles(500);
        let config = DataConfig::default();
        let feature_config = FeatureConfig::default();

        let dataset = CrossAttentionDataset::new(candles, config, feature_config);

        assert_eq!(dataset.n_assets(), 3);
        assert_eq!(dataset.n_features(), 6);
    }

    #[test]
    fn test_data_preparation() {
        let candles = create_test_candles(500);
        let config = DataConfig {
            lookback: 50,
            horizon: 10,
            train_ratio: 0.7,
            val_ratio: 0.15,
        };
        let feature_config = FeatureConfig::default();

        let dataset = CrossAttentionDataset::new(candles, config, feature_config);
        let result = dataset.prepare();

        assert!(result.is_some());
        let (x, y, _) = result.unwrap();

        assert!(!x.is_empty());
        assert_eq!(x.len(), y.len());

        // Check shape of first sample
        assert_eq!(x[0].len(), 3); // n_assets
        assert_eq!(x[0][0].len(), 50); // lookback
        assert_eq!(x[0][0][0].len(), 6); // n_features
    }

    #[test]
    fn test_data_split() {
        let candles = create_test_candles(500);
        let config = DataConfig {
            lookback: 50,
            horizon: 10,
            train_ratio: 0.7,
            val_ratio: 0.15,
        };
        let feature_config = FeatureConfig::default();

        let dataset = CrossAttentionDataset::new(candles, config, feature_config);
        let splits = dataset.split();

        assert!(splits.is_some());
        let (train, val, test) = splits.unwrap();

        assert!(!train.x.is_empty());
        assert!(!val.x.is_empty());
        assert!(!test.x.is_empty());

        // Check proportions (approximately)
        let total = train.x.len() + val.x.len() + test.x.len();
        let train_ratio = train.x.len() as f64 / total as f64;
        let val_ratio = val.x.len() as f64 / total as f64;

        assert!((train_ratio - 0.7).abs() < 0.05);
        assert!((val_ratio - 0.15).abs() < 0.05);
    }

    #[test]
    fn test_flatten_tensor_data() {
        let data = vec![vec![vec![vec![1.0, 2.0], vec![3.0, 4.0]]]];
        let flat = flatten_to_tensor_data(&data);

        assert_eq!(flat, vec![1.0f32, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_get_tensor_shape() {
        let data = vec![
            vec![
                vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]],
                vec![vec![7.0, 8.0, 9.0], vec![10.0, 11.0, 12.0]],
            ],
        ];

        let shape = get_tensor_shape(&data);
        assert_eq!(shape, (1, 2, 2, 3));
    }
}
