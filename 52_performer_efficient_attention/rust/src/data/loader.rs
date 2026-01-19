//! Data loading utilities

use crate::api::Kline;
use crate::data::{Dataset, Features};

/// Data loader for preparing datasets
pub struct DataLoader {
    /// Sequence length for windowing
    seq_len: usize,
    /// Target prediction horizon
    target_horizon: usize,
    /// Whether to normalize features
    normalize: bool,
}

impl DataLoader {
    /// Create new data loader with default settings
    pub fn new() -> Self {
        Self {
            seq_len: 168,       // 7 days of hourly data
            target_horizon: 24, // 24 hours ahead
            normalize: true,
        }
    }

    /// Create data loader with custom settings
    pub fn with_config(seq_len: usize, target_horizon: usize, normalize: bool) -> Self {
        Self {
            seq_len,
            target_horizon,
            normalize,
        }
    }

    /// Set sequence length
    pub fn seq_len(mut self, seq_len: usize) -> Self {
        self.seq_len = seq_len;
        self
    }

    /// Set target horizon
    pub fn target_horizon(mut self, target_horizon: usize) -> Self {
        self.target_horizon = target_horizon;
        self
    }

    /// Set normalization
    pub fn normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    /// Prepare dataset from klines
    pub fn prepare_dataset(&self, klines: &[Kline]) -> Result<Dataset, String> {
        if klines.len() < self.seq_len + 50 {
            return Err(format!(
                "Need at least {} klines, got {}",
                self.seq_len + 50,
                klines.len()
            ));
        }

        let mut features = Features::from_klines(klines, self.target_horizon);

        if features.is_empty() {
            return Err("Failed to compute features".to_string());
        }

        if self.normalize {
            features.normalize();
        }

        let dataset = Dataset::from_features(&features, self.seq_len);

        if dataset.is_empty() {
            return Err("Failed to create dataset".to_string());
        }

        Ok(dataset)
    }

    /// Prepare features only (without windowing)
    pub fn prepare_features(&self, klines: &[Kline]) -> Result<Features, String> {
        if klines.len() < 51 {
            return Err(format!(
                "Need at least 51 klines, got {}",
                klines.len()
            ));
        }

        let mut features = Features::from_klines(klines, self.target_horizon);

        if features.is_empty() {
            return Err("Failed to compute features".to_string());
        }

        if self.normalize {
            features.normalize();
        }

        Ok(features)
    }

    /// Load and prepare data for multiple symbols
    pub fn prepare_multi_dataset(
        &self,
        multi_klines: &[(String, Vec<Kline>)],
    ) -> Result<Vec<(String, Dataset)>, String> {
        let mut datasets = Vec::new();

        for (symbol, klines) in multi_klines {
            match self.prepare_dataset(klines) {
                Ok(dataset) => datasets.push((symbol.clone(), dataset)),
                Err(e) => {
                    tracing::warn!("Failed to prepare dataset for {}: {}", symbol, e);
                }
            }
        }

        if datasets.is_empty() {
            return Err("Failed to prepare any datasets".to_string());
        }

        Ok(datasets)
    }
}

impl Default for DataLoader {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_klines(n: usize) -> Vec<Kline> {
        (0..n)
            .map(|i| {
                let base_price = 100.0 + (i as f64 * 0.1);
                Kline {
                    timestamp: i as u64 * 3600000,
                    open: base_price,
                    high: base_price * 1.01,
                    low: base_price * 0.99,
                    close: base_price * (1.0 + (i % 2) as f64 * 0.005),
                    volume: 1000.0 + (i % 10) as f64 * 100.0,
                    turnover: 100000.0,
                }
            })
            .collect()
    }

    #[test]
    fn test_data_loader_default() {
        let loader = DataLoader::new();
        let klines = create_test_klines(500);

        let dataset = loader.prepare_dataset(&klines).unwrap();

        assert!(dataset.len() > 0);
        assert_eq!(dataset.seq_len, 168);
    }

    #[test]
    fn test_data_loader_custom() {
        let loader = DataLoader::with_config(64, 12, true);
        let klines = create_test_klines(300);

        let dataset = loader.prepare_dataset(&klines).unwrap();

        assert!(dataset.len() > 0);
        assert_eq!(dataset.seq_len, 64);
    }

    #[test]
    fn test_data_loader_builder() {
        let loader = DataLoader::new()
            .seq_len(32)
            .target_horizon(6)
            .normalize(false);

        let klines = create_test_klines(200);
        let dataset = loader.prepare_dataset(&klines).unwrap();

        assert_eq!(dataset.seq_len, 32);
    }

    #[test]
    fn test_data_loader_insufficient_data() {
        let loader = DataLoader::new();
        let klines = create_test_klines(50);

        let result = loader.prepare_dataset(&klines);
        assert!(result.is_err());
    }

    #[test]
    fn test_prepare_features() {
        let loader = DataLoader::new();
        let klines = create_test_klines(200);

        let features = loader.prepare_features(&klines).unwrap();
        assert!(!features.is_empty());
    }
}
