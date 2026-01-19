//! Data module for Cross-Attention Multi-Asset Trading
//!
//! This module provides:
//! - Bybit API client for cryptocurrency data
//! - Feature engineering functions
//! - Dataset preparation utilities

mod bybit;
mod features;
mod dataset;

pub use bybit::{BybitClient, Candle, BybitError};
pub use features::{compute_features, FeatureConfig, Features, normalize_features};
pub use dataset::{CrossAttentionDataset, DataSplit, flatten_to_tensor_data, get_tensor_shape};

use serde::{Deserialize, Serialize};

/// Data configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataConfig {
    /// Lookback window (number of time steps)
    pub lookback: usize,
    /// Prediction horizon
    pub horizon: usize,
    /// Training ratio
    pub train_ratio: f64,
    /// Validation ratio
    pub val_ratio: f64,
}

impl Default for DataConfig {
    fn default() -> Self {
        Self {
            lookback: 168,  // 7 days of hourly data
            horizon: 24,    // 1 day ahead
            train_ratio: 0.7,
            val_ratio: 0.15,
        }
    }
}

impl DataConfig {
    /// Compute test ratio
    pub fn test_ratio(&self) -> f64 {
        1.0 - self.train_ratio - self.val_ratio
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.lookback == 0 {
            return Err("lookback must be greater than 0".to_string());
        }
        if self.horizon == 0 {
            return Err("horizon must be greater than 0".to_string());
        }
        if self.train_ratio + self.val_ratio >= 1.0 {
            return Err("train_ratio + val_ratio must be less than 1.0".to_string());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = DataConfig::default();
        assert_eq!(config.lookback, 168);
        assert_eq!(config.horizon, 24);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_test_ratio() {
        let config = DataConfig::default();
        let test_ratio = config.test_ratio();
        assert!((test_ratio - 0.15).abs() < 1e-10);
    }
}
