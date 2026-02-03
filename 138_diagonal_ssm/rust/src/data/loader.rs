//! Data loader with builder pattern for constructing datasets from klines.

use anyhow::Result;
use log::info;
use ndarray::Array2;

use crate::api::types::Kline;
use crate::data::dataset::Dataset;
use crate::data::features::FeatureEngine;

/// Loads raw kline data and transforms it into training-ready datasets.
///
/// Uses a builder pattern to configure sequence length and prediction horizon.
#[derive(Debug, Clone)]
pub struct DataLoader {
    /// Length of the input sequence window.
    seq_len: usize,
    /// Number of steps ahead to predict.
    target_horizon: usize,
    /// Feature engine for computing technical indicators.
    feature_engine: FeatureEngine,
}

impl Default for DataLoader {
    fn default() -> Self {
        Self {
            seq_len: 60,
            target_horizon: 1,
            feature_engine: FeatureEngine::default(),
        }
    }
}

impl DataLoader {
    /// Create a new DataLoader with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the input sequence length.
    pub fn seq_len(mut self, seq_len: usize) -> Self {
        self.seq_len = seq_len;
        self
    }

    /// Set the target prediction horizon (number of steps ahead).
    pub fn target_horizon(mut self, horizon: usize) -> Self {
        self.target_horizon = horizon;
        self
    }

    /// Set a custom feature engine.
    pub fn feature_engine(mut self, engine: FeatureEngine) -> Self {
        self.feature_engine = engine;
        self
    }

    /// Build a dataset from kline data.
    ///
    /// Computes features, constructs sliding windows of length `seq_len`,
    /// and extracts targets at `target_horizon` steps ahead.
    ///
    /// Returns a [`Dataset`] with input sequences and target values.
    pub fn build_dataset(&self, klines: &[Kline]) -> Result<Dataset> {
        info!(
            "Building dataset from {} klines (seq_len={}, horizon={})",
            klines.len(),
            self.seq_len,
            self.target_horizon
        );

        let features = self.feature_engine.compute_all(klines)?;
        let (n_samples, n_features) = features.dim();

        let total_needed = self.seq_len + self.target_horizon;
        if n_samples < total_needed {
            anyhow::bail!(
                "Not enough data: have {} samples, need at least {} (seq_len={} + horizon={})",
                n_samples,
                total_needed,
                self.seq_len,
                self.target_horizon
            );
        }

        let n_windows = n_samples - total_needed + 1;
        let mut inputs = Array2::<f64>::zeros((n_windows * self.seq_len, n_features));
        let mut targets = Vec::with_capacity(n_windows);

        // Extract returns for targets (first feature column is returns)
        for i in 0..n_windows {
            // Copy the input window
            for j in 0..self.seq_len {
                let src_row = i + j;
                for k in 0..n_features {
                    inputs[[i * self.seq_len + j, k]] = features[[src_row, k]];
                }
            }
            // Target: sign of return at horizon
            let target_idx = i + self.seq_len + self.target_horizon - 1;
            let target_return = features[[target_idx, 0]]; // returns column
            targets.push(if target_return > 0.0 { 1.0 } else { 0.0 });
        }

        let targets = ndarray::Array1::from_vec(targets);

        info!(
            "Dataset built: {} windows, {} features per timestep",
            n_windows, n_features
        );

        Ok(Dataset {
            inputs,
            targets,
            seq_len: self.seq_len,
            n_features,
            n_windows,
        })
    }

    /// Build a dataset from raw price data (close prices only).
    ///
    /// Useful for quick testing with synthetic data.
    pub fn build_from_prices(&self, prices: &[f64]) -> Result<Dataset> {
        let klines: Vec<Kline> = prices
            .iter()
            .enumerate()
            .map(|(i, &p)| Kline {
                timestamp: (i as u64) * 60_000,
                open: p,
                high: p * 1.001,
                low: p * 0.999,
                close: p,
                volume: 100.0,
                turnover: p * 100.0,
            })
            .collect();
        self.build_dataset(&klines)
    }
}
