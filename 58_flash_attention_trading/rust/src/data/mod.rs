//! Data loading and feature engineering module.
//!
//! This module provides:
//! - Bybit API client for cryptocurrency data
//! - Yahoo Finance data loading for stocks
//! - Feature engineering for trading models

mod bybit;
mod features;
mod yahoo;

pub use bybit::{fetch_bybit_klines, BybitClient, Kline};
pub use features::{calculate_features, prepare_features, TradingFeatures};
pub use yahoo::{fetch_yahoo_data, YahooClient};

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// OHLCV data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OhlcvData {
    pub timestamp: DateTime<Utc>,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

impl OhlcvData {
    /// Calculate typical price (HLC/3)
    pub fn typical_price(&self) -> f64 {
        (self.high + self.low + self.close) / 3.0
    }

    /// Calculate true range
    pub fn true_range(&self, prev_close: Option<f64>) -> f64 {
        match prev_close {
            Some(pc) => {
                let hl = self.high - self.low;
                let hc = (self.high - pc).abs();
                let lc = (self.low - pc).abs();
                hl.max(hc).max(lc)
            }
            None => self.high - self.low,
        }
    }
}

/// Dataset for training/testing
#[derive(Debug, Clone)]
pub struct TradingDataset {
    pub features: ndarray::Array2<f32>,
    pub targets: ndarray::Array1<f32>,
    pub timestamps: Vec<DateTime<Utc>>,
    pub prices: Vec<f64>,
}

impl TradingDataset {
    /// Split dataset into train/validation/test sets
    pub fn split(&self, train_ratio: f64, val_ratio: f64) -> (Self, Self, Self) {
        let n = self.features.nrows();
        let train_end = (n as f64 * train_ratio) as usize;
        let val_end = train_end + (n as f64 * val_ratio) as usize;

        let train = Self {
            features: self.features.slice(ndarray::s![..train_end, ..]).to_owned(),
            targets: self.targets.slice(ndarray::s![..train_end]).to_owned(),
            timestamps: self.timestamps[..train_end].to_vec(),
            prices: self.prices[..train_end].to_vec(),
        };

        let val = Self {
            features: self
                .features
                .slice(ndarray::s![train_end..val_end, ..])
                .to_owned(),
            targets: self.targets.slice(ndarray::s![train_end..val_end]).to_owned(),
            timestamps: self.timestamps[train_end..val_end].to_vec(),
            prices: self.prices[train_end..val_end].to_vec(),
        };

        let test = Self {
            features: self.features.slice(ndarray::s![val_end.., ..]).to_owned(),
            targets: self.targets.slice(ndarray::s![val_end..]).to_owned(),
            timestamps: self.timestamps[val_end..].to_vec(),
            prices: self.prices[val_end..].to_vec(),
        };

        (train, val, test)
    }

    /// Create windowed sequences for attention model
    pub fn create_sequences(
        &self,
        lookback: usize,
        horizon: usize,
    ) -> Result<(ndarray::Array3<f32>, ndarray::Array1<f32>)> {
        let n = self.features.nrows();
        let n_features = self.features.ncols();

        if n < lookback + horizon {
            anyhow::bail!("Not enough data for sequences");
        }

        let n_sequences = n - lookback - horizon + 1;
        let mut x = ndarray::Array3::<f32>::zeros((n_sequences, lookback, n_features));
        let mut y = ndarray::Array1::<f32>::zeros(n_sequences);

        for i in 0..n_sequences {
            for t in 0..lookback {
                for f in 0..n_features {
                    x[[i, t, f]] = self.features[[i + t, f]];
                }
            }
            y[i] = self.targets[i + lookback + horizon - 1];
        }

        Ok((x, y))
    }
}
