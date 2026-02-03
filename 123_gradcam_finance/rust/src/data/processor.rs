//! Data processing utilities
//!
//! This module provides utilities for preparing data for CNN inference
//! and managing data sequences.

use crate::api::bybit::Kline;
use crate::data::features::FeatureEngineering;
use ndarray::{Array2, Array3};

/// Data processor for preparing CNN inputs
#[derive(Debug, Clone)]
pub struct DataProcessor {
    sequence_length: usize,
    num_channels: usize,
}

impl Default for DataProcessor {
    fn default() -> Self {
        Self::new(60, 5)
    }
}

impl DataProcessor {
    /// Create a new data processor
    ///
    /// # Arguments
    /// * `sequence_length` - Number of time periods per sequence
    /// * `num_channels` - Number of features (default: 5 for OHLCV)
    pub fn new(sequence_length: usize, num_channels: usize) -> Self {
        Self {
            sequence_length,
            num_channels,
        }
    }

    /// Get the sequence length
    pub fn sequence_length(&self) -> usize {
        self.sequence_length
    }

    /// Get the number of channels
    pub fn num_channels(&self) -> usize {
        self.num_channels
    }

    /// Create a single feature sequence from klines
    ///
    /// Returns array of shape (1, channels, sequence_length)
    pub fn create_single_sequence(&self, klines: &[Kline]) -> Option<Array3<f64>> {
        if klines.len() < self.sequence_length {
            return None;
        }

        let window = &klines[klines.len() - self.sequence_length..];
        let features = FeatureEngineering::create_sequence_features(window);

        // Reshape to (1, channels, sequence_length)
        let mut batch = Array3::zeros((1, self.num_channels, self.sequence_length));
        for c in 0..self.num_channels {
            for t in 0..self.sequence_length {
                batch[[0, c, t]] = features[[c, t]];
            }
        }

        Some(batch)
    }

    /// Create multiple feature sequences from klines (sliding window)
    ///
    /// Returns array of shape (num_sequences, channels, sequence_length)
    pub fn create_sequences(&self, klines: &[Kline]) -> Array3<f64> {
        let num_sequences = if klines.len() >= self.sequence_length {
            klines.len() - self.sequence_length + 1
        } else {
            0
        };

        let mut sequences = Array3::zeros((num_sequences, self.num_channels, self.sequence_length));

        for i in 0..num_sequences {
            let window = &klines[i..i + self.sequence_length];
            let features = FeatureEngineering::create_sequence_features(window);

            for c in 0..self.num_channels {
                for t in 0..self.sequence_length {
                    sequences[[i, c, t]] = features[[c, t]];
                }
            }
        }

        sequences
    }

    /// Create labels based on future price movement
    ///
    /// Labels: 0 = down, 1 = neutral, 2 = up
    ///
    /// # Arguments
    /// * `klines` - Input klines
    /// * `threshold` - Threshold for neutral class (e.g., 0.005 for 0.5%)
    /// * `lookahead` - Number of periods to look ahead
    pub fn create_labels(&self, klines: &[Kline], threshold: f64, lookahead: usize) -> Vec<u8> {
        let mut labels = Vec::new();

        for i in 0..(klines.len().saturating_sub(lookahead)) {
            let current_price = klines[i].close;
            let future_price = klines[i + lookahead].close;
            let returns = (future_price - current_price) / current_price;

            let label = if returns < -threshold {
                0 // Down
            } else if returns > threshold {
                2 // Up
            } else {
                1 // Neutral
            };

            labels.push(label);
        }

        labels
    }

    /// Split data into train/validation/test sets
    ///
    /// Returns (train, validation, test) splits
    pub fn train_val_test_split<T: Clone>(
        data: &[T],
        train_ratio: f64,
        val_ratio: f64,
    ) -> (Vec<T>, Vec<T>, Vec<T>) {
        let n = data.len();
        let train_size = (n as f64 * train_ratio) as usize;
        let val_size = (n as f64 * val_ratio) as usize;

        let train = data[..train_size].to_vec();
        let val = data[train_size..train_size + val_size].to_vec();
        let test = data[train_size + val_size..].to_vec();

        (train, val, test)
    }
}

/// Stores a batch of processed data ready for inference
#[derive(Debug, Clone)]
pub struct DataBatch {
    /// Features array of shape (batch_size, channels, sequence_length)
    pub features: Array3<f64>,
    /// Optional labels
    pub labels: Option<Vec<u8>>,
    /// Original klines (for reference)
    pub klines: Vec<Kline>,
}

impl DataBatch {
    /// Create a new data batch
    pub fn new(features: Array3<f64>, labels: Option<Vec<u8>>, klines: Vec<Kline>) -> Self {
        Self {
            features,
            labels,
            klines,
        }
    }

    /// Get batch size
    pub fn len(&self) -> usize {
        self.features.dim().0
    }

    /// Check if batch is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the latest close price
    pub fn latest_close(&self) -> Option<f64> {
        self.klines.last().map(|k| k.close)
    }

    /// Get the timestamp of the latest kline
    pub fn latest_timestamp(&self) -> Option<i64> {
        self.klines.last().map(|k| k.timestamp)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_klines(n: usize) -> Vec<Kline> {
        (0..n)
            .map(|i| Kline {
                timestamp: i as i64 * 60000,
                open: 100.0 + i as f64,
                high: 101.0 + i as f64,
                low: 99.0 + i as f64,
                close: 100.5 + i as f64,
                volume: 1000.0 + i as f64 * 10.0,
                turnover: 100000.0,
            })
            .collect()
    }

    #[test]
    fn test_create_single_sequence() {
        let processor = DataProcessor::new(60, 5);
        let klines = create_test_klines(60);

        let seq = processor.create_single_sequence(&klines);
        assert!(seq.is_some());

        let seq = seq.unwrap();
        assert_eq!(seq.dim(), (1, 5, 60));
    }

    #[test]
    fn test_create_single_sequence_insufficient_data() {
        let processor = DataProcessor::new(60, 5);
        let klines = create_test_klines(30);

        let seq = processor.create_single_sequence(&klines);
        assert!(seq.is_none());
    }

    #[test]
    fn test_create_sequences() {
        let processor = DataProcessor::new(10, 5);
        let klines = create_test_klines(20);

        let seqs = processor.create_sequences(&klines);
        assert_eq!(seqs.dim(), (11, 5, 10));
    }

    #[test]
    fn test_create_labels() {
        let processor = DataProcessor::new(60, 5);
        let klines = create_test_klines(100);

        let labels = processor.create_labels(&klines, 0.005, 1);
        assert_eq!(labels.len(), 99);
    }

    #[test]
    fn test_train_val_test_split() {
        let data: Vec<i32> = (0..100).collect();
        let (train, val, test) = DataProcessor::train_val_test_split(&data, 0.7, 0.15);

        assert_eq!(train.len(), 70);
        assert_eq!(val.len(), 15);
        assert_eq!(test.len(), 15);
    }
}
