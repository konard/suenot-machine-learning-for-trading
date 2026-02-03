//! Data processing utilities for trading data

use crate::api::bybit::Kline;
use ndarray::{Array1, Array2};

/// Data processor for converting raw klines to model-ready format
pub struct DataProcessor {
    /// Sequence length for model input
    pub sequence_length: usize,
    /// Feature means for normalization
    means: Option<Array1<f64>>,
    /// Feature standard deviations for normalization
    stds: Option<Array1<f64>>,
}

impl DataProcessor {
    /// Create a new data processor
    pub fn new(sequence_length: usize) -> Self {
        Self {
            sequence_length,
            means: None,
            stds: None,
        }
    }

    /// Convert klines to feature matrix
    pub fn klines_to_features(&self, klines: &[Kline]) -> Array2<f64> {
        let n = klines.len();
        let num_features = 5; // OHLCV

        let mut features = Array2::zeros((n, num_features));

        for (i, kline) in klines.iter().enumerate() {
            features[[i, 0]] = kline.open;
            features[[i, 1]] = kline.high;
            features[[i, 2]] = kline.low;
            features[[i, 3]] = kline.close;
            features[[i, 4]] = kline.volume;
        }

        features
    }

    /// Fit normalization parameters on data
    pub fn fit(&mut self, features: &Array2<f64>) {
        let means = features.mean_axis(ndarray::Axis(0)).unwrap();
        let stds = features.std_axis(ndarray::Axis(0), 0.0);

        self.means = Some(means);
        self.stds = Some(stds);
    }

    /// Normalize features using fitted parameters
    pub fn transform(&self, features: &Array2<f64>) -> Array2<f64> {
        let means = self.means.as_ref().expect("Must call fit() first");
        let stds = self.stds.as_ref().expect("Must call fit() first");

        let mut normalized = features.clone();
        for i in 0..features.ncols() {
            let std = stds[i].max(1e-10);
            for j in 0..features.nrows() {
                normalized[[j, i]] = (features[[j, i]] - means[i]) / std;
            }
        }

        normalized
    }

    /// Fit and transform in one step
    pub fn fit_transform(&mut self, features: &Array2<f64>) -> Array2<f64> {
        self.fit(features);
        self.transform(features)
    }

    /// Create sequences for model input
    pub fn create_sequences(&self, features: &Array2<f64>) -> (Vec<Array2<f64>>, Vec<f64>) {
        let n = features.nrows();
        let num_features = features.ncols();

        if n <= self.sequence_length {
            return (vec![], vec![]);
        }

        let mut sequences = Vec::new();
        let mut labels = Vec::new();

        for i in 0..(n - self.sequence_length - 1) {
            let seq = features
                .slice(ndarray::s![i..i + self.sequence_length, ..])
                .to_owned();
            sequences.push(seq);

            // Label: 1 if next close > current close, 0 otherwise
            let current_close = features[[i + self.sequence_length - 1, 3]];
            let next_close = features[[i + self.sequence_length, 3]];
            let label = if next_close > current_close { 1.0 } else { 0.0 };
            labels.push(label);
        }

        (sequences, labels)
    }

    /// Split data into train/test sets
    pub fn train_test_split(
        sequences: &[Array2<f64>],
        labels: &[f64],
        test_ratio: f64,
    ) -> (Vec<Array2<f64>>, Vec<f64>, Vec<Array2<f64>>, Vec<f64>) {
        let n = sequences.len();
        let split_idx = ((1.0 - test_ratio) * n as f64) as usize;

        let train_seq = sequences[..split_idx].to_vec();
        let train_labels = labels[..split_idx].to_vec();
        let test_seq = sequences[split_idx..].to_vec();
        let test_labels = labels[split_idx..].to_vec();

        (train_seq, train_labels, test_seq, test_labels)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_processor() {
        let klines: Vec<Kline> = (0..100)
            .map(|i| Kline {
                timestamp: i * 60000,
                open: 100.0 + i as f64,
                high: 105.0 + i as f64,
                low: 95.0 + i as f64,
                close: 102.0 + i as f64,
                volume: 1000.0,
                turnover: 100000.0,
            })
            .collect();

        let mut processor = DataProcessor::new(30);
        let features = processor.klines_to_features(&klines);
        let normalized = processor.fit_transform(&features);

        assert_eq!(normalized.shape(), features.shape());

        let (sequences, labels) = processor.create_sequences(&normalized);
        assert!(!sequences.is_empty());
        assert_eq!(sequences.len(), labels.len());
    }
}
