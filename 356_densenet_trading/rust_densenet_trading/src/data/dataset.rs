//! Trading Dataset
//!
//! Dataset for training DenseNet models.

use ndarray::Array2;
use serde::{Deserialize, Serialize};

use super::candle::{Candle, CandleSeries};

/// A single training sample
#[derive(Debug, Clone)]
pub struct TradingSample {
    /// Input features [features, sequence_length]
    pub features: Array2<f64>,

    /// Target label (0=Short, 1=Hold, 2=Long)
    pub label: usize,

    /// Future return (for regression)
    pub future_return: f64,

    /// Timestamp of the prediction point
    pub timestamp: u64,
}

/// Dataset for training
#[derive(Debug, Clone)]
pub struct TradingDataset {
    /// All samples
    pub samples: Vec<TradingSample>,

    /// Number of input features
    pub num_features: usize,

    /// Sequence length
    pub sequence_length: usize,

    /// Number of classes
    pub num_classes: usize,
}

impl TradingDataset {
    /// Create a new empty dataset
    pub fn new(num_features: usize, sequence_length: usize, num_classes: usize) -> Self {
        Self {
            samples: Vec::new(),
            num_features,
            sequence_length,
            num_classes,
        }
    }

    /// Create dataset from candle series
    ///
    /// # Arguments
    /// * `candles` - Historical candle data
    /// * `feature_matrix` - Pre-computed features [num_candles, num_features]
    /// * `sequence_length` - Lookback window
    /// * `prediction_horizon` - How far ahead to predict
    /// * `threshold` - Return threshold for labeling (e.g., 0.001 = 0.1%)
    pub fn from_features(
        candles: &CandleSeries,
        feature_matrix: &Array2<f64>,
        sequence_length: usize,
        prediction_horizon: usize,
        threshold: f64,
    ) -> Self {
        let (num_candles, num_features) = feature_matrix.dim();
        let num_samples = num_candles.saturating_sub(sequence_length + prediction_horizon);

        let mut samples = Vec::with_capacity(num_samples);

        for i in 0..num_samples {
            let start = i;
            let end = i + sequence_length;
            let future_idx = end + prediction_horizon - 1;

            if future_idx >= candles.len() {
                break;
            }

            // Extract feature window
            let mut features = Array2::zeros((num_features, sequence_length));
            for j in 0..sequence_length {
                for f in 0..num_features {
                    features[[f, j]] = feature_matrix[[start + j, f]];
                }
            }

            // Calculate future return
            let current_close = candles.candles[end - 1].close;
            let future_close = candles.candles[future_idx].close;
            let future_return = (future_close / current_close).ln();

            // Assign label based on threshold
            let label = if future_return > threshold {
                2 // Long
            } else if future_return < -threshold {
                0 // Short
            } else {
                1 // Hold
            };

            samples.push(TradingSample {
                features,
                label,
                future_return,
                timestamp: candles.candles[end - 1].timestamp,
            });
        }

        Self {
            samples,
            num_features,
            sequence_length,
            num_classes: 3,
        }
    }

    /// Add a sample
    pub fn push(&mut self, sample: TradingSample) {
        self.samples.push(sample);
    }

    /// Get number of samples
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Get a sample by index
    pub fn get(&self, index: usize) -> Option<&TradingSample> {
        self.samples.get(index)
    }

    /// Split into train and test sets
    pub fn train_test_split(&self, train_ratio: f64) -> (TradingDataset, TradingDataset) {
        let split_idx = (self.samples.len() as f64 * train_ratio) as usize;

        let train_samples = self.samples[..split_idx].to_vec();
        let test_samples = self.samples[split_idx..].to_vec();

        (
            TradingDataset {
                samples: train_samples,
                num_features: self.num_features,
                sequence_length: self.sequence_length,
                num_classes: self.num_classes,
            },
            TradingDataset {
                samples: test_samples,
                num_features: self.num_features,
                sequence_length: self.sequence_length,
                num_classes: self.num_classes,
            },
        )
    }

    /// Get label distribution
    pub fn label_distribution(&self) -> Vec<usize> {
        let mut counts = vec![0; self.num_classes];
        for sample in &self.samples {
            if sample.label < self.num_classes {
                counts[sample.label] += 1;
            }
        }
        counts
    }

    /// Get class weights for balanced training
    pub fn class_weights(&self) -> Vec<f64> {
        let dist = self.label_distribution();
        let total: usize = dist.iter().sum();

        if total == 0 {
            return vec![1.0; self.num_classes];
        }

        dist.iter()
            .map(|&count| {
                if count > 0 {
                    total as f64 / (self.num_classes as f64 * count as f64)
                } else {
                    1.0
                }
            })
            .collect()
    }

    /// Create batches for training
    pub fn batches(&self, batch_size: usize) -> Vec<Vec<&TradingSample>> {
        self.samples.chunks(batch_size).map(|c| c.iter().collect()).collect()
    }

    /// Shuffle the dataset (in-place)
    pub fn shuffle(&mut self) {
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();
        self.samples.shuffle(&mut rng);
    }

    /// Get dataset statistics
    pub fn statistics(&self) -> DatasetStatistics {
        let returns: Vec<f64> = self.samples.iter().map(|s| s.future_return).collect();

        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;
        let std = variance.sqrt();

        let mut sorted = returns.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let min = sorted.first().copied().unwrap_or(0.0);
        let max = sorted.last().copied().unwrap_or(0.0);
        let median = sorted.get(sorted.len() / 2).copied().unwrap_or(0.0);

        DatasetStatistics {
            num_samples: self.samples.len(),
            num_features: self.num_features,
            sequence_length: self.sequence_length,
            label_distribution: self.label_distribution(),
            return_mean: mean,
            return_std: std,
            return_min: min,
            return_max: max,
            return_median: median,
        }
    }
}

/// Dataset statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetStatistics {
    pub num_samples: usize,
    pub num_features: usize,
    pub sequence_length: usize,
    pub label_distribution: Vec<usize>,
    pub return_mean: f64,
    pub return_std: f64,
    pub return_min: f64,
    pub return_max: f64,
    pub return_median: f64,
}

impl std::fmt::Display for DatasetStatistics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Dataset Statistics")?;
        writeln!(f, "==================")?;
        writeln!(f, "Samples: {}", self.num_samples)?;
        writeln!(f, "Features: {}", self.num_features)?;
        writeln!(f, "Sequence Length: {}", self.sequence_length)?;
        writeln!(f, "Label Distribution: {:?}", self.label_distribution)?;
        writeln!(f, "Return Mean: {:.6}", self.return_mean)?;
        writeln!(f, "Return Std: {:.6}", self.return_std)?;
        writeln!(f, "Return Range: [{:.6}, {:.6}]", self.return_min, self.return_max)?;
        writeln!(f, "Return Median: {:.6}", self.return_median)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataset_creation() {
        let dataset = TradingDataset::new(32, 128, 3);
        assert!(dataset.is_empty());
        assert_eq!(dataset.num_features, 32);
    }

    #[test]
    fn test_train_test_split() {
        let mut dataset = TradingDataset::new(10, 20, 3);

        // Add 100 samples
        for i in 0..100 {
            let sample = TradingSample {
                features: Array2::zeros((10, 20)),
                label: i % 3,
                future_return: 0.001 * i as f64,
                timestamp: i as u64,
            };
            dataset.push(sample);
        }

        let (train, test) = dataset.train_test_split(0.8);
        assert_eq!(train.len(), 80);
        assert_eq!(test.len(), 20);
    }

    #[test]
    fn test_label_distribution() {
        let mut dataset = TradingDataset::new(10, 20, 3);

        for i in 0..90 {
            let sample = TradingSample {
                features: Array2::zeros((10, 20)),
                label: i % 3,
                future_return: 0.0,
                timestamp: i as u64,
            };
            dataset.push(sample);
        }

        let dist = dataset.label_distribution();
        assert_eq!(dist, vec![30, 30, 30]);
    }

    #[test]
    fn test_class_weights() {
        let mut dataset = TradingDataset::new(10, 20, 3);

        // Imbalanced dataset
        for _ in 0..50 {
            dataset.push(TradingSample {
                features: Array2::zeros((10, 20)),
                label: 0,
                future_return: 0.0,
                timestamp: 0,
            });
        }
        for _ in 0..30 {
            dataset.push(TradingSample {
                features: Array2::zeros((10, 20)),
                label: 1,
                future_return: 0.0,
                timestamp: 0,
            });
        }
        for _ in 0..20 {
            dataset.push(TradingSample {
                features: Array2::zeros((10, 20)),
                label: 2,
                future_return: 0.0,
                timestamp: 0,
            });
        }

        let weights = dataset.class_weights();
        // Smaller class should have higher weight
        assert!(weights[2] > weights[0]);
    }
}
