//! Trading Dataset
//!
//! Dataset implementation for training BigBird models.

use burn::data::dataloader::batcher::Batcher;
use burn::prelude::*;

use super::features::FeatureEngine;
use crate::api::MarketData;

/// A single training sample
#[derive(Clone, Debug)]
pub struct TradingSample {
    /// Input features [seq_len, n_features]
    pub features: Vec<Vec<f64>>,
    /// Target value (next period return)
    pub target: f64,
}

/// Trading dataset for time series data
#[derive(Clone, Debug)]
pub struct TradingDataset {
    /// All samples
    samples: Vec<TradingSample>,
    /// Sequence length
    seq_len: usize,
    /// Number of features per timestep
    n_features: usize,
}

impl TradingDataset {
    /// Create a new dataset from market data
    pub fn from_market_data(
        data: &MarketData,
        seq_len: usize,
        feature_engine: &FeatureEngine,
    ) -> Self {
        let closes: Vec<f64> = data.klines.iter().map(|k| k.close).collect();
        let features = feature_engine.calculate_features(&data.klines);
        let targets = feature_engine.generate_targets(&closes, 1);

        let n_features = features.first().map(|f| f.len()).unwrap_or(0);
        let mut samples = Vec::new();

        // Create sequences
        for i in seq_len..features.len().saturating_sub(1) {
            let seq_features: Vec<Vec<f64>> = features[i - seq_len..i].to_vec();
            let target = targets[i];

            samples.push(TradingSample {
                features: seq_features,
                target,
            });
        }

        Self {
            samples,
            seq_len,
            n_features,
        }
    }

    /// Create from pre-computed features and targets
    pub fn from_features(
        features: Vec<Vec<f64>>,
        targets: Vec<f64>,
        seq_len: usize,
    ) -> Self {
        let n_features = features.first().map(|f| f.len()).unwrap_or(0);
        let mut samples = Vec::new();

        for i in seq_len..features.len().min(targets.len()) {
            let seq_features: Vec<Vec<f64>> = features[i - seq_len..i].to_vec();
            let target = targets[i];

            samples.push(TradingSample {
                features: seq_features,
                target,
            });
        }

        Self {
            samples,
            seq_len,
            n_features,
        }
    }

    /// Get the number of samples
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Get sequence length
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Get number of features
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// Get a sample by index
    pub fn get(&self, index: usize) -> Option<&TradingSample> {
        self.samples.get(index)
    }

    /// Split dataset into train/val/test
    pub fn split(&self, train_ratio: f64, val_ratio: f64) -> (Self, Self, Self) {
        let n = self.samples.len();
        let train_end = (n as f64 * train_ratio) as usize;
        let val_end = train_end + (n as f64 * val_ratio) as usize;

        let train = Self {
            samples: self.samples[..train_end].to_vec(),
            seq_len: self.seq_len,
            n_features: self.n_features,
        };

        let val = Self {
            samples: self.samples[train_end..val_end].to_vec(),
            seq_len: self.seq_len,
            n_features: self.n_features,
        };

        let test = Self {
            samples: self.samples[val_end..].to_vec(),
            seq_len: self.seq_len,
            n_features: self.n_features,
        };

        (train, val, test)
    }

    /// Shuffle the dataset (in-place)
    pub fn shuffle(&mut self, seed: u64) {
        use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};
        let mut rng = StdRng::seed_from_u64(seed);
        self.samples.shuffle(&mut rng);
    }

    /// Get statistics about targets
    pub fn target_stats(&self) -> TargetStats {
        if self.samples.is_empty() {
            return TargetStats::default();
        }

        let targets: Vec<f64> = self.samples.iter().map(|s| s.target).collect();
        let mean = targets.iter().sum::<f64>() / targets.len() as f64;
        let variance = targets.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / targets.len() as f64;
        let std = variance.sqrt();

        let min = targets.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = targets.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let positive_ratio = targets.iter().filter(|&&t| t > 0.0).count() as f64 / targets.len() as f64;

        TargetStats {
            mean,
            std,
            min,
            max,
            positive_ratio,
        }
    }
}

/// Target statistics
#[derive(Debug, Clone, Default)]
pub struct TargetStats {
    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub max: f64,
    pub positive_ratio: f64,
}

impl std::fmt::Display for TargetStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "TargetStats {{ mean: {:.6}, std: {:.6}, min: {:.6}, max: {:.6}, positive: {:.2}% }}",
            self.mean,
            self.std,
            self.min,
            self.max,
            self.positive_ratio * 100.0
        )
    }
}

/// Batch of training data
#[derive(Clone, Debug)]
pub struct TradingBatch<B: Backend> {
    pub features: Tensor<B, 3>,
    pub targets: Tensor<B, 1>,
}

/// Batcher for creating training batches
#[derive(Clone, Debug)]
pub struct TradingBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> TradingBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

impl<B: Backend> Batcher<TradingSample, TradingBatch<B>> for TradingBatcher<B> {
    fn batch(&self, items: Vec<TradingSample>) -> TradingBatch<B> {
        let batch_size = items.len();
        let seq_len = items[0].features.len();
        let n_features = items[0].features[0].len();

        // Flatten features into a single vector
        let features_flat: Vec<f32> = items
            .iter()
            .flat_map(|s| {
                s.features
                    .iter()
                    .flat_map(|f| f.iter().map(|&v| v as f32))
            })
            .collect();

        let targets_flat: Vec<f32> = items.iter().map(|s| s.target as f32).collect();

        let features = Tensor::from_floats(features_flat.as_slice(), &self.device)
            .reshape([batch_size, seq_len, n_features]);

        let targets = Tensor::from_floats(targets_flat.as_slice(), &self.device);

        TradingBatch { features, targets }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::DataLoader;

    #[test]
    fn test_dataset_creation() {
        let loader = DataLoader::offline();
        let data = loader.generate_synthetic(500, 42);

        let feature_engine = FeatureEngine::default();
        let dataset = TradingDataset::from_market_data(&data, 64, &feature_engine);

        assert!(!dataset.is_empty());
        assert_eq!(dataset.seq_len(), 64);
        assert!(dataset.n_features() > 0);

        println!("Dataset size: {}", dataset.len());
        println!("Target stats: {}", dataset.target_stats());
    }

    #[test]
    fn test_dataset_split() {
        let loader = DataLoader::offline();
        let data = loader.generate_synthetic(1000, 42);

        let feature_engine = FeatureEngine::default();
        let dataset = TradingDataset::from_market_data(&data, 64, &feature_engine);

        let (train, val, test) = dataset.split(0.7, 0.15);

        assert!(train.len() > val.len());
        assert!(val.len() > 0);
        assert!(test.len() > 0);

        // Check approximate ratios
        let total = train.len() + val.len() + test.len();
        assert_eq!(total, dataset.len());
    }

    #[test]
    fn test_batcher() {
        use burn::backend::NdArray;
        type TestBackend = NdArray;

        let loader = DataLoader::offline();
        let data = loader.generate_synthetic(200, 42);

        let feature_engine = FeatureEngine::default();
        let dataset = TradingDataset::from_market_data(&data, 32, &feature_engine);

        let device = Default::default();
        let batcher = TradingBatcher::<TestBackend>::new(device);

        let samples: Vec<_> = (0..4).filter_map(|i| dataset.get(i).cloned()).collect();

        if samples.len() == 4 {
            let batch = batcher.batch(samples);

            assert_eq!(batch.features.dims()[0], 4);
            assert_eq!(batch.features.dims()[1], 32);
            assert_eq!(batch.targets.dims()[0], 4);
        }
    }
}
