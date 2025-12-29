//! Data preprocessing utilities.

use ndarray::{Array1, Array2, Array3, s, Axis};
use tch::{Tensor, Kind, Device};

/// Normalization parameters.
#[derive(Debug, Clone)]
pub struct NormParams {
    pub mean: Vec<f64>,
    pub std: Vec<f64>,
}

/// Normalize features using standardization.
pub fn normalize(data: &Array2<f64>) -> (Array2<f64>, NormParams) {
    let (n_samples, n_features) = data.dim();

    let mut mean = vec![0.0; n_features];
    let mut std = vec![1.0; n_features];

    // Compute mean and std for each feature
    for j in 0..n_features {
        let col = data.column(j);
        mean[j] = col.mean().unwrap_or(0.0);

        let variance: f64 = col.iter().map(|x| (x - mean[j]).powi(2)).sum::<f64>() / n_samples as f64;
        std[j] = variance.sqrt().max(1e-8);
    }

    // Normalize
    let mut normalized = Array2::zeros((n_samples, n_features));
    for i in 0..n_samples {
        for j in 0..n_features {
            normalized[[i, j]] = (data[[i, j]] - mean[j]) / std[j];
        }
    }

    (normalized, NormParams { mean, std })
}

/// Inverse normalize data.
pub fn inverse_normalize(data: &Array2<f64>, params: &NormParams) -> Array2<f64> {
    let (n_samples, n_features) = data.dim();
    let mut denormalized = Array2::zeros((n_samples, n_features));

    for i in 0..n_samples {
        for j in 0..n_features {
            denormalized[[i, j]] = data[[i, j]] * params.std[j] + params.mean[j];
        }
    }

    denormalized
}

/// Create sequences for time series modeling.
pub fn create_sequences(
    features: &Array2<f64>,
    targets: &Array1<f64>,
    seq_length: usize,
    forecast_horizon: usize,
) -> (Array3<f64>, Array2<f64>) {
    let n_samples = features.nrows();
    let n_features = features.ncols();

    let n_sequences = n_samples - seq_length - forecast_horizon + 1;

    let mut x = Array3::zeros((n_sequences, seq_length, n_features));
    let mut y = Array2::zeros((n_sequences, forecast_horizon));

    for i in 0..n_sequences {
        // Historical features
        for j in 0..seq_length {
            for k in 0..n_features {
                x[[i, j, k]] = features[[i + j, k]];
            }
        }

        // Future targets
        for j in 0..forecast_horizon {
            y[[i, j]] = targets[i + seq_length + j];
        }
    }

    (x, y)
}

/// Dataset for training.
pub struct TimeSeriesDataset {
    pub features: Tensor,
    pub targets: Tensor,
    pub norm_params: NormParams,
    pub target_mean: f64,
    pub target_std: f64,
}

impl TimeSeriesDataset {
    /// Create a new dataset from arrays.
    pub fn new(
        features: Array3<f64>,
        targets: Array2<f64>,
        norm_params: NormParams,
        target_mean: f64,
        target_std: f64,
        device: Device,
    ) -> Self {
        let (n, seq_len, n_feat) = features.dim();
        let (_, horizon) = targets.dim();

        // Convert to tensors
        let features_vec: Vec<f64> = features.iter().cloned().collect();
        let targets_vec: Vec<f64> = targets.iter().cloned().collect();

        let features_tensor = Tensor::from_slice(&features_vec)
            .reshape(&[n as i64, seq_len as i64, n_feat as i64])
            .to_kind(Kind::Float)
            .to(device);

        let targets_tensor = Tensor::from_slice(&targets_vec)
            .reshape(&[n as i64, horizon as i64])
            .to_kind(Kind::Float)
            .to(device);

        Self {
            features: features_tensor,
            targets: targets_tensor,
            norm_params,
            target_mean,
            target_std,
        }
    }

    /// Get the number of samples.
    pub fn len(&self) -> i64 {
        self.features.size()[0]
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get a batch of data.
    pub fn get_batch(&self, indices: &[i64]) -> (Tensor, Tensor) {
        let idx = Tensor::from_slice(indices);
        (
            self.features.index_select(0, &idx),
            self.targets.index_select(0, &idx),
        )
    }

    /// Inverse transform targets.
    pub fn inverse_transform_targets(&self, targets: &Tensor) -> Tensor {
        targets * self.target_std + self.target_mean
    }
}

/// Data loader for batching.
pub struct DataLoader {
    dataset: TimeSeriesDataset,
    batch_size: usize,
    shuffle: bool,
    indices: Vec<i64>,
    current_idx: usize,
}

impl DataLoader {
    /// Create a new data loader.
    pub fn new(dataset: TimeSeriesDataset, batch_size: usize, shuffle: bool) -> Self {
        let n = dataset.len() as usize;
        let indices: Vec<i64> = (0..n as i64).collect();

        Self {
            dataset,
            batch_size,
            shuffle,
            indices,
            current_idx: 0,
        }
    }

    /// Reset the loader for a new epoch.
    pub fn reset(&mut self) {
        self.current_idx = 0;

        if self.shuffle {
            use rand::seq::SliceRandom;
            let mut rng = rand::thread_rng();
            self.indices.shuffle(&mut rng);
        }
    }

    /// Get the number of batches.
    pub fn num_batches(&self) -> usize {
        (self.indices.len() + self.batch_size - 1) / self.batch_size
    }
}

impl Iterator for DataLoader {
    type Item = (Tensor, Tensor);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_idx >= self.indices.len() {
            return None;
        }

        let end_idx = (self.current_idx + self.batch_size).min(self.indices.len());
        let batch_indices: Vec<i64> = self.indices[self.current_idx..end_idx].to_vec();

        self.current_idx = end_idx;

        Some(self.dataset.get_batch(&batch_indices))
    }
}
