//! Dataset structure for training the Diagonal SSM model.

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

/// A dataset of input sequences and target labels for training.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dataset {
    /// Flattened input matrix of shape (n_windows * seq_len, n_features).
    /// Each contiguous block of `seq_len` rows is one input sequence.
    pub inputs: Array2<f64>,
    /// Target values, one per window (length n_windows).
    pub targets: Array1<f64>,
    /// Length of each input sequence.
    pub seq_len: usize,
    /// Number of features per timestep.
    pub n_features: usize,
    /// Number of sliding windows (samples).
    pub n_windows: usize,
}

impl Dataset {
    /// Get a specific input window as a 2D slice (seq_len x n_features).
    pub fn get_window(&self, index: usize) -> Array2<f64> {
        let start = index * self.seq_len;
        let end = start + self.seq_len;
        self.inputs.slice(ndarray::s![start..end, ..]).to_owned()
    }

    /// Get the target value for a specific window.
    pub fn get_target(&self, index: usize) -> f64 {
        self.targets[index]
    }

    /// Split the dataset into training and validation sets.
    ///
    /// # Arguments
    ///
    /// * `train_ratio` - Fraction of data to use for training (e.g. 0.8)
    pub fn train_val_split(&self, train_ratio: f64) -> (Dataset, Dataset) {
        let train_size = (self.n_windows as f64 * train_ratio).floor() as usize;
        let val_size = self.n_windows - train_size;

        let train_rows = train_size * self.seq_len;
        let val_start = train_size * self.seq_len;

        let train = Dataset {
            inputs: self.inputs.slice(ndarray::s![..train_rows, ..]).to_owned(),
            targets: self.targets.slice(ndarray::s![..train_size]).to_owned(),
            seq_len: self.seq_len,
            n_features: self.n_features,
            n_windows: train_size,
        };

        let val = Dataset {
            inputs: self.inputs.slice(ndarray::s![val_start.., ..]).to_owned(),
            targets: self.targets.slice(ndarray::s![train_size..]).to_owned(),
            seq_len: self.seq_len,
            n_features: self.n_features,
            n_windows: val_size,
        };

        (train, val)
    }

    /// Normalize feature columns to zero mean and unit variance.
    ///
    /// Returns the means and standard deviations for each feature.
    pub fn normalize(&mut self) -> (Vec<f64>, Vec<f64>) {
        let mut means = vec![0.0; self.n_features];
        let mut stds = vec![0.0; self.n_features];
        let n = self.inputs.nrows() as f64;

        for col in 0..self.n_features {
            let column = self.inputs.column(col);
            let mean = column.sum() / n;
            let variance = column.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
            let std = variance.sqrt().max(1e-8);

            means[col] = mean;
            stds[col] = std;

            for row in 0..self.inputs.nrows() {
                self.inputs[[row, col]] = (self.inputs[[row, col]] - mean) / std;
            }
        }

        (means, stds)
    }
}
