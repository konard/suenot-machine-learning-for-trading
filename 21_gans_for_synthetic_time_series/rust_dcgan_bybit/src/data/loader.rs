//! DataLoader for batching and iterating over training data
//!
//! Provides efficient batching for GAN training with support for:
//! - Random shuffling
//! - Drop last incomplete batch
//! - Iteration over batches

use ndarray::{Array3, ArrayView3, Axis};
use rand::seq::SliceRandom;

/// DataLoader for iterating over batched sequences
pub struct DataLoader {
    /// Full dataset of shape (num_sequences, sequence_length, num_features)
    data: Array3<f64>,
    /// Batch size
    batch_size: usize,
    /// Whether to shuffle data each epoch
    shuffle: bool,
    /// Whether to drop the last incomplete batch
    drop_last: bool,
    /// Current indices for iteration
    indices: Vec<usize>,
    /// Current position in iteration
    current_idx: usize,
}

impl DataLoader {
    /// Create a new DataLoader
    ///
    /// # Arguments
    ///
    /// * `data` - 3D array of shape (num_sequences, sequence_length, num_features)
    /// * `batch_size` - Number of sequences per batch
    /// * `shuffle` - Whether to shuffle data each epoch
    /// * `drop_last` - Whether to drop incomplete final batch
    pub fn new(data: Array3<f64>, batch_size: usize, shuffle: bool, drop_last: bool) -> Self {
        let num_samples = data.shape()[0];
        let indices: Vec<usize> = (0..num_samples).collect();

        let mut loader = Self {
            data,
            batch_size,
            shuffle,
            drop_last,
            indices,
            current_idx: 0,
        };

        if shuffle {
            loader.shuffle_indices();
        }

        loader
    }

    /// Get the number of batches per epoch
    pub fn num_batches(&self) -> usize {
        let num_samples = self.data.shape()[0];
        if self.drop_last {
            num_samples / self.batch_size
        } else {
            (num_samples + self.batch_size - 1) / self.batch_size
        }
    }

    /// Get total number of samples
    pub fn num_samples(&self) -> usize {
        self.data.shape()[0]
    }

    /// Get sequence length
    pub fn sequence_length(&self) -> usize {
        self.data.shape()[1]
    }

    /// Get number of features
    pub fn num_features(&self) -> usize {
        self.data.shape()[2]
    }

    /// Shuffle indices for a new epoch
    fn shuffle_indices(&mut self) {
        let mut rng = rand::thread_rng();
        self.indices.shuffle(&mut rng);
    }

    /// Reset for new epoch
    pub fn reset(&mut self) {
        self.current_idx = 0;
        if self.shuffle {
            self.shuffle_indices();
        }
    }

    /// Get next batch
    ///
    /// Returns None when epoch is complete
    pub fn next_batch(&mut self) -> Option<Array3<f64>> {
        let num_samples = self.indices.len();
        let start = self.current_idx;

        if start >= num_samples {
            return None;
        }

        let end = (start + self.batch_size).min(num_samples);
        let actual_batch_size = end - start;

        // Skip incomplete batch if drop_last
        if self.drop_last && actual_batch_size < self.batch_size {
            return None;
        }

        // Collect batch
        let seq_len = self.sequence_length();
        let num_features = self.num_features();
        let mut batch = Array3::<f64>::zeros((actual_batch_size, seq_len, num_features));

        for (batch_idx, &data_idx) in self.indices[start..end].iter().enumerate() {
            batch
                .index_axis_mut(Axis(0), batch_idx)
                .assign(&self.data.index_axis(Axis(0), data_idx));
        }

        self.current_idx = end;
        Some(batch)
    }

    /// Iterate over all batches (consuming iterator style)
    pub fn iter(&mut self) -> DataLoaderIter<'_> {
        self.reset();
        DataLoaderIter { loader: self }
    }

    /// Get a view of the underlying data
    pub fn data(&self) -> ArrayView3<'_, f64> {
        self.data.view()
    }
}

/// Iterator adapter for DataLoader
pub struct DataLoaderIter<'a> {
    loader: &'a mut DataLoader,
}

impl<'a> Iterator for DataLoaderIter<'a> {
    type Item = Array3<f64>;

    fn next(&mut self) -> Option<Self::Item> {
        self.loader.next_batch()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[test]
    fn test_dataloader_basic() {
        // Create dummy data: 10 sequences, 5 timesteps, 3 features
        let data = Array3::<f64>::zeros((10, 5, 3));
        let mut loader = DataLoader::new(data, 3, false, false);

        assert_eq!(loader.num_batches(), 4); // ceil(10/3) = 4
        assert_eq!(loader.num_samples(), 10);

        let mut batch_count = 0;
        while let Some(batch) = loader.next_batch() {
            batch_count += 1;
            if batch_count < 4 {
                assert_eq!(batch.shape()[0], 3);
            } else {
                assert_eq!(batch.shape()[0], 1); // Last batch has 1 sample
            }
        }
        assert_eq!(batch_count, 4);
    }

    #[test]
    fn test_dataloader_drop_last() {
        let data = Array3::<f64>::zeros((10, 5, 3));
        let mut loader = DataLoader::new(data, 3, false, true);

        assert_eq!(loader.num_batches(), 3); // floor(10/3) = 3

        let mut batch_count = 0;
        while let Some(batch) = loader.next_batch() {
            batch_count += 1;
            assert_eq!(batch.shape()[0], 3);
        }
        assert_eq!(batch_count, 3);
    }

    #[test]
    fn test_dataloader_iter() {
        let data = Array3::<f64>::zeros((10, 5, 3));
        let mut loader = DataLoader::new(data, 5, false, true);

        let batches: Vec<_> = loader.iter().collect();
        assert_eq!(batches.len(), 2);
    }
}
