//! Dataset structure for training and evaluation

use ndarray::{Array1, Array2, s};
use crate::data::features::Features;

/// Single data sample
#[derive(Debug, Clone)]
pub struct Sample {
    /// Input features [lookback, n_features]
    pub x: Array2<f64>,
    /// Target label (0 or 1 for direction)
    pub y: u8,
}

/// Dataset for training and evaluation
pub struct Dataset {
    /// All samples
    pub samples: Vec<Sample>,
    /// Feature names
    pub feature_names: Vec<String>,
    /// Lookback window size
    pub lookback: usize,
}

impl Dataset {
    /// Create a new dataset
    pub fn new(samples: Vec<Sample>, feature_names: Vec<String>, lookback: usize) -> Self {
        Self {
            samples,
            feature_names,
            lookback,
        }
    }

    /// Number of samples
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Get a sample by index
    pub fn get(&self, idx: usize) -> Option<&Sample> {
        self.samples.get(idx)
    }

    /// Get flattened input dimension
    pub fn input_dim(&self) -> usize {
        self.lookback * Features::count()
    }

    /// Split into train/validation/test sets
    pub fn split(
        &self,
        train_ratio: f64,
        val_ratio: f64,
    ) -> (Dataset, Dataset, Dataset) {
        let n = self.samples.len();
        let train_end = (n as f64 * train_ratio) as usize;
        let val_end = (n as f64 * (train_ratio + val_ratio)) as usize;

        let train = Dataset::new(
            self.samples[..train_end].to_vec(),
            self.feature_names.clone(),
            self.lookback,
        );

        let val = Dataset::new(
            self.samples[train_end..val_end].to_vec(),
            self.feature_names.clone(),
            self.lookback,
        );

        let test = Dataset::new(
            self.samples[val_end..].to_vec(),
            self.feature_names.clone(),
            self.lookback,
        );

        (train, val, test)
    }

    /// Get batch of samples
    pub fn get_batch(&self, indices: &[usize]) -> (Array2<f64>, Array1<u8>) {
        let batch_size = indices.len();
        let input_dim = self.input_dim();

        let mut x = Array2::zeros((batch_size, input_dim));
        let mut y = Array1::zeros(batch_size);

        for (i, &idx) in indices.iter().enumerate() {
            if let Some(sample) = self.samples.get(idx) {
                x.row_mut(i).assign(&sample.x.view().into_shape(input_dim).unwrap());
                y[i] = sample.y;
            }
        }

        (x, y)
    }

    /// Create iterator over batches
    pub fn batches(&self, batch_size: usize) -> BatchIterator {
        BatchIterator {
            dataset: self,
            batch_size,
            current: 0,
        }
    }
}

/// Iterator over dataset batches
pub struct BatchIterator<'a> {
    dataset: &'a Dataset,
    batch_size: usize,
    current: usize,
}

impl<'a> Iterator for BatchIterator<'a> {
    type Item = (Array2<f64>, Array1<u8>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.dataset.len() {
            return None;
        }

        let end = (self.current + self.batch_size).min(self.dataset.len());
        let indices: Vec<usize> = (self.current..end).collect();

        self.current = end;

        Some(self.dataset.get_batch(&indices))
    }
}

/// Prepare dataset from features
pub fn prepare_data(
    features: &[Features],
    lookback: usize,
    horizon: usize,
) -> Dataset {
    let n = features.len();
    let n_features = Features::count();

    if n < lookback + horizon {
        return Dataset::new(vec![], vec![], lookback);
    }

    let mut samples = Vec::new();

    for i in lookback..(n - horizon) {
        // Create input array
        let mut x = Array2::zeros((lookback, n_features));
        for j in 0..lookback {
            x.row_mut(j).assign(&features[i - lookback + j].to_array());
        }

        // Create target (1 if price goes up, 0 otherwise)
        let current_return = features[i].log_return;
        let future_return = features[i + horizon].log_return;
        let y = if future_return > 0.0 { 1 } else { 0 };

        samples.push(Sample { x, y });
    }

    let feature_names = vec![
        "log_return", "return_5", "return_10",
        "volume_ratio", "volume_change",
        "volatility_5", "volatility_20",
        "rsi_14", "rsi_7",
        "macd", "macd_signal", "macd_hist",
        "bb_position", "atr_ratio",
        "ma_10", "ma_20",
    ].iter().map(|s| s.to_string()).collect();

    Dataset::new(samples, feature_names, lookback)
}

/// Normalize dataset features
pub fn normalize_dataset(dataset: &mut Dataset) {
    if dataset.is_empty() {
        return;
    }

    let n = dataset.len();
    let input_dim = dataset.input_dim();

    // Compute mean and std
    let mut mean = Array1::zeros(input_dim);
    let mut var = Array1::zeros(input_dim);

    for sample in &dataset.samples {
        let flat = sample.x.view().into_shape(input_dim).unwrap();
        mean = mean + &flat;
    }
    mean = mean / n as f64;

    for sample in &dataset.samples {
        let flat = sample.x.view().into_shape(input_dim).unwrap();
        let diff = &flat - &mean;
        var = var + &(&diff * &diff);
    }
    var = var / n as f64;
    let std = var.mapv(|v| (v + 1e-8).sqrt());

    // Normalize
    for sample in &mut dataset.samples {
        let flat = sample.x.view().into_shape(input_dim).unwrap().to_owned();
        let normalized = (&flat - &mean) / &std;
        sample.x = normalized.into_shape((dataset.lookback, Features::count())).unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::loader::generate_sample_data;

    #[test]
    fn test_prepare_data() {
        let features = generate_sample_data(200);
        let dataset = prepare_data(&features, 60, 1);

        assert!(!dataset.is_empty());
        assert_eq!(dataset.input_dim(), 60 * 16);
    }

    #[test]
    fn test_split() {
        let features = generate_sample_data(200);
        let dataset = prepare_data(&features, 60, 1);

        let (train, val, test) = dataset.split(0.7, 0.15);

        assert!(train.len() > val.len());
        assert!(train.len() > test.len());
    }
}
