//! Dataset structure for machine learning

use anyhow::{Context, Result};
use ndarray::{Array1, Array2};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

/// Dataset for machine learning with features and labels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dataset {
    /// Feature matrix (n_samples x n_features)
    pub features: Vec<Vec<f64>>,
    /// Target labels
    pub labels: Vec<f64>,
    /// Feature names
    pub feature_names: Vec<String>,
    /// Timestamps for each sample
    pub timestamps: Vec<i64>,
}

/// Train/test split result
pub struct Split {
    pub train: Dataset,
    pub test: Dataset,
}

impl Dataset {
    /// Create a new empty dataset
    pub fn new(feature_names: Vec<String>) -> Self {
        Self {
            features: Vec::new(),
            labels: Vec::new(),
            feature_names,
            timestamps: Vec::new(),
        }
    }

    /// Create dataset from raw data
    pub fn from_data(
        features: Vec<Vec<f64>>,
        labels: Vec<f64>,
        feature_names: Vec<String>,
        timestamps: Vec<i64>,
    ) -> Self {
        Self {
            features,
            labels,
            feature_names,
            timestamps,
        }
    }

    /// Number of samples
    pub fn n_samples(&self) -> usize {
        self.features.len()
    }

    /// Number of features
    pub fn n_features(&self) -> usize {
        self.feature_names.len()
    }

    /// Add a sample
    pub fn add_sample(&mut self, features: Vec<f64>, label: f64, timestamp: i64) {
        assert_eq!(features.len(), self.feature_names.len());
        self.features.push(features);
        self.labels.push(label);
        self.timestamps.push(timestamp);
    }

    /// Get feature matrix as ndarray
    pub fn features_array(&self) -> Array2<f64> {
        let n_samples = self.n_samples();
        let n_features = self.n_features();

        if n_samples == 0 {
            return Array2::zeros((0, n_features));
        }

        Array2::from_shape_fn((n_samples, n_features), |(i, j)| self.features[i][j])
    }

    /// Get labels as ndarray
    pub fn labels_array(&self) -> Array1<f64> {
        Array1::from_vec(self.labels.clone())
    }

    /// Split into train and test sets (time-series aware)
    ///
    /// For time series data, we split by time, not randomly
    pub fn train_test_split(&self, test_ratio: f64) -> Split {
        let n = self.n_samples();
        let train_size = ((1.0 - test_ratio) * n as f64) as usize;

        let train = Dataset {
            features: self.features[..train_size].to_vec(),
            labels: self.labels[..train_size].to_vec(),
            feature_names: self.feature_names.clone(),
            timestamps: self.timestamps[..train_size].to_vec(),
        };

        let test = Dataset {
            features: self.features[train_size..].to_vec(),
            labels: self.labels[train_size..].to_vec(),
            feature_names: self.feature_names.clone(),
            timestamps: self.timestamps[train_size..].to_vec(),
        };

        Split { train, test }
    }

    /// Random shuffle split (not recommended for time series)
    pub fn random_split(&self, test_ratio: f64, seed: u64) -> Split {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let n = self.n_samples();

        let mut indices: Vec<usize> = (0..n).collect();
        indices.shuffle(&mut rng);

        let test_size = (test_ratio * n as f64) as usize;
        let (test_indices, train_indices) = indices.split_at(test_size);

        let train = self.subset(train_indices);
        let test = self.subset(test_indices);

        Split { train, test }
    }

    /// Create a subset of the dataset by indices
    pub fn subset(&self, indices: &[usize]) -> Dataset {
        Dataset {
            features: indices.iter().map(|&i| self.features[i].clone()).collect(),
            labels: indices.iter().map(|&i| self.labels[i]).collect(),
            feature_names: self.feature_names.clone(),
            timestamps: indices.iter().map(|&i| self.timestamps[i]).collect(),
        }
    }

    /// Bootstrap sample (random sample with replacement)
    pub fn bootstrap_sample(&self, seed: u64) -> Dataset {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let n = self.n_samples();

        let indices: Vec<usize> = (0..n).map(|_| *(&(0..n).collect::<Vec<_>>()).choose(&mut rng).unwrap()).collect();

        self.subset(&indices)
    }

    /// Normalize features (z-score)
    pub fn normalize(&mut self) {
        let n_features = self.n_features();
        let n_samples = self.n_samples();

        if n_samples == 0 {
            return;
        }

        for j in 0..n_features {
            let values: Vec<f64> = self.features.iter().map(|row| row[j]).collect();
            let mean: f64 = values.iter().sum::<f64>() / n_samples as f64;
            let variance: f64 =
                values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n_samples as f64;
            let std = variance.sqrt();

            if std > 1e-10 {
                for row in &mut self.features {
                    row[j] = (row[j] - mean) / std;
                }
            }
        }
    }

    /// Convert labels to binary classification (positive/negative returns)
    pub fn to_binary_classification(&mut self) {
        for label in &mut self.labels {
            *label = if *label > 0.0 { 1.0 } else { 0.0 };
        }
    }

    /// Save dataset to JSON file
    pub fn save(&self, path: &Path) -> Result<()> {
        let file = File::create(path).context("Failed to create file")?;
        let writer = BufWriter::new(file);
        serde_json::to_writer(writer, self).context("Failed to serialize dataset")?;
        Ok(())
    }

    /// Load dataset from JSON file
    pub fn load(path: &Path) -> Result<Self> {
        let file = File::open(path).context("Failed to open file")?;
        let reader = BufReader::new(file);
        let dataset = serde_json::from_reader(reader).context("Failed to deserialize dataset")?;
        Ok(dataset)
    }

    /// Save to CSV file
    pub fn save_csv(&self, path: &Path) -> Result<()> {
        let mut writer = csv::Writer::from_path(path)?;

        // Write header
        let mut header = self.feature_names.clone();
        header.push("label".to_string());
        header.push("timestamp".to_string());
        writer.write_record(&header)?;

        // Write data
        for i in 0..self.n_samples() {
            let mut row: Vec<String> = self.features[i].iter().map(|v| v.to_string()).collect();
            row.push(self.labels[i].to_string());
            row.push(self.timestamps[i].to_string());
            writer.write_record(&row)?;
        }

        writer.flush()?;
        Ok(())
    }

    /// Load from CSV file
    pub fn load_csv(path: &Path) -> Result<Self> {
        let mut reader = csv::Reader::from_path(path)?;

        let headers: Vec<String> = reader
            .headers()?
            .iter()
            .map(|s| s.to_string())
            .collect();

        // Assume last two columns are label and timestamp
        let n_features = headers.len() - 2;
        let feature_names: Vec<String> = headers[..n_features].to_vec();

        let mut features = Vec::new();
        let mut labels = Vec::new();
        let mut timestamps = Vec::new();

        for result in reader.records() {
            let record = result?;
            let row: Vec<f64> = record
                .iter()
                .take(n_features)
                .map(|s| s.parse().unwrap_or(0.0))
                .collect();
            features.push(row);

            let label: f64 = record.get(n_features).unwrap_or("0").parse().unwrap_or(0.0);
            labels.push(label);

            let ts: i64 = record
                .get(n_features + 1)
                .unwrap_or("0")
                .parse()
                .unwrap_or(0);
            timestamps.push(ts);
        }

        Ok(Dataset {
            features,
            labels,
            feature_names,
            timestamps,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataset_operations() {
        let mut dataset = Dataset::new(vec!["f1".to_string(), "f2".to_string()]);
        dataset.add_sample(vec![1.0, 2.0], 0.05, 1000);
        dataset.add_sample(vec![3.0, 4.0], -0.02, 2000);
        dataset.add_sample(vec![5.0, 6.0], 0.03, 3000);

        assert_eq!(dataset.n_samples(), 3);
        assert_eq!(dataset.n_features(), 2);

        let split = dataset.train_test_split(0.33);
        assert_eq!(split.train.n_samples(), 2);
        assert_eq!(split.test.n_samples(), 1);
    }
}
