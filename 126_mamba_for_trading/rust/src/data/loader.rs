//! Data loading utilities.

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::OhlcvBar;

/// Errors that can occur during data loading.
#[derive(Error, Debug)]
pub enum LoaderError {
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),

    #[error("CSV parse error: {0}")]
    CsvError(String),
}

/// Data loader for OHLCV data.
pub struct DataLoader;

impl DataLoader {
    /// Load OHLCV data from a CSV file.
    ///
    /// Expected format: timestamp,open,high,low,close,volume
    pub fn load_csv(path: &Path) -> Result<Vec<OhlcvBar>, LoaderError> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut bars = Vec::new();

        for (i, line) in reader.lines().enumerate() {
            let line = line?;

            // Skip header
            if i == 0 && line.starts_with("timestamp") {
                continue;
            }

            let parts: Vec<&str> = line.split(',').collect();
            if parts.len() < 6 {
                return Err(LoaderError::CsvError(format!(
                    "Invalid row at line {}: {}",
                    i + 1,
                    line
                )));
            }

            let bar = OhlcvBar {
                timestamp: parts[0]
                    .parse()
                    .map_err(|e| LoaderError::CsvError(format!("Parse error: {}", e)))?,
                open: parts[1]
                    .parse()
                    .map_err(|e| LoaderError::CsvError(format!("Parse error: {}", e)))?,
                high: parts[2]
                    .parse()
                    .map_err(|e| LoaderError::CsvError(format!("Parse error: {}", e)))?,
                low: parts[3]
                    .parse()
                    .map_err(|e| LoaderError::CsvError(format!("Parse error: {}", e)))?,
                close: parts[4]
                    .parse()
                    .map_err(|e| LoaderError::CsvError(format!("Parse error: {}", e)))?,
                volume: parts[5]
                    .parse()
                    .map_err(|e| LoaderError::CsvError(format!("Parse error: {}", e)))?,
            };

            bars.push(bar);
        }

        Ok(bars)
    }

    /// Save OHLCV data to a CSV file.
    pub fn save_csv(bars: &[OhlcvBar], path: &Path) -> Result<(), LoaderError> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        writeln!(writer, "timestamp,open,high,low,close,volume")?;

        for bar in bars {
            writeln!(
                writer,
                "{},{},{},{},{},{}",
                bar.timestamp, bar.open, bar.high, bar.low, bar.close, bar.volume
            )?;
        }

        Ok(())
    }

    /// Load OHLCV data from a JSON file.
    pub fn load_json(path: &Path) -> Result<Vec<OhlcvBar>, LoaderError> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let bars: Vec<OhlcvBar> = serde_json::from_reader(reader)?;
        Ok(bars)
    }

    /// Save OHLCV data to a JSON file.
    pub fn save_json(bars: &[OhlcvBar], path: &Path) -> Result<(), LoaderError> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, bars)?;
        Ok(())
    }
}

/// Dataset for training/inference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dataset {
    /// Feature sequences: (n_samples, seq_len, n_features)
    pub features: Vec<Vec<Vec<f64>>>,
    /// Labels (optional)
    pub labels: Option<Vec<usize>>,
    /// Feature names
    pub feature_names: Vec<String>,
}

impl Dataset {
    /// Create a new empty dataset.
    pub fn new(feature_names: Vec<String>) -> Self {
        Self {
            features: Vec::new(),
            labels: None,
            feature_names,
        }
    }

    /// Number of samples.
    pub fn len(&self) -> usize {
        self.features.len()
    }

    /// Check if dataset is empty.
    pub fn is_empty(&self) -> bool {
        self.features.is_empty()
    }

    /// Number of features.
    pub fn n_features(&self) -> usize {
        self.feature_names.len()
    }

    /// Add a sample.
    pub fn add_sample(&mut self, sequence: Vec<Vec<f64>>, label: Option<usize>) {
        self.features.push(sequence);
        if let Some(l) = label {
            if self.labels.is_none() {
                self.labels = Some(Vec::new());
            }
            self.labels.as_mut().unwrap().push(l);
        }
    }

    /// Split dataset into train and test sets.
    pub fn train_test_split(&self, test_ratio: f64) -> (Dataset, Dataset) {
        let split_idx = ((1.0 - test_ratio) * self.len() as f64) as usize;

        let train = Dataset {
            features: self.features[..split_idx].to_vec(),
            labels: self.labels.as_ref().map(|l| l[..split_idx].to_vec()),
            feature_names: self.feature_names.clone(),
        };

        let test = Dataset {
            features: self.features[split_idx..].to_vec(),
            labels: self.labels.as_ref().map(|l| l[split_idx..].to_vec()),
            feature_names: self.feature_names.clone(),
        };

        (train, test)
    }

    /// Save dataset to JSON file.
    pub fn save(&self, path: &Path) -> Result<(), LoaderError> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer(writer, self)?;
        Ok(())
    }

    /// Load dataset from JSON file.
    pub fn load(path: &Path) -> Result<Self, LoaderError> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let dataset: Dataset = serde_json::from_reader(reader)?;
        Ok(dataset)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_load_save_csv() {
        let bars = vec![
            OhlcvBar::new(1000, 100.0, 105.0, 95.0, 102.0, 1000.0),
            OhlcvBar::new(2000, 102.0, 108.0, 100.0, 106.0, 1200.0),
        ];

        let temp = NamedTempFile::new().unwrap();
        DataLoader::save_csv(&bars, temp.path()).unwrap();

        let loaded = DataLoader::load_csv(temp.path()).unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].timestamp, 1000);
        assert!((loaded[1].close - 106.0).abs() < 0.01);
    }
}
