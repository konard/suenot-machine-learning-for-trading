//! Input/Output utilities for loading and saving data

use crate::api::bybit::MarketDocument;
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use thiserror::Error;

/// IO errors
#[derive(Error, Debug)]
pub enum IoError {
    #[error("File not found: {0}")]
    FileNotFound(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("JSON parsing error: {0}")]
    JsonError(#[from] serde_json::Error),

    #[error("Bincode error: {0}")]
    BincodeError(#[from] bincode::Error),
}

/// Dataset of documents for topic modeling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentDataset {
    /// Documents
    pub documents: Vec<MarketDocument>,
    /// Metadata
    pub metadata: DatasetMetadata,
}

/// Dataset metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetMetadata {
    /// Dataset name
    pub name: String,
    /// Creation timestamp
    pub created_at: u64,
    /// Source description
    pub source: String,
    /// Number of documents
    pub n_documents: usize,
}

impl DocumentDataset {
    /// Create a new dataset
    pub fn new(name: &str, source: &str, documents: Vec<MarketDocument>) -> Self {
        let n_documents = documents.len();
        Self {
            documents,
            metadata: DatasetMetadata {
                name: name.to_string(),
                created_at: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                source: source.to_string(),
                n_documents,
            },
        }
    }

    /// Save dataset to JSON file
    pub fn save_json<P: AsRef<Path>>(&self, path: P) -> Result<(), IoError> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, self)?;
        Ok(())
    }

    /// Load dataset from JSON file
    pub fn load_json<P: AsRef<Path>>(path: P) -> Result<Self, IoError> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let dataset = serde_json::from_reader(reader)?;
        Ok(dataset)
    }

    /// Save dataset to binary file (more efficient)
    pub fn save_binary<P: AsRef<Path>>(&self, path: P) -> Result<(), IoError> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        bincode::serialize_into(writer, self)?;
        Ok(())
    }

    /// Load dataset from binary file
    pub fn load_binary<P: AsRef<Path>>(path: P) -> Result<Self, IoError> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let dataset = bincode::deserialize_from(reader)?;
        Ok(dataset)
    }

    /// Get document texts
    pub fn get_texts(&self) -> Vec<String> {
        self.documents.iter().map(|d| d.full_text()).collect()
    }

    /// Filter documents by type
    pub fn filter_by_type(&self, doc_type: &str) -> Vec<&MarketDocument> {
        self.documents
            .iter()
            .filter(|d| d.doc_type == doc_type)
            .collect()
    }

    /// Filter documents by symbol
    pub fn filter_by_symbol(&self, symbol: &str) -> Vec<&MarketDocument> {
        self.documents
            .iter()
            .filter(|d| d.symbols.contains(&symbol.to_string()))
            .collect()
    }

    /// Filter documents by time range
    pub fn filter_by_time_range(
        &self,
        start_timestamp: u64,
        end_timestamp: u64,
    ) -> Vec<&MarketDocument> {
        self.documents
            .iter()
            .filter(|d| d.timestamp >= start_timestamp && d.timestamp <= end_timestamp)
            .collect()
    }
}

/// Model save/load utilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SavedModel {
    /// Model type (LSI or LDA)
    pub model_type: String,
    /// Number of topics
    pub n_topics: usize,
    /// Vocabulary
    pub vocabulary: Vec<String>,
    /// Topic-word weights/probabilities
    pub topic_words: Vec<Vec<f64>>,
    /// Training metadata
    pub metadata: ModelMetadata,
}

/// Model metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Training timestamp
    pub trained_at: u64,
    /// Number of documents used for training
    pub n_documents: usize,
    /// Vocabulary size
    pub vocabulary_size: usize,
    /// Model-specific parameters
    pub parameters: std::collections::HashMap<String, String>,
}

impl SavedModel {
    /// Save model to file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), IoError> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        bincode::serialize_into(writer, self)?;
        Ok(())
    }

    /// Load model from file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, IoError> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let model = bincode::deserialize_from(reader)?;
        Ok(model)
    }
}

/// Load sample documents from a text file
/// Each line is treated as a separate document
pub fn load_text_documents<P: AsRef<Path>>(path: P) -> Result<Vec<String>, IoError> {
    let content = fs::read_to_string(path)?;
    Ok(content.lines().map(|s| s.to_string()).collect())
}

/// Save results to CSV format
pub fn save_results_csv<P: AsRef<Path>>(
    path: P,
    headers: &[&str],
    rows: &[Vec<String>],
) -> Result<(), IoError> {
    let mut file = File::create(path)?;

    // Write headers
    writeln!(file, "{}", headers.join(","))?;

    // Write rows
    for row in rows {
        writeln!(file, "{}", row.join(","))?;
    }

    Ok(())
}

/// Create directory if it doesn't exist
pub fn ensure_directory<P: AsRef<Path>>(path: P) -> Result<(), IoError> {
    fs::create_dir_all(path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_document_dataset() {
        let docs = vec![MarketDocument {
            id: "test_1".to_string(),
            title: "Test Document".to_string(),
            content: "This is a test".to_string(),
            timestamp: 1700000000000,
            symbols: vec!["BTC".to_string()],
            doc_type: "test".to_string(),
            market_context: None,
        }];

        let dataset = DocumentDataset::new("test", "unit_test", docs);
        assert_eq!(dataset.metadata.n_documents, 1);

        let texts = dataset.get_texts();
        assert_eq!(texts.len(), 1);
        assert!(texts[0].contains("Test Document"));
    }
}
