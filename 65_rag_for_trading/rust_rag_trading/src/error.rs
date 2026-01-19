//! Error handling for the RAG trading library.

use thiserror::Error;

/// Result type alias for the RAG trading library.
pub type Result<T> = std::result::Result<T, RagError>;

/// Errors that can occur in the RAG trading library.
#[derive(Error, Debug)]
pub enum RagError {
    /// Document not found
    #[error("Document not found: {0}")]
    DocumentNotFound(String),

    /// No documents in index
    #[error("No documents in index")]
    EmptyIndex,

    /// Invalid query
    #[error("Invalid query: {0}")]
    InvalidQuery(String),

    /// API error
    #[error("API error: {0}")]
    ApiError(String),

    /// Network error
    #[error("Network error: {0}")]
    NetworkError(#[from] reqwest::Error),

    /// JSON parsing error
    #[error("JSON parsing error: {0}")]
    JsonError(#[from] serde_json::Error),

    /// IO error
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// Invalid data
    #[error("Invalid data: {0}")]
    InvalidData(String),

    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// Data fetch error
    #[error("Data fetch error: {0}")]
    DataFetch(String),

    /// Parse error
    #[error("Parse error: {0}")]
    Parse(String),
}

impl RagError {
    /// Create a new API error.
    pub fn api_error(msg: impl Into<String>) -> Self {
        RagError::ApiError(msg.into())
    }

    /// Create a new invalid data error.
    pub fn invalid_data(msg: impl Into<String>) -> Self {
        RagError::InvalidData(msg.into())
    }
}
