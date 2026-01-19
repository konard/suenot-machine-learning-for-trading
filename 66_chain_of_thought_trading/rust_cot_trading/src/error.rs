//! Error types for the CoT trading library.

use thiserror::Error;

/// Result type alias for CoT trading operations.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors that can occur in the CoT trading library.
#[derive(Error, Debug)]
pub enum Error {
    /// API request failed
    #[error("API request failed: {0}")]
    ApiError(String),

    /// Invalid input data
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Data loading failed
    #[error("Failed to load data: {0}")]
    DataLoadError(String),

    /// Analysis failed
    #[error("Analysis failed: {0}")]
    AnalysisError(String),

    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// HTTP request error
    #[error("HTTP error: {0}")]
    HttpError(#[from] reqwest::Error),

    /// JSON parsing error
    #[error("JSON parsing error: {0}")]
    JsonError(#[from] serde_json::Error),

    /// IO error
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}
