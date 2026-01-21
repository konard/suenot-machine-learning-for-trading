//! Error types for the LLM Backtesting Assistant

use thiserror::Error;

/// Custom error type for the library
#[derive(Error, Debug)]
pub enum Error {
    /// HTTP request error
    #[error("HTTP request failed: {0}")]
    HttpError(#[from] reqwest::Error),

    /// JSON serialization/deserialization error
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),

    /// API error response
    #[error("API error: {0}")]
    ApiError(String),

    /// Invalid data error
    #[error("Invalid data: {0}")]
    InvalidData(String),

    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// Calculation error
    #[error("Calculation error: {0}")]
    CalculationError(String),

    /// IO error
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

/// Result type alias using our custom Error
pub type Result<T> = std::result::Result<T, Error>;
