//! Error types for the BloombergGPT Trading library

use thiserror::Error;

/// Result type alias for this crate
pub type Result<T> = std::result::Result<T, Error>;

/// Main error type for the library
#[derive(Error, Debug)]
pub enum Error {
    /// API request failed
    #[error("API request failed: {0}")]
    ApiError(String),

    /// Rate limit exceeded
    #[error("Rate limit exceeded, retry after {retry_after_secs} seconds")]
    RateLimitError { retry_after_secs: u64 },

    /// Invalid configuration
    #[error("Invalid configuration: {0}")]
    ConfigError(String),

    /// Data parsing error
    #[error("Failed to parse data: {0}")]
    ParseError(String),

    /// Network error
    #[error("Network error: {0}")]
    NetworkError(#[from] reqwest::Error),

    /// JSON serialization/deserialization error
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),

    /// IO error
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// Invalid input
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Insufficient data for analysis
    #[error("Insufficient data: {0}")]
    InsufficientData(String),

    /// Model not available
    #[error("Model not available: {0}")]
    ModelNotAvailable(String),

    /// Backtest error
    #[error("Backtest error: {0}")]
    BacktestError(String),

    /// Data source error
    #[error("Data source error: {0}")]
    DataSourceError(String),
}

impl Error {
    /// Check if error is retryable
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            Error::RateLimitError { .. } | Error::NetworkError(_)
        )
    }

    /// Get suggested retry delay in seconds
    pub fn retry_delay(&self) -> Option<u64> {
        match self {
            Error::RateLimitError { retry_after_secs } => Some(*retry_after_secs),
            Error::NetworkError(_) => Some(5),
            _ => None,
        }
    }
}
