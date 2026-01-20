//! Error types for the trading module.

use thiserror::Error;

/// Errors that can occur in the trading module.
#[derive(Error, Debug)]
pub enum TradingError {
    /// LLM API error
    #[error("LLM API error: {0}")]
    LLMError(String),

    /// JSON parsing error
    #[error("JSON parsing error: {0}")]
    ParseError(#[from] serde_json::Error),

    /// HTTP request error
    #[error("HTTP request error: {0}")]
    RequestError(#[from] reqwest::Error),

    /// Invalid configuration
    #[error("Invalid configuration: {0}")]
    ConfigError(String),

    /// Data not available
    #[error("Data not available: {0}")]
    DataError(String),

    /// Unknown error
    #[error("Unknown error: {0}")]
    Unknown(String),
}

/// Result type alias for trading operations.
pub type TradingResult<T> = Result<T, TradingError>;
