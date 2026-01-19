//! Error types for the multi-agent trading system.

use thiserror::Error;

/// Main error type for the trading system.
#[derive(Error, Debug)]
pub enum TradingError {
    #[error("Insufficient data: need at least {required} periods, got {actual}")]
    InsufficientData { required: usize, actual: usize },

    #[error("Invalid signal: {0}")]
    InvalidSignal(String),

    #[error("Agent error: {0}")]
    AgentError(String),

    #[error("Communication error: {0}")]
    CommunicationError(String),

    #[error("Data loading error: {0}")]
    DataError(String),

    #[error("API error: {0}")]
    ApiError(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("HTTP error: {0}")]
    HttpError(#[from] reqwest::Error),

    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),
}

/// Result type alias for trading operations.
pub type Result<T> = std::result::Result<T, TradingError>;
