//! Error types for the FinGPT trading library

use thiserror::Error;

/// Result type alias for FinGPT operations
pub type Result<T> = std::result::Result<T, Error>;

/// Error types for FinGPT trading operations
#[derive(Error, Debug)]
pub enum Error {
    /// API request failed
    #[error("API error: {0}")]
    Api(String),

    /// Invalid input data
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Model inference error
    #[error("Model error: {0}")]
    Model(String),

    /// Data parsing error
    #[error("Parse error: {0}")]
    Parse(String),

    /// Network error
    #[error("Network error: {0}")]
    Network(#[from] reqwest::Error),

    /// JSON serialization error
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Generic IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Backtest error
    #[error("Backtest error: {0}")]
    Backtest(String),

    /// Configuration error
    #[error("Configuration error: {0}")]
    Config(String),
}
