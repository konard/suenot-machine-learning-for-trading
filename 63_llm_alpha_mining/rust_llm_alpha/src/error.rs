//! Error types for the LLM Alpha Mining library.

use thiserror::Error;

/// Main error type for the library.
#[derive(Error, Debug)]
pub enum Error {
    /// Error fetching data from API
    #[error("API error: {0}")]
    Api(String),

    /// Network error
    #[error("Network error: {0}")]
    Network(#[from] reqwest::Error),

    /// Invalid alpha expression
    #[error("Invalid expression: {0}")]
    InvalidExpression(String),

    /// Evaluation error
    #[error("Evaluation error: {0}")]
    Evaluation(String),

    /// Insufficient data
    #[error("Insufficient data: {0}")]
    InsufficientData(String),

    /// Invalid parameter
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    /// JSON parsing error
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Result type alias for the library.
pub type Result<T> = std::result::Result<T, Error>;
