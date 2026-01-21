//! Error types for the LLM compliance check library

use thiserror::Error;

/// Result type alias for compliance operations
pub type Result<T> = std::result::Result<T, Error>;

/// Error types for compliance checking operations
#[derive(Error, Debug)]
pub enum Error {
    /// Error from HTTP client
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    /// JSON serialization/deserialization error
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// LLM API error
    #[error("LLM API error: {0}")]
    LlmApi(String),

    /// Compliance check error
    #[error("Compliance check error: {0}")]
    ComplianceCheck(String),

    /// Regulation not found
    #[error("Regulation not found: {0}")]
    RegulationNotFound(String),

    /// Invalid configuration
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Validation error
    #[error("Validation error: {0}")]
    Validation(String),

    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Generic error
    #[error("{0}")]
    Generic(String),
}

impl Error {
    /// Create a new LLM API error
    pub fn llm_api(msg: impl Into<String>) -> Self {
        Self::LlmApi(msg.into())
    }

    /// Create a new compliance check error
    pub fn compliance_check(msg: impl Into<String>) -> Self {
        Self::ComplianceCheck(msg.into())
    }

    /// Create a new regulation not found error
    pub fn regulation_not_found(msg: impl Into<String>) -> Self {
        Self::RegulationNotFound(msg.into())
    }

    /// Create a new invalid config error
    pub fn invalid_config(msg: impl Into<String>) -> Self {
        Self::InvalidConfig(msg.into())
    }

    /// Create a new validation error
    pub fn validation(msg: impl Into<String>) -> Self {
        Self::Validation(msg.into())
    }
}
