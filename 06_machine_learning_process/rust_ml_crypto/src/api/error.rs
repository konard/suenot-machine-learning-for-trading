//! API error types

use thiserror::Error;

/// Errors that can occur when interacting with the API
#[derive(Error, Debug)]
pub enum ApiError {
    #[error("HTTP request failed: {0}")]
    RequestFailed(#[from] reqwest::Error),

    #[error("JSON parsing failed: {0}")]
    JsonParseError(#[from] serde_json::Error),

    #[error("API returned error: {code} - {message}")]
    ApiResponseError { code: i32, message: String },

    #[error("Rate limit exceeded")]
    RateLimitExceeded,

    #[error("Invalid symbol: {0}")]
    InvalidSymbol(String),

    #[error("Invalid interval: {0}")]
    InvalidInterval(String),

    #[error("Authentication failed: {0}")]
    AuthenticationFailed(String),

    #[error("No data available")]
    NoData,
}

/// Result type alias for API operations
pub type ApiResult<T> = Result<T, ApiError>;
