//! API error types.

use thiserror::Error;

/// Bybit API error types.
#[derive(Error, Debug)]
pub enum ApiError {
    #[error("HTTP request failed: {0}")]
    HttpError(#[from] reqwest::Error),

    #[error("JSON parsing failed: {0}")]
    JsonError(#[from] serde_json::Error),

    #[error("API returned error: {code} - {message}")]
    BybitError { code: i32, message: String },

    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    #[error("Rate limit exceeded")]
    RateLimitExceeded,

    #[error("No data returned")]
    NoData,
}
