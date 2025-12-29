//! API Error Types

use thiserror::Error;

/// API error types
#[derive(Debug, Error)]
pub enum ApiError {
    #[error("HTTP request failed: {0}")]
    RequestError(#[from] reqwest::Error),

    #[error("Invalid interval: {0}")]
    InvalidInterval(String),

    #[error("API response error: code={code}, message={message}")]
    ApiResponseError { code: i32, message: String },

    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("No data returned")]
    NoData,
}

/// Result type for API operations
pub type ApiResult<T> = Result<T, ApiError>;
