//! API error types

use thiserror::Error;

/// API error type
#[derive(Error, Debug)]
pub enum ApiError {
    /// HTTP request error
    #[error("HTTP request failed: {0}")]
    HttpError(#[from] reqwest::Error),

    /// JSON parsing error
    #[error("JSON parsing failed: {0}")]
    JsonError(#[from] serde_json::Error),

    /// Bybit API error
    #[error("Bybit API error (code {code}): {message}")]
    BybitError { code: i32, message: String },

    /// No data returned
    #[error("No data returned for {symbol}")]
    NoData { symbol: String },

    /// Invalid interval
    #[error("Invalid interval: {0}")]
    InvalidInterval(String),

    /// Rate limit exceeded
    #[error("Rate limit exceeded, retry after {retry_after} seconds")]
    RateLimited { retry_after: u64 },
}

impl ApiError {
    /// Create a Bybit API error
    pub fn bybit_error(code: i32, message: String) -> Self {
        ApiError::BybitError { code, message }
    }

    /// Create a no data error
    pub fn no_data(symbol: impl Into<String>) -> Self {
        ApiError::NoData {
            symbol: symbol.into(),
        }
    }

    /// Check if error is retryable
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            ApiError::HttpError(_) | ApiError::RateLimited { .. }
        )
    }
}
