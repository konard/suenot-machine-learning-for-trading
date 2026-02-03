//! API types and response structures

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// API error types
#[derive(Error, Debug)]
pub enum ApiError {
    #[error("HTTP request failed: {0}")]
    RequestError(#[from] reqwest::Error),

    #[error("API returned error: {code} - {message}")]
    ApiResponseError { code: i32, message: String },

    #[error("Failed to parse response: {0}")]
    ParseError(String),
}

/// Kline (candlestick) data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Kline {
    pub timestamp: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// Bybit API response wrapper
#[derive(Debug, Deserialize)]
pub struct ApiResponse<T> {
    #[serde(rename = "retCode")]
    pub ret_code: i32,
    #[serde(rename = "retMsg")]
    pub ret_msg: String,
    pub result: T,
}

/// Kline response result
#[derive(Debug, Deserialize)]
pub struct KlineResult {
    pub symbol: String,
    pub category: String,
    pub list: Vec<Vec<String>>,
}

/// Type alias for kline API response
pub type KlineResponse = ApiResponse<KlineResult>;

impl Kline {
    /// Create from raw string values (as returned by Bybit API)
    pub fn from_strings(values: &[String]) -> Result<Self, ApiError> {
        if values.len() < 6 {
            return Err(ApiError::ParseError(
                "Insufficient kline values".to_string(),
            ));
        }

        Ok(Kline {
            timestamp: values[0]
                .parse()
                .map_err(|_| ApiError::ParseError("Invalid timestamp".to_string()))?,
            open: values[1]
                .parse()
                .map_err(|_| ApiError::ParseError("Invalid open price".to_string()))?,
            high: values[2]
                .parse()
                .map_err(|_| ApiError::ParseError("Invalid high price".to_string()))?,
            low: values[3]
                .parse()
                .map_err(|_| ApiError::ParseError("Invalid low price".to_string()))?,
            close: values[4]
                .parse()
                .map_err(|_| ApiError::ParseError("Invalid close price".to_string()))?,
            volume: values[5]
                .parse()
                .map_err(|_| ApiError::ParseError("Invalid volume".to_string()))?,
        })
    }

    /// Calculate log return
    pub fn log_return(&self, prev_close: f64) -> f64 {
        (self.close / prev_close).ln()
    }

    /// Calculate true range
    pub fn true_range(&self, prev_close: f64) -> f64 {
        let hl = self.high - self.low;
        let hc = (self.high - prev_close).abs();
        let lc = (self.low - prev_close).abs();
        hl.max(hc).max(lc)
    }
}
