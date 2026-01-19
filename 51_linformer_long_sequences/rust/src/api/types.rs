//! API types for exchange data responses.

use serde::{Deserialize, Serialize};

/// Kline (candlestick) data from Bybit API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Kline {
    /// Start timestamp in milliseconds
    pub start_time: u64,
    /// Open price
    pub open: f64,
    /// High price
    pub high: f64,
    /// Low price
    pub low: f64,
    /// Close price
    pub close: f64,
    /// Volume
    pub volume: f64,
    /// Turnover (quote volume)
    pub turnover: f64,
}

/// Bybit API response wrapper for kline data.
#[derive(Debug, Deserialize)]
pub struct KlineResponse {
    /// Return code (0 = success)
    #[serde(rename = "retCode")]
    pub ret_code: i32,
    /// Return message
    #[serde(rename = "retMsg")]
    pub ret_msg: String,
    /// Response result
    pub result: KlineResult,
}

/// Kline result data from Bybit API.
#[derive(Debug, Deserialize)]
pub struct KlineResult {
    /// Trading symbol
    pub symbol: String,
    /// Data category
    pub category: String,
    /// List of kline data
    pub list: Vec<Vec<String>>,
}

/// API error types.
#[derive(Debug, thiserror::Error)]
pub enum ApiError {
    /// HTTP request failed
    #[error("HTTP request failed: {0}")]
    RequestFailed(#[from] reqwest::Error),

    /// API returned an error
    #[error("API error (code {code}): {message}")]
    ApiError { code: i32, message: String },

    /// Failed to parse response
    #[error("Parse error: {0}")]
    ParseError(String),
}

impl Kline {
    /// Parse kline from Bybit API response format.
    /// Bybit returns: [startTime, open, high, low, close, volume, turnover]
    pub fn from_bybit_list(data: &[String]) -> Result<Self, ApiError> {
        if data.len() < 7 {
            return Err(ApiError::ParseError(format!(
                "Expected 7 fields, got {}",
                data.len()
            )));
        }

        Ok(Self {
            start_time: data[0].parse().map_err(|e| {
                ApiError::ParseError(format!("Invalid start_time: {}", e))
            })?,
            open: data[1].parse().map_err(|e| {
                ApiError::ParseError(format!("Invalid open: {}", e))
            })?,
            high: data[2].parse().map_err(|e| {
                ApiError::ParseError(format!("Invalid high: {}", e))
            })?,
            low: data[3].parse().map_err(|e| {
                ApiError::ParseError(format!("Invalid low: {}", e))
            })?,
            close: data[4].parse().map_err(|e| {
                ApiError::ParseError(format!("Invalid close: {}", e))
            })?,
            volume: data[5].parse().map_err(|e| {
                ApiError::ParseError(format!("Invalid volume: {}", e))
            })?,
            turnover: data[6].parse().map_err(|e| {
                ApiError::ParseError(format!("Invalid turnover: {}", e))
            })?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kline_from_bybit_list() {
        let data = vec![
            "1672531200000".to_string(),
            "16500.5".to_string(),
            "16600.0".to_string(),
            "16400.0".to_string(),
            "16550.25".to_string(),
            "1000.5".to_string(),
            "16525000.0".to_string(),
        ];

        let kline = Kline::from_bybit_list(&data).unwrap();
        assert_eq!(kline.start_time, 1672531200000);
        assert!((kline.open - 16500.5).abs() < 0.01);
        assert!((kline.close - 16550.25).abs() < 0.01);
    }

    #[test]
    fn test_kline_from_bybit_list_invalid() {
        let data = vec!["1672531200000".to_string()];
        assert!(Kline::from_bybit_list(&data).is_err());
    }
}
