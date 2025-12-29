//! # API модуль
//!
//! Интеграция с криптовалютными биржами для получения данных
//! и исполнения сделок.

pub mod bybit;

use thiserror::Error;

/// Ошибки API
#[derive(Error, Debug)]
pub enum ApiError {
    #[error("HTTP request failed: {0}")]
    Http(#[from] reqwest::Error),

    #[error("JSON parsing failed: {0}")]
    Json(#[from] serde_json::Error),

    #[error("API returned error: {code} - {message}")]
    ApiResponse { code: i32, message: String },

    #[error("Authentication failed: {0}")]
    Auth(String),

    #[error("Rate limit exceeded")]
    RateLimit,

    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    #[error("WebSocket error: {0}")]
    WebSocket(String),

    #[error("Connection timeout")]
    Timeout,
}

/// Результат API операции
pub type ApiResult<T> = Result<T, ApiError>;
