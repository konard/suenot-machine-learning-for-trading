//! Типы ошибок для API клиента

use thiserror::Error;

/// Ошибки API Bybit
#[derive(Error, Debug)]
pub enum ApiError {
    /// Ошибка HTTP запроса
    #[error("HTTP request failed: {0}")]
    HttpError(#[from] reqwest::Error),

    /// Ошибка парсинга JSON
    #[error("JSON parsing failed: {0}")]
    JsonError(#[from] serde_json::Error),

    /// Ошибка от API Bybit
    #[error("Bybit API error {code}: {message}")]
    BybitError {
        code: i32,
        message: String,
    },

    /// Неверный параметр
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    /// Данные не найдены
    #[error("No data found for {symbol}")]
    NoData {
        symbol: String,
    },

    /// Превышен лимит запросов (rate limit)
    #[error("Rate limit exceeded, retry after {retry_after} seconds")]
    RateLimited {
        retry_after: u64,
    },

    /// Таймаут запроса
    #[error("Request timed out")]
    Timeout,

    /// Неизвестная ошибка
    #[error("Unknown error: {0}")]
    Unknown(String),
}

impl ApiError {
    /// Создать ошибку от API Bybit
    pub fn bybit_error(code: i32, message: impl Into<String>) -> Self {
        ApiError::BybitError {
            code,
            message: message.into(),
        }
    }

    /// Создать ошибку "данные не найдены"
    pub fn no_data(symbol: impl Into<String>) -> Self {
        ApiError::NoData {
            symbol: symbol.into(),
        }
    }

    /// Создать ошибку "неверный параметр"
    pub fn invalid_param(msg: impl Into<String>) -> Self {
        ApiError::InvalidParameter(msg.into())
    }

    /// Это ошибка, которую можно повторить?
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            ApiError::HttpError(_) | ApiError::RateLimited { .. } | ApiError::Timeout
        )
    }
}
