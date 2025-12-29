//! Ошибки API

use thiserror::Error;

/// Ошибки при работе с API
#[derive(Error, Debug)]
pub enum ApiError {
    /// Ошибка HTTP запроса
    #[error("HTTP request failed: {0}")]
    RequestError(#[from] reqwest::Error),

    /// Ошибка парсинга JSON
    #[error("JSON parsing failed: {0}")]
    JsonError(#[from] serde_json::Error),

    /// Ошибка API Bybit
    #[error("Bybit API error: code={code}, message={message}")]
    BybitError { code: i32, message: String },

    /// Ошибка валидации данных
    #[error("Validation error: {0}")]
    ValidationError(String),

    /// Ошибка ограничения запросов
    #[error("Rate limit exceeded")]
    RateLimitError,

    /// Неизвестная ошибка
    #[error("Unknown error: {0}")]
    Unknown(String),
}

impl ApiError {
    /// Создать ошибку Bybit API
    pub fn bybit(code: i32, message: impl Into<String>) -> Self {
        Self::BybitError {
            code,
            message: message.into(),
        }
    }

    /// Создать ошибку валидации
    pub fn validation(message: impl Into<String>) -> Self {
        Self::ValidationError(message.into())
    }
}
