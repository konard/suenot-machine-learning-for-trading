//! Типы данных для API Bybit

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Ошибки при работе с Bybit API
#[derive(Error, Debug)]
pub enum BybitError {
    #[error("HTTP request failed: {0}")]
    RequestError(#[from] reqwest::Error),

    #[error("API error: code={code}, message={message}")]
    ApiError { code: i32, message: String },

    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("Rate limit exceeded")]
    RateLimitExceeded,
}

/// Интервалы свечей
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KlineInterval {
    Min1,
    Min3,
    Min5,
    Min15,
    Min30,
    Hour1,
    Hour2,
    Hour4,
    Hour6,
    Hour12,
    Day1,
    Week1,
    Month1,
}

impl KlineInterval {
    /// Преобразование в строку для API
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Min1 => "1",
            Self::Min3 => "3",
            Self::Min5 => "5",
            Self::Min15 => "15",
            Self::Min30 => "30",
            Self::Hour1 => "60",
            Self::Hour2 => "120",
            Self::Hour4 => "240",
            Self::Hour6 => "360",
            Self::Hour12 => "720",
            Self::Day1 => "D",
            Self::Week1 => "W",
            Self::Month1 => "M",
        }
    }

    /// Длительность интервала в минутах
    pub fn minutes(&self) -> u64 {
        match self {
            Self::Min1 => 1,
            Self::Min3 => 3,
            Self::Min5 => 5,
            Self::Min15 => 15,
            Self::Min30 => 30,
            Self::Hour1 => 60,
            Self::Hour2 => 120,
            Self::Hour4 => 240,
            Self::Hour6 => 360,
            Self::Hour12 => 720,
            Self::Day1 => 1440,
            Self::Week1 => 10080,
            Self::Month1 => 43200,
        }
    }
}

impl std::str::FromStr for KlineInterval {
    type Err = BybitError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "1" => Ok(Self::Min1),
            "3" => Ok(Self::Min3),
            "5" => Ok(Self::Min5),
            "15" => Ok(Self::Min15),
            "30" => Ok(Self::Min30),
            "60" => Ok(Self::Hour1),
            "120" => Ok(Self::Hour2),
            "240" => Ok(Self::Hour4),
            "360" => Ok(Self::Hour6),
            "720" => Ok(Self::Hour12),
            "D" => Ok(Self::Day1),
            "W" => Ok(Self::Week1),
            "M" => Ok(Self::Month1),
            _ => Err(BybitError::ParseError(format!("Unknown interval: {}", s))),
        }
    }
}

/// Структура ответа Bybit API
#[derive(Debug, Deserialize)]
pub struct BybitResponse<T> {
    #[serde(rename = "retCode")]
    pub ret_code: i32,
    #[serde(rename = "retMsg")]
    pub ret_msg: String,
    pub result: T,
}

/// Результат запроса свечей
#[derive(Debug, Deserialize)]
pub struct KlineResult {
    pub symbol: String,
    pub category: String,
    pub list: Vec<Vec<String>>,
}

/// Свеча (OHLCV данные)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Kline {
    /// Время открытия свечи (Unix timestamp в миллисекундах)
    pub timestamp: i64,
    /// Цена открытия
    pub open: f64,
    /// Максимальная цена
    pub high: f64,
    /// Минимальная цена
    pub low: f64,
    /// Цена закрытия
    pub close: f64,
    /// Объём торгов
    pub volume: f64,
    /// Объём в базовой валюте
    pub turnover: f64,
}

impl Kline {
    /// Парсинг из массива строк API Bybit
    pub fn from_api_data(data: &[String]) -> Result<Self, BybitError> {
        if data.len() < 7 {
            return Err(BybitError::ParseError(
                "Not enough fields in kline data".to_string(),
            ));
        }

        Ok(Self {
            timestamp: data[0].parse().map_err(|e| {
                BybitError::ParseError(format!("Invalid timestamp: {}", e))
            })?,
            open: data[1].parse().map_err(|e| {
                BybitError::ParseError(format!("Invalid open price: {}", e))
            })?,
            high: data[2].parse().map_err(|e| {
                BybitError::ParseError(format!("Invalid high price: {}", e))
            })?,
            low: data[3].parse().map_err(|e| {
                BybitError::ParseError(format!("Invalid low price: {}", e))
            })?,
            close: data[4].parse().map_err(|e| {
                BybitError::ParseError(format!("Invalid close price: {}", e))
            })?,
            volume: data[5].parse().map_err(|e| {
                BybitError::ParseError(format!("Invalid volume: {}", e))
            })?,
            turnover: data[6].parse().map_err(|e| {
                BybitError::ParseError(format!("Invalid turnover: {}", e))
            })?,
        })
    }

    /// Время свечи как DateTime
    pub fn datetime(&self) -> DateTime<Utc> {
        DateTime::from_timestamp_millis(self.timestamp)
            .unwrap_or_else(|| DateTime::from_timestamp(0, 0).unwrap())
    }

    /// Размер тела свечи
    pub fn body_size(&self) -> f64 {
        (self.close - self.open).abs()
    }

    /// Размах свечи (high - low)
    pub fn range(&self) -> f64 {
        self.high - self.low
    }

    /// Бычья свеча (close > open)
    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }

    /// Медвежья свеча (close < open)
    pub fn is_bearish(&self) -> bool {
        self.close < self.open
    }

    /// Доходность свечи в процентах
    pub fn return_pct(&self) -> f64 {
        if self.open == 0.0 {
            return 0.0;
        }
        (self.close - self.open) / self.open * 100.0
    }
}
