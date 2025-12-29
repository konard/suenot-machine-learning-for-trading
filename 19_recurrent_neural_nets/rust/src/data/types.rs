//! Типы данных для работы с биржей Bybit

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Ошибки при работе с Bybit API
#[derive(Error, Debug)]
pub enum BybitError {
    #[error("Ошибка HTTP запроса: {0}")]
    HttpError(#[from] reqwest::Error),

    #[error("Ошибка парсинга JSON: {0}")]
    JsonError(#[from] serde_json::Error),

    #[error("Ошибка API Bybit: {code} - {message}")]
    ApiError { code: i32, message: String },

    #[error("Неверный интервал: {0}")]
    InvalidInterval(String),

    #[error("Нет данных")]
    NoData,
}

/// Интервал свечей (таймфрейм)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Interval {
    /// 1 минута
    M1,
    /// 3 минуты
    M3,
    /// 5 минут
    M5,
    /// 15 минут
    M15,
    /// 30 минут
    M30,
    /// 1 час
    H1,
    /// 2 часа
    H2,
    /// 4 часа
    H4,
    /// 6 часов
    H6,
    /// 12 часов
    H12,
    /// 1 день
    D1,
    /// 1 неделя
    W1,
    /// 1 месяц
    M1Month,
}

impl Interval {
    /// Преобразует строку в интервал
    pub fn from_str(s: &str) -> Result<Self, BybitError> {
        match s.to_lowercase().as_str() {
            "1" | "1m" => Ok(Interval::M1),
            "3" | "3m" => Ok(Interval::M3),
            "5" | "5m" => Ok(Interval::M5),
            "15" | "15m" => Ok(Interval::M15),
            "30" | "30m" => Ok(Interval::M30),
            "60" | "1h" => Ok(Interval::H1),
            "120" | "2h" => Ok(Interval::H2),
            "240" | "4h" => Ok(Interval::H4),
            "360" | "6h" => Ok(Interval::H6),
            "720" | "12h" => Ok(Interval::H12),
            "d" | "1d" | "day" => Ok(Interval::D1),
            "w" | "1w" | "week" => Ok(Interval::W1),
            "m" | "1month" | "month" => Ok(Interval::M1Month),
            _ => Err(BybitError::InvalidInterval(s.to_string())),
        }
    }

    /// Преобразует интервал в строку для API
    pub fn to_api_string(&self) -> &'static str {
        match self {
            Interval::M1 => "1",
            Interval::M3 => "3",
            Interval::M5 => "5",
            Interval::M15 => "15",
            Interval::M30 => "30",
            Interval::H1 => "60",
            Interval::H2 => "120",
            Interval::H4 => "240",
            Interval::H6 => "360",
            Interval::H12 => "720",
            Interval::D1 => "D",
            Interval::W1 => "W",
            Interval::M1Month => "M",
        }
    }

    /// Возвращает длительность интервала в миллисекундах
    pub fn duration_ms(&self) -> i64 {
        match self {
            Interval::M1 => 60_000,
            Interval::M3 => 180_000,
            Interval::M5 => 300_000,
            Interval::M15 => 900_000,
            Interval::M30 => 1_800_000,
            Interval::H1 => 3_600_000,
            Interval::H2 => 7_200_000,
            Interval::H4 => 14_400_000,
            Interval::H6 => 21_600_000,
            Interval::H12 => 43_200_000,
            Interval::D1 => 86_400_000,
            Interval::W1 => 604_800_000,
            Interval::M1Month => 2_592_000_000, // ~30 дней
        }
    }
}

/// Свеча (OHLCV данные)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candle {
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

    /// Объём в котируемой валюте (USDT)
    pub turnover: f64,
}

impl Candle {
    /// Создаёт новую свечу
    pub fn new(
        timestamp: i64,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
        turnover: f64,
    ) -> Self {
        Self {
            timestamp,
            open,
            high,
            low,
            close,
            volume,
            turnover,
        }
    }

    /// Возвращает время как DateTime
    pub fn datetime(&self) -> DateTime<Utc> {
        DateTime::from_timestamp_millis(self.timestamp).unwrap_or_default()
    }

    /// Рассчитывает изменение цены в процентах
    pub fn price_change_pct(&self) -> f64 {
        if self.open == 0.0 {
            0.0
        } else {
            (self.close - self.open) / self.open * 100.0
        }
    }

    /// Проверяет, является ли свеча бычьей (зелёной)
    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }

    /// Проверяет, является ли свеча медвежьей (красной)
    pub fn is_bearish(&self) -> bool {
        self.close < self.open
    }

    /// Рассчитывает размер тела свечи
    pub fn body_size(&self) -> f64 {
        (self.close - self.open).abs()
    }

    /// Рассчитывает размер верхней тени
    pub fn upper_shadow(&self) -> f64 {
        self.high - self.close.max(self.open)
    }

    /// Рассчитывает размер нижней тени
    pub fn lower_shadow(&self) -> f64 {
        self.close.min(self.open) - self.low
    }

    /// Рассчитывает полный размер свечи (high - low)
    pub fn full_range(&self) -> f64 {
        self.high - self.low
    }
}

/// Ответ API Bybit для свечей
#[derive(Debug, Deserialize)]
pub struct BybitKlineResponse {
    #[serde(rename = "retCode")]
    pub ret_code: i32,
    #[serde(rename = "retMsg")]
    pub ret_msg: String,
    pub result: BybitKlineResult,
}

#[derive(Debug, Deserialize)]
pub struct BybitKlineResult {
    pub symbol: String,
    pub category: String,
    pub list: Vec<Vec<String>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_candle_bullish() {
        let candle = Candle::new(0, 100.0, 110.0, 95.0, 105.0, 1000.0, 100000.0);
        assert!(candle.is_bullish());
        assert!(!candle.is_bearish());
    }

    #[test]
    fn test_candle_bearish() {
        let candle = Candle::new(0, 105.0, 110.0, 95.0, 100.0, 1000.0, 100000.0);
        assert!(candle.is_bearish());
        assert!(!candle.is_bullish());
    }

    #[test]
    fn test_price_change_pct() {
        let candle = Candle::new(0, 100.0, 110.0, 95.0, 110.0, 1000.0, 100000.0);
        assert_eq!(candle.price_change_pct(), 10.0);
    }

    #[test]
    fn test_interval_from_str() {
        assert_eq!(Interval::from_str("1h").unwrap(), Interval::H1);
        assert_eq!(Interval::from_str("4h").unwrap(), Interval::H4);
        assert_eq!(Interval::from_str("1d").unwrap(), Interval::D1);
    }
}
