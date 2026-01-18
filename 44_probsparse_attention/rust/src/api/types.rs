//! Типы данных для Bybit API

use serde::{Deserialize, Serialize};
use thiserror::Error;
use chrono::{DateTime, Utc};

/// Ошибки Bybit API
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

    #[error("Invalid symbol: {0}")]
    InvalidSymbol(String),

    #[error("No data returned")]
    NoData,
}

/// Ответ Bybit API
#[derive(Debug, Deserialize)]
pub struct ApiResponse<T> {
    #[serde(rename = "retCode")]
    pub ret_code: i32,
    #[serde(rename = "retMsg")]
    pub ret_msg: String,
    pub result: Option<T>,
}

/// Результат запроса klines
#[derive(Debug, Deserialize)]
pub struct KlinesResult {
    pub symbol: String,
    pub category: String,
    pub list: Vec<Vec<String>>,
}

/// OHLCV свеча
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Kline {
    /// Timestamp в миллисекундах
    pub timestamp: u64,
    /// Цена открытия
    pub open: f64,
    /// Максимальная цена
    pub high: f64,
    /// Минимальная цена
    pub low: f64,
    /// Цена закрытия
    pub close: f64,
    /// Объём
    pub volume: f64,
    /// Оборот (turnover)
    pub turnover: f64,
}

impl Kline {
    /// Парсит свечу из массива строк Bybit API
    ///
    /// Формат: [timestamp, open, high, low, close, volume, turnover]
    pub fn from_bybit_array(arr: &[String]) -> Result<Self, BybitError> {
        if arr.len() < 7 {
            return Err(BybitError::ParseError(
                format!("Invalid kline array length: {}", arr.len())
            ));
        }

        let parse_f64 = |s: &str, field: &str| -> Result<f64, BybitError> {
            s.parse::<f64>()
                .map_err(|_| BybitError::ParseError(format!("Invalid {}: {}", field, s)))
        };

        let parse_u64 = |s: &str, field: &str| -> Result<u64, BybitError> {
            s.parse::<u64>()
                .map_err(|_| BybitError::ParseError(format!("Invalid {}: {}", field, s)))
        };

        Ok(Self {
            timestamp: parse_u64(&arr[0], "timestamp")?,
            open: parse_f64(&arr[1], "open")?,
            high: parse_f64(&arr[2], "high")?,
            low: parse_f64(&arr[3], "low")?,
            close: parse_f64(&arr[4], "close")?,
            volume: parse_f64(&arr[5], "volume")?,
            turnover: parse_f64(&arr[6], "turnover")?,
        })
    }

    /// Возвращает datetime UTC
    pub fn datetime(&self) -> DateTime<Utc> {
        DateTime::from_timestamp_millis(self.timestamp as i64)
            .unwrap_or_else(|| DateTime::from_timestamp(0, 0).unwrap())
    }

    /// Вычисляет log return
    pub fn log_return(&self, prev_close: f64) -> f64 {
        (self.close / prev_close).ln()
    }

    /// Вычисляет диапазон high-low
    pub fn range(&self) -> f64 {
        (self.high - self.low) / self.close
    }

    /// Вычисляет диапазон close-open
    pub fn body(&self) -> f64 {
        (self.close - self.open) / self.open
    }

    /// Проверяет, является ли свеча бычьей
    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }
}

/// Информация о тикере
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ticker {
    pub symbol: String,
    pub last_price: f64,
    pub high_24h: f64,
    pub low_24h: f64,
    pub volume_24h: f64,
    pub turnover_24h: f64,
    pub price_change_24h: f64,
    pub price_change_pct_24h: f64,
}

/// Уровень книги ордеров
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookLevel {
    pub price: f64,
    pub quantity: f64,
}

/// Книга ордеров
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBook {
    pub symbol: String,
    pub timestamp: u64,
    pub bids: Vec<OrderBookLevel>,
    pub asks: Vec<OrderBookLevel>,
}

impl OrderBook {
    /// Возвращает лучший бид
    pub fn best_bid(&self) -> Option<&OrderBookLevel> {
        self.bids.first()
    }

    /// Возвращает лучший аск
    pub fn best_ask(&self) -> Option<&OrderBookLevel> {
        self.asks.first()
    }

    /// Вычисляет спред
    pub fn spread(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some(ask.price - bid.price),
            _ => None,
        }
    }

    /// Вычисляет спред в процентах
    pub fn spread_pct(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => {
                let mid = (bid.price + ask.price) / 2.0;
                Some((ask.price - bid.price) / mid * 100.0)
            },
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kline_from_bybit_array() {
        let arr = vec![
            "1704067200000".to_string(),  // timestamp
            "42500.5".to_string(),         // open
            "43000.0".to_string(),         // high
            "42000.0".to_string(),         // low
            "42800.0".to_string(),         // close
            "1000.5".to_string(),          // volume
            "42500000.0".to_string(),      // turnover
        ];

        let kline = Kline::from_bybit_array(&arr).unwrap();

        assert_eq!(kline.timestamp, 1704067200000);
        assert!((kline.open - 42500.5).abs() < 0.01);
        assert!((kline.close - 42800.0).abs() < 0.01);
    }

    #[test]
    fn test_kline_log_return() {
        let kline = Kline {
            timestamp: 0,
            open: 100.0,
            high: 110.0,
            low: 90.0,
            close: 105.0,
            volume: 1000.0,
            turnover: 100000.0,
        };

        let prev_close = 100.0;
        let log_ret = kline.log_return(prev_close);

        // ln(105/100) ≈ 0.0488
        assert!((log_ret - 0.0488).abs() < 0.001);
    }

    #[test]
    fn test_kline_is_bullish() {
        let bullish = Kline {
            timestamp: 0,
            open: 100.0,
            high: 110.0,
            low: 95.0,
            close: 105.0,
            volume: 1000.0,
            turnover: 100000.0,
        };

        let bearish = Kline {
            timestamp: 0,
            open: 100.0,
            high: 105.0,
            low: 90.0,
            close: 95.0,
            volume: 1000.0,
            turnover: 100000.0,
        };

        assert!(bullish.is_bullish());
        assert!(!bearish.is_bullish());
    }

    #[test]
    fn test_orderbook_spread() {
        let orderbook = OrderBook {
            symbol: "BTCUSDT".to_string(),
            timestamp: 0,
            bids: vec![
                OrderBookLevel { price: 42000.0, quantity: 1.0 },
                OrderBookLevel { price: 41990.0, quantity: 2.0 },
            ],
            asks: vec![
                OrderBookLevel { price: 42010.0, quantity: 1.5 },
                OrderBookLevel { price: 42020.0, quantity: 2.5 },
            ],
        };

        let spread = orderbook.spread().unwrap();
        assert!((spread - 10.0).abs() < 0.01);

        let spread_pct = orderbook.spread_pct().unwrap();
        // 10 / 42005 * 100 ≈ 0.0238%
        assert!(spread_pct > 0.02 && spread_pct < 0.03);
    }
}
