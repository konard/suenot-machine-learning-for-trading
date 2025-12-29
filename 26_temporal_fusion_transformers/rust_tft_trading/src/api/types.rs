//! Типы данных для Bybit API

use chrono::{DateTime, TimeZone, Utc};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Ошибки при работе с Bybit API
#[derive(Error, Debug)]
pub enum BybitError {
    #[error("HTTP request failed: {0}")]
    RequestError(#[from] reqwest::Error),

    #[error("API error: {code} - {message}")]
    ApiError { code: i32, message: String },

    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("Invalid interval: {0}")]
    InvalidInterval(String),

    #[error("Rate limit exceeded")]
    RateLimitError,
}

/// Результат операции с Bybit API
pub type Result<T> = std::result::Result<T, BybitError>;

/// Свеча (OHLCV данные)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Kline {
    /// Время открытия (Unix timestamp в миллисекундах)
    pub open_time: i64,
    /// Цена открытия
    pub open: f64,
    /// Максимальная цена
    pub high: f64,
    /// Минимальная цена
    pub low: f64,
    /// Цена закрытия
    pub close: f64,
    /// Объем в базовой валюте
    pub volume: f64,
    /// Объем в котируемой валюте (turnover)
    pub turnover: f64,
}

impl Kline {
    /// Возвращает время открытия как DateTime
    pub fn datetime(&self) -> DateTime<Utc> {
        Utc.timestamp_millis_opt(self.open_time).unwrap()
    }

    /// Возвращает изменение цены (close - open)
    pub fn price_change(&self) -> f64 {
        self.close - self.open
    }

    /// Возвращает процентное изменение цены
    pub fn returns(&self) -> f64 {
        if self.open == 0.0 {
            0.0
        } else {
            (self.close - self.open) / self.open
        }
    }

    /// Возвращает процентное изменение цены (в процентах)
    pub fn returns_percent(&self) -> f64 {
        self.returns() * 100.0
    }

    /// Возвращает размах свечи (high - low)
    pub fn range(&self) -> f64 {
        self.high - self.low
    }

    /// Возвращает типичную цену (high + low + close) / 3
    pub fn typical_price(&self) -> f64 {
        (self.high + self.low + self.close) / 3.0
    }

    /// Проверяет, является ли свеча бычьей
    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }

    /// Проверяет, является ли свеча медвежьей
    pub fn is_bearish(&self) -> bool {
        self.close < self.open
    }

    /// Возвращает тело свечи (абсолютное значение close - open)
    pub fn body(&self) -> f64 {
        (self.close - self.open).abs()
    }

    /// Возвращает верхнюю тень
    pub fn upper_shadow(&self) -> f64 {
        self.high - self.close.max(self.open)
    }

    /// Возвращает нижнюю тень
    pub fn lower_shadow(&self) -> f64 {
        self.close.min(self.open) - self.low
    }
}

/// Уровень в стакане заявок
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookLevel {
    /// Цена
    pub price: f64,
    /// Объем
    pub size: f64,
}

/// Стакан заявок
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBook {
    /// Символ
    pub symbol: String,
    /// Заявки на покупку (bids) - отсортированы по убыванию цены
    pub bids: Vec<OrderBookLevel>,
    /// Заявки на продажу (asks) - отсортированы по возрастанию цены
    pub asks: Vec<OrderBookLevel>,
    /// Время обновления (Unix timestamp в миллисекундах)
    pub timestamp: i64,
}

impl OrderBook {
    /// Возвращает лучшую цену покупки (bid)
    pub fn best_bid(&self) -> Option<f64> {
        self.bids.first().map(|l| l.price)
    }

    /// Возвращает лучшую цену продажи (ask)
    pub fn best_ask(&self) -> Option<f64> {
        self.asks.first().map(|l| l.price)
    }

    /// Возвращает спред (ask - bid)
    pub fn spread(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some(ask - bid),
            _ => None,
        }
    }

    /// Возвращает процентный спред
    pub fn spread_percent(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) if bid > 0.0 => Some((ask - bid) / bid * 100.0),
            _ => None,
        }
    }

    /// Возвращает mid price (среднее между best bid и best ask)
    pub fn mid_price(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some((bid + ask) / 2.0),
            _ => None,
        }
    }

    /// Возвращает глубину заявок на покупку до определенного количества уровней
    pub fn bid_depth(&self, levels: usize) -> f64 {
        self.bids.iter().take(levels).map(|l| l.size).sum()
    }

    /// Возвращает глубину заявок на продажу до определенного количества уровней
    pub fn ask_depth(&self, levels: usize) -> f64 {
        self.asks.iter().take(levels).map(|l| l.size).sum()
    }

    /// Возвращает дисбаланс стакана
    /// Положительный = больше покупателей, отрицательный = больше продавцов
    pub fn imbalance(&self, levels: usize) -> f64 {
        let bid_depth = self.bid_depth(levels);
        let ask_depth = self.ask_depth(levels);
        let total = bid_depth + ask_depth;
        if total == 0.0 {
            0.0
        } else {
            (bid_depth - ask_depth) / total
        }
    }

    /// Возвращает VWAP цену на покупку до определенного объема
    pub fn vwap_bid(&self, target_volume: f64) -> Option<f64> {
        let mut remaining = target_volume;
        let mut total_value = 0.0;
        let mut total_volume = 0.0;

        for level in &self.bids {
            let volume = level.size.min(remaining);
            total_value += level.price * volume;
            total_volume += volume;
            remaining -= volume;

            if remaining <= 0.0 {
                break;
            }
        }

        if total_volume > 0.0 {
            Some(total_value / total_volume)
        } else {
            None
        }
    }
}

/// Тикер (текущее состояние рынка)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ticker {
    /// Символ
    pub symbol: String,
    /// Последняя цена
    pub last_price: f64,
    /// Цена 24 часа назад
    pub prev_price_24h: f64,
    /// Изменение цены за 24 часа (процент)
    pub price_24h_pcnt: f64,
    /// Максимальная цена за 24 часа
    pub high_price_24h: f64,
    /// Минимальная цена за 24 часа
    pub low_price_24h: f64,
    /// Объем за 24 часа (в базовой валюте)
    pub volume_24h: f64,
    /// Оборот за 24 часа (в котируемой валюте)
    pub turnover_24h: f64,
    /// Открытый интерес (для фьючерсов)
    pub open_interest: Option<f64>,
}

impl Ticker {
    /// Возвращает размах цены за 24 часа
    pub fn range_24h(&self) -> f64 {
        self.high_price_24h - self.low_price_24h
    }

    /// Возвращает размах в процентах от low
    pub fn range_24h_percent(&self) -> f64 {
        if self.low_price_24h > 0.0 {
            (self.high_price_24h - self.low_price_24h) / self.low_price_24h * 100.0
        } else {
            0.0
        }
    }

    /// Проверяет, растет ли актив
    pub fn is_bullish(&self) -> bool {
        self.price_24h_pcnt > 0.0
    }
}

/// Ответ от API Bybit
#[derive(Debug, Deserialize)]
pub(crate) struct ApiResponse<T> {
    #[serde(rename = "retCode")]
    pub ret_code: i32,
    #[serde(rename = "retMsg")]
    pub ret_msg: String,
    pub result: T,
}

/// Результат запроса свечей
#[derive(Debug, Deserialize)]
pub(crate) struct KlineResult {
    pub symbol: String,
    pub category: String,
    pub list: Vec<Vec<String>>,
}

/// Результат запроса стакана
#[derive(Debug, Deserialize)]
pub(crate) struct OrderBookResult {
    pub s: String,
    pub b: Vec<Vec<String>>,
    pub a: Vec<Vec<String>>,
    pub ts: i64,
}

/// Результат запроса тикеров
#[derive(Debug, Deserialize)]
pub(crate) struct TickerResult {
    pub category: String,
    pub list: Vec<TickerData>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct TickerData {
    pub symbol: String,
    #[serde(rename = "lastPrice")]
    pub last_price: String,
    #[serde(rename = "prevPrice24h")]
    pub prev_price_24h: String,
    #[serde(rename = "price24hPcnt")]
    pub price_24h_pcnt: String,
    #[serde(rename = "highPrice24h")]
    pub high_price_24h: String,
    #[serde(rename = "lowPrice24h")]
    pub low_price_24h: String,
    #[serde(rename = "volume24h")]
    pub volume_24h: String,
    #[serde(rename = "turnover24h")]
    pub turnover_24h: String,
    #[serde(rename = "openInterest")]
    pub open_interest: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kline_calculations() {
        let kline = Kline {
            open_time: 1700000000000,
            open: 100.0,
            high: 110.0,
            low: 95.0,
            close: 105.0,
            volume: 1000.0,
            turnover: 100000.0,
        };

        assert_eq!(kline.price_change(), 5.0);
        assert!((kline.returns() - 0.05).abs() < 1e-10);
        assert_eq!(kline.range(), 15.0);
        assert!((kline.typical_price() - 103.333).abs() < 0.01);
        assert!(kline.is_bullish());
        assert!(!kline.is_bearish());
        assert_eq!(kline.body(), 5.0);
        assert_eq!(kline.upper_shadow(), 5.0);
        assert_eq!(kline.lower_shadow(), 5.0);
    }

    #[test]
    fn test_orderbook_calculations() {
        let orderbook = OrderBook {
            symbol: "BTCUSDT".to_string(),
            bids: vec![
                OrderBookLevel {
                    price: 100.0,
                    size: 10.0,
                },
                OrderBookLevel {
                    price: 99.0,
                    size: 20.0,
                },
            ],
            asks: vec![
                OrderBookLevel {
                    price: 101.0,
                    size: 15.0,
                },
                OrderBookLevel {
                    price: 102.0,
                    size: 25.0,
                },
            ],
            timestamp: 1700000000000,
        };

        assert_eq!(orderbook.best_bid(), Some(100.0));
        assert_eq!(orderbook.best_ask(), Some(101.0));
        assert_eq!(orderbook.spread(), Some(1.0));
        assert_eq!(orderbook.mid_price(), Some(100.5));
        assert_eq!(orderbook.bid_depth(2), 30.0);
        assert_eq!(orderbook.ask_depth(2), 40.0);
    }
}
