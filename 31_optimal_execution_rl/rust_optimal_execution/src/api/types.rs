//! Типы данных для API

use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

/// Интервал свечей
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Interval {
    /// 1 минута
    #[serde(rename = "1")]
    Min1,
    /// 3 минуты
    #[serde(rename = "3")]
    Min3,
    /// 5 минут
    #[serde(rename = "5")]
    Min5,
    /// 15 минут
    #[serde(rename = "15")]
    Min15,
    /// 30 минут
    #[serde(rename = "30")]
    Min30,
    /// 1 час
    #[serde(rename = "60")]
    Hour1,
    /// 2 часа
    #[serde(rename = "120")]
    Hour2,
    /// 4 часа
    #[serde(rename = "240")]
    Hour4,
    /// 1 день
    #[serde(rename = "D")]
    Day1,
    /// 1 неделя
    #[serde(rename = "W")]
    Week1,
}

impl Interval {
    /// Получить строковое представление интервала для API
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
            Self::Day1 => "D",
            Self::Week1 => "W",
        }
    }

    /// Получить длительность интервала в секундах
    pub fn seconds(&self) -> u64 {
        match self {
            Self::Min1 => 60,
            Self::Min3 => 180,
            Self::Min5 => 300,
            Self::Min15 => 900,
            Self::Min30 => 1800,
            Self::Hour1 => 3600,
            Self::Hour2 => 7200,
            Self::Hour4 => 14400,
            Self::Day1 => 86400,
            Self::Week1 => 604800,
        }
    }
}

impl std::fmt::Display for Interval {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// OHLCV свеча
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candle {
    /// Время открытия (Unix timestamp в миллисекундах)
    pub timestamp: i64,
    /// Цена открытия
    pub open: f64,
    /// Максимальная цена
    pub high: f64,
    /// Минимальная цена
    pub low: f64,
    /// Цена закрытия
    pub close: f64,
    /// Объём в базовой валюте
    pub volume: f64,
    /// Объём в котируемой валюте
    pub turnover: f64,
}

impl Candle {
    /// Получить DateTime из timestamp
    pub fn datetime(&self) -> DateTime<Utc> {
        DateTime::from_timestamp_millis(self.timestamp)
            .unwrap_or_else(|| Utc::now())
    }

    /// Рассчитать типичную цену (typical price)
    pub fn typical_price(&self) -> f64 {
        (self.high + self.low + self.close) / 3.0
    }

    /// Рассчитать VWAP для свечи
    pub fn vwap(&self) -> f64 {
        if self.volume > 0.0 {
            self.turnover / self.volume
        } else {
            self.typical_price()
        }
    }

    /// Рассчитать доходность
    pub fn return_pct(&self) -> f64 {
        if self.open > 0.0 {
            (self.close - self.open) / self.open
        } else {
            0.0
        }
    }

    /// Рассчитать True Range
    pub fn true_range(&self, prev_close: Option<f64>) -> f64 {
        match prev_close {
            Some(prev) => {
                let hl = self.high - self.low;
                let hc = (self.high - prev).abs();
                let lc = (self.low - prev).abs();
                hl.max(hc).max(lc)
            }
            None => self.high - self.low,
        }
    }
}

/// Уровень книги ордеров
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookLevel {
    /// Цена
    pub price: f64,
    /// Размер
    pub size: f64,
}

/// Книга ордеров
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBook {
    /// Символ
    pub symbol: String,
    /// Время обновления
    pub timestamp: i64,
    /// Цены покупки (bids)
    pub bids: Vec<OrderBookLevel>,
    /// Цены продажи (asks)
    pub asks: Vec<OrderBookLevel>,
}

impl OrderBook {
    /// Получить лучшую цену покупки
    pub fn best_bid(&self) -> Option<f64> {
        self.bids.first().map(|l| l.price)
    }

    /// Получить лучшую цену продажи
    pub fn best_ask(&self) -> Option<f64> {
        self.asks.first().map(|l| l.price)
    }

    /// Получить спред
    pub fn spread(&self) -> Option<f64> {
        match (self.best_ask(), self.best_bid()) {
            (Some(ask), Some(bid)) => Some(ask - bid),
            _ => None,
        }
    }

    /// Получить спред в процентах
    pub fn spread_pct(&self) -> Option<f64> {
        match (self.spread(), self.mid_price()) {
            (Some(spread), Some(mid)) if mid > 0.0 => Some(spread / mid),
            _ => None,
        }
    }

    /// Получить среднюю цену
    pub fn mid_price(&self) -> Option<f64> {
        match (self.best_ask(), self.best_bid()) {
            (Some(ask), Some(bid)) => Some((ask + bid) / 2.0),
            _ => None,
        }
    }

    /// Рассчитать дисбаланс книги ордеров
    pub fn imbalance(&self, depth: usize) -> f64 {
        let bid_volume: f64 = self.bids.iter().take(depth).map(|l| l.size).sum();
        let ask_volume: f64 = self.asks.iter().take(depth).map(|l| l.size).sum();
        let total = bid_volume + ask_volume;
        if total > 0.0 {
            (bid_volume - ask_volume) / total
        } else {
            0.0
        }
    }

    /// Рассчитать глубину ликвидности до определённой цены
    pub fn depth_at_price(&self, price_distance_pct: f64) -> (f64, f64) {
        let mid = self.mid_price().unwrap_or(0.0);
        let threshold = mid * price_distance_pct;

        let bid_depth: f64 = self.bids.iter()
            .filter(|l| mid - l.price <= threshold)
            .map(|l| l.size)
            .sum();

        let ask_depth: f64 = self.asks.iter()
            .filter(|l| l.price - mid <= threshold)
            .map(|l| l.size)
            .sum();

        (bid_depth, ask_depth)
    }
}

/// Сделка
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    /// Время сделки
    pub timestamp: i64,
    /// Символ
    pub symbol: String,
    /// Сторона (Buy/Sell)
    pub side: String,
    /// Цена
    pub price: f64,
    /// Размер
    pub size: f64,
    /// ID сделки
    pub trade_id: String,
}

impl Trade {
    /// Проверить, является ли сделка покупкой
    pub fn is_buy(&self) -> bool {
        self.side.to_lowercase() == "buy"
    }
}

/// Ответ API Bybit
#[derive(Debug, Deserialize)]
pub struct BybitResponse<T> {
    /// Код ответа (0 = успех)
    #[serde(rename = "retCode")]
    pub ret_code: i32,
    /// Сообщение
    #[serde(rename = "retMsg")]
    pub ret_msg: String,
    /// Данные
    pub result: T,
    /// Время ответа
    pub time: i64,
}

/// Результат запроса klines
#[derive(Debug, Deserialize)]
pub struct KlineResult {
    /// Символ
    pub symbol: String,
    /// Категория
    pub category: String,
    /// Список свечей
    pub list: Vec<Vec<String>>,
}

/// Результат запроса книги ордеров
#[derive(Debug, Deserialize)]
pub struct OrderBookResult {
    /// Символ
    #[serde(rename = "s")]
    pub symbol: String,
    /// Bids
    #[serde(rename = "b")]
    pub bids: Vec<Vec<String>>,
    /// Asks
    #[serde(rename = "a")]
    pub asks: Vec<String>,
    /// Timestamp
    pub ts: i64,
    /// Update ID
    #[serde(rename = "u")]
    pub update_id: i64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interval_seconds() {
        assert_eq!(Interval::Min1.seconds(), 60);
        assert_eq!(Interval::Hour1.seconds(), 3600);
        assert_eq!(Interval::Day1.seconds(), 86400);
    }

    #[test]
    fn test_candle_calculations() {
        let candle = Candle {
            timestamp: 1704067200000,
            open: 100.0,
            high: 110.0,
            low: 90.0,
            close: 105.0,
            volume: 1000.0,
            turnover: 101000.0,
        };

        assert!((candle.typical_price() - 101.666).abs() < 0.01);
        assert!((candle.vwap() - 101.0).abs() < 0.01);
        assert!((candle.return_pct() - 0.05).abs() < 0.001);
    }

    #[test]
    fn test_order_book() {
        let book = OrderBook {
            symbol: "BTCUSDT".to_string(),
            timestamp: 1704067200000,
            bids: vec![
                OrderBookLevel { price: 99.0, size: 10.0 },
                OrderBookLevel { price: 98.0, size: 20.0 },
            ],
            asks: vec![
                OrderBookLevel { price: 101.0, size: 15.0 },
                OrderBookLevel { price: 102.0, size: 25.0 },
            ],
        };

        assert_eq!(book.best_bid(), Some(99.0));
        assert_eq!(book.best_ask(), Some(101.0));
        assert_eq!(book.spread(), Some(2.0));
        assert_eq!(book.mid_price(), Some(100.0));
    }
}
