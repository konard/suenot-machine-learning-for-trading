//! Структуры данных для работы с рыночными данными
//!
//! Этот модуль содержит основные типы данных:
//! - `Kline` - OHLCV свечи
//! - `Ticker` - Текущая цена и статистика
//! - `OrderBook` - Стакан заявок

mod kline;
mod ticker;
mod orderbook;

pub use kline::Kline;
pub use ticker::Ticker;
pub use orderbook::{OrderBook, OrderBookLevel};

/// Результат расчёта фактора с метаданными
#[derive(Debug, Clone)]
pub struct FactorResult {
    /// Название фактора
    pub name: String,
    /// Значения фактора
    pub values: Vec<f64>,
    /// Временные метки (опционально)
    pub timestamps: Option<Vec<i64>>,
}

impl FactorResult {
    /// Создать новый результат фактора
    pub fn new(name: impl Into<String>, values: Vec<f64>) -> Self {
        Self {
            name: name.into(),
            values,
            timestamps: None,
        }
    }

    /// Добавить временные метки
    pub fn with_timestamps(mut self, timestamps: Vec<i64>) -> Self {
        self.timestamps = Some(timestamps);
        self
    }

    /// Получить последнее значение
    pub fn last(&self) -> Option<f64> {
        self.values.last().copied()
    }

    /// Получить среднее значение
    pub fn mean(&self) -> Option<f64> {
        if self.values.is_empty() {
            return None;
        }
        let sum: f64 = self.values.iter().sum();
        Some(sum / self.values.len() as f64)
    }
}
