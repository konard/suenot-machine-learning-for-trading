//! Типы данных для работы с рыночными данными
//!
//! Этот модуль содержит основные структуры данных для представления
//! ценовых данных, OHLCV свечей и временных рядов.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// OHLCV свеча (Open, High, Low, Close, Volume)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candle {
    /// Временная метка открытия свечи
    pub timestamp: DateTime<Utc>,
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
    /// Объём в базовой валюте (опционально)
    pub turnover: Option<f64>,
}

impl Candle {
    /// Создать новую свечу
    pub fn new(
        timestamp: DateTime<Utc>,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
    ) -> Self {
        Self {
            timestamp,
            open,
            high,
            low,
            close,
            volume,
            turnover: None,
        }
    }

    /// Типичная цена (typical price)
    pub fn typical_price(&self) -> f64 {
        (self.high + self.low + self.close) / 3.0
    }

    /// Средняя цена (average price)
    pub fn average_price(&self) -> f64 {
        (self.open + self.high + self.low + self.close) / 4.0
    }

    /// Размах свечи (range)
    pub fn range(&self) -> f64 {
        self.high - self.low
    }

    /// Тело свечи (body)
    pub fn body(&self) -> f64 {
        (self.close - self.open).abs()
    }

    /// Свеча бычья?
    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }

    /// Свеча медвежья?
    pub fn is_bearish(&self) -> bool {
        self.close < self.open
    }
}

/// Временной ряд цен
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceSeries {
    /// Символ актива
    pub symbol: String,
    /// Временной интервал (1m, 5m, 1h, 1d, etc.)
    pub interval: String,
    /// Свечи
    pub candles: Vec<Candle>,
}

impl PriceSeries {
    /// Создать новый временной ряд
    pub fn new(symbol: String, interval: String) -> Self {
        Self {
            symbol,
            interval,
            candles: Vec::new(),
        }
    }

    /// Добавить свечу
    pub fn push(&mut self, candle: Candle) {
        self.candles.push(candle);
    }

    /// Получить цены закрытия
    pub fn closes(&self) -> Vec<f64> {
        self.candles.iter().map(|c| c.close).collect()
    }

    /// Получить объёмы
    pub fn volumes(&self) -> Vec<f64> {
        self.candles.iter().map(|c| c.volume).collect()
    }

    /// Получить временные метки
    pub fn timestamps(&self) -> Vec<DateTime<Utc>> {
        self.candles.iter().map(|c| c.timestamp).collect()
    }

    /// Количество свечей
    pub fn len(&self) -> usize {
        self.candles.len()
    }

    /// Пустой ряд?
    pub fn is_empty(&self) -> bool {
        self.candles.is_empty()
    }

    /// Первая свеча
    pub fn first(&self) -> Option<&Candle> {
        self.candles.first()
    }

    /// Последняя свеча
    pub fn last(&self) -> Option<&Candle> {
        self.candles.last()
    }

    /// Сортировка по времени
    pub fn sort_by_time(&mut self) {
        self.candles.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));
    }

    /// Вычислить доходности (returns)
    pub fn returns(&self) -> Vec<f64> {
        let closes = self.closes();
        if closes.len() < 2 {
            return Vec::new();
        }

        closes
            .windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect()
    }

    /// Вычислить логарифмические доходности
    pub fn log_returns(&self) -> Vec<f64> {
        let closes = self.closes();
        if closes.len() < 2 {
            return Vec::new();
        }

        closes
            .windows(2)
            .map(|w| (w[1] / w[0]).ln())
            .collect()
    }
}

/// Портфель активов
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Portfolio {
    /// Веса активов (symbol -> weight)
    pub weights: std::collections::HashMap<String, f64>,
    /// Временная метка
    pub timestamp: DateTime<Utc>,
}

impl Portfolio {
    /// Создать новый портфель
    pub fn new(timestamp: DateTime<Utc>) -> Self {
        Self {
            weights: std::collections::HashMap::new(),
            timestamp,
        }
    }

    /// Установить вес актива
    pub fn set_weight(&mut self, symbol: &str, weight: f64) {
        self.weights.insert(symbol.to_string(), weight);
    }

    /// Получить вес актива
    pub fn get_weight(&self, symbol: &str) -> f64 {
        *self.weights.get(symbol).unwrap_or(&0.0)
    }

    /// Нормализовать веса (сумма = 1)
    pub fn normalize(&mut self) {
        let sum: f64 = self.weights.values().sum();
        if sum > 0.0 {
            for weight in self.weights.values_mut() {
                *weight /= sum;
            }
        }
    }

    /// Получить список активов с положительными весами
    pub fn active_assets(&self) -> Vec<&String> {
        self.weights
            .iter()
            .filter(|(_, &w)| w > 0.0)
            .map(|(s, _)| s)
            .collect()
    }

    /// Сумма весов
    pub fn total_weight(&self) -> f64 {
        self.weights.values().sum()
    }
}

/// Сигнал для торговли
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Signal {
    /// Покупка
    Long,
    /// Продажа
    Short,
    /// Нейтральная позиция
    Neutral,
    /// Выход в кеш
    Cash,
}

impl Signal {
    /// Преобразовать в числовое значение
    pub fn to_numeric(&self) -> f64 {
        match self {
            Signal::Long => 1.0,
            Signal::Short => -1.0,
            Signal::Neutral => 0.0,
            Signal::Cash => 0.0,
        }
    }
}

/// Сигналы для нескольких активов
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Signals {
    /// Сигналы (symbol -> signal)
    pub signals: std::collections::HashMap<String, Signal>,
    /// Временная метка
    pub timestamp: DateTime<Utc>,
}

impl Signals {
    /// Создать новый набор сигналов
    pub fn new(timestamp: DateTime<Utc>) -> Self {
        Self {
            signals: std::collections::HashMap::new(),
            timestamp,
        }
    }

    /// Установить сигнал
    pub fn set(&mut self, symbol: &str, signal: Signal) {
        self.signals.insert(symbol.to_string(), signal);
    }

    /// Получить сигнал
    pub fn get(&self, symbol: &str) -> Signal {
        *self.signals.get(symbol).unwrap_or(&Signal::Neutral)
    }

    /// Получить символы с сигналом Long
    pub fn long_symbols(&self) -> Vec<&String> {
        self.signals
            .iter()
            .filter(|(_, &s)| s == Signal::Long)
            .map(|(sym, _)| sym)
            .collect()
    }

    /// Получить символы с сигналом Short
    pub fn short_symbols(&self) -> Vec<&String> {
        self.signals
            .iter()
            .filter(|(_, &s)| s == Signal::Short)
            .map(|(sym, _)| sym)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_candle_properties() {
        let candle = Candle::new(
            Utc::now(),
            100.0, // open
            110.0, // high
            95.0,  // low
            105.0, // close
            1000.0, // volume
        );

        assert!(candle.is_bullish());
        assert!(!candle.is_bearish());
        assert_eq!(candle.range(), 15.0);
        assert_eq!(candle.body(), 5.0);
    }

    #[test]
    fn test_price_series_returns() {
        let mut series = PriceSeries::new("BTCUSDT".to_string(), "1d".to_string());

        let now = Utc::now();
        series.push(Candle::new(now, 100.0, 100.0, 100.0, 100.0, 1000.0));
        series.push(Candle::new(now, 100.0, 110.0, 100.0, 110.0, 1000.0));
        series.push(Candle::new(now, 110.0, 110.0, 100.0, 100.0, 1000.0));

        let returns = series.returns();
        assert_eq!(returns.len(), 2);
        assert!((returns[0] - 0.1).abs() < 1e-10); // 10% рост
        assert!((returns[1] - (-0.0909)).abs() < 0.001); // ~9% падение
    }

    #[test]
    fn test_portfolio_normalize() {
        let mut portfolio = Portfolio::new(Utc::now());
        portfolio.set_weight("BTC", 2.0);
        portfolio.set_weight("ETH", 1.0);
        portfolio.set_weight("SOL", 1.0);

        portfolio.normalize();

        assert!((portfolio.get_weight("BTC") - 0.5).abs() < 1e-10);
        assert!((portfolio.get_weight("ETH") - 0.25).abs() < 1e-10);
        assert!((portfolio.get_weight("SOL") - 0.25).abs() < 1e-10);
    }
}
