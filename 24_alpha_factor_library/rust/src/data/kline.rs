//! Структура данных для OHLCV свечей

use serde::{Deserialize, Serialize};

/// OHLCV свеча (Candlestick)
///
/// Содержит данные о цене за определённый период:
/// - Open (Открытие) - цена в начале периода
/// - High (Максимум) - максимальная цена за период
/// - Low (Минимум) - минимальная цена за период
/// - Close (Закрытие) - цена в конце периода
/// - Volume (Объём) - объём торгов за период
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Kline {
    /// Временная метка открытия (Unix timestamp в миллисекундах)
    pub timestamp: i64,
    /// Символ торговой пары (например, "BTCUSDT")
    pub symbol: String,
    /// Интервал свечи (например, "1m", "1h", "1d")
    pub interval: String,
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
    /// Объём в котируемой валюте (turnover)
    pub turnover: f64,
}

impl Kline {
    /// Создать новую свечу
    pub fn new(
        timestamp: i64,
        symbol: impl Into<String>,
        interval: impl Into<String>,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
        turnover: f64,
    ) -> Self {
        Self {
            timestamp,
            symbol: symbol.into(),
            interval: interval.into(),
            open,
            high,
            low,
            close,
            volume,
            turnover,
        }
    }

    /// Типичная цена (Typical Price) = (High + Low + Close) / 3
    pub fn typical_price(&self) -> f64 {
        (self.high + self.low + self.close) / 3.0
    }

    /// Средняя цена (Average Price) = (Open + High + Low + Close) / 4
    pub fn average_price(&self) -> f64 {
        (self.open + self.high + self.low + self.close) / 4.0
    }

    /// Медианная цена (Median Price) = (High + Low) / 2
    pub fn median_price(&self) -> f64 {
        (self.high + self.low) / 2.0
    }

    /// Размах свечи (Range) = High - Low
    pub fn range(&self) -> f64 {
        self.high - self.low
    }

    /// Тело свечи (Body) = |Close - Open|
    pub fn body(&self) -> f64 {
        (self.close - self.open).abs()
    }

    /// Верхняя тень (Upper Shadow)
    pub fn upper_shadow(&self) -> f64 {
        self.high - self.open.max(self.close)
    }

    /// Нижняя тень (Lower Shadow)
    pub fn lower_shadow(&self) -> f64 {
        self.open.min(self.close) - self.low
    }

    /// Свеча бычья (закрытие выше открытия)?
    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }

    /// Свеча медвежья (закрытие ниже открытия)?
    pub fn is_bearish(&self) -> bool {
        self.close < self.open
    }

    /// Дожи (тело очень маленькое)?
    pub fn is_doji(&self, threshold: f64) -> bool {
        let body_ratio = self.body() / self.range();
        body_ratio < threshold
    }

    /// Изменение цены в процентах
    pub fn change_percent(&self) -> f64 {
        if self.open == 0.0 {
            return 0.0;
        }
        ((self.close - self.open) / self.open) * 100.0
    }
}

/// Вспомогательные функции для работы со списком свечей
pub trait KlineVec {
    /// Извлечь все цены закрытия
    fn closes(&self) -> Vec<f64>;
    /// Извлечь все цены открытия
    fn opens(&self) -> Vec<f64>;
    /// Извлечь все максимумы
    fn highs(&self) -> Vec<f64>;
    /// Извлечь все минимумы
    fn lows(&self) -> Vec<f64>;
    /// Извлечь все объёмы
    fn volumes(&self) -> Vec<f64>;
    /// Извлечь все временные метки
    fn timestamps(&self) -> Vec<i64>;
}

impl KlineVec for Vec<Kline> {
    fn closes(&self) -> Vec<f64> {
        self.iter().map(|k| k.close).collect()
    }

    fn opens(&self) -> Vec<f64> {
        self.iter().map(|k| k.open).collect()
    }

    fn highs(&self) -> Vec<f64> {
        self.iter().map(|k| k.high).collect()
    }

    fn lows(&self) -> Vec<f64> {
        self.iter().map(|k| k.low).collect()
    }

    fn volumes(&self) -> Vec<f64> {
        self.iter().map(|k| k.volume).collect()
    }

    fn timestamps(&self) -> Vec<i64> {
        self.iter().map(|k| k.timestamp).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_bullish_kline() -> Kline {
        Kline::new(
            1700000000000,
            "BTCUSDT",
            "1h",
            100.0, // open
            110.0, // high
            95.0,  // low
            108.0, // close
            1000.0,
            100000.0,
        )
    }

    fn sample_bearish_kline() -> Kline {
        Kline::new(
            1700000000000,
            "BTCUSDT",
            "1h",
            108.0, // open
            112.0, // high
            98.0,  // low
            100.0, // close
            800.0,
            80000.0,
        )
    }

    #[test]
    fn test_bullish_bearish() {
        let bullish = sample_bullish_kline();
        let bearish = sample_bearish_kline();

        assert!(bullish.is_bullish());
        assert!(!bullish.is_bearish());
        assert!(bearish.is_bearish());
        assert!(!bearish.is_bullish());
    }

    #[test]
    fn test_typical_price() {
        let kline = sample_bullish_kline();
        // (110 + 95 + 108) / 3 = 104.33...
        assert!((kline.typical_price() - 104.333).abs() < 0.01);
    }

    #[test]
    fn test_range() {
        let kline = sample_bullish_kline();
        // 110 - 95 = 15
        assert_eq!(kline.range(), 15.0);
    }

    #[test]
    fn test_body() {
        let kline = sample_bullish_kline();
        // |108 - 100| = 8
        assert_eq!(kline.body(), 8.0);
    }

    #[test]
    fn test_change_percent() {
        let kline = sample_bullish_kline();
        // ((108 - 100) / 100) * 100 = 8%
        assert_eq!(kline.change_percent(), 8.0);
    }
}
