//! Свечные данные (OHLCV)

use serde::{Deserialize, Serialize};
use std::fmt;

/// Интервал свечей
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
    MN,
}

impl fmt::Display for Interval {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
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
            Interval::MN => "M",
        };
        write!(f, "{}", s)
    }
}

impl Interval {
    /// Получить длительность интервала в миллисекундах
    pub fn to_millis(&self) -> u64 {
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
            Interval::MN => 2_592_000_000, // примерно 30 дней
        }
    }
}

/// Свеча (OHLCV)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Kline {
    /// Время открытия свечи (Unix timestamp в миллисекундах)
    pub timestamp: u64,
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
}

impl Kline {
    /// Создать новую свечу
    pub fn new(timestamp: u64, open: f64, high: f64, low: f64, close: f64, volume: f64) -> Self {
        Self {
            timestamp,
            open,
            high,
            low,
            close,
            volume,
        }
    }

    /// Тело свечи (разница между открытием и закрытием)
    pub fn body(&self) -> f64 {
        (self.close - self.open).abs()
    }

    /// Размах свечи (разница между максимумом и минимумом)
    pub fn range(&self) -> f64 {
        self.high - self.low
    }

    /// Бычья свеча (закрытие выше открытия)
    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }

    /// Медвежья свеча (закрытие ниже открытия)
    pub fn is_bearish(&self) -> bool {
        self.close < self.open
    }

    /// Типичная цена (среднее из high, low, close)
    pub fn typical_price(&self) -> f64 {
        (self.high + self.low + self.close) / 3.0
    }

    /// Средняя цена (среднее из open, high, low, close)
    pub fn average_price(&self) -> f64 {
        (self.open + self.high + self.low + self.close) / 4.0
    }

    /// Верхняя тень
    pub fn upper_shadow(&self) -> f64 {
        self.high - self.open.max(self.close)
    }

    /// Нижняя тень
    pub fn lower_shadow(&self) -> f64 {
        self.open.min(self.close) - self.low
    }
}

/// Набор свечей с методами для анализа
#[derive(Debug, Clone, Default)]
pub struct KlineData {
    pub klines: Vec<Kline>,
}

impl KlineData {
    /// Создать из вектора свечей
    pub fn new(klines: Vec<Kline>) -> Self {
        Self { klines }
    }

    /// Количество свечей
    pub fn len(&self) -> usize {
        self.klines.len()
    }

    /// Пустой ли набор
    pub fn is_empty(&self) -> bool {
        self.klines.is_empty()
    }

    /// Получить цены закрытия
    pub fn close_prices(&self) -> Vec<f64> {
        self.klines.iter().map(|k| k.close).collect()
    }

    /// Получить цены открытия
    pub fn open_prices(&self) -> Vec<f64> {
        self.klines.iter().map(|k| k.open).collect()
    }

    /// Получить максимальные цены
    pub fn high_prices(&self) -> Vec<f64> {
        self.klines.iter().map(|k| k.high).collect()
    }

    /// Получить минимальные цены
    pub fn low_prices(&self) -> Vec<f64> {
        self.klines.iter().map(|k| k.low).collect()
    }

    /// Получить объёмы
    pub fn volumes(&self) -> Vec<f64> {
        self.klines.iter().map(|k| k.volume).collect()
    }

    /// Получить типичные цены
    pub fn typical_prices(&self) -> Vec<f64> {
        self.klines.iter().map(|k| k.typical_price()).collect()
    }

    /// Последняя свеча
    pub fn last(&self) -> Option<&Kline> {
        self.klines.last()
    }

    /// Получить срез последних N свечей
    pub fn tail(&self, n: usize) -> &[Kline] {
        let start = self.klines.len().saturating_sub(n);
        &self.klines[start..]
    }

    /// Добавить свечу
    pub fn push(&mut self, kline: Kline) {
        self.klines.push(kline);
    }

    /// Срез данных
    pub fn slice(&self, start: usize, end: usize) -> Self {
        Self {
            klines: self.klines[start..end.min(self.klines.len())].to_vec(),
        }
    }
}

impl From<Vec<Kline>> for KlineData {
    fn from(klines: Vec<Kline>) -> Self {
        Self::new(klines)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kline_properties() {
        let kline = Kline::new(1704067200000, 42000.0, 42500.0, 41800.0, 42300.0, 1000.0);

        assert!(kline.is_bullish());
        assert!(!kline.is_bearish());
        assert_eq!(kline.body(), 300.0);
        assert_eq!(kline.range(), 700.0);
    }

    #[test]
    fn test_interval_display() {
        assert_eq!(Interval::M1.to_string(), "1");
        assert_eq!(Interval::H1.to_string(), "60");
        assert_eq!(Interval::D1.to_string(), "D");
    }
}
