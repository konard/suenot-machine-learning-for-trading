//! Типы данных для API

use chrono::{DateTime, TimeZone, Utc};
use serde::{Deserialize, Serialize};

/// OHLCV свеча (Kline)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Kline {
    /// Временная метка открытия (Unix timestamp в миллисекундах)
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
    /// Оборот в котируемой валюте
    pub turnover: f64,
}

impl Kline {
    /// Создание новой свечи
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

    /// Получение DateTime из timestamp
    pub fn datetime(&self) -> DateTime<Utc> {
        Utc.timestamp_millis_opt(self.timestamp)
            .single()
            .unwrap_or_else(Utc::now)
    }

    /// Расчёт доходности (return) свечи
    pub fn return_pct(&self) -> f64 {
        if self.open > 0.0 {
            (self.close - self.open) / self.open
        } else {
            0.0
        }
    }

    /// Размер тела свечи
    pub fn body_size(&self) -> f64 {
        (self.close - self.open).abs()
    }

    /// Размер верхней тени
    pub fn upper_shadow(&self) -> f64 {
        self.high - self.close.max(self.open)
    }

    /// Размер нижней тени
    pub fn lower_shadow(&self) -> f64 {
        self.close.min(self.open) - self.low
    }

    /// Является ли свеча бычьей (зелёной)
    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }

    /// Является ли свеча медвежьей (красной)
    pub fn is_bearish(&self) -> bool {
        self.close < self.open
    }
}

/// Набор данных свечей
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KlineData {
    /// Символ торговой пары
    pub symbol: String,
    /// Интервал
    pub interval: String,
    /// Данные свечей
    pub data: Vec<Kline>,
}

impl KlineData {
    /// Создание нового набора данных
    pub fn new(symbol: String, interval: String, data: Vec<Kline>) -> Self {
        Self {
            symbol,
            interval,
            data,
        }
    }

    /// Количество свечей
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Проверка на пустоту
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Получение цен закрытия
    pub fn close_prices(&self) -> Vec<f64> {
        self.data.iter().map(|k| k.close).collect()
    }

    /// Получение объёмов
    pub fn volumes(&self) -> Vec<f64> {
        self.data.iter().map(|k| k.volume).collect()
    }

    /// Расчёт доходностей
    pub fn returns(&self) -> Vec<f64> {
        if self.data.len() < 2 {
            return vec![];
        }

        self.data
            .windows(2)
            .map(|w| {
                if w[0].close > 0.0 {
                    (w[1].close - w[0].close) / w[0].close
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// Получение последней свечи
    pub fn last(&self) -> Option<&Kline> {
        self.data.last()
    }

    /// Получение первой свечи
    pub fn first(&self) -> Option<&Kline> {
        self.data.first()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kline_return() {
        let kline = Kline::new(1000000, 100.0, 110.0, 95.0, 105.0, 1000.0, 100000.0);

        assert!((kline.return_pct() - 0.05).abs() < 1e-10);
        assert!(kline.is_bullish());
        assert!(!kline.is_bearish());
    }

    #[test]
    fn test_kline_data_returns() {
        let data = vec![
            Kline::new(1000000, 100.0, 105.0, 98.0, 100.0, 1000.0, 100000.0),
            Kline::new(1060000, 100.0, 110.0, 99.0, 105.0, 1100.0, 110000.0),
            Kline::new(1120000, 105.0, 108.0, 102.0, 103.0, 900.0, 95000.0),
        ];

        let kline_data = KlineData::new("BTCUSDT".to_string(), "60".to_string(), data);
        let returns = kline_data.returns();

        assert_eq!(returns.len(), 2);
        assert!((returns[0] - 0.05).abs() < 1e-10); // (105-100)/100 = 0.05
    }
}
