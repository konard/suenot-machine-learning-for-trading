//! # Crypto Time Series Analysis Library
//!
//! Библиотека для анализа временных рядов криптовалют с использованием данных Bybit.
//!
//! ## Модули
//!
//! - `api` - Клиент для работы с Bybit API
//! - `analysis` - Инструменты анализа временных рядов (стационарность, ACF/PACF)
//! - `models` - Модели прогнозирования (ARIMA, GARCH)
//! - `trading` - Торговые стратегии (парная торговля)

pub mod api;
pub mod analysis;
pub mod models;
pub mod trading;

/// Общие типы данных
pub mod types {
    use chrono::{DateTime, Utc};
    use serde::{Deserialize, Serialize};

    /// OHLCV свеча (Open, High, Low, Close, Volume)
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Candle {
        pub timestamp: DateTime<Utc>,
        pub open: f64,
        pub high: f64,
        pub low: f64,
        pub close: f64,
        pub volume: f64,
    }

    /// Временной ряд цен
    #[derive(Debug, Clone)]
    pub struct TimeSeries {
        pub symbol: String,
        pub data: Vec<f64>,
        pub timestamps: Vec<DateTime<Utc>>,
    }

    impl TimeSeries {
        pub fn new(symbol: &str) -> Self {
            Self {
                symbol: symbol.to_string(),
                data: Vec::new(),
                timestamps: Vec::new(),
            }
        }

        pub fn from_candles(symbol: &str, candles: &[Candle]) -> Self {
            Self {
                symbol: symbol.to_string(),
                data: candles.iter().map(|c| c.close).collect(),
                timestamps: candles.iter().map(|c| c.timestamp).collect(),
            }
        }

        pub fn len(&self) -> usize {
            self.data.len()
        }

        pub fn is_empty(&self) -> bool {
            self.data.is_empty()
        }

        /// Вычислить доходности (returns)
        pub fn returns(&self) -> Vec<f64> {
            if self.data.len() < 2 {
                return Vec::new();
            }
            self.data
                .windows(2)
                .map(|w| (w[1] - w[0]) / w[0])
                .collect()
        }

        /// Вычислить логарифмические доходности
        pub fn log_returns(&self) -> Vec<f64> {
            if self.data.len() < 2 {
                return Vec::new();
            }
            self.data
                .windows(2)
                .map(|w| (w[1] / w[0]).ln())
                .collect()
        }

        /// Первая разность (дифференцирование)
        pub fn diff(&self) -> Vec<f64> {
            if self.data.len() < 2 {
                return Vec::new();
            }
            self.data.windows(2).map(|w| w[1] - w[0]).collect()
        }
    }

    /// Результат статистического теста
    #[derive(Debug, Clone)]
    pub struct TestResult {
        pub test_name: String,
        pub statistic: f64,
        pub p_value: f64,
        pub critical_values: Vec<(String, f64)>,
        pub is_significant: bool,
    }

    impl TestResult {
        pub fn summary(&self) -> String {
            format!(
                "{}: statistic={:.4}, p-value={:.4}, significant={}",
                self.test_name, self.statistic, self.p_value, self.is_significant
            )
        }
    }
}

pub use types::*;
