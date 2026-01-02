//! # WaveNet Trading Library
//!
//! Библиотека для прогнозирования криптовалют с использованием архитектуры WaveNet
//! и данных биржи Bybit.
//!
//! ## Модули
//!
//! - `api` - Клиент для работы с Bybit API
//! - `models` - Реализация WaveNet и его компонентов
//! - `analysis` - Технические индикаторы и подготовка признаков
//! - `trading` - Торговые стратегии и бэктестинг

pub mod api;
pub mod models;
pub mod analysis;
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

    impl Candle {
        /// Типичная цена (Typical Price)
        pub fn typical_price(&self) -> f64 {
            (self.high + self.low + self.close) / 3.0
        }

        /// True Range
        pub fn true_range(&self, prev_close: Option<f64>) -> f64 {
            let hl = self.high - self.low;
            match prev_close {
                Some(pc) => {
                    let hc = (self.high - pc).abs();
                    let lc = (self.low - pc).abs();
                    hl.max(hc).max(lc)
                }
                None => hl,
            }
        }

        /// Доходность (Return)
        pub fn return_pct(&self) -> f64 {
            (self.close - self.open) / self.open
        }
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

        /// Нормализация (z-score)
        pub fn normalize(&self) -> Vec<f64> {
            let mean = self.data.iter().sum::<f64>() / self.data.len() as f64;
            let variance = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                / self.data.len() as f64;
            let std = variance.sqrt();

            if std < 1e-10 {
                return vec![0.0; self.data.len()];
            }

            self.data.iter().map(|x| (x - mean) / std).collect()
        }

        /// Min-Max нормализация
        pub fn normalize_minmax(&self) -> Vec<f64> {
            let min = self.data.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = self.data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let range = max - min;

            if range < 1e-10 {
                return vec![0.5; self.data.len()];
            }

            self.data.iter().map(|x| (x - min) / range).collect()
        }
    }

    /// Торговый сигнал
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum Signal {
        Buy,
        Sell,
        Hold,
    }

    impl Signal {
        pub fn as_str(&self) -> &'static str {
            match self {
                Signal::Buy => "BUY",
                Signal::Sell => "SELL",
                Signal::Hold => "HOLD",
            }
        }

        pub fn to_position(&self) -> f64 {
            match self {
                Signal::Buy => 1.0,
                Signal::Sell => -1.0,
                Signal::Hold => 0.0,
            }
        }
    }

    /// Результат прогноза
    #[derive(Debug, Clone)]
    pub struct Prediction {
        pub timestamp: DateTime<Utc>,
        pub predicted_return: f64,
        pub confidence: f64,
        pub signal: Signal,
    }

    /// Метрики производительности
    #[derive(Debug, Clone, Default)]
    pub struct PerformanceMetrics {
        pub total_return: f64,
        pub sharpe_ratio: f64,
        pub sortino_ratio: f64,
        pub max_drawdown: f64,
        pub win_rate: f64,
        pub profit_factor: f64,
        pub total_trades: usize,
    }

    impl PerformanceMetrics {
        pub fn print_summary(&self) {
            println!("=== Performance Metrics ===");
            println!("Total Return:   {:.2}%", self.total_return * 100.0);
            println!("Sharpe Ratio:   {:.3}", self.sharpe_ratio);
            println!("Sortino Ratio:  {:.3}", self.sortino_ratio);
            println!("Max Drawdown:   {:.2}%", self.max_drawdown * 100.0);
            println!("Win Rate:       {:.2}%", self.win_rate * 100.0);
            println!("Profit Factor:  {:.3}", self.profit_factor);
            println!("Total Trades:   {}", self.total_trades);
        }
    }
}

pub use types::*;
