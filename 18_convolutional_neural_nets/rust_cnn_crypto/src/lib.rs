//! # CNN Crypto Trading Library
//!
//! Библиотека для торговли криптовалютой с использованием сверточных нейронных сетей (CNN).
//! Использует данные с биржи Bybit.
//!
//! ## Модули
//!
//! - `bybit` - Клиент для API Bybit
//! - `data` - Обработка и подготовка данных
//! - `indicators` - Технические индикаторы
//! - `model` - Архитектура CNN модели
//! - `trading` - Торговые стратегии и бэктестинг

pub mod bybit;
pub mod data;
pub mod indicators;
pub mod model;
pub mod trading;

// Re-export commonly used types
pub use bybit::{BybitClient, Kline, BybitError};
pub use data::{Dataset, DataProcessor, Sample};
pub use indicators::{TechnicalIndicators, IndicatorConfig};
pub use model::{CnnModel, CnnConfig, TrainingConfig};
pub use trading::{Strategy, Backtest, Signal, Position};

/// Конфигурация приложения
#[derive(Debug, Clone)]
pub struct AppConfig {
    /// Символ торговой пары (например, "BTCUSDT")
    pub symbol: String,
    /// Интервал свечей (например, "15" для 15 минут)
    pub interval: String,
    /// Размер окна для входных данных
    pub window_size: usize,
    /// Горизонт прогнозирования (в свечах)
    pub prediction_horizon: usize,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            symbol: "BTCUSDT".to_string(),
            interval: "15".to_string(),
            window_size: 60,
            prediction_horizon: 4,
        }
    }
}
