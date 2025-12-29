//! # TFT Trading
//!
//! Библиотека для прогнозирования криптовалют с использованием
//! Temporal Fusion Transformers и данных с биржи Bybit.
//!
//! ## Модули
//!
//! - `api` - Клиент для работы с Bybit API
//! - `data` - Загрузка и предобработка данных
//! - `model` - Реализация архитектуры TFT
//! - `training` - Обучение модели
//! - `strategy` - Торговая стратегия и бэктестинг
//!
//! ## Пример использования
//!
//! ```no_run
//! use tft_trading::{BybitClient, DataLoader, TFTModel};
//!
//! #[tokio::main]
//! async fn main() {
//!     // Получаем данные
//!     let client = BybitClient::new();
//!     let klines = client.get_klines("BTCUSDT", "1h", 1000).await.unwrap();
//!
//!     // Подготавливаем данные
//!     let loader = DataLoader::new();
//!     let dataset = loader.prepare_dataset(&klines, 168, 24).unwrap();
//!
//!     // Создаём модель
//!     let model = TFTModel::new(TFTConfig::default());
//!
//!     // Делаем прогноз
//!     let prediction = model.predict(&dataset);
//! }
//! ```

pub mod api;
pub mod data;
pub mod model;
pub mod strategy;
pub mod training;

// Re-exports для удобства
pub use api::{BybitClient, BybitError, Kline, OrderBook, Ticker};
pub use data::{DataLoader, Dataset, Features, TimeSeriesDataset};
pub use model::{
    Attention, GatedResidualNetwork, QuantileLoss, TFTConfig, TFTModel, VariableSelectionNetwork,
};
pub use strategy::{BacktestResult, Signal, SignalGenerator, TradingStrategy};
pub use training::{Trainer, TrainingConfig};

/// Версия библиотеки
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Настройки по умолчанию
pub mod defaults {
    /// Размер скрытого слоя
    pub const HIDDEN_SIZE: usize = 64;

    /// Количество голов внимания
    pub const NUM_ATTENTION_HEADS: usize = 4;

    /// Dropout rate
    pub const DROPOUT: f64 = 0.1;

    /// Длина encoder context
    pub const ENCODER_LENGTH: usize = 168; // 7 дней часовых данных

    /// Длина prediction horizon
    pub const PREDICTION_LENGTH: usize = 24; // 24 часа

    /// Квантили для прогноза
    pub const QUANTILES: [f64; 3] = [0.1, 0.5, 0.9];

    /// Скорость обучения
    pub const LEARNING_RATE: f64 = 0.001;

    /// Размер батча
    pub const BATCH_SIZE: usize = 32;

    /// Количество эпох
    pub const EPOCHS: usize = 100;
}
