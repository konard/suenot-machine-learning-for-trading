//! # Informer ProbSparse Attention
//!
//! Реализация модели Informer с механизмом ProbSparse Attention
//! для эффективного долгосрочного прогнозирования временных рядов.
//!
//! ## Особенности
//!
//! - ProbSparse Self-Attention: O(L·log(L)) вместо O(L²)
//! - Self-Attention Distilling: прогрессивное уменьшение длины последовательности
//! - Поддержка криптовалют (Bybit API) и акций
//!
//! ## Модули
//!
//! - `api` - Клиент для работы с Bybit API
//! - `data` - Загрузка и предобработка данных
//! - `model` - Реализация архитектуры Informer с ProbSparse
//! - `strategy` - Торговая стратегия и бэктестинг
//!
//! ## Пример использования
//!
//! ```no_run
//! use informer_probsparse::{BybitClient, DataLoader, InformerModel, InformerConfig};
//!
//! #[tokio::main]
//! async fn main() {
//!     // Получаем данные
//!     let client = BybitClient::new();
//!     let klines = client.get_klines("BTCUSDT", "60", 1000).await.unwrap();
//!
//!     // Подготавливаем данные
//!     let loader = DataLoader::new();
//!     let dataset = loader.prepare_dataset(&klines, 96, 24).unwrap();
//!
//!     // Создаём модель
//!     let config = InformerConfig {
//!         seq_len: 96,
//!         pred_len: 24,
//!         d_model: 64,
//!         n_heads: 4,
//!         sampling_factor: 5.0,
//!         use_distilling: true,
//!         ..Default::default()
//!     };
//!     let model = InformerModel::new(config);
//!
//!     // Делаем прогноз
//!     let predictions = model.predict(&dataset);
//! }
//! ```

pub mod api;
pub mod data;
pub mod model;
pub mod strategy;

// Re-exports для удобства
pub use api::{BybitClient, BybitError, Kline};
pub use data::{DataLoader, TimeSeriesDataset, Features};
pub use model::{
    InformerConfig, InformerModel, ProbSparseAttention,
    AttentionDistilling, AttentionWeights, OutputType,
};
pub use strategy::{BacktestResult, TradingSignal, SignalGenerator, TradingStrategy};

/// Версия библиотеки
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Настройки по умолчанию
pub mod defaults {
    /// Размер скрытого слоя
    pub const D_MODEL: usize = 64;

    /// Количество голов внимания
    pub const N_HEADS: usize = 4;

    /// Количество encoder layers
    pub const N_ENCODER_LAYERS: usize = 2;

    /// Dropout rate
    pub const DROPOUT: f64 = 0.1;

    /// Длина входной последовательности
    pub const SEQ_LEN: usize = 96;

    /// Длина прогноза
    pub const PRED_LEN: usize = 24;

    /// Скорость обучения
    pub const LEARNING_RATE: f64 = 0.001;

    /// Размер батча
    pub const BATCH_SIZE: usize = 32;

    /// Количество эпох
    pub const EPOCHS: usize = 100;

    /// Фактор разреженности для ProbSparse attention
    pub const SAMPLING_FACTOR: f64 = 5.0;

    /// Количество входных признаков
    pub const INPUT_FEATURES: usize = 6;
}
