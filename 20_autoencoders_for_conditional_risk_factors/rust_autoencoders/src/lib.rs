//! # Crypto Autoencoders
//!
//! Библиотека для анализа криптовалютных данных с использованием автоэнкодеров.
//! Получает данные с биржи Bybit и применяет различные модели автоэнкодеров
//! для извлечения риск-факторов.
//!
//! ## Модули
//!
//! - `bybit_client` - Клиент для работы с API Bybit
//! - `data_processor` - Предобработка и нормализация данных
//! - `autoencoder` - Реализации различных автоэнкодеров
//! - `risk_factors` - Анализ риск-факторов
//! - `utils` - Вспомогательные функции
//!
//! ## Пример использования
//!
//! ```no_run
//! use crypto_autoencoders::{BybitClient, DataProcessor, Autoencoder};
//!
//! #[tokio::main]
//! async fn main() {
//!     // Получаем данные
//!     let client = BybitClient::new();
//!     let klines = client.get_klines("BTCUSDT", "1h", 1000).await.unwrap();
//!
//!     // Обрабатываем данные
//!     let processor = DataProcessor::new();
//!     let features = processor.extract_features(&klines);
//!
//!     // Обучаем автоэнкодер
//!     let mut ae = Autoencoder::new(features.ncols(), 8);
//!     ae.fit(&features, 100, 0.001);
//! }
//! ```

pub mod autoencoder;
pub mod bybit_client;
pub mod data_processor;
pub mod risk_factors;
pub mod utils;

// Re-exports для удобства
pub use autoencoder::{
    Autoencoder, ConditionalAutoencoder, DenoisingAutoencoder, VariationalAutoencoder,
};
pub use bybit_client::{BybitClient, Kline, OrderBook, Ticker};
pub use data_processor::{DataProcessor, Features, NormalizationMethod};
pub use risk_factors::{RiskFactor, RiskFactorAnalyzer};
pub use utils::{load_csv, save_csv};

/// Версия библиотеки
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Настройки по умолчанию
pub mod defaults {
    /// Размер скрытого слоя по умолчанию
    pub const LATENT_DIM: usize = 8;

    /// Количество эпох обучения
    pub const EPOCHS: usize = 100;

    /// Скорость обучения
    pub const LEARNING_RATE: f64 = 0.001;

    /// Размер батча
    pub const BATCH_SIZE: usize = 32;
}
