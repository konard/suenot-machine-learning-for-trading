//! # Rust Sentiment-Momentum Fusion
//!
//! Модульная библиотека для реализации стратегии Sentiment-Momentum Fusion
//! на криптовалютном рынке с использованием данных Bybit.
//!
//! ## Модули
//!
//! - `api` - Клиенты для получения данных (Bybit)
//! - `sentiment` - Анализ и агрегация настроений
//! - `momentum` - Расчёт ценового и sentiment моментума
//! - `signals` - Генерация торговых сигналов
//! - `models` - Общие модели данных

pub mod api;
pub mod models;
pub mod momentum;
pub mod sentiment;
pub mod signals;

// Re-exports для удобства использования
pub use api::{BybitClient, Kline, KlineData};
pub use models::{Signal, SignalType, TradingSignal};
pub use momentum::{MomentumCalculator, MomentumResult, PriceMomentum, SentimentMomentum};
pub use sentiment::{MockSentimentGenerator, SentimentAggregator, SentimentData, SentimentScore};
pub use signals::{DivergenceDetector, DivergenceType, FusionStrategy};
