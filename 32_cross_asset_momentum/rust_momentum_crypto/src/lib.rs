//! # Momentum Crypto
//!
//! Библиотека для реализации кросс-активного моментума на криптовалютном рынке.
//!
//! ## Модули
//!
//! - `data` - Работа с данными и Bybit API
//! - `momentum` - Расчёт моментума (time-series, cross-sectional, dual)
//! - `strategy` - Генерация сигналов и расчёт весов
//! - `backtest` - Движок бэктестинга и метрики
//! - `utils` - Конфигурация и утилиты
//!
//! ## Быстрый старт
//!
//! ```no_run
//! use momentum_crypto::{
//!     data::{BybitClient, get_momentum_universe},
//!     momentum::DualMomentum,
//! };
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Создаём клиент Bybit
//!     let client = BybitClient::new();
//!
//!     // Получаем вселенную активов
//!     let universe = get_momentum_universe();
//!
//!     // Загружаем данные для первого актива
//!     let prices = client.get_klines(universe[0], "D", None, None, Some(100)).await?;
//!
//!     println!("Загружено {} свечей для {}", prices.len(), universe[0]);
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Пример расчёта dual momentum
//!
//! ```no_run
//! use momentum_crypto::{
//!     momentum::{DualMomentum, DualMomentumConfig},
//!     data::PriceSeries,
//! };
//! use std::collections::HashMap;
//!
//! fn example(price_data: HashMap<String, PriceSeries>) {
//!     let config = DualMomentumConfig::crypto();
//!     let strategy = DualMomentum::new(config);
//!
//!     // Анализируем активы
//!     if let Ok(analysis) = strategy.analyze(&price_data) {
//!         for result in &analysis {
//!             if result.selected {
//!                 println!("Выбран {} с моментумом {:.2}%",
//!                     result.symbol,
//!                     result.ts_momentum * 100.0
//!                 );
//!             }
//!         }
//!     }
//! }
//! ```

pub mod backtest;
pub mod data;
pub mod momentum;
pub mod strategy;
pub mod utils;

/// Re-exports для удобства
pub use backtest::{BacktestConfig, BacktestEngine, BacktestResult, PerformanceMetrics};
pub use data::{BybitClient, Candle, Portfolio, PriceSeries, Signal, Signals};
pub use momentum::{
    CrossSectionalMomentum, CrossSectionalMomentumConfig, DualMomentum, DualMomentumConfig,
    TimeSeriesMomentum, TimeSeriesMomentumConfig,
};
pub use strategy::{SignalGenerator, WeightCalculator, WeightConfig};
pub use utils::StrategyConfig;

/// Версия библиотеки
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
