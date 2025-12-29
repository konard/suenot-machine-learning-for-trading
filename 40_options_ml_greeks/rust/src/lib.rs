//! # Options Greeks ML
//!
//! Библиотека для торговли волатильностью опционов на криптовалютном рынке (Bybit).
//!
//! ## Модули
//!
//! - `greeks` - Расчёт греков опционов (Delta, Gamma, Theta, Vega)
//! - `volatility` - Расчёт и предсказание волатильности
//! - `strategy` - Торговые стратегии (Straddle, Delta-hedging)
//! - `api` - Интеграция с биржей Bybit
//! - `models` - Типы данных и модели
//!
//! ## Пример использования
//!
//! ```rust
//! use options_greeks_ml::greeks::BlackScholes;
//! use options_greeks_ml::volatility::RealizedVolatility;
//!
//! // Расчёт цены опциона
//! let bs = BlackScholes::new(
//!     42000.0,  // spot price
//!     42000.0,  // strike
//!     7.0 / 365.0,  // time to expiry (7 days)
//!     0.05,     // risk-free rate
//!     0.55,     // volatility (55%)
//! );
//!
//! let call_price = bs.call_price();
//! let greeks = bs.greeks();
//!
//! println!("Call price: ${:.2}", call_price);
//! println!("Delta: {:.4}", greeks.delta);
//! ```

pub mod api;
pub mod greeks;
pub mod models;
pub mod strategy;
pub mod volatility;

// Re-exports для удобства
pub use api::bybit::BybitClient;
pub use greeks::BlackScholes;
pub use models::*;
pub use strategy::{DeltaHedger, StraddleStrategy};
pub use volatility::{ImpliedVolatility, RealizedVolatility};

/// Версия библиотеки
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Результат операции с возможной ошибкой
pub type Result<T> = std::result::Result<T, Error>;

/// Ошибки библиотеки
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("API error: {0}")]
    Api(#[from] api::ApiError),

    #[error("Calculation error: {0}")]
    Calculation(String),

    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    #[error("Data error: {0}")]
    Data(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}
