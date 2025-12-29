//! # Alpha Factors Library
//!
//! Библиотека для расчёта альфа-факторов на криптовалютных данных биржи Bybit.
//!
//! ## Модули
//!
//! - `api` - Клиент API Bybit для получения рыночных данных
//! - `data` - Структуры данных (свечи, тикеры и т.д.)
//! - `factors` - Реализации технических индикаторов и альфа-факторов
//!
//! ## Пример использования
//!
//! ```rust,no_run
//! use alpha_factors::{BybitClient, factors};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let client = BybitClient::new();
//!     let klines = client.get_klines("BTCUSDT", "1h", 100).await?;
//!
//!     let closes: Vec<f64> = klines.iter().map(|k| k.close).collect();
//!     let sma = factors::sma(&closes, 20);
//!
//!     println!("SMA(20): {:?}", sma.last());
//!     Ok(())
//! }
//! ```

pub mod api;
pub mod data;
pub mod factors;

// Re-exports for convenience
pub use api::BybitClient;
pub use data::{Kline, Ticker, OrderBook};
pub use factors::momentum::{rsi, roc, momentum};
pub use factors::trend::{sma, ema, macd, bollinger_bands};
pub use factors::volume::{obv, vwap, volume_sma};
pub use factors::volatility::{atr, historical_volatility};
pub use factors::alpha::{alpha_001, alpha_002, alpha_003};

/// Версия библиотеки
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
