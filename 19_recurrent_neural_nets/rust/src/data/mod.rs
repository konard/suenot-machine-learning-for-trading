//! # Модуль данных
//!
//! Получение рыночных данных с биржи Bybit.
//!
//! ## Пример использования
//!
//! ```rust,no_run
//! use crypto_rnn::data::BybitClient;
//!
//! #[tokio::main]
//! async fn main() {
//!     let client = BybitClient::new();
//!
//!     // Получаем 1000 часовых свечей для BTC
//!     let candles = client.get_klines("BTCUSDT", "1h", 1000).await.unwrap();
//!
//!     println!("Получено {} свечей", candles.len());
//! }
//! ```

mod bybit;
mod types;

pub use bybit::BybitClient;
pub use types::{Candle, Interval, BybitError};
