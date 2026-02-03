//! MambaTS: Improved Selective State Space Models for Long-term Time Series Forecasting
//!
//! This crate provides a Rust implementation of the MambaTS architecture
//! for financial time series forecasting. It includes:
//!
//! - Bybit API client for cryptocurrency data
//! - Data preprocessing and feature engineering
//! - Model inference for predictions
//! - Backtesting framework
//!
//! # Examples
//!
//! ```no_run
//! use mambats_time_series::{BybitClient, Interval, MambaTS, prepare_features};
//! use chrono::{Utc, Duration};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Fetch data from Bybit
//!     let client = BybitClient::new();
//!     let end_time = Utc::now();
//!     let start_time = end_time - Duration::days(30);
//!
//!     let candles = client
//!         .get_klines("BTCUSDT", Interval::OneHour, start_time, end_time)
//!         .await?;
//!
//!     // Prepare features
//!     let features = prepare_features(&candles);
//!
//!     // Load model and predict
//!     let model = MambaTS::load("models/mambats.bin")?;
//!     let prediction = model.predict(&features)?;
//!
//!     println!("Prediction: {:.4}", prediction);
//!     Ok(())
//! }
//! ```

pub mod api;
pub mod data;
pub mod model;

// Re-exports for convenience
pub use api::bybit::{BybitClient, Candle, Interval, Ticker};
pub use data::features::{prepare_features, TechnicalIndicators};
pub use data::processor::{DataProcessor, TimeSeriesDataset};
pub use model::mambats::MambaTS;
