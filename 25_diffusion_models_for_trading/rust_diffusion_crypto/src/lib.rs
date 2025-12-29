//! # Diffusion Models for Cryptocurrency Forecasting
//!
//! This library implements Denoising Diffusion Probabilistic Models (DDPM)
//! for cryptocurrency price forecasting using data from Bybit exchange.
//!
//! ## Features
//!
//! - Bybit API client for fetching historical OHLCV data
//! - Technical indicator computation (RSI, MACD, Bollinger Bands, etc.)
//! - DDPM model with configurable noise schedules
//! - Training pipeline with checkpointing
//! - Probabilistic forecasting with uncertainty quantification
//!
//! ## Example
//!
//! ```rust,no_run
//! use diffusion_crypto::{
//!     data::BybitClient,
//!     model::{DDPM, NoiseSchedule},
//!     training::Trainer,
//! };
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Fetch data
//!     let client = BybitClient::new();
//!     let data = client.fetch_historical_klines("BTCUSDT", "60", 90).await?;
//!
//!     // Create and train model
//!     let schedule = NoiseSchedule::cosine(1000);
//!     let model = DDPM::new(7, 100, 24, 256, &schedule, tch::Device::Cpu);
//!
//!     Ok(())
//! }
//! ```

pub mod data;
pub mod model;
pub mod training;
pub mod utils;

pub use data::{BybitClient, OHLCV, FeatureEngineer};
pub use model::{DDPM, NoiseSchedule};
pub use training::{Trainer, TrainingConfig};
pub use utils::{Config, Checkpoint};
