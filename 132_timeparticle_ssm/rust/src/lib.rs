//! TimeParticle SSM - Particle-like Multiscale State Space Models for Trading
//!
//! This crate provides a Rust implementation of the TimeParticle architecture
//! for financial time series forecasting and trading signal generation.
//!
//! # Features
//!
//! - Multiscale temporal processing with particle-based representations
//! - Cross-scale attention for information sharing between time scales
//! - High-performance inference for real-time trading applications
//! - Integration with Bybit API for cryptocurrency data
//!
//! # Example
//!
//! ```rust,ignore
//! use timeparticle_ssm::model::TimeParticleSSM;
//! use timeparticle_ssm::data::BybitClient;
//!
//! // Create model
//! let model = TimeParticleSSM::new(20, 64, 32, 4, 3);
//!
//! // Fetch data
//! let client = BybitClient::new();
//! let data = client.fetch_klines("BTCUSDT", "60", 100)?;
//!
//! // Generate prediction
//! let signal = model.predict(&data);
//! ```

pub mod api;
pub mod data;
pub mod model;

pub use api::bybit::BybitClient;
pub use data::features::FeatureEngine;
pub use data::loader::DataLoader;
pub use model::particle::Particle;
pub use model::timeparticle::{TimeParticleSSM, TradingSignal};
