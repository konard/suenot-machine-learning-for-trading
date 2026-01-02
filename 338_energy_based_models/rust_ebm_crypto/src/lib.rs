//! Energy-Based Models for Cryptocurrency Trading
//!
//! This library provides tools for implementing Energy-Based Models (EBMs)
//! for cryptocurrency market analysis and trading, with data from Bybit exchange.
//!
//! # Modules
//!
//! - `data`: Data fetching from Bybit API and OHLCV handling
//! - `features`: Feature engineering for EBM training
//! - `ebm`: Energy-Based Model implementations
//! - `strategy`: Trading strategy based on energy signals
//!
//! # Example
//!
//! ```no_run
//! use rust_ebm_crypto::data::{BybitClient, OhlcvData};
//! use rust_ebm_crypto::ebm::EnergyModel;
//! use rust_ebm_crypto::features::FeatureEngine;
//!
//! // Fetch data from Bybit
//! let client = BybitClient::public();
//! let data = client.get_klines("BTCUSDT", "60", 1000, None, None).unwrap();
//!
//! // Extract features
//! let engine = FeatureEngine::default();
//! let features = engine.compute(&data.data);
//!
//! // Train EBM
//! let mut model = EnergyModel::new(features.ncols());
//! model.train(&features, 100);
//!
//! // Get energy scores for anomaly detection
//! let energies = model.energy_batch(&features);
//! ```

pub mod data;
pub mod ebm;
pub mod features;
pub mod strategy;

pub use data::*;
pub use ebm::*;
pub use features::*;
pub use strategy::*;
