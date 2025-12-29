//! Anomaly Detection for Cryptocurrency Trading
//!
//! This library provides tools for detecting anomalies in cryptocurrency
//! market data, with a focus on Bybit exchange data.
//!
//! # Modules
//!
//! - `data`: Data fetching from Bybit API and OHLCV handling
//! - `features`: Feature engineering for anomaly detection
//! - `anomaly`: Anomaly detection algorithms
//! - `strategy`: Trading strategy based on anomaly signals
//!
//! # Example
//!
//! ```no_run
//! use rust_anomaly_crypto::data::{BybitClient, BybitConfig};
//! use rust_anomaly_crypto::anomaly::ZScoreDetector;
//!
//! let client = BybitClient::public();
//! let data = client.get_klines("BTCUSDT", "60", 100, None, None).unwrap();
//!
//! let detector = ZScoreDetector::new(20, 3.0);
//! let closes: Vec<f64> = data.data.iter().map(|c| c.close).collect();
//! let anomalies = detector.detect(&closes);
//! ```

pub mod anomaly;
pub mod data;
pub mod features;
pub mod strategy;

pub use anomaly::*;
pub use data::*;
pub use features::*;
pub use strategy::*;
