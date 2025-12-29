//! # Crypto Event Surprise Prediction
//!
//! A library for detecting and predicting "earnings-like" surprises in cryptocurrency markets.
//!
//! This library adapts traditional earnings surprise concepts to crypto:
//! - Volume spikes as "earnings announcements"
//! - Historical patterns as "analyst estimates"
//! - Post-event drift analysis (PEAD analog)
//!
//! ## Modules
//!
//! - [`api`] - Bybit exchange API client
//! - [`data`] - Data types and processing
//! - [`events`] - Event detection and classification
//! - [`features`] - Feature engineering for predictions
//! - [`models`] - Simple prediction models
//! - [`analysis`] - Post-event analysis
//!
//! ## Example
//!
//! ```rust,no_run
//! use earnings_crypto::api::BybitClient;
//! use earnings_crypto::events::EventDetector;
//! use earnings_crypto::features::SurpriseCalculator;
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Fetch data
//!     let client = BybitClient::new();
//!     let candles = client.get_klines("BTCUSDT", "1h", 200).await?;
//!
//!     // Detect events
//!     let detector = EventDetector::default();
//!     let events = detector.detect_all_events(&candles);
//!
//!     // Calculate surprises
//!     let calculator = SurpriseCalculator::new(20);
//!     let surprises = calculator.calculate(&candles);
//!
//!     Ok(())
//! }
//! ```

pub mod api;
pub mod data;
pub mod events;
pub mod features;
pub mod models;
pub mod analysis;

// Re-exports for convenience
pub use api::BybitClient;
pub use data::types::{Candle, OrderBook, Trade};
pub use events::{CryptoEvent, EventDetector, EventType};
pub use features::{SurpriseCalculator, SurpriseMetrics};
