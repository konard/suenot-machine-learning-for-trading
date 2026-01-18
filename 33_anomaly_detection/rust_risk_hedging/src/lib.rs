//! Risk Hedging with Anomaly Detection for Cryptocurrency Trading
//!
//! This library provides tools for detecting market anomalies and
//! automatically hedging tail risks using Bybit exchange data.
//!
//! # Modules
//!
//! - `data`: Data fetching from Bybit API and OHLCV handling
//! - `features`: Feature engineering for risk detection
//! - `anomaly`: Anomaly detection algorithms (Z-Score, Isolation Forest, Mahalanobis)
//! - `risk`: Risk management and hedging strategies
//!
//! # Example
//!
//! ```no_run
//! use rust_risk_hedging::data::BybitClient;
//! use rust_risk_hedging::anomaly::EnsembleDetector;
//! use rust_risk_hedging::risk::HedgingStrategy;
//!
//! // Fetch market data
//! let client = BybitClient::public();
//! let data = client.get_klines("BTCUSDT", "60", 200, None, None).unwrap();
//!
//! // Create ensemble detector
//! let detector = EnsembleDetector::default();
//! let risk_score = detector.detect(&data);
//!
//! // Make hedging decision
//! let strategy = HedgingStrategy::default();
//! let hedge = strategy.decide(risk_score, 100_000.0);
//! println!("Hedge allocation: {:?}", hedge);
//! ```

pub mod anomaly;
pub mod data;
pub mod features;
pub mod risk;

pub use anomaly::*;
pub use data::*;
pub use features::*;
pub use risk::*;
