//! # Squeeze-and-Excitation Trading Library
//!
//! This library implements Squeeze-and-Excitation (SE) networks for
//! algorithmic trading with cryptocurrency markets (Bybit integration).
//!
//! ## Overview
//!
//! SE networks dynamically weight different market features based on
//! current market conditions, allowing the model to focus on the most
//! relevant indicators.
//!
//! ## Modules
//!
//! - `models`: Core SE block implementation and trading models
//! - `data`: Bybit API integration and feature engineering
//! - `strategies`: Trading strategies using SE networks
//! - `utils`: Utility functions and metrics

pub mod models;
pub mod data;
pub mod strategies;
pub mod utils;

// Re-export commonly used items
pub use models::se_block::SEBlock;
pub use models::se_trading::SETradingModel;
pub use data::bybit::BybitClient;
pub use data::features::FeatureEngine;
pub use strategies::se_momentum::SEMomentumStrategy;
pub use strategies::signals::{TradingSignal, Direction};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Library prelude for convenient imports
pub mod prelude {
    pub use crate::models::se_block::SEBlock;
    pub use crate::models::se_trading::SETradingModel;
    pub use crate::models::activation::{relu, sigmoid, tanh};
    pub use crate::data::bybit::BybitClient;
    pub use crate::data::features::FeatureEngine;
    pub use crate::data::normalize::Normalizer;
    pub use crate::strategies::se_momentum::SEMomentumStrategy;
    pub use crate::strategies::signals::{TradingSignal, Direction};
    pub use crate::utils::metrics::PerformanceMetrics;
}
