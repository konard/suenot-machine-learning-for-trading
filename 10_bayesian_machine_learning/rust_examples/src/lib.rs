//! # Bayesian Crypto Trading
//!
//! A Rust library for Bayesian machine learning applied to cryptocurrency trading.
//! Uses Bybit exchange data for real-world examples.
//!
//! ## Modules
//!
//! - `data` - Data fetching from Bybit API and data structures
//! - `bayesian` - Core Bayesian statistics implementations
//! - `examples` - Trading-specific Bayesian models

pub mod data;
pub mod bayesian;

// Re-export commonly used types
pub use data::bybit::{BybitClient, Kline, Symbol};
pub use data::returns::Returns;
pub use bayesian::distributions::{Beta, Normal, StudentT};
pub use bayesian::inference::{ConjugatePrior, MCMC, MetropolisHastings};
