//! # Rule Extraction Trading
//!
//! A high-performance Rust library for extracting interpretable trading rules
//! from black-box machine learning models.
//!
//! ## Overview
//!
//! This library provides tools for:
//! - Extracting if-then rules from decision trees and neural networks
//! - Fetching market data from Bybit (crypto) and stock APIs
//! - Implementing rule-based trading strategies
//! - Backtesting strategies with full explainability
//!
//! ## Example
//!
//! ```rust,no_run
//! use rule_extraction_trading::{
//!     rules::{Rule, Condition, Operator},
//!     trading::RuleBasedStrategy,
//! };
//!
//! // Create trading rules
//! let rules = vec![
//!     Rule::new(
//!         vec![Condition::new(0, Operator::GreaterThan, 70.0)],
//!         -1, // SELL
//!         0.75,
//!     ),
//! ];
//!
//! let feature_names = vec!["RSI".to_string()];
//! let strategy = RuleBasedStrategy::new(rules, feature_names);
//! ```

pub mod data;
pub mod extraction;
pub mod rules;
pub mod trading;

// Re-export commonly used types
pub use data::{BybitClient, Kline, TickerInfo};
pub use extraction::DecisionTreeExtractor;
pub use rules::{Condition, Operator, Rule};
pub use trading::{BacktestResult, RuleBasedStrategy, Trade};
