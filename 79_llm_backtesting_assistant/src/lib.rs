//! # LLM Backtesting Assistant
//!
//! A Rust library for analyzing trading strategy backtesting results using Large Language Models.
//!
//! ## Features
//!
//! - Calculate comprehensive performance metrics (Sharpe ratio, Sortino ratio, max drawdown, etc.)
//! - Fetch historical data from Bybit (crypto) and stock market APIs
//! - Generate structured analysis prompts for LLMs
//! - Parse and present LLM-generated insights
//!
//! ## Example
//!
//! ```rust,no_run
//! use llm_backtesting_assistant::prelude::*;
//!
//! #[tokio::main]
//! async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
//!     // Create sample backtest results
//!     let results = BacktestResults::sample();
//!
//!     // Create assistant and analyze
//!     let assistant = BacktestingAssistant::new("your-api-key".to_string());
//!     let analysis = assistant.analyze(&results).await?;
//!
//!     println!("{}", analysis);
//!     Ok(())
//! }
//! ```

pub mod analysis;
pub mod backtesting;
pub mod data;
pub mod metrics;
pub mod reports;

pub mod error;

pub use error::{Error, Result};

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::analysis::*;
    pub use crate::backtesting::*;
    pub use crate::data::*;
    pub use crate::metrics::*;
    pub use crate::reports::*;
    pub use crate::error::{Error, Result};
}
