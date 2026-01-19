//! FinGPT Trading - Open-source Financial LLM Trading Toolkit
//!
//! This crate provides tools for financial sentiment analysis, trading signal
//! generation, and backtesting using FinGPT-inspired approaches.
//!
//! FinGPT is an open-source alternative to proprietary financial LLMs,
//! featuring data-centric design and RLSP (Reinforcement Learning from Stock Prices).
//!
//! # Modules
//!
//! - `sentiment`: Financial sentiment analysis
//! - `signals`: Trading signal generation from sentiment scores
//! - `backtest`: Backtesting engine for LLM-derived trading strategies
//! - `data`: Market data loaders (stocks, crypto)
//! - `api`: External API clients (Bybit, etc.)
//!
//! # Example
//!
//! ```rust,no_run
//! use fingpt_trading::{SentimentAnalyzer, TradingSignalGenerator};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Initialize analyzer with mock mode for demo
//!     let analyzer = SentimentAnalyzer::new_mock();
//!
//!     // Analyze financial text
//!     let result = analyzer.analyze("Apple reports record quarterly earnings").await?;
//!     println!("Sentiment: {:?}, Score: {:.2}", result.sentiment, result.score);
//!
//!     Ok(())
//! }
//! ```

pub mod sentiment;
pub mod signals;
pub mod backtest;
pub mod data;
pub mod api;
pub mod error;

// Re-exports for convenience
pub use sentiment::{SentimentAnalyzer, SentimentResult, Sentiment};
pub use signals::{TradingSignal, SignalType, TradingSignalGenerator};
pub use backtest::{Backtester, BacktestConfig, BacktestResult, Trade};
pub use data::{MarketDataLoader, OHLCVBar, DataSource};
pub use error::{Error, Result};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
