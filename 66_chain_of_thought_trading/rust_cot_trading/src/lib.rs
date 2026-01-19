//! Chain-of-Thought Trading - Explainable AI Trading Toolkit
//!
//! This crate provides tools for explainable trading decisions using
//! Chain-of-Thought (CoT) prompting techniques with Large Language Models.
//!
//! # Key Features
//!
//! - **Transparent Reasoning**: Every trading decision includes a full reasoning chain
//! - **Multi-Step Analysis**: 6-step signal generation with trend, momentum, volume analysis
//! - **Risk Management**: Position sizing with CoT-based risk assessment
//! - **Audit Trails**: Complete backtesting with trade-by-trade reasoning logs
//!
//! # Modules
//!
//! - `analyzer`: Chain-of-Thought analysis engine
//! - `signals`: Multi-step trading signal generation
//! - `position`: Risk-aware position sizing with reasoning
//! - `backtest`: Backtesting engine with audit trails
//! - `data`: Market data loaders (Yahoo Finance, Bybit)
//! - `error`: Error types and handling
//!
//! # Example
//!
//! ```rust,no_run
//! use cot_trading::{CoTAnalyzer, SignalGenerator, Signal};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Initialize analyzer with mock (no API key needed)
//!     let analyzer = CoTAnalyzer::new_mock();
//!
//!     // Create signal generator
//!     let generator = SignalGenerator::new(analyzer);
//!
//!     // Generate trading signal with reasoning
//!     let signal = generator.generate(
//!         "AAPL",
//!         150.0,  // current price
//!         45.0,   // RSI
//!         0.5,    // MACD
//!         0.3,    // MACD signal
//!         148.0,  // SMA 20
//!         145.0,  // SMA 50
//!         1.2,    // volume ratio
//!         2.5,    // ATR
//!     ).await?;
//!
//!     println!("Signal: {:?}", signal.signal_type);
//!     println!("Reasoning: {:?}", signal.reasoning_chain);
//!
//!     Ok(())
//! }
//! ```

pub mod analyzer;
pub mod signals;
pub mod position;
pub mod backtest;
pub mod data;
pub mod error;

// Re-exports for convenience
pub use analyzer::{CoTAnalyzer, CoTAnalysis, ReasoningStep};
pub use signals::{SignalGenerator, CoTSignal, Signal};
pub use position::{PositionSizer, PositionResult};
pub use backtest::{Backtester, BacktestConfig, BacktestResult, Trade, TradeDirection};
pub use data::{DataLoader, OHLCV, YahooLoader, BybitLoader, MockLoader, add_indicators, BarWithIndicators};
pub use error::{Error, Result};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
