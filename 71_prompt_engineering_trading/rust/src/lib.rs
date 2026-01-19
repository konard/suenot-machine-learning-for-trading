//! # Prompt Engineering for Trading
//!
//! This crate provides tools for using prompt engineering techniques
//! to generate trading signals from LLM analysis.
//!
//! ## Features
//!
//! - Financial sentiment analysis
//! - Trading signal generation
//! - Market regime detection
//! - Multiple LLM provider support
//!
//! ## Example
//!
//! ```rust,no_run
//! use prompt_engineering_trading::{
//!     FinancialSentimentAnalyzer,
//!     MockLLMClient,
//!     Sentiment,
//! };
//!
//! #[tokio::main]
//! async fn main() {
//!     let client = MockLLMClient::new();
//!     let analyzer = FinancialSentimentAnalyzer::new(Box::new(client));
//!
//!     let result = analyzer
//!         .analyze("Apple beats earnings expectations", "AAPL")
//!         .await
//!         .unwrap();
//!
//!     println!("Sentiment: {:?}", result.sentiment);
//! }
//! ```

pub mod llm_client;
pub mod prompts;
pub mod sentiment;
pub mod signals;
pub mod regime;
pub mod backtest;
pub mod data;
pub mod error;

pub use llm_client::{LLMClient, MockLLMClient, OpenAIClient};
pub use sentiment::{FinancialSentimentAnalyzer, Sentiment, SentimentResult};
pub use signals::{SignalGenerator, TradingSignal, SignalDirection};
pub use regime::{MarketRegimeDetector, MarketRegime, RegimeAnalysis};
pub use backtest::{Backtester, BacktestConfig, BacktestResult};
pub use error::TradingError;
