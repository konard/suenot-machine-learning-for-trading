//! DeBERTa-based Sentiment Analysis for Trading
//!
//! This crate provides tools for applying DeBERTa-style sentiment analysis
//! to financial text for generating trading signals.
//!
//! # Overview
//!
//! DeBERTa (Decoding-enhanced BERT with Disentangled Attention) achieves
//! state-of-the-art NLP performance. This crate implements a rule-based
//! sentiment engine inspired by DeBERTa's approach, along with data fetching
//! from Bybit and a backtesting framework.
//!
//! # Modules
//!
//! - `model`: Sentiment analysis with financial lexicon
//! - `data`: Data fetching from Bybit and synthetic generation
//! - `backtest`: Backtesting framework for sentiment strategies
//! - `trading`: Signal generation and trading utilities
//!
//! # Example
//!
//! ```rust,ignore
//! use deberta_trading::{SentimentModel, BybitClient};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Analyze sentiment
//!     let model = SentimentModel::new();
//!     let result = model.predict("Bitcoin surges on ETF approval");
//!     println!("Sentiment: {:?}, Score: {:.4}", result.label, result.score);
//!
//!     // Fetch price data
//!     let client = BybitClient::new();
//!     let klines = client.fetch_klines("BTCUSDT", "D", 100).await?;
//!
//!     Ok(())
//! }
//! ```

pub mod model;
pub mod data;
pub mod backtest;
pub mod trading;

// Re-exports for convenience
pub use model::deberta::SentimentModel;
pub use data::bybit::BybitClient;
pub use backtest::engine::SentimentBacktester;
