//! # LLM Finance - Fine-tuning Large Language Models for Financial Applications
//!
//! This crate provides tools for fine-tuning LLMs for financial applications
//! using parameter-efficient methods like LoRA, QLoRA, and Prefix-Tuning.
//!
//! ## Modules
//!
//! - `model`: LoRA layer implementations and financial sentiment models
//! - `trainer`: Training pipeline with early stopping
//! - `data_loader`: Yahoo Finance and Bybit data loaders
//! - `signals`: Trading signal generation
//! - `backtest`: Backtesting framework
//! - `evaluate`: Evaluation metrics
//!
//! ## Example
//!
//! ```rust,no_run
//! use llm_finance::{LoraLayer, FinancialSentimentModel, TrainingConfig};
//!
//! // Create a LoRA layer
//! let lora = LoraLayer::new(768, 768, 8, 16.0);
//!
//! // Create a sentiment model
//! let model = FinancialSentimentModel::new(768, 12, 8, 16.0, 3);
//! ```

pub mod model;
pub mod trainer;
pub mod data_loader;
pub mod signals;
pub mod backtest;
pub mod evaluate;

// Re-export main types
pub use model::{LoraLayer, FinancialSentimentModel, PrefixTuningLayer};
pub use trainer::{FineTuningTrainer, TrainingConfig, TrainingMetrics};
pub use data_loader::{YahooFinanceLoader, BybitDataLoader, OhlcvData};
pub use signals::{TradingSignalGenerator, TradingSignal, SignalType};
pub use backtest::{LlmBacktester, BacktestResult, BacktestConfig};
pub use evaluate::{evaluate_sentiment, evaluate_trading, SentimentMetrics, TradingMetrics};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
