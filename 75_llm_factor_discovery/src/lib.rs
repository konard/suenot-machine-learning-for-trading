//! # LLM Factor Discovery for Quantitative Trading
//!
//! This crate provides tools for discovering and generating alpha factors using
//! Large Language Models (LLMs) for quantitative trading in both stock and
//! cryptocurrency markets.
//!
//! ## Features
//!
//! - Factor expression parsing and evaluation
//! - LLM-based factor generation (Alpha-GPT style)
//! - Information Coefficient (IC) calculation
//! - Factor backtesting framework
//! - Bybit exchange integration for crypto trading
//! - Multi-factor strategy combination
//!
//! ## Example
//!
//! ```rust,no_run
//! use llm_factor_discovery::{
//!     FactorDiscoverySystem, LlmClient, FactorExpr, BacktestConfig,
//!     BybitClient, BybitConfig,
//! };
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Initialize the LLM client
//!     let llm_client = LlmClient::new_openai("your-api-key")?;
//!
//!     // Create the factor discovery system
//!     let discovery = FactorDiscoverySystem::new(llm_client);
//!
//!     // Generate factor ideas based on market hypothesis
//!     let factors = discovery.generate_factors(
//!         "momentum reversal in oversold conditions"
//!     ).await?;
//!
//!     // Parse and evaluate a factor expression
//!     let expr = FactorExpr::parse("rank(ts_mean(returns(close, 5), 20))")?;
//!     println!("Factor expression: {:?}", expr);
//!
//!     // Backtest the factors
//!     let config = BacktestConfig::default();
//!     let results = discovery.backtest(&factors, &config).await?;
//!
//!     for result in results {
//!         println!("Factor: {} | IC: {:.4} | IC_IR: {:.4}",
//!             result.factor_name, result.ic, result.ic_ir);
//!     }
//!
//!     Ok(())
//! }
//! ```

pub mod data;
pub mod discovery;
pub mod evaluation;
pub mod parser;
pub mod strategy;
pub mod utils;

// Re-export main types for convenience
pub use data::{
    BybitClient, BybitConfig, MarketData, MarketDataError, OhlcvBar, OrderBook,
    PriceData, Ticker, TimeFrame,
};
pub use discovery::{
    FactorCandidate, FactorDiscoverySystem, FactorGenerator, FactorRefiner,
    GenerationConfig, LlmClient, LlmConfig, LlmError, PromptBuilder,
};
pub use evaluation::{
    BacktestConfig, BacktestResult, Backtester, FactorMetrics, IcCalculator,
    MetricsCalculator, PerformanceMetrics,
};
pub use parser::{
    FactorExpr, FactorExprError, FactorFunction, FactorLexer, FactorParser,
    FactorValidator, Token, ValidationResult,
};
pub use strategy::{
    ExecutionConfig, FactorCombiner, FactorSignal, FactorStrategy,
    SignalGenerator, TradeAction, TradingSignal,
};
pub use utils::{load_config, AppConfig, Metrics, MetricsRecorder};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default IC threshold for factor selection
pub const DEFAULT_IC_THRESHOLD: f64 = 0.03;

/// Default IC_IR threshold for factor selection
pub const DEFAULT_IC_IR_THRESHOLD: f64 = 0.5;

/// Maximum factor expression depth
pub const MAX_EXPRESSION_DEPTH: usize = 10;

/// Default lookback period for factor calculation (days)
pub const DEFAULT_LOOKBACK_DAYS: usize = 252;
