//! # LLM News Interpretation for Trading
//!
//! This crate provides tools for interpreting financial news using Large Language Models
//! to generate trading signals for cryptocurrency markets.
//!
//! ## Features
//!
//! - News collection from multiple sources (Twitter, RSS, Telegram)
//! - LLM-based news analysis and interpretation
//! - Event classification and sentiment extraction
//! - Trading signal generation
//! - Integration with Bybit exchange
//!
//! ## Example
//!
//! ```rust,no_run
//! use llm_news_interpretation::{NewsAnalyzer, LlmClient, TradingSignalGenerator};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Initialize the LLM client
//!     let llm_client = LlmClient::new_openai("your-api-key")?;
//!
//!     // Create the news analyzer
//!     let analyzer = NewsAnalyzer::new(llm_client);
//!
//!     // Analyze a news item
//!     let news = "Bitcoin ETF approved by SEC";
//!     let analysis = analyzer.analyze(news).await?;
//!
//!     println!("Sentiment: {}", analysis.sentiment);
//!     println!("Event type: {:?}", analysis.event_type);
//!
//!     Ok(())
//! }
//! ```

pub mod analysis;
pub mod data;
pub mod llm;
pub mod news;
pub mod strategy;
pub mod utils;

// Re-export main types for convenience
pub use analysis::{
    AggregatedSignal, EventClassifier, EventType, SentimentAnalyzer,
    SentimentResult, SignalAggregator,
};
pub use data::{BybitClient, BybitConfig, MarketData, MarketDataError, OHLCV, OrderBook, Ticker};
pub use llm::{LlmClient, LlmConfig, LlmError, LlmResponse, NewsAnalyzer};
pub use news::{NewsCollector, NewsItem, NewsPreprocessor, NewsSource};
pub use strategy::{
    NewsStrategy, Position, PositionSide, PositionSizer, RiskConfig,
    RiskLevel, RiskManager, StrategyConfig, StrategySignal, TradeAction,
};
pub use utils::{AppConfig, Metrics, MetricsRecorder, load_config};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default confidence threshold for trading signals
pub const DEFAULT_CONFIDENCE_THRESHOLD: f64 = 0.7;

/// Maximum latency target for signal generation (milliseconds)
pub const MAX_LATENCY_MS: u64 = 4000;
