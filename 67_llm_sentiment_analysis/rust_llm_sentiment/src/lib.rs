//! # LLM Sentiment Analysis
//!
//! Library for LLM-based sentiment analysis for cryptocurrency trading
//! with Bybit market data integration.
//!
//! ## Modules
//!
//! - `api` - Bybit API client and news fetching
//! - `data` - Text preprocessing and data loading
//! - `sentiment` - Sentiment analysis models and scoring
//! - `strategy` - Trading signals and backtesting
//!
//! ## Example Usage
//!
//! ```no_run
//! use llm_sentiment::{BybitClient, TextPreprocessor, SentimentAnalyzer, SignalGenerator};
//!
//! #[tokio::main]
//! async fn main() {
//!     // Fetch market data
//!     let client = BybitClient::new();
//!     let ticker = client.get_ticker("BTCUSDT").await.unwrap();
//!
//!     // Analyze sentiment
//!     let preprocessor = TextPreprocessor::new();
//!     let analyzer = SentimentAnalyzer::new();
//!
//!     let text = "Bitcoin surges as institutional adoption accelerates";
//!     let clean_text = preprocessor.preprocess(text);
//!     let sentiment = analyzer.analyze(&clean_text);
//!
//!     // Generate trading signal
//!     let signal_gen = SignalGenerator::default();
//!     let signal = signal_gen.generate(&sentiment);
//!
//!     println!("Signal: {:?}", signal);
//! }
//! ```

pub mod api;
pub mod data;
pub mod sentiment;
pub mod strategy;

// Re-exports for convenience
pub use api::{BybitClient, BybitError, Candle, Interval, Ticker};
pub use data::{TextPreprocessor, NewsArticle, NewsCollector};
pub use sentiment::{
    SentimentAnalyzer, SentimentResult, SentimentLevel, CryptoTerminology,
    FinancialLexicon, SentimentAggregator,
};
pub use strategy::{SignalGenerator, TradingSignal, BacktestResult, BacktestConfig, Backtester};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default configuration values
pub mod defaults {
    /// Sentiment threshold for bullish signal
    pub const BULLISH_THRESHOLD: f64 = 0.3;

    /// Sentiment threshold for bearish signal
    pub const BEARISH_THRESHOLD: f64 = -0.3;

    /// Minimum confidence for trading signal
    pub const MIN_CONFIDENCE: f64 = 0.6;

    /// Default lookback period for sentiment aggregation (hours)
    pub const LOOKBACK_HOURS: u64 = 24;

    /// Maximum position size (fraction of portfolio)
    pub const MAX_POSITION_SIZE: f64 = 0.5;

    /// Stop loss percentage
    pub const STOP_LOSS_PCT: f64 = 0.05;

    /// Take profit percentage
    pub const TAKE_PROFIT_PCT: f64 = 0.10;

    /// Sentiment decay half-life (hours)
    pub const SENTIMENT_DECAY_HOURS: f64 = 6.0;
}
