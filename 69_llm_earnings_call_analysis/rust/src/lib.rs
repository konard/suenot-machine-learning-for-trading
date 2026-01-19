//! # Earnings Call Analyzer
//!
//! A Rust implementation of LLM-powered earnings call analysis for trading.
//!
//! This library provides:
//! - Transcript parsing and section detection
//! - Sentiment analysis with financial lexicon
//! - Trading signal generation
//! - Bybit API integration for crypto data
//! - Backtesting framework
//!
//! ## Example
//!
//! ```rust,no_run
//! use earnings_call_analyzer::{
//!     EarningsAnalyzer,
//!     SignalGenerator,
//! };
//!
//! fn main() {
//!     let transcript = r#"
//!         John Smith - CEO:
//!         We delivered strong results this quarter with record revenue.
//!         We are confident in our growth trajectory.
//!     "#;
//!
//!     // Generate trading signal directly
//!     let signal_gen = SignalGenerator::new();
//!     let signal = signal_gen.generate_signal(transcript);
//!
//!     println!("{}", signal);
//! }
//! ```

pub mod analyzer;
pub mod api;
pub mod backtest;
pub mod trading;
pub mod utils;

// Re-export main types from analyzer
pub use analyzer::{
    EarningsAnalyzer, EarningsAnalysis,
    SentimentScore, ConfidenceScore, GuidanceAssessment, QAQuality,
    TranscriptParser, TranscriptSegment, SpeakerRole,
    SentimentAnalyzer, SentimentResult,
};

// Re-export from api
pub use api::{BybitClient, Candle, Ticker, OrderBook};

// Re-export from backtest
pub use backtest::{Backtester, BacktestResults, BacktestConfig, EarningsEvent, Trade};

// Re-export from trading
pub use trading::{SignalGenerator, TradingSignal, SignalType, SignalConfig};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default configuration for earnings analysis
pub mod config {
    /// Default sentiment weight in signal calculation
    pub const DEFAULT_SENTIMENT_WEIGHT: f64 = 0.35;

    /// Default confidence weight in signal calculation
    pub const DEFAULT_CONFIDENCE_WEIGHT: f64 = 0.20;

    /// Default guidance weight in signal calculation
    pub const DEFAULT_GUIDANCE_WEIGHT: f64 = 0.30;

    /// Default Q&A quality weight in signal calculation
    pub const DEFAULT_QA_WEIGHT: f64 = 0.15;

    /// Default signal threshold for trading decisions
    pub const DEFAULT_SIGNAL_THRESHOLD: f64 = 0.3;

    /// Default hold period in days
    pub const DEFAULT_HOLD_PERIOD: usize = 5;

    /// Default position size as fraction
    pub const DEFAULT_POSITION_SIZE: f64 = 0.1;
}
