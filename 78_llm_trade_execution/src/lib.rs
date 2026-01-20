//! # LLM Trade Execution
//!
//! This crate provides tools for optimizing trade execution using Large Language Models (LLMs)
//! to minimize market impact in both stock and cryptocurrency markets.
//!
//! ## Features
//!
//! - Traditional execution algorithms (TWAP, VWAP, Implementation Shortfall)
//! - Market impact estimation using Almgren-Chriss model
//! - LLM-based adaptive execution optimization
//! - Bybit exchange integration for cryptocurrency trading
//! - Real-time order book analysis
//! - Execution quality metrics and analytics
//!
//! ## Architecture
//!
//! The crate is organized into the following modules:
//!
//! - [`data`] - Market data structures and exchange connectivity (Bybit)
//! - [`execution`] - Execution engine, strategies, and LLM integration
//! - [`impact`] - Market impact models and estimation
//! - [`strategy`] - Execution strategy implementations (TWAP, VWAP, IS)
//! - [`utils`] - Configuration and utility functions
//!
//! ## Example
//!
//! ```rust,no_run
//! use llm_trade_execution::{
//!     ExecutionEngine, ExecutionConfig, ParentOrder, Side,
//!     TwapStrategy, MarketImpactEstimator, BybitClient, BybitConfig,
//! };
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Configure the execution engine
//!     let config = ExecutionConfig::default();
//!
//!     // Create a parent order
//!     let order = ParentOrder::new(
//!         "BTCUSDT".to_string(),
//!         Side::Buy,
//!         1.5,  // 1.5 BTC
//!         3600, // Execute over 1 hour
//!     );
//!
//!     // Create TWAP strategy
//!     let strategy = TwapStrategy::new(60); // 60-second slices
//!
//!     // Initialize the execution engine
//!     let mut engine = ExecutionEngine::new(config);
//!
//!     // Execute the order
//!     let result = engine.execute(order, Box::new(strategy)).await?;
//!
//!     println!("Execution complete:");
//!     println!("  Average price: {:.2}", result.average_price);
//!     println!("  Implementation shortfall: {:.4}%", result.implementation_shortfall * 100.0);
//!     println!("  VWAP slippage: {:.4}%", result.vwap_slippage * 100.0);
//!
//!     Ok(())
//! }
//! ```

pub mod data;
pub mod execution;
pub mod impact;
pub mod strategy;
pub mod utils;

// Re-export main types for convenience
pub use data::{
    BybitClient, BybitConfig, MarketData, MarketDataError, OhlcvBar, OrderBook,
    OrderBookLevel, PriceData, Ticker, TimeFrame, Trade, TradeDirection,
};
pub use execution::{
    ChildOrder, ChildOrderStatus, ExecutionConfig, ExecutionEngine, ExecutionError,
    ExecutionResult, ExecutionState, LlmAdapter, LlmConfig, LlmDecision, OrderId,
    ParentOrder, ParentOrderStatus, Side,
};
pub use impact::{
    AlmgrenChrissParams, ImpactComponent, MarketImpactError, MarketImpactEstimator,
    MarketImpactModel, TemporaryImpact, PermanentImpact,
};
pub use strategy::{
    AdaptiveStrategy, ExecutionSlice, ExecutionStrategy, ImplementationShortfallStrategy,
    LlmStrategy, StrategyConfig, TwapStrategy, VolumeProfile, VwapStrategy,
};
pub use utils::{load_config, AppConfig, MetricsRecorder};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default participation rate (fraction of volume)
pub const DEFAULT_PARTICIPATION_RATE: f64 = 0.10;

/// Maximum participation rate to avoid excessive market impact
pub const MAX_PARTICIPATION_RATE: f64 = 0.25;

/// Default urgency parameter for IS strategy
pub const DEFAULT_URGENCY: f64 = 0.5;

/// Minimum slice size as fraction of total order
pub const MIN_SLICE_FRACTION: f64 = 0.01;

/// Maximum number of child orders per parent order
pub const MAX_CHILD_ORDERS: usize = 1000;

/// Default WebSocket reconnect delay in milliseconds
pub const WS_RECONNECT_DELAY_MS: u64 = 1000;

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::data::{BybitClient, BybitConfig, OrderBook, TimeFrame};
    pub use crate::execution::{
        ExecutionConfig, ExecutionEngine, ExecutionResult, ParentOrder, Side,
    };
    pub use crate::impact::MarketImpactEstimator;
    pub use crate::strategy::{
        AdaptiveStrategy, ExecutionStrategy, TwapStrategy, VwapStrategy,
    };
}
