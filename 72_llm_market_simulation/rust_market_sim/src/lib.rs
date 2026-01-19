//! LLM Market Simulation Library
//!
//! A multi-agent market simulation framework for testing trading strategies
//! with different agent types (value investors, momentum traders, market makers).
//!
//! # Features
//! - Order book with price-time priority matching
//! - Multiple agent types with configurable parameters
//! - Bybit API integration for real market data
//! - Performance metrics calculation
//! - Bubble detection algorithms
//!
//! # Example
//! ```no_run
//! use llm_market_sim::{
//!     simulation::SimulationEngine,
//!     agents::{ValueInvestor, MomentumTrader, MarketMaker},
//! };
//!
//! #[tokio::main]
//! async fn main() {
//!     let mut engine = SimulationEngine::new(100.0, 100.0, 0.02);
//!
//!     engine.add_agent(Box::new(ValueInvestor::new(
//!         "value_1".to_string(),
//!         100000.0,
//!         100,
//!         100.0,
//!     )));
//!
//!     let result = engine.run(500);
//!     println!("Final price: {}", result.final_price);
//! }
//! ```

pub mod market;
pub mod agents;
pub mod simulation;
pub mod data;

pub use market::{OrderBook, Order, OrderType, Side, OrderResult};
pub use agents::{Agent, AgentDecision, ValueInvestor, MomentumTrader, MarketMaker};
pub use simulation::{SimulationEngine, SimulationResult};
pub use data::BybitClient;
