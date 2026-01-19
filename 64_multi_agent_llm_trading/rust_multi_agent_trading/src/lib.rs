//! Multi-Agent LLM Trading System
//!
//! A high-performance framework for building multi-agent trading systems
//! using Large Language Models.
//!
//! # Example
//!
//! ```rust,no_run
//! use multi_agent_trading::{
//!     agents::{TechnicalAgent, BullAgent, BearAgent, TraderAgent, Agent},
//!     data::create_mock_data,
//! };
//!
//! #[tokio::main]
//! async fn main() {
//!     let data = create_mock_data("DEMO", 252, 100.0);
//!
//!     let tech = TechnicalAgent::new("Tech-1");
//!     let analysis = tech.analyze("DEMO", &data, None).await.unwrap();
//!
//!     println!("Signal: {:?}", analysis.signal);
//! }
//! ```

pub mod agents;
pub mod backtest;
pub mod communication;
pub mod data;
pub mod error;

pub use agents::{Agent, Analysis, Signal};
pub use backtest::{BacktestResult, MultiAgentBacktester};
pub use communication::{Debate, Message, RoundTable};
pub use data::MarketData;
pub use error::TradingError;
