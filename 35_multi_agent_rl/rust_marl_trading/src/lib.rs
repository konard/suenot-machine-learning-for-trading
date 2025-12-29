//! # Rust MARL Trading
//!
//! A modular Multi-Agent Reinforcement Learning library for cryptocurrency trading on Bybit.
//!
//! ## Overview
//!
//! This library provides tools for simulating multi-agent market environments where
//! multiple trading agents interact, compete, and learn simultaneously.
//!
//! ## Modules
//!
//! - `api` - Bybit API client for fetching market data
//! - `orderbook` - Order book simulation with matching engine
//! - `agents` - Different types of trading agents (trend-following, mean-reversion, etc.)
//! - `environment` - Multi-agent trading environment
//! - `algorithms` - MARL algorithms (Independent Learners, Self-Play, Population-Based)
//! - `utils` - Utility functions and helpers
//!
//! ## Example
//!
//! ```rust,no_run
//! use rust_marl_trading::prelude::*;
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Create a multi-agent market environment
//!     let mut env = MultiAgentEnv::new(MultiAgentConfig {
//!         n_agents: 4,
//!         initial_cash: 100_000.0,
//!         initial_price: 50_000.0,
//!         ..Default::default()
//!     });
//!
//!     // Create different types of agents
//!     let agents: Vec<Box<dyn Agent>> = vec![
//!         Box::new(TrendFollowingAgent::default()),
//!         Box::new(MeanReversionAgent::default()),
//!         Box::new(NoiseTrader::default()),
//!         Box::new(MarketMakerAgent::default()),
//!     ];
//!
//!     // Run simulation
//!     let results = env.run_episode(&agents, 1000)?;
//!     println!("Episode results: {:?}", results.summary());
//!
//!     Ok(())
//! }
//! ```

pub mod agents;
pub mod algorithms;
pub mod api;
pub mod environment;
pub mod orderbook;
pub mod utils;

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::agents::{
        Agent, AgentAction, AgentId, InformedTrader, MarketMakerAgent, MeanReversionAgent,
        NoiseTrader, RLAgent, TrendFollowingAgent,
    };
    pub use crate::algorithms::{
        IndependentLearners, PopulationTrainer, SelfPlayTrainer, TrainingConfig,
    };
    pub use crate::api::{BybitClient, Candle, MarketData};
    pub use crate::environment::{
        EpisodeResult, MultiAgentConfig, MultiAgentEnv, Observation, StepResult,
    };
    pub use crate::orderbook::{Order, OrderBook, OrderSide, OrderType, Trade};
    pub use crate::utils::{compute_sharpe, compute_max_drawdown, compute_win_rate};
}

// Re-export main types at crate root for convenience
pub use agents::{Agent, AgentAction, AgentId};
pub use api::BybitClient;
pub use environment::{MultiAgentConfig, MultiAgentEnv};
pub use orderbook::OrderBook;
