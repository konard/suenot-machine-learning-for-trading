//! # Meta-RL (RL^2) for Trading
//!
//! This crate implements Meta-Reinforcement Learning using the RL^2 framework
//! for algorithmic trading. The agent uses a GRU-based recurrent network that
//! learns a learning algorithm within its hidden state, enabling rapid
//! adaptation to new market environments.
//!
//! ## Features
//!
//! - GRU-based Meta-RL agent (RL^2 framework)
//! - Trading environment with realistic market simulation
//! - PPO meta-training across multiple environments
//! - Bybit API integration for cryptocurrency data
//! - Backtesting framework for strategy evaluation
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use meta_rl_trading::{MetaRLAgent, TradingEnvironment, PPOTrainer, BybitClient};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Create agent and environment
//!     let agent = MetaRLAgent::new(14, 3, 128);
//!     let client = BybitClient::new();
//!     let data = client.fetch_klines("BTCUSDT", "60", 1000).await?;
//!
//!     Ok(())
//! }
//! ```

pub mod agent;
pub mod env;
pub mod trainer;
pub mod data;
pub mod backtest;

pub use agent::meta_rl::MetaRLAgent;
pub use env::trading_env::TradingEnvironment;
pub use trainer::ppo::PPOTrainer;
pub use data::bybit::BybitClient;
pub use data::features::FeatureGenerator;
pub use backtest::engine::BacktestEngine;

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::agent::meta_rl::MetaRLAgent;
    pub use crate::env::trading_env::TradingEnvironment;
    pub use crate::trainer::ppo::PPOTrainer;
    pub use crate::data::bybit::BybitClient;
    pub use crate::data::features::FeatureGenerator;
    pub use crate::backtest::engine::BacktestEngine;
}

/// Error types for the crate
#[derive(thiserror::Error, Debug)]
pub enum MetaRLError {
    #[error("Agent error: {0}")]
    AgentError(String),

    #[error("Environment error: {0}")]
    EnvironmentError(String),

    #[error("Training error: {0}")]
    TrainingError(String),

    #[error("Data error: {0}")]
    DataError(String),

    #[error("API error: {0}")]
    ApiError(String),

    #[error("Backtest error: {0}")]
    BacktestError(String),

    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
}

pub type Result<T> = std::result::Result<T, MetaRLError>;
