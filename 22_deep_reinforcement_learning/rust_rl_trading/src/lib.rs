//! # Rust RL Trading
//!
//! A modular Deep Reinforcement Learning library for cryptocurrency trading on Bybit.
//!
//! ## Modules
//!
//! - `agent` - RL agents (Q-Learning, Deep Q-Network)
//! - `environment` - Trading environment implementation
//! - `data` - Bybit API client and data management
//! - `utils` - Utility functions and helpers

pub mod agent;
pub mod data;
pub mod environment;
pub mod utils;

pub use agent::{Agent, DQNAgent, QLearningAgent};
pub use data::{BybitClient, Candle, MarketData};
pub use environment::{TradingAction, TradingEnvironment, TradingState};
