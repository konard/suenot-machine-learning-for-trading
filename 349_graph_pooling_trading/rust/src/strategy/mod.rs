//! Trading strategy module
//!
//! Implements trading strategies based on hierarchical graph pooling.

mod hierarchical;

pub use hierarchical::{GraphPoolingStrategy, Signal, Regime, StrategyConfig};
