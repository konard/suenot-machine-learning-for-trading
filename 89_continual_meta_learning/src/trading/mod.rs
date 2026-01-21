//! Trading strategy module.
//!
//! This module provides components for trading with CML:
//! - Strategy implementation
//! - Signal generation
//! - Position management

pub mod strategy;
pub mod signals;

pub use strategy::{CMLStrategy, StrategyConfig, Position, TradeAction};
pub use signals::{Signal, SignalGenerator};
