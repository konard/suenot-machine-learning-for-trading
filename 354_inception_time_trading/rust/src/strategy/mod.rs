//! Trading strategy module
//!
//! This module provides:
//! - Signal generation from model predictions
//! - Position management
//! - Risk management

mod position;
mod risk;
mod signals;

pub use position::PositionManager;
pub use risk::RiskManager;
pub use signals::{Signal, SignalGenerator, TradingStrategy};
