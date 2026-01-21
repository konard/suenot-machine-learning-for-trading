//! Trading strategy module
//!
//! This module provides:
//! - Market regime classifier
//! - Trading signal generation
//! - Position management

mod classifier;
mod signals;
mod execution;

pub use classifier::{RegimeClassifier, ClassificationResult};
pub use signals::{TradingSignal, SignalGenerator, SignalConfig, SignalType};
pub use execution::{Position, PositionManager, ExecutionConfig, PositionSide, OrderType};
