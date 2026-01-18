//! Trading strategy module
//!
//! Converts predictions into actionable trading signals

mod signals;

pub use signals::{Signal, SignalGenerator, SignalResult};
