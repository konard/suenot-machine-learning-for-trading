//! Trading strategy module
//!
//! Provides signal generation with uncertainty-aware position sizing.

pub mod signal;

pub use signal::{Signal, SignalGenerator, SignalType};
