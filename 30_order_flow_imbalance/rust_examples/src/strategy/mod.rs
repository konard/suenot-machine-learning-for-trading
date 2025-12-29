//! # Trading Strategy Module
//!
//! Signal generation and position management.

pub mod position;
pub mod signal;

pub use position::PositionManager;
pub use signal::SignalGenerator;
