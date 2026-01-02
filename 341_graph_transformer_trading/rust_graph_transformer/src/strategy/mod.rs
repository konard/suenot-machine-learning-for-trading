//! Trading strategy module
//!
//! Provides signal generation and portfolio construction based on
//! Graph Transformer predictions.

pub mod portfolio;
pub mod signal;

pub use portfolio::PortfolioBuilder;
pub use signal::{Signal, SignalGenerator, SignalType};
