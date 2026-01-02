//! Trading strategy module
//!
//! Signal generation and portfolio management using GAT.

mod portfolio;
mod signals;

pub use portfolio::Portfolio;
pub use signals::{SignalGenerator, TradingStrategy};
