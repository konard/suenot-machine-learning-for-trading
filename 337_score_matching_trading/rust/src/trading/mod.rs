//! # Trading Module
//!
//! Trading signal generation based on score matching.

mod signals;

pub use signals::{ScoreMatchingTrader, TradingSignal, TraderConfig};
