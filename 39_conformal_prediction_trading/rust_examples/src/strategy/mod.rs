//! Trading strategy modules
//!
//! - `trading` - Conformal prediction-based trading strategy
//! - `sizing` - Position sizing based on prediction intervals

pub mod sizing;
pub mod trading;

pub use sizing::PositionSizer;
pub use trading::{ConformalTradingStrategy, TradingSignal};
