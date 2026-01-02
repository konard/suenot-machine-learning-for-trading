//! Trading strategy module

mod signal;
mod position;

pub use signal::{Signal, SignalType, SignalGenerator};
pub use position::{Position, PositionManager, PositionSide};
