//! Trading strategy module
//!
//! Converts anomaly detection signals into trading decisions.

mod signals;
mod position;

pub use signals::*;
pub use position::*;
