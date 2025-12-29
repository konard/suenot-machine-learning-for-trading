//! Event detection module
//!
//! Detects significant market events that are analogous to earnings announcements:
//! - Volume spikes
//! - Price gaps
//! - Volatility expansion

mod detector;
mod types;

pub use detector::EventDetector;
pub use types::{CryptoEvent, EventType};
