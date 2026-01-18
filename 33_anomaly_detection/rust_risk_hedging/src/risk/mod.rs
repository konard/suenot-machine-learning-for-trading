//! Risk management and hedging module
//!
//! Provides tools for:
//! - Position sizing based on risk
//! - Hedging decisions
//! - Portfolio protection strategies

mod hedging;
mod portfolio;
mod signals;

pub use hedging::*;
pub use portfolio::*;
pub use signals::*;
