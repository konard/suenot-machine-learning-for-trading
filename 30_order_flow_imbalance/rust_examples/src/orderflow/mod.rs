//! # Order Flow Analysis Module
//!
//! Implementations of order flow metrics for market microstructure analysis.
//!
//! ## Metrics
//!
//! - `ofi` - Order Flow Imbalance (Cont et al., 2014)
//! - `vpin` - Volume-Synchronized Probability of Informed Trading
//! - `toxicity` - Flow toxicity indicators
//! - `kyle` - Kyle's Lambda (price impact)

pub mod kyle;
pub mod ofi;
pub mod toxicity;
pub mod vpin;

pub use kyle::KyleLambdaCalculator;
pub use ofi::OrderFlowCalculator;
pub use toxicity::ToxicityIndicator;
pub use vpin::VpinCalculator;
