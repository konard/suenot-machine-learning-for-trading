//! Neural network models for SE-based trading
//!
//! This module contains the core Squeeze-and-Excitation block implementation
//! and trading-specific model wrappers.

pub mod se_block;
pub mod se_trading;
pub mod activation;

pub use se_block::SEBlock;
pub use se_trading::SETradingModel;
