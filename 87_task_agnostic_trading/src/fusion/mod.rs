//! Decision fusion for combining multi-task predictions
//!
//! This module provides methods to combine predictions from different tasks
//! into actionable trading decisions with confidence estimates.

mod fusion;

pub use fusion::{DecisionFusion, FusionConfig, FusionMethod, TradingDecision, DecisionConfidence};
