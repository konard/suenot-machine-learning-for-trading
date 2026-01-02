//! Subgraph mining module for discovering patterns in financial networks.
//!
//! This module provides:
//! - gSpan algorithm implementation for frequent subgraph mining
//! - Pattern representation and manipulation
//! - Pattern matching for signal generation

mod gspan;
mod pattern;
mod matcher;

pub use gspan::GSpanMiner;
pub use pattern::{Pattern, PatternType, PatternEdge, PatternNode, DFSCode};
pub use matcher::{PatternMatcher, PatternMatch, MatchResult};
