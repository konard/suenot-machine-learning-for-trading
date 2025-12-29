//! Utility module with helper functions
//!
//! This module provides:
//! - Configuration handling
//! - Checkpoint save/load utilities
//! - Visualization helpers

mod config;
mod checkpoint;

pub use config::Config;
pub use checkpoint::{save_checkpoint, load_checkpoint};
