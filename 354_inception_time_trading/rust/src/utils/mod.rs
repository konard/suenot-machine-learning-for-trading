//! Utility module
//!
//! This module provides:
//! - Configuration management
//! - Logging setup

mod config;
mod logging;

pub use config::Config;
pub use logging::setup_logging;
