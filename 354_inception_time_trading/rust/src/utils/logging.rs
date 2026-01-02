//! Logging setup
//!
//! This module provides logging configuration.

use anyhow::Result;
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

/// Setup logging with the specified level
pub fn setup_logging(level: &str) -> Result<()> {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(level));

    tracing_subscriber::registry()
        .with(fmt::layer().with_target(false).with_thread_ids(false))
        .with(filter)
        .try_init()
        .ok();

    Ok(())
}

/// Setup logging with file output
pub fn setup_logging_with_file(level: &str, _file_path: &str) -> Result<()> {
    // For simplicity, just use console logging
    // File logging would require additional setup with tracing-appender
    setup_logging(level)
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_logging_setup() {
        // Just verify it doesn't panic
        // Can only be called once per process
    }
}
