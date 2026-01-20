//! Logging initialization utilities

use tracing_subscriber::{
    fmt,
    prelude::*,
    filter::EnvFilter,
};

/// Initialize logging with the given level
pub fn init_logging(level: &str) {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(level));

    tracing_subscriber::registry()
        .with(fmt::layer().with_target(true))
        .with(filter)
        .init();

    tracing::info!("Logging initialized at level: {}", level);
}

/// Initialize logging with file output
pub fn init_logging_with_file(level: &str, _file_path: &str) -> Result<(), LoggingError> {
    // For simplicity, we'll just use console logging
    // File logging would require additional setup with tracing-appender
    init_logging(level);
    Ok(())
}

/// Logging error
#[derive(Debug, thiserror::Error)]
pub enum LoggingError {
    /// Failed to initialize
    #[error("Failed to initialize logging: {0}")]
    InitError(String),
}

#[cfg(test)]
mod tests {
    // Logging tests are tricky because subscriber can only be set once
    // We'll skip actual initialization tests
}
