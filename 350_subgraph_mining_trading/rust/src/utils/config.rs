//! Configuration handling.

use serde::{Deserialize, Serialize};
use std::path::Path;

/// Application configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Data source configuration
    pub data: DataConfig,
    /// Graph building configuration
    pub graph: GraphConfig,
    /// Mining configuration
    pub mining: MiningConfig,
    /// Trading configuration
    pub trading: TradingConfig,
}

/// Data source configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataConfig {
    /// API base URL
    pub api_url: String,
    /// Symbols to track
    pub symbols: Vec<String>,
    /// Default time frame
    pub timeframe: String,
    /// Cache TTL in seconds
    pub cache_ttl: i64,
}

/// Graph building configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphConfig {
    /// Correlation threshold for edges
    pub correlation_threshold: f64,
    /// Window size for correlation calculation
    pub window_size: usize,
    /// Minimum volume filter
    pub min_volume: f64,
}

/// Mining configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MiningConfig {
    /// Minimum support for patterns
    pub min_support: usize,
    /// Maximum pattern size (nodes)
    pub max_size: usize,
    /// Maximum patterns to find
    pub max_patterns: usize,
}

/// Trading configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingConfig {
    /// Initial capital
    pub initial_capital: f64,
    /// Maximum position size (fraction)
    pub max_position_size: f64,
    /// Stop loss percentage
    pub stop_loss: f64,
    /// Take profit percentage
    pub take_profit: f64,
    /// Trading fee percentage
    pub trading_fee: f64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            data: DataConfig {
                api_url: "https://api.bybit.com".to_string(),
                symbols: vec![
                    "BTCUSDT".to_string(),
                    "ETHUSDT".to_string(),
                    "SOLUSDT".to_string(),
                    "BNBUSDT".to_string(),
                    "XRPUSDT".to_string(),
                    "ADAUSDT".to_string(),
                    "AVAXUSDT".to_string(),
                    "DOTUSDT".to_string(),
                    "MATICUSDT".to_string(),
                    "LINKUSDT".to_string(),
                ],
                timeframe: "1h".to_string(),
                cache_ttl: 300,
            },
            graph: GraphConfig {
                correlation_threshold: 0.7,
                window_size: 24,
                min_volume: 0.0,
            },
            mining: MiningConfig {
                min_support: 2,
                max_size: 6,
                max_patterns: 100,
            },
            trading: TradingConfig {
                initial_capital: 10000.0,
                max_position_size: 0.1,
                stop_loss: 0.05,
                take_profit: 0.10,
                trading_fee: 0.001,
            },
        }
    }
}

impl Config {
    /// Create new default configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Load configuration from file
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let content = std::fs::read_to_string(path)?;
        let config: Config = serde_json::from_str(&content)?;
        Ok(config)
    }

    /// Save configuration to file
    pub fn to_file(&self, path: impl AsRef<Path>) -> Result<(), ConfigError> {
        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }

    /// Load from environment variables (with prefix)
    pub fn from_env(prefix: &str) -> Self {
        let mut config = Self::default();

        // Override with environment variables
        if let Ok(url) = std::env::var(format!("{}_API_URL", prefix)) {
            config.data.api_url = url;
        }

        if let Ok(threshold) = std::env::var(format!("{}_CORRELATION_THRESHOLD", prefix)) {
            if let Ok(v) = threshold.parse() {
                config.graph.correlation_threshold = v;
            }
        }

        if let Ok(capital) = std::env::var(format!("{}_INITIAL_CAPITAL", prefix)) {
            if let Ok(v) = capital.parse() {
                config.trading.initial_capital = v;
            }
        }

        config
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.data.symbols.is_empty() {
            return Err(ConfigError::Validation("No symbols configured".to_string()));
        }

        if self.graph.correlation_threshold <= 0.0 || self.graph.correlation_threshold > 1.0 {
            return Err(ConfigError::Validation(
                "Correlation threshold must be between 0 and 1".to_string(),
            ));
        }

        if self.graph.window_size < 2 {
            return Err(ConfigError::Validation(
                "Window size must be at least 2".to_string(),
            ));
        }

        if self.trading.initial_capital <= 0.0 {
            return Err(ConfigError::Validation(
                "Initial capital must be positive".to_string(),
            ));
        }

        Ok(())
    }
}

/// Configuration errors
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Parse error: {0}")]
    Parse(#[from] serde_json::Error),
    #[error("Validation error: {0}")]
    Validation(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert!(!config.data.symbols.is_empty());
        assert!(config.graph.correlation_threshold > 0.0);
    }

    #[test]
    fn test_config_validation() {
        let config = Config::default();
        assert!(config.validate().is_ok());

        let mut invalid = Config::default();
        invalid.data.symbols.clear();
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn test_config_serialization() {
        let config = Config::default();
        let json = serde_json::to_string(&config).unwrap();
        let parsed: Config = serde_json::from_str(&json).unwrap();
        assert_eq!(config.data.symbols.len(), parsed.data.symbols.len());
    }
}
