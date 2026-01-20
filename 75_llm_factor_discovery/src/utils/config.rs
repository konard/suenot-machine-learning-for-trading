//! Configuration management
//!
//! Handles loading and validation of application configuration.

use serde::{Deserialize, Serialize};
use std::path::Path;
use thiserror::Error;

/// Configuration errors
#[derive(Error, Debug)]
pub enum ConfigError {
    #[error("Failed to read config file: {0}")]
    ReadError(#[from] std::io::Error),

    #[error("Failed to parse config: {0}")]
    ParseError(String),

    #[error("Invalid configuration: {0}")]
    ValidationError(String),

    #[error("Missing required field: {0}")]
    MissingField(String),
}

/// Application configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    /// LLM configuration
    #[serde(default)]
    pub llm: LlmSettings,

    /// Data source configuration
    #[serde(default)]
    pub data: DataSettings,

    /// Strategy configuration
    #[serde(default)]
    pub strategy: StrategySettings,

    /// Logging configuration
    #[serde(default)]
    pub logging: LoggingSettings,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            llm: LlmSettings::default(),
            data: DataSettings::default(),
            strategy: StrategySettings::default(),
            logging: LoggingSettings::default(),
        }
    }
}

/// LLM provider settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmSettings {
    /// Provider (openai, anthropic, local)
    #[serde(default = "default_provider")]
    pub provider: String,

    /// Model name
    #[serde(default = "default_model")]
    pub model: String,

    /// Maximum tokens
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,

    /// Temperature
    #[serde(default = "default_temperature")]
    pub temperature: f32,

    /// API key environment variable name
    #[serde(default = "default_api_key_env")]
    pub api_key_env: String,
}

fn default_provider() -> String {
    "openai".to_string()
}

fn default_model() -> String {
    "gpt-4".to_string()
}

fn default_max_tokens() -> u32 {
    2000
}

fn default_temperature() -> f32 {
    0.7
}

fn default_api_key_env() -> String {
    "OPENAI_API_KEY".to_string()
}

impl Default for LlmSettings {
    fn default() -> Self {
        Self {
            provider: default_provider(),
            model: default_model(),
            max_tokens: default_max_tokens(),
            temperature: default_temperature(),
            api_key_env: default_api_key_env(),
        }
    }
}

/// Data source settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSettings {
    /// Exchange
    #[serde(default = "default_exchange")]
    pub exchange: String,

    /// Use testnet
    #[serde(default)]
    pub testnet: bool,

    /// Default timeframe
    #[serde(default = "default_timeframe")]
    pub timeframe: String,

    /// Universe of symbols
    #[serde(default = "default_universe")]
    pub universe: Vec<String>,

    /// Data cache directory
    #[serde(default = "default_cache_dir")]
    pub cache_dir: String,
}

fn default_exchange() -> String {
    "bybit".to_string()
}

fn default_timeframe() -> String {
    "1d".to_string()
}

fn default_universe() -> Vec<String> {
    vec![
        "BTCUSDT".to_string(),
        "ETHUSDT".to_string(),
        "BNBUSDT".to_string(),
        "SOLUSDT".to_string(),
        "XRPUSDT".to_string(),
    ]
}

fn default_cache_dir() -> String {
    ".cache".to_string()
}

impl Default for DataSettings {
    fn default() -> Self {
        Self {
            exchange: default_exchange(),
            testnet: false,
            timeframe: default_timeframe(),
            universe: default_universe(),
            cache_dir: default_cache_dir(),
        }
    }
}

/// Strategy settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategySettings {
    /// Minimum IC threshold
    #[serde(default = "default_ic_threshold")]
    pub min_ic: f64,

    /// Minimum IC_IR threshold
    #[serde(default = "default_ir_threshold")]
    pub min_ic_ir: f64,

    /// Maximum position size
    #[serde(default = "default_max_position")]
    pub max_position: f64,

    /// Rebalance frequency (days)
    #[serde(default = "default_rebalance_days")]
    pub rebalance_days: u32,

    /// Include transaction costs
    #[serde(default)]
    pub include_costs: bool,

    /// Transaction cost (bps)
    #[serde(default = "default_tx_cost")]
    pub transaction_cost_bps: f64,
}

fn default_ic_threshold() -> f64 {
    0.03
}

fn default_ir_threshold() -> f64 {
    0.5
}

fn default_max_position() -> f64 {
    0.1
}

fn default_rebalance_days() -> u32 {
    1
}

fn default_tx_cost() -> f64 {
    10.0
}

impl Default for StrategySettings {
    fn default() -> Self {
        Self {
            min_ic: default_ic_threshold(),
            min_ic_ir: default_ir_threshold(),
            max_position: default_max_position(),
            rebalance_days: default_rebalance_days(),
            include_costs: false,
            transaction_cost_bps: default_tx_cost(),
        }
    }
}

/// Logging settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingSettings {
    /// Log level
    #[serde(default = "default_log_level")]
    pub level: String,

    /// Log format (json or text)
    #[serde(default = "default_log_format")]
    pub format: String,

    /// Output file (optional)
    pub file: Option<String>,
}

fn default_log_level() -> String {
    "info".to_string()
}

fn default_log_format() -> String {
    "text".to_string()
}

impl Default for LoggingSettings {
    fn default() -> Self {
        Self {
            level: default_log_level(),
            format: default_log_format(),
            file: None,
        }
    }
}

/// Load configuration from file
pub fn load_config<P: AsRef<Path>>(path: P) -> Result<AppConfig, ConfigError> {
    let content = std::fs::read_to_string(path)?;

    // Detect format from extension or content
    if content.trim().starts_with('{') {
        // JSON
        serde_json::from_str(&content)
            .map_err(|e| ConfigError::ParseError(e.to_string()))
    } else if content.contains(':') && !content.contains('=') {
        // YAML
        serde_yaml::from_str(&content)
            .map_err(|e| ConfigError::ParseError(e.to_string()))
    } else {
        // TOML
        toml::from_str(&content)
            .map_err(|e| ConfigError::ParseError(e.to_string()))
    }
}

/// Load configuration from environment variables
pub fn load_config_from_env() -> AppConfig {
    let mut config = AppConfig::default();

    // Override with environment variables
    if let Ok(provider) = std::env::var("LLM_PROVIDER") {
        config.llm.provider = provider;
    }

    if let Ok(model) = std::env::var("LLM_MODEL") {
        config.llm.model = model;
    }

    if let Ok(exchange) = std::env::var("EXCHANGE") {
        config.data.exchange = exchange;
    }

    if let Ok(testnet) = std::env::var("USE_TESTNET") {
        config.data.testnet = testnet.parse().unwrap_or(false);
    }

    config
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = AppConfig::default();
        assert_eq!(config.llm.provider, "openai");
        assert_eq!(config.data.exchange, "bybit");
        assert!(!config.data.testnet);
    }

    #[test]
    fn test_config_serialization() {
        let config = AppConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let parsed: AppConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.llm.provider, config.llm.provider);
    }
}
