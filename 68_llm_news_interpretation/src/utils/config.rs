//! Configuration utilities

use serde::{Deserialize, Serialize};
use std::path::Path;

/// Application configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    /// LLM settings
    pub llm: LlmSettings,
    /// Trading settings
    pub trading: TradingSettings,
    /// News source settings
    pub news: NewsSettings,
    /// Logging settings
    pub logging: LoggingSettings,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            llm: LlmSettings::default(),
            trading: TradingSettings::default(),
            news: NewsSettings::default(),
            logging: LoggingSettings::default(),
        }
    }
}

/// LLM-related settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmSettings {
    /// Provider (openai, anthropic, local)
    pub provider: String,
    /// Model name
    pub model: String,
    /// API key (can be loaded from env)
    pub api_key: Option<String>,
    /// API key environment variable name
    pub api_key_env: String,
    /// Maximum tokens for response
    pub max_tokens: usize,
    /// Temperature for generation
    pub temperature: f64,
    /// Enable response caching
    pub enable_cache: bool,
    /// Request timeout in milliseconds
    pub timeout_ms: u64,
}

impl Default for LlmSettings {
    fn default() -> Self {
        Self {
            provider: "openai".to_string(),
            model: "gpt-4".to_string(),
            api_key: None,
            api_key_env: "OPENAI_API_KEY".to_string(),
            max_tokens: 1000,
            temperature: 0.3,
            enable_cache: true,
            timeout_ms: 30000,
        }
    }
}

impl LlmSettings {
    /// Get API key from config or environment
    pub fn get_api_key(&self) -> Option<String> {
        self.api_key.clone().or_else(|| {
            std::env::var(&self.api_key_env).ok()
        })
    }
}

/// Trading-related settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingSettings {
    /// Exchange to use
    pub exchange: String,
    /// Use testnet
    pub testnet: bool,
    /// Symbols to trade
    pub symbols: Vec<String>,
    /// Maximum position size (fraction of portfolio)
    pub max_position_size: f64,
    /// Risk per trade (fraction)
    pub risk_per_trade: f64,
    /// Enable paper trading (no real orders)
    pub paper_trading: bool,
    /// Stop loss percentage
    pub stop_loss_pct: f64,
    /// Take profit percentage
    pub take_profit_pct: f64,
}

impl Default for TradingSettings {
    fn default() -> Self {
        Self {
            exchange: "bybit".to_string(),
            testnet: true,
            symbols: vec!["BTCUSDT".to_string(), "ETHUSDT".to_string()],
            max_position_size: 0.1,
            risk_per_trade: 0.02,
            paper_trading: true,
            stop_loss_pct: 0.05,
            take_profit_pct: 0.10,
        }
    }
}

/// News source settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewsSettings {
    /// Enabled news sources
    pub sources: Vec<String>,
    /// Minimum relevance score
    pub min_relevance: f64,
    /// Maximum news age in seconds
    pub max_age_secs: i64,
    /// Fetch interval in seconds
    pub fetch_interval_secs: u64,
    /// Twitter API credentials (if using)
    pub twitter_bearer_token: Option<String>,
    /// RSS feed URLs
    pub rss_feeds: Vec<String>,
}

impl Default for NewsSettings {
    fn default() -> Self {
        Self {
            sources: vec![
                "coindesk".to_string(),
                "cointelegraph".to_string(),
                "twitter".to_string(),
            ],
            min_relevance: 0.5,
            max_age_secs: 3600,
            fetch_interval_secs: 60,
            twitter_bearer_token: None,
            rss_feeds: vec![
                "https://www.coindesk.com/arc/outboundfeeds/rss/".to_string(),
                "https://cointelegraph.com/rss".to_string(),
            ],
        }
    }
}

/// Logging settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingSettings {
    /// Log level (trace, debug, info, warn, error)
    pub level: String,
    /// Log to file
    pub file_enabled: bool,
    /// Log file path
    pub file_path: String,
    /// Enable JSON format
    pub json_format: bool,
}

impl Default for LoggingSettings {
    fn default() -> Self {
        Self {
            level: "info".to_string(),
            file_enabled: false,
            file_path: "logs/app.log".to_string(),
            json_format: false,
        }
    }
}

/// Load configuration from file
pub fn load_config<P: AsRef<Path>>(path: P) -> Result<AppConfig, ConfigError> {
    let content = std::fs::read_to_string(path.as_ref())
        .map_err(|e| ConfigError::FileError(e.to_string()))?;

    let ext = path.as_ref()
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");

    match ext {
        "json" => serde_json::from_str(&content)
            .map_err(|e| ConfigError::ParseError(e.to_string())),
        "toml" => toml::from_str(&content)
            .map_err(|e| ConfigError::ParseError(e.to_string())),
        "yaml" | "yml" => serde_yaml::from_str(&content)
            .map_err(|e| ConfigError::ParseError(e.to_string())),
        _ => Err(ConfigError::UnsupportedFormat(ext.to_string())),
    }
}

/// Save configuration to file
pub fn save_config<P: AsRef<Path>>(config: &AppConfig, path: P) -> Result<(), ConfigError> {
    let ext = path.as_ref()
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");

    let content = match ext {
        "json" => serde_json::to_string_pretty(config)
            .map_err(|e| ConfigError::SerializeError(e.to_string()))?,
        "toml" => toml::to_string_pretty(config)
            .map_err(|e| ConfigError::SerializeError(e.to_string()))?,
        "yaml" | "yml" => serde_yaml::to_string(config)
            .map_err(|e| ConfigError::SerializeError(e.to_string()))?,
        _ => return Err(ConfigError::UnsupportedFormat(ext.to_string())),
    };

    std::fs::write(path, content)
        .map_err(|e| ConfigError::FileError(e.to_string()))
}

/// Configuration error
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    /// File I/O error
    #[error("File error: {0}")]
    FileError(String),
    /// Parse error
    #[error("Parse error: {0}")]
    ParseError(String),
    /// Serialization error
    #[error("Serialize error: {0}")]
    SerializeError(String),
    /// Unsupported format
    #[error("Unsupported config format: {0}")]
    UnsupportedFormat(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = AppConfig::default();
        assert_eq!(config.llm.provider, "openai");
        assert_eq!(config.trading.testnet, true);
    }

    #[test]
    fn test_api_key_from_env() {
        let settings = LlmSettings {
            api_key_env: "TEST_API_KEY_12345".to_string(),
            ..Default::default()
        };

        // Should return None if env var not set
        assert!(settings.get_api_key().is_none());
    }

    #[test]
    fn test_api_key_from_config() {
        let settings = LlmSettings {
            api_key: Some("test-key".to_string()),
            ..Default::default()
        };

        assert_eq!(settings.get_api_key(), Some("test-key".to_string()));
    }
}
