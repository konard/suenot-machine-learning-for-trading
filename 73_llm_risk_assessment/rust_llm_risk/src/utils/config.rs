//! Configuration loading and management.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Main configuration structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// LLM provider configuration.
    pub llm: LLMConfig,
    /// Trading configuration.
    pub trading: TradingConfig,
    /// Data source configuration.
    pub data: DataConfig,
}

/// LLM provider configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMConfig {
    /// Provider type (openai, anthropic, local).
    pub provider: String,
    /// Model name.
    pub model: String,
    /// API key (optional, can be from env).
    pub api_key: Option<String>,
    /// API base URL (for local providers).
    pub base_url: Option<String>,
    /// Temperature for generation.
    pub temperature: f32,
    /// Maximum tokens to generate.
    pub max_tokens: u32,
}

impl Default for LLMConfig {
    fn default() -> Self {
        Self {
            provider: "local".to_string(),
            model: "llama2".to_string(),
            api_key: None,
            base_url: Some("http://localhost:11434".to_string()),
            temperature: 0.3,
            max_tokens: 2000,
        }
    }
}

/// Trading strategy configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingConfig {
    /// Risk threshold for long positions.
    pub risk_threshold_long: f64,
    /// Risk threshold for short positions.
    pub risk_threshold_short: f64,
    /// Maximum position size (0-1).
    pub max_position_size: f64,
    /// Stop loss percentage.
    pub stop_loss_pct: f64,
    /// Take profit percentage.
    pub take_profit_pct: f64,
    /// Allow short positions.
    pub allow_shorts: bool,
    /// Initial capital for backtesting.
    pub initial_capital: f64,
    /// Trading fee percentage.
    pub trading_fee: f64,
}

impl Default for TradingConfig {
    fn default() -> Self {
        Self {
            risk_threshold_long: 4.0,
            risk_threshold_short: 7.0,
            max_position_size: 1.0,
            stop_loss_pct: 0.05,
            take_profit_pct: 0.10,
            allow_shorts: true,
            initial_capital: 10000.0,
            trading_fee: 0.001,
        }
    }
}

/// Data source configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataConfig {
    /// Default exchange (bybit).
    pub exchange: String,
    /// Default symbols to track.
    pub symbols: Vec<String>,
    /// Default interval for price data.
    pub interval: String,
    /// Number of days for historical data.
    pub historical_days: u32,
}

impl Default for DataConfig {
    fn default() -> Self {
        Self {
            exchange: "bybit".to_string(),
            symbols: vec![
                "BTCUSDT".to_string(),
                "ETHUSDT".to_string(),
                "SOLUSDT".to_string(),
            ],
            interval: "60".to_string(),  // 1 hour
            historical_days: 30,
        }
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            llm: LLMConfig::default(),
            trading: TradingConfig::default(),
            data: DataConfig::default(),
        }
    }
}

impl Config {
    /// Create a new configuration from a TOML file.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(&path)
            .with_context(|| format!("Failed to read config file: {:?}", path.as_ref()))?;

        let config: Config = toml::from_str(&content)
            .context("Failed to parse config file")?;

        Ok(config)
    }

    /// Save configuration to a TOML file.
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let content = toml::to_string_pretty(self)
            .context("Failed to serialize config")?;

        std::fs::write(&path, content)
            .with_context(|| format!("Failed to write config file: {:?}", path.as_ref()))?;

        Ok(())
    }

    /// Create a sample configuration file.
    pub fn create_sample_config<P: AsRef<Path>>(path: P) -> Result<()> {
        let config = Config::default();
        config.save_to_file(path)
    }

    /// Get API key from config or environment variable.
    pub fn get_api_key(&self) -> Option<String> {
        self.llm.api_key.clone().or_else(|| {
            match self.llm.provider.as_str() {
                "openai" => std::env::var("OPENAI_API_KEY").ok(),
                "anthropic" => std::env::var("ANTHROPIC_API_KEY").ok(),
                _ => None,
            }
        })
    }
}

/// Load configuration from file or create default.
pub fn load_config<P: AsRef<Path>>(path: P) -> Result<Config> {
    if path.as_ref().exists() {
        Config::from_file(path)
    } else {
        Ok(Config::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.llm.provider, "local");
        assert_eq!(config.trading.risk_threshold_long, 4.0);
        assert_eq!(config.data.symbols.len(), 3);
    }

    #[test]
    fn test_config_from_file() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, r#"
[llm]
provider = "openai"
model = "gpt-4"
temperature = 0.5
max_tokens = 1000

[trading]
risk_threshold_long = 3.0
risk_threshold_short = 8.0
max_position_size = 0.5
stop_loss_pct = 0.03
take_profit_pct = 0.06
allow_shorts = false
initial_capital = 5000.0
trading_fee = 0.002

[data]
exchange = "bybit"
symbols = ["BTCUSDT"]
interval = "15"
historical_days = 7
        "#).unwrap();

        let config = Config::from_file(temp_file.path()).unwrap();
        assert_eq!(config.llm.provider, "openai");
        assert_eq!(config.trading.risk_threshold_long, 3.0);
        assert!(!config.trading.allow_shorts);
    }
}
