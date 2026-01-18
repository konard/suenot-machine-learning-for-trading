//! Configuration management
//!
//! Handles loading and saving strategy configuration

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// Main configuration structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Strategy name
    pub name: String,
    /// Description
    pub description: String,
    /// Cryptocurrency universe
    pub universe: Vec<String>,
    /// Model configuration
    pub model: ModelConfig,
    /// Portfolio configuration
    pub portfolio: PortfolioConfig,
    /// Trading configuration
    pub trading: TradingConfig,
}

/// Model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Input dimension (number of features)
    pub input_dim: usize,
    /// Model hidden dimension
    pub d_model: usize,
    /// Number of attention heads
    pub n_heads: usize,
    /// Number of HOT layers
    pub n_layers: usize,
    /// CP decomposition rank
    pub rank: usize,
    /// Lookback period
    pub lookback: usize,
}

/// Portfolio configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioConfig {
    /// Initial capital
    pub initial_capital: f64,
    /// Maximum position size
    pub max_position: f64,
    /// Maximum total exposure
    pub max_exposure: f64,
    /// Target volatility
    pub target_volatility: f64,
}

/// Trading configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingConfig {
    /// Minimum confidence to trade
    pub min_confidence: f64,
    /// Rebalance period (days)
    pub rebalance_period: u32,
    /// Trading fee
    pub commission: f64,
    /// Slippage estimate
    pub slippage: f64,
    /// Stop loss percentage
    pub stop_loss: f64,
    /// Take profit percentage
    pub take_profit: f64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            name: "HOT Crypto Strategy".to_string(),
            description: "Higher Order Transformer for cryptocurrency prediction".to_string(),
            universe: vec![
                "BTCUSDT".to_string(),
                "ETHUSDT".to_string(),
                "SOLUSDT".to_string(),
                "BNBUSDT".to_string(),
                "XRPUSDT".to_string(),
            ],
            model: ModelConfig::default(),
            portfolio: PortfolioConfig::default(),
            trading: TradingConfig::default(),
        }
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            input_dim: 7,
            d_model: 128,
            n_heads: 4,
            n_layers: 3,
            rank: 16,
            lookback: 60,
        }
    }
}

impl Default for PortfolioConfig {
    fn default() -> Self {
        Self {
            initial_capital: 10000.0,
            max_position: 0.25,
            max_exposure: 0.80,
            target_volatility: 0.30,
        }
    }
}

impl Default for TradingConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.60,
            rebalance_period: 1,
            commission: 0.001,
            slippage: 0.0005,
            stop_loss: 0.03,
            take_profit: 0.05,
        }
    }
}

impl Config {
    /// Load configuration from a JSON file
    pub fn from_file<P: AsRef<Path>>(path: P) -> anyhow::Result<Self> {
        let content = fs::read_to_string(path)?;
        let config: Config = serde_json::from_str(&content)?;
        Ok(config)
    }

    /// Save configuration to a JSON file
    pub fn to_file<P: AsRef<Path>>(&self, path: P) -> anyhow::Result<()> {
        let content = serde_json::to_string_pretty(self)?;
        fs::write(path, content)?;
        Ok(())
    }

    /// Create a conservative configuration (lower risk)
    pub fn conservative() -> Self {
        Self {
            name: "HOT Crypto Conservative".to_string(),
            description: "Lower risk configuration".to_string(),
            portfolio: PortfolioConfig {
                max_position: 0.15,
                max_exposure: 0.50,
                target_volatility: 0.20,
                ..Default::default()
            },
            trading: TradingConfig {
                min_confidence: 0.70,
                stop_loss: 0.02,
                take_profit: 0.03,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    /// Create an aggressive configuration (higher risk)
    pub fn aggressive() -> Self {
        Self {
            name: "HOT Crypto Aggressive".to_string(),
            description: "Higher risk configuration".to_string(),
            portfolio: PortfolioConfig {
                max_position: 0.35,
                max_exposure: 0.90,
                target_volatility: 0.40,
                ..Default::default()
            },
            trading: TradingConfig {
                min_confidence: 0.55,
                stop_loss: 0.05,
                take_profit: 0.08,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.universe.is_empty() {
            return Err("Universe cannot be empty".to_string());
        }

        if self.model.d_model % self.model.n_heads != 0 {
            return Err("d_model must be divisible by n_heads".to_string());
        }

        if self.portfolio.max_position > self.portfolio.max_exposure {
            return Err("max_position cannot exceed max_exposure".to_string());
        }

        if self.trading.min_confidence < 0.0 || self.trading.min_confidence > 1.0 {
            return Err("min_confidence must be between 0 and 1".to_string());
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_serialization() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("config.json");

        let config = Config::default();
        config.to_file(&path).unwrap();

        let loaded = Config::from_file(&path).unwrap();
        assert_eq!(config.name, loaded.name);
    }

    #[test]
    fn test_conservative_config() {
        let config = Config::conservative();
        assert!(config.validate().is_ok());
        assert!(config.trading.min_confidence > 0.65);
    }

    #[test]
    fn test_aggressive_config() {
        let config = Config::aggressive();
        assert!(config.validate().is_ok());
        assert!(config.portfolio.max_exposure > 0.85);
    }
}
