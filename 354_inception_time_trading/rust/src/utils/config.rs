//! Configuration management
//!
//! This module handles loading and managing configuration.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Data configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataConfig {
    pub symbol: String,
    pub interval: String,
    pub days_history: u32,
    pub window_size: usize,
    pub stride: usize,
    pub features: Vec<String>,
    pub prediction_horizon: usize,
    pub threshold_percent: f64,
}

impl Default for DataConfig {
    fn default() -> Self {
        Self {
            symbol: "BTCUSDT".to_string(),
            interval: "15".to_string(),
            days_history: 90,
            window_size: 64,
            stride: 1,
            features: vec![
                "close".to_string(),
                "volume".to_string(),
                "rsi".to_string(),
                "macd".to_string(),
                "bb_upper".to_string(),
                "bb_lower".to_string(),
            ],
            prediction_horizon: 4,
            threshold_percent: 0.5,
        }
    }
}

/// Model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub num_filters: i64,
    pub depth: i64,
    pub kernel_sizes: Vec<i64>,
    pub bottleneck_size: i64,
    pub residual_interval: i64,
    pub num_classes: i64,
    pub ensemble_size: usize,
    pub dropout: f64,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            num_filters: 32,
            depth: 6,
            kernel_sizes: vec![10, 20, 40],
            bottleneck_size: 32,
            residual_interval: 3,
            num_classes: 3,
            ensemble_size: 5,
            dropout: 0.2,
        }
    }
}

/// Training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub batch_size: usize,
    pub epochs: u32,
    pub learning_rate: f64,
    pub weight_decay: f64,
    pub lr_patience: usize,
    pub lr_decay_factor: f64,
    pub early_stopping_patience: usize,
    pub min_delta: f64,
    pub train_ratio: f64,
    pub val_ratio: f64,
    pub test_ratio: f64,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            batch_size: 64,
            epochs: 1500,
            learning_rate: 0.001,
            weight_decay: 0.0001,
            lr_patience: 50,
            lr_decay_factor: 0.5,
            early_stopping_patience: 100,
            min_delta: 0.0001,
            train_ratio: 0.7,
            val_ratio: 0.15,
            test_ratio: 0.15,
        }
    }
}

/// Strategy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyConfig {
    pub min_confidence: f64,
    pub max_position_size: f64,
    pub risk_per_trade: f64,
    pub max_drawdown: f64,
    pub daily_loss_limit: f64,
    pub atr_multiplier: f64,
}

impl Default for StrategyConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.6,
            max_position_size: 0.2,
            risk_per_trade: 0.02,
            max_drawdown: 0.15,
            daily_loss_limit: 0.03,
            atr_multiplier: 2.0,
        }
    }
}

/// Backtest configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestConfig {
    pub initial_capital: f64,
    pub commission_rate: f64,
    pub slippage_rate: f64,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            initial_capital: 100000.0,
            commission_rate: 0.001,
            slippage_rate: 0.0005,
        }
    }
}

/// Logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    pub level: String,
    pub file: Option<String>,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "info".to_string(),
            file: None,
        }
    }
}

/// Main configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Config {
    pub data: DataConfig,
    pub model: ModelConfig,
    pub training: TrainingConfig,
    pub strategy: StrategyConfig,
    pub backtest: BacktestConfig,
    pub logging: LoggingConfig,
}

impl Config {
    /// Load configuration from file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Config = toml::from_str(&content)?;
        Ok(config)
    }

    /// Load configuration from file or use default
    pub fn load_or_default<P: AsRef<Path>>(path: P) -> Self {
        Self::load(path).unwrap_or_default()
    }

    /// Save configuration to file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let content = toml::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }

    /// Create default configuration file
    pub fn create_default<P: AsRef<Path>>(path: P) -> Result<()> {
        let config = Config::default();
        config.save(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = Config::default();
        assert_eq!(config.data.symbol, "BTCUSDT");
        assert_eq!(config.model.num_filters, 32);
        assert_eq!(config.training.epochs, 1500);
    }

    #[test]
    fn test_config_serialization() {
        let config = Config::default();
        let toml_str = toml::to_string(&config).unwrap();
        let parsed: Config = toml::from_str(&toml_str).unwrap();
        assert_eq!(parsed.data.symbol, config.data.symbol);
    }
}
