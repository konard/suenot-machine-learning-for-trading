//! Configuration handling.

use serde::{Deserialize, Serialize};
use std::path::Path;

/// Main configuration structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Data configuration
    pub data: DataConfig,
    /// Graph configuration
    pub graph: GraphConfig,
    /// Model configuration
    pub model: ModelConfig,
    /// Strategy configuration
    pub strategy: StrategyConfig,
    /// Backtest configuration
    pub backtest: BacktestConfig,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            data: DataConfig::default(),
            graph: GraphConfig::default(),
            model: ModelConfig::default(),
            strategy: StrategyConfig::default(),
            backtest: BacktestConfig::default(),
        }
    }
}

impl Config {
    /// Load configuration from TOML file.
    pub fn from_file(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Config = toml::from_str(&content)?;
        Ok(config)
    }

    /// Save configuration to TOML file.
    pub fn to_file(&self, path: impl AsRef<Path>) -> anyhow::Result<()> {
        let content = toml::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }
}

/// Data fetching configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataConfig {
    /// Trading symbols to fetch
    pub symbols: Vec<String>,
    /// Kline interval
    pub interval: String,
    /// Number of days of history
    pub history_days: u32,
    /// Data directory
    pub data_dir: String,
}

impl Default for DataConfig {
    fn default() -> Self {
        Self {
            symbols: vec![
                "BTCUSDT".to_string(),
                "ETHUSDT".to_string(),
                "SOLUSDT".to_string(),
                "AVAXUSDT".to_string(),
                "MATICUSDT".to_string(),
            ],
            interval: "60".to_string(),
            history_days: 90,
            data_dir: "data".to_string(),
        }
    }
}

/// Graph construction configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphConfig {
    /// Graph construction method
    pub method: GraphMethod,
    /// Correlation threshold (for correlation method)
    pub threshold: f64,
    /// Rolling window size
    pub window: usize,
    /// Number of neighbors (for k-NN method)
    pub k_neighbors: usize,
    /// Update frequency
    pub update_frequency: String,
}

impl Default for GraphConfig {
    fn default() -> Self {
        Self {
            method: GraphMethod::Correlation,
            threshold: 0.5,
            window: 60,
            k_neighbors: 5,
            update_frequency: "daily".to_string(),
        }
    }
}

/// Graph construction method.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum GraphMethod {
    /// Correlation-based graph
    Correlation,
    /// k-Nearest neighbors graph
    Knn,
    /// Sector-based graph
    Sector,
}

/// Model configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Model architecture
    pub architecture: ModelArchitecture,
    /// Hidden dimension
    pub hidden_dim: i64,
    /// Number of layers
    pub num_layers: usize,
    /// Number of attention heads (for GAT)
    pub num_heads: i64,
    /// Dropout probability
    pub dropout: f64,
    /// Learning rate
    pub learning_rate: f64,
    /// Number of training epochs
    pub epochs: usize,
    /// Batch size
    pub batch_size: usize,
    /// Checkpoint directory
    pub checkpoint_dir: String,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            architecture: ModelArchitecture::Gcn,
            hidden_dim: 64,
            num_layers: 3,
            num_heads: 4,
            dropout: 0.3,
            learning_rate: 0.001,
            epochs: 100,
            batch_size: 32,
            checkpoint_dir: "checkpoints".to_string(),
        }
    }
}

/// Model architecture.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ModelArchitecture {
    /// Graph Convolutional Network
    Gcn,
    /// Graph Attention Network
    Gat,
}

/// Strategy configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyConfig {
    /// Confidence threshold for signals
    pub confidence_threshold: f64,
    /// Position size as fraction of capital
    pub position_size: f64,
    /// Maximum number of positions
    pub max_positions: usize,
    /// Stop loss percentage
    pub stop_loss: f64,
    /// Take profit percentage
    pub take_profit: f64,
    /// Holding period (in candles)
    pub max_holding_period: Option<usize>,
}

impl Default for StrategyConfig {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.6,
            position_size: 0.1,
            max_positions: 5,
            stop_loss: 0.05,
            take_profit: 0.10,
            max_holding_period: Some(24), // 24 hours for hourly candles
        }
    }
}

/// Backtest configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestConfig {
    /// Initial capital
    pub initial_capital: f64,
    /// Transaction cost rate
    pub transaction_cost: f64,
    /// Slippage rate
    pub slippage: f64,
    /// Start date (optional)
    pub start_date: Option<String>,
    /// End date (optional)
    pub end_date: Option<String>,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            initial_capital: 100000.0,
            transaction_cost: 0.001,
            slippage: 0.0005,
            start_date: None,
            end_date: None,
        }
    }
}

/// Logging configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogConfig {
    /// Log level
    pub level: String,
    /// Log file path (optional)
    pub file: Option<String>,
}

impl Default for LogConfig {
    fn default() -> Self {
        Self {
            level: "info".to_string(),
            file: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = Config::default();
        assert_eq!(config.data.symbols.len(), 5);
        assert_eq!(config.graph.method, GraphMethod::Correlation);
        assert_eq!(config.model.architecture, ModelArchitecture::Gcn);
    }

    #[test]
    fn test_config_serialization() {
        let config = Config::default();
        let toml = toml::to_string(&config).unwrap();
        let parsed: Config = toml::from_str(&toml).unwrap();
        assert_eq!(config.data.symbols, parsed.data.symbols);
    }
}
