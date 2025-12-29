//! Application configuration.

use serde::{Deserialize, Serialize};

/// Main application configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    /// Bybit API configuration
    pub bybit: BybitConfig,
    /// Training configuration
    pub training: TrainingConfig,
    /// Environment configuration
    pub environment: EnvironmentConfig,
}

/// Bybit API configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BybitConfig {
    /// API key (optional for public endpoints)
    pub api_key: Option<String>,
    /// API secret (optional for public endpoints)
    pub api_secret: Option<String>,
    /// Use testnet
    pub use_testnet: bool,
    /// Default trading symbol
    pub symbol: String,
    /// Default interval
    pub interval: String,
}

/// Training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Number of training episodes
    pub num_episodes: usize,
    /// Maximum steps per episode
    pub max_steps: usize,
    /// Batch size for DQN training
    pub batch_size: usize,
    /// Learning rate
    pub learning_rate: f64,
    /// Discount factor (gamma)
    pub gamma: f64,
    /// Initial exploration rate (epsilon)
    pub epsilon_start: f64,
    /// Final exploration rate
    pub epsilon_end: f64,
    /// Epsilon decay rate
    pub epsilon_decay: f64,
    /// Replay buffer size
    pub buffer_size: usize,
    /// Target network update frequency
    pub target_update_freq: usize,
    /// Save model every N episodes
    pub save_freq: usize,
    /// Log progress every N episodes
    pub log_freq: usize,
    /// Hidden layer sizes
    pub hidden_layers: Vec<usize>,
    /// Use Double DQN
    pub double_dqn: bool,
}

/// Environment configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentConfig {
    /// Episode length in trading days/periods
    pub episode_length: usize,
    /// Trading cost in basis points
    pub trading_cost_bps: f64,
    /// Time cost (holding cost) in basis points
    pub time_cost_bps: f64,
    /// Initial capital
    pub initial_capital: f64,
    /// Maximum drawdown before episode termination
    pub max_drawdown: f64,
    /// Reward scaling factor
    pub reward_scale: f64,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            bybit: BybitConfig {
                api_key: None,
                api_secret: None,
                use_testnet: false,
                symbol: "BTCUSDT".to_string(),
                interval: "60".to_string(), // 1 hour
            },
            training: TrainingConfig {
                num_episodes: 1000,
                max_steps: 252,
                batch_size: 64,
                learning_rate: 0.001,
                gamma: 0.99,
                epsilon_start: 1.0,
                epsilon_end: 0.01,
                epsilon_decay: 0.995,
                buffer_size: 100_000,
                target_update_freq: 100,
                save_freq: 100,
                log_freq: 10,
                hidden_layers: vec![128, 64],
                double_dqn: true,
            },
            environment: EnvironmentConfig {
                episode_length: 252,
                trading_cost_bps: 0.001,
                time_cost_bps: 0.0001,
                initial_capital: 10000.0,
                max_drawdown: 0.3,
                reward_scale: 100.0,
            },
        }
    }
}

impl AppConfig {
    /// Load configuration from a file
    pub fn from_file(path: &str) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: AppConfig = serde_json::from_str(&content)?;
        Ok(config)
    }

    /// Save configuration to a file
    pub fn to_file(&self, path: &str) -> anyhow::Result<()> {
        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }

    /// Load from environment variables
    pub fn from_env() -> Self {
        let mut config = Self::default();

        if let Ok(key) = std::env::var("BYBIT_API_KEY") {
            config.bybit.api_key = Some(key);
        }
        if let Ok(secret) = std::env::var("BYBIT_API_SECRET") {
            config.bybit.api_secret = Some(secret);
        }
        if let Ok(symbol) = std::env::var("TRADING_SYMBOL") {
            config.bybit.symbol = symbol;
        }
        if let Ok(testnet) = std::env::var("USE_TESTNET") {
            config.bybit.use_testnet = testnet.to_lowercase() == "true";
        }

        config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = AppConfig::default();
        assert_eq!(config.bybit.symbol, "BTCUSDT");
        assert_eq!(config.training.num_episodes, 1000);
    }

    #[test]
    fn test_config_serialization() {
        let config = AppConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let loaded: AppConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config.bybit.symbol, loaded.bybit.symbol);
    }
}
