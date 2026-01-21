//! Configuration and metrics utilities.

use crate::data::BybitConfig;
use crate::execution::{ExecutionConfig, LlmConfig};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Application configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    /// Execution engine configuration
    pub execution: ExecutionConfig,
    /// Bybit exchange configuration
    pub bybit: Option<BybitConfig>,
    /// LLM configuration
    pub llm: Option<LlmConfig>,
    /// Logging level
    pub log_level: String,
    /// Enable debug mode
    pub debug: bool,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            execution: ExecutionConfig::default(),
            bybit: Some(BybitConfig::default()),
            llm: None,
            log_level: "info".to_string(),
            debug: false,
        }
    }
}

impl AppConfig {
    /// Create a new configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Load configuration from environment variables
    pub fn from_env() -> Self {
        let mut config = Self::default();

        // Bybit credentials from env
        if let Ok(api_key) = std::env::var("BYBIT_API_KEY") {
            if let Ok(api_secret) = std::env::var("BYBIT_API_SECRET") {
                let testnet = std::env::var("BYBIT_TESTNET")
                    .map(|v| v == "true" || v == "1")
                    .unwrap_or(false);

                config.bybit = Some(if testnet {
                    BybitConfig::testnet_with_credentials(api_key, api_secret)
                } else {
                    BybitConfig::with_credentials(api_key, api_secret)
                });
            }
        }

        // LLM credentials from env
        if let Ok(api_key) = std::env::var("OPENAI_API_KEY") {
            config.llm = Some(LlmConfig::openai(api_key));
        } else if let Ok(api_key) = std::env::var("ANTHROPIC_API_KEY") {
            config.llm = Some(LlmConfig::anthropic(api_key));
        }

        // Log level from env
        if let Ok(level) = std::env::var("LOG_LEVEL") {
            config.log_level = level;
        }

        // Debug mode from env
        config.debug = std::env::var("DEBUG")
            .map(|v| v == "true" || v == "1")
            .unwrap_or(false);

        config
    }

    /// Enable testnet mode
    pub fn with_testnet(mut self) -> Self {
        if let Some(ref mut bybit) = self.bybit {
            bybit.testnet = true;
        } else {
            self.bybit = Some(BybitConfig::testnet());
        }
        self
    }

    /// Enable LLM-assisted execution
    pub fn with_llm(mut self, config: LlmConfig) -> Self {
        self.llm = Some(config);
        self.execution.use_llm = true;
        self
    }

    /// Set execution to aggressive mode
    pub fn aggressive(mut self) -> Self {
        self.execution = ExecutionConfig::aggressive();
        self
    }

    /// Set execution to passive mode
    pub fn passive(mut self) -> Self {
        self.execution = ExecutionConfig::passive();
        self
    }
}

/// Load configuration from a TOML file
pub fn load_config<P: AsRef<Path>>(path: P) -> Result<AppConfig, config::ConfigError> {
    let settings = config::Config::builder()
        .add_source(config::File::from(path.as_ref()))
        .add_source(config::Environment::with_prefix("LLM_EXEC"))
        .build()?;

    settings.try_deserialize()
}

/// Metrics recorder for execution analytics
#[derive(Debug, Clone, Default)]
pub struct MetricsRecorder {
    /// Number of orders executed
    pub orders_executed: u64,
    /// Total quantity executed
    pub total_quantity: f64,
    /// Total value executed
    pub total_value: f64,
    /// Sum of implementation shortfalls (bps)
    pub total_is_bps: f64,
    /// Sum of VWAP slippages (bps)
    pub total_vwap_slippage_bps: f64,
    /// Number of child orders
    pub total_child_orders: u64,
    /// Total execution time (seconds)
    pub total_execution_time: u64,
    /// Number of LLM decisions made
    pub llm_decisions: u64,
}

impl MetricsRecorder {
    /// Create a new metrics recorder
    pub fn new() -> Self {
        Self::default()
    }

    /// Record an execution result
    pub fn record(&mut self, result: &crate::execution::ExecutionResult) {
        self.orders_executed += 1;
        self.total_quantity += result.filled_quantity;
        self.total_value += result.filled_quantity * result.average_price;
        self.total_is_bps += result.implementation_shortfall;
        self.total_vwap_slippage_bps += result.vwap_slippage;
        self.total_child_orders += result.child_order_count as u64;
        self.total_execution_time += result.duration_secs;
        self.llm_decisions += result.llm_decisions.len() as u64;
    }

    /// Get average implementation shortfall
    pub fn average_is_bps(&self) -> f64 {
        if self.orders_executed > 0 {
            self.total_is_bps / self.orders_executed as f64
        } else {
            0.0
        }
    }

    /// Get average VWAP slippage
    pub fn average_vwap_slippage_bps(&self) -> f64 {
        if self.orders_executed > 0 {
            self.total_vwap_slippage_bps / self.orders_executed as f64
        } else {
            0.0
        }
    }

    /// Get average child orders per parent order
    pub fn average_child_orders(&self) -> f64 {
        if self.orders_executed > 0 {
            self.total_child_orders as f64 / self.orders_executed as f64
        } else {
            0.0
        }
    }

    /// Get average execution time
    pub fn average_execution_time(&self) -> f64 {
        if self.orders_executed > 0 {
            self.total_execution_time as f64 / self.orders_executed as f64
        } else {
            0.0
        }
    }

    /// Print summary statistics
    pub fn print_summary(&self) {
        println!("=== Execution Metrics Summary ===");
        println!("Orders executed: {}", self.orders_executed);
        println!("Total quantity: {:.4}", self.total_quantity);
        println!("Total value: {:.2}", self.total_value);
        println!("Average IS: {:.2} bps", self.average_is_bps());
        println!("Average VWAP slippage: {:.2} bps", self.average_vwap_slippage_bps());
        println!("Average child orders: {:.1}", self.average_child_orders());
        println!("Average execution time: {:.1} seconds", self.average_execution_time());
        println!("LLM decisions: {}", self.llm_decisions);
    }

    /// Reset all metrics
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

/// Initialize logging with the specified level
pub fn init_logging(level: &str) {
    use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

    let filter = tracing_subscriber::filter::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::filter::EnvFilter::new(level));

    tracing_subscriber::registry()
        .with(filter)
        .with(tracing_subscriber::fmt::layer())
        .init();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_app_config_default() {
        let config = AppConfig::default();
        assert!(config.bybit.is_some());
        assert!(config.llm.is_none());
        assert_eq!(config.log_level, "info");
    }

    #[test]
    fn test_app_config_testnet() {
        let config = AppConfig::default().with_testnet();
        assert!(config.bybit.as_ref().unwrap().testnet);
    }

    #[test]
    fn test_app_config_aggressive() {
        let config = AppConfig::default().aggressive();
        assert!(config.execution.max_participation_rate > 0.3);
    }

    #[test]
    fn test_metrics_recorder() {
        let mut recorder = MetricsRecorder::new();

        // Create a mock result
        let result = crate::execution::ExecutionResult {
            order_id: crate::execution::OrderId::new(),
            symbol: "BTCUSDT".to_string(),
            side: crate::execution::Side::Buy,
            total_quantity: 10.0,
            filled_quantity: 10.0,
            child_order_count: 5,
            average_price: 50100.0,
            arrival_price: 50000.0,
            market_vwap: 50050.0,
            implementation_shortfall: 20.0,
            vwap_slippage: 10.0,
            participation_rate: 0.1,
            start_time: chrono::Utc::now(),
            end_time: chrono::Utc::now(),
            duration_secs: 300,
            status: crate::execution::ParentOrderStatus::Completed,
            state_history: vec![],
            llm_decisions: vec![],
        };

        recorder.record(&result);

        assert_eq!(recorder.orders_executed, 1);
        assert_eq!(recorder.total_quantity, 10.0);
        assert_eq!(recorder.average_is_bps(), 20.0);
    }
}
