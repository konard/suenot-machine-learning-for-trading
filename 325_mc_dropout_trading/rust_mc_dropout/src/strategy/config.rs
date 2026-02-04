//! Configuration for trading strategy

use serde::{Deserialize, Serialize};

/// Configuration for signal generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalConfig {
    /// Minimum z-score to generate a signal
    pub confidence_threshold: f64,
    /// Maximum uncertainty to trade
    pub uncertainty_threshold: f64,
    /// Maximum position size (fraction of portfolio)
    pub max_position_size: f64,
    /// Minimum position size
    pub min_position_size: f64,
    /// Use Kelly criterion for position sizing
    pub use_kelly: bool,
    /// Minimum expected return to trade
    pub min_return_threshold: f64,
    /// Number of MC samples for inference
    pub n_mc_samples: usize,
}

impl Default for SignalConfig {
    fn default() -> Self {
        Self {
            confidence_threshold: 1.5,
            uncertainty_threshold: 0.03,
            max_position_size: 0.1,
            min_position_size: 0.01,
            use_kelly: true,
            min_return_threshold: 0.001,
            n_mc_samples: 50,
        }
    }
}

impl SignalConfig {
    /// Create a conservative configuration
    pub fn conservative() -> Self {
        Self {
            confidence_threshold: 2.0,
            uncertainty_threshold: 0.02,
            max_position_size: 0.05,
            min_position_size: 0.01,
            use_kelly: true,
            min_return_threshold: 0.002,
            n_mc_samples: 100,
        }
    }

    /// Create an aggressive configuration
    pub fn aggressive() -> Self {
        Self {
            confidence_threshold: 1.0,
            uncertainty_threshold: 0.05,
            max_position_size: 0.2,
            min_position_size: 0.02,
            use_kelly: true,
            min_return_threshold: 0.0005,
            n_mc_samples: 30,
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.confidence_threshold < 0.0 {
            return Err("Confidence threshold must be non-negative".to_string());
        }
        if self.uncertainty_threshold <= 0.0 {
            return Err("Uncertainty threshold must be positive".to_string());
        }
        if self.max_position_size <= 0.0 || self.max_position_size > 1.0 {
            return Err("Max position size must be in (0, 1]".to_string());
        }
        if self.min_position_size < 0.0 || self.min_position_size > self.max_position_size {
            return Err("Min position size must be in [0, max_position_size]".to_string());
        }
        Ok(())
    }
}
