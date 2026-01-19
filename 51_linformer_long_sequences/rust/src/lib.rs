//! # Linformer: Self-Attention with Linear Complexity
//!
//! This crate provides a Rust implementation of Linformer, a transformer
//! architecture that achieves linear O(n) complexity instead of quadratic O(n²)
//! through low-rank approximation of the attention matrix.
//!
//! ## Key Features
//!
//! - **Linear Complexity**: Process sequences of any length efficiently
//! - **Memory Efficient**: Up to 98% reduction in memory usage for long sequences
//! - **Trading Focused**: Includes data loading, technical indicators, and backtesting
//! - **Production Ready**: Optimized for real-time financial applications
//!
//! ## Example Usage
//!
//! ```rust,no_run
//! use linformer::{
//!     model::{Linformer, LinformerConfig},
//!     data::{DataLoader, TechnicalFeatures},
//!     strategy::{Backtester, BacktestConfig},
//! };
//!
//! // Create model configuration
//! let config = LinformerConfig::new(128, 4, 512, 64, 4)
//!     .with_n_features(6)
//!     .with_n_outputs(1);
//!
//! // Build the model
//! let model = Linformer::new(config).expect("Failed to create model");
//!
//! // Print model summary
//! println!("{}", model.summary());
//! ```
//!
//! ## Architecture
//!
//! Linformer achieves linear complexity by projecting keys and values
//! to a lower dimension using learned projection matrices:
//!
//! ```text
//! Standard:  Attention = softmax(Q @ K.T) @ V           // O(n²)
//! Linformer: Attention = softmax(Q @ (E @ K).T) @ (F @ V)  // O(n × k)
//! ```
//!
//! Where k << n is the projection dimension.
//!
//! ## Modules
//!
//! - [`api`]: HTTP clients for fetching market data from exchanges
//! - [`data`]: Data loading, preprocessing, and technical indicators
//! - [`model`]: Linformer attention and transformer implementation
//! - [`strategy`]: Backtesting framework and performance metrics
//!
//! ## References
//!
//! - [Linformer: Self-Attention with Linear Complexity](https://arxiv.org/abs/2006.04768)

pub mod api;
pub mod data;
pub mod model;
pub mod strategy;

// Re-export commonly used types
pub use api::{ApiError, BybitClient, Kline};
pub use data::{DataLoader, SequenceDataset, TechnicalFeatures};
pub use model::{Linformer, LinformerAttention, LinformerConfig};
pub use strategy::{BacktestConfig, BacktestResult, Backtester, PerformanceMetrics};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Returns library information.
pub fn info() -> String {
    format!(
        "Linformer v{}\n\
         Self-Attention with Linear Complexity\n\
         For financial time series analysis",
        VERSION
    )
}

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::api::{ApiError, BybitClient, Kline};
    pub use crate::data::{DataLoader, SequenceDataset, TechnicalFeatures};
    pub use crate::model::{Linformer, LinformerAttention, LinformerConfig};
    pub use crate::strategy::{BacktestConfig, BacktestResult, Backtester, PerformanceMetrics};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }

    #[test]
    fn test_info() {
        let info = info();
        assert!(info.contains("Linformer"));
    }

    #[test]
    fn test_model_integration() {
        let config = LinformerConfig::new(32, 2, 64, 16, 2)
            .with_n_features(4)
            .with_n_outputs(1);

        let model = Linformer::new(config).unwrap();
        assert!(!model.summary().is_empty());
    }
}
