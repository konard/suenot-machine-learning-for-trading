//! # Transfer Learning for Trading
//!
//! This crate implements transfer learning techniques for financial market prediction
//! and cross-domain trading signal generation. Transfer learning enables models trained
//! on data-rich markets (source domain) to be adapted for data-scarce markets (target domain).
//!
//! ## Key Features
//!
//! - **Feature Extraction Networks**: Multi-layer feature extractors with batch normalization
//! - **Domain Adaptation**: MMD, CORAL, and adversarial domain adaptation methods
//! - **Transfer Strategies**: Freeze, partial fine-tune, and full fine-tune approaches
//! - **Bybit Integration**: Real-time cryptocurrency data from Bybit API
//! - **Stock Data Support**: Historical stock data processing
//! - **Backtesting**: Strategy backtesting with transfer learning models
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use transfer_learning_trading::{
//!     TransferNetwork, DomainAdapter, AdaptationMethod,
//!     FeatureExtractor, TradingSignal,
//! };
//!
//! // Create a transfer learning network
//! let network = TransferNetwork::new(
//!     20,   // input dimension (number of features)
//!     128,  // hidden dimension
//!     64,   // feature/embedding dimension
//!     true, // use domain adaptation
//! );
//!
//! // Pre-train on source domain data
//! network.pretrain(&source_features, &source_labels, &config);
//!
//! // Adapt to target domain
//! let adapter = DomainAdapter::new(AdaptationMethod::MMD);
//! adapter.adapt(&source_features, &target_features);
//!
//! // Generate trading signals
//! let signals = network.predict(&target_features);
//! ```
//!
//! ## Architecture
//!
//! The transfer learning pipeline consists of:
//! ```text
//! Source Data → Feature Extractor → Shared Features → Task Classifier → Predictions
//!                    ↑                    ↓
//!              (transfer weights)   Domain Adapter
//!                    ↓                    ↑
//! Target Data → Feature Extractor → Aligned Features → Task Classifier → Predictions
//! ```
//!
//! ## Modules
//!
//! - [`network`]: Neural network components (feature extractor, domain adapter, transfer network)
//! - [`data`]: Data fetching and preprocessing (Bybit, stock data, feature engineering)
//! - [`strategy`]: Trading strategies based on transfer learning predictions
//! - [`training`]: Training utilities (pre-training, fine-tuning, adaptation)
//! - [`utils`]: Metrics, evaluation, and helper functions

pub mod network;
pub mod data;
pub mod strategy;
pub mod training;
pub mod utils;

// Re-export main types for convenience
pub use network::{
    TransferNetwork,
    FeatureExtractor,
    DomainAdapter,
    AdaptationMethod,
    FineTuneStrategy,
};

pub use data::{
    MarketFeatures,
    OHLCVBar,
};

#[cfg(feature = "bybit")]
pub use data::BybitClient;

pub use strategy::{
    TransferTradingStrategy,
    TradingSignal,
    Position,
};

pub use training::{
    TransferTrainer,
    TrainingConfig,
    AdaptationConfig,
};

pub use utils::{
    Metrics,
    ConfusionMatrix,
};

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::network::*;
    pub use crate::data::*;
    pub use crate::strategy::*;
    pub use crate::training::*;
    pub use crate::utils::*;
}

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_library_exports() {
        let _ = AdaptationMethod::MMD;
        let _ = FineTuneStrategy::FeatureExtraction;
        let _ = TradingSignal::Buy;
        assert!(!VERSION.is_empty());
    }
}
