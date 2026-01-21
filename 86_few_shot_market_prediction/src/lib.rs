//! # Few-Shot Market Prediction
//!
//! This library implements multiple few-shot learning approaches for financial market
//! prediction, including metric-based methods, MAML-inspired optimization, and hybrid
//! approaches.
//!
//! ## Overview
//!
//! Few-shot learning enables rapid adaptation to new market conditions with minimal
//! examples. This is particularly valuable for:
//!
//! - **New Asset Analysis**: Predict patterns for newly listed cryptocurrencies
//! - **Rare Event Detection**: Recognize crashes, squeezes with few historical examples
//! - **Cross-Asset Transfer**: Apply knowledge from one asset to another
//! - **Regime Change Adaptation**: Quickly adjust to market regime shifts
//!
//! ## Supported Methods
//!
//! 1. **Metric-Based** (Prototypical Networks): Classify based on distance to prototypes
//! 2. **MAML-Inspired**: Gradient-based quick adaptation
//! 3. **Siamese Networks**: Pairwise similarity learning
//! 4. **Hybrid**: Combine metric and optimization approaches
//!
//! ## Example Usage
//!
//! ```rust,ignore
//! use few_shot_market_prediction::prelude::*;
//!
//! // Create a metric-based few-shot predictor
//! let config = FewShotConfig::default()
//!     .with_method(FewShotMethod::Metric)
//!     .with_embedding_dim(32);
//!
//! let predictor = FewShotPredictor::new(config);
//!
//! // Fit on support set
//! predictor.fit(&support_features, &support_labels);
//!
//! // Predict on query
//! let (predictions, confidences) = predictor.predict(&query_features);
//! ```
//!
//! ## Modules
//!
//! - `network` - Neural network components (embedding, distance computation)
//! - `methods` - Few-shot learning methods (metric, MAML, siamese, hybrid)
//! - `data` - Bybit API integration and feature extraction
//! - `training` - Episodic training framework
//! - `strategy` - Trading signal generation and regime classification

pub mod network;
pub mod methods;
pub mod data;
pub mod training;
pub mod strategy;

/// Prelude module for convenient imports
pub mod prelude {
    // Network components
    pub use crate::network::{
        EmbeddingNetwork, EmbeddingConfig, ActivationType, NetworkParams,
        DistanceComputer, DistanceMetric,
    };

    // Few-shot methods
    pub use crate::methods::{
        FewShotMethod, FewShotConfig, FewShotPredictor, FewShotLearner,
        MetricBasedLearner, MAMLLearner, SiameseLearner, HybridLearner,
        PredictionResult, AdaptationResult,
    };

    // Data components
    pub use crate::data::{
        BybitClient, BybitConfig,
        MarketFeatures, FeatureExtractor, FeatureConfig,
        Kline, Trade, OrderBook, FundingRate, Ticker,
    };

    // Training components
    pub use crate::training::{
        Episode, EpisodeGenerator, EpisodeConfig,
        MetaTrainer, MetaTrainerConfig, TrainingResult,
    };

    // Strategy components
    pub use crate::strategy::{
        MarketRegime, RegimeClassifier, RegimeClassification,
        TradingSignal, SignalType, SignalGenerator, SignalConfig,
        RiskManager, RiskConfig, RiskCheckResult, RiskSummary, Position,
    };
}

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }
}
