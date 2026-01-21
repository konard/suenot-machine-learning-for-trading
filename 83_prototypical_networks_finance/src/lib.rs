//! # Prototypical Networks for Finance
//!
//! This library implements Prototypical Networks for few-shot market regime classification
//! in cryptocurrency trading on Bybit.
//!
//! ## Overview
//!
//! Prototypical Networks are a meta-learning approach that learns to classify new examples
//! by computing distances to class prototypes (centroids) in an embedding space. This is
//! particularly useful for financial markets where:
//!
//! - Rare events (crashes, squeezes) have few historical examples
//! - Market regimes can shift rapidly
//! - Quick adaptation to new patterns is essential
//!
//! ## Modules
//!
//! - `network` - Neural network components for embedding and prototype computation
//! - `training` - Episodic training and learning rate scheduling
//! - `data` - Bybit API client and feature engineering
//! - `strategy` - Trading signal generation and execution
//! - `utils` - Utilities and performance metrics

pub mod network;
pub mod training;
pub mod data;
pub mod strategy;
pub mod utils;

/// Prelude module for convenient imports
pub mod prelude {
    // Network components
    pub use crate::network::{
        EmbeddingNetwork, EmbeddingConfig, ActivationType,
        PrototypeComputer, Prototype,
        DistanceFunction,
    };

    // Training components
    pub use crate::training::{
        Episode, EpisodeGenerator, EpisodeConfig,
        PrototypicalTrainer, TrainerConfig, TrainingResult,
        LearningRateScheduler, SchedulerType,
    };

    // Data types
    pub use crate::data::{
        BybitClient, BybitConfig, BybitError,
        Kline, OrderBook, OrderBookLevel, Ticker, Trade, FundingRate, OpenInterest,
        FeatureExtractor, MarketFeatures, FeatureConfig,
        MarketRegime, TradingBias,
    };

    // Strategy components
    pub use crate::strategy::{
        RegimeClassifier, ClassificationResult,
        TradingSignal, SignalGenerator, SignalConfig, SignalType,
        Position, PositionManager, PositionSide, ExecutionConfig, OrderType,
    };

    // Utilities
    pub use crate::utils::{
        PerformanceMetrics, MetricsCalculator,
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
