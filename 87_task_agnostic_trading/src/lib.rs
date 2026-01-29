//! # Task-Agnostic Trading
//!
//! This library implements task-agnostic representation learning for financial market
//! trading. Instead of training separate models for each trading task, we learn universal
//! representations that transfer across multiple tasks simultaneously.
//!
//! ## Overview
//!
//! Task-agnostic representation learning addresses a core challenge in ML-driven trading:
//!
//! - **Trend prediction** models don't transfer to volatility forecasting
//! - **Regime detection** models are separate from portfolio optimization
//! - **Risk assessment** requires yet another specialized model
//!
//! By learning universal encodings of market state, a single model backbone serves
//! all downstream trading tasks with task-specific lightweight heads.
//!
//! ## Key Components
//!
//! 1. **Universal Encoder**: Shared backbone that maps raw market features into a
//!    task-agnostic representation space
//! 2. **Task Heads**: Lightweight task-specific layers for trend prediction,
//!    volatility forecasting, regime detection, and risk assessment
//! 3. **Gradient Harmonization**: Balances gradients from different tasks to prevent
//!    any single task from dominating training
//! 4. **Decision Fusion**: Combines outputs from multiple task heads into unified
//!    trading signals
//!
//! ## Example Usage
//!
//! ```rust,ignore
//! use task_agnostic_trading::prelude::*;
//!
//! // Create a universal encoder with multi-task heads
//! let config = TaskAgnosticConfig::default();
//! let mut model = TaskAgnosticModel::new(config);
//!
//! // Train on multiple tasks simultaneously
//! let trainer = MultiTaskTrainer::new(TrainerConfig::default());
//! trainer.train(&mut model, &train_data);
//!
//! // Generate unified trading signals
//! let signals = model.predict(&market_features);
//! ```
//!
//! ## Modules
//!
//! - `model` - Universal encoder and task-specific heads
//! - `data` - Bybit API integration and feature extraction
//! - `training` - Multi-task training with gradient harmonization
//! - `strategy` - Trading signal generation and regime classification
//! - `backtest` - Backtesting engine for strategy evaluation

pub mod model;
pub mod data;
pub mod training;
pub mod strategy;
pub mod backtest;

/// Prelude module for convenient imports
pub mod prelude {
    // Model components
    pub use crate::model::{
        UniversalEncoder, EncoderConfig,
        TaskHead, TaskHeadConfig, TaskType,
        DecisionFusion, FusionConfig, FusionMethod,
        TaskAgnosticModel, TaskAgnosticConfig,
    };

    // Data components
    pub use crate::data::{
        BybitClient, BybitConfig,
        MarketFeatures, FeatureExtractor, FeatureConfig,
        Kline, Trade, OrderBook, FundingRate, Ticker,
    };

    // Training components
    pub use crate::training::{
        MultiTaskTrainer, TrainerConfig,
        GradientHarmonizer, HarmonizerConfig, HarmonizationMethod,
        TaskLoss, TrainingResult, EpochMetrics,
    };

    // Strategy components
    pub use crate::strategy::{
        MarketRegime, RegimeClassifier, RegimeConfig,
        TradingSignal, SignalType, SignalGenerator, SignalConfig,
    };

    // Backtest components
    pub use crate::backtest::{
        BacktestEngine, BacktestConfig, BacktestResult,
        PerformanceMetrics,
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
