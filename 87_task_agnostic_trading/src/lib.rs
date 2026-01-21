//! # Task-Agnostic Representation Learning for Trading
//!
//! This library implements task-agnostic representation learning for financial markets,
//! enabling a single model to solve multiple trading tasks simultaneously through
//! shared representations.
//!
//! ## Overview
//!
//! Task-agnostic learning creates universal market representations that work across
//! multiple prediction tasks:
//!
//! - **Direction Prediction**: Classify market movement (up/down/sideways)
//! - **Volatility Estimation**: Predict future price volatility
//! - **Regime Classification**: Identify market regime (trending/ranging/volatile)
//! - **Return Prediction**: Forecast expected returns
//!
//! ## Architecture
//!
//! The system uses a shared encoder with task-specific heads:
//!
//! ```text
//! Market Data → [Shared Encoder] → Universal Representation
//!                                         ↓
//!                    ┌────────────────────┼────────────────────┐
//!                    ↓                    ↓                    ↓
//!              [Direction]          [Volatility]         [Regime]
//!                 Head                  Head                Head
//! ```
//!
//! ## Example Usage
//!
//! ```rust,ignore
//! use task_agnostic_trading::prelude::*;
//!
//! // Create multi-task model
//! let config = MultiTaskConfig::default()
//!     .with_encoder_type(EncoderType::Transformer)
//!     .with_embedding_dim(64);
//!
//! let model = MultiTaskModel::new(config);
//!
//! // Train on multiple tasks simultaneously
//! model.fit(&features, &task_labels);
//!
//! // Get predictions for all tasks
//! let predictions = model.predict_all(&query_features);
//! ```
//!
//! ## Modules
//!
//! - `encoder` - Shared encoder architectures (Transformer, CNN, MoE)
//! - `tasks` - Task-specific prediction heads
//! - `data` - Bybit API integration and feature extraction
//! - `training` - Multi-task training with gradient harmonization
//! - `fusion` - Decision fusion combining multiple task outputs

pub mod encoder;
pub mod tasks;
pub mod data;
pub mod training;
pub mod fusion;

/// Prelude module for convenient imports
pub mod prelude {
    // Encoder components
    pub use crate::encoder::{
        SharedEncoder, EncoderConfig, EncoderType,
        TransformerEncoder, TransformerConfig,
        CNNEncoder, CNNConfig,
        MoEEncoder, MoEConfig,
    };

    // Task heads
    pub use crate::tasks::{
        TaskHead, TaskType,
        DirectionHead, DirectionConfig, DirectionPrediction,
        VolatilityHead, VolatilityConfig, VolatilityPrediction,
        RegimeHead, RegimeConfig, RegimePrediction, MarketRegime,
        ReturnsHead, ReturnsConfig, ReturnsPrediction,
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
        GradientHarmonizer, HarmonizerType,
        TaskWeighter, WeightingStrategy,
    };

    // Fusion components
    pub use crate::fusion::{
        DecisionFusion, FusionConfig, FusionMethod,
        TradingDecision, DecisionConfidence,
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
