//! # Matching Networks for Finance
//!
//! This crate implements Matching Networks for financial market pattern recognition
//! and trading signal generation. Matching Networks are a meta-learning approach
//! that uses attention-based classification over learned embeddings.
//!
//! ## Key Features
//!
//! - **Attention-based Classification**: Uses weighted voting over support examples
//! - **Full Context Embeddings (FCE)**: Context-dependent embeddings for better accuracy
//! - **Market Pattern Recognition**: Identify trend continuations, reversals, breakouts, etc.
//! - **Bybit Integration**: Real-time cryptocurrency data from Bybit API
//! - **Stock Data Support**: Historical stock data processing
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use matching_networks_finance::{
//!     MatchingNetwork, MarketPattern, MarketPatternClassifier,
//! };
//!
//! // Create a matching network
//! let network = MatchingNetwork::new(
//!     20,   // input dimension (number of features)
//!     64,   // hidden dimension
//!     32,   // embedding dimension
//!     true, // use Full Context Embeddings
//! );
//!
//! // Create a pattern classifier
//! let classifier = MarketPatternClassifier::new(network);
//!
//! // Fit on support examples and predict
//! classifier.fit(&support_features, &support_labels);
//! let predictions = classifier.predict(&query_features);
//! ```
//!
//! ## Architecture
//!
//! The Matching Network computes:
//! ```text
//! P(ŷ|x̂,S) = Σᵢ a(x̂,xᵢ)yᵢ
//! ```
//!
//! Where:
//! - `a(x̂,xᵢ)` is the attention weight (softmax of cosine similarities)
//! - `xᵢ` are support set examples
//! - `yᵢ` are support set labels
//!
//! ## Modules
//!
//! - [`network`]: Core matching network implementation
//! - [`data`]: Data fetching and preprocessing (Bybit, stock data)
//! - [`strategy`]: Trading strategies based on pattern classification
//! - [`training`]: Episodic training utilities
//! - [`utils`]: Metrics, evaluation, and helper functions

pub mod network;
pub mod data;
pub mod strategy;
pub mod training;
pub mod utils;

// Re-export main types for convenience
pub use network::{
    MatchingNetwork,
    EmbeddingNetwork,
    AttentionModule,
    FullContextEmbedding,
    DistanceFunction,
};

pub use data::{
    MarketPattern,
    MarketFeatures,
    FeatureExtractor,
};

#[cfg(feature = "bybit")]
pub use data::BybitClient;

pub use strategy::{
    PatternTradingStrategy,
    TradingSignal,
    Position,
};

pub use training::{
    EpisodicTrainer,
    Episode,
    TrainingConfig,
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
        // Verify main types are exported
        let _ = MarketPattern::TrendContinuation;
        assert!(!VERSION.is_empty());
    }
}
