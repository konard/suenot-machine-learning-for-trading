//! Core Matching Network implementation
//!
//! This module contains the neural network components for Matching Networks:
//! - Embedding networks for feature transformation
//! - Attention mechanisms for weighted voting
//! - Full Context Embeddings (FCE) for context-dependent representations

mod embedding;
mod attention;
mod fce;
mod matching;

pub use embedding::EmbeddingNetwork;
pub use attention::{AttentionModule, DistanceFunction};
pub use fce::FullContextEmbedding;
pub use matching::MatchingNetwork;
