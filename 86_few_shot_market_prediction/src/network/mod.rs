//! Neural network components for few-shot learning
//!
//! This module provides:
//! - Embedding networks for converting market features to embeddings
//! - Distance computation for similarity-based classification
//! - Support for MAML-style gradient computation

mod embedding;
mod distance;

pub use embedding::{EmbeddingNetwork, EmbeddingConfig, ActivationType, NetworkParams};
pub use distance::{DistanceComputer, DistanceMetric};
