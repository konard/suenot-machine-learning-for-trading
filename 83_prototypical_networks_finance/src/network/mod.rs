//! Neural network components for prototypical networks
//!
//! This module provides:
//! - Embedding networks for converting market features to embeddings
//! - Prototype computation from support set embeddings
//! - Distance functions for classification

mod embedding;
mod prototype;
mod distance;

pub use embedding::{EmbeddingNetwork, EmbeddingConfig, ActivationType};
pub use prototype::{PrototypeComputer, Prototype};
pub use distance::{DistanceFunction, DistanceType};
