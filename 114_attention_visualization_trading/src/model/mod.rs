//! Transformer model with attention weight extraction.

mod transformer;

pub use transformer::{
    AttentionTransformer,
    AttentionWeights,
    MultiHeadAttention,
    TransformerConfig,
};
