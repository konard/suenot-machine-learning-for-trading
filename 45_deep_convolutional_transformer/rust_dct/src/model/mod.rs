//! DCT Model module
//!
//! Implements the Deep Convolutional Transformer architecture:
//! - Inception Convolutional Token Embedding
//! - Multi-Head Self-Attention
//! - Transformer Encoder
//! - Movement Classifier

mod attention;
mod config;
mod dct;
mod encoder;
mod inception;

pub use attention::MultiHeadAttention;
pub use config::DCTConfig;
pub use dct::{DCTModel, Prediction};
pub use encoder::TransformerEncoder;
pub use inception::InceptionEmbedding;
