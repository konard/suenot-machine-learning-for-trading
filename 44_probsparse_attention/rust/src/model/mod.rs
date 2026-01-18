//! Модули модели Informer с ProbSparse Attention
//!
//! - `config` - Конфигурация модели
//! - `attention` - Механизм ProbSparse Attention
//! - `embedding` - Слои эмбеддинга
//! - `informer` - Полная модель Informer

pub mod config;
pub mod attention;
pub mod embedding;
pub mod informer;

pub use config::{InformerConfig, OutputType};
pub use attention::{ProbSparseAttention, AttentionDistilling, AttentionWeights};
pub use embedding::{TokenEmbedding, PositionalEncoding};
pub use informer::InformerModel;
