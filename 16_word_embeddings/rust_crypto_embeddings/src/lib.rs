//! # Crypto Embeddings
//!
//! A Rust library for word embeddings analysis of cryptocurrency trading data.
//!
//! This library provides tools for:
//! - Fetching cryptocurrency data from Bybit exchange
//! - Text preprocessing for trading-related content
//! - Training Word2Vec-style embeddings
//! - Analyzing sentiment and similarity in crypto texts
//!
//! ## Modules
//!
//! - `api`: Bybit API client for market data
//! - `preprocessing`: Text tokenization and cleaning
//! - `embeddings`: Word2Vec implementation
//! - `analysis`: Similarity and sentiment analysis
//! - `utils`: Common utilities and error types

pub mod api;
pub mod embeddings;
pub mod preprocessing;
pub mod analysis;
pub mod utils;

pub use api::BybitClient;
pub use embeddings::Word2Vec;
pub use preprocessing::Tokenizer;
pub use analysis::SimilarityAnalyzer;

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
