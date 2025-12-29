//! # Topic Modeling for Cryptocurrency Market Analysis
//!
//! This library provides tools for topic modeling applied to cryptocurrency
//! market data from Bybit exchange.
//!
//! ## Modules
//!
//! - `api` - Bybit API client for fetching market data and announcements
//! - `preprocessing` - Text preprocessing and tokenization
//! - `models` - Topic modeling algorithms (LSI, LDA)
//! - `utils` - Utility functions and data structures

pub mod api;
pub mod models;
pub mod preprocessing;
pub mod utils;

pub use api::bybit::BybitClient;
pub use models::lda::LDA;
pub use models::lsi::LSI;
pub use preprocessing::tokenizer::Tokenizer;
pub use preprocessing::vectorizer::TfIdfVectorizer;
