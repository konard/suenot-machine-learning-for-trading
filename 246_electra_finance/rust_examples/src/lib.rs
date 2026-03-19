//! # ELECTRA for Cryptocurrency Trading
//!
//! This library provides implementations of ELECTRA-based sentiment analysis
//! and trading strategies for cryptocurrency markets using Bybit data.
//!
//! ## Modules
//!
//! - `api` - Bybit API client for fetching market data
//! - `data` - Data processing and feature engineering
//! - `models` - Neural network models for sentiment classification
//! - `nlp` - NLP utilities for text processing and tokenization

pub mod api;
pub mod data;
pub mod models;
pub mod nlp;

pub use api::bybit::BybitClient;
pub use data::features::FeatureEngineering;
pub use data::processor::DataProcessor;
pub use models::network::SentimentNetwork;
pub use nlp::tokenizer::SimpleTokenizer;
