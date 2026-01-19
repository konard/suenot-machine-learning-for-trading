//! # Sentiment Analysis Module
//!
//! Sentiment analysis models and scoring for financial text.

mod analyzer;
mod lexicon;
pub mod crypto;
mod aggregator;

pub use analyzer::{SentimentAnalyzer, SentimentResult, SentimentLevel};
pub use lexicon::FinancialLexicon;
pub use crypto::CryptoTerminology;
pub use aggregator::SentimentAggregator;
