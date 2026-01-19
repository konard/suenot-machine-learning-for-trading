//! # Data Module
//!
//! Text preprocessing and news data handling for sentiment analysis.

mod preprocessing;
mod news;

pub use preprocessing::TextPreprocessor;
pub use news::{NewsArticle, NewsCollector, NewsSource};
