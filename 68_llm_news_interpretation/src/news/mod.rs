//! News module for collecting and preprocessing financial news.
//!
//! This module provides functionality for:
//! - Connecting to various news sources
//! - Preprocessing raw news text
//! - Deduplicating news items
//! - Managing news data types

mod preprocessor;
mod sources;
mod types;

pub use preprocessor::NewsPreprocessor;
pub use sources::{NewsSourceConnector, RssSource, TwitterSource};
pub use types::{NewsItem, NewsSource, RawNews};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// News collection manager that aggregates news from multiple sources
#[derive(Debug)]
pub struct NewsCollector {
    sources: Vec<Box<dyn NewsSourceConnector>>,
    preprocessor: NewsPreprocessor,
    seen_hashes: HashSet<String>,
}

impl NewsCollector {
    /// Create a new news collector with default sources
    pub fn new() -> Self {
        Self {
            sources: Vec::new(),
            preprocessor: NewsPreprocessor::new(),
            seen_hashes: HashSet::new(),
        }
    }

    /// Add a news source connector
    pub fn add_source(&mut self, source: Box<dyn NewsSourceConnector>) {
        self.sources.push(source);
    }

    /// Collect news from all sources within a time range
    pub async fn collect(
        &mut self,
        since: DateTime<Utc>,
    ) -> Result<Vec<NewsItem>, NewsCollectionError> {
        let mut all_news = Vec::new();

        for source in &self.sources {
            match source.fetch_news(since).await {
                Ok(news) => all_news.extend(news),
                Err(e) => {
                    tracing::warn!("Failed to fetch from source: {}", e);
                }
            }
        }

        // Deduplicate based on content hash
        let unique_news: Vec<NewsItem> = all_news
            .into_iter()
            .filter(|item| {
                let hash = self.preprocessor.hash_content(&item.text);
                if self.seen_hashes.contains(&hash) {
                    false
                } else {
                    self.seen_hashes.insert(hash);
                    true
                }
            })
            .collect();

        // Preprocess each news item
        let processed: Vec<NewsItem> = unique_news
            .into_iter()
            .map(|item| self.preprocessor.process(item))
            .collect();

        Ok(processed)
    }

    /// Clear the seen hashes cache
    pub fn clear_cache(&mut self) {
        self.seen_hashes.clear();
    }
}

impl Default for NewsCollector {
    fn default() -> Self {
        Self::new()
    }
}

/// Errors that can occur during news collection
#[derive(Debug, thiserror::Error)]
pub enum NewsCollectionError {
    #[error("Failed to connect to news source: {0}")]
    ConnectionError(String),

    #[error("Failed to parse news data: {0}")]
    ParseError(String),

    #[error("Rate limit exceeded")]
    RateLimitExceeded,

    #[error("Authentication failed: {0}")]
    AuthenticationError(String),

    #[error("Unknown error: {0}")]
    Unknown(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_news_collector_creation() {
        let collector = NewsCollector::new();
        assert!(collector.sources.is_empty());
        assert!(collector.seen_hashes.is_empty());
    }

    #[test]
    fn test_default_implementation() {
        let collector = NewsCollector::default();
        assert!(collector.sources.is_empty());
    }
}
