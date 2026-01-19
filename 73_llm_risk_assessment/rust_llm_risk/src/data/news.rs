//! News data structures and processing.

use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

/// Source of the news item.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum NewsSource {
    /// Twitter/X posts
    Twitter,
    /// Reddit posts/comments
    Reddit,
    /// CoinDesk articles
    CoinDesk,
    /// CoinTelegraph articles
    CoinTelegraph,
    /// Bloomberg articles
    Bloomberg,
    /// Reuters articles
    Reuters,
    /// SEC filings
    SEC,
    /// Company press releases
    PressRelease,
    /// Other/unknown source
    Other(String),
}

impl Default for NewsSource {
    fn default() -> Self {
        NewsSource::Other("unknown".to_string())
    }
}

/// A news item for risk analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewsItem {
    /// Unique identifier for the news item.
    pub id: String,
    /// Title or headline.
    pub title: String,
    /// Full content/body text.
    pub content: String,
    /// Source of the news.
    pub source: NewsSource,
    /// Publication timestamp.
    pub published_at: DateTime<Utc>,
    /// Related symbols (e.g., ["BTC", "ETH"]).
    pub symbols: Vec<String>,
    /// URL to the original source.
    pub url: Option<String>,
    /// Author name if available.
    pub author: Option<String>,
}

impl NewsItem {
    /// Create a new news item.
    pub fn new(
        id: String,
        title: String,
        content: String,
        source: NewsSource,
        published_at: DateTime<Utc>,
    ) -> Self {
        Self {
            id,
            title,
            content,
            source,
            published_at,
            symbols: Vec::new(),
            url: None,
            author: None,
        }
    }

    /// Add related symbols to the news item.
    pub fn with_symbols(mut self, symbols: Vec<String>) -> Self {
        self.symbols = symbols;
        self
    }

    /// Add URL to the news item.
    pub fn with_url(mut self, url: String) -> Self {
        self.url = Some(url);
        self
    }

    /// Add author to the news item.
    pub fn with_author(mut self, author: String) -> Self {
        self.author = Some(author);
        self
    }

    /// Get the combined text for analysis (title + content).
    pub fn full_text(&self) -> String {
        format!("{}\n\n{}", self.title, self.content)
    }

    /// Get word count of the content.
    pub fn word_count(&self) -> usize {
        self.content.split_whitespace().count()
    }

    /// Check if the news item mentions a specific symbol.
    pub fn mentions_symbol(&self, symbol: &str) -> bool {
        let symbol_upper = symbol.to_uppercase();
        self.symbols.iter().any(|s| s.to_uppercase() == symbol_upper)
            || self.title.to_uppercase().contains(&symbol_upper)
            || self.content.to_uppercase().contains(&symbol_upper)
    }

    /// Extract mentioned cryptocurrencies from text.
    pub fn extract_crypto_mentions(&self) -> Vec<String> {
        let text = self.full_text().to_uppercase();
        let mut mentions = Vec::new();

        // Common cryptocurrency names and symbols
        let cryptos = [
            ("BTC", "BITCOIN"),
            ("ETH", "ETHEREUM"),
            ("SOL", "SOLANA"),
            ("BNB", "BINANCE"),
            ("XRP", "RIPPLE"),
            ("ADA", "CARDANO"),
            ("AVAX", "AVALANCHE"),
            ("DOT", "POLKADOT"),
            ("MATIC", "POLYGON"),
            ("DOGE", "DOGECOIN"),
            ("SHIB", "SHIBA"),
            ("LTC", "LITECOIN"),
            ("LINK", "CHAINLINK"),
            ("UNI", "UNISWAP"),
            ("ATOM", "COSMOS"),
        ];

        for (symbol, name) in cryptos.iter() {
            if text.contains(symbol) || text.contains(name) {
                mentions.push(symbol.to_string());
            }
        }

        mentions.sort();
        mentions.dedup();
        mentions
    }
}

/// Collection of news items.
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub struct NewsFeed {
    items: Vec<NewsItem>,
}

#[allow(dead_code)]
impl NewsFeed {
    /// Create a new empty news feed.
    pub fn new() -> Self {
        Self { items: Vec::new() }
    }

    /// Add a news item to the feed.
    pub fn add(&mut self, item: NewsItem) {
        self.items.push(item);
    }

    /// Get all items.
    pub fn items(&self) -> &[NewsItem] {
        &self.items
    }

    /// Get number of items.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Check if the feed is empty.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Filter items by symbol.
    pub fn filter_by_symbol(&self, symbol: &str) -> Vec<&NewsItem> {
        self.items
            .iter()
            .filter(|item| item.mentions_symbol(symbol))
            .collect()
    }

    /// Filter items by source.
    pub fn filter_by_source(&self, source: &NewsSource) -> Vec<&NewsItem> {
        self.items
            .iter()
            .filter(|item| &item.source == source)
            .collect()
    }

    /// Get items within a time range.
    pub fn filter_by_time(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Vec<&NewsItem> {
        self.items
            .iter()
            .filter(|item| item.published_at >= start && item.published_at <= end)
            .collect()
    }

    /// Sort items by publication time (newest first).
    pub fn sort_by_time(&mut self) {
        self.items.sort_by(|a, b| b.published_at.cmp(&a.published_at));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_news_item_creation() {
        let news = NewsItem::new(
            "1".to_string(),
            "Bitcoin reaches new high".to_string(),
            "Bitcoin price surged to $100,000 today.".to_string(),
            NewsSource::CoinDesk,
            Utc::now(),
        )
        .with_symbols(vec!["BTC".to_string()])
        .with_url("https://coindesk.com/article".to_string());

        assert!(news.mentions_symbol("BTC"));
        assert!(news.mentions_symbol("bitcoin"));
        assert!(!news.mentions_symbol("ETH"));
    }

    #[test]
    fn test_extract_crypto_mentions() {
        let news = NewsItem::new(
            "1".to_string(),
            "Market update".to_string(),
            "Bitcoin and Ethereum are both up today. SOL also showing strength.".to_string(),
            NewsSource::Twitter,
            Utc::now(),
        );

        let mentions = news.extract_crypto_mentions();
        assert!(mentions.contains(&"BTC".to_string()));
        assert!(mentions.contains(&"ETH".to_string()));
        assert!(mentions.contains(&"SOL".to_string()));
    }

    #[test]
    fn test_news_feed() {
        let mut feed = NewsFeed::new();

        feed.add(NewsItem::new(
            "1".to_string(),
            "BTC news".to_string(),
            "Bitcoin content".to_string(),
            NewsSource::Twitter,
            Utc::now(),
        ));

        feed.add(NewsItem::new(
            "2".to_string(),
            "ETH news".to_string(),
            "Ethereum content".to_string(),
            NewsSource::Reddit,
            Utc::now(),
        ));

        assert_eq!(feed.len(), 2);
        assert_eq!(feed.filter_by_symbol("BTC").len(), 1);
        assert_eq!(feed.filter_by_source(&NewsSource::Twitter).len(), 1);
    }
}
