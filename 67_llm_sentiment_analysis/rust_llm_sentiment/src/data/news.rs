//! # News Data Module
//!
//! News article structures and mock news collection for sentiment analysis.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// News source type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NewsSource {
    /// Twitter/X posts
    Twitter,
    /// Reddit posts and comments
    Reddit,
    /// Telegram messages
    Telegram,
    /// Discord messages
    Discord,
    /// News articles
    News,
    /// StockTwits posts
    StockTwits,
    /// Unknown source
    Unknown,
}

impl NewsSource {
    /// Get source weight for sentiment aggregation
    pub fn weight(&self) -> f64 {
        match self {
            NewsSource::News => 1.2,
            NewsSource::Twitter => 1.0,
            NewsSource::Reddit => 0.9,
            NewsSource::Telegram => 0.7,
            NewsSource::Discord => 0.6,
            NewsSource::StockTwits => 0.5,
            NewsSource::Unknown => 0.5,
        }
    }

    /// Get source name
    pub fn name(&self) -> &'static str {
        match self {
            NewsSource::Twitter => "twitter",
            NewsSource::Reddit => "reddit",
            NewsSource::Telegram => "telegram",
            NewsSource::Discord => "discord",
            NewsSource::News => "news",
            NewsSource::StockTwits => "stocktwits",
            NewsSource::Unknown => "unknown",
        }
    }
}

/// News article or social media post
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewsArticle {
    /// Unique identifier
    pub id: String,
    /// Article title or post summary
    pub title: String,
    /// Full text content
    pub content: String,
    /// Data source
    pub source: NewsSource,
    /// Publication timestamp
    pub timestamp: DateTime<Utc>,
    /// URL to original content
    pub url: Option<String>,
    /// Author/username
    pub author: Option<String>,
    /// Related symbols/tickers
    pub symbols: Vec<String>,
    /// Engagement metrics (likes, retweets, etc.)
    pub engagement: Option<u64>,
}

impl NewsArticle {
    /// Create a new news article
    pub fn new(
        id: impl Into<String>,
        title: impl Into<String>,
        content: impl Into<String>,
        source: NewsSource,
    ) -> Self {
        Self {
            id: id.into(),
            title: title.into(),
            content: content.into(),
            source,
            timestamp: Utc::now(),
            url: None,
            author: None,
            symbols: Vec::new(),
            engagement: None,
        }
    }

    /// Set timestamp
    pub fn with_timestamp(mut self, timestamp: DateTime<Utc>) -> Self {
        self.timestamp = timestamp;
        self
    }

    /// Set URL
    pub fn with_url(mut self, url: impl Into<String>) -> Self {
        self.url = Some(url.into());
        self
    }

    /// Set author
    pub fn with_author(mut self, author: impl Into<String>) -> Self {
        self.author = Some(author.into());
        self
    }

    /// Set symbols
    pub fn with_symbols(mut self, symbols: Vec<String>) -> Self {
        self.symbols = symbols;
        self
    }

    /// Set engagement
    pub fn with_engagement(mut self, engagement: u64) -> Self {
        self.engagement = Some(engagement);
        self
    }

    /// Get full text for analysis (title + content)
    pub fn full_text(&self) -> String {
        if self.title.is_empty() {
            self.content.clone()
        } else if self.content.is_empty() {
            self.title.clone()
        } else {
            format!("{} {}", self.title, self.content)
        }
    }
}

/// News collector for gathering articles from various sources
///
/// Note: In production, this would connect to actual news APIs.
/// For demonstration, it provides mock data.
pub struct NewsCollector {
    /// Supported symbols
    symbols: Vec<String>,
}

impl Default for NewsCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl NewsCollector {
    /// Create a new news collector
    pub fn new() -> Self {
        Self {
            symbols: vec![
                "BTC".to_string(),
                "ETH".to_string(),
                "SOL".to_string(),
                "DOGE".to_string(),
            ],
        }
    }

    /// Get mock crypto news for demonstration
    ///
    /// In production, this would fetch from actual APIs like:
    /// - CoinDesk API
    /// - CryptoCompare News API
    /// - Twitter API
    /// - Reddit API
    pub fn get_mock_crypto_news(&self, symbol: &str) -> Vec<NewsArticle> {
        let symbol_upper = symbol.to_uppercase();

        // Mock news data for demonstration
        vec![
            NewsArticle::new(
                "1",
                format!("{} surges 10% as institutional adoption accelerates", symbol_upper),
                format!(
                    "Major banks announce {} custody services, driving institutional demand. \
                     Analysts predict continued growth as regulatory clarity improves.",
                    symbol_upper
                ),
                NewsSource::News,
            )
            .with_symbols(vec![symbol_upper.clone()]),

            NewsArticle::new(
                "2",
                format!("Breaking: {} hits new monthly high!", symbol_upper),
                format!(
                    "${} breaks key resistance level at $50k. Bulls are back! #crypto #moon",
                    symbol_upper
                ),
                NewsSource::Twitter,
            )
            .with_symbols(vec![symbol_upper.clone()])
            .with_engagement(15000),

            NewsArticle::new(
                "3",
                format!("Whale alert: Large {} accumulation detected", symbol_upper),
                format!(
                    "On-chain data shows whales accumulating {} during the dip. \
                     Diamond hands holding strong. WAGMI!",
                    symbol_upper
                ),
                NewsSource::Reddit,
            )
            .with_symbols(vec![symbol_upper.clone()])
            .with_engagement(5000),

            NewsArticle::new(
                "4",
                format!("{} faces regulatory concerns", symbol_upper),
                format!(
                    "SEC announces investigation into {} exchanges. \
                     Market reacts with caution. FUD spreading on social media.",
                    symbol_upper
                ),
                NewsSource::News,
            )
            .with_symbols(vec![symbol_upper.clone()]),

            NewsArticle::new(
                "5",
                format!("Warning: {} leverage positions at risk", symbol_upper),
                format!(
                    "{} high leverage longs at risk of liquidation. \
                     Bearish sentiment growing. Paper hands getting rekt!",
                    symbol_upper
                ),
                NewsSource::Telegram,
            )
            .with_symbols(vec![symbol_upper.clone()])
            .with_engagement(2000),

            NewsArticle::new(
                "6",
                format!("{} technical analysis: Key levels to watch", symbol_upper),
                format!(
                    "{} testing support at $45k. RSI neutral at 50. \
                     Volume declining, suggesting consolidation phase.",
                    symbol_upper
                ),
                NewsSource::StockTwits,
            )
            .with_symbols(vec![symbol_upper.clone()])
            .with_engagement(800),

            NewsArticle::new(
                "7",
                format!("{} network upgrade successful", symbol_upper),
                format!(
                    "{} major network upgrade completed! Transaction speeds improved 5x. \
                     Bullish for long-term adoption. HODL!",
                    symbol_upper
                ),
                NewsSource::Twitter,
            )
            .with_symbols(vec![symbol_upper.clone()])
            .with_engagement(20000),

            NewsArticle::new(
                "8",
                format!("Market analysis: {} outlook for Q1", symbol_upper),
                format!(
                    "{} expected to consolidate before next leg up. \
                     DCA remains the safest strategy. Not financial advice.",
                    symbol_upper
                ),
                NewsSource::News,
            )
            .with_symbols(vec![symbol_upper]),
        ]
    }

    /// Get supported symbols
    pub fn supported_symbols(&self) -> &[String] {
        &self.symbols
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_news_source_weight() {
        assert_eq!(NewsSource::News.weight(), 1.2);
        assert_eq!(NewsSource::Twitter.weight(), 1.0);
        assert!(NewsSource::Discord.weight() < NewsSource::Reddit.weight());
    }

    #[test]
    fn test_news_article_creation() {
        let article = NewsArticle::new(
            "test-1",
            "Test Title",
            "Test content",
            NewsSource::Twitter,
        )
        .with_engagement(1000);

        assert_eq!(article.id, "test-1");
        assert_eq!(article.engagement, Some(1000));
    }

    #[test]
    fn test_full_text() {
        let article = NewsArticle::new(
            "1",
            "Title",
            "Content",
            NewsSource::News,
        );
        assert_eq!(article.full_text(), "Title Content");
    }

    #[test]
    fn test_mock_news_collection() {
        let collector = NewsCollector::new();
        let news = collector.get_mock_crypto_news("BTC");
        assert!(!news.is_empty());
        assert!(news.iter().all(|n| n.symbols.contains(&"BTC".to_string())));
    }
}
