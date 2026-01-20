//! News source connectors for various platforms

use super::types::{NewsItem, NewsSource, RawNews};
use super::NewsCollectionError;
use async_trait::async_trait;
use chrono::{DateTime, Utc};

/// Trait for news source connectors
#[async_trait]
pub trait NewsSourceConnector: Send + Sync + std::fmt::Debug {
    /// Fetch news items since a given timestamp
    async fn fetch_news(&self, since: DateTime<Utc>) -> Result<Vec<NewsItem>, NewsCollectionError>;

    /// Get the source type
    fn source_type(&self) -> NewsSource;

    /// Check if the connector is healthy
    async fn health_check(&self) -> bool;
}

/// Twitter/X news source connector
#[derive(Debug)]
pub struct TwitterSource {
    /// API bearer token
    bearer_token: String,
    /// List of accounts to follow
    accounts: Vec<String>,
    /// Base API URL
    base_url: String,
}

impl TwitterSource {
    /// Create a new Twitter source
    pub fn new(bearer_token: String, accounts: Vec<String>) -> Self {
        Self {
            bearer_token,
            accounts,
            base_url: "https://api.twitter.com/2".to_string(),
        }
    }

    /// Add an account to follow
    pub fn add_account(&mut self, account: String) {
        if !self.accounts.contains(&account) {
            self.accounts.push(account);
        }
    }
}

#[async_trait]
impl NewsSourceConnector for TwitterSource {
    async fn fetch_news(&self, since: DateTime<Utc>) -> Result<Vec<NewsItem>, NewsCollectionError> {
        // In production, this would make actual API calls
        // For now, return empty vec as placeholder
        tracing::debug!(
            "Fetching Twitter news since {} for {} accounts",
            since,
            self.accounts.len()
        );

        // Placeholder implementation
        Ok(Vec::new())
    }

    fn source_type(&self) -> NewsSource {
        NewsSource::Twitter
    }

    async fn health_check(&self) -> bool {
        !self.bearer_token.is_empty()
    }
}

/// RSS feed news source connector
#[derive(Debug)]
pub struct RssSource {
    /// List of RSS feed URLs
    feeds: Vec<String>,
    /// HTTP client
    client: reqwest::Client,
}

impl RssSource {
    /// Create a new RSS source
    pub fn new(feeds: Vec<String>) -> Self {
        Self {
            feeds,
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .build()
                .unwrap_or_default(),
        }
    }

    /// Add a feed URL
    pub fn add_feed(&mut self, url: String) {
        if !self.feeds.contains(&url) {
            self.feeds.push(url);
        }
    }

    /// Parse RSS XML into news items
    fn parse_rss(&self, _xml: &str, source: NewsSource) -> Vec<RawNews> {
        // Simplified RSS parsing - in production would use proper XML parser
        Vec::new()
    }
}

#[async_trait]
impl NewsSourceConnector for RssSource {
    async fn fetch_news(&self, since: DateTime<Utc>) -> Result<Vec<NewsItem>, NewsCollectionError> {
        let mut all_news = Vec::new();

        for feed_url in &self.feeds {
            match self.client.get(feed_url).send().await {
                Ok(response) => {
                    if let Ok(text) = response.text().await {
                        let source = if feed_url.contains("coindesk") {
                            NewsSource::CoinDesk
                        } else if feed_url.contains("cointelegraph") {
                            NewsSource::CoinTelegraph
                        } else {
                            NewsSource::Rss
                        };

                        let raw_items = self.parse_rss(&text, source);

                        for raw in raw_items {
                            if raw.published_at > since {
                                let item = NewsItem::from_raw(
                                    raw.clone(),
                                    raw.content.clone(),
                                    Vec::new(),
                                );
                                all_news.push(item);
                            }
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!("Failed to fetch RSS feed {}: {}", feed_url, e);
                }
            }
        }

        Ok(all_news)
    }

    fn source_type(&self) -> NewsSource {
        NewsSource::Rss
    }

    async fn health_check(&self) -> bool {
        !self.feeds.is_empty()
    }
}

/// Telegram channel news source connector
#[derive(Debug)]
pub struct TelegramSource {
    /// Bot API token
    bot_token: String,
    /// List of channel usernames or IDs
    channels: Vec<String>,
}

impl TelegramSource {
    /// Create a new Telegram source
    pub fn new(bot_token: String, channels: Vec<String>) -> Self {
        Self { bot_token, channels }
    }
}

#[async_trait]
impl NewsSourceConnector for TelegramSource {
    async fn fetch_news(&self, since: DateTime<Utc>) -> Result<Vec<NewsItem>, NewsCollectionError> {
        // Placeholder implementation
        tracing::debug!(
            "Fetching Telegram news since {} from {} channels",
            since,
            self.channels.len()
        );
        Ok(Vec::new())
    }

    fn source_type(&self) -> NewsSource {
        NewsSource::Telegram
    }

    async fn health_check(&self) -> bool {
        !self.bot_token.is_empty()
    }
}

/// Mock news source for testing
#[derive(Debug)]
pub struct MockNewsSource {
    items: Vec<NewsItem>,
    source: NewsSource,
}

impl MockNewsSource {
    /// Create a new mock source with predefined items
    pub fn new(items: Vec<NewsItem>, source: NewsSource) -> Self {
        Self { items, source }
    }

    /// Create an empty mock source
    pub fn empty(source: NewsSource) -> Self {
        Self {
            items: Vec::new(),
            source,
        }
    }
}

#[async_trait]
impl NewsSourceConnector for MockNewsSource {
    async fn fetch_news(&self, since: DateTime<Utc>) -> Result<Vec<NewsItem>, NewsCollectionError> {
        Ok(self
            .items
            .iter()
            .filter(|item| item.published_at > since)
            .cloned()
            .collect())
    }

    fn source_type(&self) -> NewsSource {
        self.source
    }

    async fn health_check(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_twitter_source_creation() {
        let source = TwitterSource::new(
            "test_token".to_string(),
            vec!["@bitcoin".to_string(), "@ethereum".to_string()],
        );
        assert_eq!(source.source_type(), NewsSource::Twitter);
        assert_eq!(source.accounts.len(), 2);
    }

    #[test]
    fn test_rss_source_creation() {
        let source = RssSource::new(vec!["https://example.com/rss".to_string()]);
        assert_eq!(source.source_type(), NewsSource::Rss);
        assert_eq!(source.feeds.len(), 1);
    }

    #[tokio::test]
    async fn test_mock_source() {
        let mock = MockNewsSource::empty(NewsSource::Twitter);
        let news = mock.fetch_news(Utc::now()).await.unwrap();
        assert!(news.is_empty());
        assert!(mock.health_check().await);
    }
}
