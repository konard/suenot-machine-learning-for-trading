//! News data types and structures

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Enumeration of supported news sources
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NewsSource {
    /// Twitter/X platform
    Twitter,
    /// Reddit forums
    Reddit,
    /// Telegram channels
    Telegram,
    /// CoinDesk news
    CoinDesk,
    /// CoinTelegraph news
    CoinTelegraph,
    /// The Block news
    TheBlock,
    /// Bloomberg
    Bloomberg,
    /// Reuters
    Reuters,
    /// Official project blogs
    OfficialBlog,
    /// GitHub releases and commits
    GitHub,
    /// On-chain data with context
    OnChain,
    /// Generic RSS feed
    Rss,
    /// Other specific source
    Other,
    /// Unknown source
    Unknown,
}

impl NewsSource {
    /// Get the reliability score of this news source (0.0 to 1.0)
    pub fn reliability_score(&self) -> f64 {
        match self {
            NewsSource::OfficialBlog => 0.95,
            NewsSource::GitHub => 0.95,
            NewsSource::Bloomberg => 0.90,
            NewsSource::Reuters => 0.90,
            NewsSource::CoinDesk => 0.80,
            NewsSource::CoinTelegraph => 0.75,
            NewsSource::TheBlock => 0.80,
            NewsSource::Twitter => 0.50,
            NewsSource::Reddit => 0.40,
            NewsSource::Telegram => 0.45,
            NewsSource::OnChain => 0.85,
            NewsSource::Rss => 0.60,
            NewsSource::Other => 0.50,
            NewsSource::Unknown => 0.30,
        }
    }

    /// Get the typical latency for this source (milliseconds)
    pub fn typical_latency_ms(&self) -> u64 {
        match self {
            NewsSource::Twitter => 500,
            NewsSource::Telegram => 1000,
            NewsSource::OnChain => 2000,
            NewsSource::Reddit => 3000,
            NewsSource::CoinDesk => 60000,
            NewsSource::CoinTelegraph => 60000,
            NewsSource::TheBlock => 60000,
            NewsSource::Bloomberg => 30000,
            NewsSource::Reuters => 30000,
            NewsSource::OfficialBlog => 300000,
            NewsSource::GitHub => 600000,
            NewsSource::Rss => 60000,
            NewsSource::Other => 60000,
            NewsSource::Unknown => 120000,
        }
    }
}

impl std::fmt::Display for NewsSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl NewsSource {
    /// Get the name of this news source
    pub fn name(&self) -> &'static str {
        match self {
            NewsSource::Twitter => "Twitter",
            NewsSource::Reddit => "Reddit",
            NewsSource::Telegram => "Telegram",
            NewsSource::CoinDesk => "CoinDesk",
            NewsSource::CoinTelegraph => "CoinTelegraph",
            NewsSource::TheBlock => "The Block",
            NewsSource::Bloomberg => "Bloomberg",
            NewsSource::Reuters => "Reuters",
            NewsSource::OfficialBlog => "Official Blog",
            NewsSource::GitHub => "GitHub",
            NewsSource::OnChain => "On-Chain",
            NewsSource::Rss => "RSS",
            NewsSource::Other => "Other",
            NewsSource::Unknown => "Unknown",
        }
    }
}

/// Raw news data as received from a source
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawNews {
    /// The raw text content
    pub content: String,
    /// Source of the news
    pub source: NewsSource,
    /// Original URL if available
    pub url: Option<String>,
    /// Author or account name
    pub author: Option<String>,
    /// Timestamp when the news was published
    pub published_at: DateTime<Utc>,
    /// Any additional metadata
    pub metadata: Option<serde_json::Value>,
}

/// Processed news item ready for analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewsItem {
    /// Unique identifier for this news item
    pub id: String,
    /// News headline/title
    pub title: String,
    /// Full content of the news
    pub content: String,
    /// Source of the news
    pub source: NewsSource,
    /// URL if available
    pub url: Option<String>,
    /// When the news was published
    pub published_at: DateTime<Utc>,
    /// Affected cryptocurrency symbols
    pub symbols: Vec<String>,
    /// Relevance score (0.0 to 1.0)
    pub relevance_score: f64,
}

impl NewsItem {
    /// Create a new news item from raw news
    pub fn from_raw(raw: RawNews, cleaned_text: String, entities: Vec<String>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            title: cleaned_text.chars().take(100).collect(),
            content: cleaned_text,
            source: raw.source,
            url: raw.url,
            published_at: raw.published_at,
            symbols: entities,
            relevance_score: 0.5, // Default relevance
        }
    }

    /// Create a new news item with all fields
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
            url: None,
            published_at,
            symbols: vec![],
            relevance_score: 0.5,
        }
    }

    /// Calculate the age of this news item in seconds
    pub fn age_seconds(&self) -> i64 {
        (Utc::now() - self.published_at).num_seconds()
    }

    /// Check if this news is still fresh (less than threshold seconds old)
    pub fn is_fresh(&self, threshold_seconds: i64) -> bool {
        self.age_seconds() < threshold_seconds
    }

    /// Get the recency weight for this news (decays over time)
    pub fn recency_weight(&self, half_life_seconds: f64) -> f64 {
        let age = self.age_seconds() as f64;
        (-age / half_life_seconds).exp()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_news_source_reliability() {
        assert!(NewsSource::OfficialBlog.reliability_score() > NewsSource::Twitter.reliability_score());
        assert!(NewsSource::Bloomberg.reliability_score() > NewsSource::Reddit.reliability_score());
    }

    #[test]
    fn test_news_source_display() {
        assert_eq!(format!("{}", NewsSource::Twitter), "Twitter");
        assert_eq!(format!("{}", NewsSource::CoinDesk), "CoinDesk");
    }

    #[test]
    fn test_news_source_name() {
        assert_eq!(NewsSource::Twitter.name(), "Twitter");
        assert_eq!(NewsSource::Bloomberg.name(), "Bloomberg");
    }

    #[test]
    fn test_raw_news_creation() {
        let raw = RawNews {
            content: "Test news content".to_string(),
            source: NewsSource::Twitter,
            url: Some("https://example.com".to_string()),
            author: Some("test_user".to_string()),
            published_at: Utc::now(),
            metadata: None,
        };

        assert_eq!(raw.source, NewsSource::Twitter);
    }

    #[test]
    fn test_news_item_from_raw() {
        let raw = RawNews {
            content: "Bitcoin price surges".to_string(),
            source: NewsSource::CoinDesk,
            url: None,
            author: None,
            published_at: Utc::now(),
            metadata: None,
        };

        let item = NewsItem::from_raw(
            raw,
            "Bitcoin price surges".to_string(),
            vec!["BTC".to_string()],
        );

        assert!(!item.id.is_empty());
        assert_eq!(item.symbols, vec!["BTC"]);
    }

    #[test]
    fn test_news_item_new() {
        let item = NewsItem::new(
            "test-id".to_string(),
            "Bitcoin ETF Approved".to_string(),
            "The SEC has approved the first Bitcoin ETF".to_string(),
            NewsSource::Bloomberg,
            Utc::now(),
        );

        assert_eq!(item.id, "test-id");
        assert_eq!(item.title, "Bitcoin ETF Approved");
        assert_eq!(item.source, NewsSource::Bloomberg);
    }

    #[test]
    fn test_news_freshness() {
        let item = NewsItem::new(
            "test".to_string(),
            "Test".to_string(),
            "Test content".to_string(),
            NewsSource::Twitter,
            Utc::now(),
        );

        assert!(item.is_fresh(60)); // Fresh within 60 seconds
        assert!(item.age_seconds() < 5); // Should be very recent
    }
}
