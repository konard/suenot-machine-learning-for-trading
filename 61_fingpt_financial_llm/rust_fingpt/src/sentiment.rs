//! Financial sentiment analysis module
//!
//! This module provides sentiment analysis capabilities inspired by FinGPT,
//! supporting both mock analysis for demos and real LLM-based analysis.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Sentiment classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Sentiment {
    /// Positive sentiment (bullish)
    Positive,
    /// Negative sentiment (bearish)
    Negative,
    /// Neutral sentiment
    Neutral,
}

impl std::fmt::Display for Sentiment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Sentiment::Positive => write!(f, "positive"),
            Sentiment::Negative => write!(f, "negative"),
            Sentiment::Neutral => write!(f, "neutral"),
        }
    }
}

/// Result of sentiment analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentimentResult {
    /// The classified sentiment
    pub sentiment: Sentiment,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,
    /// Raw sentiment score (-1.0 to 1.0)
    pub score: f64,
    /// Analysis metadata
    pub metadata: HashMap<String, String>,
}

impl SentimentResult {
    /// Create a new sentiment result
    pub fn new(sentiment: Sentiment, confidence: f64, score: f64) -> Self {
        Self {
            sentiment,
            confidence,
            score,
            metadata: HashMap::new(),
        }
    }

    /// Add metadata to the result
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// Financial sentiment analyzer using FinGPT-inspired approach
#[derive(Clone)]
pub struct SentimentAnalyzer {
    use_mock: bool,
    positive_keywords: Vec<&'static str>,
    negative_keywords: Vec<&'static str>,
}

impl SentimentAnalyzer {
    /// Create a new mock sentiment analyzer for demos
    pub fn new_mock() -> Self {
        Self {
            use_mock: true,
            positive_keywords: vec![
                "record", "surge", "beat", "exceed", "growth", "profit", "gain",
                "bullish", "rally", "soar", "jump", "rise", "high", "strong",
                "positive", "upgrade", "buy", "outperform", "success", "boom",
                "breakthrough", "milestone", "innovation", "expansion", "demand",
            ],
            negative_keywords: vec![
                "drop", "fall", "decline", "loss", "miss", "below", "concern",
                "bearish", "plunge", "crash", "sink", "weak", "negative", "sell",
                "downgrade", "layoff", "cut", "warning", "risk", "debt", "default",
                "lawsuit", "investigation", "recall", "bankruptcy", "fraud",
            ],
        }
    }

    /// Create an analyzer with real LLM backend (placeholder for future implementation)
    pub fn new_with_model(_model_path: &str) -> Result<Self> {
        // In a real implementation, this would load a FinGPT model
        Err(Error::Model("Real model loading not implemented. Use new_mock() for demos.".into()))
    }

    /// Analyze sentiment of a financial text
    pub async fn analyze(&self, text: &str) -> Result<SentimentResult> {
        if self.use_mock {
            Ok(self.mock_analyze(text))
        } else {
            Err(Error::Model("Real model inference not implemented".into()))
        }
    }

    /// Mock sentiment analysis based on keyword matching
    fn mock_analyze(&self, text: &str) -> SentimentResult {
        let text_lower = text.to_lowercase();

        let positive_count = self.positive_keywords
            .iter()
            .filter(|kw| text_lower.contains(*kw))
            .count();

        let negative_count = self.negative_keywords
            .iter()
            .filter(|kw| text_lower.contains(*kw))
            .count();

        let total = positive_count + negative_count;

        let (sentiment, score, confidence) = if total == 0 {
            (Sentiment::Neutral, 0.0, 0.5)
        } else {
            let pos_ratio = positive_count as f64 / total as f64;
            let neg_ratio = negative_count as f64 / total as f64;

            if pos_ratio > neg_ratio {
                let score = pos_ratio - neg_ratio;
                let confidence = 0.6 + (score * 0.3);
                (Sentiment::Positive, score, confidence.min(0.95))
            } else if neg_ratio > pos_ratio {
                let score = neg_ratio - pos_ratio;
                let confidence = 0.6 + (score * 0.3);
                (Sentiment::Negative, -score, confidence.min(0.95))
            } else {
                (Sentiment::Neutral, 0.0, 0.6)
            }
        };

        SentimentResult::new(sentiment, confidence, score)
            .with_metadata("method", "keyword_matching")
            .with_metadata("positive_matches", positive_count.to_string())
            .with_metadata("negative_matches", negative_count.to_string())
    }

    /// Analyze sentiment for multiple entities/aspects in a text
    pub async fn analyze_aspects(
        &self,
        text: &str,
        entities: &[&str],
    ) -> Result<HashMap<String, SentimentResult>> {
        let mut results = HashMap::new();

        // Split text into sentences for aspect analysis
        let sentences: Vec<&str> = text.split(|c| c == '.' || c == '!' || c == '?')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect();

        for entity in entities {
            // Find sentences mentioning this entity
            let relevant_text: String = sentences
                .iter()
                .filter(|s| s.to_lowercase().contains(&entity.to_lowercase()))
                .copied()
                .collect::<Vec<_>>()
                .join(". ");

            let result = if relevant_text.is_empty() {
                SentimentResult::new(Sentiment::Neutral, 0.5, 0.0)
                    .with_metadata("note", "No relevant text found for entity")
            } else {
                self.analyze(&relevant_text).await?
            };

            results.insert(entity.to_string(), result);
        }

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_positive_sentiment() {
        let analyzer = SentimentAnalyzer::new_mock();
        let result = analyzer.analyze("Apple reports record quarterly revenue, beating expectations").await.unwrap();
        assert_eq!(result.sentiment, Sentiment::Positive);
        assert!(result.score > 0.0);
    }

    #[tokio::test]
    async fn test_negative_sentiment() {
        let analyzer = SentimentAnalyzer::new_mock();
        let result = analyzer.analyze("Stock prices crash amid concerns about debt default").await.unwrap();
        assert_eq!(result.sentiment, Sentiment::Negative);
        assert!(result.score < 0.0);
    }

    #[tokio::test]
    async fn test_neutral_sentiment() {
        let analyzer = SentimentAnalyzer::new_mock();
        let result = analyzer.analyze("The company scheduled a meeting for next week").await.unwrap();
        assert_eq!(result.sentiment, Sentiment::Neutral);
    }
}
