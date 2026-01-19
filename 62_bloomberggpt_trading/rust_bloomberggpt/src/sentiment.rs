//! Financial Sentiment Analysis Module
//!
//! Provides sentiment analysis for financial texts using LLM APIs.
//! Supports multiple backends: OpenAI, local models, and mock for testing.

use crate::error::{Error, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Sentiment classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Sentiment {
    /// Very negative sentiment
    VeryNegative,
    /// Negative sentiment
    Negative,
    /// Neutral sentiment
    Neutral,
    /// Positive sentiment
    Positive,
    /// Very positive sentiment
    VeryPositive,
}

impl Sentiment {
    /// Convert sentiment to numeric score (-1.0 to 1.0)
    pub fn to_score(&self) -> f64 {
        match self {
            Sentiment::VeryNegative => -1.0,
            Sentiment::Negative => -0.5,
            Sentiment::Neutral => 0.0,
            Sentiment::Positive => 0.5,
            Sentiment::VeryPositive => 1.0,
        }
    }

    /// Create sentiment from numeric score
    pub fn from_score(score: f64) -> Self {
        if score <= -0.6 {
            Sentiment::VeryNegative
        } else if score <= -0.2 {
            Sentiment::Negative
        } else if score <= 0.2 {
            Sentiment::Neutral
        } else if score <= 0.6 {
            Sentiment::Positive
        } else {
            Sentiment::VeryPositive
        }
    }
}

impl std::fmt::Display for Sentiment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Sentiment::VeryNegative => write!(f, "Very Negative"),
            Sentiment::Negative => write!(f, "Negative"),
            Sentiment::Neutral => write!(f, "Neutral"),
            Sentiment::Positive => write!(f, "Positive"),
            Sentiment::VeryPositive => write!(f, "Very Positive"),
        }
    }
}

/// Result of sentiment analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentimentResult {
    /// The analyzed text
    pub text: String,
    /// Overall sentiment classification
    pub sentiment: Sentiment,
    /// Sentiment score (-1.0 to 1.0)
    pub score: f64,
    /// Confidence in the prediction (0.0 to 1.0)
    pub confidence: f64,
    /// Entity-specific sentiments (if detected)
    pub entity_sentiments: HashMap<String, EntitySentiment>,
    /// Source of analysis (model name)
    pub source: String,
}

/// Entity-specific sentiment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntitySentiment {
    /// Entity name (e.g., "AAPL", "Tim Cook")
    pub entity: String,
    /// Entity type
    pub entity_type: EntityType,
    /// Sentiment for this entity
    pub sentiment: Sentiment,
    /// Sentiment score
    pub score: f64,
}

/// Types of entities that can be detected
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EntityType {
    /// Stock ticker
    Ticker,
    /// Company name
    Company,
    /// Person name
    Person,
    /// Sector/industry
    Sector,
    /// Geographic location
    Location,
    /// Cryptocurrency
    Crypto,
    /// Other entity
    Other,
}

/// Trait for sentiment analyzers
#[async_trait]
pub trait SentimentAnalyzerTrait: Send + Sync {
    /// Analyze sentiment of a single text
    async fn analyze(&self, text: &str) -> Result<SentimentResult>;

    /// Analyze sentiment of multiple texts in batch
    async fn analyze_batch(&self, texts: &[String]) -> Result<Vec<SentimentResult>>;

    /// Get the analyzer name/model
    fn name(&self) -> &str;
}

/// Main sentiment analyzer that can use different backends
pub struct SentimentAnalyzer {
    backend: Box<dyn SentimentAnalyzerTrait>,
}

impl SentimentAnalyzer {
    /// Create analyzer with OpenAI backend
    pub fn new_openai(api_key: &str) -> Self {
        Self {
            backend: Box::new(OpenAISentimentAnalyzer::new(api_key)),
        }
    }

    /// Create analyzer with mock backend (for testing)
    pub fn new_mock() -> Self {
        Self {
            backend: Box::new(MockSentimentAnalyzer::new()),
        }
    }

    /// Analyze sentiment of text
    pub async fn analyze(&self, text: &str) -> Result<SentimentResult> {
        self.backend.analyze(text).await
    }

    /// Analyze multiple texts
    pub async fn analyze_batch(&self, texts: &[String]) -> Result<Vec<SentimentResult>> {
        self.backend.analyze_batch(texts).await
    }

    /// Get backend name
    pub fn name(&self) -> &str {
        self.backend.name()
    }
}

/// OpenAI-based sentiment analyzer
pub struct OpenAISentimentAnalyzer {
    api_key: String,
    client: reqwest::Client,
    model: String,
}

impl OpenAISentimentAnalyzer {
    /// Create new OpenAI analyzer
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            client: reqwest::Client::new(),
            model: "gpt-4o-mini".to_string(),
        }
    }

    /// Set the model to use
    pub fn with_model(mut self, model: &str) -> Self {
        self.model = model.to_string();
        self
    }

    fn build_prompt(&self, text: &str) -> String {
        format!(
            r#"Analyze the financial sentiment of the following text. Respond with a JSON object containing:
- "sentiment": one of "very_negative", "negative", "neutral", "positive", "very_positive"
- "score": a number from -1.0 (very negative) to 1.0 (very positive)
- "confidence": a number from 0.0 to 1.0 indicating confidence
- "entities": array of objects with "entity", "type" (ticker/company/person/sector/crypto/other), "sentiment", "score"

Text to analyze:
"{}"

Respond only with valid JSON, no other text."#,
            text
        )
    }
}

#[async_trait]
impl SentimentAnalyzerTrait for OpenAISentimentAnalyzer {
    async fn analyze(&self, text: &str) -> Result<SentimentResult> {
        let prompt = self.build_prompt(text);

        let request_body = serde_json::json!({
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a financial sentiment analysis expert. Analyze texts for market sentiment and identify relevant entities like stock tickers, companies, and key people."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.1,
            "max_tokens": 500
        });

        let response = self.client
            .post("https://api.openai.com/v1/chat/completions")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(Error::ApiError(format!(
                "OpenAI API error ({}): {}",
                status, error_text
            )));
        }

        let response_json: serde_json::Value = response.json().await?;

        // Parse the response
        let content = response_json["choices"][0]["message"]["content"]
            .as_str()
            .ok_or_else(|| Error::ParseError("No content in response".to_string()))?;

        // Parse the JSON content
        let parsed: serde_json::Value = serde_json::from_str(content)
            .map_err(|e| Error::ParseError(format!("Failed to parse LLM response: {}", e)))?;

        let sentiment_str = parsed["sentiment"]
            .as_str()
            .unwrap_or("neutral");

        let sentiment = match sentiment_str {
            "very_negative" => Sentiment::VeryNegative,
            "negative" => Sentiment::Negative,
            "positive" => Sentiment::Positive,
            "very_positive" => Sentiment::VeryPositive,
            _ => Sentiment::Neutral,
        };

        let score = parsed["score"].as_f64().unwrap_or(0.0);
        let confidence = parsed["confidence"].as_f64().unwrap_or(0.5);

        // Parse entities
        let mut entity_sentiments = HashMap::new();
        if let Some(entities) = parsed["entities"].as_array() {
            for entity in entities {
                let entity_name = entity["entity"].as_str().unwrap_or("").to_string();
                if entity_name.is_empty() {
                    continue;
                }

                let entity_type = match entity["type"].as_str().unwrap_or("other") {
                    "ticker" => EntityType::Ticker,
                    "company" => EntityType::Company,
                    "person" => EntityType::Person,
                    "sector" => EntityType::Sector,
                    "crypto" => EntityType::Crypto,
                    _ => EntityType::Other,
                };

                let entity_sentiment_str = entity["sentiment"].as_str().unwrap_or("neutral");
                let entity_sentiment = match entity_sentiment_str {
                    "very_negative" => Sentiment::VeryNegative,
                    "negative" => Sentiment::Negative,
                    "positive" => Sentiment::Positive,
                    "very_positive" => Sentiment::VeryPositive,
                    _ => Sentiment::Neutral,
                };

                let entity_score = entity["score"].as_f64().unwrap_or(0.0);

                entity_sentiments.insert(
                    entity_name.clone(),
                    EntitySentiment {
                        entity: entity_name,
                        entity_type,
                        sentiment: entity_sentiment,
                        score: entity_score,
                    },
                );
            }
        }

        Ok(SentimentResult {
            text: text.to_string(),
            sentiment,
            score,
            confidence,
            entity_sentiments,
            source: format!("openai/{}", self.model),
        })
    }

    async fn analyze_batch(&self, texts: &[String]) -> Result<Vec<SentimentResult>> {
        // Process in parallel with rate limiting
        let mut results = Vec::with_capacity(texts.len());
        for text in texts {
            results.push(self.analyze(text).await?);
        }
        Ok(results)
    }

    fn name(&self) -> &str {
        "OpenAI"
    }
}

/// Mock sentiment analyzer for testing
pub struct MockSentimentAnalyzer {
    /// Predefined responses for testing
    responses: HashMap<String, SentimentResult>,
}

impl MockSentimentAnalyzer {
    /// Create new mock analyzer
    pub fn new() -> Self {
        Self {
            responses: HashMap::new(),
        }
    }

    /// Add a predefined response
    pub fn add_response(&mut self, text: &str, result: SentimentResult) {
        self.responses.insert(text.to_string(), result);
    }

    /// Simple keyword-based sentiment for demo purposes
    fn analyze_keywords(&self, text: &str) -> (Sentiment, f64) {
        let text_lower = text.to_lowercase();

        // Positive keywords
        let positive_keywords = [
            "beat", "surge", "record", "growth", "profit", "success",
            "breakthrough", "bullish", "rally", "gain", "soar", "exceed",
            "outperform", "strong", "positive", "upgrade", "buy",
        ];

        // Negative keywords
        let negative_keywords = [
            "miss", "drop", "fall", "decline", "loss", "fail", "crash",
            "bearish", "plunge", "cut", "layoff", "weak", "negative",
            "downgrade", "sell", "warning", "concern", "risk",
        ];

        let positive_count: i32 = positive_keywords
            .iter()
            .filter(|k| text_lower.contains(*k))
            .count() as i32;

        let negative_count: i32 = negative_keywords
            .iter()
            .filter(|k| text_lower.contains(*k))
            .count() as i32;

        let net_score = positive_count - negative_count;
        let total = (positive_count + negative_count).max(1) as f64;

        let score = (net_score as f64 / total).clamp(-1.0, 1.0);
        let sentiment = Sentiment::from_score(score);

        (sentiment, score)
    }

    /// Extract entities from text
    fn extract_entities(&self, text: &str) -> HashMap<String, EntitySentiment> {
        let mut entities = HashMap::new();

        // Simple ticker detection (uppercase 1-5 letter words)
        let ticker_regex = regex::Regex::new(r"\b([A-Z]{1,5})\b").ok();
        if let Some(re) = ticker_regex {
            for cap in re.captures_iter(text) {
                let ticker = &cap[1];
                // Filter out common words
                if !["A", "I", "IN", "ON", "AT", "TO", "THE", "AND", "OR", "FOR", "CEO", "CFO", "COO"]
                    .contains(&ticker)
                {
                    let (sentiment, score) = self.analyze_keywords(text);
                    entities.insert(
                        ticker.to_string(),
                        EntitySentiment {
                            entity: ticker.to_string(),
                            entity_type: EntityType::Ticker,
                            sentiment,
                            score,
                        },
                    );
                }
            }
        }

        entities
    }
}

impl Default for MockSentimentAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl SentimentAnalyzerTrait for MockSentimentAnalyzer {
    async fn analyze(&self, text: &str) -> Result<SentimentResult> {
        // Check for predefined response
        if let Some(result) = self.responses.get(text) {
            return Ok(result.clone());
        }

        // Keyword-based analysis
        let (sentiment, score) = self.analyze_keywords(text);
        let entity_sentiments = self.extract_entities(text);

        // Confidence based on keyword matches
        let confidence = if score.abs() > 0.5 { 0.85 } else { 0.65 };

        Ok(SentimentResult {
            text: text.to_string(),
            sentiment,
            score,
            confidence,
            entity_sentiments,
            source: "mock/keyword".to_string(),
        })
    }

    async fn analyze_batch(&self, texts: &[String]) -> Result<Vec<SentimentResult>> {
        let mut results = Vec::with_capacity(texts.len());
        for text in texts {
            results.push(self.analyze(text).await?);
        }
        Ok(results)
    }

    fn name(&self) -> &str {
        "Mock"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sentiment_from_score() {
        assert_eq!(Sentiment::from_score(-0.8), Sentiment::VeryNegative);
        assert_eq!(Sentiment::from_score(-0.4), Sentiment::Negative);
        assert_eq!(Sentiment::from_score(0.0), Sentiment::Neutral);
        assert_eq!(Sentiment::from_score(0.4), Sentiment::Positive);
        assert_eq!(Sentiment::from_score(0.8), Sentiment::VeryPositive);
    }

    #[test]
    fn test_sentiment_to_score() {
        assert_eq!(Sentiment::VeryNegative.to_score(), -1.0);
        assert_eq!(Sentiment::Negative.to_score(), -0.5);
        assert_eq!(Sentiment::Neutral.to_score(), 0.0);
        assert_eq!(Sentiment::Positive.to_score(), 0.5);
        assert_eq!(Sentiment::VeryPositive.to_score(), 1.0);
    }

    #[tokio::test]
    async fn test_mock_analyzer_positive() {
        let analyzer = MockSentimentAnalyzer::new();
        let result = analyzer
            .analyze("Apple reports record quarterly earnings, beating analyst expectations")
            .await
            .unwrap();

        assert!(result.score > 0.0);
        assert!(matches!(
            result.sentiment,
            Sentiment::Positive | Sentiment::VeryPositive
        ));
    }

    #[tokio::test]
    async fn test_mock_analyzer_negative() {
        let analyzer = MockSentimentAnalyzer::new();
        let result = analyzer
            .analyze("Company reports massive loss, announces layoffs")
            .await
            .unwrap();

        assert!(result.score < 0.0);
        assert!(matches!(
            result.sentiment,
            Sentiment::Negative | Sentiment::VeryNegative
        ));
    }

    #[tokio::test]
    async fn test_entity_extraction() {
        let analyzer = MockSentimentAnalyzer::new();
        let result = analyzer
            .analyze("AAPL stock surges after earnings beat")
            .await
            .unwrap();

        assert!(result.entity_sentiments.contains_key("AAPL"));
    }
}
