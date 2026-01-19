//! Financial sentiment analysis using LLMs.

use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::error::{TradingError, TradingResult};
use crate::llm_client::{LLMClient, LLMConfig};
use crate::prompts::sentiment as prompts;

/// Sentiment classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum Sentiment {
    Positive,
    Negative,
    Neutral,
}

/// Market impact classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum MarketImpact {
    Bullish,
    Bearish,
    Neutral,
}

/// Time horizon for impact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum TimeHorizon {
    ShortTerm,
    MediumTerm,
    LongTerm,
}

/// Result of sentiment analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentimentResult {
    pub sentiment: Sentiment,
    pub confidence: f32,
    pub market_impact: MarketImpact,
    pub time_horizon: TimeHorizon,
    pub key_factors: Vec<String>,
    pub reasoning: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_response: Option<String>,
}

/// Prompt type for sentiment analysis.
#[derive(Debug, Clone, Copy)]
pub enum SentimentPromptType {
    ZeroShot,
    FewShot,
    ChainOfThought,
}

/// Financial sentiment analyzer.
pub struct FinancialSentimentAnalyzer {
    client: Arc<dyn LLMClient>,
    prompt_type: SentimentPromptType,
    config: LLMConfig,
}

impl FinancialSentimentAnalyzer {
    /// Create a new sentiment analyzer.
    pub fn new(client: Box<dyn LLMClient>) -> Self {
        Self {
            client: Arc::from(client),
            prompt_type: SentimentPromptType::FewShot,
            config: LLMConfig {
                temperature: 0.3,
                max_tokens: 500,
                ..Default::default()
            },
        }
    }

    /// Create with specific prompt type.
    pub fn with_prompt_type(client: Box<dyn LLMClient>, prompt_type: SentimentPromptType) -> Self {
        Self {
            client: Arc::from(client),
            prompt_type,
            config: LLMConfig {
                temperature: 0.3,
                max_tokens: 500,
                ..Default::default()
            },
        }
    }

    /// Analyze sentiment of financial text.
    pub async fn analyze(&self, text: &str, symbol: &str) -> TradingResult<SentimentResult> {
        let prompt = self.build_prompt(text, symbol);
        let response = self.client.complete(&prompt, &self.config).await?;
        self.parse_response(&response)
    }

    /// Analyze with self-consistency (multiple samples).
    pub async fn analyze_with_consistency(
        &self,
        text: &str,
        symbol: &str,
        num_samples: usize,
    ) -> TradingResult<SentimentResult> {
        let prompt = self.build_prompt(text, symbol);

        // Generate multiple samples
        let mut results = Vec::new();
        for _ in 0..num_samples {
            let response = self.client.complete(&prompt, &self.config).await?;
            if let Ok(result) = self.parse_response(&response) {
                results.push(result);
            }
        }

        if results.is_empty() {
            return Err(TradingError::LLMError(
                "Failed to generate consistent results".to_string(),
            ));
        }

        // Count sentiment occurrences
        let mut positive_count = 0;
        let mut negative_count = 0;
        let mut neutral_count = 0;

        for result in &results {
            match result.sentiment {
                Sentiment::Positive => positive_count += 1,
                Sentiment::Negative => negative_count += 1,
                Sentiment::Neutral => neutral_count += 1,
            }
        }

        // Find majority sentiment
        let (majority_sentiment, majority_count) = if positive_count >= negative_count
            && positive_count >= neutral_count
        {
            (Sentiment::Positive, positive_count)
        } else if negative_count >= neutral_count {
            (Sentiment::Negative, negative_count)
        } else {
            (Sentiment::Neutral, neutral_count)
        };

        // Use first matching result as base
        let base_result = results
            .iter()
            .find(|r| r.sentiment == majority_sentiment)
            .unwrap();

        let agreement_ratio = majority_count as f32 / results.len() as f32;

        Ok(SentimentResult {
            sentiment: majority_sentiment,
            confidence: base_result.confidence * agreement_ratio,
            market_impact: base_result.market_impact,
            time_horizon: base_result.time_horizon,
            key_factors: base_result.key_factors.clone(),
            reasoning: format!(
                "[Agreement: {:.0}%] {}",
                agreement_ratio * 100.0,
                base_result.reasoning
            ),
            raw_response: None,
        })
    }

    /// Build the prompt based on prompt type.
    fn build_prompt(&self, text: &str, symbol: &str) -> String {
        match self.prompt_type {
            SentimentPromptType::ZeroShot => {
                prompts::ZERO_SHOT.replace("{text}", text)
            }
            SentimentPromptType::FewShot => prompts::FEW_SHOT
                .replace("{text}", text)
                .replace("{symbol}", symbol),
            SentimentPromptType::ChainOfThought => prompts::CHAIN_OF_THOUGHT
                .replace("{text}", text)
                .replace("{symbol}", symbol),
        }
    }

    /// Parse LLM response into SentimentResult.
    fn parse_response(&self, response: &str) -> TradingResult<SentimentResult> {
        // Find JSON in response
        let start_idx = response.find('{');
        let end_idx = response.rfind('}');

        match (start_idx, end_idx) {
            (Some(start), Some(end)) if start < end => {
                let json_str = &response[start..=end];

                // Parse JSON
                let parsed: serde_json::Value = serde_json::from_str(json_str)?;

                // Extract fields with defaults
                let sentiment_str = parsed
                    .get("sentiment")
                    .and_then(|v| v.as_str())
                    .unwrap_or("NEUTRAL");

                let sentiment = match sentiment_str.to_uppercase().as_str() {
                    "POSITIVE" => Sentiment::Positive,
                    "NEGATIVE" => Sentiment::Negative,
                    _ => Sentiment::Neutral,
                };

                let confidence = parsed
                    .get("confidence")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(50.0) as f32;

                let market_impact_str = parsed
                    .get("market_impact")
                    .and_then(|v| v.as_str())
                    .unwrap_or("NEUTRAL");

                let market_impact = match market_impact_str.to_uppercase().as_str() {
                    "BULLISH" => MarketImpact::Bullish,
                    "BEARISH" => MarketImpact::Bearish,
                    _ => MarketImpact::Neutral,
                };

                let time_horizon_str = parsed
                    .get("time_horizon")
                    .and_then(|v| v.as_str())
                    .unwrap_or("SHORT_TERM");

                let time_horizon = match time_horizon_str.to_uppercase().as_str() {
                    "MEDIUM_TERM" => TimeHorizon::MediumTerm,
                    "LONG_TERM" => TimeHorizon::LongTerm,
                    _ => TimeHorizon::ShortTerm,
                };

                let key_factors = parsed
                    .get("key_factors")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_str().map(String::from))
                            .collect()
                    })
                    .unwrap_or_default();

                let reasoning = parsed
                    .get("reasoning")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();

                Ok(SentimentResult {
                    sentiment,
                    confidence,
                    market_impact,
                    time_horizon,
                    key_factors,
                    reasoning,
                    raw_response: Some(response.to_string()),
                })
            }
            _ => Err(TradingError::ParseError(serde_json::Error::io(
                std::io::Error::new(std::io::ErrorKind::InvalidData, "No JSON found in response"),
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm_client::MockLLMClient;
    use std::collections::HashMap;

    #[tokio::test]
    async fn test_sentiment_analysis() {
        let mut responses = HashMap::new();
        responses.insert(
            "earnings".to_string(),
            r#"{"sentiment": "POSITIVE", "confidence": 85, "market_impact": "BULLISH", "time_horizon": "SHORT_TERM", "key_factors": ["earnings beat"], "reasoning": "Strong results"}"#.to_string(),
        );

        let client = MockLLMClient::with_responses(responses);
        let analyzer = FinancialSentimentAnalyzer::new(Box::new(client));

        let result = analyzer
            .analyze("Apple beats earnings expectations", "AAPL")
            .await
            .unwrap();

        assert_eq!(result.sentiment, Sentiment::Positive);
        assert_eq!(result.confidence, 85.0);
        assert_eq!(result.market_impact, MarketImpact::Bullish);
    }
}
