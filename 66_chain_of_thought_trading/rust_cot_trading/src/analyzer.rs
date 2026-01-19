//! Chain-of-Thought Analyzer
//!
//! This module provides the core CoT analysis functionality for trading.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use crate::error::{Error, Result};

/// A single reasoning step in the chain of thought.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningStep {
    /// Name of this analysis step
    pub step_name: String,
    /// Input data for this step
    pub input_data: String,
    /// The reasoning process
    pub reasoning: String,
    /// Conclusion from this step
    pub conclusion: String,
    /// Confidence in this step (0.0-1.0)
    pub confidence: f64,
}

impl ReasoningStep {
    /// Create a new reasoning step.
    pub fn new(
        step_name: impl Into<String>,
        input_data: impl Into<String>,
        reasoning: impl Into<String>,
        conclusion: impl Into<String>,
        confidence: f64,
    ) -> Self {
        Self {
            step_name: step_name.into(),
            input_data: input_data.into(),
            reasoning: reasoning.into(),
            conclusion: conclusion.into(),
            confidence: confidence.clamp(0.0, 1.0),
        }
    }
}

/// Complete CoT analysis result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoTAnalysis {
    /// The original query/prompt
    pub query: String,
    /// Chain of reasoning steps
    pub reasoning_steps: Vec<ReasoningStep>,
    /// Final answer/recommendation
    pub final_answer: String,
    /// Overall confidence (0.0-1.0)
    pub confidence: f64,
    /// Processing time in milliseconds
    pub processing_time_ms: u64,
}

/// Trait for Chain-of-Thought analyzers.
#[async_trait]
pub trait Analyzer: Send + Sync {
    /// Analyze a query and return a CoT analysis.
    async fn analyze(&self, query: &str) -> Result<CoTAnalysis>;
}

/// CoT Analyzer implementation.
pub struct CoTAnalyzer {
    /// The underlying analyzer implementation
    inner: Box<dyn Analyzer>,
}

impl CoTAnalyzer {
    /// Create a new CoT analyzer with a custom implementation.
    pub fn new(analyzer: impl Analyzer + 'static) -> Self {
        Self {
            inner: Box::new(analyzer),
        }
    }

    /// Create a mock analyzer for testing (no API key needed).
    pub fn new_mock() -> Self {
        Self::new(MockAnalyzer::default())
    }

    /// Analyze a query and return a CoT analysis.
    pub async fn analyze(&self, query: &str) -> Result<CoTAnalysis> {
        self.inner.analyze(query).await
    }

    /// Analyze with self-consistency (multiple samples).
    pub async fn analyze_with_consistency(
        &self,
        query: &str,
        num_samples: usize,
    ) -> Result<CoTAnalysis> {
        let mut analyses = Vec::with_capacity(num_samples);

        for _ in 0..num_samples {
            let analysis = self.analyze(query).await?;
            analyses.push(analysis);
        }

        // Find the most common answer and average confidence
        let total_confidence: f64 = analyses.iter().map(|a| a.confidence).sum();
        let avg_confidence = total_confidence / num_samples as f64;

        // Use the first analysis as base (in production, would aggregate properly)
        let mut result = analyses.remove(0);
        result.confidence = avg_confidence;

        Ok(result)
    }
}

/// Mock analyzer for testing without API calls.
#[derive(Debug, Clone)]
pub struct MockAnalyzer {
    /// Base confidence for mock responses
    pub base_confidence: f64,
}

impl Default for MockAnalyzer {
    fn default() -> Self {
        Self {
            base_confidence: 0.75,
        }
    }
}

#[async_trait]
impl Analyzer for MockAnalyzer {
    async fn analyze(&self, query: &str) -> Result<CoTAnalysis> {
        use std::time::Instant;
        let start = Instant::now();

        // Simulate processing delay
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

        // Create mock reasoning steps
        let steps = vec![
            ReasoningStep::new(
                "Data Gathering",
                query,
                "First, I need to gather and organize all the relevant market data provided in the query.",
                "Data gathered and organized for analysis",
                0.9,
            ),
            ReasoningStep::new(
                "Trend Analysis",
                "Price and moving average data",
                "Looking at the price relative to moving averages to determine the overall trend direction.",
                "Trend appears moderately bullish based on price position",
                0.8,
            ),
            ReasoningStep::new(
                "Momentum Check",
                "RSI and MACD indicators",
                "Checking momentum indicators to confirm trend strength and potential reversals.",
                "Momentum indicators show positive confirmation",
                0.75,
            ),
            ReasoningStep::new(
                "Volume Analysis",
                "Volume data",
                "Analyzing volume to validate price movements and gauge market participation.",
                "Volume supports the current price action",
                0.7,
            ),
            ReasoningStep::new(
                "Risk Assessment",
                "ATR and price levels",
                "Calculating appropriate stop loss and take profit levels based on volatility.",
                "Risk/reward ratio is favorable at approximately 1:2",
                0.8,
            ),
            ReasoningStep::new(
                "Final Decision",
                "All previous analysis",
                "Aggregating all factors to make a final trading decision with appropriate confidence.",
                "BUY signal with moderate confidence",
                self.base_confidence,
            ),
        ];

        let confidence = steps.iter().map(|s| s.confidence).sum::<f64>() / steps.len() as f64;

        Ok(CoTAnalysis {
            query: query.to_string(),
            reasoning_steps: steps,
            final_answer: "BUY - Positive trend with momentum confirmation".to_string(),
            confidence,
            processing_time_ms: start.elapsed().as_millis() as u64,
        })
    }
}

/// OpenAI-based analyzer (requires API key).
pub struct OpenAIAnalyzer {
    api_key: String,
    model: String,
    client: reqwest::Client,
}

impl OpenAIAnalyzer {
    /// Create a new OpenAI analyzer.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            model: "gpt-4".to_string(),
            client: reqwest::Client::new(),
        }
    }

    /// Set the model to use.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }
}

#[async_trait]
impl Analyzer for OpenAIAnalyzer {
    async fn analyze(&self, query: &str) -> Result<CoTAnalysis> {
        use std::time::Instant;
        let start = Instant::now();

        let system_prompt = r#"You are a financial analyst using Chain-of-Thought reasoning.
For each analysis, provide:
1. Clear step-by-step reasoning
2. Confidence levels for each step
3. A final recommendation with overall confidence

Format your response as JSON with this structure:
{
    "steps": [
        {"name": "...", "input": "...", "reasoning": "...", "conclusion": "...", "confidence": 0.8}
    ],
    "final_answer": "...",
    "confidence": 0.75
}"#;

        let request_body = serde_json::json!({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            "temperature": 0.3,
            "max_tokens": 2000
        });

        let response = self.client
            .post("https://api.openai.com/v1/chat/completions")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| Error::ApiError(e.to_string()))?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(Error::ApiError(format!("API request failed: {}", error_text)));
        }

        let response_json: serde_json::Value = response.json().await?;

        // Parse the response (simplified - production would need better parsing)
        let content = response_json["choices"][0]["message"]["content"]
            .as_str()
            .ok_or_else(|| Error::ApiError("No content in response".to_string()))?;

        // Try to parse as JSON, fall back to creating a simple analysis
        let analysis: CoTAnalysis = match serde_json::from_str(content) {
            Ok(parsed) => parsed,
            Err(_) => {
                // Fallback: create a simple analysis from the text response
                CoTAnalysis {
                    query: query.to_string(),
                    reasoning_steps: vec![ReasoningStep::new(
                        "Analysis",
                        query,
                        content,
                        "See reasoning above",
                        0.7,
                    )],
                    final_answer: content.to_string(),
                    confidence: 0.7,
                    processing_time_ms: start.elapsed().as_millis() as u64,
                }
            }
        };

        Ok(analysis)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mock_analyzer() {
        let analyzer = CoTAnalyzer::new_mock();
        let result = analyzer.analyze("Test query").await.unwrap();

        assert!(!result.reasoning_steps.is_empty());
        assert!(result.confidence > 0.0 && result.confidence <= 1.0);
        assert!(!result.final_answer.is_empty());
    }

    #[tokio::test]
    async fn test_self_consistency() {
        let analyzer = CoTAnalyzer::new_mock();
        let result = analyzer.analyze_with_consistency("Test query", 3).await.unwrap();

        assert!(!result.reasoning_steps.is_empty());
        assert!(result.confidence > 0.0);
    }
}
