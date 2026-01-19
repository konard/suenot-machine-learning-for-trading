//! Risk assessor using LLM for text analysis.

use anyhow::{Context, Result};
use serde::Deserialize;
use tracing::{debug, info, warn};

use crate::llm::{LLMClient, Message};
use super::score::{Confidence, RiskDirection, RiskScore};

/// Prompt template for risk assessment.
const RISK_ASSESSMENT_PROMPT: &str = r#"Analyze the following financial text and provide a risk assessment.

Text: {text}

Evaluate the following risk dimensions on a scale of 1-10 (1=lowest risk, 10=highest risk):

1. Market Risk: Exposure to market volatility and systematic risk
2. Credit Risk: Counterparty or default risk indicators
3. Liquidity Risk: Trading and market depth concerns
4. Operational Risk: Business execution and management risks
5. Regulatory Risk: Legal and compliance exposure
6. Sentiment Risk: Market perception and reputation

For each dimension, provide a score (1-10) and identify key factors.

Also provide:
- Overall Risk Score (weighted average)
- Confidence (low/medium/high)
- Direction (increasing/stable/decreasing)
- Key factors (list of main risk drivers)

IMPORTANT: Respond ONLY with valid JSON in this exact format:
{
  "market_risk": 5.0,
  "credit_risk": 3.0,
  "liquidity_risk": 4.0,
  "operational_risk": 5.0,
  "regulatory_risk": 6.0,
  "sentiment_risk": 4.0,
  "overall_score": 4.5,
  "confidence": "medium",
  "direction": "stable",
  "key_factors": ["factor 1", "factor 2"]
}"#;

/// Raw response from LLM for risk assessment.
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct RiskAssessmentResponse {
    market_risk: Option<f64>,
    credit_risk: Option<f64>,
    liquidity_risk: Option<f64>,
    operational_risk: Option<f64>,
    regulatory_risk: Option<f64>,
    sentiment_risk: Option<f64>,
    overall_score: Option<f64>,
    confidence: Option<String>,
    direction: Option<String>,
    key_factors: Option<Vec<String>>,
}

/// Risk assessor that uses LLM for text analysis.
pub struct RiskAssessor {
    llm_client: LLMClient,
    system_prompt: String,
}

impl RiskAssessor {
    /// Create a new risk assessor with an LLM client.
    pub fn new(llm_client: LLMClient) -> Self {
        Self {
            llm_client,
            system_prompt: "You are an expert financial risk analyst. Analyze text for risk factors and provide structured risk assessments. Always respond with valid JSON.".to_string(),
        }
    }

    /// Set a custom system prompt.
    pub fn with_system_prompt(mut self, prompt: String) -> Self {
        self.system_prompt = prompt;
        self
    }

    /// Assess risk from text content.
    pub async fn assess(&self, text: &str) -> Result<RiskScore> {
        if text.trim().is_empty() {
            warn!("Empty text provided for risk assessment");
            return Ok(RiskScore::default());
        }

        let prompt = RISK_ASSESSMENT_PROMPT.replace("{text}", text);

        let messages = vec![
            Message::system(&self.system_prompt),
            Message::user(&prompt),
        ];

        debug!("Sending risk assessment request to LLM");
        let response = self.llm_client.complete(messages).await?;
        debug!("Received response from LLM");

        self.parse_response(&response)
    }

    /// Assess risk for a specific symbol with context.
    pub async fn assess_with_context(
        &self,
        text: &str,
        symbol: &str,
        additional_context: Option<&str>,
    ) -> Result<RiskScore> {
        let mut full_text = format!("Asset/Symbol: {}\n\n{}", symbol, text);

        if let Some(context) = additional_context {
            full_text = format!("{}\n\nAdditional Context:\n{}", full_text, context);
        }

        let mut score = self.assess(&full_text).await?;
        score.symbol = Some(symbol.to_string());

        Ok(score)
    }

    /// Parse the LLM response into a RiskScore.
    fn parse_response(&self, response: &str) -> Result<RiskScore> {
        // Try to extract JSON from the response
        let json_str = self.extract_json(response)?;

        let parsed: RiskAssessmentResponse = serde_json::from_str(&json_str)
            .context("Failed to parse risk assessment JSON")?;

        let confidence = match parsed.confidence.as_deref() {
            Some("high") => Confidence::High,
            Some("low") => Confidence::Low,
            _ => Confidence::Medium,
        };

        let direction = match parsed.direction.as_deref() {
            Some("increasing") => RiskDirection::Increasing,
            Some("decreasing") => RiskDirection::Decreasing,
            _ => RiskDirection::Stable,
        };

        let score = RiskScore::new(
            parsed.market_risk.unwrap_or(5.0),
            parsed.credit_risk.unwrap_or(5.0),
            parsed.liquidity_risk.unwrap_or(5.0),
            parsed.operational_risk.unwrap_or(5.0),
            parsed.regulatory_risk.unwrap_or(5.0),
            parsed.sentiment_risk.unwrap_or(5.0),
        )
        .with_confidence(confidence)
        .with_direction(direction)
        .with_factors(parsed.key_factors.unwrap_or_default());

        info!("Risk assessment complete: overall score {:.1}", score.overall_score);

        Ok(score)
    }

    /// Extract JSON from potentially mixed content.
    fn extract_json(&self, text: &str) -> Result<String> {
        // Try to find JSON object in the response
        let text = text.trim();

        // If it already looks like JSON, return it
        if text.starts_with('{') && text.ends_with('}') {
            return Ok(text.to_string());
        }

        // Try to find JSON between code blocks
        if let Some(start) = text.find("```json") {
            if let Some(end) = text[start..].find("```\n") {
                let json_start = start + 7; // Skip "```json"
                let json_end = start + end;
                if json_start < json_end {
                    return Ok(text[json_start..json_end].trim().to_string());
                }
            }
        }

        // Try to find first { and last }
        if let (Some(start), Some(end)) = (text.find('{'), text.rfind('}')) {
            if start < end {
                return Ok(text[start..=end].to_string());
            }
        }

        anyhow::bail!("Could not find JSON in response: {}", &text[..text.len().min(100)])
    }

    /// Assess risk from multiple text sources and aggregate.
    pub async fn assess_multiple(&self, texts: &[&str]) -> Result<RiskScore> {
        if texts.is_empty() {
            return Ok(RiskScore::default());
        }

        let mut scores: Vec<RiskScore> = Vec::new();

        for text in texts {
            match self.assess(text).await {
                Ok(score) => scores.push(score),
                Err(e) => {
                    warn!("Failed to assess text: {}", e);
                }
            }
        }

        if scores.is_empty() {
            return Ok(RiskScore::default());
        }

        // Aggregate scores (simple average for now)
        let n = scores.len() as f64;
        let aggregated = RiskScore::new(
            scores.iter().map(|s| s.market_risk).sum::<f64>() / n,
            scores.iter().map(|s| s.credit_risk).sum::<f64>() / n,
            scores.iter().map(|s| s.liquidity_risk).sum::<f64>() / n,
            scores.iter().map(|s| s.operational_risk).sum::<f64>() / n,
            scores.iter().map(|s| s.regulatory_risk).sum::<f64>() / n,
            scores.iter().map(|s| s.sentiment_risk).sum::<f64>() / n,
        );

        // Collect all key factors
        let all_factors: Vec<String> = scores
            .into_iter()
            .flat_map(|s| s.key_factors)
            .collect();

        Ok(aggregated.with_factors(all_factors))
    }
}

/// Builder for creating risk assessors.
#[allow(dead_code)]
pub struct RiskAssessorBuilder {
    llm_client: Option<LLMClient>,
    system_prompt: Option<String>,
}

#[allow(dead_code)]
impl RiskAssessorBuilder {
    pub fn new() -> Self {
        Self {
            llm_client: None,
            system_prompt: None,
        }
    }

    pub fn with_llm_client(mut self, client: LLMClient) -> Self {
        self.llm_client = Some(client);
        self
    }

    pub fn with_system_prompt(mut self, prompt: String) -> Self {
        self.system_prompt = Some(prompt);
        self
    }

    pub fn build(self) -> Result<RiskAssessor> {
        let client = self.llm_client.ok_or_else(|| {
            anyhow::anyhow!("LLM client is required to build RiskAssessor")
        })?;

        let mut assessor = RiskAssessor::new(client);

        if let Some(prompt) = self.system_prompt {
            assessor = assessor.with_system_prompt(prompt);
        }

        Ok(assessor)
    }
}

impl Default for RiskAssessorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_json_plain() {
        let assessor = RiskAssessor::new(
            LLMClient::new(crate::llm::LLMProvider::ollama("test".to_string()))
        );

        let json = r#"{"market_risk": 5.0, "credit_risk": 3.0}"#;
        let result = assessor.extract_json(json).unwrap();
        assert!(result.contains("market_risk"));
    }

    #[test]
    fn test_extract_json_with_text() {
        let assessor = RiskAssessor::new(
            LLMClient::new(crate::llm::LLMProvider::ollama("test".to_string()))
        );

        let response = r#"Here is the analysis:
        {"market_risk": 5.0, "credit_risk": 3.0}
        That's my assessment."#;

        let result = assessor.extract_json(response).unwrap();
        assert!(result.contains("market_risk"));
    }

    #[test]
    fn test_parse_response() {
        let assessor = RiskAssessor::new(
            LLMClient::new(crate::llm::LLMProvider::ollama("test".to_string()))
        );

        let response = r#"{
            "market_risk": 4.0,
            "credit_risk": 3.0,
            "liquidity_risk": 2.0,
            "operational_risk": 5.0,
            "regulatory_risk": 6.0,
            "sentiment_risk": 4.0,
            "overall_score": 4.0,
            "confidence": "high",
            "direction": "decreasing",
            "key_factors": ["Strong earnings", "Low debt"]
        }"#;

        let score = assessor.parse_response(response).unwrap();

        assert_eq!(score.market_risk, 4.0);
        assert_eq!(score.credit_risk, 3.0);
        assert_eq!(score.confidence, Confidence::High);
        assert_eq!(score.direction, RiskDirection::Decreasing);
        assert_eq!(score.key_factors.len(), 2);
    }
}
