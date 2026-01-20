//! Response parser for LLM outputs

use super::LlmError;
use crate::analysis::{EventType, SentimentResult};
use serde::{Deserialize, Serialize};

/// Response from LLM completion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmResponse {
    /// The content/text response
    pub content: String,
    /// Model used for generation
    pub model: String,
    /// Number of tokens used
    pub tokens_used: usize,
    /// Latency in milliseconds
    pub latency_ms: u64,
}

/// Parse an analysis response from LLM output
pub fn parse_analysis_response(response: &str) -> Result<ParsedAnalysis, LlmError> {
    let parser = ResponseParser::new();
    parser.parse(response)
}

/// Parsed analysis result from LLM response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedAnalysis {
    /// Extracted sentiment
    pub sentiment: SentimentResult,
    /// Identified event type
    pub event_type: EventType,
    /// Affected assets/symbols
    pub affected_assets: Vec<String>,
    /// Key entities mentioned
    pub entities: Vec<String>,
    /// Confidence in the analysis
    pub confidence: f64,
    /// Brief summary
    pub summary: String,
    /// Suggested trading action
    pub suggested_action: Option<TradingAction>,
}

/// Trading action suggestion from LLM
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TradingAction {
    /// Strong buy signal
    StrongBuy,
    /// Buy signal
    Buy,
    /// Hold/no action
    Hold,
    /// Sell signal
    Sell,
    /// Strong sell signal
    StrongSell,
}

impl TradingAction {
    /// Convert sentiment score to trading action
    pub fn from_sentiment(score: f64, confidence: f64) -> Self {
        let weighted_score = score * confidence;

        if weighted_score > 0.6 {
            TradingAction::StrongBuy
        } else if weighted_score > 0.2 {
            TradingAction::Buy
        } else if weighted_score > -0.2 {
            TradingAction::Hold
        } else if weighted_score > -0.6 {
            TradingAction::Sell
        } else {
            TradingAction::StrongSell
        }
    }

    /// Get action as a numeric signal (-1.0 to 1.0)
    pub fn as_signal(&self) -> f64 {
        match self {
            TradingAction::StrongBuy => 1.0,
            TradingAction::Buy => 0.5,
            TradingAction::Hold => 0.0,
            TradingAction::Sell => -0.5,
            TradingAction::StrongSell => -1.0,
        }
    }
}

/// Parser for LLM responses
#[derive(Debug, Default)]
pub struct ResponseParser {
    /// Whether to use strict JSON parsing
    strict_mode: bool,
}

impl ResponseParser {
    /// Create a new response parser
    pub fn new() -> Self {
        Self { strict_mode: false }
    }

    /// Enable strict JSON parsing mode
    pub fn with_strict_mode(mut self, strict: bool) -> Self {
        self.strict_mode = strict;
        self
    }

    /// Parse LLM response into structured analysis
    pub fn parse(&self, response: &str) -> Result<ParsedAnalysis, LlmError> {
        // Try to extract JSON from response
        let json_str = self.extract_json(response)?;

        // Parse the JSON
        let raw: RawLlmResponse = serde_json::from_str(&json_str)
            .map_err(|e| LlmError::ParseError(format!("JSON parse error: {}", e)))?;

        // Convert to ParsedAnalysis
        self.convert_response(raw)
    }

    /// Extract JSON from potentially mixed text/JSON response
    fn extract_json(&self, response: &str) -> Result<String, LlmError> {
        let trimmed = response.trim();

        // If it starts with {, assume it's JSON
        if trimmed.starts_with('{') {
            // Find matching closing brace
            let mut depth = 0;
            let mut end_idx = 0;

            for (i, c) in trimmed.char_indices() {
                match c {
                    '{' => depth += 1,
                    '}' => {
                        depth -= 1;
                        if depth == 0 {
                            end_idx = i + 1;
                            break;
                        }
                    }
                    _ => {}
                }
            }

            if end_idx > 0 {
                return Ok(trimmed[..end_idx].to_string());
            }
        }

        // Try to find JSON block in markdown code fence
        if let Some(start) = trimmed.find("```json") {
            let json_start = start + 7;
            if let Some(end) = trimmed[json_start..].find("```") {
                return Ok(trimmed[json_start..json_start + end].trim().to_string());
            }
        }

        // Try to find any JSON object
        if let Some(start) = trimmed.find('{') {
            if let Some(end) = trimmed.rfind('}') {
                if end > start {
                    return Ok(trimmed[start..=end].to_string());
                }
            }
        }

        if self.strict_mode {
            Err(LlmError::ParseError(
                "No valid JSON found in response".to_string(),
            ))
        } else {
            // Create a minimal response from text
            Ok(format!(
                r#"{{"sentiment": 0.0, "confidence": 0.3, "summary": "{}"}}"#,
                response.chars().take(100).collect::<String>()
            ))
        }
    }

    /// Convert raw LLM response to ParsedAnalysis
    fn convert_response(&self, raw: RawLlmResponse) -> Result<ParsedAnalysis, LlmError> {
        let sentiment_score = raw.sentiment.unwrap_or(0.0);
        let confidence = raw.confidence.unwrap_or(0.5).clamp(0.0, 1.0);

        let sentiment = SentimentResult {
            score: sentiment_score,
            confidence,
            label: if sentiment_score > 0.2 {
                "positive".to_string()
            } else if sentiment_score < -0.2 {
                "negative".to_string()
            } else {
                "neutral".to_string()
            },
        };

        let event_type = raw
            .event_type
            .as_deref()
            .map(EventType::from_str)
            .unwrap_or(EventType::Unknown);

        let suggested_action = Some(TradingAction::from_sentiment(sentiment_score, confidence));

        Ok(ParsedAnalysis {
            sentiment,
            event_type,
            affected_assets: raw.affected_assets.unwrap_or_default(),
            entities: raw.entities.unwrap_or_default(),
            confidence,
            summary: raw.summary.unwrap_or_default(),
            suggested_action,
        })
    }
}

/// Raw response structure from LLM
#[derive(Debug, Deserialize)]
struct RawLlmResponse {
    sentiment: Option<f64>,
    confidence: Option<f64>,
    event_type: Option<String>,
    affected_assets: Option<Vec<String>>,
    entities: Option<Vec<String>>,
    summary: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trading_action_from_sentiment() {
        assert_eq!(
            TradingAction::from_sentiment(0.8, 1.0),
            TradingAction::StrongBuy
        );
        assert_eq!(TradingAction::from_sentiment(0.3, 1.0), TradingAction::Buy);
        assert_eq!(TradingAction::from_sentiment(0.0, 1.0), TradingAction::Hold);
        assert_eq!(TradingAction::from_sentiment(-0.3, 1.0), TradingAction::Sell);
        assert_eq!(
            TradingAction::from_sentiment(-0.8, 1.0),
            TradingAction::StrongSell
        );
    }

    #[test]
    fn test_parse_json_response() {
        let parser = ResponseParser::new();
        let response = r#"{"sentiment": 0.5, "confidence": 0.8, "summary": "Positive news"}"#;

        let result = parser.parse(response);
        assert!(result.is_ok());

        let analysis = result.unwrap();
        assert_eq!(analysis.sentiment.score, 0.5);
        assert_eq!(analysis.confidence, 0.8);
    }

    #[test]
    fn test_parse_markdown_json() {
        let parser = ResponseParser::new();
        let response = r#"
Here is my analysis:

```json
{"sentiment": -0.3, "confidence": 0.7, "summary": "Bearish outlook"}
```

This indicates negative sentiment.
"#;

        let result = parser.parse(response);
        assert!(result.is_ok());

        let analysis = result.unwrap();
        assert_eq!(analysis.sentiment.score, -0.3);
    }

    #[test]
    fn test_action_signal() {
        assert_eq!(TradingAction::StrongBuy.as_signal(), 1.0);
        assert_eq!(TradingAction::Hold.as_signal(), 0.0);
        assert_eq!(TradingAction::StrongSell.as_signal(), -1.0);
    }
}
