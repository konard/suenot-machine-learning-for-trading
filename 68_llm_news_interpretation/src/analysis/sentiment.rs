//! Sentiment analysis for news content

use serde::{Deserialize, Serialize};

/// Result of sentiment analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentimentResult {
    /// Sentiment score from -1.0 (very negative) to +1.0 (very positive)
    pub score: f64,
    /// Confidence in the sentiment assessment (0.0 to 1.0)
    pub confidence: f64,
    /// Human-readable label
    pub label: String,
}

impl Default for SentimentResult {
    fn default() -> Self {
        Self {
            score: 0.0,
            confidence: 0.0,
            label: "neutral".to_string(),
        }
    }
}

impl SentimentResult {
    /// Create a new sentiment result
    pub fn new(score: f64, confidence: f64) -> Self {
        let label = Self::score_to_label(score);
        Self {
            score: score.clamp(-1.0, 1.0),
            confidence: confidence.clamp(0.0, 1.0),
            label,
        }
    }

    /// Convert score to human-readable label
    fn score_to_label(score: f64) -> String {
        if score >= 0.6 {
            "very_positive".to_string()
        } else if score >= 0.2 {
            "positive".to_string()
        } else if score > -0.2 {
            "neutral".to_string()
        } else if score > -0.6 {
            "negative".to_string()
        } else {
            "very_negative".to_string()
        }
    }

    /// Check if sentiment is bullish (positive)
    pub fn is_bullish(&self) -> bool {
        self.score > 0.1 && self.confidence > 0.5
    }

    /// Check if sentiment is bearish (negative)
    pub fn is_bearish(&self) -> bool {
        self.score < -0.1 && self.confidence > 0.5
    }

    /// Get weighted score (score * confidence)
    pub fn weighted_score(&self) -> f64 {
        self.score * self.confidence
    }
}

/// Sentiment analyzer using rule-based and LLM approaches
#[derive(Debug, Default)]
pub struct SentimentAnalyzer {
    /// Positive keywords with weights
    positive_keywords: Vec<(String, f64)>,
    /// Negative keywords with weights
    negative_keywords: Vec<(String, f64)>,
    /// Use LLM for analysis (vs rule-based)
    use_llm: bool,
}

impl SentimentAnalyzer {
    /// Create a new sentiment analyzer
    pub fn new() -> Self {
        Self {
            positive_keywords: Self::default_positive_keywords(),
            negative_keywords: Self::default_negative_keywords(),
            use_llm: false,
        }
    }

    /// Enable LLM-based analysis
    pub fn with_llm(mut self, use_llm: bool) -> Self {
        self.use_llm = use_llm;
        self
    }

    /// Analyze sentiment using rule-based approach
    pub fn analyze_rules(&self, text: &str) -> SentimentResult {
        let text_lower = text.to_lowercase();
        let words: Vec<&str> = text_lower.split_whitespace().collect();
        let word_count = words.len() as f64;

        if word_count == 0.0 {
            return SentimentResult::default();
        }

        let mut positive_score = 0.0;
        let mut negative_score = 0.0;
        let mut matches = 0;

        // Check positive keywords
        for (keyword, weight) in &self.positive_keywords {
            if text_lower.contains(keyword) {
                positive_score += weight;
                matches += 1;
            }
        }

        // Check negative keywords
        for (keyword, weight) in &self.negative_keywords {
            if text_lower.contains(keyword) {
                negative_score += weight;
                matches += 1;
            }
        }

        // Calculate final score
        let raw_score = positive_score - negative_score;
        let score = (raw_score / (1.0 + raw_score.abs())).clamp(-1.0, 1.0);

        // Confidence based on keyword matches and text length
        let keyword_confidence = (matches as f64 / 5.0).min(1.0);
        let length_confidence = (word_count / 50.0).min(1.0);
        let confidence = (keyword_confidence * 0.7 + length_confidence * 0.3).clamp(0.1, 0.9);

        SentimentResult::new(score, confidence)
    }

    /// Get default positive keywords for crypto/finance
    fn default_positive_keywords() -> Vec<(String, f64)> {
        vec![
            ("bullish".to_string(), 0.8),
            ("surge".to_string(), 0.6),
            ("rally".to_string(), 0.6),
            ("soar".to_string(), 0.7),
            ("breakout".to_string(), 0.5),
            ("approved".to_string(), 0.7),
            ("adoption".to_string(), 0.5),
            ("partnership".to_string(), 0.4),
            ("upgrade".to_string(), 0.4),
            ("growth".to_string(), 0.4),
            ("gains".to_string(), 0.5),
            ("profit".to_string(), 0.4),
            ("milestone".to_string(), 0.4),
            ("breakthrough".to_string(), 0.6),
            ("accumulation".to_string(), 0.3),
            ("institutional".to_string(), 0.3),
            ("moon".to_string(), 0.5),
            ("pump".to_string(), 0.4),
            ("ath".to_string(), 0.6),
            ("all-time high".to_string(), 0.6),
        ]
    }

    /// Get default negative keywords for crypto/finance
    fn default_negative_keywords() -> Vec<(String, f64)> {
        vec![
            ("bearish".to_string(), 0.8),
            ("crash".to_string(), 0.8),
            ("plunge".to_string(), 0.7),
            ("dump".to_string(), 0.6),
            ("hack".to_string(), 0.9),
            ("exploit".to_string(), 0.8),
            ("vulnerability".to_string(), 0.6),
            ("ban".to_string(), 0.7),
            ("lawsuit".to_string(), 0.6),
            ("investigation".to_string(), 0.5),
            ("fraud".to_string(), 0.8),
            ("scam".to_string(), 0.9),
            ("rug".to_string(), 0.9),
            ("bankrupt".to_string(), 0.9),
            ("insolvency".to_string(), 0.8),
            ("liquidation".to_string(), 0.6),
            ("fear".to_string(), 0.5),
            ("panic".to_string(), 0.7),
            ("sell-off".to_string(), 0.6),
            ("collapse".to_string(), 0.8),
        ]
    }

    /// Add custom positive keyword
    pub fn add_positive_keyword(&mut self, keyword: &str, weight: f64) {
        self.positive_keywords
            .push((keyword.to_lowercase(), weight));
    }

    /// Add custom negative keyword
    pub fn add_negative_keyword(&mut self, keyword: &str, weight: f64) {
        self.negative_keywords
            .push((keyword.to_lowercase(), weight));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sentiment_result_new() {
        let result = SentimentResult::new(0.5, 0.8);
        assert_eq!(result.score, 0.5);
        assert_eq!(result.confidence, 0.8);
        assert_eq!(result.label, "positive");
    }

    #[test]
    fn test_sentiment_clamping() {
        let result = SentimentResult::new(1.5, 1.5);
        assert_eq!(result.score, 1.0);
        assert_eq!(result.confidence, 1.0);
    }

    #[test]
    fn test_is_bullish_bearish() {
        let bullish = SentimentResult::new(0.5, 0.8);
        let bearish = SentimentResult::new(-0.5, 0.8);
        let neutral = SentimentResult::new(0.0, 0.8);

        assert!(bullish.is_bullish());
        assert!(!bullish.is_bearish());
        assert!(bearish.is_bearish());
        assert!(!bearish.is_bullish());
        assert!(!neutral.is_bullish());
        assert!(!neutral.is_bearish());
    }

    #[test]
    fn test_rule_based_sentiment() {
        let analyzer = SentimentAnalyzer::new();

        let positive = analyzer.analyze_rules("Bitcoin shows bullish breakout with massive gains");
        assert!(positive.score > 0.0);

        let negative = analyzer.analyze_rules("Major hack causes crypto crash and panic selling");
        assert!(negative.score < 0.0);
    }

    #[test]
    fn test_weighted_score() {
        let result = SentimentResult::new(0.6, 0.5);
        assert!((result.weighted_score() - 0.3).abs() < 0.001);
    }
}
