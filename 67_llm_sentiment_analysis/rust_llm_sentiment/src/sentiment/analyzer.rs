//! # Sentiment Analyzer
//!
//! Main sentiment analysis module combining lexicon and crypto terminology.

use super::crypto::CryptoTerminology;
use super::lexicon::FinancialLexicon;
use crate::data::TextPreprocessor;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Sentiment level classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SentimentLevel {
    /// Very negative sentiment (score < -0.5)
    VeryNegative,
    /// Negative sentiment (-0.5 <= score < -0.2)
    Negative,
    /// Neutral sentiment (-0.2 <= score <= 0.2)
    Neutral,
    /// Positive sentiment (0.2 < score <= 0.5)
    Positive,
    /// Very positive sentiment (score > 0.5)
    VeryPositive,
}

impl SentimentLevel {
    /// Convert numeric score to sentiment level
    pub fn from_score(score: f64) -> Self {
        if score < -0.5 {
            SentimentLevel::VeryNegative
        } else if score < -0.2 {
            SentimentLevel::Negative
        } else if score <= 0.2 {
            SentimentLevel::Neutral
        } else if score <= 0.5 {
            SentimentLevel::Positive
        } else {
            SentimentLevel::VeryPositive
        }
    }

    /// Get string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            SentimentLevel::VeryNegative => "Very Negative",
            SentimentLevel::Negative => "Negative",
            SentimentLevel::Neutral => "Neutral",
            SentimentLevel::Positive => "Positive",
            SentimentLevel::VeryPositive => "Very Positive",
        }
    }

    /// Check if sentiment is bullish (positive or very positive)
    pub fn is_bullish(&self) -> bool {
        matches!(self, SentimentLevel::Positive | SentimentLevel::VeryPositive)
    }

    /// Check if sentiment is bearish (negative or very negative)
    pub fn is_bearish(&self) -> bool {
        matches!(self, SentimentLevel::Negative | SentimentLevel::VeryNegative)
    }
}

/// Sentiment analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentimentResult {
    /// Original text
    pub text: String,
    /// Overall sentiment score (-1 to 1)
    pub score: f64,
    /// Classified sentiment level
    pub level: SentimentLevel,
    /// Confidence in the prediction (0 to 1)
    pub confidence: f64,
    /// Score from financial lexicon
    pub lexicon_score: f64,
    /// Score from crypto terminology
    pub crypto_score: f64,
    /// Key sentiment words found
    pub key_words: Vec<String>,
    /// Crypto slang terms found
    pub crypto_terms: Vec<String>,
    /// Timestamp of analysis
    pub timestamp: DateTime<Utc>,
}

/// Sentiment analyzer combining multiple analysis methods
pub struct SentimentAnalyzer {
    /// Text preprocessor
    preprocessor: TextPreprocessor,
    /// Financial sentiment lexicon
    lexicon: FinancialLexicon,
    /// Crypto terminology processor
    crypto: CryptoTerminology,
    /// Weight for lexicon score
    lexicon_weight: f64,
    /// Weight for crypto score
    crypto_weight: f64,
}

impl Default for SentimentAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl SentimentAnalyzer {
    /// Create a new sentiment analyzer with default settings
    pub fn new() -> Self {
        Self {
            preprocessor: TextPreprocessor::new(),
            lexicon: FinancialLexicon::new(),
            crypto: CryptoTerminology::new(),
            lexicon_weight: 0.5,
            crypto_weight: 0.5,
        }
    }

    /// Set weights for combining scores
    pub fn with_weights(mut self, lexicon_weight: f64, crypto_weight: f64) -> Self {
        let total = lexicon_weight + crypto_weight;
        self.lexicon_weight = lexicon_weight / total;
        self.crypto_weight = crypto_weight / total;
        self
    }

    /// Analyze sentiment of a single text
    pub fn analyze(&self, text: &str) -> SentimentResult {
        // Preprocess for lexicon analysis
        let clean_text = self.preprocessor.preprocess(text);

        // Financial lexicon analysis
        let lexicon_result = self.lexicon.analyze(&clean_text);

        // Crypto terminology analysis (use original text to catch exact terms)
        let crypto_result = self.crypto.analyze(text);

        // Combine scores
        let combined_score = if lexicon_result.word_count == 0 && crypto_result.term_count == 0 {
            0.0
        } else if lexicon_result.word_count == 0 {
            crypto_result.score
        } else if crypto_result.term_count == 0 {
            lexicon_result.score
        } else {
            self.lexicon_weight * lexicon_result.score
                + self.crypto_weight * crypto_result.score
        };

        // Calculate confidence based on word coverage
        let total_words = text.split_whitespace().count() as f64;
        let matched_words = (lexicon_result.word_count + crypto_result.term_count) as f64;
        let word_coverage = if total_words > 0.0 {
            (matched_words / total_words).min(1.0)
        } else {
            0.0
        };

        // Confidence increases with more matched words, decreases with neutral scores
        let score_strength = combined_score.abs();
        let confidence = (0.3 + 0.4 * word_coverage + 0.3 * score_strength).min(0.95);

        // Extract key words
        let key_words: Vec<String> = lexicon_result
            .matched_words
            .into_iter()
            .map(|(word, _)| word)
            .collect();

        let crypto_terms: Vec<String> = crypto_result
            .found_terms
            .into_iter()
            .map(|(term, _, _)| term)
            .collect();

        SentimentResult {
            text: text.to_string(),
            score: combined_score,
            level: SentimentLevel::from_score(combined_score),
            confidence,
            lexicon_score: lexicon_result.score,
            crypto_score: crypto_result.score,
            key_words,
            crypto_terms,
            timestamp: Utc::now(),
        }
    }

    /// Analyze multiple texts
    pub fn analyze_batch(&self, texts: &[&str]) -> Vec<SentimentResult> {
        texts.iter().map(|text| self.analyze(text)).collect()
    }

    /// Analyze with additional context (e.g., market conditions)
    pub fn analyze_with_context(
        &self,
        text: &str,
        context: &AnalysisContext,
    ) -> SentimentResult {
        let mut result = self.analyze(text);

        // Adjust score based on context
        let context_adjustment = context.calculate_adjustment();
        result.score = (result.score + context_adjustment * 0.1).clamp(-1.0, 1.0);
        result.level = SentimentLevel::from_score(result.score);

        // Adjust confidence based on context alignment
        if (result.score > 0.0 && context.market_trend > 0.0)
            || (result.score < 0.0 && context.market_trend < 0.0)
        {
            result.confidence = (result.confidence * 1.1).min(0.95);
        }

        result
    }
}

/// Context for sentiment analysis
#[derive(Debug, Clone, Default)]
pub struct AnalysisContext {
    /// Current market trend (-1 to 1)
    pub market_trend: f64,
    /// Recent price change percentage
    pub price_change: f64,
    /// Market volatility (0 to 1)
    pub volatility: f64,
    /// Social media volume (normalized)
    pub social_volume: f64,
}

impl AnalysisContext {
    /// Calculate context-based adjustment
    pub fn calculate_adjustment(&self) -> f64 {
        // Weighted combination of context factors
        let trend_factor = self.market_trend * 0.4;
        let price_factor = (self.price_change / 10.0).clamp(-0.3, 0.3);
        let volatility_factor = self.volatility * -0.1; // High volatility increases uncertainty

        trend_factor + price_factor + volatility_factor
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sentiment_levels() {
        assert_eq!(SentimentLevel::from_score(-0.8), SentimentLevel::VeryNegative);
        assert_eq!(SentimentLevel::from_score(-0.3), SentimentLevel::Negative);
        assert_eq!(SentimentLevel::from_score(0.0), SentimentLevel::Neutral);
        assert_eq!(SentimentLevel::from_score(0.3), SentimentLevel::Positive);
        assert_eq!(SentimentLevel::from_score(0.8), SentimentLevel::VeryPositive);
    }

    #[test]
    fn test_analyze_positive() {
        let analyzer = SentimentAnalyzer::new();
        let result = analyzer.analyze("Bitcoin is bullish! Mooning to new ATH!");
        assert!(result.score > 0.0);
        assert!(result.level.is_bullish());
    }

    #[test]
    fn test_analyze_negative() {
        let analyzer = SentimentAnalyzer::new();
        let result = analyzer.analyze("Market crash! Everyone getting rekt. Rug pull!");
        assert!(result.score < 0.0);
        assert!(result.level.is_bearish());
    }

    #[test]
    fn test_analyze_neutral() {
        let analyzer = SentimentAnalyzer::new();
        let result = analyzer.analyze("The meeting is scheduled for tomorrow.");
        assert!((result.score.abs()) < 0.3);
    }

    #[test]
    fn test_confidence() {
        let analyzer = SentimentAnalyzer::new();

        // More sentiment words = higher confidence
        let weak = analyzer.analyze("It might be good.");
        let strong = analyzer.analyze("This is extremely bullish! Huge gains! Massive rally!");

        assert!(strong.confidence >= weak.confidence);
    }

    #[test]
    fn test_batch_analysis() {
        let analyzer = SentimentAnalyzer::new();
        let texts = vec![
            "Bitcoin is mooning!",
            "Market is crashing",
            "Price is stable",
        ];
        let results = analyzer.analyze_batch(&texts);

        assert_eq!(results.len(), 3);
        assert!(results[0].score > 0.0);
        assert!(results[1].score < 0.0);
    }

    #[test]
    fn test_context_adjustment() {
        let analyzer = SentimentAnalyzer::new();

        let bullish_context = AnalysisContext {
            market_trend: 0.5,
            price_change: 5.0,
            volatility: 0.3,
            social_volume: 0.7,
        };

        let without_context = analyzer.analyze("Market looking interesting today");
        let with_context = analyzer.analyze_with_context(
            "Market looking interesting today",
            &bullish_context,
        );

        // Context should affect the score
        assert_ne!(without_context.score, with_context.score);
    }
}
