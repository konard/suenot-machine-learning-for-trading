//! Sentiment analysis module
//!
//! Analyzes sentiment in earnings call transcripts using financial lexicon.

use std::collections::HashSet;

/// Sentiment analyzer for financial text
pub struct SentimentAnalyzer {
    positive_words: HashSet<&'static str>,
    negative_words: HashSet<&'static str>,
    hedging_words: HashSet<&'static str>,
    confidence_words: HashSet<&'static str>,
}

/// Result of sentiment analysis
#[derive(Debug, Clone)]
pub struct SentimentResult {
    pub score: f64,
    pub positive_score: f64,
    pub negative_score: f64,
    pub hedging_score: f64,
    pub confidence: f64,
    pub positive_phrases: Vec<String>,
    pub negative_phrases: Vec<String>,
    pub hedging_words: Vec<String>,
    pub confidence_words: Vec<String>,
    pub word_count: usize,
}

impl SentimentAnalyzer {
    /// Create a new sentiment analyzer with financial lexicon
    pub fn new() -> Self {
        Self {
            positive_words: Self::build_positive_lexicon(),
            negative_words: Self::build_negative_lexicon(),
            hedging_words: Self::build_hedging_lexicon(),
            confidence_words: Self::build_confidence_lexicon(),
        }
    }

    /// Analyze sentiment in text
    pub fn analyze(&self, text: &str) -> SentimentResult {
        let text_lower = text.to_lowercase();
        let words: Vec<&str> = text_lower.split_whitespace().collect();
        let word_count = words.len();

        if word_count == 0 {
            return SentimentResult {
                score: 0.0,
                positive_score: 0.0,
                negative_score: 0.0,
                hedging_score: 0.0,
                confidence: 0.0,
                positive_phrases: vec![],
                negative_phrases: vec![],
                hedging_words: vec![],
                confidence_words: vec![],
                word_count: 0,
            };
        }

        let (positive_count, positive_found) = self.count_and_collect(&text_lower, &self.positive_words);
        let (negative_count, negative_found) = self.count_and_collect(&text_lower, &self.negative_words);
        let (hedging_count, hedging_found) = self.count_and_collect(&text_lower, &self.hedging_words);
        let (confidence_count, confidence_found) = self.count_and_collect(&text_lower, &self.confidence_words);

        let word_count_f64 = word_count as f64;
        let positive_score = positive_count as f64 / word_count_f64;
        let negative_score = negative_count as f64 / word_count_f64;
        let hedging_score = hedging_count as f64 / word_count_f64;
        let confidence_score = confidence_count as f64 / word_count_f64;

        // Net score considers confidence as positive and hedging as negative
        let net_score = (positive_score + confidence_score * 0.5)
            - (negative_score + hedging_score * 0.5);

        // Confidence level based on confidence words vs hedging
        let confidence = if hedging_count + confidence_count > 0 {
            confidence_count as f64 / (hedging_count + confidence_count) as f64
        } else {
            0.5
        };

        SentimentResult {
            score: net_score,
            positive_score,
            negative_score,
            hedging_score,
            confidence,
            positive_phrases: positive_found,
            negative_phrases: negative_found,
            hedging_words: hedging_found,
            confidence_words: confidence_found,
            word_count,
        }
    }

    /// Analyze management confidence level
    pub fn analyze_confidence(&self, text: &str) -> f64 {
        let result = self.analyze(text);
        result.confidence
    }

    /// Count word matches and collect found words
    fn count_and_collect(&self, text: &str, words: &HashSet<&'static str>) -> (usize, Vec<String>) {
        let found: Vec<String> = words
            .iter()
            .filter(|word| text.contains(*word))
            .map(|word| word.to_string())
            .collect();
        (found.len(), found)
    }

    /// Build positive word lexicon
    fn build_positive_lexicon() -> HashSet<&'static str> {
        [
            // Growth and performance
            "growth", "growing", "grew", "increase", "increased", "increasing",
            "improve", "improved", "improvement", "improving", "strong", "stronger",
            "strength", "robust", "solid", "excellent", "exceptional", "outstanding",
            "record", "beat", "exceeded", "surpassed", "outperformed",

            // Financial positives
            "profit", "profitable", "profitability", "margin", "margins",
            "revenue", "earnings", "momentum", "accelerate", "accelerated",
            "expansion", "expand", "expanded", "expanding",

            // Market position
            "leader", "leading", "leadership", "dominant", "competitive",
            "advantage", "opportunity", "opportunities", "potential",

            // Confidence indicators
            "confident", "confidence", "optimistic", "positive", "pleased",
            "excited", "encouraged", "favorable", "success", "successful",
            "achieve", "achieved", "achievement", "deliver", "delivered",

            // Forward looking
            "guidance", "raise", "raised", "upgrade", "upside",
        ].into_iter().collect()
    }

    /// Build negative word lexicon
    fn build_negative_lexicon() -> HashSet<&'static str> {
        [
            // Decline indicators
            "decline", "declined", "declining", "decrease", "decreased",
            "decreasing", "drop", "dropped", "dropping", "fall", "fell",
            "falling", "weak", "weaker", "weakness", "soft", "softer",

            // Problems
            "challenge", "challenges", "challenging", "difficult", "difficulty",
            "headwind", "headwinds", "pressure", "pressured", "pressures",
            "concern", "concerns", "concerned", "risk", "risks", "risky",

            // Financial negatives
            "loss", "losses", "miss", "missed", "below", "shortfall",
            "disappointing", "disappointed", "disappoints", "underperformed",

            // Uncertainty
            "uncertain", "uncertainty", "volatile", "volatility", "unstable",
            "downturn", "recession", "slowdown", "slowing", "slower",

            // Actions
            "restructure", "restructuring", "layoff", "layoffs", "cut", "cuts",
            "reduce", "reduced", "reduction", "impairment", "writedown",

            // Forward looking
            "lower", "lowered", "downgrade", "downside",
        ].into_iter().collect()
    }

    /// Build hedging word lexicon
    fn build_hedging_lexicon() -> HashSet<&'static str> {
        [
            // Uncertainty markers
            "may", "might", "could", "possibly", "perhaps", "potentially",
            "likely", "unlikely", "probable", "probably", "approximately",
            "roughly", "around", "about", "estimate", "estimated",

            // Conditional
            "assuming", "depending", "contingent", "conditional",

            // Vague qualifiers
            "somewhat", "relatively", "fairly", "moderately", "slightly",
            "partially", "largely", "generally", "typically", "usually",

            // Disclaimers
            "believe", "expect", "anticipate", "intend", "plan",
        ].into_iter().collect()
    }

    /// Build confidence word lexicon
    fn build_confidence_lexicon() -> HashSet<&'static str> {
        [
            // Certainty markers
            "will", "definitely", "certainly", "absolutely", "clearly",
            "undoubtedly", "guaranteed",

            // Commitment
            "committed", "commitment", "dedicated", "focused", "determined",
            "resolute", "unwavering",

            // Evidence-based
            "proven", "demonstrated", "evidenced", "confirmed", "validated",
            "established",

            // Strength indicators
            "continue", "continued", "continuing", "maintain", "maintained",
            "sustain", "sustained", "sustainable", "consistent", "consistently",

            // Action
            "execute", "executed", "executing", "execution", "implement",
            "implemented", "implementing",
        ].into_iter().collect()
    }
}

impl Default for SentimentAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_positive_sentiment() {
        let analyzer = SentimentAnalyzer::new();
        let text = "We achieved strong growth and exceeded expectations.
                    Revenue increased significantly with improved margins.";

        let result = analyzer.analyze(text);

        assert!(result.positive_score > 0.0);
        assert!(result.score > 0.0);
    }

    #[test]
    fn test_negative_sentiment() {
        let analyzer = SentimentAnalyzer::new();
        let text = "We faced significant challenges with declining revenue.
                    The weak market conditions created headwinds.";

        let result = analyzer.analyze(text);

        assert!(result.negative_score > 0.0);
        assert!(result.score < 0.0);
    }

    #[test]
    fn test_hedging_detection() {
        let analyzer = SentimentAnalyzer::new();
        let text = "We believe results may improve, potentially leading to
                    approximately better outcomes.";

        let result = analyzer.analyze(text);

        assert!(result.hedging_score > 0.0);
        assert!(!result.hedging_words.is_empty());
    }

    #[test]
    fn test_empty_text() {
        let analyzer = SentimentAnalyzer::new();
        let result = analyzer.analyze("");

        assert_eq!(result.word_count, 0);
        assert_eq!(result.score, 0.0);
    }

    #[test]
    fn test_confidence_analysis() {
        let analyzer = SentimentAnalyzer::new();

        let confident_text = "We will definitely continue our commitment to execution.";
        let conf1 = analyzer.analyze_confidence(confident_text);

        let hedging_text = "We believe we may potentially see improvement.";
        let conf2 = analyzer.analyze_confidence(hedging_text);

        assert!(conf1 > conf2);
    }
}
