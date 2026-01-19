//! # Crypto Terminology
//!
//! Cryptocurrency-specific slang and terminology for sentiment analysis.

use std::collections::HashMap;

/// Sentiment type classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlangSentiment {
    Positive,
    Negative,
    Neutral,
}

/// Crypto terminology processor
///
/// Handles cryptocurrency-specific slang that standard sentiment
/// models may not understand correctly.
pub struct CryptoTerminology {
    /// Slang to (sentiment, intensity) mapping
    slang: HashMap<String, (SlangSentiment, f64)>,
}

impl Default for CryptoTerminology {
    fn default() -> Self {
        Self::new()
    }
}

impl CryptoTerminology {
    /// Create a new crypto terminology processor
    pub fn new() -> Self {
        let mut slang = HashMap::new();

        // Bullish/positive terms
        let positive_terms = vec![
            ("moon", 0.9),
            ("mooning", 0.95),
            ("hodl", 0.6),
            ("hodling", 0.6),
            ("diamond hands", 0.7),
            ("diamondhands", 0.7),
            ("bullish", 0.8),
            ("pump", 0.7),
            ("pumping", 0.7),
            ("ath", 0.8),
            ("fomo", 0.5),  // Can be context-dependent
            ("wagmi", 0.7),
            ("gm", 0.3),
            ("lfg", 0.7),
            ("to the moon", 0.9),
            ("buy the dip", 0.6),
            ("btd", 0.6),
            ("accumulate", 0.5),
            ("accumulating", 0.5),
            ("stacking", 0.5),
            ("stacking sats", 0.6),
            ("bullrun", 0.8),
            ("bull run", 0.8),
            ("breakout", 0.6),
            ("lambo", 0.7),
            ("green candle", 0.6),
            ("green candles", 0.6),
            ("alpha", 0.5),
        ];

        // Bearish/negative terms
        let negative_terms = vec![
            ("rekt", 0.9),
            ("rekted", 0.9),
            ("dump", 0.8),
            ("dumping", 0.8),
            ("bearish", 0.8),
            ("rug", 0.95),
            ("rug pull", 0.95),
            ("rugpull", 0.95),
            ("rugged", 0.95),
            ("scam", 0.9),
            ("scammer", 0.9),
            ("fud", 0.6),
            ("paper hands", 0.5),
            ("paperhands", 0.5),
            ("ngmi", 0.7),
            ("crash", 0.85),
            ("crashing", 0.85),
            ("bloodbath", 0.85),
            ("capitulation", 0.8),
            ("liquidated", 0.9),
            ("liquidation", 0.9),
            ("exit scam", 0.95),
            ("dead", 0.8),
            ("dead coin", 0.9),
            ("shitcoin", 0.6),
            ("ponzi", 0.9),
            ("bubble", 0.6),
            ("red candle", 0.6),
            ("red candles", 0.6),
            ("bagholder", 0.5),
            ("bag holder", 0.5),
        ];

        // Neutral terms
        let neutral_terms = vec![
            ("dyor", 0.0),
            ("nfa", 0.0),
            ("dca", 0.0),
            ("defi", 0.0),
            ("dex", 0.0),
            ("cex", 0.0),
            ("kyc", 0.0),
            ("apy", 0.0),
            ("tvl", 0.0),
            ("gas", 0.0),
            ("gas fees", 0.0),
            ("whale", 0.0),
            ("whales", 0.0),
            ("altcoin", 0.0),
            ("altcoins", 0.0),
            ("stablecoin", 0.0),
            ("blockchain", 0.0),
            ("smart contract", 0.0),
            ("nft", 0.0),
            ("dao", 0.0),
            ("yield", 0.0),
            ("stake", 0.0),
            ("staking", 0.0),
            ("mining", 0.0),
            ("halving", 0.0),
        ];

        for (term, score) in positive_terms {
            slang.insert(term.to_string(), (SlangSentiment::Positive, score));
        }

        for (term, score) in negative_terms {
            slang.insert(term.to_string(), (SlangSentiment::Negative, score));
        }

        for (term, score) in neutral_terms {
            slang.insert(term.to_string(), (SlangSentiment::Neutral, score));
        }

        Self { slang }
    }

    /// Tokenize text into words, removing punctuation
    fn tokenize(text: &str) -> Vec<String> {
        text.to_lowercase()
            .split_whitespace()
            .map(|t| t.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
            .filter(|t| !t.is_empty())
            .collect()
    }

    /// Check if tokens contain a term (single word or phrase)
    fn tokens_contain_term(tokens: &[String], term: &str) -> bool {
        let term_tokens: Vec<&str> = term.split_whitespace().collect();
        if term_tokens.len() == 1 {
            tokens.iter().any(|t| t == term_tokens[0])
        } else {
            tokens
                .windows(term_tokens.len())
                .any(|w| w.iter().map(|s| s.as_str()).eq(term_tokens.iter().copied()))
        }
    }

    /// Process text for crypto terminology
    pub fn analyze(&self, text: &str) -> CryptoSlangResult {
        let tokens = Self::tokenize(text);
        let mut found_terms: Vec<(String, SlangSentiment, f64)> = Vec::new();
        let mut sentiment_scores: Vec<f64> = Vec::new();

        // Sort terms by length (descending) to match longer phrases first
        let mut sorted_terms: Vec<_> = self.slang.iter().collect();
        sorted_terms.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

        for (term, (sentiment, intensity)) in sorted_terms {
            if Self::tokens_contain_term(&tokens, term) {
                found_terms.push((term.clone(), *sentiment, *intensity));

                match sentiment {
                    SlangSentiment::Positive => sentiment_scores.push(*intensity),
                    SlangSentiment::Negative => sentiment_scores.push(-intensity),
                    SlangSentiment::Neutral => {}
                }
            }
        }

        let avg_score = if sentiment_scores.is_empty() {
            0.0
        } else {
            sentiment_scores.iter().sum::<f64>() / sentiment_scores.len() as f64
        };

        let term_count = found_terms.len();

        CryptoSlangResult {
            score: avg_score.clamp(-1.0, 1.0),
            found_terms,
            term_count,
        }
    }

    /// Check if text contains specific crypto slang
    pub fn contains_term(&self, text: &str, term: &str) -> bool {
        let tokens = Self::tokenize(text);
        Self::tokens_contain_term(&tokens, &term.to_lowercase())
    }

    /// Get sentiment for a specific term
    pub fn get_term_sentiment(&self, term: &str) -> Option<(SlangSentiment, f64)> {
        self.slang.get(&term.to_lowercase()).copied()
    }

    /// Add a custom term
    pub fn add_term(&mut self, term: &str, sentiment: SlangSentiment, intensity: f64) {
        self.slang.insert(term.to_lowercase(), (sentiment, intensity));
    }

    /// Get all bullish terms
    pub fn bullish_terms(&self) -> Vec<&String> {
        self.slang
            .iter()
            .filter(|(_, (s, _))| *s == SlangSentiment::Positive)
            .map(|(t, _)| t)
            .collect()
    }

    /// Get all bearish terms
    pub fn bearish_terms(&self) -> Vec<&String> {
        self.slang
            .iter()
            .filter(|(_, (s, _))| *s == SlangSentiment::Negative)
            .map(|(t, _)| t)
            .collect()
    }
}

/// Result from crypto slang analysis
#[derive(Debug, Clone)]
pub struct CryptoSlangResult {
    /// Overall sentiment score (-1 to 1)
    pub score: f64,
    /// Found terms with their sentiment and intensity
    pub found_terms: Vec<(String, SlangSentiment, f64)>,
    /// Number of slang terms found
    pub term_count: usize,
}

/// Fear & Greed level for crypto markets
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FearGreedLevel {
    ExtremeFear,
    Fear,
    Neutral,
    Greed,
    ExtremeGreed,
}

impl FearGreedLevel {
    /// Convert a score (0-100) to Fear & Greed level
    pub fn from_score(score: f64) -> Self {
        if score < 25.0 {
            FearGreedLevel::ExtremeFear
        } else if score < 45.0 {
            FearGreedLevel::Fear
        } else if score < 55.0 {
            FearGreedLevel::Neutral
        } else if score < 75.0 {
            FearGreedLevel::Greed
        } else {
            FearGreedLevel::ExtremeGreed
        }
    }

    /// Get string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            FearGreedLevel::ExtremeFear => "Extreme Fear",
            FearGreedLevel::Fear => "Fear",
            FearGreedLevel::Neutral => "Neutral",
            FearGreedLevel::Greed => "Greed",
            FearGreedLevel::ExtremeGreed => "Extreme Greed",
        }
    }

    /// Convert sentiment score (-1 to 1) to Fear & Greed (0 to 100)
    pub fn sentiment_to_fg(sentiment: f64) -> f64 {
        (sentiment + 1.0) * 50.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bullish_terms() {
        let crypto = CryptoTerminology::new();
        let result = crypto.analyze("BTC is mooning! Diamond hands! WAGMI");
        assert!(result.score > 0.0);
        assert!(result.term_count >= 2);
    }

    #[test]
    fn test_bearish_terms() {
        let crypto = CryptoTerminology::new();
        let result = crypto.analyze("Got rekt! This is a rug pull scam!");
        assert!(result.score < 0.0);
        assert!(result.term_count >= 2);
    }

    #[test]
    fn test_neutral_terms() {
        let crypto = CryptoTerminology::new();
        let result = crypto.analyze("DYOR and NFA. Always DCA.");
        assert_eq!(result.score, 0.0);
        assert!(result.term_count >= 2);
    }

    #[test]
    fn test_mixed_sentiment() {
        let crypto = CryptoTerminology::new();
        let result = crypto.analyze("HODL through the FUD! Diamond hands win!");
        // Should be positive overall (hodl + diamond hands > fud)
        assert!(result.score > 0.0);
    }

    #[test]
    fn test_fear_greed_levels() {
        assert_eq!(FearGreedLevel::from_score(10.0), FearGreedLevel::ExtremeFear);
        assert_eq!(FearGreedLevel::from_score(50.0), FearGreedLevel::Neutral);
        assert_eq!(FearGreedLevel::from_score(90.0), FearGreedLevel::ExtremeGreed);
    }

    #[test]
    fn test_sentiment_to_fg() {
        assert_eq!(FearGreedLevel::sentiment_to_fg(-1.0), 0.0);
        assert_eq!(FearGreedLevel::sentiment_to_fg(0.0), 50.0);
        assert_eq!(FearGreedLevel::sentiment_to_fg(1.0), 100.0);
    }
}
