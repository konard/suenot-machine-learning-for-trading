//! # Financial Lexicon
//!
//! Financial sentiment lexicon for rule-based sentiment analysis.

use std::collections::HashMap;

/// Financial sentiment lexicon
///
/// Contains word-sentiment mappings specific to financial text.
pub struct FinancialLexicon {
    /// Word to sentiment score mapping
    words: HashMap<String, f64>,
    /// Negation words
    negations: Vec<String>,
    /// Intensifier words
    intensifiers: HashMap<String, f64>,
}

impl Default for FinancialLexicon {
    fn default() -> Self {
        Self::new()
    }
}

impl FinancialLexicon {
    /// Create a new financial lexicon with default words
    pub fn new() -> Self {
        let mut words = HashMap::new();

        // Positive financial terms
        let positive_words = vec![
            ("bullish", 0.8),
            ("surge", 0.7),
            ("rally", 0.7),
            ("soar", 0.8),
            ("gain", 0.5),
            ("profit", 0.6),
            ("growth", 0.6),
            ("rise", 0.5),
            ("increase", 0.5),
            ("improve", 0.5),
            ("outperform", 0.7),
            ("beat", 0.6),
            ("exceed", 0.6),
            ("strong", 0.5),
            ("positive", 0.5),
            ("optimistic", 0.6),
            ("confident", 0.5),
            ("record", 0.6),
            ("high", 0.4),
            ("upgrade", 0.6),
            ("buy", 0.5),
            ("long", 0.4),
            ("accumulate", 0.5),
            ("breakout", 0.6),
            ("momentum", 0.4),
            ("support", 0.4),
            ("recovery", 0.5),
            ("rebound", 0.5),
        ];

        // Negative financial terms
        let negative_words = vec![
            ("bearish", -0.8),
            ("crash", -0.9),
            ("plunge", -0.8),
            ("drop", -0.6),
            ("fall", -0.5),
            ("decline", -0.6),
            ("loss", -0.6),
            ("down", -0.4),
            ("decrease", -0.5),
            ("weak", -0.5),
            ("negative", -0.5),
            ("pessimistic", -0.6),
            ("concern", -0.5),
            ("worry", -0.5),
            ("fear", -0.6),
            ("risk", -0.4),
            ("volatile", -0.3),
            ("uncertainty", -0.5),
            ("miss", -0.6),
            ("disappoint", -0.7),
            ("underperform", -0.6),
            ("downgrade", -0.6),
            ("sell", -0.5),
            ("short", -0.4),
            ("dump", -0.7),
            ("breakdown", -0.6),
            ("resistance", -0.3),
            ("correction", -0.4),
            ("crisis", -0.8),
            ("warning", -0.5),
            ("trouble", -0.6),
            ("problem", -0.5),
            ("fail", -0.7),
            ("scam", -0.9),
            ("fraud", -0.9),
        ];

        for (word, score) in positive_words {
            words.insert(word.to_string(), score);
        }

        for (word, score) in negative_words {
            words.insert(word.to_string(), score);
        }

        let negations = vec![
            "not", "no", "never", "neither", "nobody", "nothing",
            "nowhere", "none", "cannot", "cant", "don't", "dont",
            "doesn't", "doesnt", "didn't", "didnt", "won't", "wont",
            "wouldn't", "wouldnt", "shouldn't", "shouldnt", "couldn't",
            "couldnt", "isn't", "isnt", "aren't", "arent", "wasn't",
            "wasnt", "weren't", "werent", "hardly", "barely", "scarcely",
        ]
        .into_iter()
        .map(String::from)
        .collect();

        let mut intensifiers = HashMap::new();
        intensifiers.insert("very".to_string(), 1.5);
        intensifiers.insert("extremely".to_string(), 2.0);
        intensifiers.insert("highly".to_string(), 1.5);
        intensifiers.insert("significantly".to_string(), 1.5);
        intensifiers.insert("substantially".to_string(), 1.5);
        intensifiers.insert("dramatically".to_string(), 1.8);
        intensifiers.insert("massively".to_string(), 1.8);
        intensifiers.insert("hugely".to_string(), 1.7);
        intensifiers.insert("slightly".to_string(), 0.5);
        intensifiers.insert("somewhat".to_string(), 0.7);
        intensifiers.insert("marginally".to_string(), 0.5);
        intensifiers.insert("relatively".to_string(), 0.8);

        Self {
            words,
            negations,
            intensifiers,
        }
    }

    /// Get sentiment score for a word
    pub fn get_score(&self, word: &str) -> Option<f64> {
        self.words.get(&word.to_lowercase()).copied()
    }

    /// Check if a word is a negation
    pub fn is_negation(&self, word: &str) -> bool {
        self.negations.contains(&word.to_lowercase())
    }

    /// Get intensifier multiplier
    pub fn get_intensifier(&self, word: &str) -> Option<f64> {
        self.intensifiers.get(&word.to_lowercase()).copied()
    }

    /// Calculate sentiment score for a text
    ///
    /// Uses simple rule-based approach:
    /// 1. Look up word scores
    /// 2. Apply negation handling (flip sign for words after negation)
    /// 3. Apply intensifier scaling
    pub fn analyze(&self, text: &str) -> LexiconResult {
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut scores: Vec<f64> = Vec::new();
        let mut matched_words: Vec<(String, f64)> = Vec::new();

        let mut negate_next = false;
        let mut intensifier: f64 = 1.0;

        for word in words {
            let word_lower = word.to_lowercase();

            // Check for negation
            if self.is_negation(&word_lower) {
                negate_next = true;
                continue;
            }

            // Check for intensifier
            if let Some(mult) = self.get_intensifier(&word_lower) {
                intensifier = mult;
                continue;
            }

            // Check for sentiment word
            if let Some(mut score) = self.get_score(&word_lower) {
                // Apply negation
                if negate_next {
                    score = -score;
                    negate_next = false;
                }

                // Apply intensifier
                score *= intensifier;
                intensifier = 1.0;

                scores.push(score);
                matched_words.push((word_lower, score));
            } else {
                // Reset modifiers if word not found
                negate_next = false;
                intensifier = 1.0;
            }
        }

        let sentiment_score = if scores.is_empty() {
            0.0
        } else {
            scores.iter().sum::<f64>() / scores.len() as f64
        };

        LexiconResult {
            score: sentiment_score.clamp(-1.0, 1.0),
            matched_words,
            word_count: scores.len(),
        }
    }

    /// Add a custom word to the lexicon
    pub fn add_word(&mut self, word: &str, score: f64) {
        self.words.insert(word.to_lowercase(), score);
    }
}

/// Result from lexicon-based analysis
#[derive(Debug, Clone)]
pub struct LexiconResult {
    /// Overall sentiment score (-1 to 1)
    pub score: f64,
    /// Words that matched with their scores
    pub matched_words: Vec<(String, f64)>,
    /// Number of sentiment words found
    pub word_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_positive_words() {
        let lexicon = FinancialLexicon::new();
        assert!(lexicon.get_score("bullish").unwrap() > 0.0);
        assert!(lexicon.get_score("surge").unwrap() > 0.0);
    }

    #[test]
    fn test_negative_words() {
        let lexicon = FinancialLexicon::new();
        assert!(lexicon.get_score("bearish").unwrap() < 0.0);
        assert!(lexicon.get_score("crash").unwrap() < 0.0);
    }

    #[test]
    fn test_negation() {
        let lexicon = FinancialLexicon::new();
        assert!(lexicon.is_negation("not"));
        assert!(lexicon.is_negation("never"));
        assert!(!lexicon.is_negation("bullish"));
    }

    #[test]
    fn test_analyze_positive() {
        let lexicon = FinancialLexicon::new();
        let result = lexicon.analyze("Bitcoin is bullish with strong momentum");
        assert!(result.score > 0.0);
    }

    #[test]
    fn test_analyze_negative() {
        let lexicon = FinancialLexicon::new();
        let result = lexicon.analyze("Market crash fears grow as prices plunge");
        assert!(result.score < 0.0);
    }

    #[test]
    fn test_negation_handling() {
        let lexicon = FinancialLexicon::new();
        let positive = lexicon.analyze("the market is bullish");
        let negated = lexicon.analyze("the market is not bullish");
        assert!(positive.score > 0.0);
        assert!(negated.score < 0.0);
    }

    #[test]
    fn test_intensifier_handling() {
        let lexicon = FinancialLexicon::new();
        let normal = lexicon.analyze("market is bullish");
        let intensified = lexicon.analyze("market is extremely bullish");
        assert!(intensified.score > normal.score);
    }
}
