//! Text tokenization and preprocessing
//!
//! This module provides tools for:
//! - Text cleaning and normalization
//! - Tokenization (splitting text into words)
//! - Stop word removal
//! - Stemming/lemmatization (basic)

use regex::Regex;
use std::collections::HashSet;
use unicode_segmentation::UnicodeSegmentation;

/// Tokenizer configuration and functionality
#[derive(Debug, Clone)]
pub struct Tokenizer {
    /// Stop words to filter out
    stop_words: HashSet<String>,
    /// Minimum token length
    min_length: usize,
    /// Maximum token length
    max_length: usize,
    /// Convert to lowercase
    lowercase: bool,
    /// Remove numbers
    remove_numbers: bool,
    /// Custom patterns to remove
    remove_patterns: Vec<Regex>,
}

impl Tokenizer {
    /// Create a new tokenizer with default English stop words
    pub fn new() -> Self {
        Self {
            stop_words: default_stop_words(),
            min_length: 2,
            max_length: 50,
            lowercase: true,
            remove_numbers: true,
            remove_patterns: vec![],
        }
    }

    /// Create a tokenizer optimized for cryptocurrency text
    pub fn for_crypto() -> Self {
        let mut tokenizer = Self::new();
        // Add crypto-specific stop words
        tokenizer.add_stop_words(&[
            "crypto",
            "cryptocurrency",
            "blockchain",
            "token",
            "coin",
            "trading",
            "trade",
            "market",
            "price",
            "usdt",
            "usd",
            "btc",
            "eth",
            // Common announcement words
            "announcement",
            "update",
            "new",
            "please",
            "note",
            "important",
            "dear",
            "users",
            "user",
        ]);
        // Keep numbers for crypto context (amounts, percentages)
        tokenizer.remove_numbers = false;
        tokenizer
    }

    /// Add custom stop words
    pub fn add_stop_words(&mut self, words: &[&str]) {
        for word in words {
            self.stop_words.insert(word.to_lowercase());
        }
    }

    /// Set minimum token length
    pub fn min_length(mut self, len: usize) -> Self {
        self.min_length = len;
        self
    }

    /// Set maximum token length
    pub fn max_length(mut self, len: usize) -> Self {
        self.max_length = len;
        self
    }

    /// Enable/disable lowercase conversion
    pub fn lowercase(mut self, enable: bool) -> Self {
        self.lowercase = enable;
        self
    }

    /// Enable/disable number removal
    pub fn remove_numbers(mut self, enable: bool) -> Self {
        self.remove_numbers = enable;
        self
    }

    /// Add a pattern to remove from text
    pub fn add_remove_pattern(&mut self, pattern: &str) -> Result<(), regex::Error> {
        let regex = Regex::new(pattern)?;
        self.remove_patterns.push(regex);
        Ok(())
    }

    /// Clean and normalize text
    pub fn clean(&self, text: &str) -> String {
        let mut cleaned = text.to_string();

        // Apply custom removal patterns
        for pattern in &self.remove_patterns {
            cleaned = pattern.replace_all(&cleaned, " ").to_string();
        }

        // Remove URLs
        let url_pattern = Regex::new(r"https?://\S+").unwrap();
        cleaned = url_pattern.replace_all(&cleaned, " ").to_string();

        // Remove email addresses
        let email_pattern = Regex::new(r"\S+@\S+\.\S+").unwrap();
        cleaned = email_pattern.replace_all(&cleaned, " ").to_string();

        // Remove HTML tags
        let html_pattern = Regex::new(r"<[^>]+>").unwrap();
        cleaned = html_pattern.replace_all(&cleaned, " ").to_string();

        // Remove special characters but keep letters and numbers
        let special_pattern = Regex::new(r"[^\w\s]").unwrap();
        cleaned = special_pattern.replace_all(&cleaned, " ").to_string();

        // Remove numbers if configured
        if self.remove_numbers {
            let number_pattern = Regex::new(r"\b\d+\b").unwrap();
            cleaned = number_pattern.replace_all(&cleaned, " ").to_string();
        }

        // Convert to lowercase if configured
        if self.lowercase {
            cleaned = cleaned.to_lowercase();
        }

        // Normalize whitespace
        let whitespace_pattern = Regex::new(r"\s+").unwrap();
        cleaned = whitespace_pattern.replace_all(&cleaned, " ").to_string();

        cleaned.trim().to_string()
    }

    /// Tokenize text into words
    pub fn tokenize(&self, text: &str) -> Vec<String> {
        let cleaned = self.clean(text);

        cleaned
            .unicode_words()
            .filter(|word| {
                let len = word.len();
                len >= self.min_length
                    && len <= self.max_length
                    && !self.stop_words.contains(&word.to_lowercase())
            })
            .map(|s| s.to_string())
            .collect()
    }

    /// Tokenize multiple documents
    pub fn tokenize_documents(&self, documents: &[String]) -> Vec<Vec<String>> {
        documents.iter().map(|doc| self.tokenize(doc)).collect()
    }

    /// Get unique vocabulary from tokenized documents
    pub fn build_vocabulary(&self, tokenized_docs: &[Vec<String>]) -> Vec<String> {
        let mut vocab_set: HashSet<String> = HashSet::new();

        for doc in tokenized_docs {
            for token in doc {
                vocab_set.insert(token.clone());
            }
        }

        let mut vocab: Vec<String> = vocab_set.into_iter().collect();
        vocab.sort();
        vocab
    }

    /// Get vocabulary with document frequencies
    pub fn vocabulary_with_frequencies(
        &self,
        tokenized_docs: &[Vec<String>],
    ) -> Vec<(String, usize)> {
        let mut doc_freq: std::collections::HashMap<String, usize> = std::collections::HashMap::new();

        for doc in tokenized_docs {
            let unique_tokens: HashSet<&String> = doc.iter().collect();
            for token in unique_tokens {
                *doc_freq.entry(token.clone()).or_insert(0) += 1;
            }
        }

        let mut vocab: Vec<(String, usize)> = doc_freq.into_iter().collect();
        vocab.sort_by(|a, b| b.1.cmp(&a.1)); // Sort by frequency descending
        vocab
    }

    /// Filter vocabulary by document frequency
    pub fn filter_by_document_frequency(
        &self,
        tokenized_docs: &[Vec<String>],
        min_df: usize,
        max_df_ratio: f64,
    ) -> Vec<String> {
        let n_docs = tokenized_docs.len();
        let max_df = (n_docs as f64 * max_df_ratio) as usize;

        self.vocabulary_with_frequencies(tokenized_docs)
            .into_iter()
            .filter(|(_, freq)| *freq >= min_df && *freq <= max_df)
            .map(|(token, _)| token)
            .collect()
    }
}

impl Default for Tokenizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Default English stop words
fn default_stop_words() -> HashSet<String> {
    let words = [
        // Articles
        "a", "an", "the",
        // Pronouns
        "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
        "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself",
        "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which",
        "who", "whom", "this", "that", "these", "those",
        // Verbs
        "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having",
        "do", "does", "did", "doing", "would", "should", "could", "ought", "might", "must",
        "shall", "will", "can", "may",
        // Prepositions
        "at", "by", "for", "from", "in", "into", "of", "on", "to", "with", "about", "against",
        "between", "during", "before", "after", "above", "below", "up", "down", "out", "off",
        "over", "under", "again", "further", "then", "once",
        // Conjunctions
        "and", "but", "or", "nor", "so", "yet", "both", "either", "neither", "not", "only",
        "than", "when", "where", "while", "if", "because", "as", "until", "although",
        // Other common words
        "here", "there", "all", "each", "few", "more", "most", "other", "some", "such", "no",
        "any", "own", "same", "too", "very", "just", "also", "now", "how", "why", "well",
    ];

    words.iter().map(|s| s.to_string()).collect()
}

/// N-gram generator
pub struct NGramGenerator {
    n: usize,
}

impl NGramGenerator {
    /// Create a new n-gram generator
    pub fn new(n: usize) -> Self {
        Self { n }
    }

    /// Generate n-grams from tokens
    pub fn generate(&self, tokens: &[String]) -> Vec<String> {
        if tokens.len() < self.n {
            return vec![];
        }

        tokens
            .windows(self.n)
            .map(|window| window.join("_"))
            .collect()
    }

    /// Generate n-grams for multiple documents
    pub fn generate_for_documents(&self, tokenized_docs: &[Vec<String>]) -> Vec<Vec<String>> {
        tokenized_docs
            .iter()
            .map(|doc| self.generate(doc))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer_basic() {
        let tokenizer = Tokenizer::new();
        let tokens = tokenizer.tokenize("Hello World! This is a test.");

        assert!(!tokens.contains(&"a".to_string())); // Stop word
        assert!(!tokens.contains(&"is".to_string())); // Stop word
        assert!(tokens.contains(&"hello".to_string()));
        assert!(tokens.contains(&"world".to_string()));
        assert!(tokens.contains(&"test".to_string()));
    }

    #[test]
    fn test_tokenizer_crypto() {
        let tokenizer = Tokenizer::for_crypto();
        let text = "Bitcoin BTC reached $50,000 USD. ETH is also performing well.";
        let tokens = tokenizer.tokenize(text);

        // Should filter out common crypto terms
        assert!(!tokens.contains(&"btc".to_string()));
        assert!(!tokens.contains(&"eth".to_string()));
        assert!(!tokens.contains(&"usd".to_string()));
    }

    #[test]
    fn test_clean_text() {
        let tokenizer = Tokenizer::new();
        let cleaned = tokenizer.clean("Visit https://example.com for more info!");

        assert!(!cleaned.contains("https://"));
        assert!(!cleaned.contains("!"));
    }

    #[test]
    fn test_ngram_generator() {
        let generator = NGramGenerator::new(2);
        let tokens = vec![
            "hello".to_string(),
            "world".to_string(),
            "test".to_string(),
        ];
        let ngrams = generator.generate(&tokens);

        assert_eq!(ngrams.len(), 2);
        assert!(ngrams.contains(&"hello_world".to_string()));
        assert!(ngrams.contains(&"world_test".to_string()));
    }

    #[test]
    fn test_vocabulary_building() {
        let tokenizer = Tokenizer::new();
        let docs = vec![
            "bitcoin trading analysis".to_string(),
            "ethereum smart contracts".to_string(),
            "bitcoin ethereum comparison".to_string(),
        ];

        let tokenized = tokenizer.tokenize_documents(&docs);
        let vocab = tokenizer.build_vocabulary(&tokenized);

        assert!(vocab.contains(&"bitcoin".to_string()));
        assert!(vocab.contains(&"ethereum".to_string()));
        assert!(vocab.contains(&"trading".to_string()));
        assert!(vocab.contains(&"analysis".to_string()));
    }
}
