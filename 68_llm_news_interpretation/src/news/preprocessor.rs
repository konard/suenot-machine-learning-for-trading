//! News text preprocessing functionality

use super::types::{NewsItem, RawNews};
use sha2::{Digest, Sha256};
use std::collections::HashSet;

/// Preprocessor for cleaning and normalizing news text
#[derive(Debug, Clone)]
pub struct NewsPreprocessor {
    /// Common crypto symbols to extract
    crypto_symbols: HashSet<String>,
    /// Stop words to filter
    stop_words: HashSet<String>,
}

impl NewsPreprocessor {
    /// Create a new preprocessor with default settings
    pub fn new() -> Self {
        let crypto_symbols: HashSet<String> = [
            "BTC", "ETH", "SOL", "AVAX", "ARB", "OP", "MATIC", "BNB", "XRP", "ADA",
            "DOT", "LINK", "UNI", "AAVE", "COMP", "MKR", "SNX", "CRV", "LDO", "RPL",
            "DOGE", "SHIB", "PEPE", "WIF", "BONK", "APE", "SAND", "MANA", "AXS", "GMT",
            "ATOM", "NEAR", "FTM", "ALGO", "VET", "HBAR", "ICP", "FIL", "THETA", "GRT",
            "BITCOIN", "ETHEREUM", "SOLANA", "AVALANCHE", "ARBITRUM", "OPTIMISM",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();

        let stop_words: HashSet<String> = [
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "must", "shall", "can", "need", "dare",
            "ought", "used", "to", "of", "in", "for", "on", "with", "at", "by",
            "from", "as", "into", "through", "during", "before", "after",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();

        Self {
            crypto_symbols,
            stop_words,
        }
    }

    /// Process a raw news item into a cleaned NewsItem
    pub fn process(&self, raw: NewsItem) -> NewsItem {
        let cleaned_text = self.clean_text(&raw.text);
        let entities = self.extract_entities(&raw.text);

        NewsItem {
            text: cleaned_text,
            entities,
            ..raw
        }
    }

    /// Clean and normalize text
    pub fn clean_text(&self, text: &str) -> String {
        let mut cleaned = text.to_string();

        // Remove URLs
        cleaned = self.remove_urls(&cleaned);

        // Remove excessive whitespace
        cleaned = self.normalize_whitespace(&cleaned);

        // Remove common noise patterns
        cleaned = self.remove_noise(&cleaned);

        cleaned.trim().to_string()
    }

    /// Remove URLs from text
    fn remove_urls(&self, text: &str) -> String {
        let url_pattern = regex::Regex::new(r"https?://\S+").unwrap();
        url_pattern.replace_all(text, "").to_string()
    }

    /// Normalize whitespace
    fn normalize_whitespace(&self, text: &str) -> String {
        let ws_pattern = regex::Regex::new(r"\s+").unwrap();
        ws_pattern.replace_all(text, " ").to_string()
    }

    /// Remove common noise patterns
    fn remove_noise(&self, text: &str) -> String {
        let mut cleaned = text.to_string();

        // Remove RT prefix
        if cleaned.starts_with("RT ") {
            cleaned = cleaned[3..].to_string();
        }

        // Remove @mentions (but keep for entity extraction first)
        let mention_pattern = regex::Regex::new(r"@\w+").unwrap();
        cleaned = mention_pattern.replace_all(&cleaned, "").to_string();

        // Remove hashtag symbols but keep the text
        cleaned = cleaned.replace('#', "");

        cleaned
    }

    /// Extract entities (crypto symbols, companies, etc.) from text
    pub fn extract_entities(&self, text: &str) -> Vec<String> {
        let mut entities = Vec::new();
        let upper_text = text.to_uppercase();
        let words: Vec<&str> = upper_text.split_whitespace().collect();

        for word in words {
            // Remove punctuation for matching
            let clean_word: String = word.chars().filter(|c| c.is_alphanumeric()).collect();

            if self.crypto_symbols.contains(&clean_word) {
                if !entities.contains(&clean_word) {
                    entities.push(clean_word);
                }
            }
        }

        // Also check for common patterns
        let patterns = [
            (r"\$([A-Z]{2,5})", 1), // $BTC style
            (r"#([A-Z]{2,5})", 1),  // #BTC style
        ];

        for (pattern, group) in patterns {
            if let Ok(re) = regex::Regex::new(pattern) {
                for cap in re.captures_iter(&text.to_uppercase()) {
                    if let Some(m) = cap.get(group) {
                        let symbol = m.as_str().to_string();
                        if self.crypto_symbols.contains(&symbol) && !entities.contains(&symbol) {
                            entities.push(symbol);
                        }
                    }
                }
            }
        }

        entities
    }

    /// Generate a hash of the content for deduplication
    pub fn hash_content(&self, text: &str) -> String {
        let normalized = text.to_lowercase();
        let normalized: String = normalized
            .chars()
            .filter(|c| c.is_alphanumeric() || c.is_whitespace())
            .collect();

        let mut hasher = Sha256::new();
        hasher.update(normalized.as_bytes());
        hex::encode(hasher.finalize())
    }

    /// Check if two texts are semantically similar (simple approach)
    pub fn is_similar(&self, text1: &str, text2: &str, threshold: f64) -> bool {
        let hash1 = self.hash_content(text1);
        let hash2 = self.hash_content(text2);

        if hash1 == hash2 {
            return true;
        }

        // Simple word overlap similarity
        let text1_lower = text1.to_lowercase();
        let text2_lower = text2.to_lowercase();
        let words1: HashSet<&str> = text1_lower.split_whitespace().collect();
        let words2: HashSet<&str> = text2_lower.split_whitespace().collect();

        let intersection = words1.intersection(&words2).count();
        let union = words1.union(&words2).count();

        if union == 0 {
            return false;
        }

        let jaccard = intersection as f64 / union as f64;
        jaccard >= threshold
    }
}

impl Default for NewsPreprocessor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clean_text() {
        let preprocessor = NewsPreprocessor::new();

        let text = "Check out https://example.com for more info!";
        let cleaned = preprocessor.clean_text(text);
        assert!(!cleaned.contains("https://"));

        let text = "RT @user: Bitcoin is up!";
        let cleaned = preprocessor.clean_text(text);
        assert!(!cleaned.starts_with("RT"));
        assert!(!cleaned.contains("@user"));
    }

    #[test]
    fn test_extract_entities() {
        let preprocessor = NewsPreprocessor::new();

        let text = "Bitcoin and Ethereum are pumping! $SOL also up.";
        let entities = preprocessor.extract_entities(text);

        assert!(entities.contains(&"BITCOIN".to_string()));
        assert!(entities.contains(&"ETHEREUM".to_string()));
        assert!(entities.contains(&"SOL".to_string()));
    }

    #[test]
    fn test_hash_content() {
        let preprocessor = NewsPreprocessor::new();

        let hash1 = preprocessor.hash_content("Hello World");
        let hash2 = preprocessor.hash_content("hello world");

        assert_eq!(hash1, hash2); // Should be same after normalization
    }

    #[test]
    fn test_similarity() {
        let preprocessor = NewsPreprocessor::new();

        let text1 = "Bitcoin price surges to new high";
        let text2 = "Bitcoin price surges to new high today";
        let text3 = "Ethereum network upgrade announced";

        assert!(preprocessor.is_similar(text1, text2, 0.6));
        assert!(!preprocessor.is_similar(text1, text3, 0.6));
    }
}
