//! Text Preprocessing Module
//!
//! This module provides text preprocessing utilities for cryptocurrency
//! and trading-related content. It includes:
//!
//! - Tokenization
//! - Lowercasing and normalization
//! - Stopword removal
//! - Crypto-specific token handling
//! - N-gram generation
//!
//! # Example
//!
//! ```rust
//! use crypto_embeddings::Tokenizer;
//!
//! let tokenizer = Tokenizer::new();
//! let tokens = tokenizer.tokenize("BTC is showing bullish momentum on the 4H chart");
//! println!("{:?}", tokens);
//! ```

use regex::Regex;
use std::collections::HashSet;

/// Tokenizer for cryptocurrency trading texts
pub struct Tokenizer {
    /// Stopwords to remove
    stopwords: HashSet<String>,
    /// Regex for tokenization
    word_regex: Regex,
    /// Regex for crypto symbols (e.g., $BTC, $ETH)
    crypto_symbol_regex: Regex,
    /// Whether to lowercase tokens
    lowercase: bool,
    /// Minimum token length
    min_length: usize,
}

impl Tokenizer {
    /// Create a new tokenizer with default settings
    pub fn new() -> Self {
        Self {
            stopwords: Self::default_stopwords(),
            word_regex: Regex::new(r"[a-zA-Z0-9$#@]+").unwrap(),
            crypto_symbol_regex: Regex::new(r"\$[A-Z]{2,10}").unwrap(),
            lowercase: true,
            min_length: 2,
        }
    }

    /// Create a tokenizer with custom settings
    pub fn with_options(lowercase: bool, min_length: usize, stopwords: Option<HashSet<String>>) -> Self {
        Self {
            stopwords: stopwords.unwrap_or_else(Self::default_stopwords),
            word_regex: Regex::new(r"[a-zA-Z0-9$#@]+").unwrap(),
            crypto_symbol_regex: Regex::new(r"\$[A-Z]{2,10}").unwrap(),
            lowercase,
            min_length,
        }
    }

    /// Default English stopwords plus some common web/trading noise
    fn default_stopwords() -> HashSet<String> {
        let words = vec![
            // Common English stopwords
            "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
            "be", "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "must", "shall", "can", "need", "dare", "ought",
            "used", "it", "its", "this", "that", "these", "those", "i", "you", "he",
            "she", "we", "they", "what", "which", "who", "whom", "whose", "where",
            "when", "why", "how", "all", "each", "every", "both", "few", "more",
            "most", "other", "some", "such", "no", "nor", "not", "only", "own",
            "same", "so", "than", "too", "very", "just", "also", "now", "here",
            "there", "then", "if", "because", "about", "into", "through", "during",
            "before", "after", "above", "below", "between", "under", "again", "once",
            // Web/social media noise
            "http", "https", "www", "com", "co", "io", "org", "net", "rt", "via",
            "amp", "gt", "lt", "re", "fw", "fwd",
        ];
        words.into_iter().map(String::from).collect()
    }

    /// Add custom stopwords
    pub fn add_stopwords(&mut self, words: &[&str]) {
        for word in words {
            self.stopwords.insert(word.to_lowercase());
        }
    }

    /// Tokenize a text string
    pub fn tokenize(&self, text: &str) -> Vec<String> {
        let mut tokens = Vec::new();

        // First, extract crypto symbols (preserve them)
        let _crypto_symbols: Vec<String> = self.crypto_symbol_regex
            .find_iter(text)
            .map(|m| m.as_str().to_uppercase())
            .collect();

        // Extract regular tokens
        for mat in self.word_regex.find_iter(text) {
            let mut token = mat.as_str().to_string();

            // Apply lowercase if enabled (but not for crypto symbols starting with $)
            if self.lowercase && !token.starts_with('$') {
                token = token.to_lowercase();
            }

            // Skip if too short
            if token.len() < self.min_length {
                continue;
            }

            // Skip if stopword
            if self.stopwords.contains(&token.to_lowercase()) {
                continue;
            }

            tokens.push(token);
        }

        tokens
    }

    /// Tokenize into sentences first, then tokenize each sentence
    pub fn tokenize_sentences(&self, text: &str) -> Vec<Vec<String>> {
        let sentence_regex = Regex::new(r"[.!?]+").unwrap();
        let sentences: Vec<&str> = sentence_regex.split(text).collect();

        sentences
            .into_iter()
            .map(|s| self.tokenize(s))
            .filter(|tokens| !tokens.is_empty())
            .collect()
    }

    /// Generate n-grams from tokens
    pub fn ngrams(tokens: &[String], n: usize) -> Vec<String> {
        if tokens.len() < n {
            return Vec::new();
        }

        tokens
            .windows(n)
            .map(|window| window.join("_"))
            .collect()
    }

    /// Detect and combine common phrases (bigrams)
    pub fn detect_phrases(&self, sentences: &[Vec<String>], min_count: usize) -> HashSet<String> {
        use std::collections::HashMap;

        // Count bigrams
        let mut bigram_counts: HashMap<String, usize> = HashMap::new();
        let mut word_counts: HashMap<String, usize> = HashMap::new();

        for sentence in sentences {
            for token in sentence {
                *word_counts.entry(token.clone()).or_insert(0) += 1;
            }

            for window in sentence.windows(2) {
                let bigram = format!("{}_{}", window[0], window[1]);
                *bigram_counts.entry(bigram).or_insert(0) += 1;
            }
        }

        // Filter bigrams by frequency and score
        let mut phrases = HashSet::new();

        for (bigram, count) in bigram_counts {
            if count < min_count {
                continue;
            }

            let parts: Vec<&str> = bigram.split('_').collect();
            if parts.len() != 2 {
                continue;
            }

            let word1_count = word_counts.get(parts[0]).unwrap_or(&1);
            let word2_count = word_counts.get(parts[1]).unwrap_or(&1);

            // PMI-like score
            let score = (count as f64).ln() - ((*word1_count as f64).ln() + (*word2_count as f64).ln());

            if score > 0.0 {
                phrases.insert(bigram);
            }
        }

        phrases
    }

    /// Apply detected phrases to sentences
    pub fn apply_phrases(&self, sentences: &[Vec<String>], phrases: &HashSet<String>) -> Vec<Vec<String>> {
        sentences
            .iter()
            .map(|sentence| {
                let mut result = Vec::new();
                let mut i = 0;

                while i < sentence.len() {
                    if i + 1 < sentence.len() {
                        let bigram = format!("{}_{}", sentence[i], sentence[i + 1]);
                        if phrases.contains(&bigram) {
                            result.push(bigram);
                            i += 2;
                            continue;
                        }
                    }
                    result.push(sentence[i].clone());
                    i += 1;
                }

                result
            })
            .collect()
    }
}

impl Default for Tokenizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Crypto-specific vocabulary and patterns
pub struct CryptoVocab {
    /// Known cryptocurrency symbols
    pub symbols: HashSet<String>,
    /// Trading-specific terms
    pub trading_terms: HashSet<String>,
    /// Technical analysis terms
    pub ta_terms: HashSet<String>,
}

impl CryptoVocab {
    /// Create a new crypto vocabulary with common terms
    pub fn new() -> Self {
        Self {
            symbols: Self::default_symbols(),
            trading_terms: Self::default_trading_terms(),
            ta_terms: Self::default_ta_terms(),
        }
    }

    fn default_symbols() -> HashSet<String> {
        let symbols = vec![
            "BTC", "ETH", "USDT", "USDC", "BNB", "XRP", "SOL", "ADA", "DOGE",
            "DOT", "AVAX", "MATIC", "SHIB", "LTC", "TRX", "ATOM", "LINK", "XMR",
            "ETC", "XLM", "ALGO", "NEAR", "ICP", "FIL", "VET", "AAVE", "SAND",
            "MANA", "AXS", "UNI", "SUSHI", "CRV", "MKR", "COMP", "SNX", "YFI",
        ];
        symbols.into_iter().map(String::from).collect()
    }

    fn default_trading_terms() -> HashSet<String> {
        let terms = vec![
            "long", "short", "buy", "sell", "hold", "hodl", "moon", "dump", "pump",
            "bullish", "bearish", "breakout", "breakdown", "support", "resistance",
            "leverage", "margin", "liquidation", "ath", "atl", "dip", "correction",
            "rally", "consolidation", "accumulation", "distribution", "fomo", "fud",
            "whale", "altcoin", "shitcoin", "defi", "nft", "staking", "yield",
            "apy", "apr", "tvl", "mcap", "volume", "spread", "slippage",
        ];
        terms.into_iter().map(String::from).collect()
    }

    fn default_ta_terms() -> HashSet<String> {
        let terms = vec![
            "rsi", "macd", "ema", "sma", "bollinger", "fibonacci", "fib",
            "divergence", "convergence", "overbought", "oversold", "trend",
            "uptrend", "downtrend", "sideways", "channel", "triangle", "wedge",
            "flag", "pennant", "double_top", "double_bottom", "head_shoulders",
            "candlestick", "doji", "hammer", "engulfing", "morning_star",
            "ichimoku", "vwap", "obv", "atr", "adx", "stochastic",
        ];
        terms.into_iter().map(String::from).collect()
    }

    /// Check if a token is a crypto symbol
    pub fn is_crypto_symbol(&self, token: &str) -> bool {
        self.symbols.contains(&token.to_uppercase())
    }

    /// Check if a token is a trading term
    pub fn is_trading_term(&self, token: &str) -> bool {
        self.trading_terms.contains(&token.to_lowercase())
    }

    /// Check if a token is a TA term
    pub fn is_ta_term(&self, token: &str) -> bool {
        self.ta_terms.contains(&token.to_lowercase())
    }
}

impl Default for CryptoVocab {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer_basic() {
        let tokenizer = Tokenizer::new();
        let tokens = tokenizer.tokenize("BTC is showing bullish momentum");

        assert!(tokens.contains(&"btc".to_string()));
        assert!(tokens.contains(&"bullish".to_string()));
        assert!(tokens.contains(&"momentum".to_string()));
        assert!(!tokens.contains(&"is".to_string())); // stopword
    }

    #[test]
    fn test_tokenizer_sentences() {
        let tokenizer = Tokenizer::new();
        let sentences = tokenizer.tokenize_sentences("BTC is up. ETH is down. SOL is sideways.");

        assert_eq!(sentences.len(), 3);
    }

    #[test]
    fn test_ngrams() {
        let tokens = vec!["btc".to_string(), "showing".to_string(), "bullish".to_string()];
        let bigrams = Tokenizer::ngrams(&tokens, 2);

        assert_eq!(bigrams.len(), 2);
        assert!(bigrams.contains(&"btc_showing".to_string()));
        assert!(bigrams.contains(&"showing_bullish".to_string()));
    }

    #[test]
    fn test_crypto_vocab() {
        let vocab = CryptoVocab::new();

        assert!(vocab.is_crypto_symbol("BTC"));
        assert!(vocab.is_crypto_symbol("btc"));
        assert!(vocab.is_trading_term("bullish"));
        assert!(vocab.is_ta_term("rsi"));
    }
}
