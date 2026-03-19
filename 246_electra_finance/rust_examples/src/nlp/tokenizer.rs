//! Simple tokenizer for financial text processing
//!
//! This module provides a lightweight tokenizer for processing
//! financial news text in the ELECTRA pipeline.

use std::collections::HashMap;

/// Special token IDs
pub const PAD_TOKEN_ID: u32 = 0;
pub const UNK_TOKEN_ID: u32 = 1;
pub const CLS_TOKEN_ID: u32 = 2;
pub const SEP_TOKEN_ID: u32 = 3;

/// Simple whitespace-based tokenizer with vocabulary support
pub struct SimpleTokenizer {
    vocab: HashMap<String, u32>,
    id_to_token: HashMap<u32, String>,
    max_length: usize,
}

impl SimpleTokenizer {
    /// Create a new tokenizer with a default financial vocabulary
    pub fn new(max_length: usize) -> Self {
        let mut tokenizer = Self {
            vocab: HashMap::new(),
            id_to_token: HashMap::new(),
            max_length,
        };

        // Add special tokens
        tokenizer.add_token("[PAD]", PAD_TOKEN_ID);
        tokenizer.add_token("[UNK]", UNK_TOKEN_ID);
        tokenizer.add_token("[CLS]", CLS_TOKEN_ID);
        tokenizer.add_token("[SEP]", SEP_TOKEN_ID);

        // Add common financial vocabulary
        let financial_terms = vec![
            "stock", "price", "market", "trading", "volume", "shares",
            "revenue", "earnings", "profit", "loss", "growth", "decline",
            "bullish", "bearish", "rally", "crash", "surge", "plunge",
            "bitcoin", "crypto", "blockchain", "token", "exchange",
            "buy", "sell", "hold", "long", "short", "position",
            "resistance", "support", "breakout", "trend", "momentum",
            "volatility", "risk", "return", "yield", "dividend",
            "portfolio", "asset", "fund", "index", "benchmark",
            "analyst", "forecast", "estimate", "consensus", "target",
            "beat", "miss", "exceed", "below", "above", "strong",
            "weak", "record", "quarterly", "annual", "report",
            "the", "a", "an", "is", "was", "are", "were", "has", "had",
            "of", "in", "on", "at", "to", "for", "with", "by", "from",
            "up", "down", "after", "before", "today", "yesterday",
            "rose", "fell", "dropped", "climbed", "gained", "lost",
            "increased", "decreased", "jumped", "tumbled", "surged",
            "pushed", "drove", "led", "showed", "traded", "remained",
            "heavy", "light", "high", "low", "flat", "mixed",
            "pressure", "sentiment", "session", "investors", "traders",
        ];

        let mut next_id = 4u32;
        for term in financial_terms {
            tokenizer.add_token(term, next_id);
            next_id += 1;
        }

        tokenizer
    }

    /// Add a token to the vocabulary
    fn add_token(&mut self, token: &str, id: u32) {
        self.vocab.insert(token.to_lowercase(), id);
        self.id_to_token.insert(id, token.to_lowercase());
    }

    /// Build vocabulary from a corpus of texts
    pub fn build_vocab(&mut self, texts: &[String], max_vocab_size: usize) {
        let mut word_counts: HashMap<String, usize> = HashMap::new();

        for text in texts {
            for word in text.to_lowercase().split_whitespace() {
                let clean_word = word
                    .trim_matches(|c: char| !c.is_alphanumeric() && c != '$' && c != '%')
                    .to_string();
                if !clean_word.is_empty() {
                    *word_counts.entry(clean_word).or_insert(0) += 1;
                }
            }
        }

        let mut sorted_words: Vec<_> = word_counts.into_iter().collect();
        sorted_words.sort_by(|a, b| b.1.cmp(&a.1));

        let mut next_id = self.vocab.len() as u32;
        for (word, _) in sorted_words.iter().take(max_vocab_size) {
            if !self.vocab.contains_key(word) {
                self.add_token(word, next_id);
                next_id += 1;
            }
        }
    }

    /// Tokenize a single text into token IDs
    pub fn tokenize(&self, text: &str) -> (Vec<u32>, Vec<u32>) {
        let words: Vec<&str> = text.to_lowercase().split_whitespace().collect();
        let max_tokens = self.max_length - 2; // Reserve space for [CLS] and [SEP]

        let mut token_ids = vec![CLS_TOKEN_ID];
        for word in words.iter().take(max_tokens) {
            let clean_word = word.trim_matches(|c: char| !c.is_alphanumeric() && c != '$' && c != '%');
            let id = self.vocab.get(clean_word).copied().unwrap_or(UNK_TOKEN_ID);
            token_ids.push(id);
        }
        token_ids.push(SEP_TOKEN_ID);

        let seq_len = token_ids.len();
        let mut attention_mask = vec![1u32; seq_len];

        // Pad to max_length
        while token_ids.len() < self.max_length {
            token_ids.push(PAD_TOKEN_ID);
            attention_mask.push(0);
        }

        (token_ids, attention_mask)
    }

    /// Tokenize a batch of texts
    pub fn tokenize_batch(&self, texts: &[String]) -> (Vec<Vec<u32>>, Vec<Vec<u32>>) {
        let mut all_ids = Vec::with_capacity(texts.len());
        let mut all_masks = Vec::with_capacity(texts.len());

        for text in texts {
            let (ids, mask) = self.tokenize(text);
            all_ids.push(ids);
            all_masks.push(mask);
        }

        (all_ids, all_masks)
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Decode token IDs back to text
    pub fn decode(&self, token_ids: &[u32]) -> String {
        token_ids
            .iter()
            .filter(|&&id| id != PAD_TOKEN_ID && id != CLS_TOKEN_ID && id != SEP_TOKEN_ID)
            .filter_map(|id| self.id_to_token.get(id))
            .cloned()
            .collect::<Vec<_>>()
            .join(" ")
    }
}
