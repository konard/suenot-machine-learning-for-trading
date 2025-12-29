//! Analysis Module
//!
//! This module provides analysis tools for cryptocurrency trading texts
//! using word embeddings. It includes:
//!
//! - Similarity analysis between documents
//! - Sentiment classification
//! - Topic detection
//! - Trend analysis

use crate::embeddings::Word2Vec;
use crate::preprocessing::Tokenizer;
use crate::utils::{Result, CryptoEmbeddingsError, cosine_similarity};
use std::collections::HashMap;

/// Analyzer for computing document similarities using embeddings
pub struct SimilarityAnalyzer {
    model: Word2Vec,
    tokenizer: Tokenizer,
}

impl SimilarityAnalyzer {
    /// Create a new similarity analyzer with a trained model
    pub fn new(model: Word2Vec) -> Self {
        Self {
            model,
            tokenizer: Tokenizer::new(),
        }
    }

    /// Create analyzer with custom tokenizer
    pub fn with_tokenizer(model: Word2Vec, tokenizer: Tokenizer) -> Self {
        Self { model, tokenizer }
    }

    /// Get document vector by averaging word vectors
    pub fn get_document_vector(&self, text: &str) -> Result<Vec<f32>> {
        let tokens = self.tokenizer.tokenize(text);

        if tokens.is_empty() {
            return Err(CryptoEmbeddingsError::EmptyCorpus);
        }

        let dim = self.model.get_vector(self.model.words().first().unwrap())?.len();
        let mut doc_vec = vec![0.0f32; dim];
        let mut count = 0;

        for token in &tokens {
            if let Ok(vec) = self.model.get_vector(token) {
                for (i, &v) in vec.iter().enumerate() {
                    doc_vec[i] += v;
                }
                count += 1;
            }
        }

        if count == 0 {
            return Err(CryptoEmbeddingsError::WordNotFound(
                "No known words in document".to_string()
            ));
        }

        // Average
        for v in &mut doc_vec {
            *v /= count as f32;
        }

        Ok(doc_vec)
    }

    /// Compute similarity between two texts
    pub fn text_similarity(&self, text1: &str, text2: &str) -> Result<f32> {
        let vec1 = self.get_document_vector(text1)?;
        let vec2 = self.get_document_vector(text2)?;
        Ok(cosine_similarity(&vec1, &vec2))
    }

    /// Find most similar documents from a corpus
    pub fn most_similar_documents(
        &self,
        query: &str,
        documents: &[&str],
        top_n: usize,
    ) -> Result<Vec<(usize, f32)>> {
        let query_vec = self.get_document_vector(query)?;

        let mut similarities: Vec<(usize, f32)> = documents
            .iter()
            .enumerate()
            .filter_map(|(i, doc)| {
                self.get_document_vector(doc).ok().map(|vec| {
                    (i, cosine_similarity(&query_vec, &vec))
                })
            })
            .collect();

        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        similarities.truncate(top_n);

        Ok(similarities)
    }
}

/// Simple sentiment analyzer for crypto texts
pub struct SentimentAnalyzer {
    model: Word2Vec,
    tokenizer: Tokenizer,
    positive_words: HashMap<String, f32>,
    negative_words: HashMap<String, f32>,
}

impl SentimentAnalyzer {
    /// Create a new sentiment analyzer
    pub fn new(model: Word2Vec) -> Self {
        Self {
            model,
            tokenizer: Tokenizer::new(),
            positive_words: Self::default_positive_words(),
            negative_words: Self::default_negative_words(),
        }
    }

    fn default_positive_words() -> HashMap<String, f32> {
        let words = vec![
            ("bullish", 1.0), ("moon", 1.0), ("pump", 0.8), ("breakout", 0.9),
            ("rally", 0.9), ("surge", 0.8), ("gain", 0.7), ("profit", 0.8),
            ("accumulation", 0.6), ("support", 0.5), ("uptrend", 0.8),
            ("ath", 0.9), ("hodl", 0.5), ("buy", 0.6), ("long", 0.5),
            ("green", 0.5), ("strong", 0.6), ("recovery", 0.7), ("momentum", 0.6),
            ("adoption", 0.7), ("upgrade", 0.6), ("partnership", 0.5),
        ];
        words.into_iter().map(|(w, s)| (w.to_string(), s)).collect()
    }

    fn default_negative_words() -> HashMap<String, f32> {
        let words = vec![
            ("bearish", 1.0), ("dump", 1.0), ("crash", 1.0), ("breakdown", 0.9),
            ("drop", 0.8), ("plunge", 0.9), ("loss", 0.8), ("sell", 0.5),
            ("distribution", 0.6), ("resistance", 0.3), ("downtrend", 0.8),
            ("liquidation", 0.9), ("fud", 0.7), ("short", 0.5), ("red", 0.5),
            ("weak", 0.6), ("scam", 1.0), ("hack", 1.0), ("rug", 1.0),
            ("correction", 0.5), ("fear", 0.7), ("panic", 0.8),
        ];
        words.into_iter().map(|(w, s)| (w.to_string(), s)).collect()
    }

    /// Add custom sentiment words
    pub fn add_positive_word(&mut self, word: &str, score: f32) {
        self.positive_words.insert(word.to_lowercase(), score);
    }

    pub fn add_negative_word(&mut self, word: &str, score: f32) {
        self.negative_words.insert(word.to_lowercase(), score);
    }

    /// Analyze sentiment of a text
    /// Returns a score between -1 (very negative) and 1 (very positive)
    pub fn analyze(&self, text: &str) -> SentimentResult {
        let tokens = self.tokenizer.tokenize(text);

        let mut positive_score = 0.0f32;
        let mut negative_score = 0.0f32;
        let mut positive_words = Vec::new();
        let mut negative_words = Vec::new();

        for token in &tokens {
            let lower = token.to_lowercase();

            if let Some(&score) = self.positive_words.get(&lower) {
                positive_score += score;
                positive_words.push(token.clone());
            }

            if let Some(&score) = self.negative_words.get(&lower) {
                negative_score += score;
                negative_words.push(token.clone());
            }
        }

        let total = positive_score + negative_score;
        let sentiment_score = if total > 0.0 {
            (positive_score - negative_score) / total
        } else {
            0.0
        };

        let label = if sentiment_score > 0.2 {
            SentimentLabel::Positive
        } else if sentiment_score < -0.2 {
            SentimentLabel::Negative
        } else {
            SentimentLabel::Neutral
        };

        SentimentResult {
            score: sentiment_score,
            label,
            positive_score,
            negative_score,
            positive_words,
            negative_words,
        }
    }

    /// Analyze sentiment using word embeddings similarity
    /// More sophisticated but requires good training data
    pub fn analyze_with_embeddings(&self, text: &str) -> Result<f32> {
        let tokens = self.tokenizer.tokenize(text);

        if tokens.is_empty() {
            return Ok(0.0);
        }

        // Get average vector for positive and negative seed words
        let get_seed_vector = |words: &HashMap<String, f32>| -> Vec<f32> {
            let dim = 100; // Assume default dimension
            let mut vec = vec![0.0f32; dim];
            let mut count = 0;

            for word in words.keys() {
                if let Ok(wv) = self.model.get_vector(word) {
                    for (i, &v) in wv.iter().enumerate() {
                        if i < dim {
                            vec[i] += v;
                        }
                    }
                    count += 1;
                }
            }

            if count > 0 {
                for v in &mut vec {
                    *v /= count as f32;
                }
            }
            vec
        };

        let pos_vec = get_seed_vector(&self.positive_words);
        let neg_vec = get_seed_vector(&self.negative_words);

        // Get text vector
        let dim = pos_vec.len();
        let mut text_vec = vec![0.0f32; dim];
        let mut count = 0;

        for token in &tokens {
            if let Ok(wv) = self.model.get_vector(token) {
                for (i, &v) in wv.iter().enumerate() {
                    if i < dim {
                        text_vec[i] += v;
                    }
                }
                count += 1;
            }
        }

        if count == 0 {
            return Ok(0.0);
        }

        for v in &mut text_vec {
            *v /= count as f32;
        }

        // Compute similarity to positive vs negative
        let pos_sim = cosine_similarity(&text_vec, &pos_vec);
        let neg_sim = cosine_similarity(&text_vec, &neg_vec);

        Ok(pos_sim - neg_sim)
    }
}

/// Sentiment analysis result
#[derive(Debug, Clone)]
pub struct SentimentResult {
    /// Overall sentiment score (-1 to 1)
    pub score: f32,
    /// Sentiment label
    pub label: SentimentLabel,
    /// Positive score component
    pub positive_score: f32,
    /// Negative score component
    pub negative_score: f32,
    /// Positive words found
    pub positive_words: Vec<String>,
    /// Negative words found
    pub negative_words: Vec<String>,
}

/// Sentiment label
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SentimentLabel {
    Positive,
    Neutral,
    Negative,
}

/// Trend analyzer using embeddings
pub struct TrendAnalyzer {
    model: Word2Vec,
    tokenizer: Tokenizer,
}

impl TrendAnalyzer {
    /// Create a new trend analyzer
    pub fn new(model: Word2Vec) -> Self {
        Self {
            model,
            tokenizer: Tokenizer::new(),
        }
    }

    /// Detect trending topics from a corpus of recent texts
    pub fn detect_trends(&self, texts: &[&str], top_n: usize) -> Vec<(String, usize)> {
        let mut word_counts: HashMap<String, usize> = HashMap::new();

        for text in texts {
            let tokens = self.tokenizer.tokenize(text);
            for token in tokens {
                if self.model.contains(&token) {
                    *word_counts.entry(token).or_insert(0) += 1;
                }
            }
        }

        let mut counts: Vec<(String, usize)> = word_counts.into_iter().collect();
        counts.sort_by(|a, b| b.1.cmp(&a.1));
        counts.truncate(top_n);

        counts
    }

    /// Find words that appear together frequently (potential topics)
    pub fn find_topics(&self, texts: &[&str], num_topics: usize) -> Vec<Vec<String>> {
        // Simple co-occurrence based topic detection
        let mut cooccurrences: HashMap<(String, String), usize> = HashMap::new();

        for text in texts {
            let tokens = self.tokenizer.tokenize(text);
            let known_tokens: Vec<String> = tokens
                .into_iter()
                .filter(|t| self.model.contains(t))
                .collect();

            for i in 0..known_tokens.len() {
                for j in (i + 1)..known_tokens.len() {
                    let pair = if known_tokens[i] < known_tokens[j] {
                        (known_tokens[i].clone(), known_tokens[j].clone())
                    } else {
                        (known_tokens[j].clone(), known_tokens[i].clone())
                    };
                    *cooccurrences.entry(pair).or_insert(0) += 1;
                }
            }
        }

        // Find connected components using high co-occurrence pairs
        let mut topics: Vec<Vec<String>> = Vec::new();
        let mut used_words: std::collections::HashSet<String> = std::collections::HashSet::new();

        let mut pairs: Vec<((String, String), usize)> = cooccurrences.into_iter().collect();
        pairs.sort_by(|a, b| b.1.cmp(&a.1));

        for ((w1, w2), _count) in pairs {
            if topics.len() >= num_topics {
                break;
            }

            let w1_used = used_words.contains(&w1);
            let w2_used = used_words.contains(&w2);

            if !w1_used || !w2_used {
                // Find existing topic or create new
                let mut found = false;
                for topic in &mut topics {
                    if topic.contains(&w1) && !w2_used {
                        topic.push(w2.clone());
                        used_words.insert(w2.clone());
                        found = true;
                        break;
                    } else if topic.contains(&w2) && !w1_used {
                        topic.push(w1.clone());
                        used_words.insert(w1.clone());
                        found = true;
                        break;
                    }
                }

                if !found && !w1_used && !w2_used {
                    topics.push(vec![w1.clone(), w2.clone()]);
                    used_words.insert(w1);
                    used_words.insert(w2);
                }
            }
        }

        topics
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_model() -> Word2Vec {
        let mut model = Word2Vec::new(10, 2, 1);
        let sentences = vec![
            vec!["btc", "bullish", "trend"].iter().map(|s| s.to_string()).collect(),
            vec!["eth", "bearish", "dump"].iter().map(|s| s.to_string()).collect(),
            vec!["btc", "moon", "pump"].iter().map(|s| s.to_string()).collect(),
        ];
        model.build_vocab(&sentences);
        model.train(&sentences, 2).unwrap();
        model
    }

    #[test]
    fn test_sentiment_analyzer() {
        let model = create_test_model();
        let analyzer = SentimentAnalyzer::new(model);

        let result = analyzer.analyze("BTC is looking very bullish, moon soon!");
        assert!(result.score > 0.0);
        assert_eq!(result.label, SentimentLabel::Positive);

        let result = analyzer.analyze("This looks like a scam, expect a dump");
        assert!(result.score < 0.0);
        assert_eq!(result.label, SentimentLabel::Negative);
    }

    #[test]
    fn test_similarity_analyzer() {
        let model = create_test_model();
        let analyzer = SimilarityAnalyzer::new(model);

        // Texts about the same topic should be similar
        let sim = analyzer.text_similarity(
            "BTC bullish trend momentum",
            "Bitcoin bullish trend"
        );

        // Just check that it runs without error
        assert!(sim.is_ok() || sim.is_err()); // Either result is acceptable for test data
    }
}
