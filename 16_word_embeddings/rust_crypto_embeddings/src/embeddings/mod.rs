//! Word Embeddings Module
//!
//! This module provides a Word2Vec-style implementation for learning
//! word embeddings from cryptocurrency trading texts.
//!
//! # Architecture
//!
//! The implementation uses the Skip-gram model with negative sampling,
//! which is efficient and produces high-quality embeddings for trading vocabulary.
//!
//! # Example
//!
//! ```rust,no_run
//! use crypto_embeddings::Word2Vec;
//!
//! let mut model = Word2Vec::new(100, 5, 5);
//!
//! let sentences = vec![
//!     vec!["btc".to_string(), "bullish".to_string(), "trend".to_string()],
//!     vec!["eth".to_string(), "breakout".to_string(), "resistance".to_string()],
//! ];
//!
//! model.build_vocab(&sentences);
//! model.train(&sentences, 5).unwrap();
//!
//! let similar = model.most_similar("btc", 5).unwrap();
//! ```

use ndarray::Array2;
use rand::Rng;
use std::collections::HashMap;
use std::path::Path;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use crate::utils::{Result, CryptoEmbeddingsError, cosine_similarity, sigmoid};
use indicatif::{ProgressBar, ProgressStyle};

/// Word2Vec model for learning word embeddings
#[derive(Clone)]
pub struct Word2Vec {
    /// Embedding dimension
    dim: usize,
    /// Context window size
    window: usize,
    /// Minimum word frequency
    min_count: usize,
    /// Learning rate
    learning_rate: f32,
    /// Number of negative samples
    negative_samples: usize,
    /// Word to index mapping
    word2idx: HashMap<String, usize>,
    /// Index to word mapping
    idx2word: Vec<String>,
    /// Word counts
    word_counts: Vec<usize>,
    /// Input embeddings (context vectors)
    syn0: Option<Array2<f32>>,
    /// Output embeddings (target vectors)
    syn1: Option<Array2<f32>>,
    /// Negative sampling distribution (unigram^0.75)
    sampling_table: Vec<usize>,
}

impl Word2Vec {
    /// Create a new Word2Vec model
    pub fn new(dim: usize, window: usize, min_count: usize) -> Self {
        Self {
            dim,
            window,
            min_count,
            learning_rate: 0.025,
            negative_samples: 5,
            word2idx: HashMap::new(),
            idx2word: Vec::new(),
            word_counts: Vec::new(),
            syn0: None,
            syn1: None,
            sampling_table: Vec::new(),
        }
    }

    /// Set learning rate
    pub fn with_learning_rate(mut self, lr: f32) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set number of negative samples
    pub fn with_negative_samples(mut self, n: usize) -> Self {
        self.negative_samples = n;
        self
    }

    /// Build vocabulary from sentences
    pub fn build_vocab(&mut self, sentences: &[Vec<String>]) {
        let mut counts: HashMap<String, usize> = HashMap::new();

        // Count word frequencies
        for sentence in sentences {
            for word in sentence {
                *counts.entry(word.clone()).or_insert(0) += 1;
            }
        }

        // Filter by min_count and build mappings
        let mut words: Vec<(String, usize)> = counts
            .into_iter()
            .filter(|(_, count)| *count >= self.min_count)
            .collect();

        // Sort by frequency (descending)
        words.sort_by(|a, b| b.1.cmp(&a.1));

        self.word2idx.clear();
        self.idx2word.clear();
        self.word_counts.clear();

        for (idx, (word, count)) in words.into_iter().enumerate() {
            self.word2idx.insert(word.clone(), idx);
            self.idx2word.push(word);
            self.word_counts.push(count);
        }

        // Build negative sampling table
        self.build_sampling_table();

        log::info!("Vocabulary size: {}", self.vocab_size());
    }

    /// Build the negative sampling table
    fn build_sampling_table(&mut self) {
        let table_size = 100_000_000;
        self.sampling_table = Vec::with_capacity(table_size);

        // Calculate sum of count^0.75
        let sum: f64 = self.word_counts
            .iter()
            .map(|&c| (c as f64).powf(0.75))
            .sum();

        // Fill table proportionally
        let mut cumulative = 0.0;
        let mut idx = 0;

        for i in 0..table_size {
            self.sampling_table.push(idx);

            let expected = (i as f64 / table_size as f64) * sum;
            while cumulative < expected && idx < self.word_counts.len() {
                cumulative += (self.word_counts[idx] as f64).powf(0.75);
                idx += 1;
            }
            if idx > 0 {
                idx -= 1;
            }
        }
    }

    /// Get a negative sample (used internally via sampling_table)
    #[allow(dead_code)]
    fn sample_negative(&self, rng: &mut impl Rng) -> usize {
        let idx = rng.gen_range(0..self.sampling_table.len());
        self.sampling_table[idx]
    }

    /// Initialize embeddings
    fn init_embeddings(&mut self) {
        let vocab_size = self.vocab_size();
        let mut rng = rand::thread_rng();

        // Initialize with small random values
        let scale = 1.0 / self.dim as f32;

        let mut syn0 = Array2::zeros((vocab_size, self.dim));
        let mut syn1 = Array2::zeros((vocab_size, self.dim));

        for i in 0..vocab_size {
            for j in 0..self.dim {
                syn0[[i, j]] = (rng.gen::<f32>() - 0.5) * scale;
                syn1[[i, j]] = 0.0; // Output vectors start at zero
            }
        }

        self.syn0 = Some(syn0);
        self.syn1 = Some(syn1);
    }

    /// Train the model on sentences
    pub fn train(&mut self, sentences: &[Vec<String>], epochs: usize) -> Result<()> {
        if self.word2idx.is_empty() {
            return Err(CryptoEmbeddingsError::EmptyCorpus);
        }

        self.init_embeddings();

        let total_words: usize = sentences.iter().map(|s| s.len()).sum();
        let total_iterations = total_words * epochs;

        let pb = ProgressBar::new(total_iterations as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
            .unwrap()
            .progress_chars("#>-"));

        let mut rng = rand::thread_rng();

        for epoch in 0..epochs {
            // Decay learning rate
            let lr = self.learning_rate * (1.0 - epoch as f32 / epochs as f32);
            let lr = lr.max(self.learning_rate * 0.0001);

            for sentence in sentences {
                self.train_sentence(sentence, lr, &mut rng)?;
                pb.inc(sentence.len() as u64);
            }
        }

        pb.finish_with_message("Training complete");
        Ok(())
    }

    /// Train on a single sentence using skip-gram with negative sampling
    fn train_sentence(
        &mut self,
        sentence: &[String],
        lr: f32,
        rng: &mut impl Rng,
    ) -> Result<()> {
        let indices: Vec<usize> = sentence
            .iter()
            .filter_map(|w| self.word2idx.get(w).copied())
            .collect();

        if indices.len() < 2 {
            return Ok(());
        }

        // Pre-sample negative indices to avoid borrowing issues
        let table_len = self.sampling_table.len();
        let dim = self.dim;
        let negative_samples = self.negative_samples;

        let syn0 = self.syn0.as_mut().unwrap();
        let syn1 = self.syn1.as_mut().unwrap();

        for (pos, &center_idx) in indices.iter().enumerate() {
            // Random window size for each position (like original word2vec)
            let actual_window = rng.gen_range(1..=self.window);

            // Context indices
            let start = pos.saturating_sub(actual_window);
            let end = (pos + actual_window + 1).min(indices.len());

            for ctx_pos in start..end {
                if ctx_pos == pos {
                    continue;
                }

                let context_idx = indices[ctx_pos];

                // Pre-sample negative indices
                let neg_indices: Vec<usize> = (0..negative_samples)
                    .map(|_| {
                        let idx = rng.gen_range(0..table_len);
                        self.sampling_table[idx]
                    })
                    .collect();

                // Get center word vector
                let center_vec: Vec<f32> = syn0.row(center_idx).to_vec();

                // Positive sample (context word, label = 1)
                let mut neu1e = vec![0.0f32; dim];

                // Train on positive sample
                {
                    let target_vec: Vec<f32> = syn1.row(context_idx).to_vec();
                    let dot: f32 = center_vec.iter().zip(&target_vec).map(|(a, b)| a * b).sum();
                    let pred = sigmoid(dot);
                    let g = lr * (1.0 - pred);

                    for j in 0..dim {
                        neu1e[j] += g * target_vec[j];
                        syn1[[context_idx, j]] += g * center_vec[j];
                    }
                }

                // Negative samples
                for neg_idx in neg_indices {
                    if neg_idx == context_idx {
                        continue;
                    }

                    let neg_vec: Vec<f32> = syn1.row(neg_idx).to_vec();
                    let dot: f32 = center_vec.iter().zip(&neg_vec).map(|(a, b)| a * b).sum();
                    let pred = sigmoid(dot);
                    let g = lr * (0.0 - pred);

                    for j in 0..dim {
                        neu1e[j] += g * neg_vec[j];
                        syn1[[neg_idx, j]] += g * center_vec[j];
                    }
                }

                // Update center word vector
                for j in 0..dim {
                    syn0[[center_idx, j]] += neu1e[j];
                }
            }
        }

        Ok(())
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.idx2word.len()
    }

    /// Get the embedding vector for a word
    pub fn get_vector(&self, word: &str) -> Result<Vec<f32>> {
        let idx = self.word2idx.get(word)
            .ok_or_else(|| CryptoEmbeddingsError::WordNotFound(word.to_string()))?;

        let syn0 = self.syn0.as_ref()
            .ok_or_else(|| CryptoEmbeddingsError::TrainingError("Model not trained".to_string()))?;

        Ok(syn0.row(*idx).to_vec())
    }

    /// Find most similar words
    pub fn most_similar(&self, word: &str, top_n: usize) -> Result<Vec<(String, f32)>> {
        let word_vec = self.get_vector(word)?;
        let syn0 = self.syn0.as_ref()
            .ok_or_else(|| CryptoEmbeddingsError::TrainingError("Model not trained".to_string()))?;

        let word_idx = *self.word2idx.get(word).unwrap();

        let mut similarities: Vec<(usize, f32)> = (0..self.vocab_size())
            .filter(|&i| i != word_idx)
            .map(|i| {
                let vec: Vec<f32> = syn0.row(i).to_vec();
                (i, cosine_similarity(&word_vec, &vec))
            })
            .collect();

        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let results: Vec<(String, f32)> = similarities
            .into_iter()
            .take(top_n)
            .map(|(i, sim)| (self.idx2word[i].clone(), sim))
            .collect();

        Ok(results)
    }

    /// Perform analogy: positive - negative = ?
    pub fn analogy(
        &self,
        positive: &[String],
        negative: &[String],
        top_n: usize,
    ) -> Result<Vec<(String, f32)>> {
        let syn0 = self.syn0.as_ref()
            .ok_or_else(|| CryptoEmbeddingsError::TrainingError("Model not trained".to_string()))?;

        // Start with zero vector
        let mut result_vec = vec![0.0f32; self.dim];

        // Add positive words
        let mut exclude_indices = Vec::new();
        for word in positive {
            let vec = self.get_vector(word)?;
            for (i, &v) in vec.iter().enumerate() {
                result_vec[i] += v;
            }
            exclude_indices.push(*self.word2idx.get(word).unwrap());
        }

        // Subtract negative words
        for word in negative {
            let vec = self.get_vector(word)?;
            for (i, &v) in vec.iter().enumerate() {
                result_vec[i] -= v;
            }
            exclude_indices.push(*self.word2idx.get(word).unwrap());
        }

        // Find most similar (excluding input words)
        let mut similarities: Vec<(usize, f32)> = (0..self.vocab_size())
            .filter(|i| !exclude_indices.contains(i))
            .map(|i| {
                let vec: Vec<f32> = syn0.row(i).to_vec();
                (i, cosine_similarity(&result_vec, &vec))
            })
            .collect();

        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let results: Vec<(String, f32)> = similarities
            .into_iter()
            .take(top_n)
            .map(|(i, sim)| (self.idx2word[i].clone(), sim))
            .collect();

        Ok(results)
    }

    /// Save model to file
    pub fn save(&self, path: &Path) -> Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        let syn0 = self.syn0.as_ref()
            .ok_or_else(|| CryptoEmbeddingsError::TrainingError("Model not trained".to_string()))?;

        // Write header: vocab_size dim
        writeln!(writer, "{} {}", self.vocab_size(), self.dim)?;

        // Write each word and its vector
        for (idx, word) in self.idx2word.iter().enumerate() {
            write!(writer, "{}", word)?;
            for j in 0..self.dim {
                write!(writer, " {}", syn0[[idx, j]])?;
            }
            writeln!(writer)?;
        }

        Ok(())
    }

    /// Load model from file
    pub fn load(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut lines = reader.lines();

        // Read header
        let header = lines.next()
            .ok_or_else(|| CryptoEmbeddingsError::InvalidModel("Empty file".to_string()))??;
        let parts: Vec<&str> = header.split_whitespace().collect();
        if parts.len() != 2 {
            return Err(CryptoEmbeddingsError::InvalidModel("Invalid header".to_string()));
        }
        let vocab_size: usize = parts[0].parse().map_err(|_| CryptoEmbeddingsError::InvalidModel("Invalid vocab size".to_string()))?;
        let dim: usize = parts[1].parse().map_err(|_| CryptoEmbeddingsError::InvalidModel("Invalid dimension".to_string()))?;

        let mut word2idx = HashMap::new();
        let mut idx2word = Vec::with_capacity(vocab_size);
        let mut syn0 = Array2::zeros((vocab_size, dim));

        for (idx, line) in lines.enumerate() {
            let line = line?;
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() != dim + 1 {
                continue;
            }

            let word = parts[0].to_string();
            word2idx.insert(word.clone(), idx);
            idx2word.push(word);

            for (j, &val) in parts[1..].iter().enumerate() {
                syn0[[idx, j]] = val.parse().unwrap_or(0.0);
            }
        }

        Ok(Self {
            dim,
            window: 5,
            min_count: 5,
            learning_rate: 0.025,
            negative_samples: 5,
            word2idx,
            idx2word,
            word_counts: vec![0; vocab_size],
            syn0: Some(syn0),
            syn1: None,
            sampling_table: Vec::new(),
        })
    }

    /// Check if a word is in vocabulary
    pub fn contains(&self, word: &str) -> bool {
        self.word2idx.contains_key(word)
    }

    /// Get all words in vocabulary
    pub fn words(&self) -> &[String] {
        &self.idx2word
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_sentences() -> Vec<Vec<String>> {
        vec![
            vec!["btc", "bullish", "trend", "support", "resistance"].iter().map(|s| s.to_string()).collect(),
            vec!["eth", "bearish", "trend", "breakdown", "support"].iter().map(|s| s.to_string()).collect(),
            vec!["btc", "breakout", "resistance", "bullish", "momentum"].iter().map(|s| s.to_string()).collect(),
            vec!["sol", "bullish", "breakout", "ath", "volume"].iter().map(|s| s.to_string()).collect(),
            vec!["eth", "bullish", "trend", "accumulation", "support"].iter().map(|s| s.to_string()).collect(),
        ]
    }

    #[test]
    fn test_build_vocab() {
        let mut model = Word2Vec::new(10, 2, 1);
        let sentences = create_test_sentences();

        model.build_vocab(&sentences);

        assert!(model.vocab_size() > 0);
        assert!(model.word2idx.contains_key("btc"));
        assert!(model.word2idx.contains_key("bullish"));
    }

    #[test]
    fn test_train() {
        let mut model = Word2Vec::new(10, 2, 1);
        let sentences = create_test_sentences();

        model.build_vocab(&sentences);
        let result = model.train(&sentences, 2);

        assert!(result.is_ok());
        assert!(model.syn0.is_some());
    }

    #[test]
    fn test_get_vector() {
        let mut model = Word2Vec::new(10, 2, 1);
        let sentences = create_test_sentences();

        model.build_vocab(&sentences);
        model.train(&sentences, 2).unwrap();

        let vec = model.get_vector("btc");
        assert!(vec.is_ok());
        assert_eq!(vec.unwrap().len(), 10);
    }

    #[test]
    fn test_most_similar() {
        let mut model = Word2Vec::new(10, 2, 1);
        let sentences = create_test_sentences();

        model.build_vocab(&sentences);
        model.train(&sentences, 5).unwrap();

        let similar = model.most_similar("bullish", 3);
        assert!(similar.is_ok());

        let results = similar.unwrap();
        assert!(!results.is_empty());
    }
}
