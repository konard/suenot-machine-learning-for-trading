//! Pattern Memory Manager
//!
//! Handles pattern storage with capacity limits, relevance scoring,
//! and automatic eviction of old/unused patterns.

use chrono::{DateTime, Utc};
use ndarray::Array1;
use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};

/// Stored pattern with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredPattern {
    /// Pattern features
    pub features: Vec<f64>,
    /// Associated outcome (label)
    pub outcome: f64,
    /// Timestamp when pattern was stored
    pub timestamp: DateTime<Utc>,
    /// Number of times this pattern was retrieved
    pub retrieval_count: usize,
    /// Last retrieval timestamp
    pub last_retrieved: Option<DateTime<Utc>>,
}

impl StoredPattern {
    pub fn new(features: Vec<f64>, outcome: f64, timestamp: DateTime<Utc>) -> Self {
        Self {
            features,
            outcome,
            timestamp,
            retrieval_count: 0,
            last_retrieved: None,
        }
    }

    /// Compute cosine similarity with another pattern
    pub fn similarity(&self, other: &[f64]) -> f64 {
        if self.features.len() != other.len() {
            return 0.0;
        }

        let dot: f64 = self
            .features
            .iter()
            .zip(other.iter())
            .map(|(a, b)| a * b)
            .sum();

        let norm_a: f64 = self.features.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
        let norm_b: f64 = other.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }

        dot / (norm_a * norm_b)
    }

    /// Get usefulness score based on recency and usage
    pub fn usefulness_score(&self, now: DateTime<Utc>) -> f64 {
        // Recency score (exponential decay over 30 days)
        let age_days = (now - self.timestamp).num_hours() as f64 / 24.0;
        let recency_score = (-age_days / 30.0).exp();

        // Usage score
        let usage_score = (self.retrieval_count as f64).ln_1p() / 5.0;

        // Combined score (weighted)
        0.6 * recency_score + 0.4 * usage_score.min(1.0)
    }
}

/// Pattern Memory Manager with automatic capacity management
#[derive(Debug, Clone)]
pub struct PatternMemoryManager {
    /// Stored patterns
    patterns: Vec<StoredPattern>,
    /// Maximum number of patterns
    max_patterns: usize,
    /// Pattern dimension
    pattern_dim: usize,
    /// Similarity threshold for merging patterns
    similarity_threshold: f64,
    /// EMA alpha for merging patterns
    merge_alpha: f64,
}

impl PatternMemoryManager {
    /// Create a new pattern memory manager
    pub fn new(max_patterns: usize, pattern_dim: usize) -> Self {
        Self {
            patterns: Vec::new(),
            max_patterns,
            pattern_dim,
            similarity_threshold: 0.95,
            merge_alpha: 0.3,
        }
    }

    /// Set similarity threshold for merging
    pub fn with_similarity_threshold(mut self, threshold: f64) -> Self {
        self.similarity_threshold = threshold;
        self
    }

    /// Get current number of stored patterns
    pub fn len(&self) -> usize {
        self.patterns.len()
    }

    /// Check if memory is empty
    pub fn is_empty(&self) -> bool {
        self.patterns.is_empty()
    }

    /// Add a pattern to memory
    ///
    /// If a very similar pattern exists, it will be updated instead.
    /// If memory is full, the least useful pattern will be evicted.
    pub fn add_pattern(&mut self, features: &[f64], outcome: f64, timestamp: DateTime<Utc>) {
        if features.len() != self.pattern_dim {
            log::error!(
                "Pattern dimension mismatch: expected {}, got {}",
                self.pattern_dim,
                features.len()
            );
            return;
        }

        // Normalize features
        let norm: f64 = features.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
        let normalized: Vec<f64> = if norm > 0.0 {
            features.iter().map(|x| x / norm).collect()
        } else {
            features.to_vec()
        };

        // Check for similar existing pattern
        let mut most_similar_idx = None;
        let mut max_similarity = 0.0;

        for (idx, pattern) in self.patterns.iter().enumerate() {
            let sim = pattern.similarity(&normalized);
            if sim > max_similarity {
                max_similarity = sim;
                most_similar_idx = Some(idx);
            }
        }

        // If very similar pattern exists, update it
        if max_similarity > self.similarity_threshold {
            if let Some(idx) = most_similar_idx {
                let pattern = &mut self.patterns[idx];

                // EMA update
                for i in 0..self.pattern_dim {
                    pattern.features[i] =
                        self.merge_alpha * normalized[i] + (1.0 - self.merge_alpha) * pattern.features[i];
                }
                pattern.outcome =
                    self.merge_alpha * outcome + (1.0 - self.merge_alpha) * pattern.outcome;

                // Re-normalize
                let new_norm: f64 = pattern.features.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
                if new_norm > 0.0 {
                    for f in &mut pattern.features {
                        *f /= new_norm;
                    }
                }

                log::debug!("Updated existing pattern (similarity: {:.3})", max_similarity);
                return;
            }
        }

        // Evict if at capacity
        if self.patterns.len() >= self.max_patterns {
            self.evict_pattern();
        }

        // Add new pattern
        let new_pattern = StoredPattern::new(normalized, outcome, timestamp);
        self.patterns.push(new_pattern);
        log::debug!("Added new pattern, total: {}", self.patterns.len());
    }

    /// Evict the least useful pattern
    fn evict_pattern(&mut self) {
        if self.patterns.is_empty() {
            return;
        }

        let now = Utc::now();

        // Find pattern with lowest usefulness score
        let mut min_score = f64::INFINITY;
        let mut evict_idx = 0;

        for (idx, pattern) in self.patterns.iter().enumerate() {
            let score = pattern.usefulness_score(now);
            if score < min_score {
                min_score = score;
                evict_idx = idx;
            }
        }

        self.patterns.swap_remove(evict_idx);
        log::debug!("Evicted pattern with score {:.3}", min_score);
    }

    /// Query memory for similar patterns
    ///
    /// Returns (patterns, outcomes, similarities)
    pub fn query(&mut self, query: &[f64], k: usize) -> (Vec<Vec<f64>>, Vec<f64>, Vec<f64>) {
        if self.patterns.is_empty() || query.len() != self.pattern_dim {
            return (Vec::new(), Vec::new(), Vec::new());
        }

        // Normalize query
        let norm: f64 = query.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
        let normalized: Vec<f64> = if norm > 0.0 {
            query.iter().map(|x| x / norm).collect()
        } else {
            query.to_vec()
        };

        // Compute similarities and sort
        let mut scored: Vec<(usize, f64)> = self
            .patterns
            .iter()
            .enumerate()
            .map(|(idx, p)| (idx, p.similarity(&normalized)))
            .collect();

        scored.sort_by_key(|(_, s)| std::cmp::Reverse(OrderedFloat(*s)));

        // Update retrieval counts for top-k
        let now = Utc::now();
        let top_k: Vec<(usize, f64)> = scored.into_iter().take(k).collect();

        for &(idx, _) in &top_k {
            self.patterns[idx].retrieval_count += 1;
            self.patterns[idx].last_retrieved = Some(now);
        }

        // Collect results
        let patterns: Vec<Vec<f64>> = top_k
            .iter()
            .map(|&(idx, _)| self.patterns[idx].features.clone())
            .collect();

        let outcomes: Vec<f64> = top_k
            .iter()
            .map(|&(idx, _)| self.patterns[idx].outcome)
            .collect();

        let similarities: Vec<f64> = top_k.iter().map(|&(_, s)| s).collect();

        (patterns, outcomes, similarities)
    }

    /// Predict outcome based on similar patterns
    pub fn predict(&mut self, query: &[f64], k: usize) -> (f64, f64) {
        let (_, outcomes, similarities) = self.query(query, k);

        if outcomes.is_empty() {
            return (0.0, 0.0);
        }

        // Weighted average prediction
        let total_weight: f64 = similarities.iter().sum();
        let prediction: f64 = if total_weight > 0.0 {
            outcomes
                .iter()
                .zip(similarities.iter())
                .map(|(o, s)| o * s)
                .sum::<f64>()
                / total_weight
        } else {
            outcomes.iter().sum::<f64>() / outcomes.len() as f64
        };

        // Confidence based on average similarity
        let confidence = similarities.iter().sum::<f64>() / similarities.len() as f64;

        (prediction, confidence)
    }

    /// Convert to ndarray format for use with DenseAssociativeMemory
    pub fn to_arrays(&self) -> (ndarray::Array2<f64>, ndarray::Array1<f64>) {
        if self.patterns.is_empty() {
            return (
                ndarray::Array2::zeros((0, self.pattern_dim)),
                ndarray::Array1::zeros(0),
            );
        }

        let n = self.patterns.len();
        let mut features = ndarray::Array2::zeros((n, self.pattern_dim));
        let mut outcomes = ndarray::Array1::zeros(n);

        for (i, pattern) in self.patterns.iter().enumerate() {
            for (j, &f) in pattern.features.iter().enumerate() {
                features[[i, j]] = f;
            }
            outcomes[i] = pattern.outcome;
        }

        (features, outcomes)
    }

    /// Get memory statistics
    pub fn stats(&self) -> MemoryManagerStats {
        let now = Utc::now();

        let total_retrievals: usize = self.patterns.iter().map(|p| p.retrieval_count).sum();
        let avg_usefulness = if self.patterns.is_empty() {
            0.0
        } else {
            self.patterns.iter().map(|p| p.usefulness_score(now)).sum::<f64>()
                / self.patterns.len() as f64
        };

        let avg_outcome = if self.patterns.is_empty() {
            0.0
        } else {
            self.patterns.iter().map(|p| p.outcome).sum::<f64>() / self.patterns.len() as f64
        };

        MemoryManagerStats {
            n_patterns: self.patterns.len(),
            capacity: self.max_patterns,
            utilization: self.patterns.len() as f64 / self.max_patterns as f64,
            total_retrievals,
            avg_usefulness,
            avg_outcome,
        }
    }

    /// Save memory to JSON file
    pub fn save(&self, path: &str) -> anyhow::Result<()> {
        let json = serde_json::to_string_pretty(&self.patterns)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load memory from JSON file
    pub fn load(&mut self, path: &str) -> anyhow::Result<()> {
        let json = std::fs::read_to_string(path)?;
        self.patterns = serde_json::from_str(&json)?;

        // Validate dimension
        if !self.patterns.is_empty() && self.patterns[0].features.len() != self.pattern_dim {
            return Err(anyhow::anyhow!(
                "Pattern dimension mismatch: expected {}, got {}",
                self.pattern_dim,
                self.patterns[0].features.len()
            ));
        }

        Ok(())
    }
}

/// Memory manager statistics
#[derive(Debug, Clone)]
pub struct MemoryManagerStats {
    pub n_patterns: usize,
    pub capacity: usize,
    pub utilization: f64,
    pub total_retrievals: usize,
    pub avg_usefulness: f64,
    pub avg_outcome: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_and_query() {
        let mut manager = PatternMemoryManager::new(10, 3);

        manager.add_pattern(&[1.0, 0.0, 0.0], 1.0, Utc::now());
        manager.add_pattern(&[0.0, 1.0, 0.0], -1.0, Utc::now());

        assert_eq!(manager.len(), 2);

        let (_, outcomes, similarities) = manager.query(&[1.0, 0.0, 0.0], 2);

        assert_eq!(outcomes.len(), 2);
        assert!(similarities[0] > similarities[1]);
    }

    #[test]
    fn test_merge_similar() {
        let mut manager = PatternMemoryManager::new(10, 2)
            .with_similarity_threshold(0.95);

        manager.add_pattern(&[1.0, 0.0], 1.0, Utc::now());
        manager.add_pattern(&[0.99, 0.01], 0.8, Utc::now());

        // Should have merged
        assert_eq!(manager.len(), 1);
    }

    #[test]
    fn test_eviction() {
        let mut manager = PatternMemoryManager::new(2, 2);

        manager.add_pattern(&[1.0, 0.0], 1.0, Utc::now());
        manager.add_pattern(&[0.0, 1.0], -1.0, Utc::now());
        manager.add_pattern(&[0.7, 0.7], 0.5, Utc::now());

        // Should have evicted one
        assert_eq!(manager.len(), 2);
    }

    #[test]
    fn test_predict() {
        let mut manager = PatternMemoryManager::new(10, 2);

        manager.add_pattern(&[1.0, 0.0], 1.0, Utc::now());
        manager.add_pattern(&[0.0, 1.0], -1.0, Utc::now());

        let (pred, conf) = manager.predict(&[1.0, 0.0], 2);

        assert!(pred > 0.0);
        assert!(conf > 0.0);
    }
}
