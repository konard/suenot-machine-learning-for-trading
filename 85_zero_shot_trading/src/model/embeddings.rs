//! Embedding space utilities for zero-shot learning.

use crate::{MarketRegime, Result, ZeroShotError};
use std::collections::HashMap;

/// Embedding space for zero-shot learning.
///
/// Manages the shared embedding space where market features
/// and regime attributes are mapped.
#[derive(Debug, Clone)]
pub struct EmbeddingSpace {
    /// Embedding dimension
    pub embed_dim: usize,
    /// Stored embeddings for regimes
    regime_embeddings: HashMap<MarketRegime, Vec<f64>>,
    /// Temperature for similarity computation
    temperature: f64,
}

impl EmbeddingSpace {
    /// Create a new embedding space.
    pub fn new(embed_dim: usize, temperature: f64) -> Self {
        Self {
            embed_dim,
            regime_embeddings: HashMap::new(),
            temperature,
        }
    }

    /// L2 normalize a vector.
    pub fn l2_normalize(v: &[f64]) -> Vec<f64> {
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-8 {
            v.iter().map(|x| x / norm).collect()
        } else {
            v.to_vec()
        }
    }

    /// Compute cosine similarity between two embeddings.
    pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(a.len(), b.len());
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    /// Compute Euclidean distance between two embeddings.
    pub fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Register a regime embedding.
    pub fn register_regime(&mut self, regime: MarketRegime, embedding: Vec<f64>) -> Result<()> {
        if embedding.len() != self.embed_dim {
            return Err(ZeroShotError::InvalidParameter(format!(
                "Embedding dimension mismatch: expected {}, got {}",
                self.embed_dim,
                embedding.len()
            )));
        }
        let normalized = Self::l2_normalize(&embedding);
        self.regime_embeddings.insert(regime, normalized);
        Ok(())
    }

    /// Get regime embedding.
    pub fn get_regime_embedding(&self, regime: &MarketRegime) -> Option<&Vec<f64>> {
        self.regime_embeddings.get(regime)
    }

    /// Find nearest regime to a market embedding.
    pub fn find_nearest_regime(
        &self,
        market_embedding: &[f64],
    ) -> Result<(MarketRegime, f64, HashMap<MarketRegime, f64>)> {
        if self.regime_embeddings.is_empty() {
            return Err(ZeroShotError::ModelError("No regimes registered".into()));
        }

        // Compute similarities
        let mut similarities = HashMap::new();
        for (regime, regime_embed) in &self.regime_embeddings {
            let sim = Self::cosine_similarity(market_embedding, regime_embed);
            similarities.insert(*regime, sim);
        }

        // Convert to probabilities via softmax
        let sim_values: Vec<f64> = similarities.values().cloned().collect();
        let scaled: Vec<f64> = sim_values.iter().map(|s| s / self.temperature).collect();
        let max_val = scaled.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_vals: Vec<f64> = scaled.iter().map(|s| (s - max_val).exp()).collect();
        let sum_exp: f64 = exp_vals.iter().sum();

        let mut probabilities = HashMap::new();
        for (regime, exp_val) in similarities.keys().zip(exp_vals.iter()) {
            probabilities.insert(*regime, exp_val / sum_exp);
        }

        // Find best match
        let (best_regime, best_prob) = probabilities
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(r, p)| (*r, *p))
            .unwrap();

        Ok((best_regime, best_prob, probabilities))
    }

    /// Compute centroid of multiple embeddings.
    pub fn compute_centroid(embeddings: &[Vec<f64>]) -> Result<Vec<f64>> {
        if embeddings.is_empty() {
            return Err(ZeroShotError::DataError("Empty embeddings list".into()));
        }

        let dim = embeddings[0].len();
        let mut centroid = vec![0.0; dim];

        for embedding in embeddings {
            for (i, val) in embedding.iter().enumerate() {
                centroid[i] += val / embeddings.len() as f64;
            }
        }

        Ok(Self::l2_normalize(&centroid))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_space() {
        let mut space = EmbeddingSpace::new(64, 0.1);

        let embedding = vec![1.0; 64];
        space
            .register_regime(MarketRegime::StrongUptrend, embedding)
            .unwrap();

        assert!(space.get_regime_embedding(&MarketRegime::StrongUptrend).is_some());
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let c = vec![0.0, 1.0, 0.0];

        assert!((EmbeddingSpace::cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);
        assert!((EmbeddingSpace::cosine_similarity(&a, &c) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_l2_normalize() {
        let v = vec![3.0, 4.0];
        let normalized = EmbeddingSpace::l2_normalize(&v);
        let norm: f64 = normalized.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }
}
