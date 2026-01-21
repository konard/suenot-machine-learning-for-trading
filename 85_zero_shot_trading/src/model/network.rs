//! Neural network model for zero-shot trading.

use crate::{MarketRegime, Result, ZeroShotError};
use rand::Rng;
use std::collections::HashMap;

/// Zero-shot trading model.
///
/// This model maps market features and regime attributes to a shared
/// embedding space for zero-shot prediction.
#[derive(Debug, Clone)]
pub struct ZeroShotModel {
    /// Input feature dimension
    pub input_dim: usize,
    /// Embedding dimension
    pub embed_dim: usize,
    /// Temperature for softmax scaling
    pub temperature: f64,
    /// Market encoder weights (simplified linear projection)
    market_weights: Vec<Vec<f64>>,
    /// Regime embeddings
    regime_embeddings: HashMap<MarketRegime, Vec<f64>>,
}

impl ZeroShotModel {
    /// Create a new zero-shot model.
    ///
    /// # Arguments
    /// * `input_dim` - Number of input features per timestep
    /// * `embed_dim` - Dimension of the embedding space
    pub fn new(input_dim: usize, embed_dim: usize) -> Self {
        let mut rng = rand::thread_rng();

        // Initialize market encoder weights (Xavier initialization)
        let scale = (2.0 / (input_dim + embed_dim) as f64).sqrt();
        let market_weights: Vec<Vec<f64>> = (0..embed_dim)
            .map(|_| {
                (0..input_dim)
                    .map(|_| rng.gen_range(-scale..scale))
                    .collect()
            })
            .collect();

        // Initialize regime embeddings
        let mut regime_embeddings = HashMap::new();
        for regime in MarketRegime::all() {
            let embedding: Vec<f64> = (0..embed_dim)
                .map(|_| rng.gen_range(-1.0..1.0))
                .collect();
            let normalized = Self::l2_normalize(&embedding);
            regime_embeddings.insert(*regime, normalized);
        }

        Self {
            input_dim,
            embed_dim,
            temperature: 0.1,
            market_weights,
            regime_embeddings,
        }
    }

    /// Load model from file.
    pub fn load(path: &str) -> Result<Self> {
        // In production, this would load from a serialized file
        // For now, create a new model
        tracing::warn!("Model loading not implemented, creating new model");
        Ok(Self::new(11, 64))
    }

    /// Save model to file.
    pub fn save(&self, path: &str) -> Result<()> {
        // In production, this would serialize the model
        tracing::warn!("Model saving not implemented");
        Ok(())
    }

    /// L2 normalize a vector.
    fn l2_normalize(v: &[f64]) -> Vec<f64> {
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-8 {
            v.iter().map(|x| x / norm).collect()
        } else {
            v.to_vec()
        }
    }

    /// Encode market features to embedding space.
    ///
    /// # Arguments
    /// * `features` - Market features (seq_len x input_dim)
    ///
    /// # Returns
    /// * Embedding vector of size embed_dim
    pub fn encode_market(&self, features: &[Vec<f64>]) -> Result<Vec<f64>> {
        if features.is_empty() {
            return Err(ZeroShotError::FeatureError("Empty feature matrix".into()));
        }

        let seq_len = features.len();

        // Validate input dimension
        if features[0].len() != self.input_dim {
            return Err(ZeroShotError::FeatureError(format!(
                "Expected {} features, got {}",
                self.input_dim,
                features[0].len()
            )));
        }

        // Compute mean across sequence (temporal pooling)
        let mut pooled = vec![0.0; self.input_dim];
        for row in features {
            for (i, val) in row.iter().enumerate() {
                pooled[i] += val / seq_len as f64;
            }
        }

        // Linear projection: embedding = W * pooled
        let mut embedding = vec![0.0; self.embed_dim];
        for (i, weights) in self.market_weights.iter().enumerate() {
            for (j, w) in weights.iter().enumerate() {
                embedding[i] += w * pooled[j];
            }
            // Apply ReLU activation
            if embedding[i] < 0.0 {
                embedding[i] = 0.0;
            }
        }

        // L2 normalize
        Ok(Self::l2_normalize(&embedding))
    }

    /// Get regime embedding.
    pub fn get_regime_embedding(&self, regime: MarketRegime) -> &[f64] {
        self.regime_embeddings.get(&regime).expect("Regime should exist")
    }

    /// Compute cosine similarity between two embeddings.
    pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(a.len(), b.len());
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    /// Predict market regime from features.
    ///
    /// # Arguments
    /// * `features` - Market features (seq_len x input_dim)
    ///
    /// # Returns
    /// * Tuple of (predicted regime, confidence, all regime probabilities)
    pub fn predict_regime(
        &self,
        features: &[Vec<f64>],
    ) -> Result<(MarketRegime, f64, HashMap<MarketRegime, f64>)> {
        // Encode market features
        let market_embed = self.encode_market(features)?;

        // Compute similarities to all regimes
        let mut similarities = HashMap::new();
        for regime in MarketRegime::all() {
            let regime_embed = self.get_regime_embedding(*regime);
            let sim = Self::cosine_similarity(&market_embed, regime_embed);
            similarities.insert(*regime, sim);
        }

        // Convert to probabilities via softmax
        let sim_values: Vec<f64> = MarketRegime::all()
            .iter()
            .map(|r| similarities[r] / self.temperature)
            .collect();

        let max_sim = sim_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_sims: Vec<f64> = sim_values.iter().map(|s| (s - max_sim).exp()).collect();
        let sum_exp: f64 = exp_sims.iter().sum();

        let mut probabilities = HashMap::new();
        for (i, regime) in MarketRegime::all().iter().enumerate() {
            probabilities.insert(*regime, exp_sims[i] / sum_exp);
        }

        // Get prediction (highest probability)
        let (predicted_regime, confidence) = probabilities
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(r, p)| (*r, *p))
            .unwrap();

        Ok((predicted_regime, confidence, probabilities))
    }

    /// Update regime embedding (for training).
    pub fn update_regime_embedding(&mut self, regime: MarketRegime, embedding: Vec<f64>) {
        let normalized = Self::l2_normalize(&embedding);
        self.regime_embeddings.insert(regime, normalized);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_creation() {
        let model = ZeroShotModel::new(11, 64);
        assert_eq!(model.input_dim, 11);
        assert_eq!(model.embed_dim, 64);
    }

    #[test]
    fn test_market_encoding() {
        let model = ZeroShotModel::new(11, 64);

        // Create dummy features
        let features: Vec<Vec<f64>> = (0..50)
            .map(|_| (0..11).map(|_| rand::random::<f64>()).collect())
            .collect();

        let embedding = model.encode_market(&features).unwrap();
        assert_eq!(embedding.len(), 64);

        // Check normalization
        let norm: f64 = embedding.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_regime_prediction() {
        let model = ZeroShotModel::new(11, 64);

        let features: Vec<Vec<f64>> = (0..50)
            .map(|_| (0..11).map(|_| rand::random::<f64>()).collect())
            .collect();

        let (regime, confidence, probs) = model.predict_regime(&features).unwrap();

        // Check confidence is valid probability
        assert!(confidence >= 0.0 && confidence <= 1.0);

        // Check probabilities sum to 1
        let sum: f64 = probs.values().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }
}
