//! SSM-Transformer Hybrid model combining SSM and Transformer blocks.
//!
//! Interleaves Mamba-style SSM blocks with Transformer blocks following
//! the Jamba architecture pattern. Produces multi-task outputs for trading:
//! direction probability, volatility, and return magnitude.

use crate::model::ssm_block::{SSMBlock, LinearLayer};
use crate::model::transformer_block::TransformerBlock;

/// Layer type in the hybrid model.
#[derive(Debug, Clone)]
pub enum HybridLayer {
    SSM(SSMBlock),
    Transformer(TransformerBlock),
}

/// Predictions from the hybrid model.
#[derive(Debug, Clone)]
pub struct HybridPredictions {
    /// Probability of price going up (0.0 to 1.0)
    pub direction: f64,
    /// Predicted next-bar volatility
    pub volatility: f64,
    /// Predicted next-bar return magnitude
    pub return_magnitude: f64,
}

/// SSM-Transformer Hybrid model.
///
/// Architecture: Input projection → [SSM/Transformer layers] → Task heads
///
/// The ratio of SSM to Transformer layers is configurable. By default,
/// follows the Jamba pattern where most layers are SSM with periodic
/// Transformer layers.
#[derive(Debug, Clone)]
pub struct SSMTransformerModel {
    pub input_proj: LinearLayer,
    pub layers: Vec<HybridLayer>,
    pub direction_head: LinearLayer,
    pub volatility_head: LinearLayer,
    pub return_head: LinearLayer,
    pub d_model: usize,
}

impl SSMTransformerModel {
    /// Create a new SSM-Transformer Hybrid model.
    ///
    /// # Arguments
    /// * `input_size` - Number of input features
    /// * `d_model` - Hidden dimension
    /// * `n_ssm_layers` - Number of SSM (Mamba) layers
    /// * `n_attn_layers` - Number of Transformer layers
    /// * `ssm_state_size` - SSM hidden state dimension
    /// * `n_heads` - Number of attention heads
    pub fn new(
        input_size: usize,
        d_model: usize,
        n_ssm_layers: usize,
        n_attn_layers: usize,
        ssm_state_size: usize,
        n_heads: usize,
    ) -> Self {
        let total_layers = n_ssm_layers + n_attn_layers;
        let attn_interval = if n_attn_layers > 0 {
            total_layers / n_attn_layers
        } else {
            total_layers + 1
        };
        let d_ff = d_model * 2;

        let mut layers = Vec::with_capacity(total_layers);
        let mut attn_placed = 0;

        for i in 0..total_layers {
            if (i + 1) % attn_interval == 0 && attn_placed < n_attn_layers {
                layers.push(HybridLayer::Transformer(TransformerBlock::new(
                    d_model, n_heads, d_ff,
                )));
                attn_placed += 1;
            } else {
                layers.push(HybridLayer::SSM(SSMBlock::new(d_model, ssm_state_size)));
            }
        }

        Self {
            input_proj: LinearLayer::new(input_size, d_model),
            layers,
            direction_head: LinearLayer::new(d_model, 1),
            volatility_head: LinearLayer::new(d_model, 1),
            return_head: LinearLayer::new(d_model, 1),
            d_model,
        }
    }

    /// Run forward pass on a sequence.
    ///
    /// # Arguments
    /// * `sequence` - Input: Vec of timesteps, each Vec<f64> of input_size
    ///
    /// # Returns
    /// Multi-task predictions based on the last timestep
    pub fn forward(&self, sequence: &[Vec<f64>]) -> HybridPredictions {
        // Project input to d_model
        let mut x: Vec<Vec<f64>> = sequence
            .iter()
            .map(|step| self.input_proj.forward(step))
            .collect();

        // Pass through interleaved layers
        for layer in &self.layers {
            x = match layer {
                HybridLayer::SSM(ssm) => ssm.forward(&x),
                HybridLayer::Transformer(tf) => tf.forward(&x),
            };
        }

        // Use last timestep for predictions
        let last = &x[x.len() - 1];

        let dir_raw = self.direction_head.forward(last)[0];
        let vol_raw = self.volatility_head.forward(last)[0];
        let ret_raw = self.return_head.forward(last)[0];

        HybridPredictions {
            direction: sigmoid(dir_raw),
            volatility: softplus(vol_raw),
            return_magnitude: ret_raw,
        }
    }

    /// Process a batch of sequences.
    pub fn forward_batch(&self, batch: &[Vec<Vec<f64>>]) -> Vec<HybridPredictions> {
        batch.iter().map(|seq| self.forward(seq)).collect()
    }

    /// Count total number of parameters in the model.
    pub fn num_parameters(&self) -> usize {
        let layer_params: usize = self.layers.iter().map(|l| match l {
            HybridLayer::SSM(ssm) => ssm.num_parameters(),
            HybridLayer::Transformer(tf) => tf.num_parameters(),
        }).sum();

        self.input_proj.num_parameters()
            + layer_params
            + self.direction_head.num_parameters()
            + self.volatility_head.num_parameters()
            + self.return_head.num_parameters()
    }

    /// Get a summary of the model architecture.
    pub fn summary(&self) -> String {
        let n_ssm = self.layers.iter().filter(|l| matches!(l, HybridLayer::SSM(_))).count();
        let n_attn = self.layers.iter().filter(|l| matches!(l, HybridLayer::Transformer(_))).count();
        format!(
            "SSMTransformerModel(d_model={}, layers={} SSM + {} Attention, params={})",
            self.d_model, n_ssm, n_attn, self.num_parameters()
        )
    }
}

/// Sigmoid activation.
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Softplus activation: log(1 + exp(x))
fn softplus(x: f64) -> f64 {
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hybrid_model_creation() {
        let model = SSMTransformerModel::new(15, 32, 4, 1, 8, 4);
        assert!(model.num_parameters() > 0);
        println!("{}", model.summary());
    }

    #[test]
    fn test_hybrid_forward() {
        let model = SSMTransformerModel::new(10, 16, 3, 1, 8, 4);
        let sequence: Vec<Vec<f64>> = (0..20)
            .map(|_| (0..10).map(|_| 0.1).collect())
            .collect();

        let preds = model.forward(&sequence);
        assert!(preds.direction >= 0.0 && preds.direction <= 1.0);
        assert!(preds.volatility >= 0.0);
    }

    #[test]
    fn test_hybrid_batch() {
        let model = SSMTransformerModel::new(10, 16, 2, 1, 8, 4);
        let batch: Vec<Vec<Vec<f64>>> = (0..3)
            .map(|_| {
                (0..10)
                    .map(|_| (0..10).map(|_| 0.1).collect())
                    .collect()
            })
            .collect();

        let preds = model.forward_batch(&batch);
        assert_eq!(preds.len(), 3);
    }

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 0.01);
        assert!(sigmoid(10.0) > 0.99);
        assert!(sigmoid(-10.0) < 0.01);
    }
}
