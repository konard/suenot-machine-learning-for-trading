//! Gated State Space Model block.
//!
//! Implements the GSS (Gated State Spaces) architecture from Mehta et al. (2022):
//!   z = SSM(V * input)
//!   g = GELU(G * input)
//!   output = O * (z ⊙ g)

use super::ssm::DiagonalSSMState;
use std::f64;

/// GELU activation function (approximate).
fn gelu(x: f64) -> f64 {
    0.5 * x * (1.0 + ((2.0 / f64::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
}

/// Sigmoid activation function.
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Simple linear layer: y = W * x + b
fn linear_forward(weights: &[Vec<f64>], bias: &[f64], input: &[f64]) -> Vec<f64> {
    weights
        .iter()
        .zip(bias.iter())
        .map(|(w, b)| {
            let sum: f64 = w.iter().zip(input.iter()).map(|(wi, xi)| wi * xi).sum();
            sum + b
        })
        .collect()
}

/// Initialize weight matrix with scaled random-like values.
fn init_weights(rows: usize, cols: usize, seed: usize) -> Vec<Vec<f64>> {
    let scale = 1.0 / (cols as f64).sqrt();
    (0..rows)
        .map(|i| {
            (0..cols)
                .map(|j| {
                    let hash = ((i.wrapping_mul(7) + j.wrapping_mul(13) + seed) as f64) * 0.1;
                    hash.sin() * scale
                })
                .collect()
        })
        .collect()
}

/// Gated SSM block: SSM with output gating.
///
/// Architecture:
///   1. Value projection: v = W_v * input
///   2. SSM pass: z = SSM(v)
///   3. Gate projection: g = GELU(W_g * input)
///   4. Merge: merged = z ⊙ g
///   5. Output projection: output = W_o * merged
pub struct GatedSSMBlock {
    /// SSM layers (one per inner dimension)
    pub ssm_layers: Vec<DiagonalSSMState>,
    /// Value projection weights (d_inner x d_model)
    pub value_weights: Vec<Vec<f64>>,
    /// Value projection bias
    pub value_bias: Vec<f64>,
    /// Gate projection weights (d_inner x d_model)
    pub gate_weights: Vec<Vec<f64>>,
    /// Gate projection bias
    pub gate_bias: Vec<f64>,
    /// Output projection weights (d_model x d_inner)
    pub output_weights: Vec<Vec<f64>>,
    /// Output projection bias
    pub output_bias: Vec<f64>,
    /// Model dimension
    pub d_model: usize,
    /// Inner dimension (d_model * expand)
    pub d_inner: usize,
}

impl GatedSSMBlock {
    /// Create a new Gated SSM block.
    ///
    /// # Arguments
    /// * `d_model` - Input/output dimension
    /// * `d_state` - SSM hidden state dimension
    /// * `expand` - Expansion factor for inner dimension
    pub fn new(d_model: usize, d_state: usize, expand: usize) -> Self {
        let d_inner = d_model * expand;

        // Initialize SSM layers for each inner dimension
        let ssm_layers: Vec<_> = (0..d_inner)
            .map(|_| DiagonalSSMState::new(d_state, 0.01))
            .collect();

        // Projection weights
        let value_weights = init_weights(d_inner, d_model, 42);
        let value_bias = vec![0.0; d_inner];
        let gate_weights = init_weights(d_inner, d_model, 137);
        let gate_bias = vec![0.0; d_inner];
        let output_weights = init_weights(d_model, d_inner, 256);
        let output_bias = vec![0.0; d_model];

        GatedSSMBlock {
            ssm_layers,
            value_weights,
            value_bias,
            gate_weights,
            gate_bias,
            output_weights,
            output_bias,
            d_model,
            d_inner,
        }
    }

    /// Process a single time step through the gated SSM block.
    ///
    /// # Arguments
    /// * `input` - Feature vector of length d_model
    ///
    /// # Returns
    /// Output vector of length d_model
    pub fn step(&mut self, input: &[f64]) -> Vec<f64> {
        // Value projection
        let v = linear_forward(&self.value_weights, &self.value_bias, input);

        // SSM pass: each SSM processes one dimension of v
        let ssm_out: Vec<f64> = self
            .ssm_layers
            .iter_mut()
            .enumerate()
            .map(|(i, ssm)| ssm.step(v[i]))
            .collect();

        // Gate: linear + GELU
        let gate: Vec<f64> = linear_forward(&self.gate_weights, &self.gate_bias, input)
            .into_iter()
            .map(gelu)
            .collect();

        // Merge: element-wise multiply
        let merged: Vec<f64> = ssm_out
            .iter()
            .zip(gate.iter())
            .map(|(s, g)| s * g)
            .collect();

        // Output projection
        linear_forward(&self.output_weights, &self.output_bias, &merged)
    }

    /// Process a full sequence through the gated SSM block.
    ///
    /// # Arguments
    /// * `sequence` - Sequence of feature vectors, each of length d_model
    ///
    /// # Returns
    /// Sequence of output vectors, each of length d_model
    pub fn forward(&mut self, sequence: &[Vec<f64>]) -> Vec<Vec<f64>> {
        self.reset();
        sequence.iter().map(|input| self.step(input)).collect()
    }

    /// Reset all SSM states to zero.
    pub fn reset(&mut self) {
        for ssm in &mut self.ssm_layers {
            ssm.reset();
        }
    }
}

/// Multi-layer Gated SSM model with residual connections.
pub struct GatedSSMModel {
    /// Stacked Gated SSM blocks
    pub layers: Vec<GatedSSMBlock>,
    /// Output projection weights
    pub head_weights: Vec<f64>,
    /// Output projection bias
    pub head_bias: f64,
    /// Model dimension
    pub d_model: usize,
}

impl GatedSSMModel {
    /// Create a new multi-layer Gated SSM model.
    ///
    /// # Arguments
    /// * `d_model` - Model dimension
    /// * `d_state` - SSM hidden state dimension
    /// * `n_layers` - Number of Gated SSM blocks
    /// * `expand` - Expansion factor
    pub fn new(d_model: usize, d_state: usize, n_layers: usize, expand: usize) -> Self {
        let layers: Vec<_> = (0..n_layers)
            .map(|_| GatedSSMBlock::new(d_model, d_state, expand))
            .collect();

        // Output head (d_model -> 1)
        let scale = 1.0 / (d_model as f64).sqrt();
        let head_weights: Vec<f64> = (0..d_model)
            .map(|i| ((i as f64) * 0.31).sin() * scale)
            .collect();

        GatedSSMModel {
            layers,
            head_weights,
            head_bias: 0.0,
            d_model,
        }
    }

    /// Process a sequence and return the final prediction.
    ///
    /// # Arguments
    /// * `sequence` - Sequence of feature vectors
    ///
    /// # Returns
    /// Scalar prediction (logit)
    pub fn predict(&mut self, sequence: &[Vec<f64>]) -> f64 {
        let mut h: Vec<Vec<f64>> = sequence.to_vec();

        for layer in &mut self.layers {
            layer.reset();
            let output = layer.forward(&h);
            // Residual connection
            h = h
                .iter()
                .zip(output.iter())
                .map(|(res, out)| {
                    res.iter().zip(out.iter()).map(|(r, o)| r + o).collect()
                })
                .collect();
        }

        // Take last time step
        let last = &h[h.len() - 1];

        // Linear head
        let logit: f64 = self
            .head_weights
            .iter()
            .zip(last.iter())
            .map(|(w, x)| w * x)
            .sum::<f64>()
            + self.head_bias;

        sigmoid(logit)
    }

    /// Reset all layers.
    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.reset();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gated_ssm_block() {
        let mut block = GatedSSMBlock::new(4, 16, 2);
        let input = vec![0.1, -0.2, 0.3, 0.0];
        let output = block.step(&input);
        assert_eq!(output.len(), 4);
        assert!(output.iter().all(|y| y.is_finite()));
    }

    #[test]
    fn test_gated_ssm_sequence() {
        let mut block = GatedSSMBlock::new(4, 16, 2);
        let sequence: Vec<Vec<f64>> = (0..60)
            .map(|i| vec![(i as f64 * 0.1).sin(), 0.0, 0.1, -0.05])
            .collect();
        let output = block.forward(&sequence);
        assert_eq!(output.len(), 60);
        assert_eq!(output[0].len(), 4);
    }

    #[test]
    fn test_gated_ssm_model() {
        let mut model = GatedSSMModel::new(4, 16, 2, 2);
        let sequence: Vec<Vec<f64>> = (0..60)
            .map(|i| vec![(i as f64 * 0.1).sin(), 0.0, 0.1, -0.05])
            .collect();
        let pred = model.predict(&sequence);
        assert!(pred.is_finite());
        assert!(pred >= 0.0 && pred <= 1.0);
    }
}
