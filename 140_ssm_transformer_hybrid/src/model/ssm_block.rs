//! Mamba-style Selective State Space Model block.
//!
//! Implements input-dependent SSM with selective scan:
//! - Input projection to create B, C, delta parameters
//! - Discretization with softplus for delta
//! - Sequential scan through the state space
//! - Output gating with SiLU activation

use rand_distr::{Distribution, Normal};

/// A single linear layer: y = Wx + b
#[derive(Debug, Clone)]
pub struct LinearLayer {
    pub weights: Vec<Vec<f64>>,
    pub biases: Vec<f64>,
    pub input_size: usize,
    pub output_size: usize,
}

impl LinearLayer {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let std = (2.0 / (input_size + output_size) as f64).sqrt();
        let normal = Normal::new(0.0, std).unwrap();

        let weights = (0..output_size)
            .map(|_| (0..input_size).map(|_| normal.sample(&mut rng)).collect())
            .collect();
        let biases = vec![0.0; output_size];

        Self {
            weights,
            biases,
            input_size,
            output_size,
        }
    }

    pub fn forward(&self, input: &[f64]) -> Vec<f64> {
        assert_eq!(input.len(), self.input_size);
        (0..self.output_size)
            .map(|o| {
                let dot: f64 = self.weights[o].iter().zip(input).map(|(w, x)| w * x).sum();
                dot + self.biases[o]
            })
            .collect()
    }

    pub fn num_parameters(&self) -> usize {
        self.input_size * self.output_size + self.output_size
    }
}

/// Softplus activation: log(1 + exp(x))
fn softplus(x: f64) -> f64 {
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// SiLU (Swish) activation: x * sigmoid(x)
fn silu(x: f64) -> f64 {
    x / (1.0 + (-x).exp())
}

/// SSM Block implementing selective state space processing.
///
/// This block maintains a hidden state that is updated for each
/// timestep in the sequence, with input-dependent parameters
/// (B, C, delta) that control information flow.
#[derive(Debug, Clone)]
pub struct SSMBlock {
    pub d_model: usize,
    pub state_size: usize,
    /// Projects input to B parameters
    pub b_proj: LinearLayer,
    /// Projects input to C parameters
    pub c_proj: LinearLayer,
    /// Projects input to delta (step size) parameters
    pub delta_proj: LinearLayer,
    /// Input projection (expand to 2x for gating)
    pub in_proj: LinearLayer,
    /// Output projection
    pub out_proj: LinearLayer,
    /// Log-space diagonal A matrix
    pub a_log: Vec<Vec<f64>>,
}

impl SSMBlock {
    pub fn new(d_model: usize, state_size: usize) -> Self {
        // Initialize A_log as log of range 1..state_size
        let a_log: Vec<Vec<f64>> = (0..d_model)
            .map(|_| {
                (1..=state_size)
                    .map(|i| (i as f64).ln())
                    .collect()
            })
            .collect();

        Self {
            d_model,
            state_size,
            b_proj: LinearLayer::new(d_model, state_size),
            c_proj: LinearLayer::new(d_model, state_size),
            delta_proj: LinearLayer::new(d_model, d_model),
            in_proj: LinearLayer::new(d_model, d_model * 2),
            out_proj: LinearLayer::new(d_model, d_model),
            a_log,
        }
    }

    /// Process a sequence through the SSM block.
    ///
    /// # Arguments
    /// * `sequence` - Vec of timesteps, each a Vec<f64> of length d_model
    ///
    /// # Returns
    /// Output sequence of same shape
    pub fn forward(&self, sequence: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let seq_len = sequence.len();
        let mut outputs = Vec::with_capacity(seq_len);

        // Initialize hidden state
        let mut h = vec![vec![0.0; self.state_size]; self.d_model];

        for t in 0..seq_len {
            let x = &sequence[t];

            // Input projection and split into x_main and gate
            let xz = self.in_proj.forward(x);
            let (x_main, z): (Vec<f64>, Vec<f64>) = (
                xz[..self.d_model].to_vec(),
                xz[self.d_model..].to_vec(),
            );

            // Input-dependent parameters
            let b = self.b_proj.forward(&x_main);
            let c = self.c_proj.forward(&x_main);
            let delta_raw = self.delta_proj.forward(&x_main);
            let delta: Vec<f64> = delta_raw.iter().map(|&d| softplus(d)).collect();

            // A matrix (negative exponential)
            // Selective scan step: h = dA * h + dB * x
            let mut y = vec![0.0; self.d_model];

            for d in 0..self.d_model {
                for s in 0..self.state_size {
                    let a_val = -self.a_log[d][s].exp();
                    let da = (delta[d] * a_val).exp();
                    let db = delta[d] * b[s];

                    h[d][s] = da * h[d][s] + db * x_main[d];
                    y[d] += h[d][s] * c[s];
                }
            }

            // Gated output: y = y * silu(z)
            let gated: Vec<f64> = y
                .iter()
                .zip(z.iter())
                .map(|(&yi, &zi)| yi * silu(zi))
                .collect();

            // Output projection
            let out = self.out_proj.forward(&gated);

            // Residual connection
            let residual: Vec<f64> = out
                .iter()
                .zip(sequence[t].iter())
                .map(|(&o, &r)| o + r)
                .collect();

            outputs.push(residual);
        }

        outputs
    }

    pub fn num_parameters(&self) -> usize {
        self.b_proj.num_parameters()
            + self.c_proj.num_parameters()
            + self.delta_proj.num_parameters()
            + self.in_proj.num_parameters()
            + self.out_proj.num_parameters()
            + self.d_model * self.state_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_layer() {
        let layer = LinearLayer::new(4, 3);
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let output = layer.forward(&input);
        assert_eq!(output.len(), 3);
    }

    #[test]
    fn test_ssm_block() {
        let block = SSMBlock::new(8, 4);
        let sequence: Vec<Vec<f64>> = (0..10)
            .map(|_| (0..8).map(|_| 0.1).collect())
            .collect();
        let output = block.forward(&sequence);
        assert_eq!(output.len(), 10);
        assert_eq!(output[0].len(), 8);
    }

    #[test]
    fn test_softplus() {
        assert!((softplus(0.0) - 0.6931).abs() < 0.01);
        assert!((softplus(25.0) - 25.0).abs() < 0.01);
    }

    #[test]
    fn test_silu() {
        assert!((silu(0.0) - 0.0).abs() < 0.01);
        assert!(silu(5.0) > 4.9);
    }
}
