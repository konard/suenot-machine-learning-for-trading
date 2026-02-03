//! Hierarchical State Space Model network implementation.
//!
//! Implements a multi-level SSM architecture where each level operates
//! at a different temporal resolution, capturing patterns from tick-level
//! microstructure to weekly macro trends.

use rand::Rng;
use rand_distr::Normal;
use std::f64;

/// Configuration for the Hierarchical SSM.
#[derive(Debug, Clone)]
pub struct HiSSConfig {
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub output_dim: usize,
    pub num_levels: usize,
    pub downsample_factors: Vec<usize>,
    pub ssm_state_dim: usize,
    pub dropout: f64,
}

impl Default for HiSSConfig {
    fn default() -> Self {
        Self {
            input_dim: 8,
            hidden_dim: 64,
            output_dim: 3,
            num_levels: 3,
            downsample_factors: vec![4, 4],
            ssm_state_dim: 16,
            dropout: 0.1,
        }
    }
}

/// Model prediction output.
#[derive(Debug, Clone)]
pub struct Prediction {
    pub predicted_return: f64,
    pub direction_probability: f64,
    pub predicted_volatility: f64,
}

/// Single SSM layer implementing discrete state space dynamics.
///
/// x_k = A_bar * x_{k-1} + B_bar * u_k
/// y_k = C * x_k + D * u_k
#[derive(Debug, Clone)]
pub struct SSMLayer {
    state_dim: usize,
    input_dim: usize,
    output_dim: usize,
    /// Diagonal state matrix (log-space for stability)
    log_a: Vec<f64>,
    /// Input projection matrix (state_dim x input_dim)
    b: Vec<Vec<f64>>,
    /// Output projection matrix (output_dim x state_dim)
    c: Vec<Vec<f64>>,
    /// Direct feedthrough (output_dim x input_dim)
    d: Vec<Vec<f64>>,
    /// Discretization step (log-space)
    log_delta: f64,
}

impl SSMLayer {
    pub fn new(input_dim: usize, state_dim: usize, output_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 0.01).unwrap();

        let log_a: Vec<f64> = (0..state_dim)
            .map(|_| (0.5 + 0.5 * rng.gen::<f64>()).ln())
            .collect();

        let b: Vec<Vec<f64>> = (0..state_dim)
            .map(|_| (0..input_dim).map(|_| rng.sample(normal)).collect())
            .collect();

        let c: Vec<Vec<f64>> = (0..output_dim)
            .map(|_| (0..state_dim).map(|_| rng.sample(normal)).collect())
            .collect();

        let d: Vec<Vec<f64>> = (0..output_dim)
            .map(|_| (0..input_dim).map(|_| 0.0).collect())
            .collect();

        Self {
            state_dim,
            input_dim,
            output_dim,
            log_a,
            b,
            c,
            d,
            log_delta: (-2.0_f64).ln(), // delta = 0.01 approx
        }
    }

    /// Forward pass: process a sequence and return output sequence.
    ///
    /// Input shape: (seq_len, input_dim)
    /// Output shape: (seq_len, output_dim)
    pub fn forward(&self, input: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let seq_len = input.len();
        let delta = self.log_delta.exp();

        // Discretize
        let a_bar: Vec<f64> = self.log_a.iter().map(|la| (delta * (-la.exp())).exp()).collect();

        let b_bar: Vec<Vec<f64>> = (0..self.state_dim)
            .map(|i| {
                let a = -self.log_a[i].exp();
                let scale = (1.0 / a) * (a_bar[i] - 1.0) * delta;
                self.b[i].iter().map(|&bij| scale * bij).collect()
            })
            .collect();

        // Sequential scan
        let mut state = vec![0.0; self.state_dim];
        let mut outputs = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            // Update state: x = A_bar * x + B_bar * u
            for i in 0..self.state_dim {
                let mut bu = 0.0;
                for j in 0..self.input_dim {
                    bu += b_bar[i][j] * input[t][j];
                }
                state[i] = a_bar[i] * state[i] + bu;
            }

            // Output: y = C * x + D * u
            let mut output = vec![0.0; self.output_dim];
            for i in 0..self.output_dim {
                for j in 0..self.state_dim {
                    output[i] += self.c[i][j] * state[j];
                }
                for j in 0..self.input_dim {
                    output[i] += self.d[i][j] * input[t][j];
                }
            }
            outputs.push(output);
        }

        outputs
    }
}

/// SSM Block with normalization, activation, and residual connection.
#[derive(Debug, Clone)]
pub struct SSMBlock {
    ssm: SSMLayer,
    dim: usize,
    /// Linear projection weights (dim x dim)
    proj_weight: Vec<Vec<f64>>,
    proj_bias: Vec<f64>,
}

impl SSMBlock {
    pub fn new(dim: usize, state_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 0.01).unwrap();

        let proj_weight: Vec<Vec<f64>> = (0..dim)
            .map(|_| (0..dim).map(|_| rng.sample(normal)).collect())
            .collect();
        let proj_bias = vec![0.0; dim];

        Self {
            ssm: SSMLayer::new(dim, state_dim, dim),
            dim,
            proj_weight,
            proj_bias,
        }
    }

    /// Forward pass with residual connection.
    pub fn forward(&self, input: &[Vec<f64>]) -> Vec<Vec<f64>> {
        // Layer normalization
        let normed = layer_norm(input);

        // SSM forward
        let ssm_out = self.ssm.forward(&normed);

        // GELU activation + linear projection + residual
        let seq_len = input.len();
        let mut output = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            let activated: Vec<f64> = ssm_out[t].iter().map(|&x| gelu(x)).collect();

            // Linear projection
            let mut projected = vec![0.0; self.dim];
            for i in 0..self.dim {
                for j in 0..self.dim {
                    projected[i] += self.proj_weight[i][j] * activated[j];
                }
                projected[i] += self.proj_bias[i];
            }

            // Residual connection
            let residual: Vec<f64> = projected
                .iter()
                .zip(input[t].iter())
                .map(|(p, r)| p + r)
                .collect();
            output.push(residual);
        }

        output
    }
}

/// Hierarchical State Space Model for multi-scale financial prediction.
#[derive(Debug)]
pub struct HierarchicalSSM {
    config: HiSSConfig,
    /// Input projection weights (hidden_dim x input_dim)
    input_proj_weight: Vec<Vec<f64>>,
    input_proj_bias: Vec<f64>,
    /// SSM blocks at each hierarchy level
    ssm_blocks: Vec<SSMBlock>,
    /// Fusion layer weights (hidden_dim x (hidden_dim * num_levels))
    fusion_weight: Vec<Vec<f64>>,
    fusion_bias: Vec<f64>,
    /// Prediction head weights
    return_weight: Vec<f64>,
    return_bias: f64,
    direction_weight: Vec<f64>,
    direction_bias: f64,
    volatility_weight: Vec<f64>,
    volatility_bias: f64,
}

impl HierarchicalSSM {
    pub fn new(config: HiSSConfig) -> Self {
        assert_eq!(
            config.downsample_factors.len(),
            config.num_levels - 1,
            "downsample_factors length must be num_levels - 1"
        );

        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 0.01).unwrap();

        let input_proj_weight: Vec<Vec<f64>> = (0..config.hidden_dim)
            .map(|_| (0..config.input_dim).map(|_| rng.sample(normal)).collect())
            .collect();
        let input_proj_bias = vec![0.0; config.hidden_dim];

        let ssm_blocks: Vec<SSMBlock> = (0..config.num_levels)
            .map(|_| SSMBlock::new(config.hidden_dim, config.ssm_state_dim))
            .collect();

        let fusion_in = config.hidden_dim * config.num_levels;
        let fusion_weight: Vec<Vec<f64>> = (0..config.hidden_dim)
            .map(|_| (0..fusion_in).map(|_| rng.sample(normal)).collect())
            .collect();
        let fusion_bias = vec![0.0; config.hidden_dim];

        let return_weight: Vec<f64> = (0..config.hidden_dim).map(|_| rng.sample(normal)).collect();
        let direction_weight: Vec<f64> =
            (0..config.hidden_dim).map(|_| rng.sample(normal)).collect();
        let volatility_weight: Vec<f64> =
            (0..config.hidden_dim).map(|_| rng.sample(normal)).collect();

        Self {
            config,
            input_proj_weight,
            input_proj_bias,
            ssm_blocks,
            fusion_weight,
            fusion_bias,
            return_weight,
            return_bias: 0.0,
            direction_weight,
            direction_bias: 0.0,
            volatility_weight,
            volatility_bias: 0.0,
        }
    }

    /// Run forward pass on a sequence of features.
    ///
    /// Input: (seq_len, input_dim)
    /// Returns: Prediction with return, direction probability, and volatility
    pub fn predict(&self, input: &[Vec<f64>]) -> Prediction {
        let seq_len = input.len();
        let hidden_dim = self.config.hidden_dim;

        // Input projection
        let mut projected: Vec<Vec<f64>> = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            let mut h = vec![0.0; hidden_dim];
            for i in 0..hidden_dim {
                for j in 0..self.config.input_dim {
                    h[i] += self.input_proj_weight[i][j] * input[t][j];
                }
                h[i] += self.input_proj_bias[i];
            }
            projected.push(h);
        }

        // Process each hierarchy level
        let mut level_last_features: Vec<Vec<f64>> = Vec::new();

        // Level 0: full resolution
        let h0 = self.ssm_blocks[0].forward(&projected);
        level_last_features.push(h0.last().unwrap().clone());

        let mut current = h0;
        for level in 1..self.config.num_levels {
            // Downsample by averaging
            let factor = self.config.downsample_factors[level - 1];
            let downsampled = downsample(&current, factor);

            // Process at this level
            let hl = self.ssm_blocks[level].forward(&downsampled);
            level_last_features.push(hl.last().unwrap().clone());
            current = hl;
        }

        // Fuse last features from all levels
        let mut fused_input: Vec<f64> = Vec::new();
        for features in &level_last_features {
            fused_input.extend(features);
        }

        // Fusion linear layer + GELU
        let mut fused = vec![0.0; hidden_dim];
        for i in 0..hidden_dim {
            for j in 0..fused_input.len() {
                fused[i] += self.fusion_weight[i][j] * fused_input[j];
            }
            fused[i] = gelu(fused[i] + self.fusion_bias[i]);
        }

        // Prediction heads
        let mut pred_return = self.return_bias;
        let mut pred_dir = self.direction_bias;
        let mut pred_vol = self.volatility_bias;

        for i in 0..hidden_dim {
            pred_return += self.return_weight[i] * fused[i];
            pred_dir += self.direction_weight[i] * fused[i];
            pred_vol += self.volatility_weight[i] * fused[i];
        }

        Prediction {
            predicted_return: pred_return,
            direction_probability: sigmoid(pred_dir),
            predicted_volatility: softplus(pred_vol),
        }
    }
}

// --- Helper functions ---

/// Downsample a sequence by averaging groups of `factor` timesteps.
fn downsample(seq: &[Vec<f64>], factor: usize) -> Vec<Vec<f64>> {
    let dim = seq[0].len();
    let out_len = seq.len() / factor;
    let mut result = Vec::with_capacity(out_len.max(1));

    for i in 0..out_len {
        let mut avg = vec![0.0; dim];
        for j in 0..factor {
            let idx = i * factor + j;
            if idx < seq.len() {
                for d in 0..dim {
                    avg[d] += seq[idx][d];
                }
            }
        }
        for d in 0..dim {
            avg[d] /= factor as f64;
        }
        result.push(avg);
    }

    if result.is_empty() {
        // If sequence is shorter than factor, average all
        let mut avg = vec![0.0; dim];
        for step in seq {
            for d in 0..dim {
                avg[d] += step[d];
            }
        }
        let n = seq.len() as f64;
        for d in 0..dim {
            avg[d] /= n.max(1.0);
        }
        result.push(avg);
    }

    result
}

/// Layer normalization over feature dimension.
fn layer_norm(input: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let eps = 1e-5;
    input
        .iter()
        .map(|x| {
            let mean: f64 = x.iter().sum::<f64>() / x.len() as f64;
            let var: f64 = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / x.len() as f64;
            let std = (var + eps).sqrt();
            x.iter().map(|v| (v - mean) / std).collect()
        })
        .collect()
}

/// GELU activation function.
fn gelu(x: f64) -> f64 {
    0.5 * x * (1.0 + (f64::consts::SQRT_2.recip() * x).tanh())
}

/// Sigmoid activation function.
fn sigmoid(x: f64) -> f64 {
    if x.is_nan() {
        return 0.5;
    }
    let clamped = x.clamp(-500.0, 500.0);
    1.0 / (1.0 + (-clamped).exp())
}

/// Softplus activation function (ensures positive output).
fn softplus(x: f64) -> f64 {
    if x.is_nan() {
        return 0.0;
    }
    let clamped = x.clamp(-500.0, 500.0);
    (1.0 + clamped.exp()).ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ssm_layer_forward() {
        let layer = SSMLayer::new(4, 8, 4);
        let input = vec![vec![1.0, 0.5, -0.3, 0.8]; 16];
        let output = layer.forward(&input);
        assert_eq!(output.len(), 16);
        assert_eq!(output[0].len(), 4);
    }

    #[test]
    fn test_ssm_block_residual() {
        let block = SSMBlock::new(4, 8);
        let input = vec![vec![1.0, 0.5, -0.3, 0.8]; 16];
        let output = block.forward(&input);
        assert_eq!(output.len(), 16);
        assert_eq!(output[0].len(), 4);
    }

    #[test]
    fn test_hierarchical_ssm_predict() {
        let config = HiSSConfig {
            input_dim: 4,
            hidden_dim: 8,
            num_levels: 3,
            downsample_factors: vec![4, 4],
            ssm_state_dim: 4,
            ..Default::default()
        };
        let model = HierarchicalSSM::new(config);
        let input = vec![vec![0.1, -0.2, 0.5, 0.3]; 64];
        let pred = model.predict(&input);

        assert!(pred.direction_probability >= 0.0 && pred.direction_probability <= 1.0);
        assert!(pred.predicted_volatility >= 0.0);
    }

    #[test]
    fn test_downsample() {
        let seq = vec![vec![1.0, 2.0]; 8];
        let ds = downsample(&seq, 4);
        assert_eq!(ds.len(), 2);
    }

    #[test]
    fn test_gelu() {
        assert!((gelu(0.0) - 0.0).abs() < 1e-6);
        assert!(gelu(1.0) > 0.0);
        assert!(gelu(-1.0) < 0.0);
    }

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(10.0) > 0.99);
        assert!(sigmoid(-10.0) < 0.01);
    }
}
