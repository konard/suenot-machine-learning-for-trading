//! Core Mamba4Cast block implementation.
//!
//! This module implements the selective state space model (SSM) core
//! of Mamba4Cast, optimized for inference performance.

use ndarray::{Array1, Array2, Array3, Axis, s};

/// Mamba4Cast block implementing selective state space model.
///
/// The block processes sequences through:
/// 1. Input projection with gating
/// 2. Causal convolution for local context
/// 3. Selective SSM for long-range dependencies
/// 4. Output projection
pub struct Mamba4CastBlock {
    /// Model dimension
    pub d_model: usize,
    /// SSM state dimension
    pub d_state: usize,
    /// Inner dimension after expansion
    pub d_inner: usize,
    /// Convolution kernel size
    pub d_conv: usize,
    /// Maximum forecast horizon
    pub max_horizon: usize,

    // Model weights
    in_proj_weight: Array2<f32>,
    conv_weight: Array2<f32>,
    x_proj_weight: Array2<f32>,
    dt_proj_weight: Array2<f32>,
    dt_proj_bias: Array1<f32>,
    a_log: Array1<f32>,
    d_param: Array1<f32>,
    out_proj_weight: Array2<f32>,
}

impl Mamba4CastBlock {
    /// Create a new Mamba4Cast block with random initialization.
    ///
    /// # Arguments
    ///
    /// * `d_model` - Model dimension
    /// * `d_state` - SSM state dimension
    /// * `d_conv` - Convolution kernel size
    /// * `expand` - Expansion factor for inner dimension
    /// * `max_horizon` - Maximum forecast horizon
    pub fn new(
        d_model: usize,
        d_state: usize,
        d_conv: usize,
        expand: usize,
        max_horizon: usize,
    ) -> Self {
        let d_inner = expand * d_model;
        let dt_rank = (d_model + 15) / 16; // ceil(d_model / 16)

        // Initialize weights with small random values
        // In production, these would be loaded from a trained model
        let in_proj_weight = Array2::from_shape_fn(
            (d_model, d_inner * 2),
            |_| rand::random::<f32>() * 0.02 - 0.01
        );

        let conv_weight = Array2::from_shape_fn(
            (d_inner, d_conv),
            |_| rand::random::<f32>() * 0.02 - 0.01
        );

        let x_proj_weight = Array2::from_shape_fn(
            (d_inner, dt_rank + d_state * 2),
            |_| rand::random::<f32>() * 0.02 - 0.01
        );

        let dt_proj_weight = Array2::from_shape_fn(
            (dt_rank, d_inner),
            |_| rand::random::<f32>() * 0.02 - 0.01
        );

        let dt_proj_bias = Array1::from_shape_fn(
            d_inner,
            |_| rand::random::<f32>() * 0.1
        );

        // A parameter initialized with log of 1..d_state
        let a_log = Array1::from_iter(
            (1..=d_state).map(|i| (i as f32).ln())
        );

        // D parameter (skip connection)
        let d_param = Array1::ones(d_inner);

        let out_proj_weight = Array2::from_shape_fn(
            (d_inner, d_model),
            |_| rand::random::<f32>() * 0.02 - 0.01
        );

        Self {
            d_model,
            d_state,
            d_inner,
            d_conv,
            max_horizon,
            in_proj_weight,
            conv_weight,
            x_proj_weight,
            dt_proj_weight,
            dt_proj_bias,
            a_log,
            d_param,
            out_proj_weight,
        }
    }

    /// Forward pass through the Mamba4Cast block.
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor of shape (batch, seq_len, d_model)
    ///
    /// # Returns
    ///
    /// Output tensor of shape (batch, seq_len, d_model)
    pub fn forward(&self, x: &Array3<f32>) -> Array3<f32> {
        let (batch, seq_len, _) = x.dim();

        // Input projection
        let xz = self.linear_3d(x, &self.in_proj_weight);
        let (x_part, z_part) = self.split_last_dim(&xz);

        // Causal convolution
        let x_conv = self.causal_conv1d(&x_part);
        let x_act = self.silu_3d(&x_conv);

        // SSM computation
        let y = self.ssm(&x_act);

        // Gating
        let z_act = self.silu_3d(&z_part);
        let y_gated = self.elementwise_mul(&y, &z_act);

        // Output projection
        self.linear_3d(&y_gated, &self.out_proj_weight)
    }

    /// Selective State Space Model computation.
    fn ssm(&self, x: &Array3<f32>) -> Array3<f32> {
        let (batch, seq_len, d_inner) = x.dim();
        let dt_rank = (self.d_model + 15) / 16;

        // Project for dt, B, C
        let x_proj = self.linear_3d(x, &self.x_proj_weight);

        // Get A from log space (negative for stability)
        let a: Vec<f32> = self.a_log.iter().map(|v| -v.exp()).collect();

        // Initialize output
        let mut output = Array3::<f32>::zeros((batch, seq_len, d_inner));

        // Process each batch
        for b in 0..batch {
            // Initialize hidden state
            let mut h = Array2::<f32>::zeros((d_inner, self.d_state));

            for t in 0..seq_len {
                // Extract dt, B, C for this timestep
                let dt_raw: Vec<f32> = (0..dt_rank)
                    .map(|i| x_proj[[b, t, i]])
                    .collect();

                let b_vec: Vec<f32> = (0..self.d_state)
                    .map(|i| x_proj[[b, t, dt_rank + i]])
                    .collect();

                let c_vec: Vec<f32> = (0..self.d_state)
                    .map(|i| x_proj[[b, t, dt_rank + self.d_state + i]])
                    .collect();

                // Compute dt through projection and softplus
                let mut dt = vec![0.0f32; d_inner];
                for i in 0..d_inner {
                    let mut sum = self.dt_proj_bias[i];
                    for j in 0..dt_rank {
                        sum += dt_raw[j] * self.dt_proj_weight[[j, i]];
                    }
                    dt[i] = softplus(sum);
                }

                // Discretize and update state
                for i in 0..d_inner {
                    let x_t = x[[b, t, i]];

                    for j in 0..self.d_state {
                        let da = (dt[i] * a[j]).exp();
                        let db = dt[i] * b_vec[j];

                        h[[i, j]] = da * h[[i, j]] + db * x_t;
                    }

                    // Compute output
                    let mut y_t = 0.0f32;
                    for j in 0..self.d_state {
                        y_t += h[[i, j]] * c_vec[j];
                    }

                    // Add skip connection
                    output[[b, t, i]] = y_t + x[[b, t, i]] * self.d_param[i];
                }
            }
        }

        output
    }

    /// Linear transformation for 3D tensor.
    fn linear_3d(&self, x: &Array3<f32>, weight: &Array2<f32>) -> Array3<f32> {
        let (batch, seq_len, in_dim) = x.dim();
        let out_dim = weight.dim().1;

        let mut result = Array3::<f32>::zeros((batch, seq_len, out_dim));

        for b in 0..batch {
            for t in 0..seq_len {
                for o in 0..out_dim {
                    let mut sum = 0.0f32;
                    for i in 0..in_dim {
                        sum += x[[b, t, i]] * weight[[i, o]];
                    }
                    result[[b, t, o]] = sum;
                }
            }
        }

        result
    }

    /// Split tensor along last dimension.
    fn split_last_dim(&self, x: &Array3<f32>) -> (Array3<f32>, Array3<f32>) {
        let (batch, seq_len, dim) = x.dim();
        let half = dim / 2;

        let first = x.slice(s![.., .., 0..half]).to_owned();
        let second = x.slice(s![.., .., half..]).to_owned();

        (first, second)
    }

    /// Causal 1D convolution.
    fn causal_conv1d(&self, x: &Array3<f32>) -> Array3<f32> {
        let (batch, seq_len, d_inner) = x.dim();

        let mut result = Array3::<f32>::zeros((batch, seq_len, d_inner));

        for b in 0..batch {
            for i in 0..d_inner {
                for t in 0..seq_len {
                    let mut sum = 0.0f32;
                    for k in 0..self.d_conv {
                        let idx = t as i32 - k as i32;
                        if idx >= 0 {
                            sum += x[[b, idx as usize, i]] * self.conv_weight[[i, k]];
                        }
                    }
                    result[[b, t, i]] = sum;
                }
            }
        }

        result
    }

    /// SiLU activation for 3D tensor.
    fn silu_3d(&self, x: &Array3<f32>) -> Array3<f32> {
        x.mapv(|v| v * sigmoid(v))
    }

    /// Element-wise multiplication of two 3D tensors.
    fn elementwise_mul(&self, a: &Array3<f32>, b: &Array3<f32>) -> Array3<f32> {
        a * b
    }
}

/// Softplus activation function.
fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// Sigmoid activation function.
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_creation() {
        let block = Mamba4CastBlock::new(64, 16, 4, 2, 96);
        assert_eq!(block.d_model, 64);
        assert_eq!(block.d_state, 16);
        assert_eq!(block.d_inner, 128);
    }

    #[test]
    fn test_forward_pass() {
        let block = Mamba4CastBlock::new(32, 8, 4, 2, 48);

        let x = Array3::<f32>::zeros((1, 10, 32));
        let y = block.forward(&x);

        assert_eq!(y.dim(), (1, 10, 32));
    }

    #[test]
    fn test_softplus() {
        assert!(softplus(0.0) > 0.0);
        assert!(softplus(1.0) > 1.0);
        assert!((softplus(30.0) - 30.0).abs() < 0.001);
    }
}
