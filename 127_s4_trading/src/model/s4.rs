//! S4 (Structured State Space) layer implementation.
//!
//! This module implements the S4 architecture for sequence modeling,
//! with efficient handling of long-range dependencies.

use num_complex::Complex64;
use rand::Rng;
use rand_distr::{Distribution, Normal};

/// Configuration for S4 model.
#[derive(Debug, Clone)]
pub struct S4Config {
    /// Input/output dimension
    pub d_model: usize,
    /// State dimension (N)
    pub d_state: usize,
    /// Number of S4 layers
    pub n_layers: usize,
    /// Dropout rate
    pub dropout: f64,
    /// Whether to use bidirectional S4
    pub bidirectional: bool,
    /// Learning rate for step size
    pub dt_min: f64,
    /// Maximum step size
    pub dt_max: f64,
}

impl Default for S4Config {
    fn default() -> Self {
        Self {
            d_model: 64,
            d_state: 64,
            n_layers: 4,
            dropout: 0.1,
            bidirectional: false,
            dt_min: 0.001,
            dt_max: 0.1,
        }
    }
}

/// S4 Layer implementing structured state space.
///
/// The layer processes sequences through the state space equations:
/// x_k = A_bar * x_{k-1} + B_bar * u_k
/// y_k = C * x_k + D * u_k
#[derive(Debug, Clone)]
pub struct S4Layer {
    /// Input/output dimension
    pub d_model: usize,
    /// State dimension
    pub d_state: usize,
    /// A matrix eigenvalues (complex, diagonal)
    pub a_real: Vec<f64>,
    pub a_imag: Vec<f64>,
    /// B matrix (input projection)
    pub b: Vec<f64>,
    /// C matrix (output projection)
    pub c_real: Vec<f64>,
    pub c_imag: Vec<f64>,
    /// D scalar (feedthrough)
    pub d: f64,
    /// Log step size (learnable)
    pub log_dt: f64,
    /// Discretized matrices (cached)
    a_bar_real: Vec<f64>,
    a_bar_imag: Vec<f64>,
    b_bar: Vec<f64>,
}

impl S4Layer {
    /// Create a new S4 layer with HiPPO initialization.
    pub fn new(d_model: usize, d_state: usize) -> Self {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 1.0).unwrap();

        // Initialize HiPPO-LegS diagonal approximation
        // A eigenvalues: -(n + 1/2) + i * sqrt(n*(n+1))
        let mut a_real = Vec::with_capacity(d_state);
        let mut a_imag = Vec::with_capacity(d_state);

        for n in 0..d_state {
            let n_f = n as f64;
            a_real.push(-(n_f + 0.5));
            a_imag.push((n_f * (n_f + 1.0)).sqrt());
        }

        // Initialize B with sqrt(2n+1)
        let b: Vec<f64> = (0..d_state)
            .map(|n| (2.0 * n as f64 + 1.0).sqrt())
            .collect();

        // Initialize C randomly
        let c_real: Vec<f64> = (0..d_state)
            .map(|_| normal.sample(&mut rng) * 0.1)
            .collect();
        let c_imag: Vec<f64> = (0..d_state)
            .map(|_| normal.sample(&mut rng) * 0.1)
            .collect();

        // Initialize D
        let d = 0.0;

        // Initialize log_dt
        let log_dt = rng.gen_range(-4.0..-1.0); // dt between ~0.02 and ~0.4

        let mut layer = Self {
            d_model,
            d_state,
            a_real,
            a_imag,
            b,
            c_real,
            c_imag,
            d,
            log_dt,
            a_bar_real: vec![0.0; d_state],
            a_bar_imag: vec![0.0; d_state],
            b_bar: vec![0.0; d_state],
        };

        layer.discretize();
        layer
    }

    /// Discretize continuous-time SSM to discrete-time using ZOH.
    fn discretize(&mut self) {
        let dt = self.log_dt.exp();

        for i in 0..self.d_state {
            // A_bar = exp(A * dt) for diagonal A
            let a = Complex64::new(self.a_real[i], self.a_imag[i]);
            let a_bar = (a * dt).exp();
            self.a_bar_real[i] = a_bar.re;
            self.a_bar_imag[i] = a_bar.im;

            // B_bar = (A^-1)(exp(A*dt) - I) * B ≈ dt * B for small dt
            // For diagonal: B_bar_i = (exp(a_i * dt) - 1) / a_i * B_i
            let b_bar = if a.norm() > 1e-8 {
                ((a_bar - Complex64::new(1.0, 0.0)) / a) * self.b[i]
            } else {
                Complex64::new(dt * self.b[i], 0.0)
            };
            self.b_bar[i] = b_bar.re;
        }
    }

    /// Process a single timestep (recurrent mode).
    ///
    /// # Arguments
    /// * `state_real` - Real part of hidden state
    /// * `state_imag` - Imaginary part of hidden state
    /// * `input` - Input value
    ///
    /// # Returns
    /// Output value
    pub fn step(
        &self,
        state_real: &mut [f64],
        state_imag: &mut [f64],
        input: f64,
    ) -> f64 {
        let mut output = self.d * input;

        for i in 0..self.d_state {
            // x_k = A_bar * x_{k-1} + B_bar * u_k
            let new_real = self.a_bar_real[i] * state_real[i]
                         - self.a_bar_imag[i] * state_imag[i]
                         + self.b_bar[i] * input;
            let new_imag = self.a_bar_real[i] * state_imag[i]
                         + self.a_bar_imag[i] * state_real[i];

            state_real[i] = new_real;
            state_imag[i] = new_imag;

            // y_k = Re(C * x_k)
            output += self.c_real[i] * state_real[i] - self.c_imag[i] * state_imag[i];
        }

        output
    }

    /// Process a sequence (convolutional mode for training).
    ///
    /// # Arguments
    /// * `input` - Input sequence of length L
    ///
    /// # Returns
    /// Output sequence of length L
    pub fn forward(&self, input: &[f64]) -> Vec<f64> {
        let seq_len = input.len();
        let mut output = vec![0.0; seq_len];
        let mut state_real = vec![0.0; self.d_state];
        let mut state_imag = vec![0.0; self.d_state];

        for (t, &inp) in input.iter().enumerate() {
            output[t] = self.step(&mut state_real, &mut state_imag, inp);
        }

        output
    }

    /// Compute the SSM kernel for convolution.
    ///
    /// K = [C*B_bar, C*A_bar*B_bar, C*A_bar^2*B_bar, ...]
    pub fn compute_kernel(&self, length: usize) -> Vec<f64> {
        let mut kernel = vec![0.0; length];
        let mut power_real = vec![1.0; self.d_state];
        let mut power_imag = vec![0.0; self.d_state];

        for t in 0..length {
            for i in 0..self.d_state {
                // K[t] = C * A_bar^t * B_bar
                let contrib = (self.c_real[i] * power_real[i] - self.c_imag[i] * power_imag[i])
                            * self.b_bar[i];
                kernel[t] += contrib;

                // Update A_bar^t -> A_bar^(t+1)
                let new_real = self.a_bar_real[i] * power_real[i] - self.a_bar_imag[i] * power_imag[i];
                let new_imag = self.a_bar_real[i] * power_imag[i] + self.a_bar_imag[i] * power_real[i];
                power_real[i] = new_real;
                power_imag[i] = new_imag;
            }
        }

        kernel
    }

    /// Get the current step size.
    pub fn dt(&self) -> f64 {
        self.log_dt.exp()
    }
}

/// Simplified Diagonal S4 (S4D) layer.
///
/// Uses purely diagonal A matrix for simpler implementation.
#[derive(Debug, Clone)]
pub struct S4DLayer {
    /// State dimension
    pub d_state: usize,
    /// Diagonal of A (real eigenvalues)
    pub a_diag: Vec<f64>,
    /// B vector
    pub b: Vec<f64>,
    /// C vector
    pub c: Vec<f64>,
    /// D scalar
    pub d: f64,
    /// Log step size
    pub log_dt: f64,
    /// Discretized A diagonal
    a_bar: Vec<f64>,
    /// Discretized B
    b_bar: Vec<f64>,
}

impl S4DLayer {
    /// Create a new S4D layer.
    pub fn new(d_state: usize) -> Self {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 1.0).unwrap();

        // Initialize A with negative eigenvalues (stable system)
        let a_diag: Vec<f64> = (0..d_state)
            .map(|n| -(n as f64 + 0.5))
            .collect();

        // Initialize B
        let b: Vec<f64> = (0..d_state)
            .map(|n| (2.0 * n as f64 + 1.0).sqrt())
            .collect();

        // Initialize C randomly
        let c: Vec<f64> = (0..d_state)
            .map(|_| normal.sample(&mut rng) * 0.1)
            .collect();

        let d = 0.0;
        let log_dt = rng.gen_range(-4.0..-1.0);

        let mut layer = Self {
            d_state,
            a_diag,
            b,
            c,
            d,
            log_dt,
            a_bar: vec![0.0; d_state],
            b_bar: vec![0.0; d_state],
        };

        layer.discretize();
        layer
    }

    /// Discretize the layer.
    fn discretize(&mut self) {
        let dt = self.log_dt.exp();

        for i in 0..self.d_state {
            // A_bar = exp(a * dt)
            self.a_bar[i] = (self.a_diag[i] * dt).exp();

            // B_bar = (exp(a*dt) - 1) / a * b
            self.b_bar[i] = if self.a_diag[i].abs() > 1e-8 {
                (self.a_bar[i] - 1.0) / self.a_diag[i] * self.b[i]
            } else {
                dt * self.b[i]
            };
        }
    }

    /// Process a single timestep.
    pub fn step(&self, state: &mut [f64], input: f64) -> f64 {
        let mut output = self.d * input;

        for i in 0..self.d_state {
            state[i] = self.a_bar[i] * state[i] + self.b_bar[i] * input;
            output += self.c[i] * state[i];
        }

        output
    }

    /// Process a sequence.
    pub fn forward(&self, input: &[f64]) -> Vec<f64> {
        let seq_len = input.len();
        let mut output = vec![0.0; seq_len];
        let mut state = vec![0.0; self.d_state];

        for (t, &inp) in input.iter().enumerate() {
            output[t] = self.step(&mut state, inp);
        }

        output
    }
}

/// Complete S4 model with multiple layers.
#[derive(Debug, Clone)]
pub struct S4Model {
    /// Configuration
    pub config: S4Config,
    /// S4 layers
    pub layers: Vec<S4Layer>,
    /// Input embedding weights
    pub embedding: Vec<Vec<f64>>,
    /// Output weights
    pub output_weights: Vec<Vec<f64>>,
    /// Output bias
    pub output_bias: Vec<f64>,
}

impl S4Model {
    /// Create a new S4 model.
    pub fn new(config: S4Config) -> Self {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 1.0).unwrap();
        let scale = (1.0 / config.d_model as f64).sqrt();

        // Create S4 layers
        let layers: Vec<S4Layer> = (0..config.n_layers)
            .map(|_| S4Layer::new(config.d_model, config.d_state))
            .collect();

        // Input embedding (input_dim -> d_model)
        let input_dim = 5; // OHLCV
        let embedding: Vec<Vec<f64>> = (0..input_dim)
            .map(|_| {
                (0..config.d_model)
                    .map(|_| normal.sample(&mut rng) * scale)
                    .collect()
            })
            .collect();

        // Output weights (d_model -> 3 classes: BUY, HOLD, SELL)
        let output_weights: Vec<Vec<f64>> = (0..config.d_model)
            .map(|_| {
                (0..3)
                    .map(|_| normal.sample(&mut rng) * scale)
                    .collect()
            })
            .collect();

        let output_bias = vec![0.0; 3];

        Self {
            config,
            layers,
            embedding,
            output_weights,
            output_bias,
        }
    }

    /// Process input features through the model.
    ///
    /// # Arguments
    /// * `features` - Input features of shape [seq_len, input_dim]
    ///
    /// # Returns
    /// Trading signal prediction
    pub fn predict(&self, features: &[Vec<f64>]) -> crate::trading::signals::TradingSignal {
        let seq_len = features.len();
        if seq_len == 0 {
            return crate::trading::signals::TradingSignal::Hold { confidence: 0.5 };
        }

        // Embed input
        let mut hidden: Vec<Vec<f64>> = features
            .iter()
            .map(|f| self.embed(f))
            .collect();

        // Process through S4 layers
        for layer in &self.layers {
            hidden = self.process_layer(layer, &hidden);
        }

        // Get last timestep output
        let last_hidden = &hidden[seq_len - 1];

        // Compute logits
        let mut logits = self.output_bias.clone();
        for (i, &h) in last_hidden.iter().enumerate() {
            for (j, logit) in logits.iter_mut().enumerate() {
                *logit += h * self.output_weights[i][j];
            }
        }

        // Softmax
        let max_logit = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_sum: f64 = logits.iter().map(|&l| (l - max_logit).exp()).sum();
        let probs: Vec<f64> = logits.iter().map(|&l| (l - max_logit).exp() / exp_sum).collect();

        // Return signal
        let (max_idx, max_prob) = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        match max_idx {
            0 => crate::trading::signals::TradingSignal::Buy { confidence: *max_prob },
            1 => crate::trading::signals::TradingSignal::Hold { confidence: *max_prob },
            2 => crate::trading::signals::TradingSignal::Sell { confidence: *max_prob },
            _ => crate::trading::signals::TradingSignal::Hold { confidence: 0.5 },
        }
    }

    /// Embed input features.
    fn embed(&self, features: &[f64]) -> Vec<f64> {
        let mut embedded = vec![0.0; self.config.d_model];

        for (i, &f) in features.iter().enumerate() {
            if i < self.embedding.len() {
                for (j, &w) in self.embedding[i].iter().enumerate() {
                    embedded[j] += f * w;
                }
            }
        }

        embedded
    }

    /// Process hidden states through an S4 layer.
    fn process_layer(&self, layer: &S4Layer, hidden: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let seq_len = hidden.len();
        let d_model = self.config.d_model;
        let mut output = vec![vec![0.0; d_model]; seq_len];

        // Process each dimension independently
        for dim in 0..d_model {
            let input: Vec<f64> = hidden.iter().map(|h| h[dim]).collect();
            let out = layer.forward(&input);
            for (t, &o) in out.iter().enumerate() {
                output[t][dim] = o;
            }
        }

        // Apply residual connection and simple nonlinearity
        for t in 0..seq_len {
            for dim in 0..d_model {
                output[t][dim] = output[t][dim] + hidden[t][dim]; // Residual
                output[t][dim] = output[t][dim].max(0.0); // ReLU
            }
        }

        output
    }

    /// Get the hidden state for the last timestep.
    pub fn get_state(&self, features: &[Vec<f64>]) -> Vec<f64> {
        let seq_len = features.len();
        if seq_len == 0 {
            return vec![0.0; self.config.d_model];
        }

        let mut hidden: Vec<Vec<f64>> = features
            .iter()
            .map(|f| self.embed(f))
            .collect();

        for layer in &self.layers {
            hidden = self.process_layer(layer, &hidden);
        }

        hidden[seq_len - 1].clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_s4_layer_creation() {
        let layer = S4Layer::new(64, 32);
        assert_eq!(layer.d_model, 64);
        assert_eq!(layer.d_state, 32);
        assert!(layer.dt() > 0.0);
    }

    #[test]
    fn test_s4_forward() {
        let layer = S4Layer::new(1, 16);
        let input: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        let output = layer.forward(&input);
        assert_eq!(output.len(), input.len());
    }

    #[test]
    fn test_s4d_layer() {
        let layer = S4DLayer::new(32);
        let input: Vec<f64> = (0..50).map(|i| (i as f64 * 0.1).sin()).collect();
        let output = layer.forward(&input);
        assert_eq!(output.len(), 50);
    }

    #[test]
    fn test_s4_model() {
        let config = S4Config::default();
        let model = S4Model::new(config);

        let features: Vec<Vec<f64>> = (0..100)
            .map(|i| vec![
                100.0 + (i as f64 * 0.01).sin(),
                101.0,
                99.0,
                100.0 + (i as f64 * 0.01).sin(),
                1000.0,
            ])
            .collect();

        let signal = model.predict(&features);
        assert!(matches!(signal,
            crate::trading::signals::TradingSignal::Buy { .. } |
            crate::trading::signals::TradingSignal::Hold { .. } |
            crate::trading::signals::TradingSignal::Sell { .. }
        ));
    }
}
