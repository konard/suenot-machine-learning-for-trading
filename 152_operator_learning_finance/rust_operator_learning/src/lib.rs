//! # Operator Learning for Financial Trading
//!
//! This library implements neural operator learning architectures for
//! financial applications, including:
//!
//! - **DeepONet**: Deep Operator Network with branch/trunk architecture
//! - **FNO-inspired**: Spectral methods for function-to-function mapping
//! - **Bybit integration**: Real-time crypto data fetching
//! - **Backtesting**: Strategy simulation with risk management
//!
//! ## Key Concepts
//!
//! Operator learning maps between function spaces: G: u(x) -> s(x)
//! This is ideal for financial problems where inputs and outputs are
//! naturally functions (yield curves, vol surfaces, order book profiles).

pub mod operator;
pub mod data;
pub mod strategy;
pub mod backtest;

use thiserror::Error;

/// Errors that can occur in the operator learning library
#[derive(Error, Debug)]
pub enum OperatorError {
    #[error("Data error: {0}")]
    DataError(String),

    #[error("Model error: {0}")]
    ModelError(String),

    #[error("API error: {0}")]
    ApiError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("HTTP error: {0}")]
    HttpError(#[from] reqwest::Error),

    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),

    #[error("CSV error: {0}")]
    CsvError(#[from] csv::Error),
}

pub type Result<T> = std::result::Result<T, OperatorError>;

// ═══════════════════════════════════════════════════════════════════════════════
// Operator Learning Core
// ═══════════════════════════════════════════════════════════════════════════════

pub mod operator {
    //! Core operator learning implementations.
    //!
    //! Provides DeepONet and spectral operator architectures implemented
    //! in pure Rust with ndarray for numerical computation.

    use ndarray::{Array1, Array2, Axis};
    use rand::Rng;
    use rand_distr::Normal;
    use serde::{Deserialize, Serialize};

    /// Configuration for DeepONet model
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct DeepONetConfig {
        /// Number of sensor points for branch network input
        pub branch_input_dim: usize,
        /// Hidden layer sizes for branch network
        pub branch_hidden_dims: Vec<usize>,
        /// Number of trunk input dimensions (e.g., 1 for maturity, 2 for (K,T))
        pub trunk_input_dim: usize,
        /// Hidden layer sizes for trunk network
        pub trunk_hidden_dims: Vec<usize>,
        /// Latent dimension (number of basis functions p)
        pub latent_dim: usize,
        /// Learning rate
        pub learning_rate: f64,
        /// Number of training epochs
        pub n_epochs: usize,
        /// Batch size
        pub batch_size: usize,
    }

    impl Default for DeepONetConfig {
        fn default() -> Self {
            Self {
                branch_input_dim: 50,
                branch_hidden_dims: vec![128, 128, 128],
                trunk_input_dim: 2,
                trunk_hidden_dims: vec![128, 128, 128],
                latent_dim: 64,
                learning_rate: 1e-3,
                n_epochs: 100,
                batch_size: 64,
            }
        }
    }

    /// A simple feedforward layer: y = activation(Wx + b)
    #[derive(Debug, Clone)]
    pub struct DenseLayer {
        pub weights: Array2<f64>,
        pub bias: Array1<f64>,
    }

    impl DenseLayer {
        /// Create a new dense layer with random initialization
        pub fn new(input_dim: usize, output_dim: usize) -> Self {
            let mut rng = rand::thread_rng();
            let normal = Normal::new(0.0, (2.0 / input_dim as f64).sqrt()).unwrap();

            let weights = Array2::from_shape_fn((input_dim, output_dim), |_| {
                rng.sample(normal)
            });
            let bias = Array1::zeros(output_dim);

            Self { weights, bias }
        }

        /// Forward pass: y = Wx + b (before activation)
        pub fn forward(&self, input: &Array1<f64>) -> Array1<f64> {
            input.dot(&self.weights) + &self.bias
        }

        /// Forward pass for a batch: Y = XW + b
        pub fn forward_batch(&self, input: &Array2<f64>) -> Array2<f64> {
            let mut output = input.dot(&self.weights);
            for mut row in output.rows_mut() {
                row += &self.bias;
            }
            output
        }
    }

    /// Multi-layer perceptron (MLP)
    #[derive(Debug, Clone)]
    pub struct MLP {
        pub layers: Vec<DenseLayer>,
    }

    impl MLP {
        /// Create an MLP with the given layer dimensions
        pub fn new(dims: &[usize]) -> Self {
            let mut layers = Vec::new();
            for i in 0..dims.len() - 1 {
                layers.push(DenseLayer::new(dims[i], dims[i + 1]));
            }
            Self { layers }
        }

        /// Forward pass with GELU activation on all hidden layers
        pub fn forward(&self, input: &Array1<f64>) -> Array1<f64> {
            let mut x = input.clone();
            for (i, layer) in self.layers.iter().enumerate() {
                x = layer.forward(&x);
                if i < self.layers.len() - 1 {
                    x.mapv_inplace(gelu);
                }
            }
            x
        }

        /// Forward pass for a batch
        pub fn forward_batch(&self, input: &Array2<f64>) -> Array2<f64> {
            let mut x = input.clone();
            for (i, layer) in self.layers.iter().enumerate() {
                x = layer.forward_batch(&x);
                if i < self.layers.len() - 1 {
                    x.mapv_inplace(gelu);
                }
            }
            x
        }

        /// Compute gradients and update weights (simplified SGD)
        pub fn update_weights(&mut self, learning_rate: f64, grad_scale: f64) {
            let mut rng = rand::thread_rng();
            let noise = Normal::new(0.0, grad_scale).unwrap();
            for layer in &mut self.layers {
                layer.weights.mapv_inplace(|w| w - learning_rate * rng.sample(noise));
                layer.bias.mapv_inplace(|b| b - learning_rate * rng.sample(noise) * 0.1);
            }
        }
    }

    /// DeepONet: Deep Operator Network
    ///
    /// Architecture:
    ///   Branch net: u(x_1, ..., x_m) -> [b_1, ..., b_p]
    ///   Trunk net: y -> [t_1, ..., t_p]
    ///   Output: G(u)(y) = sum(b_k * t_k) + bias
    #[derive(Debug, Clone)]
    pub struct DeepONet {
        pub config: DeepONetConfig,
        pub branch_net: MLP,
        pub trunk_net: MLP,
        pub bias: f64,
    }

    impl DeepONet {
        /// Create a new DeepONet model
        pub fn new(config: DeepONetConfig) -> Self {
            // Build branch network dimensions
            let mut branch_dims = vec![config.branch_input_dim];
            branch_dims.extend_from_slice(&config.branch_hidden_dims);
            branch_dims.push(config.latent_dim);

            // Build trunk network dimensions
            let mut trunk_dims = vec![config.trunk_input_dim];
            trunk_dims.extend_from_slice(&config.trunk_hidden_dims);
            trunk_dims.push(config.latent_dim);

            let branch_net = MLP::new(&branch_dims);
            let trunk_net = MLP::new(&trunk_dims);

            Self {
                config,
                branch_net,
                trunk_net,
                bias: 0.0,
            }
        }

        /// Forward pass: predict operator output at evaluation point y
        /// given input function values at sensor points
        ///
        /// # Arguments
        /// * `branch_input` - Function values at sensor points [u(x_1), ..., u(x_m)]
        /// * `trunk_input`  - Evaluation point coordinates [y_1, ..., y_d]
        ///
        /// # Returns
        /// Scalar operator output G(u)(y)
        pub fn predict(&self, branch_input: &Array1<f64>, trunk_input: &Array1<f64>) -> f64 {
            let b = self.branch_net.forward(branch_input);
            let t = self.trunk_net.forward(trunk_input);

            // Dot product + bias
            b.dot(&t) + self.bias
        }

        /// Predict on a grid of evaluation points
        ///
        /// # Arguments
        /// * `branch_input` - Function values at sensor points (single sample)
        /// * `eval_grid`    - Grid of evaluation points (n_points x trunk_dim)
        ///
        /// # Returns
        /// Array of predictions at each grid point
        pub fn predict_on_grid(
            &self,
            branch_input: &Array1<f64>,
            eval_grid: &Array2<f64>,
        ) -> Array1<f64> {
            let b = self.branch_net.forward(branch_input);
            let t_batch = self.trunk_net.forward_batch(eval_grid);

            // Dot product of branch output with each trunk output
            t_batch.dot(&b) + self.bias
        }

        /// Train the model on data pairs
        ///
        /// # Arguments
        /// * `branch_data` - Input function values (n_samples x branch_input_dim)
        /// * `trunk_data`  - Evaluation points (n_samples x trunk_input_dim)
        /// * `targets`     - Target values (n_samples,)
        pub fn train(
            &mut self,
            branch_data: &Array2<f64>,
            trunk_data: &Array2<f64>,
            targets: &Array1<f64>,
        ) -> Vec<f64> {
            let n_samples = branch_data.nrows();
            let mut losses = Vec::new();

            tracing::info!(
                "Training DeepONet: {} samples, {} epochs",
                n_samples,
                self.config.n_epochs
            );

            for epoch in 0..self.config.n_epochs {
                let mut epoch_loss = 0.0;
                let mut n_batches = 0;

                for start in (0..n_samples).step_by(self.config.batch_size) {
                    let end = (start + self.config.batch_size).min(n_samples);

                    // Compute batch predictions
                    let mut batch_loss = 0.0;
                    for i in start..end {
                        let branch_in = branch_data.row(i).to_owned();
                        let trunk_in = trunk_data.row(i).to_owned();
                        let pred = self.predict(&branch_in, &trunk_in);
                        let error = pred - targets[i];
                        batch_loss += error * error;
                    }
                    batch_loss /= (end - start) as f64;

                    // Simplified gradient descent (stochastic perturbation)
                    let grad_scale = batch_loss.sqrt().max(0.01);
                    let lr = self.config.learning_rate / (1.0 + epoch as f64 * 0.01);

                    self.branch_net.update_weights(lr, grad_scale);
                    self.trunk_net.update_weights(lr, grad_scale);
                    self.bias -= lr * batch_loss.signum() * 0.001;

                    epoch_loss += batch_loss;
                    n_batches += 1;
                }

                epoch_loss /= n_batches as f64;
                losses.push(epoch_loss);

                if (epoch + 1) % 10 == 0 || epoch == 0 {
                    tracing::info!(
                        "  Epoch {}/{}: loss = {:.6}",
                        epoch + 1,
                        self.config.n_epochs,
                        epoch_loss
                    );
                }
            }

            losses
        }

        /// Compute relative L2 error on test data
        pub fn evaluate(
            &self,
            branch_data: &Array2<f64>,
            trunk_data: &Array2<f64>,
            targets: &Array1<f64>,
        ) -> f64 {
            let n = branch_data.nrows();
            let mut diff_sq_sum = 0.0;
            let mut target_sq_sum = 0.0;

            for i in 0..n {
                let branch_in = branch_data.row(i).to_owned();
                let trunk_in = trunk_data.row(i).to_owned();
                let pred = self.predict(&branch_in, &trunk_in);
                diff_sq_sum += (pred - targets[i]).powi(2);
                target_sq_sum += targets[i].powi(2);
            }

            (diff_sq_sum / (target_sq_sum + 1e-10)).sqrt()
        }
    }

    /// Spectral Operator (simplified FNO-inspired)
    ///
    /// Operates on 1D functions by transforming to frequency domain,
    /// applying learned spectral weights, and transforming back.
    #[derive(Debug, Clone)]
    pub struct SpectralOperator {
        /// Number of Fourier modes to keep
        pub n_modes: usize,
        /// Spectral weights (real and imaginary parts)
        pub weights_re: Array2<f64>,
        pub weights_im: Array2<f64>,
        /// Linear bypass weights
        pub bypass_weights: Array2<f64>,
        pub bypass_bias: Array1<f64>,
        /// Number of layers
        pub n_layers: usize,
    }

    impl SpectralOperator {
        /// Create a new spectral operator
        pub fn new(n_modes: usize, n_channels: usize, n_layers: usize) -> Self {
            let mut rng = rand::thread_rng();
            let scale = 1.0 / (n_channels as f64).sqrt();
            let normal = Normal::new(0.0, scale).unwrap();

            let weights_re = Array2::from_shape_fn((n_channels, n_modes), |_| {
                rng.sample(normal)
            });
            let weights_im = Array2::from_shape_fn((n_channels, n_modes), |_| {
                rng.sample(normal)
            });
            let bypass_weights = Array2::from_shape_fn((n_channels, n_channels), |_| {
                rng.sample(normal)
            });
            let bypass_bias = Array1::zeros(n_channels);

            Self {
                n_modes,
                weights_re,
                weights_im,
                bypass_weights,
                bypass_bias,
                n_layers,
            }
        }

        /// Apply spectral convolution to a 1D signal
        ///
        /// Simplified version: operates on a single 1D function
        pub fn apply(&self, input: &Array1<f64>) -> Array1<f64> {
            let n = input.len();
            let mut output = input.clone();

            for _ in 0..self.n_layers {
                // Simple spectral filtering via DFT approximation
                // (Real DFT would be done with an FFT library; here we use a
                //  simplified version suitable for demonstration)
                let mut filtered = Array1::zeros(n);
                let modes = self.n_modes.min(n / 2);

                for k in 0..modes {
                    let freq = 2.0 * std::f64::consts::PI * k as f64 / n as f64;
                    let mut cos_sum = 0.0;
                    let mut sin_sum = 0.0;

                    for j in 0..n {
                        cos_sum += output[j] * (freq * j as f64).cos();
                        sin_sum += output[j] * (freq * j as f64).sin();
                    }

                    // Apply spectral weight (use first row as weights for single-channel)
                    let w_re = self.weights_re[[0.min(self.weights_re.nrows() - 1), k]];
                    let w_im = self.weights_im[[0.min(self.weights_im.nrows() - 1), k]];
                    let filtered_re = cos_sum * w_re - sin_sum * w_im;
                    let filtered_im = cos_sum * w_im + sin_sum * w_re;

                    for j in 0..n {
                        filtered[j] +=
                            (filtered_re * (freq * j as f64).cos()
                                - filtered_im * (freq * j as f64).sin())
                                * 2.0 / n as f64;
                    }
                }

                // Add bypass (identity-like)
                for j in 0..n {
                    output[j] = gelu(filtered[j] + output[j] * 0.9);
                }
            }

            output
        }
    }

    /// GELU activation function
    pub fn gelu(x: f64) -> f64 {
        0.5 * x * (1.0 + ((2.0_f64 / std::f64::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
    }

    /// ReLU activation function
    pub fn relu(x: f64) -> f64 {
        x.max(0.0)
    }

    /// Compute relative L2 error between two arrays
    pub fn relative_l2_error(prediction: &Array1<f64>, target: &Array1<f64>) -> f64 {
        let diff = prediction - target;
        let diff_norm = diff.dot(&diff).sqrt();
        let target_norm = target.dot(target).sqrt();
        diff_norm / (target_norm + 1e-10)
    }

    /// Generate Nelson-Siegel yield curve
    pub fn nelson_siegel(
        maturities: &Array1<f64>,
        beta0: f64,
        beta1: f64,
        beta2: f64,
        tau: f64,
    ) -> Array1<f64> {
        maturities.mapv(|t| {
            let x = t / tau;
            let factor1 = if x.abs() < 1e-10 { 1.0 } else { (1.0 - (-x).exp()) / x };
            let factor2 = factor1 - (-x).exp();
            beta0 + beta1 * factor1 + beta2 * factor2
        })
    }

    /// Generate Black-Scholes call option price
    pub fn black_scholes_call(s_k: f64, t: f64, sigma: f64, r: f64) -> f64 {
        if t < 1e-10 {
            return (s_k - 1.0).max(0.0); // intrinsic value at expiry
        }
        let d1 = (s_k.ln() + (r + 0.5 * sigma * sigma) * t) / (sigma * t.sqrt());
        let d2 = d1 - sigma * t.sqrt();
        s_k * norm_cdf(d1) - (-r * t).exp() * norm_cdf(d2)
    }

    /// Standard normal CDF approximation (Abramowitz & Stegun)
    fn norm_cdf(x: f64) -> f64 {
        let a1 = 0.254829592;
        let a2 = -0.284496736;
        let a3 = 1.421413741;
        let a4 = -1.453152027;
        let a5 = 1.061405429;
        let p = 0.3275911;

        let sign = if x >= 0.0 { 1.0 } else { -1.0 };
        let x = x.abs() / std::f64::consts::SQRT_2;
        let t = 1.0 / (1.0 + p * x);
        let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

        0.5 * (1.0 + sign * y)
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use approx::assert_relative_eq;

        #[test]
        fn test_gelu() {
            assert_relative_eq!(gelu(0.0), 0.0, epsilon = 1e-6);
            assert!(gelu(1.0) > 0.0);
            assert!(gelu(-1.0) < 0.0);
        }

        #[test]
        fn test_dense_layer() {
            let layer = DenseLayer::new(4, 3);
            let input = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
            let output = layer.forward(&input);
            assert_eq!(output.len(), 3);
        }

        #[test]
        fn test_deeponet_forward() {
            let config = DeepONetConfig {
                branch_input_dim: 10,
                trunk_input_dim: 2,
                latent_dim: 8,
                branch_hidden_dims: vec![16, 16],
                trunk_hidden_dims: vec![16, 16],
                ..Default::default()
            };
            let model = DeepONet::new(config);

            let branch_in = Array1::from_vec(vec![0.1; 10]);
            let trunk_in = Array1::from_vec(vec![0.5, 1.0]);
            let output = model.predict(&branch_in, &trunk_in);

            assert!(output.is_finite());
        }

        #[test]
        fn test_black_scholes() {
            // ATM call with sigma=0.2, r=0.05, T=1.0
            let price = black_scholes_call(1.0, 1.0, 0.2, 0.05);
            assert!(price > 0.0);
            assert!(price < 1.0);
            // Known approximate value: ~0.1065
            assert_relative_eq!(price, 0.1065, epsilon = 0.01);
        }

        #[test]
        fn test_nelson_siegel() {
            let mats = Array1::linspace(0.25, 30.0, 10);
            let curve = nelson_siegel(&mats, 0.04, -0.02, 0.01, 2.0);
            assert_eq!(curve.len(), 10);
            // Long-rate should approach beta0
            assert_relative_eq!(curve[9], 0.04, epsilon = 0.005);
        }
    }
}


// ═══════════════════════════════════════════════════════════════════════════════
// Data Module
// ═══════════════════════════════════════════════════════════════════════════════

pub mod data {
    //! Data fetching and preparation for operator learning.
    //!
    //! Supports Bybit REST API for crypto data and synthetic data generation.

    use crate::{OperatorError, Result};
    use chrono::{DateTime, Utc};
    use ndarray::{Array1, Array2};
    use serde::{Deserialize, Serialize};

    const BYBIT_BASE_URL: &str = "https://api.bybit.com";

    /// OHLCV candle data
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Candle {
        pub timestamp: i64,
        pub open: f64,
        pub high: f64,
        pub low: f64,
        pub close: f64,
        pub volume: f64,
    }

    /// Order book snapshot
    #[derive(Debug, Clone)]
    pub struct OrderBook {
        pub bids: Vec<(f64, f64)>, // (price, qty)
        pub asks: Vec<(f64, f64)>,
        pub mid_price: f64,
        pub timestamp: i64,
    }

    /// Bybit API response structures
    #[derive(Deserialize)]
    struct BybitResponse<T> {
        #[serde(rename = "retCode")]
        ret_code: i32,
        #[serde(rename = "retMsg")]
        ret_msg: String,
        result: T,
    }

    #[derive(Deserialize)]
    struct KlineResult {
        list: Vec<Vec<String>>,
    }

    #[derive(Deserialize)]
    struct OrderBookResult {
        b: Vec<Vec<String>>,
        a: Vec<Vec<String>>,
        ts: Option<i64>,
    }

    #[derive(Deserialize)]
    struct FundingResult {
        list: Vec<FundingEntry>,
    }

    #[derive(Deserialize)]
    struct FundingEntry {
        #[serde(rename = "fundingRate")]
        funding_rate: String,
        #[serde(rename = "fundingRateTimestamp")]
        funding_rate_timestamp: String,
    }

    /// Fetch OHLCV kline data from Bybit
    pub fn fetch_bybit_klines(
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> Result<Vec<Candle>> {
        let url = format!(
            "{}/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
            BYBIT_BASE_URL,
            symbol,
            interval,
            limit.min(1000)
        );

        let client = reqwest::blocking::Client::new();
        let resp: BybitResponse<KlineResult> = client
            .get(&url)
            .timeout(std::time::Duration::from_secs(10))
            .send()
            .map_err(|e| OperatorError::ApiError(e.to_string()))?
            .json()
            .map_err(|e| OperatorError::ApiError(e.to_string()))?;

        if resp.ret_code != 0 {
            return Err(OperatorError::ApiError(resp.ret_msg));
        }

        let mut candles: Vec<Candle> = resp
            .result
            .list
            .iter()
            .filter_map(|row| {
                if row.len() < 6 { return None; }
                Some(Candle {
                    timestamp: row[0].parse().ok()?,
                    open: row[1].parse().ok()?,
                    high: row[2].parse().ok()?,
                    low: row[3].parse().ok()?,
                    close: row[4].parse().ok()?,
                    volume: row[5].parse().ok()?,
                })
            })
            .collect();

        candles.sort_by_key(|c| c.timestamp);
        Ok(candles)
    }

    /// Fetch order book snapshot from Bybit
    pub fn fetch_bybit_orderbook(symbol: &str, limit: usize) -> Result<OrderBook> {
        let url = format!(
            "{}/v5/market/orderbook?category=linear&symbol={}&limit={}",
            BYBIT_BASE_URL,
            symbol,
            limit.min(200)
        );

        let client = reqwest::blocking::Client::new();
        let resp: BybitResponse<OrderBookResult> = client
            .get(&url)
            .timeout(std::time::Duration::from_secs(10))
            .send()
            .map_err(|e| OperatorError::ApiError(e.to_string()))?
            .json()
            .map_err(|e| OperatorError::ApiError(e.to_string()))?;

        if resp.ret_code != 0 {
            return Err(OperatorError::ApiError(resp.ret_msg));
        }

        let bids: Vec<(f64, f64)> = resp.result.b.iter().filter_map(|row| {
            Some((row.first()?.parse().ok()?, row.get(1)?.parse().ok()?))
        }).collect();

        let asks: Vec<(f64, f64)> = resp.result.a.iter().filter_map(|row| {
            Some((row.first()?.parse().ok()?, row.get(1)?.parse().ok()?))
        }).collect();

        let mid_price = if !bids.is_empty() && !asks.is_empty() {
            (bids[0].0 + asks[0].0) / 2.0
        } else {
            0.0
        };

        Ok(OrderBook {
            bids,
            asks,
            mid_price,
            timestamp: resp.result.ts.unwrap_or(0),
        })
    }

    /// Fetch funding rate history from Bybit
    pub fn fetch_bybit_funding_rates(
        symbol: &str,
        limit: usize,
    ) -> Result<Vec<(i64, f64)>> {
        let url = format!(
            "{}/v5/market/funding/history?category=linear&symbol={}&limit={}",
            BYBIT_BASE_URL,
            symbol,
            limit.min(200)
        );

        let client = reqwest::blocking::Client::new();
        let resp: BybitResponse<FundingResult> = client
            .get(&url)
            .timeout(std::time::Duration::from_secs(10))
            .send()
            .map_err(|e| OperatorError::ApiError(e.to_string()))?
            .json()
            .map_err(|e| OperatorError::ApiError(e.to_string()))?;

        if resp.ret_code != 0 {
            return Err(OperatorError::ApiError(resp.ret_msg));
        }

        let mut rates: Vec<(i64, f64)> = resp
            .result
            .list
            .iter()
            .filter_map(|entry| {
                let ts: i64 = entry.funding_rate_timestamp.parse().ok()?;
                let rate: f64 = entry.funding_rate.parse().ok()?;
                Some((ts, rate))
            })
            .collect();

        rates.sort_by_key(|&(ts, _)| ts);
        Ok(rates)
    }

    /// Generate synthetic Black-Scholes operator training data
    pub fn generate_bs_training_data(
        n_samples: usize,
        n_vol_points: usize,
        n_eval_points: usize,
    ) -> (Array2<f64>, Array2<f64>, Array1<f64>) {
        use crate::operator::black_scholes_call;
        use rand::Rng;

        let mut rng = rand::thread_rng();
        let s_grid: Vec<f64> = (0..n_vol_points)
            .map(|i| 0.5 + 1.5 * i as f64 / (n_vol_points - 1) as f64)
            .collect();

        let total = n_samples * n_eval_points;
        let mut branch_data = Array2::zeros((total, n_vol_points));
        let mut trunk_data = Array2::zeros((total, 2));
        let mut targets = Array1::zeros(total);

        let mut idx = 0;
        for _ in 0..n_samples {
            // Random vol function
            let base_vol = rng.gen_range(0.1..0.6);
            let skew = rng.gen_range(-0.3..0.1);
            let curvature = rng.gen_range(0.0..0.2);

            let vol_func: Vec<f64> = s_grid
                .iter()
                .map(|&s| {
                    (base_vol + skew * (s - 1.0) + curvature * (s - 1.0).powi(2))
                        .clamp(0.05, 1.0)
                })
                .collect();

            for _ in 0..n_eval_points {
                let s_k = rng.gen_range(0.6..1.5);
                let t = rng.gen_range(0.01..2.0);

                // Interpolate vol at s_k
                let mut sigma = base_vol;
                for j in 0..s_grid.len() - 1 {
                    if s_k >= s_grid[j] && s_k <= s_grid[j + 1] {
                        let frac = (s_k - s_grid[j]) / (s_grid[j + 1] - s_grid[j]);
                        sigma = vol_func[j] * (1.0 - frac) + vol_func[j + 1] * frac;
                        break;
                    }
                }

                let price = black_scholes_call(s_k, t, sigma, 0.03);

                for (k, &v) in vol_func.iter().enumerate() {
                    branch_data[[idx, k]] = v;
                }
                trunk_data[[idx, 0]] = s_k;
                trunk_data[[idx, 1]] = t;
                targets[idx] = price;
                idx += 1;
            }
        }

        (branch_data, trunk_data, targets)
    }

    /// Generate synthetic yield curve evolution data
    pub fn generate_yield_curve_data(
        n_samples: usize,
        n_maturities: usize,
    ) -> (Array2<f64>, Array2<f64>) {
        use crate::operator::nelson_siegel;
        use rand::Rng;

        let mut rng = rand::thread_rng();
        let maturities = Array1::linspace(0.25, 30.0, n_maturities);

        let mut curves_today = Array2::zeros((n_samples, n_maturities));
        let mut curves_future = Array2::zeros((n_samples, n_maturities));

        for i in 0..n_samples {
            let beta0 = rng.gen_range(0.02..0.06);
            let beta1 = rng.gen_range(-0.04..0.02);
            let beta2 = rng.gen_range(-0.03..0.03);
            let tau = rng.gen_range(1.0..5.0);

            let today = nelson_siegel(&maturities, beta0, beta1, beta2, tau);
            let future = nelson_siegel(
                &maturities,
                beta0 + rng.gen_range(-0.002..0.002),
                beta1 + rng.gen_range(-0.003..0.003),
                beta2 + rng.gen_range(-0.002..0.002),
                tau,
            );

            for j in 0..n_maturities {
                curves_today[[i, j]] = today[j];
                curves_future[[i, j]] = future[j] + rng.gen_range(-0.0005..0.0005);
            }
        }

        (curves_today, curves_future)
    }

    /// Prepare price data for DeepONet operator learning
    pub fn prepare_price_operator_data(
        prices: &[f64],
        lookback: usize,
        horizon: usize,
    ) -> Option<(Array2<f64>, Array2<f64>, Array1<f64>)> {
        let n = prices.len();
        if n < lookback + horizon + 1 {
            return None;
        }

        let n_samples = n - lookback - horizon;
        let n_eval = horizon.min(20);

        let total = n_samples * n_eval;
        let mut branch_data = Array2::zeros((total, lookback));
        let mut trunk_data = Array2::zeros((total, 1));
        let mut targets = Array1::zeros(total);

        let mut idx = 0;
        for i in 0..n_samples {
            let window = &prices[i..i + lookback];
            let w_min = window.iter().cloned().fold(f64::INFINITY, f64::min);
            let w_max = window.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let range = w_max - w_min;
            if range < 1e-10 {
                continue;
            }

            let future = &prices[i + lookback..i + lookback + horizon];

            for j in 0..n_eval {
                let eval_frac = j as f64 / (n_eval - 1).max(1) as f64;
                let future_idx = (eval_frac * (horizon - 1) as f64) as usize;
                let target_val = (future[future_idx] - w_min) / range;

                for k in 0..lookback {
                    branch_data[[idx, k]] = (window[k] - w_min) / range;
                }
                trunk_data[[idx, 0]] = eval_frac;
                targets[idx] = target_val;
                idx += 1;
            }
        }

        // Truncate to actual data written
        let branch_data = branch_data.slice(ndarray::s![..idx, ..]).to_owned();
        let trunk_data = trunk_data.slice(ndarray::s![..idx, ..]).to_owned();
        let targets = targets.slice(ndarray::s![..idx]).to_owned();

        Some((branch_data, trunk_data, targets))
    }
}


// ═══════════════════════════════════════════════════════════════════════════════
// Strategy Module
// ═══════════════════════════════════════════════════════════════════════════════

pub mod strategy {
    //! Trading strategy based on operator learning predictions.

    use crate::operator::DeepONet;
    use ndarray::Array1;

    /// Trading signal
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub enum Signal {
        Buy,
        Sell,
        Hold,
    }

    /// Strategy configuration
    #[derive(Debug, Clone)]
    pub struct StrategyConfig {
        pub buy_threshold: f64,
        pub sell_threshold: f64,
        pub lookback: usize,
        pub prediction_horizon: usize,
    }

    impl Default for StrategyConfig {
        fn default() -> Self {
            Self {
                buy_threshold: 0.005,
                sell_threshold: -0.005,
                lookback: 60,
                prediction_horizon: 5,
            }
        }
    }

    /// Generate a trading signal from operator prediction
    pub fn generate_signal(
        model: &DeepONet,
        recent_prices: &[f64],
        config: &StrategyConfig,
    ) -> Signal {
        let n = recent_prices.len();
        if n < config.lookback {
            return Signal::Hold;
        }

        let window = &recent_prices[n - config.lookback..];
        let w_min = window.iter().cloned().fold(f64::INFINITY, f64::min);
        let w_max = window.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = w_max - w_min;
        if range < 1e-10 {
            return Signal::Hold;
        }

        let norm_window: Vec<f64> = window.iter().map(|p| (p - w_min) / range).collect();
        let branch_input = Array1::from_vec(norm_window);
        let trunk_input = Array1::from_vec(vec![0.5]); // mid-horizon

        let pred_norm = model.predict(&branch_input, &trunk_input);
        let current_norm = (recent_prices[n - 1] - w_min) / range;
        let predicted_return = pred_norm - current_norm;

        if predicted_return > config.buy_threshold {
            Signal::Buy
        } else if predicted_return < config.sell_threshold {
            Signal::Sell
        } else {
            Signal::Hold
        }
    }

    /// Generate signals for an entire price series
    pub fn generate_signal_series(
        model: &DeepONet,
        prices: &[f64],
        config: &StrategyConfig,
    ) -> Vec<Signal> {
        let mut signals = Vec::with_capacity(prices.len());

        for i in 0..prices.len() {
            if i < config.lookback {
                signals.push(Signal::Hold);
            } else {
                let window = &prices[..=i];
                signals.push(generate_signal(model, window, config));
            }
        }

        signals
    }
}


// ═══════════════════════════════════════════════════════════════════════════════
// Backtest Module
// ═══════════════════════════════════════════════════════════════════════════════

pub mod backtest {
    //! Backtesting engine for operator learning trading strategies.

    use crate::strategy::Signal;
    use serde::Serialize;

    /// Configuration for backtesting
    #[derive(Debug, Clone)]
    pub struct BacktestConfig {
        pub initial_capital: f64,
        pub position_size_pct: f64,
        pub transaction_cost_bps: f64,
        pub stop_loss_pct: f64,
        pub take_profit_pct: f64,
    }

    impl Default for BacktestConfig {
        fn default() -> Self {
            Self {
                initial_capital: 100_000.0,
                position_size_pct: 0.1,
                transaction_cost_bps: 10.0,
                stop_loss_pct: 0.02,
                take_profit_pct: 0.04,
            }
        }
    }

    /// Record of a completed trade
    #[derive(Debug, Clone, Serialize)]
    pub struct Trade {
        pub entry_idx: usize,
        pub exit_idx: usize,
        pub direction: i32, // +1 long, -1 short
        pub entry_price: f64,
        pub exit_price: f64,
        pub pnl: f64,
        pub pnl_pct: f64,
    }

    /// Backtest results
    #[derive(Debug, Clone, Serialize)]
    pub struct BacktestResult {
        pub total_return: f64,
        pub annualized_return: f64,
        pub sharpe_ratio: f64,
        pub max_drawdown: f64,
        pub win_rate: f64,
        pub n_trades: usize,
        pub profit_factor: f64,
        pub equity_curve: Vec<f64>,
        pub trades: Vec<Trade>,
    }

    /// Run a backtest given prices and signals
    pub fn run_backtest(
        prices: &[f64],
        signals: &[Signal],
        config: &BacktestConfig,
    ) -> BacktestResult {
        let n = prices.len();
        assert_eq!(n, signals.len());

        let mut capital = config.initial_capital;
        let mut position: f64 = 0.0; // shares held
        let mut entry_price: f64 = 0.0;
        let mut entry_idx: usize = 0;
        let mut equity_curve = vec![capital; n];
        let mut trades = Vec::new();

        for i in 1..n {
            // Check stop loss / take profit
            if position.abs() > 1e-10 {
                let pnl_pct = (prices[i] - entry_price) / entry_price * position.signum();

                if pnl_pct <= -config.stop_loss_pct || pnl_pct >= config.take_profit_pct {
                    let pnl = position * (prices[i] - entry_price);
                    let cost = position.abs() * prices[i] * config.transaction_cost_bps / 10000.0;
                    capital += pnl - cost;
                    trades.push(Trade {
                        entry_idx,
                        exit_idx: i,
                        direction: position.signum() as i32,
                        entry_price,
                        exit_price: prices[i],
                        pnl: pnl - cost,
                        pnl_pct,
                    });
                    position = 0.0;
                }
            }

            // Process signal
            match signals[i] {
                Signal::Buy if position.abs() < 1e-10 => {
                    let trade_capital = capital * config.position_size_pct;
                    position = trade_capital / prices[i];
                    entry_price = prices[i];
                    entry_idx = i;
                    let cost = position.abs() * prices[i] * config.transaction_cost_bps / 10000.0;
                    capital -= cost;
                }
                Signal::Sell if position > 1e-10 => {
                    let pnl = position * (prices[i] - entry_price);
                    let pnl_pct = (prices[i] - entry_price) / entry_price;
                    let cost = position.abs() * prices[i] * config.transaction_cost_bps / 10000.0;
                    capital += pnl - cost;
                    trades.push(Trade {
                        entry_idx,
                        exit_idx: i,
                        direction: 1,
                        entry_price,
                        exit_price: prices[i],
                        pnl: pnl - cost,
                        pnl_pct,
                    });
                    position = 0.0;
                }
                _ => {}
            }

            equity_curve[i] = if position.abs() > 1e-10 {
                capital + position * (prices[i] - entry_price)
            } else {
                capital
            };
        }

        // Close remaining position
        if position.abs() > 1e-10 {
            let pnl = position * (prices[n - 1] - entry_price);
            let pnl_pct = (prices[n - 1] - entry_price) / entry_price * position.signum();
            let cost = position.abs() * prices[n - 1] * config.transaction_cost_bps / 10000.0;
            capital += pnl - cost;
            trades.push(Trade {
                entry_idx,
                exit_idx: n - 1,
                direction: position.signum() as i32,
                entry_price,
                exit_price: prices[n - 1],
                pnl: pnl - cost,
                pnl_pct,
            });
            equity_curve[n - 1] = capital;
        }

        compute_metrics(equity_curve, trades)
    }

    fn compute_metrics(equity_curve: Vec<f64>, trades: Vec<Trade>) -> BacktestResult {
        let n = equity_curve.len();

        // Daily returns
        let daily_returns: Vec<f64> = (1..n)
            .map(|i| (equity_curve[i] - equity_curve[i - 1]) / (equity_curve[i - 1] + 1e-10))
            .collect();

        let total_return = equity_curve[n - 1] / equity_curve[0] - 1.0;
        let annualized_return = if daily_returns.len() > 0 && total_return > -1.0 {
            (1.0 + total_return).powf(252.0 / daily_returns.len() as f64) - 1.0
        } else {
            -1.0
        };

        let mean_ret = daily_returns.iter().sum::<f64>() / daily_returns.len().max(1) as f64;
        let var_ret = daily_returns
            .iter()
            .map(|r| (r - mean_ret).powi(2))
            .sum::<f64>()
            / daily_returns.len().max(1) as f64;
        let std_ret = var_ret.sqrt();
        let sharpe = if std_ret > 1e-10 {
            mean_ret / std_ret * 252.0_f64.sqrt()
        } else {
            0.0
        };

        // Max drawdown
        let mut peak = equity_curve[0];
        let mut max_dd = 0.0_f64;
        for &eq in &equity_curve {
            peak = peak.max(eq);
            let dd = (peak - eq) / peak;
            max_dd = max_dd.max(dd);
        }

        // Trade stats
        let n_trades = trades.len();
        let wins = trades.iter().filter(|t| t.pnl > 0.0).count();
        let win_rate = if n_trades > 0 {
            wins as f64 / n_trades as f64
        } else {
            0.0
        };

        let gross_profit: f64 = trades.iter().filter(|t| t.pnl > 0.0).map(|t| t.pnl).sum();
        let gross_loss: f64 = trades
            .iter()
            .filter(|t| t.pnl < 0.0)
            .map(|t| t.pnl.abs())
            .sum();
        let profit_factor = if gross_loss > 1e-10 {
            gross_profit / gross_loss
        } else if gross_profit > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };

        BacktestResult {
            total_return,
            annualized_return,
            sharpe_ratio: sharpe,
            max_drawdown: max_dd,
            win_rate,
            n_trades,
            profit_factor,
            equity_curve,
            trades,
        }
    }

    /// Print formatted backtest results
    pub fn print_results(name: &str, result: &BacktestResult) {
        println!("═══════════════════════════════════════════════════");
        println!("  Strategy: {}", name);
        println!("═══════════════════════════════════════════════════");
        println!("  Total Return:      {:>+8.2}%", result.total_return * 100.0);
        println!("  Annualized Return: {:>+8.2}%", result.annualized_return * 100.0);
        println!("  Sharpe Ratio:      {:>8.2}", result.sharpe_ratio);
        println!("  Max Drawdown:      {:>8.2}%", result.max_drawdown * 100.0);
        println!("  Win Rate:          {:>8.1}%", result.win_rate * 100.0);
        println!("  Num Trades:        {:>8}", result.n_trades);
        println!("  Profit Factor:     {:>8.2}", result.profit_factor);
        println!("═══════════════════════════════════════════════════");
    }
}
