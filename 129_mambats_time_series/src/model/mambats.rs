//! MambaTS model for time series forecasting.
//!
//! This module provides a simplified inference-only implementation of MambaTS.
//! For training, use the Python implementation with PyTorch.

use anyhow::{Context, Result};
use ndarray::{Array1, Array2, Axis};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MambaTSConfig {
    /// Number of input variables
    pub n_variables: usize,
    /// Model dimension
    pub d_model: usize,
    /// Number of Mamba layers
    pub n_layers: usize,
    /// Patch size for temporal patching
    pub patch_size: usize,
    /// Forecast horizon
    pub forecast_horizon: usize,
    /// State dimension for SSM
    pub d_state: usize,
}

impl Default for MambaTSConfig {
    fn default() -> Self {
        Self {
            n_variables: 10,
            d_model: 128,
            n_layers: 4,
            patch_size: 16,
            forecast_horizon: 24,
            d_state: 16,
        }
    }
}

/// Linear layer weights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearWeights {
    pub weight: Array2<f64>,
    pub bias: Option<Array1<f64>>,
}

impl LinearWeights {
    /// Create random weights for testing
    pub fn random(in_features: usize, out_features: usize, with_bias: bool) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let scale = (2.0 / in_features as f64).sqrt();
        let weight = Array2::from_shape_fn((out_features, in_features), |_| {
            rng.gen::<f64>() * scale - scale / 2.0
        });

        let bias = if with_bias {
            Some(Array1::zeros(out_features))
        } else {
            None
        };

        Self { weight, bias }
    }

    /// Forward pass
    pub fn forward(&self, x: &Array2<f64>) -> Array2<f64> {
        let mut result = x.dot(&self.weight.t());
        if let Some(ref bias) = self.bias {
            result = result + bias;
        }
        result
    }
}

/// Simplified Mamba block for inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MambaBlock {
    /// Input projection
    pub in_proj: LinearWeights,
    /// Output projection
    pub out_proj: LinearWeights,
    /// State matrix A (diagonal, stored as log)
    pub a_log: Array2<f64>,
    /// Skip connection D
    pub d: Array1<f64>,
    /// Model dimension
    pub d_model: usize,
    /// State dimension
    pub d_state: usize,
    /// Expansion factor
    pub expand: usize,
}

impl MambaBlock {
    /// Create a new Mamba block with random weights
    pub fn new(d_model: usize, d_state: usize, expand: usize) -> Self {
        let d_inner = d_model * expand;

        Self {
            in_proj: LinearWeights::random(d_model, d_inner * 2, false),
            out_proj: LinearWeights::random(d_inner, d_model, false),
            a_log: Array2::from_shape_fn((d_inner, d_state), |_| 0.5),
            d: Array1::ones(d_inner),
            d_model,
            d_state,
            expand,
        }
    }

    /// Forward pass (simplified)
    pub fn forward(&self, x: &Array2<f64>) -> Array2<f64> {
        let d_inner = self.d_model * self.expand;
        let seq_len = x.nrows();

        // Input projection
        let xz = self.in_proj.forward(x);

        // Split into x and z
        let x_part = xz.slice(ndarray::s![.., ..d_inner]).to_owned();
        let z_part = xz.slice(ndarray::s![.., d_inner..]).to_owned();

        // Simplified SSM (linear approximation for inference)
        // In production, use optimized scan implementation
        let mut h: Array1<f64> = Array1::zeros(d_inner);
        let mut y = Array2::zeros((seq_len, d_inner));

        let a = self.a_log.mapv(|x| -x.exp());

        for t in 0..seq_len {
            let x_t = x_part.row(t);

            // State update (simplified)
            for i in 0..d_inner {
                h[i] = h[i] * a[[i, 0]] + x_t[i];
            }

            // Output
            for i in 0..d_inner {
                y[[t, i]] = h[i] + x_t[i] * self.d[i];
            }
        }

        // Gating with SiLU
        let z_activated = z_part.mapv(|x| x * (1.0 / (1.0 + (-x).exp())));
        let y_gated = &y * &z_activated;

        // Output projection
        self.out_proj.forward(&y_gated)
    }
}

/// MambaTS model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MambaTS {
    /// Model configuration
    pub config: MambaTSConfig,
    /// Patch embedding projection
    pub patch_proj: LinearWeights,
    /// Mamba encoder layers
    pub layers: Vec<MambaBlock>,
    /// Prediction head
    pub pred_head: LinearWeights,
}

impl MambaTS {
    /// Create a new model with random weights (for testing)
    pub fn new(config: MambaTSConfig) -> Self {
        let patch_input_dim = config.patch_size * config.n_variables;

        let layers: Vec<MambaBlock> = (0..config.n_layers)
            .map(|_| MambaBlock::new(config.d_model, config.d_state, 2))
            .collect();

        Self {
            patch_proj: LinearWeights::random(patch_input_dim, config.d_model, true),
            layers,
            pred_head: LinearWeights::random(
                config.d_model,
                config.forecast_horizon * config.n_variables,
                true,
            ),
            config,
        }
    }

    /// Load model from file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let bytes = std::fs::read(path.as_ref())
            .context("Failed to read model file")?;

        // Try JSON format first
        if let Ok(model) = serde_json::from_slice(&bytes) {
            return Ok(model);
        }

        // Try bincode format
        bincode::deserialize(&bytes)
            .map_err(|e| anyhow::anyhow!("Failed to deserialize model: {}", e))
    }

    /// Save model to file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let bytes = bincode::serialize(self)
            .map_err(|e| anyhow::anyhow!("Failed to serialize model: {}", e))?;
        std::fs::write(path, bytes)?;
        Ok(())
    }

    /// Make prediction
    pub fn predict(&self, features: &Array2<f64>) -> Result<Array2<f64>> {
        let (seq_len, n_vars) = features.dim();

        if n_vars != self.config.n_variables {
            anyhow::bail!(
                "Expected {} variables, got {}",
                self.config.n_variables,
                n_vars
            );
        }

        // Create patches
        let n_patches = seq_len / self.config.patch_size;
        if n_patches == 0 {
            anyhow::bail!(
                "Sequence length {} is too short for patch size {}",
                seq_len,
                self.config.patch_size
            );
        }

        // Reshape into patches
        let truncated_len = n_patches * self.config.patch_size;
        let truncated = features.slice(ndarray::s![..truncated_len, ..]);

        let mut patches = Array2::zeros((n_patches, self.config.patch_size * n_vars));
        for i in 0..n_patches {
            let start = i * self.config.patch_size;
            let end = start + self.config.patch_size;
            let patch = truncated.slice(ndarray::s![start..end, ..]);
            for j in 0..self.config.patch_size {
                for k in 0..n_vars {
                    patches[[i, j * n_vars + k]] = patch[[j, k]];
                }
            }
        }

        // Patch embedding
        let mut x = self.patch_proj.forward(&patches);

        // Mamba encoder layers
        for layer in &self.layers {
            let residual = x.clone();
            x = layer.forward(&x);
            x = x + residual;
        }

        // Prediction head (use last patch)
        let last_patch = x.row(x.nrows() - 1).to_owned();
        let last_patch_2d = last_patch.insert_axis(Axis(0));
        let pred_flat = self.pred_head.forward(&last_patch_2d);

        // Reshape to (forecast_horizon, n_variables)
        let pred = Array2::from_shape_fn(
            (self.config.forecast_horizon, self.config.n_variables),
            |(i, j)| pred_flat[[0, i * self.config.n_variables + j]],
        );

        Ok(pred)
    }

    /// Predict single value (first variable, first horizon)
    pub fn predict_single(&self, features: &Array2<f64>) -> Result<f64> {
        let pred = self.predict(features)?;
        Ok(pred[[0, 0]])
    }
}

/// Backtesting results
#[derive(Debug, Clone, Default)]
pub struct BacktestResult {
    pub total_return: f64,
    pub sharpe_ratio: f64,
    pub max_drawdown: f64,
    pub n_trades: usize,
    pub win_rate: f64,
}

/// Simple backtester for MambaTS strategies
pub struct Backtester {
    pub model: MambaTS,
    pub threshold: f64,
    pub lookback: usize,
}

impl Backtester {
    /// Create a new backtester
    pub fn new(model: MambaTS, threshold: f64, lookback: usize) -> Self {
        Self {
            model,
            threshold,
            lookback,
        }
    }

    /// Run backtest on price data
    pub fn run(&self, prices: &[f64], features: &Array2<f64>) -> Result<BacktestResult> {
        let n = prices.len().min(features.nrows());

        if n < self.lookback + 1 {
            anyhow::bail!("Not enough data for backtest");
        }

        let mut equity = vec![1.0];
        let mut position = 0;
        let mut n_trades = 0;
        let mut wins = 0;

        for i in self.lookback..n - 1 {
            // Get features for prediction
            let input = features.slice(ndarray::s![i - self.lookback..i, ..]).to_owned();

            // Make prediction
            let pred = self.model.predict_single(&input).unwrap_or(0.0);

            // Generate signal
            let new_position = if pred > self.threshold {
                1
            } else if pred < -self.threshold {
                -1
            } else {
                0
            };

            // Calculate return
            let ret = (prices[i + 1] - prices[i]) / prices[i];
            let pnl = position as f64 * ret;

            // Update equity
            let last_equity = *equity.last().unwrap();
            equity.push(last_equity * (1.0 + pnl));

            // Track trades
            if new_position != position && new_position != 0 {
                n_trades += 1;
                if pnl > 0.0 {
                    wins += 1;
                }
            }

            position = new_position;
        }

        // Calculate metrics
        let total_return = equity.last().unwrap() / equity.first().unwrap() - 1.0;

        // Sharpe ratio (simplified)
        let returns: Vec<f64> = equity.windows(2).map(|w| w[1] / w[0] - 1.0).collect();
        let mean_ret: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
        let var: f64 = returns.iter().map(|&r| (r - mean_ret).powi(2)).sum::<f64>()
            / returns.len() as f64;
        let std = var.sqrt();
        let sharpe_ratio = if std > 0.0 {
            mean_ret / std * (252.0_f64).sqrt()
        } else {
            0.0
        };

        // Max drawdown
        let mut peak = equity[0];
        let mut max_drawdown = 0.0;
        for &e in &equity {
            if e > peak {
                peak = e;
            }
            let drawdown = (peak - e) / peak;
            if drawdown > max_drawdown {
                max_drawdown = drawdown;
            }
        }

        let win_rate = if n_trades > 0 {
            wins as f64 / n_trades as f64
        } else {
            0.0
        };

        Ok(BacktestResult {
            total_return,
            sharpe_ratio,
            max_drawdown,
            n_trades,
            win_rate,
        })
    }
}

// Add bincode support
mod bincode {
    use serde::{Deserialize, Serialize};

    pub fn serialize<T: Serialize>(value: &T) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        Ok(serde_json::to_vec(value)?)
    }

    pub fn deserialize<'a, T: Deserialize<'a>>(bytes: &'a [u8]) -> Result<T, Box<dyn std::error::Error>> {
        Ok(serde_json::from_slice(bytes)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_creation() {
        let config = MambaTSConfig::default();
        let model = MambaTS::new(config);
        assert_eq!(model.layers.len(), 4);
    }

    #[test]
    fn test_model_predict() {
        let config = MambaTSConfig {
            n_variables: 5,
            d_model: 32,
            n_layers: 2,
            patch_size: 4,
            forecast_horizon: 2,
            d_state: 8,
        };
        let model = MambaTS::new(config);

        let features = Array2::from_shape_fn((20, 5), |(i, j)| (i * j) as f64 / 100.0);
        let pred = model.predict(&features).unwrap();

        assert_eq!(pred.dim(), (2, 5));
    }

    #[test]
    fn test_linear_weights() {
        let linear = LinearWeights::random(10, 5, true);
        let x = Array2::from_shape_fn((3, 10), |(i, j)| (i + j) as f64);
        let y = linear.forward(&x);
        assert_eq!(y.dim(), (3, 5));
    }
}
