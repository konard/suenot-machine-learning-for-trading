//! Update functions for MPNN.
//!
//! Update functions combine aggregated messages with node state to produce new representations.

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

/// Types of update functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UpdateType {
    /// Simple addition: h' = h + m
    Additive,
    /// Concatenation with MLP: h' = MLP([h || m])
    Concatenate,
    /// GRU-style update
    GRU,
    /// Residual update: h' = h + MLP(m)
    Residual,
    /// Highway update with gating
    Highway,
}

/// Update function that combines node state with aggregated messages.
pub struct UpdateFunction {
    /// Type of update
    pub update_type: UpdateType,
    /// Weight matrices
    pub weights: UpdateWeights,
}

/// Weights for update functions.
pub struct UpdateWeights {
    /// Main transformation weights
    pub w: Option<Array2<f64>>,
    /// Reset gate weights (for GRU)
    pub w_r: Option<Array2<f64>>,
    /// Update gate weights (for GRU)
    pub w_z: Option<Array2<f64>>,
    /// Candidate weights (for GRU)
    pub w_h: Option<Array2<f64>>,
    /// Highway gate weights
    pub w_gate: Option<Array2<f64>>,
}

impl UpdateFunction {
    /// Create an additive update function.
    pub fn additive() -> Self {
        Self {
            update_type: UpdateType::Additive,
            weights: UpdateWeights {
                w: None,
                w_r: None,
                w_z: None,
                w_h: None,
                w_gate: None,
            },
        }
    }

    /// Create a concatenation update function.
    pub fn concatenate(hidden_dim: usize, output_dim: usize) -> Self {
        use rand_distr::{Distribution, Normal};
        let mut rng = rand::thread_rng();
        let scale = (2.0 / (2 * hidden_dim + output_dim) as f64).sqrt();
        let normal = Normal::new(0.0, scale).unwrap();

        Self {
            update_type: UpdateType::Concatenate,
            weights: UpdateWeights {
                w: Some(Array2::from_shape_fn((2 * hidden_dim, output_dim), |_| {
                    normal.sample(&mut rng)
                })),
                w_r: None,
                w_z: None,
                w_h: None,
                w_gate: None,
            },
        }
    }

    /// Create a GRU update function.
    pub fn gru(hidden_dim: usize) -> Self {
        use rand_distr::{Distribution, Normal};
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 0.1).unwrap();

        let input_dim = 2 * hidden_dim;

        Self {
            update_type: UpdateType::GRU,
            weights: UpdateWeights {
                w: None,
                w_r: Some(Array2::from_shape_fn((input_dim, hidden_dim), |_| {
                    normal.sample(&mut rng)
                })),
                w_z: Some(Array2::from_shape_fn((input_dim, hidden_dim), |_| {
                    normal.sample(&mut rng)
                })),
                w_h: Some(Array2::from_shape_fn((input_dim, hidden_dim), |_| {
                    normal.sample(&mut rng)
                })),
                w_gate: None,
            },
        }
    }

    /// Create a residual update function.
    pub fn residual(hidden_dim: usize) -> Self {
        use rand_distr::{Distribution, Normal};
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, (2.0 / hidden_dim as f64).sqrt()).unwrap();

        Self {
            update_type: UpdateType::Residual,
            weights: UpdateWeights {
                w: Some(Array2::from_shape_fn((hidden_dim, hidden_dim), |_| {
                    normal.sample(&mut rng)
                })),
                w_r: None,
                w_z: None,
                w_h: None,
                w_gate: None,
            },
        }
    }

    /// Create a highway update function.
    pub fn highway(hidden_dim: usize) -> Self {
        use rand_distr::{Distribution, Normal};
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 0.1).unwrap();

        Self {
            update_type: UpdateType::Highway,
            weights: UpdateWeights {
                w: Some(Array2::from_shape_fn((hidden_dim, hidden_dim), |_| {
                    normal.sample(&mut rng)
                })),
                w_r: None,
                w_z: None,
                w_h: None,
                w_gate: Some(Array2::from_shape_fn((2 * hidden_dim, hidden_dim), |_| {
                    normal.sample(&mut rng)
                })),
            },
        }
    }

    /// Update node state with aggregated message.
    pub fn update(&self, node_state: &Array1<f64>, message: &Array1<f64>) -> Array1<f64> {
        match self.update_type {
            UpdateType::Additive => self.additive_update(node_state, message),
            UpdateType::Concatenate => self.concatenate_update(node_state, message),
            UpdateType::GRU => self.gru_update(node_state, message),
            UpdateType::Residual => self.residual_update(node_state, message),
            UpdateType::Highway => self.highway_update(node_state, message),
        }
    }

    /// Simple additive update.
    fn additive_update(&self, node_state: &Array1<f64>, message: &Array1<f64>) -> Array1<f64> {
        let result = node_state + message;
        // Apply ReLU
        result.mapv(|x| x.max(0.0))
    }

    /// Concatenation-based update.
    fn concatenate_update(&self, node_state: &Array1<f64>, message: &Array1<f64>) -> Array1<f64> {
        if let Some(ref w) = self.weights.w {
            // Concatenate state and message
            let mut concat = Vec::with_capacity(node_state.len() + message.len());
            concat.extend(node_state.iter());
            concat.extend(message.iter());
            let concat = Array1::from_vec(concat);

            if concat.len() != w.nrows() {
                return node_state.clone();
            }

            // Transform and apply ReLU
            let result = concat.dot(w);
            result.mapv(|x| x.max(0.0))
        } else {
            node_state.clone()
        }
    }

    /// GRU-style update.
    fn gru_update(&self, node_state: &Array1<f64>, message: &Array1<f64>) -> Array1<f64> {
        let w_r = match &self.weights.w_r {
            Some(w) => w,
            None => return node_state.clone(),
        };
        let w_z = match &self.weights.w_z {
            Some(w) => w,
            None => return node_state.clone(),
        };
        let w_h = match &self.weights.w_h {
            Some(w) => w,
            None => return node_state.clone(),
        };

        // Concatenate state and message
        let mut concat = Vec::with_capacity(node_state.len() + message.len());
        concat.extend(node_state.iter());
        concat.extend(message.iter());
        let concat = Array1::from_vec(concat);

        if concat.len() != w_r.nrows() {
            return node_state.clone();
        }

        // Reset gate
        let r = concat.dot(w_r).mapv(|x| 1.0 / (1.0 + (-x).exp()));

        // Update gate
        let z = concat.dot(w_z).mapv(|x| 1.0 / (1.0 + (-x).exp()));

        // Candidate state (using reset gate on node_state part)
        let mut reset_concat = Vec::with_capacity(node_state.len() + message.len());
        for i in 0..node_state.len() {
            reset_concat.push(r[i.min(r.len() - 1)] * node_state[i]);
        }
        reset_concat.extend(message.iter());
        let reset_concat = Array1::from_vec(reset_concat);

        let h_candidate = if reset_concat.len() == w_h.nrows() {
            reset_concat.dot(w_h).mapv(|x| x.tanh())
        } else {
            message.mapv(|x| x.tanh())
        };

        // Final update: h' = (1-z) * h + z * h_candidate
        let mut result = Array1::zeros(node_state.len());
        for i in 0..node_state.len() {
            let zi = z[i.min(z.len() - 1)];
            let hi = h_candidate[i.min(h_candidate.len() - 1)];
            result[i] = (1.0 - zi) * node_state[i] + zi * hi;
        }

        result
    }

    /// Residual update.
    fn residual_update(&self, node_state: &Array1<f64>, message: &Array1<f64>) -> Array1<f64> {
        if let Some(ref w) = self.weights.w {
            if message.len() != w.nrows() {
                return node_state.clone();
            }

            // Transform message and add residual
            let transformed = message.dot(w).mapv(|x| x.max(0.0));

            if transformed.len() == node_state.len() {
                node_state + &transformed
            } else {
                node_state.clone()
            }
        } else {
            node_state + message
        }
    }

    /// Highway-style update with gating.
    fn highway_update(&self, node_state: &Array1<f64>, message: &Array1<f64>) -> Array1<f64> {
        let w = match &self.weights.w {
            Some(w) => w,
            None => return node_state.clone(),
        };
        let w_gate = match &self.weights.w_gate {
            Some(w) => w,
            None => return node_state.clone(),
        };

        // Compute transform gate
        let mut concat = Vec::with_capacity(node_state.len() + message.len());
        concat.extend(node_state.iter());
        concat.extend(message.iter());
        let concat = Array1::from_vec(concat);

        if concat.len() != w_gate.nrows() || message.len() != w.nrows() {
            return node_state.clone();
        }

        let gate = concat.dot(w_gate).mapv(|x| 1.0 / (1.0 + (-x).exp()));

        // Transform message
        let transformed = message.dot(w).mapv(|x| x.max(0.0));

        // Highway: h' = gate * transformed + (1-gate) * h
        let mut result = Array1::zeros(node_state.len());
        for i in 0..node_state.len() {
            let gi = gate[i.min(gate.len() - 1)];
            let ti = transformed[i.min(transformed.len() - 1)];
            result[i] = gi * ti + (1.0 - gi) * node_state[i];
        }

        result
    }
}

/// Trait for custom update functions.
pub trait Update {
    /// Update node state with aggregated message.
    fn update(&self, state: &Array1<f64>, message: &Array1<f64>) -> Array1<f64>;
}

/// Layer normalization for stabilizing training.
pub struct LayerNorm {
    /// Learned scale parameter
    pub gamma: Array1<f64>,
    /// Learned shift parameter
    pub beta: Array1<f64>,
    /// Small constant for numerical stability
    pub eps: f64,
}

impl LayerNorm {
    /// Create a new layer normalization.
    pub fn new(dim: usize) -> Self {
        Self {
            gamma: Array1::ones(dim),
            beta: Array1::zeros(dim),
            eps: 1e-6,
        }
    }

    /// Apply layer normalization.
    pub fn normalize(&self, x: &Array1<f64>) -> Array1<f64> {
        let mean: f64 = x.iter().sum::<f64>() / x.len() as f64;
        let var: f64 = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / x.len() as f64;
        let std = (var + self.eps).sqrt();

        let mut normalized = Array1::zeros(x.len());
        for i in 0..x.len() {
            normalized[i] = self.gamma[i] * (x[i] - mean) / std + self.beta[i];
        }

        normalized
    }
}

/// Dropout for regularization.
pub struct Dropout {
    /// Dropout probability
    pub p: f64,
    /// Whether in training mode
    pub training: bool,
}

impl Dropout {
    /// Create a new dropout layer.
    pub fn new(p: f64) -> Self {
        Self { p, training: true }
    }

    /// Apply dropout to input.
    pub fn forward(&self, x: &Array1<f64>) -> Array1<f64> {
        if !self.training || self.p <= 0.0 {
            return x.clone();
        }

        use rand::Rng;
        let mut rng = rand::thread_rng();
        let scale = 1.0 / (1.0 - self.p);

        x.mapv(|v| {
            if rng.gen::<f64>() < self.p {
                0.0
            } else {
                v * scale
            }
        })
    }

    /// Set training mode.
    pub fn train(&mut self) {
        self.training = true;
    }

    /// Set evaluation mode.
    pub fn eval(&mut self) {
        self.training = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_additive_update() {
        let update = UpdateFunction::additive();
        let state = array![1.0, 2.0, 3.0];
        let message = array![0.5, 0.5, 0.5];

        let result = update.update(&state, &message);
        assert_eq!(result, array![1.5, 2.5, 3.5]);
    }

    #[test]
    fn test_concatenate_update() {
        let update = UpdateFunction::concatenate(3, 3);
        let state = array![1.0, 2.0, 3.0];
        let message = array![0.5, 0.5, 0.5];

        let result = update.update(&state, &message);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_residual_update() {
        let update = UpdateFunction::residual(3);
        let state = array![1.0, 2.0, 3.0];
        let message = array![0.5, 0.5, 0.5];

        let result = update.update(&state, &message);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_layer_norm() {
        let ln = LayerNorm::new(3);
        let x = array![1.0, 2.0, 3.0];

        let result = ln.normalize(&x);
        assert_eq!(result.len(), 3);

        // Check mean is close to 0
        let mean: f64 = result.iter().sum::<f64>() / 3.0;
        assert!(mean.abs() < 1e-5);
    }

    #[test]
    fn test_dropout_eval() {
        let mut dropout = Dropout::new(0.5);
        dropout.eval();

        let x = array![1.0, 2.0, 3.0];
        let result = dropout.forward(&x);

        assert_eq!(result, x);
    }
}
