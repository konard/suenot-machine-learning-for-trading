//! # Portfolio Neural ODE
//!
//! Neural ODE models for portfolio optimization.

use ndarray::Array1;
use serde::{Deserialize, Serialize};

use super::network::{MLP, Activation};
use crate::ode::{ODEFunc as ODEFuncTrait, ODESolver, Dopri5Solver};
use crate::data::Features;

/// Portfolio state at a point in time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioState {
    /// Asset weights (sum to 1)
    pub weights: Vec<f64>,
    /// Latent representation
    pub latent: Vec<f64>,
    /// Current time
    pub time: f64,
}

impl PortfolioState {
    /// Create a new portfolio state
    pub fn new(weights: Vec<f64>, latent: Vec<f64>, time: f64) -> Self {
        Self { weights, latent, time }
    }

    /// Create from equal weights
    pub fn equal_weights(n_assets: usize, hidden_dim: usize) -> Self {
        let weights = vec![1.0 / n_assets as f64; n_assets];
        let latent = vec![0.0; hidden_dim];
        Self { weights, latent, time: 0.0 }
    }

    /// Convert to ndarray for ODE solving
    pub fn to_array(&self) -> Array1<f64> {
        let mut arr = Vec::with_capacity(self.weights.len() + self.latent.len());
        arr.extend(&self.weights);
        arr.extend(&self.latent);
        Array1::from_vec(arr)
    }

    /// Create from ndarray
    pub fn from_array(arr: &Array1<f64>, n_assets: usize, time: f64) -> Self {
        let weights = arr.iter().take(n_assets).cloned().collect();
        let latent = arr.iter().skip(n_assets).cloned().collect();
        Self { weights, latent, time }
    }

    /// Normalize weights to sum to 1
    pub fn normalize(&mut self) {
        let sum: f64 = self.weights.iter().sum();
        if sum > 0.0 {
            for w in &mut self.weights {
                *w /= sum;
            }
        }
    }
}

/// Trait for ODE function (portfolio dynamics)
pub trait ODEFunc: Send + Sync {
    fn evaluate(&self, state: &Array1<f64>, t: f64) -> Array1<f64>;
    fn dim(&self) -> usize;
}

/// Portfolio dynamics: dw/dt = f(w, t, context)
#[derive(Debug, Clone)]
pub struct PortfolioDynamics {
    /// Number of assets
    n_assets: usize,
    /// Hidden dimension for latent state
    hidden_dim: usize,
    /// Neural network for dynamics
    dynamics_net: MLP,
    /// Transaction cost penalty
    cost_weight: f64,
    /// Context features (cached)
    context: Option<Vec<f64>>,
}

impl PortfolioDynamics {
    /// Create new portfolio dynamics
    pub fn new(n_assets: usize, hidden_dim: usize) -> Self {
        // Network: [hidden_dim + 1 (time)] -> hidden_dim
        let dynamics_net = MLP::new(
            &[hidden_dim + 1, hidden_dim * 2, hidden_dim * 2, hidden_dim],
            Activation::Tanh,
            Activation::Identity,
        );

        Self {
            n_assets,
            hidden_dim,
            dynamics_net,
            cost_weight: 0.01,
            context: None,
        }
    }

    /// Set context features for conditional dynamics
    pub fn set_context(&mut self, context: Vec<f64>) {
        self.context = Some(context);
    }

    /// Set transaction cost weight
    pub fn set_cost_weight(&mut self, weight: f64) {
        self.cost_weight = weight;
    }

    /// Get mutable reference to dynamics network
    pub fn dynamics_net_mut(&mut self) -> &mut MLP {
        &mut self.dynamics_net
    }
}

impl ODEFunc for PortfolioDynamics {
    fn evaluate(&self, state: &Array1<f64>, t: f64) -> Array1<f64> {
        // Extract latent state (skip weights)
        let latent: Array1<f64> = state.slice(ndarray::s![self.n_assets..]).to_owned();

        // Create input with time
        let mut input = Vec::with_capacity(self.hidden_dim + 1);
        input.extend(latent.iter());
        input.push(t);
        let input_arr = Array1::from_vec(input);

        // Compute latent drift
        let latent_drift = self.dynamics_net.forward(&input_arr);

        // Apply cost-aware scaling
        let drift_magnitude: f64 = latent_drift.iter().map(|x| x.abs()).sum();
        let cost_penalty = 1.0 / (1.0 + self.cost_weight * drift_magnitude);

        let scaled_drift = latent_drift * cost_penalty;

        // Weights drift is zero in latent space (weights decoded later)
        let mut full_drift = Array1::zeros(state.len());
        for i in 0..self.hidden_dim {
            full_drift[self.n_assets + i] = scaled_drift[i];
        }

        full_drift
    }

    fn dim(&self) -> usize {
        self.n_assets + self.hidden_dim
    }
}

// Implement the solver trait for PortfolioDynamics
impl crate::ode::ODEFunc for PortfolioDynamics {
    fn evaluate(&self, z: &Array1<f64>, t: f64) -> Array1<f64> {
        ODEFunc::evaluate(self, z, t)
    }

    fn dim(&self) -> usize {
        ODEFunc::dim(self)
    }
}

/// Full Neural ODE Portfolio model
#[derive(Debug, Clone)]
pub struct NeuralODEPortfolio {
    /// Number of assets
    pub n_assets: usize,
    /// Number of input features per asset
    pub n_features: usize,
    /// Hidden dimension
    pub hidden_dim: usize,
    /// Encoder: features -> latent
    encoder: MLP,
    /// ODE dynamics
    dynamics: PortfolioDynamics,
    /// Decoder: latent -> weights
    decoder: MLP,
}

impl NeuralODEPortfolio {
    /// Create a new Neural ODE Portfolio model
    pub fn new(n_assets: usize, n_features: usize, hidden_dim: usize) -> Self {
        // Encoder: [n_assets * n_features + n_assets] -> hidden_dim
        let encoder_input = n_assets * n_features + n_assets;
        let encoder = MLP::new(
            &[encoder_input, hidden_dim * 2, hidden_dim],
            Activation::SiLU,
            Activation::Identity,
        );

        // Dynamics
        let dynamics = PortfolioDynamics::new(n_assets, hidden_dim);

        // Decoder: hidden_dim -> n_assets (with softmax)
        let decoder = MLP::new(
            &[hidden_dim, hidden_dim, n_assets],
            Activation::ReLU,
            Activation::Softmax,
        );

        Self {
            n_assets,
            n_features,
            hidden_dim,
            encoder,
            dynamics,
            decoder,
        }
    }

    /// Encode initial state
    pub fn encode(&self, weights: &[f64], features: &Features) -> Array1<f64> {
        // Flatten features
        let mut input = Vec::with_capacity(self.n_assets * self.n_features + self.n_assets);
        input.extend(features.flatten());
        input.extend(weights);

        let input_arr = Array1::from_vec(input);
        self.encoder.forward(&input_arr)
    }

    /// Decode latent to weights
    pub fn decode(&self, latent: &Array1<f64>) -> Array1<f64> {
        self.decoder.forward(latent)
    }

    /// Solve ODE trajectory
    ///
    /// # Arguments
    ///
    /// * `initial_weights` - Starting portfolio weights
    /// * `features` - Market features
    /// * `t_span` - (t0, t1) time interval
    /// * `n_steps` - Number of output points
    ///
    /// # Returns
    ///
    /// Vector of (time, weights) pairs
    pub fn solve_trajectory(
        &self,
        initial_weights: &[f64],
        features: &Features,
        t_span: (f64, f64),
        n_steps: usize,
    ) -> Vec<(f64, Vec<f64>)> {
        // Encode initial state
        let z0 = self.encode(initial_weights, features);

        // Create full initial state [weights, latent]
        let mut state0 = Vec::with_capacity(self.n_assets + self.hidden_dim);
        state0.extend(initial_weights);
        state0.extend(z0.iter());
        let state0_arr = Array1::from_vec(state0);

        // Solve ODE
        let solver = Dopri5Solver::default();
        let (times, states) = solver.solve(&self.dynamics, state0_arr, t_span, n_steps);

        // Decode trajectory
        let mut trajectory = Vec::with_capacity(n_steps);
        for (t, state) in times.into_iter().zip(states.into_iter()) {
            let latent = state.slice(ndarray::s![self.n_assets..]).to_owned();
            let weights = self.decode(&latent);
            trajectory.push((t, weights.to_vec()));
        }

        trajectory
    }

    /// Get target weights at time horizon
    pub fn get_target_weights(
        &self,
        current_weights: &[f64],
        features: &Features,
        horizon: f64,
    ) -> Vec<f64> {
        let trajectory = self.solve_trajectory(
            current_weights,
            features,
            (0.0, horizon),
            2,  // Just need start and end
        );

        trajectory.last()
            .map(|(_, w)| w.clone())
            .unwrap_or_else(|| current_weights.to_vec())
    }

    /// Get total number of parameters
    pub fn num_params(&self) -> usize {
        self.encoder.num_params()
            + self.dynamics.dynamics_net.num_params()
            + self.decoder.num_params()
    }

    /// Get mutable reference to dynamics
    pub fn dynamics_mut(&mut self) -> &mut PortfolioDynamics {
        &mut self.dynamics
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_features(n_assets: usize, n_features: usize) -> Features {
        Features {
            n_assets,
            n_features,
            data: vec![vec![0.1; n_features]; n_assets],
            names: vec!["test".to_string(); n_features],
        }
    }

    #[test]
    fn test_portfolio_state() {
        let state = PortfolioState::equal_weights(3, 8);
        assert_eq!(state.weights.len(), 3);
        assert!((state.weights.iter().sum::<f64>() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_portfolio_dynamics() {
        let dynamics = PortfolioDynamics::new(3, 8);
        assert_eq!(dynamics.dim(), 11); // 3 assets + 8 hidden

        let state = Array1::from_vec(vec![0.33, 0.33, 0.34, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let drift = ODEFunc::evaluate(&dynamics, &state, 0.0);
        assert_eq!(drift.len(), 11);
    }

    #[test]
    fn test_neural_ode_portfolio() {
        let model = NeuralODEPortfolio::new(3, 10, 16);

        let features = create_test_features(3, 10);
        let initial_weights = vec![0.4, 0.35, 0.25];

        // Test encoding
        let latent = model.encode(&initial_weights, &features);
        assert_eq!(latent.len(), 16);

        // Test decoding
        let decoded = model.decode(&latent);
        assert_eq!(decoded.len(), 3);
        assert!((decoded.sum() - 1.0).abs() < 1e-6); // Softmax should sum to 1

        // Test trajectory
        let trajectory = model.solve_trajectory(
            &initial_weights,
            &features,
            (0.0, 1.0),
            11,
        );
        assert_eq!(trajectory.len(), 11);

        // All weights should sum to 1
        for (_, weights) in &trajectory {
            let sum: f64 = weights.iter().sum();
            assert!((sum - 1.0).abs() < 1e-5, "Weights sum: {}", sum);
        }
    }

    #[test]
    fn test_target_weights() {
        let model = NeuralODEPortfolio::new(2, 5, 8);
        let features = create_test_features(2, 5);
        let initial = vec![0.6, 0.4];

        let target = model.get_target_weights(&initial, &features, 0.5);
        assert_eq!(target.len(), 2);
        assert!((target.iter().sum::<f64>() - 1.0).abs() < 1e-5);
    }
}
