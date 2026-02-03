//! Single particle implementation for TimeParticle SSM.

use ndarray::{Array1, Array2, Axis};
use rand::prelude::*;
use rand_distr::Normal;

/// Single particle operating at a specific time scale.
///
/// Each particle captures temporal dynamics at its designated scale using
/// a gated state space model.
pub struct Particle {
    /// Scale factor (1, 2, 3, ...) determines temporal resolution
    pub scale_factor: usize,
    /// Hidden dimension
    pub d_hidden: usize,
    /// State dimension
    pub d_state: usize,
    /// Gate weights [d_state x (d_hidden + d_state)]
    pub gate_weights: Array2<f64>,
    /// Gate bias [d_state]
    pub gate_bias: Array1<f64>,
    /// State weights [d_state x d_hidden]
    pub state_weights: Array2<f64>,
    /// Recurrent weights [d_state x d_state]
    pub recurrent_weights: Array2<f64>,
    /// Output weights [d_hidden x d_state]
    pub output_weights: Array2<f64>,
}

impl Particle {
    /// Create a new particle with random initialization.
    pub fn new(d_hidden: usize, d_state: usize, scale_factor: usize) -> Self {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 0.1).unwrap();

        Self {
            scale_factor,
            d_hidden,
            d_state,
            gate_weights: Array2::from_shape_fn((d_state, d_hidden + d_state), |_| {
                normal.sample(&mut rng)
            }),
            gate_bias: Array1::zeros(d_state),
            state_weights: Array2::from_shape_fn((d_state, d_hidden), |_| {
                normal.sample(&mut rng)
            }),
            recurrent_weights: Array2::from_shape_fn((d_state, d_state), |_| {
                normal.sample(&mut rng) * 0.5
            }),
            output_weights: Array2::from_shape_fn((d_hidden, d_state), |_| {
                normal.sample(&mut rng)
            }),
        }
    }

    /// Forward pass through the particle.
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor [seq_len x d_hidden]
    /// * `prev_state` - Previous hidden state [d_state]
    ///
    /// # Returns
    ///
    /// * Output tensor [seq_len x d_hidden]
    /// * Final hidden state [d_state]
    pub fn forward(&self, x: &Array2<f64>, prev_state: &Array1<f64>) -> (Array2<f64>, Array1<f64>) {
        let seq_len = x.nrows();
        let mut outputs = Array2::zeros((seq_len, self.d_hidden));
        let mut state = prev_state.clone();

        for t in 0..seq_len {
            let x_t = x.row(t).to_owned();

            // Concatenate input and state for gate
            let gate_input = concatenate(&x_t, &state);

            // Compute gate
            let gate = sigmoid(&(self.gate_weights.dot(&gate_input) + &self.gate_bias));

            // Compute state candidate
            let state_candidate = tanh(&(self.state_weights.dot(&x_t) + self.recurrent_weights.dot(&state)));

            // Gated state update
            state = &gate * &state_candidate + &(1.0 - &gate) * &state;

            // Output projection
            let output = self.output_weights.dot(&state);
            outputs.row_mut(t).assign(&output);
        }

        (outputs, state)
    }

    /// Single step forward (for recurrent inference).
    pub fn step(&mut self, x: &Array1<f64>, state: &mut Array1<f64>) -> Array1<f64> {
        // Concatenate input and state for gate
        let gate_input = concatenate(x, state);

        // Compute gate
        let gate = sigmoid(&(self.gate_weights.dot(&gate_input) + &self.gate_bias));

        // Compute state candidate
        let state_candidate = tanh(&(self.state_weights.dot(x) + self.recurrent_weights.dot(&*state)));

        // Gated state update
        *state = &gate * &state_candidate + &(1.0 - &gate) * &*state;

        // Output projection
        self.output_weights.dot(state)
    }

    /// Get initial state.
    pub fn initial_state(&self) -> Array1<f64> {
        Array1::zeros(self.d_state)
    }
}

/// Concatenate two 1D arrays.
fn concatenate(a: &Array1<f64>, b: &Array1<f64>) -> Array1<f64> {
    let mut result = Array1::zeros(a.len() + b.len());
    result.slice_mut(ndarray::s![..a.len()]).assign(a);
    result.slice_mut(ndarray::s![a.len()..]).assign(b);
    result
}

/// Sigmoid activation function.
fn sigmoid(x: &Array1<f64>) -> Array1<f64> {
    x.mapv(|v| 1.0 / (1.0 + (-v).exp()))
}

/// Tanh activation function.
fn tanh(x: &Array1<f64>) -> Array1<f64> {
    x.mapv(|v| v.tanh())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_particle_creation() {
        let particle = Particle::new(64, 32, 1);
        assert_eq!(particle.d_hidden, 64);
        assert_eq!(particle.d_state, 32);
        assert_eq!(particle.scale_factor, 1);
    }

    #[test]
    fn test_particle_forward() {
        let particle = Particle::new(8, 4, 1);
        let x = Array2::zeros((10, 8));
        let state = particle.initial_state();

        let (output, final_state) = particle.forward(&x, &state);

        assert_eq!(output.shape(), &[10, 8]);
        assert_eq!(final_state.len(), 4);
    }

    #[test]
    fn test_particle_step() {
        let mut particle = Particle::new(8, 4, 1);
        let x = Array1::zeros(8);
        let mut state = particle.initial_state();

        let output = particle.step(&x, &mut state);

        assert_eq!(output.len(), 8);
    }

    #[test]
    fn test_sigmoid() {
        let x = Array1::from_vec(vec![0.0, 1.0, -1.0]);
        let y = sigmoid(&x);

        assert!((y[0] - 0.5).abs() < 1e-10);
        assert!(y[1] > 0.5);
        assert!(y[2] < 0.5);
    }

    #[test]
    fn test_tanh() {
        let x = Array1::from_vec(vec![0.0, 1.0, -1.0]);
        let y = tanh(&x);

        assert!((y[0]).abs() < 1e-10);
        assert!(y[1] > 0.0);
        assert!(y[2] < 0.0);
    }
}
