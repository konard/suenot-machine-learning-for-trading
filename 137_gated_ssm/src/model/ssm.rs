//! Diagonal State Space Model layer.
//!
//! Implements a diagonal SSM with O(N) per-step computation.
//! The state matrix A is diagonal, so each state dimension evolves independently.

use std::f64;

/// Diagonal SSM state for a single feature dimension.
///
/// The SSM computes:
///   x_k = A_bar * x_{k-1} + B_bar * u_k
///   y_k = C * x_k + D * u_k
pub struct DiagonalSSMState {
    /// Hidden state vector
    pub state: Vec<f64>,
    /// Discretized diagonal of A (exp(dt * a_i))
    pub a_bar: Vec<f64>,
    /// Discretized B (dt * b_i)
    pub b_bar: Vec<f64>,
    /// Output projection C
    pub c: Vec<f64>,
    /// Skip connection scalar D
    pub d: f64,
    /// State dimension
    pub d_state: usize,
}

impl DiagonalSSMState {
    /// Create a new diagonal SSM with HiPPO-like initialization.
    ///
    /// # Arguments
    /// * `d_state` - Dimension of the hidden state
    /// * `dt` - Discretization step size
    pub fn new(d_state: usize, dt: f64) -> Self {
        let mut a_bar = Vec::with_capacity(d_state);
        let mut b_bar = Vec::with_capacity(d_state);
        let mut c = Vec::with_capacity(d_state);

        for i in 0..d_state {
            // HiPPO-like initialization: A_ii = -(i+1)
            let a_val = -((i as f64) + 1.0);
            let a_disc = (dt * a_val).exp();
            a_bar.push(a_disc);
            b_bar.push(dt);
            // Alternating sign initialization for C
            c.push(if i % 2 == 0 { 1.0 } else { -1.0 } / (d_state as f64).sqrt());
        }

        DiagonalSSMState {
            state: vec![0.0; d_state],
            a_bar,
            b_bar,
            c,
            d: 1.0,
            d_state,
        }
    }

    /// Process a single input step through the SSM recurrence.
    ///
    /// Returns the scalar output y_k = C * x_k + D * u_k.
    pub fn step(&mut self, u: f64) -> f64 {
        let mut y = 0.0;
        for i in 0..self.d_state {
            self.state[i] = self.a_bar[i] * self.state[i] + self.b_bar[i] * u;
            y += self.c[i] * self.state[i];
        }
        y + self.d * u
    }

    /// Process an entire sequence through the SSM.
    ///
    /// Resets state before processing.
    pub fn forward(&mut self, input: &[f64]) -> Vec<f64> {
        self.reset();
        input.iter().map(|&u| self.step(u)).collect()
    }

    /// Reset the hidden state to zeros.
    pub fn reset(&mut self) {
        self.state.iter_mut().for_each(|s| *s = 0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ssm_basic() {
        let mut ssm = DiagonalSSMState::new(16, 0.01);
        let output = ssm.step(1.0);
        // Output should be finite
        assert!(output.is_finite());
    }

    #[test]
    fn test_ssm_sequence() {
        let mut ssm = DiagonalSSMState::new(16, 0.01);
        let input: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        let output = ssm.forward(&input);
        assert_eq!(output.len(), 100);
        assert!(output.iter().all(|y| y.is_finite()));
    }

    #[test]
    fn test_ssm_reset() {
        let mut ssm = DiagonalSSMState::new(16, 0.01);
        ssm.step(1.0);
        assert!(ssm.state.iter().any(|s| *s != 0.0));
        ssm.reset();
        assert!(ssm.state.iter().all(|s| *s == 0.0));
    }
}
