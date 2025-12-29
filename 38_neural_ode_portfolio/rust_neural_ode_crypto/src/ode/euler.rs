//! # Euler Method
//!
//! First-order explicit method for solving ODEs.
//!
//! ## Algorithm
//!
//! ```text
//! z_{n+1} = z_n + h * f(z_n, t_n)
//! ```
//!
//! Simple but can accumulate errors over long integrations.

use ndarray::Array1;
use super::{ODEFunc, ODESolver};

/// Euler method ODE solver
#[derive(Debug, Clone)]
pub struct EulerSolver {
    /// Fixed step size
    step_size: f64,
}

impl EulerSolver {
    /// Create new Euler solver with given step size
    pub fn new(step_size: f64) -> Self {
        Self { step_size }
    }

    /// Create with default step size of 0.01
    pub fn default() -> Self {
        Self::new(0.01)
    }
}

impl ODESolver for EulerSolver {
    fn solve(
        &self,
        func: &dyn ODEFunc,
        z0: Array1<f64>,
        t_span: (f64, f64),
        n_steps: usize,
    ) -> (Vec<f64>, Vec<Array1<f64>>) {
        let (t0, t1) = t_span;
        let dt_output = (t1 - t0) / (n_steps - 1) as f64;

        let mut times = Vec::with_capacity(n_steps);
        let mut states = Vec::with_capacity(n_steps);

        let mut t = t0;
        let mut z = z0;

        // Store initial state
        times.push(t);
        states.push(z.clone());

        let mut output_idx = 1;
        let mut next_output_time = t0 + dt_output;

        while output_idx < n_steps && t < t1 {
            // Compute step size (don't overshoot output time or t1)
            let h = self.step_size.min(next_output_time - t).min(t1 - t);

            // Euler step: z_{n+1} = z_n + h * f(z_n, t_n)
            let dz = func.evaluate(&z, t);
            z = &z + &(&dz * h);
            t += h;

            // Check if we've reached an output time
            if t >= next_output_time - 1e-10 {
                times.push(t);
                states.push(z.clone());
                output_idx += 1;
                next_output_time = t0 + output_idx as f64 * dt_output;
            }
        }

        (times, states)
    }

    fn name(&self) -> &'static str {
        "Euler"
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ode::ClosureODE;

    #[test]
    fn test_linear_ode() {
        // dz/dt = 1, z(0) = 0 => z(t) = t
        let ode = ClosureODE::new(
            |_z: &Array1<f64>, _t: f64| Array1::from_vec(vec![1.0]),
            1,
        );

        let solver = EulerSolver::new(0.1);
        let z0 = Array1::from_vec(vec![0.0]);

        let (times, states) = solver.solve(&ode, z0, (0.0, 1.0), 11);

        // z(1) should be approximately 1
        let error = (states.last().unwrap()[0] - 1.0).abs();
        assert!(error < 0.01);
    }

    #[test]
    fn test_step_size() {
        let solver = EulerSolver::new(0.05);
        assert!((solver.step_size - 0.05).abs() < 1e-10);
    }
}
