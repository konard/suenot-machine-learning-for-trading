//! # Runge-Kutta 4th Order Method
//!
//! Classic fourth-order explicit Runge-Kutta method.
//!
//! ## Algorithm
//!
//! ```text
//! k1 = f(z_n, t_n)
//! k2 = f(z_n + h/2 * k1, t_n + h/2)
//! k3 = f(z_n + h/2 * k2, t_n + h/2)
//! k4 = f(z_n + h * k3, t_n + h)
//! z_{n+1} = z_n + h/6 * (k1 + 2*k2 + 2*k3 + k4)
//! ```
//!
//! Error is O(h^5) per step, O(h^4) globally.

use ndarray::Array1;
use super::{ODEFunc, ODESolver};

/// Fourth-order Runge-Kutta ODE solver
#[derive(Debug, Clone)]
pub struct RK4Solver {
    /// Fixed step size
    step_size: f64,
}

impl RK4Solver {
    /// Create new RK4 solver with given step size
    pub fn new(step_size: f64) -> Self {
        Self { step_size }
    }
}

impl Default for RK4Solver {
    fn default() -> Self {
        Self::new(0.01)
    }
}

impl RK4Solver {
    /// Perform a single RK4 step
    fn step(
        &self,
        func: &dyn ODEFunc,
        z: &Array1<f64>,
        t: f64,
        h: f64,
    ) -> Array1<f64> {
        let k1 = func.evaluate(z, t);
        let k2 = func.evaluate(&(z + &(&k1 * (h / 2.0))), t + h / 2.0);
        let k3 = func.evaluate(&(z + &(&k2 * (h / 2.0))), t + h / 2.0);
        let k4 = func.evaluate(&(z + &(&k3 * h)), t + h);

        // z_{n+1} = z_n + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        z + &((&k1 + &(&k2 * 2.0) + &(&k3 * 2.0) + &k4) * (h / 6.0))
    }
}

impl ODESolver for RK4Solver {
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
            // Compute step size
            let h = self.step_size.min(next_output_time - t).min(t1 - t);

            // RK4 step
            z = self.step(func, &z, t, h);
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
        "RK4"
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
    fn test_harmonic_oscillator() {
        // dx/dt = v, dv/dt = -x (simple harmonic oscillator)
        // Solution: x(t) = cos(t), v(t) = -sin(t) for x(0)=1, v(0)=0
        let ode = ClosureODE::new(
            |z: &Array1<f64>, _t: f64| {
                Array1::from_vec(vec![z[1], -z[0]])
            },
            2,
        );

        let solver = RK4Solver::new(0.01);
        let z0 = Array1::from_vec(vec![1.0, 0.0]); // x=1, v=0

        let (times, states) = solver.solve(&ode, z0, (0.0, std::f64::consts::PI), 32);

        // At t=π, x should be approximately -1
        let final_x = states.last().unwrap()[0];
        let error = (final_x - (-1.0)).abs();
        assert!(error < 0.001, "Error: {}", error);
    }

    #[test]
    fn test_single_step() {
        // dz/dt = z => z(t) = z(0)*e^t
        let ode = ClosureODE::new(
            |z: &Array1<f64>, _t: f64| z.clone(),
            1,
        );

        let solver = RK4Solver::new(0.1);
        let z = Array1::from_vec(vec![1.0]);

        let z_new = solver.step(&ode, &z, 0.0, 0.1);

        // Should be approximately e^0.1 ≈ 1.1052
        let expected = 0.1_f64.exp();
        let error = (z_new[0] - expected).abs();
        assert!(error < 1e-6, "Error: {}", error);
    }
}
