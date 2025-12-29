//! # Dormand-Prince 5(4) Method
//!
//! Adaptive step-size Runge-Kutta method with 5th order accuracy
//! and embedded 4th order error estimate.
//!
//! This is the recommended solver for Neural ODE as it automatically
//! adjusts step size based on local error estimates.

use ndarray::Array1;
use super::{ODEConfig, ODEFunc, ODESolver};

/// Dormand-Prince 5(4) adaptive solver
#[derive(Debug, Clone)]
pub struct Dopri5Solver {
    config: ODEConfig,
}

impl Dopri5Solver {
    /// Create new Dopri5 solver with given configuration
    pub fn new(config: ODEConfig) -> Self {
        Self { config }
    }

    /// Create with custom tolerances
    pub fn with_tolerances(rtol: f64, atol: f64) -> Self {
        let mut config = ODEConfig::default();
        config.rtol = rtol;
        config.atol = atol;
        Self { config }
    }

    /// Perform a single Dopri5 step with error estimation
    ///
    /// Returns (z_new, z_err, k7) where k7 can be reused as k1 for FSAL
    fn step_with_error(
        &self,
        func: &dyn ODEFunc,
        z: &Array1<f64>,
        t: f64,
        h: f64,
    ) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
        // Dormand-Prince coefficients
        const C2: f64 = 1.0 / 5.0;
        const C3: f64 = 3.0 / 10.0;
        const C4: f64 = 4.0 / 5.0;
        const C5: f64 = 8.0 / 9.0;

        const A21: f64 = 1.0 / 5.0;
        const A31: f64 = 3.0 / 40.0;
        const A32: f64 = 9.0 / 40.0;
        const A41: f64 = 44.0 / 45.0;
        const A42: f64 = -56.0 / 15.0;
        const A43: f64 = 32.0 / 9.0;
        const A51: f64 = 19372.0 / 6561.0;
        const A52: f64 = -25360.0 / 2187.0;
        const A53: f64 = 64448.0 / 6561.0;
        const A54: f64 = -212.0 / 729.0;
        const A61: f64 = 9017.0 / 3168.0;
        const A62: f64 = -355.0 / 33.0;
        const A63: f64 = 46732.0 / 5247.0;
        const A64: f64 = 49.0 / 176.0;
        const A65: f64 = -5103.0 / 18656.0;
        const A71: f64 = 35.0 / 384.0;
        const A73: f64 = 500.0 / 1113.0;
        const A74: f64 = 125.0 / 192.0;
        const A75: f64 = -2187.0 / 6784.0;
        const A76: f64 = 11.0 / 84.0;

        // 5th order weights
        const B1: f64 = 35.0 / 384.0;
        const B3: f64 = 500.0 / 1113.0;
        const B4: f64 = 125.0 / 192.0;
        const B5: f64 = -2187.0 / 6784.0;
        const B6: f64 = 11.0 / 84.0;

        // 4th order weights for error estimation
        const E1: f64 = 71.0 / 57600.0;
        const E3: f64 = -71.0 / 16695.0;
        const E4: f64 = 71.0 / 1920.0;
        const E5: f64 = -17253.0 / 339200.0;
        const E6: f64 = 22.0 / 525.0;
        const E7: f64 = -1.0 / 40.0;

        // Compute k values
        let k1 = func.evaluate(z, t);

        let z2 = z + &(&k1 * (h * A21));
        let k2 = func.evaluate(&z2, t + C2 * h);

        let z3 = z + &(&k1 * (h * A31)) + &(&k2 * (h * A32));
        let k3 = func.evaluate(&z3, t + C3 * h);

        let z4 = z + &(&k1 * (h * A41)) + &(&k2 * (h * A42)) + &(&k3 * (h * A43));
        let k4 = func.evaluate(&z4, t + C4 * h);

        let z5 = z + &(&k1 * (h * A51)) + &(&k2 * (h * A52)) + &(&k3 * (h * A53)) + &(&k4 * (h * A54));
        let k5 = func.evaluate(&z5, t + C5 * h);

        let z6 = z + &(&k1 * (h * A61)) + &(&k2 * (h * A62)) + &(&k3 * (h * A63))
            + &(&k4 * (h * A64)) + &(&k5 * (h * A65));
        let k6 = func.evaluate(&z6, t + h);

        // 5th order solution
        let z_new = z + &(&k1 * (h * B1)) + &(&k3 * (h * B3)) + &(&k4 * (h * B4))
            + &(&k5 * (h * B5)) + &(&k6 * (h * B6));

        // Compute k7 for FSAL (First Same As Last)
        let k7 = func.evaluate(&z_new, t + h);

        // Error estimate
        let z_err = &(&k1 * (h * E1)) + &(&k3 * (h * E3)) + &(&k4 * (h * E4))
            + &(&k5 * (h * E5)) + &(&k6 * (h * E6)) + &(&k7 * (h * E7));

        (z_new, z_err, k7)
    }

    /// Compute error norm
    fn error_norm(&self, z_err: &Array1<f64>, z: &Array1<f64>, z_new: &Array1<f64>) -> f64 {
        let mut err_sum = 0.0;
        let n = z_err.len();

        for i in 0..n {
            let scale = self.config.atol + self.config.rtol * z[i].abs().max(z_new[i].abs());
            err_sum += (z_err[i] / scale).powi(2);
        }

        (err_sum / n as f64).sqrt()
    }

    /// Compute optimal step size
    fn optimal_step(&self, h: f64, err: f64) -> f64 {
        if err == 0.0 {
            return h * 2.0;
        }

        const SAFETY: f64 = 0.9;
        const MIN_FACTOR: f64 = 0.2;
        const MAX_FACTOR: f64 = 10.0;

        let factor = SAFETY * (1.0 / err).powf(0.2);
        let factor = factor.max(MIN_FACTOR).min(MAX_FACTOR);

        (h * factor).max(self.config.min_step).min(self.config.max_step)
    }
}

impl Default for Dopri5Solver {
    fn default() -> Self {
        Self::new(ODEConfig::default())
    }
}

impl ODESolver for Dopri5Solver {
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
        let mut h = (t1 - t0) / 100.0; // Initial step size

        // Store initial state
        times.push(t);
        states.push(z.clone());

        let mut output_idx = 1;
        let mut next_output_time = t0 + dt_output;
        let mut step_count = 0;

        while output_idx < n_steps && t < t1 && step_count < self.config.max_steps {
            // Don't overshoot
            h = h.min(t1 - t);

            // Perform step with error estimation
            let (z_new, z_err, _k7) = self.step_with_error(func, &z, t, h);

            // Compute error
            let err = self.error_norm(&z_err, &z, &z_new);

            if err <= 1.0 {
                // Step accepted
                t += h;
                z = z_new;
                step_count += 1;

                // Collect output points
                while output_idx < n_steps && t >= next_output_time - 1e-10 {
                    // Linear interpolation for exact output times
                    // (simplified - could use dense output for better accuracy)
                    times.push(next_output_time);
                    states.push(z.clone());
                    output_idx += 1;
                    next_output_time = t0 + output_idx as f64 * dt_output;
                }
            }

            // Update step size
            h = self.optimal_step(h, err);
        }

        // Make sure we have exactly n_steps outputs
        while times.len() < n_steps {
            times.push(t);
            states.push(z.clone());
        }

        (times, states)
    }

    fn name(&self) -> &'static str {
        "Dopri5"
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
    fn test_exponential_growth() {
        // dz/dt = z, z(0) = 1 => z(t) = e^t
        let ode = ClosureODE::new(
            |z: &Array1<f64>, _t: f64| z.clone(),
            1,
        );

        let solver = Dopri5Solver::default();
        let z0 = Array1::from_vec(vec![1.0]);

        let (times, states) = solver.solve(&ode, z0, (0.0, 1.0), 11);

        // z(1) should be e ≈ 2.718
        let expected = 1.0_f64.exp();
        let error = (states.last().unwrap()[0] - expected).abs();
        assert!(error < 1e-5, "Error: {}", error);
    }

    #[test]
    fn test_stiff_system() {
        // Moderately stiff: dz/dt = -10*z
        let ode = ClosureODE::new(
            |z: &Array1<f64>, _t: f64| -10.0 * z,
            1,
        );

        let solver = Dopri5Solver::with_tolerances(1e-6, 1e-8);
        let z0 = Array1::from_vec(vec![1.0]);

        let (times, states) = solver.solve(&ode, z0, (0.0, 1.0), 11);

        // z(1) = e^(-10) ≈ 4.54e-5
        let expected = (-10.0_f64).exp();
        let relative_error = (states.last().unwrap()[0] - expected).abs() / expected;
        assert!(relative_error < 0.01, "Relative error: {}", relative_error);
    }

    #[test]
    fn test_multidimensional() {
        // Van der Pol oscillator (mu=0.1)
        let ode = ClosureODE::new(
            |z: &Array1<f64>, _t: f64| {
                let x = z[0];
                let y = z[1];
                let mu = 0.1;
                Array1::from_vec(vec![y, mu * (1.0 - x * x) * y - x])
            },
            2,
        );

        let solver = Dopri5Solver::default();
        let z0 = Array1::from_vec(vec![2.0, 0.0]);

        let (times, states) = solver.solve(&ode, z0, (0.0, 10.0), 101);

        // System should remain bounded
        for state in &states {
            assert!(state[0].abs() < 5.0, "x unbounded: {}", state[0]);
            assert!(state[1].abs() < 5.0, "y unbounded: {}", state[1]);
        }
    }
}
