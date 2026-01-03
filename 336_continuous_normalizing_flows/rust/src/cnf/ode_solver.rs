//! # ODE Solver
//!
//! Numerical ODE solvers for continuous normalizing flows.
//!
//! Supports:
//! - Euler method (fast, less accurate)
//! - RK4 (balanced accuracy and speed)
//! - Hutchinson trace estimator for log-det computation

use ndarray::Array1;
use rand::Rng;
use rand_distr::{Distribution, StandardNormal};

use super::VelocityField;

/// ODE integration method
#[derive(Debug, Clone, Copy)]
pub enum ODEMethod {
    /// Euler method (first-order)
    Euler,
    /// 4th-order Runge-Kutta
    RK4,
}

/// ODE Solver for continuous normalizing flows
#[derive(Debug, Clone)]
pub struct ODESolver {
    /// Integration method
    pub method: ODEMethod,
    /// Number of integration steps
    pub num_steps: usize,
    /// Number of samples for Hutchinson trace estimator
    pub num_trace_samples: usize,
}

impl Default for ODESolver {
    fn default() -> Self {
        Self {
            method: ODEMethod::RK4,
            num_steps: 50,
            num_trace_samples: 1,
        }
    }
}

impl ODESolver {
    /// Create a new ODE solver
    pub fn new(method: ODEMethod, num_steps: usize) -> Self {
        Self {
            method,
            num_steps,
            num_trace_samples: 1,
        }
    }

    /// Create with custom trace samples
    pub fn with_trace_samples(mut self, n: usize) -> Self {
        self.num_trace_samples = n;
        self
    }

    /// Solve ODE from t0 to t1
    ///
    /// Returns the final state z(t1)
    pub fn solve(
        &self,
        velocity_field: &VelocityField,
        z0: &Array1<f64>,
        t_span: (f64, f64),
    ) -> Array1<f64> {
        match self.method {
            ODEMethod::Euler => self.euler(velocity_field, z0, t_span),
            ODEMethod::RK4 => self.rk4(velocity_field, z0, t_span),
        }
    }

    /// Solve ODE and compute trace integral for log-det Jacobian
    ///
    /// Returns (z(t1), trace_integral) where:
    /// trace_integral = ∫ tr(∂f/∂z) dt
    pub fn solve_with_trace(
        &self,
        velocity_field: &VelocityField,
        z0: &Array1<f64>,
        t_span: (f64, f64),
    ) -> (Array1<f64>, f64) {
        match self.method {
            ODEMethod::Euler => self.euler_with_trace(velocity_field, z0, t_span),
            ODEMethod::RK4 => self.rk4_with_trace(velocity_field, z0, t_span),
        }
    }

    /// Euler method
    fn euler(
        &self,
        velocity_field: &VelocityField,
        z0: &Array1<f64>,
        t_span: (f64, f64),
    ) -> Array1<f64> {
        let (t0, t1) = t_span;
        let dt = (t1 - t0) / self.num_steps as f64;
        let mut z = z0.clone();
        let mut t = t0;

        for _ in 0..self.num_steps {
            let dz = velocity_field.forward(&z, t);
            z = &z + &(dz * dt);
            t += dt;
        }

        z
    }

    /// RK4 method
    fn rk4(
        &self,
        velocity_field: &VelocityField,
        z0: &Array1<f64>,
        t_span: (f64, f64),
    ) -> Array1<f64> {
        let (t0, t1) = t_span;
        let dt = (t1 - t0) / self.num_steps as f64;
        let mut z = z0.clone();
        let mut t = t0;

        for _ in 0..self.num_steps {
            let k1 = velocity_field.forward(&z, t);
            let k2 = velocity_field.forward(&(&z + &(&k1 * (0.5 * dt))), t + 0.5 * dt);
            let k3 = velocity_field.forward(&(&z + &(&k2 * (0.5 * dt))), t + 0.5 * dt);
            let k4 = velocity_field.forward(&(&z + &(&k3 * dt)), t + dt);

            z = &z + &((&k1 + &(&k2 * 2.0) + &(&k3 * 2.0) + &k4) * (dt / 6.0));
            t += dt;
        }

        z
    }

    /// Euler method with trace estimation
    fn euler_with_trace(
        &self,
        velocity_field: &VelocityField,
        z0: &Array1<f64>,
        t_span: (f64, f64),
    ) -> (Array1<f64>, f64) {
        let (t0, t1) = t_span;
        let dt = (t1 - t0) / self.num_steps as f64;
        let mut z = z0.clone();
        let mut t = t0;
        let mut trace_integral = 0.0;

        let mut rng = rand::thread_rng();

        for _ in 0..self.num_steps {
            // Compute trace using Hutchinson estimator
            let trace_est = self.hutchinson_trace(velocity_field, &z, t, &mut rng);

            // Euler step
            let dz = velocity_field.forward(&z, t);
            z = &z + &(dz * dt);

            trace_integral += dt * trace_est;
            t += dt;
        }

        (z, trace_integral)
    }

    /// RK4 method with trace estimation at midpoint
    fn rk4_with_trace(
        &self,
        velocity_field: &VelocityField,
        z0: &Array1<f64>,
        t_span: (f64, f64),
    ) -> (Array1<f64>, f64) {
        let (t0, t1) = t_span;
        let dt = (t1 - t0) / self.num_steps as f64;
        let mut z = z0.clone();
        let mut t = t0;
        let mut trace_integral = 0.0;

        let mut rng = rand::thread_rng();

        for _ in 0..self.num_steps {
            // Estimate trace at midpoint
            let z_mid = &z + &(velocity_field.forward(&z, t) * (0.5 * dt));
            let trace_est = self.hutchinson_trace(velocity_field, &z_mid, t + 0.5 * dt, &mut rng);

            // RK4 step
            let k1 = velocity_field.forward(&z, t);
            let k2 = velocity_field.forward(&(&z + &(&k1 * (0.5 * dt))), t + 0.5 * dt);
            let k3 = velocity_field.forward(&(&z + &(&k2 * (0.5 * dt))), t + 0.5 * dt);
            let k4 = velocity_field.forward(&(&z + &(&k3 * dt)), t + dt);

            z = &z + &((&k1 + &(&k2 * 2.0) + &(&k3 * 2.0) + &k4) * (dt / 6.0));

            trace_integral += dt * trace_est;
            t += dt;
        }

        (z, trace_integral)
    }

    /// Hutchinson trace estimator
    ///
    /// tr(J) = E_ε[ε^T J ε] where J = ∂f/∂z
    fn hutchinson_trace<R: Rng>(
        &self,
        velocity_field: &VelocityField,
        z: &Array1<f64>,
        t: f64,
        rng: &mut R,
    ) -> f64 {
        let dim = z.len();
        let mut total_trace = 0.0;

        for _ in 0..self.num_trace_samples {
            // Sample random vector (Rademacher or Gaussian)
            let epsilon: Array1<f64> = (0..dim)
                .map(|_| StandardNormal.sample(rng))
                .collect();

            // Compute v^T J v using VJP
            let (_dz, vjp) = velocity_field.forward_with_vjp(z, t, &epsilon);

            // tr(J) ≈ ε^T J ε = ε^T vjp
            let trace_est: f64 = epsilon.iter().zip(vjp.iter()).map(|(e, v)| e * v).sum();
            total_trace += trace_est;
        }

        total_trace / self.num_trace_samples as f64
    }
}

/// Result of ODE solving with trace
#[derive(Debug, Clone)]
pub struct ODESolution {
    /// Final state
    pub z_final: Array1<f64>,
    /// Trace integral (for log-det computation)
    pub trace_integral: f64,
}

impl ODESolution {
    /// Create new solution
    pub fn new(z_final: Array1<f64>, trace_integral: f64) -> Self {
        Self {
            z_final,
            trace_integral,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_euler_solve() {
        let vf = VelocityField::new(5, 32, 2);
        let solver = ODESolver::new(ODEMethod::Euler, 100);
        let z0 = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5]);

        let z1 = solver.solve(&vf, &z0, (0.0, 1.0));

        assert_eq!(z1.len(), 5);
        assert!(z1.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_rk4_solve() {
        let vf = VelocityField::new(5, 32, 2);
        let solver = ODESolver::new(ODEMethod::RK4, 50);
        let z0 = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5]);

        let z1 = solver.solve(&vf, &z0, (0.0, 1.0));

        assert_eq!(z1.len(), 5);
        assert!(z1.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_solve_with_trace() {
        let vf = VelocityField::new(5, 32, 2);
        let solver = ODESolver::new(ODEMethod::RK4, 50).with_trace_samples(5);
        let z0 = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5]);

        let (z1, trace) = solver.solve_with_trace(&vf, &z0, (0.0, 1.0));

        assert_eq!(z1.len(), 5);
        assert!(z1.iter().all(|&x| x.is_finite()));
        assert!(trace.is_finite());
    }

    #[test]
    fn test_backward_forward_consistency() {
        // Test that solving forward then backward returns to approximately the same point
        let vf = VelocityField::new(5, 32, 2);
        let solver = ODESolver::new(ODEMethod::RK4, 100);
        let z0 = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5]);

        // Forward: 0 -> 1
        let z1 = solver.solve(&vf, &z0, (0.0, 1.0));

        // Backward: 1 -> 0
        let z0_reconstructed = solver.solve(&vf, &z1, (1.0, 0.0));

        // Check reconstruction error
        let error: f64 = z0.iter()
            .zip(z0_reconstructed.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();

        // Should be reasonably small
        assert!(error < 1.0, "Reconstruction error too large: {}", error);
    }
}
