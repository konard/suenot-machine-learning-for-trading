//! # ODE Solvers
//!
//! Numerical integration methods for solving ordinary differential equations.
//!
//! ## Available Solvers
//!
//! - [`EulerSolver`]: Simple first-order method, fast but less accurate
//! - [`RK4Solver`]: Fourth-order Runge-Kutta, good balance of speed/accuracy
//! - [`Dopri5Solver`]: Adaptive step-size Dormand-Prince 5(4), recommended for Neural ODE

mod euler;
mod rk4;
mod dopri5;

pub use euler::EulerSolver;
pub use rk4::RK4Solver;
pub use dopri5::Dopri5Solver;

use ndarray::Array1;

/// Trait for ODE right-hand side function
///
/// Represents dz/dt = f(z, t)
pub trait ODEFunc: Send + Sync {
    /// Evaluate the ODE function at state z and time t
    fn evaluate(&self, z: &Array1<f64>, t: f64) -> Array1<f64>;

    /// Get the dimension of the state vector
    fn dim(&self) -> usize;
}

/// Trait for ODE solvers
pub trait ODESolver: Send + Sync {
    /// Solve the ODE from t0 to t1 with initial state z0
    ///
    /// # Arguments
    ///
    /// * `func` - The ODE function dz/dt = f(z, t)
    /// * `z0` - Initial state
    /// * `t_span` - (t0, t1) time interval
    /// * `n_steps` - Number of output points
    ///
    /// # Returns
    ///
    /// Tuple of (times, states) where states[i] is the state at times[i]
    fn solve(
        &self,
        func: &dyn ODEFunc,
        z0: Array1<f64>,
        t_span: (f64, f64),
        n_steps: usize,
    ) -> (Vec<f64>, Vec<Array1<f64>>);

    /// Get solver name
    fn name(&self) -> &'static str;
}

/// Simple wrapper for closure-based ODE functions
pub struct ClosureODE<F>
where
    F: Fn(&Array1<f64>, f64) -> Array1<f64> + Send + Sync,
{
    func: F,
    dim: usize,
}

impl<F> ClosureODE<F>
where
    F: Fn(&Array1<f64>, f64) -> Array1<f64> + Send + Sync,
{
    pub fn new(func: F, dim: usize) -> Self {
        Self { func, dim }
    }
}

impl<F> ODEFunc for ClosureODE<F>
where
    F: Fn(&Array1<f64>, f64) -> Array1<f64> + Send + Sync,
{
    fn evaluate(&self, z: &Array1<f64>, t: f64) -> Array1<f64> {
        (self.func)(z, t)
    }

    fn dim(&self) -> usize {
        self.dim
    }
}

/// ODE solver configuration
#[derive(Debug, Clone)]
pub struct ODEConfig {
    /// Relative tolerance for adaptive methods
    pub rtol: f64,
    /// Absolute tolerance for adaptive methods
    pub atol: f64,
    /// Maximum number of internal steps
    pub max_steps: usize,
    /// Minimum step size
    pub min_step: f64,
    /// Maximum step size
    pub max_step: f64,
}

impl Default for ODEConfig {
    fn default() -> Self {
        Self {
            rtol: 1e-4,
            atol: 1e-6,
            max_steps: 10000,
            min_step: 1e-10,
            max_step: 1.0,
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Test exponential decay: dz/dt = -z, solution: z(t) = z0 * exp(-t)
    fn exponential_decay(z: &Array1<f64>, _t: f64) -> Array1<f64> {
        -z.clone()
    }

    #[test]
    fn test_closure_ode() {
        let ode = ClosureODE::new(exponential_decay, 1);
        let z = Array1::from_vec(vec![1.0]);
        let result = ode.evaluate(&z, 0.0);
        assert!((result[0] - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_euler_solver() {
        let solver = EulerSolver::new(0.01);
        let ode = ClosureODE::new(exponential_decay, 1);
        let z0 = Array1::from_vec(vec![1.0]);

        let (times, states) = solver.solve(&ode, z0, (0.0, 1.0), 11);

        assert_eq!(times.len(), 11);
        assert_eq!(states.len(), 11);

        // Check final value: should be approximately exp(-1) ≈ 0.368
        let expected = (-1.0_f64).exp();
        let error = (states.last().unwrap()[0] - expected).abs();
        assert!(error < 0.05, "Euler error too large: {}", error);
    }

    #[test]
    fn test_rk4_solver() {
        let solver = RK4Solver::new(0.01);
        let ode = ClosureODE::new(exponential_decay, 1);
        let z0 = Array1::from_vec(vec![1.0]);

        let (times, states) = solver.solve(&ode, z0, (0.0, 1.0), 11);

        // RK4 should be more accurate than Euler
        let expected = (-1.0_f64).exp();
        let error = (states.last().unwrap()[0] - expected).abs();
        assert!(error < 0.001, "RK4 error too large: {}", error);
    }

    #[test]
    fn test_dopri5_solver() {
        let solver = Dopri5Solver::default();
        let ode = ClosureODE::new(exponential_decay, 1);
        let z0 = Array1::from_vec(vec![1.0]);

        let (times, states) = solver.solve(&ode, z0, (0.0, 1.0), 11);

        // Dopri5 should be very accurate
        let expected = (-1.0_f64).exp();
        let error = (states.last().unwrap()[0] - expected).abs();
        assert!(error < 1e-5, "Dopri5 error too large: {}", error);
    }
}
