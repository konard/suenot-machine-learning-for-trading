//! # Continuous Normalizing Flow
//!
//! Main CNF model combining velocity field and ODE solver.

use ndarray::{Array1, Array2};
use rand_distr::{Distribution, StandardNormal};
use std::f64::consts::PI;

use super::{VelocityField, ODESolver, ODEMethod};

/// Continuous Normalizing Flow model
///
/// Transforms data between a simple base distribution (standard normal)
/// and a complex target distribution using neural ODE dynamics.
#[derive(Debug, Clone)]
pub struct ContinuousNormalizingFlow {
    /// Velocity field network defining ODE dynamics
    pub velocity_field: VelocityField,
    /// ODE solver
    pub solver: ODESolver,
    /// Data dimension
    pub dim: usize,
    /// Time span for flow [t0, t1]
    pub t_span: (f64, f64),
}

impl ContinuousNormalizingFlow {
    /// Create a new CNF model
    ///
    /// # Arguments
    ///
    /// * `dim` - Data dimension
    /// * `hidden_dim` - Hidden layer dimension for velocity field
    /// * `num_layers` - Number of residual blocks in velocity field
    pub fn new(dim: usize, hidden_dim: usize, num_layers: usize) -> Self {
        let velocity_field = VelocityField::new(dim, hidden_dim, num_layers);
        let solver = ODESolver::new(ODEMethod::RK4, 50);

        Self {
            velocity_field,
            solver,
            dim,
            t_span: (0.0, 1.0),
        }
    }

    /// Create with custom solver settings
    pub fn with_solver(mut self, solver: ODESolver) -> Self {
        self.solver = solver;
        self
    }

    /// Create with custom time span
    pub fn with_t_span(mut self, t_span: (f64, f64)) -> Self {
        self.t_span = t_span;
        self
    }

    /// Encode data to latent space (x → z)
    ///
    /// Solves the ODE backward from t1 to t0.
    ///
    /// Returns (z, log_det) where:
    /// - z is the latent representation
    /// - log_det is the log determinant of the Jacobian
    pub fn encode(&self, x: &Array1<f64>) -> (Array1<f64>, f64) {
        // Solve ODE backward in time
        let t_span_backward = (self.t_span.1, self.t_span.0);
        let (z, neg_trace) = self.solver.solve_with_trace(
            &self.velocity_field,
            x,
            t_span_backward,
        );

        // log_det = -∫ tr(∂f/∂z) dt (negative because we solve backward)
        let log_det = -neg_trace;

        (z, log_det)
    }

    /// Encode batch of data points
    pub fn encode_batch(&self, x: &Array2<f64>) -> (Array2<f64>, Array1<f64>) {
        let batch_size = x.nrows();
        let mut z = Array2::zeros((batch_size, self.dim));
        let mut log_dets = Array1::zeros(batch_size);

        for i in 0..batch_size {
            let x_i = x.row(i).to_owned();
            let (z_i, log_det_i) = self.encode(&x_i);
            for j in 0..self.dim {
                z[[i, j]] = z_i[j];
            }
            log_dets[i] = log_det_i;
        }

        (z, log_dets)
    }

    /// Decode from latent space (z → x)
    ///
    /// Solves the ODE forward from t0 to t1.
    ///
    /// Returns (x, log_det)
    pub fn decode(&self, z: &Array1<f64>) -> (Array1<f64>, f64) {
        let (x, trace) = self.solver.solve_with_trace(
            &self.velocity_field,
            z,
            self.t_span,
        );

        let log_det = -trace;

        (x, log_det)
    }

    /// Decode batch of latent vectors
    pub fn decode_batch(&self, z: &Array2<f64>) -> (Array2<f64>, Array1<f64>) {
        let batch_size = z.nrows();
        let mut x = Array2::zeros((batch_size, self.dim));
        let mut log_dets = Array1::zeros(batch_size);

        for i in 0..batch_size {
            let z_i = z.row(i).to_owned();
            let (x_i, log_det_i) = self.decode(&z_i);
            for j in 0..self.dim {
                x[[i, j]] = x_i[j];
            }
            log_dets[i] = log_det_i;
        }

        (x, log_dets)
    }

    /// Compute log probability of data under the flow
    ///
    /// log p(x) = log p(z) + log |det(dz/dx)|
    ///          = log p(z) - ∫ tr(∂f/∂z) dt
    pub fn log_prob(&self, x: &Array1<f64>) -> f64 {
        let (z, log_det) = self.encode(x);
        let log_p_z = self.log_prob_base(&z);

        log_p_z + log_det
    }

    /// Compute log probabilities for batch
    pub fn log_prob_batch(&self, x: &Array2<f64>) -> Array1<f64> {
        let batch_size = x.nrows();
        let mut log_probs = Array1::zeros(batch_size);

        for i in 0..batch_size {
            let x_i = x.row(i).to_owned();
            log_probs[i] = self.log_prob(&x_i);
        }

        log_probs
    }

    /// Log probability under base distribution (standard normal)
    fn log_prob_base(&self, z: &Array1<f64>) -> f64 {
        let log_2pi = (2.0 * PI).ln();
        let n = self.dim as f64;
        let z_squared: f64 = z.iter().map(|&v| v * v).sum();

        -0.5 * (n * log_2pi + z_squared)
    }

    /// Generate samples from the learned distribution
    pub fn sample(&self, num_samples: usize) -> Array2<f64> {
        let mut rng = rand::thread_rng();
        let mut samples = Array2::zeros((num_samples, self.dim));

        for i in 0..num_samples {
            // Sample from base distribution
            let z: Array1<f64> = (0..self.dim)
                .map(|_| StandardNormal.sample(&mut rng))
                .collect();

            // Transform to data space
            let (x, _) = self.decode(&z);

            for j in 0..self.dim {
                samples[[i, j]] = x[j];
            }
        }

        samples
    }

    /// Sample conditional on partial observations
    ///
    /// Given observed dimensions and their values, sample from p(x_unobserved | x_observed)
    /// Uses simple rejection-like approach with perturbed latent codes.
    pub fn sample_conditional(
        &self,
        observed_dims: &[usize],
        observed_values: &Array1<f64>,
        num_samples: usize,
        max_attempts: usize,
    ) -> Array2<f64> {
        let mut rng = rand::thread_rng();
        let mut samples = Vec::new();
        let mut attempts = 0;

        while samples.len() < num_samples && attempts < max_attempts {
            // Sample from base
            let z: Array1<f64> = (0..self.dim)
                .map(|_| StandardNormal.sample(&mut rng))
                .collect();

            // Transform
            let (x, _) = self.decode(&z);

            // Check if observed dimensions match (with tolerance)
            let tolerance = 0.1;
            let matches = observed_dims.iter().zip(observed_values.iter()).all(|(&d, &v)| {
                (x[d] - v).abs() < tolerance
            });

            if matches {
                samples.push(x);
            }

            attempts += 1;
        }

        // If not enough samples, just return what we have with observed dims set
        while samples.len() < num_samples {
            let z: Array1<f64> = (0..self.dim)
                .map(|_| StandardNormal.sample(&mut rng))
                .collect();
            let (mut x, _) = self.decode(&z);

            // Force observed values
            for (&d, &v) in observed_dims.iter().zip(observed_values.iter()) {
                x[d] = v;
            }
            samples.push(x);
        }

        // Convert to array
        let mut result = Array2::zeros((num_samples, self.dim));
        for (i, sample) in samples.iter().take(num_samples).enumerate() {
            for j in 0..self.dim {
                result[[i, j]] = sample[j];
            }
        }

        result
    }

    /// Get parameters for serialization
    pub fn get_params(&self) -> Vec<f64> {
        self.velocity_field.get_params()
    }

    /// Set parameters from serialization
    pub fn set_params(&mut self, params: &[f64]) {
        self.velocity_field.set_params(params);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode() {
        let cnf = ContinuousNormalizingFlow::new(5, 32, 2);
        let x = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5]);

        let (z, log_det_encode) = cnf.encode(&x);
        let (x_reconstructed, log_det_decode) = cnf.decode(&z);

        assert_eq!(z.len(), 5);
        assert_eq!(x_reconstructed.len(), 5);
        assert!(log_det_encode.is_finite());
        assert!(log_det_decode.is_finite());
    }

    #[test]
    fn test_log_prob() {
        let cnf = ContinuousNormalizingFlow::new(5, 32, 2);
        let x = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5]);

        let log_prob = cnf.log_prob(&x);

        assert!(log_prob.is_finite());
        assert!(log_prob < 0.0); // Log probabilities should be negative
    }

    #[test]
    fn test_sample() {
        let cnf = ContinuousNormalizingFlow::new(5, 32, 2);

        let samples = cnf.sample(10);

        assert_eq!(samples.shape(), &[10, 5]);
        assert!(samples.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_batch_operations() {
        let cnf = ContinuousNormalizingFlow::new(5, 32, 2);
        let x = Array2::from_shape_fn((3, 5), |(i, j)| (i + j) as f64 * 0.1);

        let (z, log_dets_encode) = cnf.encode_batch(&x);
        let log_probs = cnf.log_prob_batch(&x);

        assert_eq!(z.shape(), &[3, 5]);
        assert_eq!(log_dets_encode.len(), 3);
        assert_eq!(log_probs.len(), 3);
    }
}
