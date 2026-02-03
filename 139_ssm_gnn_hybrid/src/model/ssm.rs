//! Diagonal State Space Model implementation.
//!
//! Implements a simplified S4-style SSM with diagonal state matrix
//! for efficient sequential processing of per-asset time series.

use rand::Rng;
use rand_distr::StandardNormal;

/// Diagonal State Space Model.
///
/// Uses diagonal A matrix for O(N) computation per time step.
/// Initialized with HiPPO-inspired diagonal values for optimal
/// long-range dependency capture.
pub struct DiagonalSSM {
    pub d_input: usize,
    pub d_model: usize,
    pub d_state: usize,
    /// Diagonal of A matrix (d_state,)
    pub a_diag: Vec<f64>,
    /// B matrix (d_state x d_model)
    pub b: Vec<Vec<f64>>,
    /// C matrix (d_model x d_state)
    pub c: Vec<Vec<f64>>,
    /// D vector (d_model,)
    pub d: Vec<f64>,
    /// Input projection (d_input x d_model)
    pub w_in: Vec<Vec<f64>>,
    /// Log step size (d_model,)
    pub log_dt: Vec<f64>,
}

impl DiagonalSSM {
    /// Create a new DiagonalSSM with random initialization.
    pub fn new(d_input: usize, d_model: usize, d_state: usize) -> Self {
        let mut rng = rand::thread_rng();

        // HiPPO-inspired diagonal A: -1, -2, ..., -d_state
        let a_diag: Vec<f64> = (1..=d_state).map(|i| -(i as f64)).collect();

        // Random B, C initialization
        let b = (0..d_state)
            .map(|_| {
                (0..d_model)
                    .map(|_| rng.sample::<f64, _>(StandardNormal) * 0.01)
                    .collect()
            })
            .collect();

        let c = (0..d_model)
            .map(|_| {
                (0..d_state)
                    .map(|_| rng.sample::<f64, _>(StandardNormal) * 0.01)
                    .collect()
            })
            .collect();

        let d = vec![0.0; d_model];

        let w_in = (0..d_input)
            .map(|_| {
                (0..d_model)
                    .map(|_| rng.sample::<f64, _>(StandardNormal) * 0.01)
                    .collect()
            })
            .collect();

        let log_dt = vec![-4.6_f64; d_model]; // log(0.01)

        Self {
            d_input,
            d_model,
            d_state,
            a_diag,
            b,
            c,
            d,
            w_in,
            log_dt,
        }
    }

    /// Discretize continuous parameters using zero-order hold.
    fn discretize(&self) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let dt: Vec<f64> = self.log_dt.iter().map(|x| x.exp()).collect();

        // A_bar[s][m] = exp(dt[m] * a_diag[s])
        let a_bar: Vec<Vec<f64>> = (0..self.d_state)
            .map(|s| {
                (0..self.d_model)
                    .map(|m| (dt[m] * self.a_diag[s]).exp())
                    .collect()
            })
            .collect();

        // B_bar[s][m] = dt[m] * B[s][m]
        let b_bar: Vec<Vec<f64>> = (0..self.d_state)
            .map(|s| {
                (0..self.d_model)
                    .map(|m| dt[m] * self.b[s][m])
                    .collect()
            })
            .collect();

        (a_bar, b_bar)
    }

    /// Forward pass through SSM.
    ///
    /// # Arguments
    /// * `input` - Input of shape (seq_len, d_input)
    ///
    /// # Returns
    /// Output of shape (seq_len, d_model)
    pub fn forward(&self, input: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let seq_len = input.len();

        // Project input: u_proj = input @ W_in
        let u_proj: Vec<Vec<f64>> = input
            .iter()
            .map(|inp| {
                (0..self.d_model)
                    .map(|m| {
                        (0..self.d_input)
                            .map(|f| inp[f] * self.w_in[f][m])
                            .sum::<f64>()
                    })
                    .collect()
            })
            .collect();

        let (a_bar, b_bar) = self.discretize();

        // Initialize state
        let mut state = vec![vec![0.0_f64; self.d_model]; self.d_state];
        let mut outputs = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            // State update: x = A_bar * x + B_bar * u_t
            for s in 0..self.d_state {
                for m in 0..self.d_model {
                    state[s][m] = a_bar[s][m] * state[s][m] + b_bar[s][m] * u_proj[t][m];
                }
            }

            // Output: y = C @ x + D * u_t
            let y: Vec<f64> = (0..self.d_model)
                .map(|m| {
                    let cx: f64 = (0..self.d_state)
                        .map(|s| self.c[m][s] * state[s][m])
                        .sum();
                    cx + self.d[m] * u_proj[t][m]
                })
                .collect();

            outputs.push(y);
        }

        outputs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ssm_forward() {
        let ssm = DiagonalSSM::new(5, 8, 4);
        let input = vec![vec![0.1, 0.2, 0.3, 0.4, 0.5]; 10];
        let output = ssm.forward(&input);
        assert_eq!(output.len(), 10);
        assert_eq!(output[0].len(), 8);
    }

    #[test]
    fn test_ssm_output_varies() {
        let ssm = DiagonalSSM::new(3, 4, 2);
        let input = vec![vec![1.0, 0.0, 0.0]; 5];
        let output = ssm.forward(&input);
        // Output should change over time as state evolves
        assert_ne!(output[0], output[4]);
    }
}
