//! HiPPO (High-order Polynomial Projection Operators) implementation.
//!
//! Provides HiPPO-LegS (Scaled Legendre) and HiPPO-LagT (Translated Laguerre)
//! operators for optimal polynomial projection of continuous signals.

use crate::HiPPOError;

/// HiPPO-LegS (Scaled Legendre) operator.
///
/// Uses the measure μ(t) = 1/t · I_{[0,t]} which uniformly weights
/// the entire history up to the current time.
///
/// The state matrix A is lower-triangular:
///   A_{nk} = -(2n+1)^{1/2} (2k+1)^{1/2}  if n > k
///   A_{nk} = -(n+1)                         if n = k
///   A_{nk} = 0                              if n < k
///
/// B_n = (2n+1)^{1/2}
#[derive(Debug, Clone)]
pub struct HiPPOLegS {
    pub n: usize,
    pub a: Vec<Vec<f64>>,
    pub b: Vec<f64>,
}

impl HiPPOLegS {
    /// Create a new HiPPO-LegS operator with N polynomial coefficients.
    pub fn new(n: usize) -> Self {
        let (a, b) = Self::build_matrices(n);
        Self { n, a, b }
    }

    /// Build the continuous-time A and B matrices.
    fn build_matrices(n: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
        let mut a = vec![vec![0.0; n]; n];
        let mut b = vec![0.0; n];

        for i in 0..n {
            b[i] = ((2 * i + 1) as f64).sqrt();
            for k in 0..=i {
                if i > k {
                    a[i][k] = -((2 * i + 1) as f64).sqrt() * ((2 * k + 1) as f64).sqrt();
                } else {
                    // i == k
                    a[i][k] = -((i + 1) as f64);
                }
            }
        }

        (a, b)
    }

    /// Discretize using forward Euler method.
    ///
    /// Returns (A_d, B_d) where:
    ///   A_d = I + dt * A
    ///   B_d = dt * B
    pub fn discretize_euler(&self, dt: f64) -> (Vec<Vec<f64>>, Vec<f64>) {
        let n = self.n;
        let mut a_d = vec![vec![0.0; n]; n];
        let b_d: Vec<f64> = self.b.iter().map(|&bi| dt * bi).collect();

        for i in 0..n {
            for j in 0..n {
                a_d[i][j] = dt * self.a[i][j];
                if i == j {
                    a_d[i][j] += 1.0;
                }
            }
        }

        (a_d, b_d)
    }

    /// Discretize using the bilinear (Tustin) transform for better stability.
    ///
    /// Returns (A_bar, B_bar) where:
    ///   A_bar = (I - dt/2 * A)^{-1} (I + dt/2 * A)
    ///   B_bar = (I - dt/2 * A)^{-1} dt * B
    pub fn discretize_bilinear(&self, dt: f64) -> Result<(Vec<Vec<f64>>, Vec<f64>), HiPPOError> {
        let n = self.n;

        // Build (I - dt/2 * A) and (I + dt/2 * A)
        let mut lhs = vec![vec![0.0; n]; n]; // I - dt/2 * A
        let mut rhs = vec![vec![0.0; n]; n]; // I + dt/2 * A

        for i in 0..n {
            for j in 0..n {
                let half_dt_a = dt / 2.0 * self.a[i][j];
                lhs[i][j] = -half_dt_a;
                rhs[i][j] = half_dt_a;
                if i == j {
                    lhs[i][j] += 1.0;
                    rhs[i][j] += 1.0;
                }
            }
        }

        // Solve lhs * A_bar = rhs using Gaussian elimination
        let lhs_inv = invert_matrix(&lhs)
            .map_err(|e| HiPPOError::ComputationError(e))?;

        let a_bar = mat_mul(&lhs_inv, &rhs);
        let dt_b: Vec<f64> = self.b.iter().map(|&bi| dt * bi).collect();
        let b_bar = mat_vec_mul(&lhs_inv, &dt_b);

        Ok((a_bar, b_bar))
    }

    /// Process a sequence through HiPPO-LegS dynamics (forward Euler).
    ///
    /// Returns the coefficient history: Vec of state vectors, one per time step.
    pub fn process_sequence(&self, input: &[f64], dt: f64) -> Vec<Vec<f64>> {
        let (a_d, b_d) = self.discretize_euler(dt);
        let n = self.n;
        let mut state = vec![0.0; n];
        let mut history = Vec::with_capacity(input.len());

        for &f in input {
            let mut new_state = vec![0.0; n];
            for i in 0..n {
                let mut sum = 0.0;
                for j in 0..n {
                    sum += a_d[i][j] * state[j];
                }
                new_state[i] = sum + b_d[i] * f;
            }
            state = new_state;
            history.push(state.clone());
        }

        history
    }

    /// Process a sequence using bilinear discretization (more stable).
    pub fn process_sequence_bilinear(
        &self,
        input: &[f64],
        dt: f64,
    ) -> Result<Vec<Vec<f64>>, HiPPOError> {
        let (a_bar, b_bar) = self.discretize_bilinear(dt)?;
        let n = self.n;
        let mut state = vec![0.0; n];
        let mut history = Vec::with_capacity(input.len());

        for &f in input {
            let mut new_state = vec![0.0; n];
            for i in 0..n {
                let mut sum = 0.0;
                for j in 0..n {
                    sum += a_bar[i][j] * state[j];
                }
                new_state[i] = sum + b_bar[i] * f;
            }
            state = new_state;
            history.push(state.clone());
        }

        Ok(history)
    }
}

/// HiPPO-LagT (Translated Laguerre) operator.
///
/// Uses exponentially decaying measure μ(t) = e^{-(t-s)} · I_{[0,t]}.
///
/// A_{nk} = -1 if n >= k, 0 otherwise.
/// B_n = 1 for all n.
#[derive(Debug, Clone)]
pub struct HiPPOLagT {
    pub n: usize,
    pub a: Vec<Vec<f64>>,
    pub b: Vec<f64>,
}

impl HiPPOLagT {
    /// Create a new HiPPO-LagT operator with N polynomial coefficients.
    pub fn new(n: usize) -> Self {
        let (a, b) = Self::build_matrices(n);
        Self { n, a, b }
    }

    fn build_matrices(n: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
        let mut a = vec![vec![0.0; n]; n];
        let b = vec![1.0; n];

        for i in 0..n {
            for k in 0..=i {
                a[i][k] = -1.0;
            }
        }

        (a, b)
    }

    /// Process a sequence through HiPPO-LagT dynamics (forward Euler).
    pub fn process_sequence(&self, input: &[f64], dt: f64) -> Vec<Vec<f64>> {
        let n = self.n;

        // Forward Euler: A_d = I + dt*A, B_d = dt*B
        let mut a_d = vec![vec![0.0; n]; n];
        let b_d: Vec<f64> = self.b.iter().map(|&bi| dt * bi).collect();

        for i in 0..n {
            for j in 0..n {
                a_d[i][j] = dt * self.a[i][j];
                if i == j {
                    a_d[i][j] += 1.0;
                }
            }
        }

        let mut state = vec![0.0; n];
        let mut history = Vec::with_capacity(input.len());

        for &f in input {
            let mut new_state = vec![0.0; n];
            for i in 0..n {
                let mut sum = 0.0;
                for j in 0..n {
                    sum += a_d[i][j] * state[j];
                }
                new_state[i] = sum + b_d[i] * f;
            }
            state = new_state;
            history.push(state.clone());
        }

        history
    }
}

// --- Linear algebra helpers ---

/// Multiply two square matrices.
fn mat_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let mut c = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..n {
                sum += a[i][k] * b[k][j];
            }
            c[i][j] = sum;
        }
    }
    c
}

/// Multiply a matrix by a vector.
fn mat_vec_mul(a: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    let n = a.len();
    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..n {
            sum += a[i][j] * x[j];
        }
        y[i] = sum;
    }
    y
}

/// Invert a square matrix using Gauss-Jordan elimination.
fn invert_matrix(mat: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
    let n = mat.len();
    // Augmented matrix [mat | I]
    let mut aug = vec![vec![0.0; 2 * n]; n];
    for i in 0..n {
        for j in 0..n {
            aug[i][j] = mat[i][j];
        }
        aug[i][n + i] = 1.0;
    }

    for col in 0..n {
        // Find pivot
        let mut max_row = col;
        let mut max_val = aug[col][col].abs();
        for row in (col + 1)..n {
            if aug[row][col].abs() > max_val {
                max_val = aug[row][col].abs();
                max_row = row;
            }
        }

        if max_val < 1e-12 {
            return Err(format!("Singular matrix at column {col}"));
        }

        aug.swap(col, max_row);

        let pivot = aug[col][col];
        for j in 0..(2 * n) {
            aug[col][j] /= pivot;
        }

        for row in 0..n {
            if row != col {
                let factor = aug[row][col];
                for j in 0..(2 * n) {
                    aug[row][j] -= factor * aug[col][j];
                }
            }
        }
    }

    let mut inv = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            inv[i][j] = aug[i][n + j];
        }
    }

    Ok(inv)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_legs_matrix_construction() {
        let hippo = HiPPOLegS::new(4);
        assert_eq!(hippo.n, 4);
        // A[0][0] should be -(0+1) = -1
        assert!((hippo.a[0][0] - (-1.0)).abs() < 1e-10);
        // B[0] should be sqrt(1) = 1
        assert!((hippo.b[0] - 1.0).abs() < 1e-10);
        // B[1] should be sqrt(3)
        assert!((hippo.b[1] - 3.0_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_lagt_matrix_construction() {
        let hippo = HiPPOLagT::new(4);
        assert_eq!(hippo.n, 4);
        // All B values should be 1
        for &bi in &hippo.b {
            assert!((bi - 1.0).abs() < 1e-10);
        }
        // A[2][1] should be -1
        assert!((hippo.a[2][1] - (-1.0)).abs() < 1e-10);
        // A[0][1] should be 0
        assert!((hippo.a[0][1]).abs() < 1e-10);
    }

    #[test]
    fn test_legs_process_sequence() {
        let hippo = HiPPOLegS::new(8);
        let input: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        let history = hippo.process_sequence(&input, 0.1);
        assert_eq!(history.len(), 100);
        assert_eq!(history[0].len(), 8);
    }

    #[test]
    fn test_legs_bilinear_discretization() {
        let hippo = HiPPOLegS::new(4);
        let result = hippo.discretize_bilinear(0.1);
        assert!(result.is_ok());
        let (a_bar, b_bar) = result.unwrap();
        assert_eq!(a_bar.len(), 4);
        assert_eq!(b_bar.len(), 4);
    }

    #[test]
    fn test_lagt_process_sequence() {
        let hippo = HiPPOLagT::new(8);
        let input = vec![1.0; 50];
        let history = hippo.process_sequence(&input, 0.1);
        assert_eq!(history.len(), 50);
    }

    #[test]
    fn test_matrix_inversion() {
        let mat = vec![
            vec![2.0, 1.0],
            vec![1.0, 3.0],
        ];
        let inv = invert_matrix(&mat).unwrap();
        // Check that mat * inv ≈ I
        let product = mat_mul(&mat, &inv);
        assert!((product[0][0] - 1.0).abs() < 1e-10);
        assert!((product[1][1] - 1.0).abs() < 1e-10);
        assert!((product[0][1]).abs() < 1e-10);
        assert!((product[1][0]).abs() < 1e-10);
    }
}
