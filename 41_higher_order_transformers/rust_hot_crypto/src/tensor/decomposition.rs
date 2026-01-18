//! Tensor decomposition algorithms
//!
//! Implements CP (CANDECOMP/PARAFAC) decomposition for approximating
//! third-order tensors as a sum of rank-1 components.

use ndarray::{Array1, Array2, Axis};
use rand::Rng;
use rand_distr::StandardNormal;

use super::Tensor3D;

/// Result of CP decomposition
///
/// A tensor T is approximated as:
/// T[i,j,k] ≈ Σ_r λ_r * A[i,r] * B[j,r] * C[k,r]
#[derive(Debug, Clone)]
pub struct CPResult {
    /// Factor matrix A (mode-0)
    pub factor_a: Array2<f64>,
    /// Factor matrix B (mode-1)
    pub factor_b: Array2<f64>,
    /// Factor matrix C (mode-2)
    pub factor_c: Array2<f64>,
    /// Weights (λ values)
    pub weights: Array1<f64>,
    /// Final reconstruction error
    pub final_error: f64,
    /// Number of iterations
    pub iterations: usize,
}

impl CPResult {
    /// Reconstruct the tensor from the decomposition
    pub fn reconstruct(&self) -> Tensor3D {
        let (n0, rank) = (self.factor_a.nrows(), self.factor_a.ncols());
        let n1 = self.factor_b.nrows();
        let n2 = self.factor_c.nrows();

        let mut tensor = Tensor3D::zeros((n0, n1, n2));

        for r in 0..rank {
            let weight = self.weights[r];
            for i in 0..n0 {
                for j in 0..n1 {
                    for k in 0..n2 {
                        tensor[(i, j, k)] += weight
                            * self.factor_a[[i, r]]
                            * self.factor_b[[j, r]]
                            * self.factor_c[[k, r]];
                    }
                }
            }
        }
        tensor
    }

    /// Compute the reconstruction error
    pub fn reconstruction_error(&self, original: &Tensor3D) -> f64 {
        let reconstructed = self.reconstruct();
        let diff = original.sub(&reconstructed);
        diff.frobenius_norm()
    }
}

/// CP Decomposition algorithm
pub struct CPDecomposition {
    /// Target rank of decomposition
    pub rank: usize,
    /// Maximum iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tolerance: f64,
}

impl Default for CPDecomposition {
    fn default() -> Self {
        Self {
            rank: 10,
            max_iter: 100,
            tolerance: 1e-6,
        }
    }
}

impl CPDecomposition {
    /// Create a new CP decomposition with given rank
    pub fn new(rank: usize) -> Self {
        Self {
            rank,
            ..Default::default()
        }
    }

    /// Create with custom parameters
    pub fn with_params(rank: usize, max_iter: usize, tolerance: f64) -> Self {
        Self {
            rank,
            max_iter,
            tolerance,
        }
    }

    /// Perform CP decomposition using Alternating Least Squares (ALS)
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to decompose
    ///
    /// # Returns
    ///
    /// The decomposition result containing factor matrices and weights
    pub fn decompose(&self, tensor: &Tensor3D) -> CPResult {
        let (n0, n1, n2) = tensor.shape();
        let rank = self.rank;

        let mut rng = rand::thread_rng();

        // Initialize factor matrices randomly
        let mut a = Array2::from_shape_fn((n0, rank), |_| {
            rng.sample::<f64, _>(StandardNormal)
        });
        let mut b = Array2::from_shape_fn((n1, rank), |_| {
            rng.sample::<f64, _>(StandardNormal)
        });
        let mut c = Array2::from_shape_fn((n2, rank), |_| {
            rng.sample::<f64, _>(StandardNormal)
        });

        // Initialize weights to 1
        let mut weights = Array1::ones(rank);

        let mut prev_error = f64::MAX;
        let mut iterations = 0;

        for iter in 0..self.max_iter {
            iterations = iter + 1;

            // Update A: solve X_(0) ≈ A (C ⊙ B)^T
            // Where ⊙ is Khatri-Rao product
            a = self.update_factor(tensor, &b, &c, 0);

            // Update B: solve X_(1) ≈ B (C ⊙ A)^T
            b = self.update_factor(tensor, &a, &c, 1);

            // Update C: solve X_(2) ≈ C (B ⊙ A)^T
            c = self.update_factor(tensor, &a, &b, 2);

            // Normalize factors and update weights
            for r in 0..rank {
                let norm_a: f64 = a.column(r).iter().map(|x| x * x).sum::<f64>().sqrt();
                let norm_b: f64 = b.column(r).iter().map(|x| x * x).sum::<f64>().sqrt();
                let norm_c: f64 = c.column(r).iter().map(|x| x * x).sum::<f64>().sqrt();

                if norm_a > 1e-10 {
                    for i in 0..n0 {
                        a[[i, r]] /= norm_a;
                    }
                }
                if norm_b > 1e-10 {
                    for j in 0..n1 {
                        b[[j, r]] /= norm_b;
                    }
                }
                if norm_c > 1e-10 {
                    for k in 0..n2 {
                        c[[k, r]] /= norm_c;
                    }
                }

                weights[r] = norm_a * norm_b * norm_c;
            }

            // Check convergence
            let result = CPResult {
                factor_a: a.clone(),
                factor_b: b.clone(),
                factor_c: c.clone(),
                weights: weights.clone(),
                final_error: 0.0,
                iterations,
            };

            let error = result.reconstruction_error(tensor);

            if (prev_error - error).abs() < self.tolerance {
                return CPResult {
                    factor_a: a,
                    factor_b: b,
                    factor_c: c,
                    weights,
                    final_error: error,
                    iterations,
                };
            }

            prev_error = error;
        }

        CPResult {
            factor_a: a,
            factor_b: b,
            factor_c: c,
            weights,
            final_error: prev_error,
            iterations,
        }
    }

    /// Update one factor matrix using ALS
    fn update_factor(
        &self,
        tensor: &Tensor3D,
        other1: &Array2<f64>,
        other2: &Array2<f64>,
        mode: usize,
    ) -> Array2<f64> {
        // Unfold tensor along the target mode
        let unfolded = match mode {
            0 => tensor.unfold_mode0(),
            1 => tensor.unfold_mode1(),
            2 => tensor.unfold_mode2(),
            _ => panic!("Invalid mode"),
        };

        // Khatri-Rao product of the other factors
        let kr = khatri_rao(other2, other1);

        // Solve: unfolded ≈ result @ kr^T
        // Using normal equations: result = unfolded @ kr @ (kr^T @ kr)^-1
        let gram = kr.t().dot(&kr);

        // Regularization for numerical stability
        let eye = Array2::eye(self.rank);
        let gram_reg = &gram + &(&eye * 1e-8);

        // Simple pseudo-inverse (for small matrices)
        // In production, use proper linear algebra library
        let gram_inv = pseudo_inverse(&gram_reg);

        unfolded.dot(&kr).dot(&gram_inv)
    }
}

/// Khatri-Rao product (column-wise Kronecker product)
pub fn khatri_rao(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let (m, r) = (a.nrows(), a.ncols());
    let n = b.nrows();

    assert_eq!(r, b.ncols(), "Matrices must have same number of columns");

    let mut result = Array2::zeros((m * n, r));

    for col in 0..r {
        for i in 0..m {
            for j in 0..n {
                result[[i * n + j, col]] = a[[i, col]] * b[[j, col]];
            }
        }
    }

    result
}

/// Simple pseudo-inverse for small matrices
fn pseudo_inverse(a: &Array2<f64>) -> Array2<f64> {
    let n = a.nrows();

    // For small matrices, use Gaussian elimination
    let mut aug = Array2::zeros((n, 2 * n));

    // Set up augmented matrix [A | I]
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = a[[i, j]];
        }
        aug[[i, n + i]] = 1.0;
    }

    // Gaussian elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let mut max_row = col;
        let mut max_val = aug[[col, col]].abs();
        for row in (col + 1)..n {
            if aug[[row, col]].abs() > max_val {
                max_val = aug[[row, col]].abs();
                max_row = row;
            }
        }

        // Swap rows
        if max_row != col {
            for j in 0..(2 * n) {
                let temp = aug[[col, j]];
                aug[[col, j]] = aug[[max_row, j]];
                aug[[max_row, j]] = temp;
            }
        }

        // Eliminate
        let pivot = aug[[col, col]];
        if pivot.abs() < 1e-10 {
            continue;
        }

        for j in 0..(2 * n) {
            aug[[col, j]] /= pivot;
        }

        for row in 0..n {
            if row != col {
                let factor = aug[[row, col]];
                for j in 0..(2 * n) {
                    aug[[row, j]] -= factor * aug[[col, j]];
                }
            }
        }
    }

    // Extract inverse
    let mut inv = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            inv[[i, j]] = aug[[i, n + j]];
        }
    }

    inv
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_khatri_rao() {
        let a = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = Array2::from_shape_vec((2, 2), vec![5.0, 6.0, 7.0, 8.0]).unwrap();

        let kr = khatri_rao(&a, &b);

        assert_eq!(kr.shape(), &[4, 2]);

        // First column: a[:, 0] ⊗ b[:, 0]
        assert_eq!(kr[[0, 0]], 1.0 * 5.0);
        assert_eq!(kr[[1, 0]], 1.0 * 7.0);
        assert_eq!(kr[[2, 0]], 3.0 * 5.0);
        assert_eq!(kr[[3, 0]], 3.0 * 7.0);
    }

    #[test]
    fn test_cp_decomposition() {
        // Create a simple rank-2 tensor
        let mut tensor = Tensor3D::zeros((3, 4, 5));
        for i in 0..3 {
            for j in 0..4 {
                for k in 0..5 {
                    tensor[(i, j, k)] = (i * j * k) as f64;
                }
            }
        }

        let cp = CPDecomposition::new(5);
        let result = cp.decompose(&tensor);

        // The reconstruction error should be small
        let error = result.reconstruction_error(&tensor);
        println!("Reconstruction error: {}", error);

        // Relative error
        let original_norm = tensor.frobenius_norm();
        let relative_error = error / original_norm.max(1e-10);
        assert!(relative_error < 0.5); // Allow some error for low-rank approximation
    }
}
