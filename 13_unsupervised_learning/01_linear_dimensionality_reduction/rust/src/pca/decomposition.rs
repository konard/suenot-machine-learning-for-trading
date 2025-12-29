//! Matrix decomposition utilities for PCA

use ndarray::{Array1, Array2, Axis};

/// Eigenvalue decomposition result
#[derive(Debug, Clone)]
pub struct EigenDecomposition {
    /// Eigenvalues (sorted in descending order)
    pub eigenvalues: Array1<f64>,
    /// Eigenvectors (columns correspond to eigenvalues)
    pub eigenvectors: Array2<f64>,
}

impl EigenDecomposition {
    /// Perform eigenvalue decomposition of a symmetric matrix
    /// Uses power iteration method for simplicity
    pub fn from_symmetric(matrix: &Array2<f64>) -> Self {
        let n = matrix.nrows();
        let mut eigenvalues = Array1::zeros(n);
        let mut eigenvectors = Array2::zeros((n, n));
        let mut deflated = matrix.clone();

        for i in 0..n {
            // Power iteration to find largest eigenvalue
            let (eigenvalue, eigenvector) = power_iteration(&deflated, 100, 1e-10);

            eigenvalues[i] = eigenvalue;
            for j in 0..n {
                eigenvectors[[j, i]] = eigenvector[j];
            }

            // Deflate matrix: A = A - λ * v * v^T
            let v = eigenvector.clone();
            let outer = outer_product(&v, &v);
            deflated = deflated - eigenvalue * outer;
        }

        // Sort by eigenvalue (descending)
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| {
            eigenvalues[b]
                .partial_cmp(&eigenvalues[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let sorted_eigenvalues = Array1::from_vec(indices.iter().map(|&i| eigenvalues[i]).collect());

        let mut sorted_eigenvectors = Array2::zeros((n, n));
        for (new_idx, &old_idx) in indices.iter().enumerate() {
            for j in 0..n {
                sorted_eigenvectors[[j, new_idx]] = eigenvectors[[j, old_idx]];
            }
        }

        Self {
            eigenvalues: sorted_eigenvalues,
            eigenvectors: sorted_eigenvectors,
        }
    }
}

/// Power iteration to find the largest eigenvalue and its eigenvector
fn power_iteration(matrix: &Array2<f64>, max_iter: usize, tol: f64) -> (f64, Array1<f64>) {
    let n = matrix.nrows();
    let mut v = Array1::from_vec(vec![1.0 / (n as f64).sqrt(); n]);
    let mut eigenvalue = 0.0;

    for _ in 0..max_iter {
        // Multiply: v = A * v
        let mut new_v = Array1::zeros(n);
        for i in 0..n {
            for j in 0..n {
                new_v[i] += matrix[[i, j]] * v[j];
            }
        }

        // Compute eigenvalue as Rayleigh quotient: λ = (v^T * A * v) / (v^T * v)
        let new_eigenvalue: f64 = v.iter().zip(new_v.iter()).map(|(&a, &b)| a * b).sum();

        // Normalize
        let norm: f64 = new_v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-10 {
            new_v /= norm;
        }

        // Check convergence
        if (new_eigenvalue - eigenvalue).abs() < tol {
            return (new_eigenvalue, new_v);
        }

        eigenvalue = new_eigenvalue;
        v = new_v;
    }

    (eigenvalue, v)
}

/// Compute outer product of two vectors
fn outer_product(a: &Array1<f64>, b: &Array1<f64>) -> Array2<f64> {
    let n = a.len();
    let m = b.len();
    let mut result = Array2::zeros((n, m));

    for i in 0..n {
        for j in 0..m {
            result[[i, j]] = a[i] * b[j];
        }
    }

    result
}

/// Singular Value Decomposition result (simplified)
#[derive(Debug, Clone)]
pub struct SVDResult {
    /// Left singular vectors (U)
    pub u: Array2<f64>,
    /// Singular values
    pub singular_values: Array1<f64>,
    /// Right singular vectors (V^T)
    pub vt: Array2<f64>,
}

impl SVDResult {
    /// Compute SVD using eigendecomposition of A^T * A
    /// This is a simplified implementation for educational purposes
    pub fn compute(matrix: &Array2<f64>) -> Self {
        let (m, n) = matrix.dim();

        // Compute A^T * A
        let ata = matrix.t().dot(matrix);

        // Eigendecomposition of A^T * A gives V and σ²
        let eigen = EigenDecomposition::from_symmetric(&ata);

        // Singular values are square roots of eigenvalues
        let singular_values = eigen.eigenvalues.mapv(|x| if x > 0.0 { x.sqrt() } else { 0.0 });

        // V^T are the eigenvectors
        let vt = eigen.eigenvectors.t().to_owned();

        // U = A * V * Σ^(-1)
        let v = &eigen.eigenvectors;
        let mut u = Array2::zeros((m, n.min(m)));

        for i in 0..n.min(m) {
            if singular_values[i] > 1e-10 {
                let av = matrix.dot(&v.column(i));
                for j in 0..m {
                    u[[j, i]] = av[j] / singular_values[i];
                }
            }
        }

        Self {
            u,
            singular_values,
            vt,
        }
    }
}

/// Calculate covariance matrix
pub fn covariance_matrix(data: &Array2<f64>) -> Array2<f64> {
    let n = data.nrows() as f64;
    let mean = data.mean_axis(Axis(0)).unwrap();

    // Center the data
    let centered = data - &mean;

    // Covariance = (X^T * X) / (n - 1)
    centered.t().dot(&centered) / (n - 1.0)
}

/// Calculate correlation matrix from covariance matrix
pub fn correlation_from_covariance(cov: &Array2<f64>) -> Array2<f64> {
    let n = cov.nrows();
    let mut corr = Array2::zeros((n, n));

    // Get standard deviations from diagonal
    let std_devs: Vec<f64> = (0..n).map(|i| cov[[i, i]].sqrt()).collect();

    for i in 0..n {
        for j in 0..n {
            if std_devs[i] > 1e-10 && std_devs[j] > 1e-10 {
                corr[[i, j]] = cov[[i, j]] / (std_devs[i] * std_devs[j]);
            } else {
                corr[[i, j]] = if i == j { 1.0 } else { 0.0 };
            }
        }
    }

    corr
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_eigen_decomposition() {
        // Simple 2x2 symmetric matrix
        let matrix = array![[4.0, 2.0], [2.0, 3.0]];
        let eigen = EigenDecomposition::from_symmetric(&matrix);

        // Eigenvalues should be approximately 5.56 and 1.44
        assert!(eigen.eigenvalues[0] > eigen.eigenvalues[1]);
        assert!((eigen.eigenvalues.sum() - 7.0).abs() < 0.1); // trace = sum of eigenvalues
    }

    #[test]
    fn test_covariance_matrix() {
        let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let cov = covariance_matrix(&data);

        // Should be 2x2 symmetric matrix
        assert_eq!(cov.shape(), &[2, 2]);
        assert!((cov[[0, 1]] - cov[[1, 0]]).abs() < 1e-10);
    }

    #[test]
    fn test_correlation_from_covariance() {
        let cov = array![[1.0, 0.5], [0.5, 1.0]];
        let corr = correlation_from_covariance(&cov);

        // Diagonal should be 1
        assert!((corr[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((corr[[1, 1]] - 1.0).abs() < 1e-10);

        // Off-diagonal should be 0.5
        assert!((corr[[0, 1]] - 0.5).abs() < 1e-10);
    }
}
