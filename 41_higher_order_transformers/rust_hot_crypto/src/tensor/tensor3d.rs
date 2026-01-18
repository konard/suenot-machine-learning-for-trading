//! Three-dimensional tensor type
//!
//! A wrapper around ndarray::Array3 with additional operations
//! useful for higher-order attention computation.

use ndarray::{Array2, Array3, Axis};

/// A three-dimensional tensor
#[derive(Debug, Clone)]
pub struct Tensor3D {
    /// The underlying data
    pub data: Array3<f64>,
}

impl Tensor3D {
    /// Create a new tensor from an ndarray
    pub fn new(data: Array3<f64>) -> Self {
        Self { data }
    }

    /// Create a tensor of zeros
    pub fn zeros(shape: (usize, usize, usize)) -> Self {
        Self {
            data: Array3::zeros(shape),
        }
    }

    /// Create a tensor from random values
    pub fn random(shape: (usize, usize, usize)) -> Self {
        Self {
            data: Array3::from_shape_fn(shape, |_| rand::random::<f64>()),
        }
    }

    /// Get the shape of the tensor
    pub fn shape(&self) -> (usize, usize, usize) {
        let s = self.data.shape();
        (s[0], s[1], s[2])
    }

    /// Get a value at the given indices
    pub fn get(&self, i: usize, j: usize, k: usize) -> f64 {
        self.data[[i, j, k]]
    }

    /// Set a value at the given indices
    pub fn set(&mut self, i: usize, j: usize, k: usize, value: f64) {
        self.data[[i, j, k]] = value;
    }

    /// Unfold (matricize) the tensor along mode 0
    ///
    /// Converts the tensor into a matrix where mode-0 fibers become columns
    pub fn unfold_mode0(&self) -> Array2<f64> {
        let (n0, n1, n2) = self.shape();
        let mut matrix = Array2::zeros((n0, n1 * n2));

        for i in 0..n0 {
            for j in 0..n1 {
                for k in 0..n2 {
                    matrix[[i, j * n2 + k]] = self.data[[i, j, k]];
                }
            }
        }
        matrix
    }

    /// Unfold (matricize) the tensor along mode 1
    pub fn unfold_mode1(&self) -> Array2<f64> {
        let (n0, n1, n2) = self.shape();
        let mut matrix = Array2::zeros((n1, n0 * n2));

        for i in 0..n0 {
            for j in 0..n1 {
                for k in 0..n2 {
                    matrix[[j, i * n2 + k]] = self.data[[i, j, k]];
                }
            }
        }
        matrix
    }

    /// Unfold (matricize) the tensor along mode 2
    pub fn unfold_mode2(&self) -> Array2<f64> {
        let (n0, n1, n2) = self.shape();
        let mut matrix = Array2::zeros((n2, n0 * n1));

        for i in 0..n0 {
            for j in 0..n1 {
                for k in 0..n2 {
                    matrix[[k, i * n1 + j]] = self.data[[i, j, k]];
                }
            }
        }
        matrix
    }

    /// Fold a matrix back into a tensor (mode 0)
    pub fn fold_mode0(matrix: &Array2<f64>, shape: (usize, usize, usize)) -> Self {
        let (n0, n1, n2) = shape;
        let mut tensor = Self::zeros(shape);

        for i in 0..n0 {
            for j in 0..n1 {
                for k in 0..n2 {
                    tensor.data[[i, j, k]] = matrix[[i, j * n2 + k]];
                }
            }
        }
        tensor
    }

    /// Contract the tensor along one dimension with a vector
    pub fn contract_mode0(&self, vec: &ndarray::Array1<f64>) -> Array2<f64> {
        let (n0, n1, n2) = self.shape();
        let mut result = Array2::zeros((n1, n2));

        for j in 0..n1 {
            for k in 0..n2 {
                let mut sum = 0.0;
                for i in 0..n0 {
                    sum += self.data[[i, j, k]] * vec[i];
                }
                result[[j, k]] = sum;
            }
        }
        result
    }

    /// Compute the Frobenius norm
    pub fn frobenius_norm(&self) -> f64 {
        self.data.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    /// Scale the tensor
    pub fn scale(&mut self, factor: f64) {
        self.data.mapv_inplace(|x| x * factor);
    }

    /// Add two tensors element-wise
    pub fn add(&self, other: &Tensor3D) -> Self {
        Self {
            data: &self.data + &other.data,
        }
    }

    /// Subtract two tensors element-wise
    pub fn sub(&self, other: &Tensor3D) -> Self {
        Self {
            data: &self.data - &other.data,
        }
    }

    /// Apply softmax along the last two dimensions (for attention)
    pub fn softmax_last_dims(&self) -> Self {
        let (n0, n1, n2) = self.shape();
        let mut result = Self::zeros((n0, n1, n2));

        for i in 0..n0 {
            // Get the slice for this i
            let slice = self.data.slice(ndarray::s![i, .., ..]);

            // Find max for numerical stability
            let max_val = slice.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

            // Compute exp and sum
            let mut sum = 0.0;
            for j in 0..n1 {
                for k in 0..n2 {
                    let exp_val = (slice[[j, k]] - max_val).exp();
                    result.data[[i, j, k]] = exp_val;
                    sum += exp_val;
                }
            }

            // Normalize
            for j in 0..n1 {
                for k in 0..n2 {
                    result.data[[i, j, k]] /= sum;
                }
            }
        }
        result
    }
}

impl std::ops::Index<(usize, usize, usize)> for Tensor3D {
    type Output = f64;

    fn index(&self, (i, j, k): (usize, usize, usize)) -> &Self::Output {
        &self.data[[i, j, k]]
    }
}

impl std::ops::IndexMut<(usize, usize, usize)> for Tensor3D {
    fn index_mut(&mut self, (i, j, k): (usize, usize, usize)) -> &mut Self::Output {
        &mut self.data[[i, j, k]]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation() {
        let t = Tensor3D::zeros((3, 4, 5));
        assert_eq!(t.shape(), (3, 4, 5));
        assert_eq!(t.get(0, 0, 0), 0.0);
    }

    #[test]
    fn test_unfold_fold_roundtrip() {
        let t = Tensor3D::random((3, 4, 5));
        let unfolded = t.unfold_mode0();
        let folded = Tensor3D::fold_mode0(&unfolded, (3, 4, 5));

        for i in 0..3 {
            for j in 0..4 {
                for k in 0..5 {
                    assert!((t.get(i, j, k) - folded.get(i, j, k)).abs() < 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_frobenius_norm() {
        let mut t = Tensor3D::zeros((2, 2, 2));
        t.set(0, 0, 0, 1.0);
        t.set(0, 0, 1, 2.0);
        t.set(0, 1, 0, 3.0);

        // sqrt(1 + 4 + 9) = sqrt(14)
        let norm = t.frobenius_norm();
        assert!((norm - (14.0_f64).sqrt()).abs() < 1e-10);
    }
}
