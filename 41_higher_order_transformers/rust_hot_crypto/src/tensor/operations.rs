//! Tensor operations and utilities
//!
//! Common operations for tensor manipulation

use ndarray::{Array1, Array2, Array3};

/// Outer product of three vectors: a ⊗ b ⊗ c
pub fn outer_product_3(a: &Array1<f64>, b: &Array1<f64>, c: &Array1<f64>) -> Array3<f64> {
    let n0 = a.len();
    let n1 = b.len();
    let n2 = c.len();

    Array3::from_shape_fn((n0, n1, n2), |(i, j, k)| a[i] * b[j] * c[k])
}

/// Mode-n product: tensor ×_n matrix
///
/// Multiplies a tensor by a matrix along the n-th mode
pub fn mode_n_product(tensor: &Array3<f64>, matrix: &Array2<f64>, mode: usize) -> Array3<f64> {
    let shape = tensor.shape();
    let (n0, n1, n2) = (shape[0], shape[1], shape[2]);
    let m_rows = matrix.nrows();

    match mode {
        0 => {
            assert_eq!(matrix.ncols(), n0);
            let mut result = Array3::zeros((m_rows, n1, n2));
            for j in 0..n1 {
                for k in 0..n2 {
                    for new_i in 0..m_rows {
                        let mut sum = 0.0;
                        for i in 0..n0 {
                            sum += matrix[[new_i, i]] * tensor[[i, j, k]];
                        }
                        result[[new_i, j, k]] = sum;
                    }
                }
            }
            result
        }
        1 => {
            assert_eq!(matrix.ncols(), n1);
            let mut result = Array3::zeros((n0, m_rows, n2));
            for i in 0..n0 {
                for k in 0..n2 {
                    for new_j in 0..m_rows {
                        let mut sum = 0.0;
                        for j in 0..n1 {
                            sum += matrix[[new_j, j]] * tensor[[i, j, k]];
                        }
                        result[[i, new_j, k]] = sum;
                    }
                }
            }
            result
        }
        2 => {
            assert_eq!(matrix.ncols(), n2);
            let mut result = Array3::zeros((n0, n1, m_rows));
            for i in 0..n0 {
                for j in 0..n1 {
                    for new_k in 0..m_rows {
                        let mut sum = 0.0;
                        for k in 0..n2 {
                            sum += matrix[[new_k, k]] * tensor[[i, j, k]];
                        }
                        result[[i, j, new_k]] = sum;
                    }
                }
            }
            result
        }
        _ => panic!("Mode must be 0, 1, or 2"),
    }
}

/// Tensor contraction along specified modes
pub fn contract(
    tensor1: &Array3<f64>,
    tensor2: &Array3<f64>,
    mode1: usize,
    mode2: usize,
) -> Array3<f64> {
    // Simplified contraction for matching dimensions
    let s1 = tensor1.shape();
    let s2 = tensor2.shape();

    let dim1 = s1[mode1];
    let dim2 = s2[mode2];

    assert_eq!(dim1, dim2, "Contraction dimensions must match");

    // This is a simplified implementation
    // Full tensor contraction is more complex
    match (mode1, mode2) {
        (2, 0) => {
            // Contract tensor1's mode-2 with tensor2's mode-0
            let mut result = Array3::zeros((s1[0], s1[1], s2[2]));
            for i in 0..s1[0] {
                for j in 0..s1[1] {
                    for l in 0..s2[2] {
                        let mut sum = 0.0;
                        for k in 0..dim1 {
                            sum += tensor1[[i, j, k]] * tensor2[[k, 0, l]].max(0.0);
                        }
                        result[[i, j, l]] = sum;
                    }
                }
            }
            result
        }
        _ => {
            // Default: just return first tensor (placeholder)
            tensor1.clone()
        }
    }
}

/// Compute tensor norm
pub fn tensor_norm(tensor: &Array3<f64>) -> f64 {
    tensor.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Normalize tensor to unit norm
pub fn normalize_tensor(tensor: &Array3<f64>) -> Array3<f64> {
    let norm = tensor_norm(tensor);
    if norm > 1e-10 {
        tensor / norm
    } else {
        tensor.clone()
    }
}

/// Inner product of two tensors
pub fn tensor_inner_product(t1: &Array3<f64>, t2: &Array3<f64>) -> f64 {
    assert_eq!(t1.shape(), t2.shape());
    t1.iter().zip(t2.iter()).map(|(a, b)| a * b).sum()
}

/// Create a diagonal tensor
pub fn diagonal_tensor(diag: &Array1<f64>) -> Array3<f64> {
    let n = diag.len();
    let mut tensor = Array3::zeros((n, n, n));
    for i in 0..n {
        tensor[[i, i, i]] = diag[i];
    }
    tensor
}

/// Apply element-wise function to tensor
pub fn tensor_map<F>(tensor: &Array3<f64>, f: F) -> Array3<f64>
where
    F: Fn(f64) -> f64,
{
    tensor.mapv(&f)
}

/// ReLU activation for tensor
pub fn tensor_relu(tensor: &Array3<f64>) -> Array3<f64> {
    tensor_map(tensor, |x| x.max(0.0))
}

/// Sigmoid activation for tensor
pub fn tensor_sigmoid(tensor: &Array3<f64>) -> Array3<f64> {
    tensor_map(tensor, |x| 1.0 / (1.0 + (-x).exp()))
}

/// Tanh activation for tensor
pub fn tensor_tanh(tensor: &Array3<f64>) -> Array3<f64> {
    tensor_map(tensor, |x| x.tanh())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_outer_product() {
        let a = Array1::from_vec(vec![1.0, 2.0]);
        let b = Array1::from_vec(vec![3.0, 4.0]);
        let c = Array1::from_vec(vec![5.0, 6.0]);

        let result = outer_product_3(&a, &b, &c);

        assert_eq!(result[[0, 0, 0]], 1.0 * 3.0 * 5.0);
        assert_eq!(result[[1, 1, 1]], 2.0 * 4.0 * 6.0);
    }

    #[test]
    fn test_tensor_norm() {
        let mut tensor = Array3::zeros((2, 2, 2));
        tensor[[0, 0, 0]] = 3.0;
        tensor[[0, 0, 1]] = 4.0;

        let norm = tensor_norm(&tensor);
        assert!((norm - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_diagonal_tensor() {
        let diag = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let tensor = diagonal_tensor(&diag);

        assert_eq!(tensor[[0, 0, 0]], 1.0);
        assert_eq!(tensor[[1, 1, 1]], 2.0);
        assert_eq!(tensor[[2, 2, 2]], 3.0);
        assert_eq!(tensor[[0, 0, 1]], 0.0);
    }
}
