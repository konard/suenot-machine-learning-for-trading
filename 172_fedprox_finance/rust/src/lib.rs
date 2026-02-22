use ndarray::{Array1, ArrayView1};
use rayon::prelude::*;

/// Optimized Proximal Engine for FedProx.
pub struct FedProxEngine;

impl FedProxEngine {
    /// Calculates the proximal term (squared L2 norm) between two sets of weights.
    /// penalty = (mu / 2) * ||w - w_t||^2
    ///
    /// This function parallelizes the calculation over the flattened weight tensors.
    pub fn calculate_proximal_term(w: &Array1<f32>, wt: &Array1<f32>, mu: f32) -> f32 {
        assert_eq!(w.len(), wt.len(), "Weight dimensions must match");
        
        let squared_diff_sum: f32 = w.as_slice().unwrap()
            .par_iter()
            .zip(wt.as_slice().unwrap().par_iter())
            .map(|(wi, wti)| (wi - wti).powi(2))
            .sum();
            
        (mu / 2.0) * squared_diff_sum
    }

    /// Computes the proximal update direction (w - wt).
    pub fn compute_proximal_gradient(w: &Array1<f32>, wt: &Array1<f32>, mu: f32) -> Array1<f32> {
        let mut grad = Array1::zeros(w.len());
        
        grad.as_slice_mut().unwrap()
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, val)| {
                *val = mu * (w[i] - wt[i]);
            });
            
        grad
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_proximal_term_calculation() {
        let w = array![1.0, 2.0, 3.0];
        let wt = array![0.0, 0.0, 0.0];
        let mu = 1.0;
        
        // penalty = (1.0 / 2.0) * (1^2 + 2^2 + 3^2) = 0.5 * 14 = 7.0
        let penalty = FedProxEngine::calculate_proximal_term(&w, &wt, mu);
        assert!((penalty - 7.0).abs() < 1e-6);
    }

    #[test]
    fn test_proximal_gradient_calculation() {
        let w = array![2.0, 4.0];
        let wt = array![1.0, 1.0];
        let mu = 0.5;
        
        // grad = 0.5 * (2-1) = 0.5
        // grad = 0.5 * (4-1) = 1.5
        let grad = FedProxEngine::compute_proximal_gradient(&w, &wt, mu);
        assert!((grad[0] - 0.5).abs() < 1e-6);
        assert!((grad[1] - 1.5).abs() < 1e-6);
    }
}
