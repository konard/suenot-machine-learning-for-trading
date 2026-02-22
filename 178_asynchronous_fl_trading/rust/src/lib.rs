use ndarray::Array1;
use rayon::prelude::*;

pub struct AFLEngine;

impl AFLEngine {
    /// Calculates staleness decay coefficient.
    /// Formula (polynomial decay): s(tau) = (tau + 1)^(-a)
    pub fn calculate_decay(tau: u32, a: f32) -> f32 {
        ((tau as f32) + 1.0).powf(-a)
    }

    /// Blends global model and asynchronous client update with staleness weighting.
    /// Formula: W_t+1 = (1 - alpha*s(tau)) * W_t + (alpha*s(tau)) * W_client
    pub fn blend_update(
        global_params: &Array1<f32>,
        client_params: &Array1<f32>,
        tau: u32,
        base_alpha: f32,
        decay_exponent: f32,
    ) -> Array1<f32> {
        assert_eq!(global_params.len(), client_params.len(), "Parameter vectors must have the same length");
        
        let s_tau = Self::calculate_decay(tau, decay_exponent);
        let learning_rate = base_alpha * s_tau;
        
        let mut updated = Array1::zeros(global_params.len());
        
        let global_slice = global_params.as_slice().unwrap();
        let client_slice = client_params.as_slice().unwrap();
        let upd_slice = updated.as_slice_mut().unwrap();

        // High-performance parallel blending
        upd_slice.par_iter_mut()
            .enumerate()
            .for_each(|(i, val)| {
                *val = (1.0 - learning_rate) * global_slice[i] + learning_rate * client_slice[i];
            });

        updated
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_decay_calculation() {
        // tau = 0 should give 1.0
        assert!((AFLEngine::calculate_decay(0, 0.5) - 1.0).abs() < 1e-6);
        // tau = 3, a = 0.5 should give 4^(-0.5) = 0.5
        assert!((AFLEngine::calculate_decay(3, 0.5) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_asynchronous_blending() {
        let global = array![1.0, 1.0];
        let client = array![2.0, 2.0];
        let tau = 3; 
        let alpha = 1.0;
        let decay = 0.5; // weight = 4^(-0.5) = 0.5
        
        let updated = AFLEngine::blend_update(&global, &client, tau, alpha, decay);
        
        // W_new = (1 - 0.5) * 1.0 + 0.5 * 2.0 = 0.5 + 1.0 = 1.5
        assert!((updated[0] - 1.5).abs() < 1e-6);
        assert!((updated[1] - 1.5).abs() < 1e-6);
    }
}
