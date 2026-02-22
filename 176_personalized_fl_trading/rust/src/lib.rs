use ndarray::Array1;
use rayon::prelude::*;

pub struct PFLEngine;

impl PFLEngine {
    /// Blends global and local model parameters using weighted interpolation.
    /// Formula: W_personalized = alpha * W_global + (1 - alpha) * W_local
    /// This is performed in parallel across high-dimensional parameter vectors.
    pub fn interpolate(
        global_params: &Array1<f32>,
        local_params: &Array1<f32>,
        alpha: f32,
    ) -> Array1<f32> {
        assert_eq!(global_params.len(), local_params.len(), "Parameter vectors must have the same length");
        
        // Parallel computation of W_personalized = alpha * W_global + (1 - alpha) * W_local
        let mut personalized = Array1::zeros(global_params.len());
        
        let global_slice = global_params.as_slice().unwrap();
        let local_slice = local_params.as_slice().unwrap();
        let pers_slice = personalized.as_slice_mut().unwrap();

        pers_slice.par_iter_mut()
            .enumerate()
            .for_each(|(i, val)| {
                *val = alpha * global_slice[i] + (1.0 - alpha) * local_slice[i];
            });

        personalized
    }

    /// Optimized calculation of personalized model update for multiple layers.
    pub fn blend_layers(
        layers_global: Vec<Array1<f32>>,
        layers_local: Vec<Array1<f32>>,
        alpha: f32,
    ) -> Vec<Array1<f32>> {
        layers_global.into_iter()
            .zip(layers_local.into_iter())
            .map(|(g, l)| Self::interpolate(&g, &l, alpha))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_model_interpolation() {
        let global = array![1.0, 2.0, 3.0];
        let local = array![0.0, 0.0, 0.0];
        let alpha = 0.5;
        
        let personalized = PFLEngine::interpolate(&global, &local, alpha);
        
        assert!((personalized[0] - 0.5).abs() < 1e-6);
        assert!((personalized[1] - 1.0).abs() < 1e-6);
        assert!((personalized[2] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_interpolation_bounds() {
        let global = array![1.0, 1.0];
        let local = array![5.0, 5.0];
        
        // alpha = 1.0 should give 100% global
        let pers_full_global = PFLEngine::interpolate(&global, &local, 1.0);
        assert!((pers_full_global[0] - 1.0).abs() < 1e-6);
        
        // alpha = 0.0 should give 100% local
        let pers_full_local = PFLEngine::interpolate(&global, &local, 0.0);
        assert!((pers_full_local[0] - 5.0).abs() < 1e-6);
    }
}
