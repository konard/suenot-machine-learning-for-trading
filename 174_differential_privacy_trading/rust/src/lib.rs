use ndarray::{Array1, ArrayView1};
use rayon::prelude::*;
use rand_distr::{Normal, Distribution};
use rand::thread_rng;

pub struct DPEngine;

impl DPEngine {
    /// Optimized per-sample gradient clipping.
    /// Scales the gradient vector if its L2 norm exceeds the threshold C.
    pub fn clip_gradient(grad: &mut Array1<f32>, c: f32) -> f32 {
        let l2_norm = grad.as_slice().unwrap()
            .par_iter()
            .map(|&x| x * x)
            .sum::<f32>()
            .sqrt();
        
        let scale = if l2_norm > c {
            c / l2_norm
        } else {
            1.0
        };

        if scale < 1.0 {
            grad.as_slice_mut().unwrap()
                .par_iter_mut()
                .for_each(|x| *x *= scale);
        }

        scale
    }

    /// Optimized noise injection (Gaussian noise).
    /// Adds noise with scale sigma * C to the gradient vector.
    pub fn add_noise(grad: &mut Array1<f32>, sigma: f32, c: f32) {
        let std_dev = sigma * c;
        if std_dev <= 0.0 { return; }
        
        let normal = Normal::new(0.0, std_dev).unwrap();
        
        // Chunk-based parallel noise generation
        let chunk_size = 1024;
        grad.as_slice_mut().unwrap()
            .par_chunks_mut(chunk_size)
            .for_each(|chunk| {
                let mut rng = thread_rng();
                for val in chunk.iter_mut() {
                    *val += normal.sample(&mut rng);
                }
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_gradient_clipping() {
        let mut grad = array![10.0, 0.0];
        let c = 1.0;
        
        // l2_norm is 10.0, scale = 1/10 = 0.1
        let scale = DPEngine::clip_gradient(&mut grad, c);
        
        assert!((scale - 0.1).abs() < 1e-6);
        assert!((grad[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_noise_injection() {
        let mut grad = Array1::zeros(1000);
        let sigma = 1.0;
        let c = 1.0;
        
        DPEngine::add_noise(&mut grad, sigma, c);
        
        // Simple statistical check: mean should be near 0
        let sum: f32 = grad.sum();
        let mean = sum / 1000.0;
        assert!(mean.abs() < 0.2); // Loose bound for randomness
    }
}
