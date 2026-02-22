use ndarray::Array1;
use rayon::prelude::*;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

pub struct SecureAggEngine;

impl SecureAggEngine {
    /// Generates a mask (noise) from a seed and applies it to a weight vector.
    /// If is_adder is true, adds the mask. Otherwise, subtracts it.
    pub fn apply_mask(weights: &mut Array1<f32>, seed: u64, is_adder: bool) {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        
        // Generate a vector of noise efficiently
        // Note: For large vectors, we want to parallelize the noise application
        // but generate the noise deterministically.
        
        let noise_strength = 10.0;
        
        // chunk-based parallel generation for performance on large tensors
        let chunk_size = 1024;
        weights.as_slice_mut().unwrap()
            .par_chunks_mut(chunk_size)
            .enumerate()
            .for_each(|(i, chunk)| {
                // Each chunk needs its own deterministically derived RNG 
                // to maintain consistency with other clients.
                let mut chunk_rng = ChaCha20Rng::seed_from_u64(seed + i as u64);
                for val in chunk.iter_mut() {
                    let noise: f32 = chunk_rng.gen::<f32>() * noise_strength;
                    if is_adder {
                        *val += noise;
                    } else {
                        *val -= noise;
                    }
                }
            });
    }

    /// Optimized server-side sum of blinded vectors.
    pub fn sum_blinded(blinded_vectors: &[Array1<f32>]) -> Array1<f32> {
        let n = blinded_vectors.len();
        if n == 0 { return Array1::zeros(0); }
        let dim = blinded_vectors[0].len();
        
        let mut sum = Array1::zeros(dim);
        
        sum.as_slice_mut().unwrap()
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, val)| {
                let mut s = 0.0;
                for v in blinded_vectors {
                    s += v[i];
                }
                *val = s;
            });
            
        sum
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_mask_cancellation() {
        let mut w1 = array![1.0, 2.0, 3.0];
        let mut w2 = array![10.0, 20.0, 30.0];
        let original_sum = &w1 + &w2;
        
        let shared_seed = 12345;
        
        // Client 1 adds mask
        SecureAggEngine::apply_mask(&mut w1, shared_seed, true);
        // Client 2 subtracts same mask
        SecureAggEngine::apply_mask(&mut w2, shared_seed, false);
        
        // Server sums them
        let result = SecureAggEngine::sum_blinded(&[w1.clone(), w2.clone()]);
        
        // Verification: result should equal original_sum
        for i in 0..3 {
            assert!((result[i] - original_sum[i]).abs() < 1e-4);
        }
        
        // Verification: Individual blinded vectors should be very different from original
        assert!((w1[0] - 1.0).abs() > 1.0);
        assert!((w2[0] - 10.0).abs() > 1.0);
    }
}
