use ndarray::{Array1, ArrayView1, Axis};
use rayon::prelude::*;

/// Optimized Federated Averaging Aggregator.
pub struct FedAvgOptimizer;

impl FedAvgOptimizer {
    /// Aggregates multiple model updates (flattened tensors) using weighted averaging.
    ///
    /// Args:
    ///     client_models: Matrix of shape (num_clients, total_params)
    ///     client_samples: Vector of length num_clients
    /// 
    /// Returns:
    ///     A flattened array of averaged global weights.
    pub fn aggregate(
        client_models: &ndarray::Array2<f32>,
        client_samples: &Array1<f32>,
    ) -> Array1<f32> {
        let total_samples: f32 = client_samples.sum();
        let num_params = client_models.ncols();
        
        // Normalize weights
        let weights = client_samples / total_samples;
        
        let mut global_model = Array1::zeros(num_params);
        
        // Parallelize over parameters for high efficiency
        global_model.as_slice_mut().unwrap()
            .par_iter_mut()
            .enumerate()
            .for_each(|(j, val)| {
                let mut sum = 0.0;
                for i in 0..client_models.nrows() {
                    sum += client_models[[i, j]] * weights[i];
                }
                *val = sum;
            });
            
        global_model
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_fedavg_aggregation() {
        // 2 clients, 3 parameters
        let client_models = array![
            [1.0, 2.0, 3.0], // Client 1
            [2.0, 4.0, 6.0]  // Client 2
        ];
        
        // Client 1 has 100 samples, Client 2 has 300 samples (1:3 ratio)
        let client_samples = array![100.0, 300.0];
        
        let result = FedAvgOptimizer::aggregate(&client_models, &client_samples);
        
        // Expected: 1.0*0.25 + 2.0*0.75 = 0.25 + 1.5 = 1.75
        // Expected: 2.0*0.25 + 4.0*0.75 = 0.5 + 3.0 = 3.5
        // Expected: 3.0*0.25 + 6.0*0.75 = 0.75 + 4.5 = 5.25
        
        assert!((result[0] - 1.75).abs() < 1e-6);
        assert!((result[1] - 3.50).abs() < 1e-6);
        assert!((result[2] - 5.25).abs() < 1e-6);
    }
}
