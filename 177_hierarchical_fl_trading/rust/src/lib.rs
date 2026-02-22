use ndarray::Array1;
use rayon::prelude::*;

pub struct HierarchicalAggregator;

impl HierarchicalAggregator {
    /// Aggregates multiple model updates (tensors) into a single average.
    /// Optimized for high-throughput regional hubs.
    pub fn aggregate_tensors(
        updates: Vec<Array1<f32>>,
        weights: Vec<f32>,
    ) -> Array1<f32> {
        assert!(!updates.is_empty(), "Updates list cannot be empty");
        assert_eq!(updates.len(), weights.len(), "Weights must match number of updates");
        
        let length = updates[0].len();
        let total_weight: f32 = weights.iter().sum();
        
        let mut result = Array1::zeros(length);
        let res_slice = result.as_slice_mut().unwrap();

        // Parallelize across the parameter space
        res_slice.par_iter_mut().enumerate().for_each(|(i, val)| {
            let mut sum = 0.0;
            for (update, weight) in updates.iter().zip(weights.iter()) {
                sum += update[i] * weight;
            }
            *val = sum / total_weight;
        });

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_hierarchical_aggregation() {
        let u1 = array![1.0, 2.0, 3.0];
        let u2 = array![3.0, 4.0, 5.0];
        let weights = vec![1.0, 1.0];
        
        let aggregated = HierarchicalAggregator::aggregate_tensors(vec![u1, u2], weights);
        
        // Expected: (1+3)/2=2, (2+4)/2=3, (3+5)/2=4
        assert!((aggregated[0] - 2.0).abs() < 1e-6);
        assert!((aggregated[1] - 3.0).abs() < 1e-6);
        assert!((aggregated[2] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_weighted_aggregation() {
        let u1 = array![1.0, 1.0];
        let u2 = array![10.0, 10.0];
        let weights = vec![0.9, 0.1]; // 9:1 ratio
        
        let aggregated = HierarchicalAggregator::aggregate_tensors(vec![u1, u2], weights);
        
        // Expected: 1*0.9 + 10*0.1 = 0.9 + 1.0 = 1.9
        assert!((aggregated[0] - 1.9).abs() < 1e-6);
    }
}
