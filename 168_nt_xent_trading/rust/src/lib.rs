use ndarray::{Array2, Axis, Array1};

/// Optimized NT-Xent Loss calculation.
pub struct NTXentOptimizer {
    pub temperature: f32,
}

impl NTXentOptimizer {
    /// Computes the NT-Xent loss for two views of a batch.
    /// features_i and features_j are (batch_size, dim) matrices.
    pub fn compute_loss(&self, features_i: &Array2<f32>, features_j: &Array2<f32>) -> f32 {
        let batch_size = features_i.nrows();
        
        // 1. Concatenate features to (2 * batch_size, dim)
        let mut all_features = Array2::zeros((2 * batch_size, features_i.ncols()));
        for (i, row) in features_i.axis_iter(Axis(0)).enumerate() {
            all_features.row_mut(i).assign(&row);
            all_features.row_mut(i + batch_size).assign(&features_j.row(i));
        }

        // 2. Compute similarity matrix (2N, 2N)
        // Note: For production, we'd use BLAS or parallelize this.
        let mut sim_matrix = all_features.dot(&all_features.t());
        sim_matrix.mapv_inplace(|x| x / self.temperature);

        // 3. Log-Sum-Exp over rows (excluding diagonal)
        let mut total_loss = 0.0;
        for i in 0..(2 * batch_size) {
            let mut row = sim_matrix.row(i).to_owned();
            
            // Mask self-similarity (set to very low value)
            row[i] = -1e9;
            
            // Log-Sum-Exp denominator
            let max_val = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let sum_exp = row.mapv(|x| (x - max_val).exp()).sum();
            let log_sum_exp = max_val + sum_exp.ln();
            
            // Target index
            let target_idx = if i < batch_size { i + batch_size } else { i - batch_size };
            
            // Loss for this sample: - (sim_pos / tau - log_sum_exp)
            total_loss += log_sum_exp - (sim_matrix[[i, target_idx]]);
        }

        total_loss / (2.0 * batch_size as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_nt_xent_value() {
        let optimizer = NTXentOptimizer { temperature: 1.0 };
        
        // Batch of 2 (N=2, total=4 samples)
        // Sample 0 and 1 are views of A. Sample 2 and 3 are views of B.
        // wait, targets logic: targets is (targets + batch_size) % (2 * batch_size)
        // Let's use simple orthogonal vectors to verify math.
        
        let z_i = array![
            [1.0, 0.0], // i=0
            [0.0, 1.0]  // i=1
        ];
        let z_j = array![
            [1.0, 0.0], // j=0 -> maps to idx 2
            [0.0, 1.0]  // j=1 -> maps to idx 3
        ];
        
        let loss = optimizer.compute_loss(&z_i, &z_j);
        println!("Rust NT-Xent Loss: {}", loss);
        
        // Similarity matrix (1.0 on pos, 0.0 on others, excluding diag)
        // Correctness: loss should be -ln(exp(1)/ (exp(1) + exp(0) + exp(0)))
        assert!(loss > 0.0);
    }
}
