use ndarray::{Array2, Axis};
use rayon::prelude::*;

/// Selects the Top-K hardest negatives for each anchor in the batch.
/// Hard negatives are samples with the highest cosine similarity that are NOT the positive match.
pub struct HardMiner {
    pub top_k: usize,
}

impl HardMiner {
    pub fn new(top_k: usize) -> Self {
        Self { top_k }
    }

    /// Finds the indices of Top-K hardest negatives for each anchor.
    /// v_anchor: (N, D) - batch of anchor embeddings
    /// v_candidates: (N, D) - batch of candidate embeddings
    /// Returns: Vec<Vec<usize>> (N rows, each containing Top-K indices)
    pub fn mine(&self, v_anchor: &Array2<f32>, v_candidates: &Array2<f32>) -> Vec<Vec<usize>> {
        let n = v_anchor.nrows();
        
        // Normalize rows to unit length for cosine similarity calculation
        let norm_anchor = self.normalize_rows(v_anchor);
        let norm_candidates = self.normalize_rows(v_candidates);

        // Compute similarity matrix: N x N
        // In Rust, we can compute this row by row in parallel
        (0..n).into_par_iter().map(|i| {
            let row_anchor = norm_anchor.row(i);
            let mut similarities: Vec<(usize, f32)> = (0..n)
                .map(|j| {
                    if i == j {
                        // Mask positive match (diagonal) with lowest possible similarity
                        (j, -1.0)
                    } else {
                        let row_candidate = norm_candidates.row(j);
                        let sim = row_anchor.dot(&row_candidate);
                        (j, sim)
                    }
                })
                .collect();

            // Sort by similarity descending
            similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // Take Top-K indices
            similarities.iter().take(self.top_k).map(|&(idx, _)| idx).collect()
        }).collect()
    }

    fn normalize_rows(&self, matrix: &Array2<f32>) -> Array2<f32> {
        let mut norm_matrix = matrix.clone();
        for mut row in norm_matrix.axis_iter_mut(Axis(0)) {
            let norm = row.dot(&row).sqrt();
            if norm > 1e-9 {
                row.mapv_inplace(|x| x / norm);
            }
        }
        norm_matrix
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_mining_logic() {
        let miner = HardMiner::new(2);
        
        // Let's create a small batch
        // Row 0 is the anchor.
        // Row 1 is the positive match (identically 1.0 similarity if normalized).
        // Row 2 is a "hard" negative (very close to row 0).
        // Row 3 is an "easy" negative (far from row 0).
        
        let batch = array![
            [1.0, 0.0, 0.0], // Anchor 0
            [1.0, 0.0, 0.0], // Anchor 1 (Positive match for 0)
            [0.9, 0.1, 0.0], // Anchor 2 (Hard negative for 0)
            [0.0, 0.0, 1.0], // Anchor 3 (Easy negative for 0)
        ];

        let results = miner.mine(&batch, &batch);
        
        // For Anchor 0:
        // Index 1: sim = 1.0 (Hardest Negative)
        // Index 2: sim = 0.9 (Hard)
        // Index 3: sim = 0.0 (Easy)
        // Expected Top-2 for Anchor 0: [1, 2] (since 0 is masked)
        
        assert_eq!(results[0][0], 1);
        assert_eq!(results[0][1], 2);
        
        println!("Rust Miner results for Anchor 0: {:?}", results[0]);
    }
}
