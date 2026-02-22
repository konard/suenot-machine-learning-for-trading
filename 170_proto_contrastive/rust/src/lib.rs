use ndarray::{Array2, Axis, Array1};
use rayon::prelude::*;

/// Optimized Prototypical Centroid Calculator.
pub struct ProtoCentroidCalculator;

impl ProtoCentroidCalculator {
    /// Computes centroids (prototypes) for each cluster given features and labels.
    /// features: (N, D) matrix
    /// labels: (N) vector of cluster indices [0, K-1]
    /// num_clusters: K
    pub fn compute_centroids(
        features: &Array2<f32>,
        labels: &[usize],
        num_clusters: usize,
    ) -> Array2<f32> {
        let dim = features.ncols();
        let mut centroids = Array2::zeros((num_clusters, dim));
        let mut counts = vec![0.0; num_clusters];

        // 1. Accumulate features for each cluster
        // We do this sequentially to avoid mutex overhead on centroids
        for (i, row) in features.axis_iter(Axis(0)).enumerate() {
            let label = labels[i];
            if label < num_clusters {
                let mut centroid_row = centroids.row_mut(label);
                centroid_row += &row;
                counts[label] += 1.0;
            }
        }

        // 2. Divide by counts to get means (prototypes)
        for i in 0..num_clusters {
            if counts[i] > 0.0 {
                let mut centroid_row = centroids.row_mut(i);
                centroid_row /= counts[i];
            }
        }

        centroids
    }

    /// Computes Euclidean distances between features and prototypes.
    /// returns: (N, K) distance matrix
    pub fn compute_distances(features: &Array2<f32>, prototypes: &Array2<f32>) -> Array2<f32> {
        let n = features.nrows();
        let k = prototypes.nrows();
        let mut distances = Array2::zeros((n, k));

        // Parallelize over samples
        distances.axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(i, mut dist_row)| {
                let feat_row = features.row(i);
                for j in 0..k {
                    let proto_row = prototypes.row(j);
                    let diff = &feat_row - &proto_row;
                    dist_row[j] = diff.dot(&diff).sqrt();
                }
            });

        distances
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_centroid_calculation() {
        let features = array![
            [1.0, 0.0],
            [2.0, 0.0],
            [0.0, 10.0],
            [0.0, 20.0]
        ];
        let labels = vec![0, 0, 1, 1];
        let centroids = ProtoCentroidCalculator::compute_centroids(&features, &labels, 2);
        
        // Expected: [1.5, 0.0] and [0.0, 15.0]
        assert_eq!(centroids[[0, 0]], 1.5);
        assert_eq!(centroids[[1, 1]], 15.0);
    }

    #[test]
    fn test_distance_calculation() {
        let features = array![[3.0, 4.0]];
        let prototypes = array![[0.0, 0.0]];
        let dist = ProtoCentroidCalculator::compute_distances(&features, &prototypes);
        
        // Expected: 5.0 (Euclidean distance from origin)
        assert_eq!(dist[[0, 0]], 5.0);
    }
}
