//! Isolation Forest anomaly detection
//!
//! Implements the Isolation Forest algorithm for unsupervised anomaly detection.
//! Key insight: Anomalies are easier to isolate and require fewer splits.

use super::{AnomalyResult, MultivariateDetector};
use ndarray::{Array1, Array2, Axis};
use rand::prelude::*;

/// A node in an isolation tree
#[derive(Debug, Clone)]
enum IsolationNode {
    /// Internal node with split information
    Internal {
        feature: usize,
        threshold: f64,
        left: Box<IsolationNode>,
        right: Box<IsolationNode>,
    },
    /// Leaf node with size (number of samples)
    Leaf { size: usize },
}

/// Single isolation tree
#[derive(Debug, Clone)]
struct IsolationTree {
    root: IsolationNode,
    max_depth: usize,
}

impl IsolationTree {
    /// Build an isolation tree from data
    fn build(data: &Array2<f64>, max_depth: usize, rng: &mut impl Rng) -> Self {
        let n_samples = data.nrows();
        let root = Self::build_node(data, 0, max_depth, rng);

        Self { root, max_depth }
    }

    /// Recursively build tree nodes
    fn build_node(
        data: &Array2<f64>,
        depth: usize,
        max_depth: usize,
        rng: &mut impl Rng,
    ) -> IsolationNode {
        let n_samples = data.nrows();
        let n_features = data.ncols();

        // Stop conditions: max depth reached or only one sample
        if depth >= max_depth || n_samples <= 1 {
            return IsolationNode::Leaf { size: n_samples };
        }

        // Randomly select a feature
        let feature = rng.gen_range(0..n_features);

        // Get min and max values for the feature
        let col = data.column(feature);
        let min_val = col.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = col.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        // If all values are the same, return leaf
        if (max_val - min_val).abs() < 1e-10 {
            return IsolationNode::Leaf { size: n_samples };
        }

        // Random threshold between min and max
        let threshold = rng.gen_range(min_val..max_val);

        // Split data
        let mut left_indices = Vec::new();
        let mut right_indices = Vec::new();

        for i in 0..n_samples {
            if data[[i, feature]] < threshold {
                left_indices.push(i);
            } else {
                right_indices.push(i);
            }
        }

        // If one side is empty, return leaf
        if left_indices.is_empty() || right_indices.is_empty() {
            return IsolationNode::Leaf { size: n_samples };
        }

        // Create subsets
        let left_data = Array2::from_shape_fn((left_indices.len(), n_features), |(i, j)| {
            data[[left_indices[i], j]]
        });

        let right_data = Array2::from_shape_fn((right_indices.len(), n_features), |(i, j)| {
            data[[right_indices[i], j]]
        });

        // Recursively build children
        let left = Self::build_node(&left_data, depth + 1, max_depth, rng);
        let right = Self::build_node(&right_data, depth + 1, max_depth, rng);

        IsolationNode::Internal {
            feature,
            threshold,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    /// Compute path length for a single sample
    fn path_length(&self, sample: &Array1<f64>) -> f64 {
        Self::path_length_node(&self.root, sample, 0)
    }

    /// Recursive path length computation
    fn path_length_node(node: &IsolationNode, sample: &Array1<f64>, depth: usize) -> f64 {
        match node {
            IsolationNode::Leaf { size } => {
                depth as f64 + Self::c(*size)
            }
            IsolationNode::Internal {
                feature,
                threshold,
                left,
                right,
            } => {
                if sample[*feature] < *threshold {
                    Self::path_length_node(left, sample, depth + 1)
                } else {
                    Self::path_length_node(right, sample, depth + 1)
                }
            }
        }
    }

    /// Average path length in unsuccessful BST search
    fn c(n: usize) -> f64 {
        if n <= 1 {
            0.0
        } else if n == 2 {
            1.0
        } else {
            let n = n as f64;
            2.0 * (n.ln() + 0.5772156649) - 2.0 * (n - 1.0) / n
        }
    }
}

/// Isolation Forest for anomaly detection
#[derive(Clone)]
pub struct IsolationForest {
    /// Number of trees in the forest
    pub n_estimators: usize,
    /// Maximum depth of each tree
    pub max_depth: usize,
    /// Maximum number of samples per tree
    pub max_samples: usize,
    /// Contamination rate (expected proportion of anomalies)
    pub contamination: f64,
    /// Random seed
    pub seed: u64,
    /// Trained trees
    trees: Vec<IsolationTree>,
    /// Threshold for anomaly detection
    threshold: Option<f64>,
}

impl IsolationForest {
    /// Create a new Isolation Forest
    ///
    /// # Arguments
    /// * `n_estimators` - Number of trees (default: 100)
    /// * `contamination` - Expected anomaly rate (default: 0.01)
    pub fn new(n_estimators: usize, contamination: f64) -> Self {
        Self {
            n_estimators,
            max_depth: 0, // Will be set during fit
            max_samples: 256,
            contamination,
            seed: 42,
            trees: Vec::new(),
            threshold: None,
        }
    }

    /// Create with default parameters
    pub fn default_params() -> Self {
        Self::new(100, 0.01)
    }

    /// Set random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Set max samples per tree
    pub fn with_max_samples(mut self, max_samples: usize) -> Self {
        self.max_samples = max_samples;
        self
    }

    /// Compute anomaly scores for samples
    pub fn score_samples(&self, data: &Array2<f64>) -> Array1<f64> {
        let n_samples = data.nrows();
        let mut scores = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let sample = data.row(i).to_owned();
            let avg_path_length: f64 = self
                .trees
                .iter()
                .map(|tree| tree.path_length(&sample))
                .sum::<f64>()
                / self.trees.len() as f64;

            // Anomaly score: 2^(-E[h(x)] / c(n))
            let c = IsolationTree::c(self.max_samples);
            if c > 0.0 {
                scores[i] = 2.0_f64.powf(-avg_path_length / c);
            } else {
                scores[i] = 0.5;
            }
        }

        scores
    }

    /// Compute decision function (lower = more anomalous)
    pub fn decision_function(&self, data: &Array2<f64>) -> Array1<f64> {
        let scores = self.score_samples(data);
        // Convert to decision function: negative = anomaly, positive = normal
        scores.mapv(|s| 0.5 - s)
    }
}

impl MultivariateDetector for IsolationForest {
    fn fit(&mut self, data: &Array2<f64>) {
        let n_samples = data.nrows();
        let actual_samples = self.max_samples.min(n_samples);

        // Set max depth based on sample size
        self.max_depth = (actual_samples as f64).log2().ceil() as usize;

        let mut rng = StdRng::seed_from_u64(self.seed);

        // Build trees
        self.trees = (0..self.n_estimators)
            .map(|_| {
                // Sample data for this tree
                let indices: Vec<usize> = (0..n_samples)
                    .choose_multiple(&mut rng, actual_samples)
                    .into_iter()
                    .collect();

                let sampled_data =
                    Array2::from_shape_fn((indices.len(), data.ncols()), |(i, j)| {
                        data[[indices[i], j]]
                    });

                IsolationTree::build(&sampled_data, self.max_depth, &mut rng)
            })
            .collect();

        // Compute threshold based on contamination
        let scores = self.score_samples(data);
        let mut sorted_scores: Vec<f64> = scores.iter().cloned().collect();
        sorted_scores.sort_by(|a, b| b.partial_cmp(a).unwrap()); // Descending

        let threshold_idx = (n_samples as f64 * self.contamination).ceil() as usize;
        self.threshold = Some(sorted_scores[threshold_idx.min(n_samples - 1)]);
    }

    fn detect(&self, data: &Array2<f64>) -> AnomalyResult {
        if self.trees.is_empty() {
            return AnomalyResult::new(vec![], vec![], vec![]);
        }

        let scores = self.score_samples(data);
        let threshold = self.threshold.unwrap_or(0.5);

        let is_anomaly: Vec<bool> = scores.iter().map(|&s| s > threshold).collect();
        let scores_vec: Vec<f64> = scores.iter().cloned().collect();
        let normalized_scores: Vec<f64> = scores.iter().map(|&s| s / threshold).collect();

        AnomalyResult::new(is_anomaly, scores_vec, normalized_scores)
    }

    fn name(&self) -> &str {
        "IsolationForest"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_isolation_forest_basic() {
        // Create normal data
        let mut rng = StdRng::seed_from_u64(42);
        let n_normal = 100;
        let n_features = 2;

        let mut data = Array2::zeros((n_normal + 2, n_features));

        // Normal points around (0, 0)
        for i in 0..n_normal {
            data[[i, 0]] = rng.gen_range(-1.0..1.0);
            data[[i, 1]] = rng.gen_range(-1.0..1.0);
        }

        // Anomalies far from the cluster
        data[[n_normal, 0]] = 10.0;
        data[[n_normal, 1]] = 10.0;
        data[[n_normal + 1, 0]] = -10.0;
        data[[n_normal + 1, 1]] = -10.0;

        let mut forest = IsolationForest::new(50, 0.02);
        forest.fit(&data);
        let result = forest.detect(&data);

        // The last two points should have high anomaly scores
        assert!(result.scores[n_normal] > result.scores[0]);
        assert!(result.scores[n_normal + 1] > result.scores[0]);
    }

    #[test]
    fn test_c_function() {
        assert_eq!(IsolationTree::c(1), 0.0);
        assert_eq!(IsolationTree::c(2), 1.0);
        assert!(IsolationTree::c(100) > IsolationTree::c(10));
    }
}
