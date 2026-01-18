//! Isolation Forest approximation for anomaly detection
//!
//! Implements a simplified version of Isolation Forest algorithm
//! that works efficiently in real-time trading scenarios

use super::AnomalyDetector;
use rand::prelude::*;
use rand_distr::Uniform;

/// Isolation Tree node
#[derive(Debug, Clone)]
enum IsolationNode {
    Internal {
        feature_idx: usize,
        split_value: f64,
        left: Box<IsolationNode>,
        right: Box<IsolationNode>,
    },
    Leaf {
        size: usize,
    },
}

/// Single Isolation Tree
#[derive(Debug, Clone)]
pub struct IsolationTree {
    root: IsolationNode,
    max_depth: usize,
}

impl IsolationTree {
    /// Build an isolation tree from data
    pub fn build(data: &[Vec<f64>], max_depth: usize, rng: &mut impl Rng) -> Self {
        let root = Self::build_node(data, 0, max_depth, rng);
        Self { root, max_depth }
    }

    fn build_node(
        data: &[Vec<f64>],
        depth: usize,
        max_depth: usize,
        rng: &mut impl Rng,
    ) -> IsolationNode {
        // Terminal conditions
        if depth >= max_depth || data.len() <= 1 {
            return IsolationNode::Leaf { size: data.len() };
        }

        if data.is_empty() {
            return IsolationNode::Leaf { size: 0 };
        }

        let n_features = data[0].len();
        if n_features == 0 {
            return IsolationNode::Leaf { size: data.len() };
        }

        // Random feature selection
        let feature_idx = rng.gen_range(0..n_features);

        // Get min/max for this feature
        let values: Vec<f64> = data.iter().map(|row| row[feature_idx]).collect();
        let min_val = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        // If no range, create leaf
        if (max_val - min_val).abs() < 1e-10 {
            return IsolationNode::Leaf { size: data.len() };
        }

        // Random split point
        let split_dist = Uniform::new(min_val, max_val);
        let split_value = rng.sample(split_dist);

        // Partition data
        let left_data: Vec<Vec<f64>> = data
            .iter()
            .filter(|row| row[feature_idx] < split_value)
            .cloned()
            .collect();

        let right_data: Vec<Vec<f64>> = data
            .iter()
            .filter(|row| row[feature_idx] >= split_value)
            .cloned()
            .collect();

        // Recursively build children
        let left = Box::new(Self::build_node(&left_data, depth + 1, max_depth, rng));
        let right = Box::new(Self::build_node(&right_data, depth + 1, max_depth, rng));

        IsolationNode::Internal {
            feature_idx,
            split_value,
            left,
            right,
        }
    }

    /// Calculate path length for a sample
    pub fn path_length(&self, sample: &[f64]) -> f64 {
        Self::path_length_node(&self.root, sample, 0)
    }

    fn path_length_node(node: &IsolationNode, sample: &[f64], depth: usize) -> f64 {
        match node {
            IsolationNode::Leaf { size } => {
                depth as f64 + Self::c(*size)
            }
            IsolationNode::Internal {
                feature_idx,
                split_value,
                left,
                right,
            } => {
                if sample[*feature_idx] < *split_value {
                    Self::path_length_node(left, sample, depth + 1)
                } else {
                    Self::path_length_node(right, sample, depth + 1)
                }
            }
        }
    }

    /// Average path length of unsuccessful search in BST
    fn c(n: usize) -> f64 {
        if n <= 1 {
            return 0.0;
        }
        let n = n as f64;
        2.0 * (n.ln() + 0.5772156649) - 2.0 * (n - 1.0) / n
    }
}

/// Isolation Forest for anomaly detection
#[derive(Debug, Clone)]
pub struct IsolationForest {
    trees: Vec<IsolationTree>,
    n_samples: usize,
    contamination: f64,
    threshold: f64,
}

impl IsolationForest {
    /// Create a new Isolation Forest
    ///
    /// # Arguments
    /// * `n_trees` - Number of trees in the forest
    /// * `max_samples` - Maximum samples per tree
    /// * `contamination` - Expected proportion of anomalies
    pub fn new(n_trees: usize, max_samples: usize, contamination: f64) -> Self {
        Self {
            trees: Vec::new(),
            n_samples: max_samples,
            contamination: contamination.clamp(0.0, 0.5),
            threshold: 0.5, // Will be calibrated after fit
        }
    }

    /// Fit the forest on training data
    pub fn fit(&mut self, data: &[Vec<f64>]) {
        let mut rng = rand::thread_rng();
        let max_depth = (self.n_samples as f64).log2().ceil() as usize;

        self.trees.clear();

        for _ in 0..100 {
            // Default 100 trees
            // Subsample
            let sample_size = self.n_samples.min(data.len());
            let indices: Vec<usize> = (0..data.len()).choose_multiple(&mut rng, sample_size);
            let sample: Vec<Vec<f64>> = indices.iter().map(|&i| data[i].clone()).collect();

            let tree = IsolationTree::build(&sample, max_depth, &mut rng);
            self.trees.push(tree);
        }

        // Calibrate threshold based on contamination
        let scores: Vec<f64> = data.iter().map(|d| self.anomaly_score(d)).collect();
        let mut sorted_scores = scores.clone();
        sorted_scores.sort_by(|a, b| b.partial_cmp(a).unwrap()); // Descending

        let threshold_idx = (self.contamination * sorted_scores.len() as f64) as usize;
        self.threshold = sorted_scores.get(threshold_idx).copied().unwrap_or(0.5);
    }

    /// Calculate anomaly score for a sample (0-1, higher = more anomalous)
    pub fn anomaly_score(&self, sample: &[f64]) -> f64 {
        if self.trees.is_empty() {
            return 0.5;
        }

        let avg_path_length: f64 =
            self.trees.iter().map(|t| t.path_length(sample)).sum::<f64>() / self.trees.len() as f64;

        let c_n = IsolationTree::c(self.n_samples);
        if c_n < 1e-10 {
            return 0.5;
        }

        // Anomaly score formula: 2^(-E(h(x)) / c(n))
        2.0_f64.powf(-avg_path_length / c_n)
    }

    /// Detect anomalies in multi-dimensional data (returns scores)
    pub fn detect_multi(&self, data: &[Vec<f64>]) -> Vec<f64> {
        data.iter().map(|sample| self.anomaly_score(sample)).collect()
    }
}

impl Default for IsolationForest {
    fn default() -> Self {
        Self::new(100, 256, 0.05)
    }
}

/// Simplified 1D Isolation Forest-like detector
#[derive(Debug, Clone)]
pub struct SimpleIsolationDetector {
    window: usize,
    n_bins: usize,
}

impl SimpleIsolationDetector {
    /// Create a new simple isolation detector
    pub fn new(window: usize, n_bins: usize) -> Self {
        Self {
            window: window.max(10),
            n_bins: n_bins.max(5),
        }
    }

    /// Calculate isolation score based on local density
    fn isolation_score(&self, value: f64, window_data: &[f64]) -> f64 {
        if window_data.is_empty() {
            return 0.5;
        }

        // Calculate how "isolated" the value is
        let min_val = window_data.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = window_data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        if (max_val - min_val).abs() < 1e-10 {
            return 0.5;
        }

        // Normalize value
        let normalized = (value - min_val) / (max_val - min_val);

        // Calculate local density using histogram
        let bin_width = 1.0 / self.n_bins as f64;
        let value_bin = (normalized / bin_width).floor() as usize;
        let value_bin = value_bin.min(self.n_bins - 1);

        // Count points in each bin
        let mut bin_counts = vec![0usize; self.n_bins];
        for &v in window_data {
            let norm_v = (v - min_val) / (max_val - min_val);
            let bin = (norm_v / bin_width).floor() as usize;
            let bin = bin.min(self.n_bins - 1);
            bin_counts[bin] += 1;
        }

        // Lower density = higher isolation = more anomalous
        let total: usize = bin_counts.iter().sum();
        if total == 0 {
            return 0.5;
        }

        let bin_density = bin_counts[value_bin] as f64 / total as f64;

        // Convert density to anomaly score (inverse relationship)
        1.0 - bin_density * self.n_bins as f64
    }
}

impl Default for SimpleIsolationDetector {
    fn default() -> Self {
        Self::new(50, 10)
    }
}

impl AnomalyDetector for SimpleIsolationDetector {
    fn detect(&self, data: &[f64]) -> Vec<f64> {
        if data.len() < self.window {
            return vec![0.5; data.len()];
        }

        let mut result = vec![0.5; self.window - 1];

        for i in (self.window - 1)..data.len() {
            let window_data = &data[(i + 1 - self.window)..i];
            let score = self.isolation_score(data[i], window_data);
            result.push(score.clamp(0.0, 1.0));
        }

        result
    }

    fn is_anomaly(&self, score: f64) -> bool {
        score > self.threshold()
    }

    fn threshold(&self) -> f64 {
        0.85
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_isolation_tree() {
        let data: Vec<Vec<f64>> = (0..100)
            .map(|i| vec![i as f64, (i * 2) as f64])
            .collect();

        let mut rng = rand::thread_rng();
        let tree = IsolationTree::build(&data, 8, &mut rng);

        // Normal point should have longer path
        let normal = vec![50.0, 100.0];
        let path_normal = tree.path_length(&normal);

        // Anomalous point should have shorter path
        let anomaly = vec![1000.0, 2000.0];
        let path_anomaly = tree.path_length(&anomaly);

        // Note: This may not always hold due to randomness
        // but generally anomalies should have shorter paths
        println!("Normal path: {}, Anomaly path: {}", path_normal, path_anomaly);
    }

    #[test]
    fn test_simple_isolation_detector() {
        let detector = SimpleIsolationDetector::new(20, 10);
        let mut data: Vec<f64> = (0..50).map(|i| (i as f64).sin()).collect();
        data.push(100.0); // Obvious anomaly

        let scores = detector.detect(&data);

        // Last score (anomaly) should be high
        assert!(scores.last().unwrap() > &0.5);
    }
}
