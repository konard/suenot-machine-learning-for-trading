//! Simple decision tree implementation for ensemble building.

use anyhow::Result;
use ndarray::{Array1, Array2, Axis};
use rand::prelude::*;
use rand::SeedableRng;

/// A node in the decision tree
#[derive(Debug, Clone)]
pub enum TreeNode {
    /// Leaf node with prediction value and variance
    Leaf {
        value: f64,
        variance: f64,
        n_samples: usize,
    },
    /// Internal node with split information
    Internal {
        feature_index: usize,
        threshold: f64,
        left: Box<TreeNode>,
        right: Box<TreeNode>,
    },
}

/// Decision tree for regression
#[derive(Debug, Clone)]
pub struct DecisionTree {
    root: Option<TreeNode>,
    max_depth: Option<usize>,
    min_samples_split: usize,
    min_samples_leaf: usize,
    max_features: usize,
    random_state: Option<u64>,
}

impl DecisionTree {
    /// Create a new decision tree
    pub fn new(
        max_depth: Option<usize>,
        min_samples_split: usize,
        min_samples_leaf: usize,
        max_features: usize,
        random_state: Option<u64>,
    ) -> Self {
        Self {
            root: None,
            max_depth,
            min_samples_split,
            min_samples_leaf,
            max_features,
            random_state,
        }
    }

    /// Fit the tree to data
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        let indices: Vec<usize> = (0..x.nrows()).collect();
        let mut rng = match self.random_state {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };

        self.root = Some(self.build_tree(x, y, &indices, 0, &mut rng));
        Ok(())
    }

    /// Build tree recursively
    fn build_tree(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        indices: &[usize],
        depth: usize,
        rng: &mut StdRng,
    ) -> TreeNode {
        let n_samples = indices.len();

        // Calculate mean and variance for this node
        let sum: f64 = indices.iter().map(|&i| y[i]).sum();
        let mean = sum / n_samples as f64;
        let variance: f64 = indices.iter().map(|&i| (y[i] - mean).powi(2)).sum::<f64>()
            / n_samples as f64;

        // Check stopping conditions
        let should_stop = n_samples < self.min_samples_split
            || self.max_depth.map_or(false, |d| depth >= d)
            || variance < 1e-10;

        if should_stop {
            return TreeNode::Leaf {
                value: mean,
                variance,
                n_samples,
            };
        }

        // Find best split
        let n_features = x.ncols();
        let features_to_try: Vec<usize> = {
            let mut all_features: Vec<usize> = (0..n_features).collect();
            all_features.shuffle(rng);
            all_features.into_iter().take(self.max_features).collect()
        };

        let mut best_feature = 0;
        let mut best_threshold = 0.0;
        let mut best_gain = f64::NEG_INFINITY;
        let mut best_left_indices = Vec::new();
        let mut best_right_indices = Vec::new();

        for &feature in &features_to_try {
            // Get unique values for this feature
            let mut values: Vec<f64> = indices.iter().map(|&i| x[[i, feature]]).collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            values.dedup();

            // Try thresholds between consecutive values
            for window in values.windows(2) {
                let threshold = (window[0] + window[1]) / 2.0;

                let (left_idx, right_idx): (Vec<usize>, Vec<usize>) = indices
                    .iter()
                    .copied()
                    .partition(|&i| x[[i, feature]] <= threshold);

                // Check minimum leaf size
                if left_idx.len() < self.min_samples_leaf
                    || right_idx.len() < self.min_samples_leaf
                {
                    continue;
                }

                // Calculate information gain (variance reduction)
                let left_mean: f64 = left_idx.iter().map(|&i| y[i]).sum::<f64>()
                    / left_idx.len() as f64;
                let right_mean: f64 = right_idx.iter().map(|&i| y[i]).sum::<f64>()
                    / right_idx.len() as f64;

                let left_var: f64 = left_idx.iter()
                    .map(|&i| (y[i] - left_mean).powi(2))
                    .sum::<f64>() / left_idx.len() as f64;
                let right_var: f64 = right_idx.iter()
                    .map(|&i| (y[i] - right_mean).powi(2))
                    .sum::<f64>() / right_idx.len() as f64;

                let left_weight = left_idx.len() as f64 / n_samples as f64;
                let right_weight = right_idx.len() as f64 / n_samples as f64;
                let weighted_var = left_weight * left_var + right_weight * right_var;
                let gain = variance - weighted_var;

                if gain > best_gain {
                    best_gain = gain;
                    best_feature = feature;
                    best_threshold = threshold;
                    best_left_indices = left_idx;
                    best_right_indices = right_idx;
                }
            }
        }

        // If no good split found, return leaf
        if best_gain <= 0.0 || best_left_indices.is_empty() || best_right_indices.is_empty() {
            return TreeNode::Leaf {
                value: mean,
                variance,
                n_samples,
            };
        }

        // Build child nodes
        let left = self.build_tree(x, y, &best_left_indices, depth + 1, rng);
        let right = self.build_tree(x, y, &best_right_indices, depth + 1, rng);

        TreeNode::Internal {
            feature_index: best_feature,
            threshold: best_threshold,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    /// Predict for a single sample
    pub fn predict_single(&self, x: &Array1<f64>) -> (f64, f64) {
        match &self.root {
            Some(node) => self.traverse(node, x),
            None => (0.0, 0.0),
        }
    }

    /// Traverse tree to find prediction
    fn traverse(&self, node: &TreeNode, x: &Array1<f64>) -> (f64, f64) {
        match node {
            TreeNode::Leaf { value, variance, .. } => (*value, *variance),
            TreeNode::Internal {
                feature_index,
                threshold,
                left,
                right,
            } => {
                if x[*feature_index] <= *threshold {
                    self.traverse(left, x)
                } else {
                    self.traverse(right, x)
                }
            }
        }
    }

    /// Predict for multiple samples
    pub fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        let mut predictions = Array1::zeros(x.nrows());
        for (i, row) in x.axis_iter(Axis(0)).enumerate() {
            let (pred, _) = self.predict_single(&row.to_owned());
            predictions[i] = pred;
        }
        predictions
    }

    /// Predict with leaf variance
    pub fn predict_with_variance(&self, x: &Array2<f64>) -> (Array1<f64>, Array1<f64>) {
        let mut predictions = Array1::zeros(x.nrows());
        let mut variances = Array1::zeros(x.nrows());
        for (i, row) in x.axis_iter(Axis(0)).enumerate() {
            let (pred, var) = self.predict_single(&row.to_owned());
            predictions[i] = pred;
            variances[i] = var;
        }
        (predictions, variances)
    }
}
