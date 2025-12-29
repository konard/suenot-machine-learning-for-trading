//! Decision Tree implementation

use crate::data::Dataset;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

/// Decision tree configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreeConfig {
    /// Maximum depth of tree
    pub max_depth: usize,
    /// Minimum samples required to split
    pub min_samples_split: usize,
    /// Minimum samples in leaf node
    pub min_samples_leaf: usize,
    /// Maximum features to consider for split (None = all)
    pub max_features: Option<usize>,
    /// Random seed for reproducibility
    pub seed: u64,
    /// Task type: "regression" or "classification"
    pub task: TaskType,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum TaskType {
    Regression,
    Classification,
}

impl Default for TreeConfig {
    fn default() -> Self {
        Self {
            max_depth: 10,
            min_samples_split: 5,
            min_samples_leaf: 2,
            max_features: None,
            seed: 42,
            task: TaskType::Regression,
        }
    }
}

/// Tree node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreeNode {
    /// Feature index for split
    pub feature_idx: Option<usize>,
    /// Threshold for split
    pub threshold: Option<f64>,
    /// Prediction value (for leaf nodes)
    pub value: f64,
    /// Class probabilities (for classification)
    pub class_probs: Option<Vec<f64>>,
    /// Number of samples in this node
    pub n_samples: usize,
    /// Left child
    pub left: Option<Box<TreeNode>>,
    /// Right child
    pub right: Option<Box<TreeNode>>,
    /// Impurity at this node
    pub impurity: f64,
}

impl TreeNode {
    fn leaf(value: f64, n_samples: usize, impurity: f64) -> Self {
        Self {
            feature_idx: None,
            threshold: None,
            value,
            class_probs: None,
            n_samples,
            left: None,
            right: None,
            impurity,
        }
    }

    fn leaf_classification(class_probs: Vec<f64>, n_samples: usize, impurity: f64) -> Self {
        let value = class_probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i as f64)
            .unwrap_or(0.0);

        Self {
            feature_idx: None,
            threshold: None,
            value,
            class_probs: Some(class_probs),
            n_samples,
            left: None,
            right: None,
            impurity,
        }
    }

    pub fn is_leaf(&self) -> bool {
        self.left.is_none() && self.right.is_none()
    }

    pub fn depth(&self) -> usize {
        if self.is_leaf() {
            1
        } else {
            1 + self
                .left
                .as_ref()
                .map(|n| n.depth())
                .unwrap_or(0)
                .max(self.right.as_ref().map(|n| n.depth()).unwrap_or(0))
        }
    }

    pub fn n_leaves(&self) -> usize {
        if self.is_leaf() {
            1
        } else {
            self.left.as_ref().map(|n| n.n_leaves()).unwrap_or(0)
                + self.right.as_ref().map(|n| n.n_leaves()).unwrap_or(0)
        }
    }
}

/// Decision Tree model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionTree {
    config: TreeConfig,
    root: Option<TreeNode>,
    feature_names: Vec<String>,
    feature_importances: Vec<f64>,
}

impl DecisionTree {
    /// Create a new decision tree with config
    pub fn new(config: TreeConfig) -> Self {
        Self {
            config,
            root: None,
            feature_names: Vec::new(),
            feature_importances: Vec::new(),
        }
    }

    /// Create with default config
    pub fn default_regression() -> Self {
        Self::new(TreeConfig {
            task: TaskType::Regression,
            ..Default::default()
        })
    }

    pub fn default_classification() -> Self {
        Self::new(TreeConfig {
            task: TaskType::Classification,
            ..Default::default()
        })
    }

    /// Train the decision tree
    pub fn fit(&mut self, dataset: &Dataset) {
        self.feature_names = dataset.feature_names.clone();
        let n_features = dataset.n_features();
        self.feature_importances = vec![0.0; n_features];

        let indices: Vec<usize> = (0..dataset.n_samples()).collect();
        let mut rng = ChaCha8Rng::seed_from_u64(self.config.seed);

        self.root = Some(self.build_tree(dataset, &indices, 0, &mut rng));

        // Normalize feature importances
        let sum: f64 = self.feature_importances.iter().sum();
        if sum > 0.0 {
            for imp in &mut self.feature_importances {
                *imp /= sum;
            }
        }
    }

    /// Build tree recursively
    fn build_tree(
        &mut self,
        dataset: &Dataset,
        indices: &[usize],
        depth: usize,
        rng: &mut ChaCha8Rng,
    ) -> TreeNode {
        let n = indices.len();
        let labels: Vec<f64> = indices.iter().map(|&i| dataset.labels[i]).collect();

        // Calculate impurity
        let impurity = match self.config.task {
            TaskType::Regression => self.mse(&labels),
            TaskType::Classification => self.gini(&labels),
        };

        // Check stopping conditions
        if depth >= self.config.max_depth
            || n < self.config.min_samples_split
            || impurity < 1e-10
        {
            return self.create_leaf(&labels, impurity);
        }

        // Find best split
        let best_split = self.find_best_split(dataset, indices, rng);

        match best_split {
            Some((feature_idx, threshold, left_indices, right_indices, importance)) => {
                if left_indices.len() < self.config.min_samples_leaf
                    || right_indices.len() < self.config.min_samples_leaf
                {
                    return self.create_leaf(&labels, impurity);
                }

                // Update feature importance
                self.feature_importances[feature_idx] += importance;

                // Build children
                let left = self.build_tree(dataset, &left_indices, depth + 1, rng);
                let right = self.build_tree(dataset, &right_indices, depth + 1, rng);

                TreeNode {
                    feature_idx: Some(feature_idx),
                    threshold: Some(threshold),
                    value: self.mean(&labels),
                    class_probs: None,
                    n_samples: n,
                    left: Some(Box::new(left)),
                    right: Some(Box::new(right)),
                    impurity,
                }
            }
            None => self.create_leaf(&labels, impurity),
        }
    }

    /// Create a leaf node
    fn create_leaf(&self, labels: &[f64], impurity: f64) -> TreeNode {
        match self.config.task {
            TaskType::Regression => TreeNode::leaf(self.mean(labels), labels.len(), impurity),
            TaskType::Classification => {
                let probs = self.class_probabilities(labels);
                TreeNode::leaf_classification(probs, labels.len(), impurity)
            }
        }
    }

    /// Find the best split
    fn find_best_split(
        &self,
        dataset: &Dataset,
        indices: &[usize],
        rng: &mut ChaCha8Rng,
    ) -> Option<(usize, f64, Vec<usize>, Vec<usize>, f64)> {
        let n_features = dataset.n_features();
        let max_features = self.config.max_features.unwrap_or(n_features);

        // Select features to consider
        let mut feature_indices: Vec<usize> = (0..n_features).collect();
        feature_indices.shuffle(rng);
        feature_indices.truncate(max_features);

        let mut best_gain = 0.0;
        let mut best_split: Option<(usize, f64, Vec<usize>, Vec<usize>, f64)> = None;

        let labels: Vec<f64> = indices.iter().map(|&i| dataset.labels[i]).collect();
        let parent_impurity = match self.config.task {
            TaskType::Regression => self.mse(&labels),
            TaskType::Classification => self.gini(&labels),
        };

        for &feature_idx in &feature_indices {
            // Get unique values for this feature
            let mut values: Vec<f64> = indices
                .iter()
                .map(|&i| dataset.features[i][feature_idx])
                .collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap());
            values.dedup();

            // Try midpoints as thresholds
            for window in values.windows(2) {
                let threshold = (window[0] + window[1]) / 2.0;

                let (left_idx, right_idx): (Vec<usize>, Vec<usize>) = indices
                    .iter()
                    .partition(|&&i| dataset.features[i][feature_idx] <= threshold);

                if left_idx.is_empty() || right_idx.is_empty() {
                    continue;
                }

                let left_labels: Vec<f64> = left_idx.iter().map(|&i| dataset.labels[i]).collect();
                let right_labels: Vec<f64> = right_idx.iter().map(|&i| dataset.labels[i]).collect();

                let left_impurity = match self.config.task {
                    TaskType::Regression => self.mse(&left_labels),
                    TaskType::Classification => self.gini(&left_labels),
                };

                let right_impurity = match self.config.task {
                    TaskType::Regression => self.mse(&right_labels),
                    TaskType::Classification => self.gini(&right_labels),
                };

                let n_left = left_idx.len() as f64;
                let n_right = right_idx.len() as f64;
                let n_total = n_left + n_right;

                let weighted_impurity =
                    (n_left * left_impurity + n_right * right_impurity) / n_total;
                let gain = parent_impurity - weighted_impurity;

                if gain > best_gain {
                    best_gain = gain;
                    let importance = gain * indices.len() as f64;
                    best_split = Some((feature_idx, threshold, left_idx, right_idx, importance));
                }
            }
        }

        best_split
    }

    /// Predict for a single sample
    pub fn predict_one(&self, features: &[f64]) -> f64 {
        match &self.root {
            Some(node) => self.traverse(node, features),
            None => 0.0,
        }
    }

    /// Predict probabilities for classification
    pub fn predict_proba_one(&self, features: &[f64]) -> Vec<f64> {
        match &self.root {
            Some(node) => self.traverse_proba(node, features),
            None => vec![0.5, 0.5],
        }
    }

    fn traverse(&self, node: &TreeNode, features: &[f64]) -> f64 {
        if node.is_leaf() {
            return node.value;
        }

        let feature_idx = node.feature_idx.unwrap();
        let threshold = node.threshold.unwrap();

        if features[feature_idx] <= threshold {
            self.traverse(node.left.as_ref().unwrap(), features)
        } else {
            self.traverse(node.right.as_ref().unwrap(), features)
        }
    }

    fn traverse_proba(&self, node: &TreeNode, features: &[f64]) -> Vec<f64> {
        if node.is_leaf() {
            return node.class_probs.clone().unwrap_or(vec![0.5, 0.5]);
        }

        let feature_idx = node.feature_idx.unwrap();
        let threshold = node.threshold.unwrap();

        if features[feature_idx] <= threshold {
            self.traverse_proba(node.left.as_ref().unwrap(), features)
        } else {
            self.traverse_proba(node.right.as_ref().unwrap(), features)
        }
    }

    /// Predict for multiple samples
    pub fn predict(&self, dataset: &Dataset) -> Vec<f64> {
        dataset
            .features
            .iter()
            .map(|f| self.predict_one(f))
            .collect()
    }

    /// Get feature importances
    pub fn feature_importances(&self) -> &[f64] {
        &self.feature_importances
    }

    /// Get feature names with importances
    pub fn feature_importance_map(&self) -> Vec<(&str, f64)> {
        self.feature_names
            .iter()
            .zip(self.feature_importances.iter())
            .map(|(n, &i)| (n.as_str(), i))
            .collect()
    }

    /// Print tree structure
    pub fn print_tree(&self) {
        if let Some(ref root) = self.root {
            self.print_node(root, 0);
        }
    }

    fn print_node(&self, node: &TreeNode, indent: usize) {
        let prefix = "  ".repeat(indent);

        if node.is_leaf() {
            println!(
                "{}Leaf: value={:.4}, samples={}",
                prefix, node.value, node.n_samples
            );
        } else {
            let feature_name = node
                .feature_idx
                .and_then(|i| self.feature_names.get(i))
                .map(|s| s.as_str())
                .unwrap_or("?");

            println!(
                "{}Split: {} <= {:.4} (samples={}, impurity={:.4})",
                prefix,
                feature_name,
                node.threshold.unwrap_or(0.0),
                node.n_samples,
                node.impurity
            );

            if let Some(ref left) = node.left {
                println!("{}Left:", prefix);
                self.print_node(left, indent + 1);
            }
            if let Some(ref right) = node.right {
                println!("{}Right:", prefix);
                self.print_node(right, indent + 1);
            }
        }
    }

    // Helper functions

    fn mean(&self, values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        values.iter().sum::<f64>() / values.len() as f64
    }

    fn mse(&self, values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        let mean = self.mean(values);
        values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64
    }

    fn gini(&self, values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }

        let n = values.len() as f64;
        let n_positive = values.iter().filter(|&&v| v > 0.0).count() as f64;
        let p = n_positive / n;

        2.0 * p * (1.0 - p)
    }

    fn class_probabilities(&self, values: &[f64]) -> Vec<f64> {
        if values.is_empty() {
            return vec![0.5, 0.5];
        }

        let n = values.len() as f64;
        let n_positive = values.iter().filter(|&&v| v > 0.0).count() as f64;

        vec![1.0 - n_positive / n, n_positive / n]
    }

    /// Calculate accuracy for classification
    pub fn accuracy(&self, dataset: &Dataset) -> f64 {
        let predictions = self.predict(dataset);
        let correct: usize = predictions
            .iter()
            .zip(dataset.labels.iter())
            .filter(|(&pred, &label)| {
                let pred_class = if pred > 0.5 { 1.0 } else { 0.0 };
                let label_class = if label > 0.0 { 1.0 } else { 0.0 };
                pred_class == label_class
            })
            .count();

        correct as f64 / dataset.n_samples() as f64
    }

    /// Calculate MSE for regression
    pub fn mse_score(&self, dataset: &Dataset) -> f64 {
        let predictions = self.predict(dataset);
        predictions
            .iter()
            .zip(dataset.labels.iter())
            .map(|(p, l)| (p - l).powi(2))
            .sum::<f64>()
            / dataset.n_samples() as f64
    }

    /// Calculate R² score
    pub fn r2_score(&self, dataset: &Dataset) -> f64 {
        let predictions = self.predict(dataset);
        let mean_label = dataset.labels.iter().sum::<f64>() / dataset.n_samples() as f64;

        let ss_res: f64 = predictions
            .iter()
            .zip(dataset.labels.iter())
            .map(|(p, l)| (l - p).powi(2))
            .sum();

        let ss_tot: f64 = dataset.labels.iter().map(|l| (l - mean_label).powi(2)).sum();

        if ss_tot == 0.0 {
            0.0
        } else {
            1.0 - ss_res / ss_tot
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decision_tree_regression() {
        let mut dataset = Dataset::new(vec!["x".to_string()]);

        // Simple linear relationship
        for i in 0..100 {
            let x = i as f64 / 10.0;
            let y = 2.0 * x + 1.0;
            dataset.add_sample(vec![x], y, i as i64);
        }

        let mut tree = DecisionTree::default_regression();
        tree.fit(&dataset);

        let predictions = tree.predict(&dataset);
        assert!(!predictions.is_empty());
    }

    #[test]
    fn test_decision_tree_classification() {
        let mut dataset = Dataset::new(vec!["x".to_string()]);

        for i in 0..100 {
            let x = i as f64 / 10.0;
            let y = if x > 5.0 { 1.0 } else { 0.0 };
            dataset.add_sample(vec![x], y, i as i64);
        }

        let mut tree = DecisionTree::default_classification();
        tree.fit(&dataset);

        let accuracy = tree.accuracy(&dataset);
        assert!(accuracy > 0.9);
    }
}
