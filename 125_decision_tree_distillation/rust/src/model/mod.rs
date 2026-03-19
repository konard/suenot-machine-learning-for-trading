//! Decision Tree Model Implementation
//!
//! This module provides a decision tree implementation optimized for
//! knowledge distillation in trading applications.

use ndarray::{Array1, Array2, ArrayView1};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for decision tree distillation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistillationConfig {
    /// Maximum depth of the tree
    pub max_depth: usize,
    /// Minimum samples required in a leaf node
    pub min_samples_leaf: usize,
    /// Minimum samples required to split a node
    pub min_samples_split: usize,
    /// Feature names for rule extraction
    pub feature_names: Option<Vec<String>>,
    /// Class names for classification
    pub class_names: Option<Vec<String>>,
}

impl Default for DistillationConfig {
    fn default() -> Self {
        Self {
            max_depth: 5,
            min_samples_leaf: 50,
            min_samples_split: 100,
            feature_names: None,
            class_names: Some(vec!["DOWN".to_string(), "UP".to_string()]),
        }
    }
}

/// A node in the decision tree
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreeNode {
    /// Feature index for split (None for leaf)
    pub feature_idx: Option<usize>,
    /// Threshold for split
    pub threshold: Option<f64>,
    /// Prediction value (for leaves)
    pub value: f64,
    /// Confidence/probability
    pub confidence: f64,
    /// Number of samples at this node
    pub n_samples: usize,
    /// Left child index
    pub left: Option<usize>,
    /// Right child index
    pub right: Option<usize>,
    /// Depth in the tree
    pub depth: usize,
}

impl TreeNode {
    /// Check if this node is a leaf
    pub fn is_leaf(&self) -> bool {
        self.feature_idx.is_none()
    }
}

/// A rule extracted from the decision tree
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistilledRule {
    /// Conditions leading to this rule
    pub conditions: Vec<String>,
    /// Prediction (class name or value)
    pub prediction: String,
    /// Confidence of the prediction
    pub confidence: f64,
    /// Number of training samples
    pub n_samples: usize,
    /// Node indices in the path
    pub path: Vec<usize>,
}

impl std::fmt::Display for DistilledRule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let conditions = if self.conditions.is_empty() {
            "TRUE".to_string()
        } else {
            self.conditions.join(" AND ")
        };
        write!(
            f,
            "IF {} THEN {} (conf: {:.2}, n={})",
            conditions, self.prediction, self.confidence, self.n_samples
        )
    }
}

/// Decision Tree for distillation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionTree {
    config: DistillationConfig,
    nodes: Vec<TreeNode>,
    fitted: bool,
    n_features: usize,
}

impl DecisionTree {
    /// Create a new decision tree with the given configuration
    pub fn new(config: DistillationConfig) -> Self {
        Self {
            config,
            nodes: Vec::new(),
            fitted: false,
            n_features: 0,
        }
    }

    /// Fit the decision tree on soft labels from a teacher model
    pub fn fit(&mut self, features: &Array2<f64>, soft_labels: &Array1<f64>) -> crate::Result<()> {
        if features.nrows() != soft_labels.len() {
            return Err(crate::DistillationError::DataError(
                "Features and labels must have same number of samples".to_string(),
            ));
        }

        self.n_features = features.ncols();
        self.nodes.clear();

        // Build tree recursively
        let indices: Vec<usize> = (0..features.nrows()).collect();
        self.build_node(features, soft_labels, &indices, 0);

        self.fitted = true;
        Ok(())
    }

    fn build_node(
        &mut self,
        features: &Array2<f64>,
        soft_labels: &Array1<f64>,
        indices: &[usize],
        depth: usize,
    ) -> usize {
        let n_samples = indices.len();
        let node_idx = self.nodes.len();

        // Calculate mean prediction and confidence
        let values: Vec<f64> = indices.iter().map(|&i| soft_labels[i]).collect();
        let mean_value = values.iter().sum::<f64>() / n_samples as f64;
        let confidence = if mean_value > 0.5 {
            mean_value
        } else {
            1.0 - mean_value
        };

        // Check stopping conditions
        if depth >= self.config.max_depth
            || n_samples < self.config.min_samples_split
            || n_samples < 2 * self.config.min_samples_leaf
        {
            // Create leaf node
            self.nodes.push(TreeNode {
                feature_idx: None,
                threshold: None,
                value: mean_value,
                confidence,
                n_samples,
                left: None,
                right: None,
                depth,
            });
            return node_idx;
        }

        // Find best split
        if let Some((best_feature, best_threshold, _best_impurity)) =
            self.find_best_split(features, soft_labels, indices)
        {
            // Split indices
            let (left_indices, right_indices): (Vec<usize>, Vec<usize>) = indices
                .iter()
                .partition(|&&i| features[[i, best_feature]] <= best_threshold);

            // Check minimum samples constraint
            if left_indices.len() < self.config.min_samples_leaf
                || right_indices.len() < self.config.min_samples_leaf
            {
                // Create leaf node
                self.nodes.push(TreeNode {
                    feature_idx: None,
                    threshold: None,
                    value: mean_value,
                    confidence,
                    n_samples,
                    left: None,
                    right: None,
                    depth,
                });
                return node_idx;
            }

            // Create internal node (placeholder)
            self.nodes.push(TreeNode {
                feature_idx: Some(best_feature),
                threshold: Some(best_threshold),
                value: mean_value,
                confidence,
                n_samples,
                left: None,
                right: None,
                depth,
            });

            // Build children
            let left_idx = self.build_node(features, soft_labels, &left_indices, depth + 1);
            let right_idx = self.build_node(features, soft_labels, &right_indices, depth + 1);

            // Update parent with children indices
            self.nodes[node_idx].left = Some(left_idx);
            self.nodes[node_idx].right = Some(right_idx);

            node_idx
        } else {
            // No valid split found, create leaf
            self.nodes.push(TreeNode {
                feature_idx: None,
                threshold: None,
                value: mean_value,
                confidence,
                n_samples,
                left: None,
                right: None,
                depth,
            });
            node_idx
        }
    }

    fn find_best_split(
        &self,
        features: &Array2<f64>,
        soft_labels: &Array1<f64>,
        indices: &[usize],
    ) -> Option<(usize, f64, f64)> {
        let mut best_feature = 0;
        let mut best_threshold = 0.0;
        let mut best_impurity = f64::MAX;

        for feature_idx in 0..self.n_features {
            // Get unique values for this feature
            let mut values: Vec<f64> = indices
                .iter()
                .map(|&i| features[[i, feature_idx]])
                .collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap());
            values.dedup();

            // Try each midpoint as threshold
            for i in 0..values.len().saturating_sub(1) {
                let threshold = (values[i] + values[i + 1]) / 2.0;

                // Calculate impurity reduction
                let (left, right): (Vec<usize>, Vec<usize>) = indices
                    .iter()
                    .partition(|&&idx| features[[idx, feature_idx]] <= threshold);

                if left.len() < self.config.min_samples_leaf
                    || right.len() < self.config.min_samples_leaf
                {
                    continue;
                }

                let impurity = self.calculate_impurity(soft_labels, &left, &right);

                if impurity < best_impurity {
                    best_impurity = impurity;
                    best_feature = feature_idx;
                    best_threshold = threshold;
                }
            }
        }

        if best_impurity < f64::MAX {
            Some((best_feature, best_threshold, best_impurity))
        } else {
            None
        }
    }

    fn calculate_impurity(
        &self,
        soft_labels: &Array1<f64>,
        left: &[usize],
        right: &[usize],
    ) -> f64 {
        let n_total = (left.len() + right.len()) as f64;

        let left_mse = self.calculate_mse(soft_labels, left);
        let right_mse = self.calculate_mse(soft_labels, right);

        (left.len() as f64 / n_total) * left_mse + (right.len() as f64 / n_total) * right_mse
    }

    fn calculate_mse(&self, soft_labels: &Array1<f64>, indices: &[usize]) -> f64 {
        if indices.is_empty() {
            return 0.0;
        }

        let values: Vec<f64> = indices.iter().map(|&i| soft_labels[i]).collect();
        let mean = values.iter().sum::<f64>() / values.len() as f64;

        values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64
    }

    /// Make predictions for new samples
    pub fn predict(&self, features: &Array2<f64>) -> crate::Result<Array1<f64>> {
        if !self.fitted {
            return Err(crate::DistillationError::NotFitted);
        }

        let mut predictions = Vec::with_capacity(features.nrows());

        for i in 0..features.nrows() {
            let row = features.row(i);
            let pred = self.predict_single(&row);
            predictions.push(pred);
        }

        Ok(Array1::from_vec(predictions))
    }

    fn predict_single(&self, features: &ArrayView1<f64>) -> f64 {
        let mut node_idx = 0;

        loop {
            let node = &self.nodes[node_idx];

            if node.is_leaf() {
                return node.value;
            }

            let feature_val = features[node.feature_idx.unwrap()];
            let threshold = node.threshold.unwrap();

            node_idx = if feature_val <= threshold {
                node.left.unwrap()
            } else {
                node.right.unwrap()
            };
        }
    }

    /// Extract human-readable rules from the tree
    pub fn extract_rules(&self) -> crate::Result<Vec<DistilledRule>> {
        if !self.fitted {
            return Err(crate::DistillationError::NotFitted);
        }

        let mut rules = Vec::new();
        self.extract_rules_recursive(0, Vec::new(), Vec::new(), &mut rules);
        Ok(rules)
    }

    fn extract_rules_recursive(
        &self,
        node_idx: usize,
        conditions: Vec<String>,
        path: Vec<usize>,
        rules: &mut Vec<DistilledRule>,
    ) {
        let node = &self.nodes[node_idx];

        if node.is_leaf() {
            let prediction = if node.value > 0.5 {
                self.config
                    .class_names
                    .as_ref()
                    .and_then(|names| names.get(1))
                    .map(|s| s.clone())
                    .unwrap_or_else(|| "UP".to_string())
            } else {
                self.config
                    .class_names
                    .as_ref()
                    .and_then(|names| names.get(0))
                    .map(|s| s.clone())
                    .unwrap_or_else(|| "DOWN".to_string())
            };

            rules.push(DistilledRule {
                conditions,
                prediction,
                confidence: node.confidence,
                n_samples: node.n_samples,
                path,
            });
            return;
        }

        let feature_name = self
            .config
            .feature_names
            .as_ref()
            .and_then(|names| names.get(node.feature_idx.unwrap()))
            .map(|s| s.clone())
            .unwrap_or_else(|| format!("feature_{}", node.feature_idx.unwrap()));

        let threshold = node.threshold.unwrap();

        // Left branch (<=)
        let mut left_conditions = conditions.clone();
        left_conditions.push(format!("{} <= {:.4}", feature_name, threshold));
        let mut left_path = path.clone();
        left_path.push(node.left.unwrap());
        self.extract_rules_recursive(node.left.unwrap(), left_conditions, left_path, rules);

        // Right branch (>)
        let mut right_conditions = conditions;
        right_conditions.push(format!("{} > {:.4}", feature_name, threshold));
        let mut right_path = path;
        right_path.push(node.right.unwrap());
        self.extract_rules_recursive(node.right.unwrap(), right_conditions, right_path, rules);
    }

    /// Get feature importance scores
    pub fn feature_importance(&self) -> crate::Result<HashMap<usize, f64>> {
        if !self.fitted {
            return Err(crate::DistillationError::NotFitted);
        }

        let mut importance: HashMap<usize, f64> = HashMap::new();
        let mut total_samples = 0.0;

        for node in &self.nodes {
            if let Some(feature_idx) = node.feature_idx {
                let weight = node.n_samples as f64;
                *importance.entry(feature_idx).or_insert(0.0) += weight;
                total_samples += weight;
            }
        }

        // Normalize
        for value in importance.values_mut() {
            *value /= total_samples;
        }

        Ok(importance)
    }

    /// Calculate fidelity score (agreement with teacher predictions)
    pub fn fidelity_score(
        &self,
        features: &Array2<f64>,
        teacher_predictions: &Array1<f64>,
    ) -> crate::Result<f64> {
        let student_predictions = self.predict(features)?;

        let agreements: usize = student_predictions
            .iter()
            .zip(teacher_predictions.iter())
            .map(|(s, t)| {
                let s_class = if *s > 0.5 { 1 } else { 0 };
                let t_class = if *t > 0.5 { 1 } else { 0 };
                if s_class == t_class {
                    1
                } else {
                    0
                }
            })
            .sum();

        Ok(agreements as f64 / student_predictions.len() as f64)
    }

    /// Get tree depth
    pub fn depth(&self) -> usize {
        self.nodes.iter().map(|n| n.depth).max().unwrap_or(0)
    }

    /// Get number of leaves
    pub fn n_leaves(&self) -> usize {
        self.nodes.iter().filter(|n| n.is_leaf()).count()
    }

    /// Check if the tree has been fitted
    pub fn is_fitted(&self) -> bool {
        self.fitted
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_tree_creation() {
        let config = DistillationConfig::default();
        let tree = DecisionTree::new(config);
        assert!(!tree.is_fitted());
    }

    #[test]
    fn test_tree_fit_and_predict() {
        let config = DistillationConfig {
            max_depth: 3,
            min_samples_leaf: 1,
            min_samples_split: 2,
            ..Default::default()
        };

        let mut tree = DecisionTree::new(config);

        // Simple dataset
        let features = array![[0.0], [0.5], [1.0], [1.5], [2.0], [2.5], [3.0], [3.5]];
        let labels = array![0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9];

        tree.fit(&features, &labels).unwrap();
        assert!(tree.is_fitted());

        let predictions = tree.predict(&features).unwrap();
        assert_eq!(predictions.len(), 8);
    }

    #[test]
    fn test_rule_extraction() {
        let config = DistillationConfig {
            max_depth: 2,
            min_samples_leaf: 1,
            min_samples_split: 2,
            feature_names: Some(vec!["RSI".to_string()]),
            ..Default::default()
        };

        let mut tree = DecisionTree::new(config);

        let features = array![[10.0], [20.0], [30.0], [70.0], [80.0], [90.0]];
        let labels = array![0.8, 0.7, 0.6, 0.3, 0.2, 0.1];

        tree.fit(&features, &labels).unwrap();

        let rules = tree.extract_rules().unwrap();
        assert!(!rules.is_empty());

        for rule in &rules {
            println!("{}", rule);
        }
    }
}
