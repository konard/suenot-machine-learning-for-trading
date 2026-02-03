//! Rule extraction algorithms for deriving interpretable rules from models.
//!
//! This module provides methods to extract if-then rules from:
//! - Decision tree structures
//! - Ensemble model predictions
//! - Neural network approximations (via surrogate trees)

use crate::rules::{Condition, Operator, Rule, RuleSet};
use std::collections::HashMap;

/// A node in a decision tree structure.
#[derive(Debug, Clone)]
pub struct TreeNode {
    /// Feature index for splitting (-1 for leaf nodes)
    pub feature: i32,
    /// Threshold for the split
    pub threshold: f64,
    /// Left child index (-1 if none)
    pub left_child: i32,
    /// Right child index (-1 if none)
    pub right_child: i32,
    /// Prediction value (for leaf nodes)
    pub value: Vec<f64>,
    /// Number of samples in this node
    pub n_samples: usize,
}

impl TreeNode {
    /// Check if this is a leaf node.
    pub fn is_leaf(&self) -> bool {
        self.feature == -1 || (self.left_child == -1 && self.right_child == -1)
    }

    /// Get the prediction class for a leaf node.
    pub fn prediction(&self) -> i32 {
        if self.value.is_empty() {
            return 0;
        }

        // Find argmax
        let (max_idx, _) = self
            .value
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        // Map to trading signals: 0 -> SELL (-1), 1 -> BUY (1)
        if max_idx == 0 {
            -1
        } else {
            1
        }
    }

    /// Get the confidence for this node's prediction.
    pub fn confidence(&self) -> f64 {
        if self.value.is_empty() {
            return 0.0;
        }

        let total: f64 = self.value.iter().sum();
        if total == 0.0 {
            return 0.0;
        }

        let max_val = self.value.iter().cloned().fold(0.0_f64, f64::max);
        max_val / total
    }
}

/// Decision tree structure for rule extraction.
#[derive(Debug, Clone)]
pub struct DecisionTree {
    /// Tree nodes
    pub nodes: Vec<TreeNode>,
    /// Feature names
    pub feature_names: Vec<String>,
}

impl DecisionTree {
    /// Create a new decision tree.
    pub fn new(nodes: Vec<TreeNode>, feature_names: Vec<String>) -> Self {
        Self {
            nodes,
            feature_names,
        }
    }

    /// Get the root node.
    pub fn root(&self) -> Option<&TreeNode> {
        self.nodes.first()
    }
}

/// Extractor for deriving rules from decision trees.
pub struct DecisionTreeExtractor {
    /// Feature names for rule interpretation
    feature_names: Vec<String>,
    /// Minimum samples required for a valid rule
    min_samples: usize,
    /// Maximum rule depth
    max_depth: usize,
}

impl DecisionTreeExtractor {
    /// Create a new extractor.
    pub fn new(feature_names: Vec<String>) -> Self {
        Self {
            feature_names,
            min_samples: 10,
            max_depth: 10,
        }
    }

    /// Set minimum samples threshold.
    pub fn with_min_samples(mut self, min_samples: usize) -> Self {
        self.min_samples = min_samples;
        self
    }

    /// Set maximum rule depth.
    pub fn with_max_depth(mut self, max_depth: usize) -> Self {
        self.max_depth = max_depth;
        self
    }

    /// Extract rules from a decision tree.
    pub fn extract_rules(&self, tree: &DecisionTree) -> Vec<Rule> {
        let mut rules = Vec::new();
        let mut path: Vec<Condition> = Vec::new();

        if let Some(root) = tree.root() {
            self.extract_recursive(tree, 0, &mut path, &mut rules, 0);
        }

        rules
    }

    /// Recursively extract rules from tree nodes.
    fn extract_recursive(
        &self,
        tree: &DecisionTree,
        node_idx: usize,
        path: &mut Vec<Condition>,
        rules: &mut Vec<Rule>,
        depth: usize,
    ) {
        if node_idx >= tree.nodes.len() || depth > self.max_depth {
            return;
        }

        let node = &tree.nodes[node_idx];

        if node.is_leaf() {
            // Create rule from current path
            if node.n_samples >= self.min_samples {
                let rule = Rule::new(path.clone(), node.prediction(), node.confidence());
                rules.push(rule);
            }
            return;
        }

        let feature_idx = node.feature as usize;

        // Left branch: feature <= threshold
        if node.left_child >= 0 {
            let left_condition = Condition::new(feature_idx, Operator::LessOrEqual, node.threshold);
            path.push(left_condition);
            self.extract_recursive(tree, node.left_child as usize, path, rules, depth + 1);
            path.pop();
        }

        // Right branch: feature > threshold
        if node.right_child >= 0 {
            let right_condition = Condition::new(feature_idx, Operator::GreaterThan, node.threshold);
            path.push(right_condition);
            self.extract_recursive(tree, node.right_child as usize, path, rules, depth + 1);
            path.pop();
        }
    }

    /// Extract rules from a decision tree and return as RuleSet.
    pub fn extract_ruleset(&self, tree: &DecisionTree) -> RuleSet {
        let rules = self.extract_rules(tree);
        RuleSet::new(rules, self.feature_names.clone())
    }
}

/// Consolidator for merging and simplifying rules.
pub struct RuleConsolidator {
    /// Minimum confidence threshold
    min_confidence: f64,
    /// Maximum number of rules to keep
    max_rules: usize,
}

impl RuleConsolidator {
    /// Create a new consolidator.
    pub fn new() -> Self {
        Self {
            min_confidence: 0.5,
            max_rules: 50,
        }
    }

    /// Set minimum confidence threshold.
    pub fn with_min_confidence(mut self, min_confidence: f64) -> Self {
        self.min_confidence = min_confidence;
        self
    }

    /// Set maximum number of rules.
    pub fn with_max_rules(mut self, max_rules: usize) -> Self {
        self.max_rules = max_rules;
        self
    }

    /// Consolidate rules from multiple trees.
    pub fn consolidate(&self, rules: Vec<Rule>) -> Vec<Rule> {
        // Filter by confidence
        let mut filtered: Vec<Rule> = rules
            .into_iter()
            .filter(|r| r.confidence >= self.min_confidence)
            .collect();

        // Sort by confidence * weight (descending)
        filtered.sort_by(|a, b| {
            let score_a = a.confidence * a.weight;
            let score_b = b.confidence * b.weight;
            score_b.partial_cmp(&score_a).unwrap()
        });

        // Keep top rules
        filtered.truncate(self.max_rules);

        // Merge similar rules
        self.merge_similar_rules(filtered)
    }

    /// Merge rules with identical conditions.
    fn merge_similar_rules(&self, rules: Vec<Rule>) -> Vec<Rule> {
        let mut merged: HashMap<String, Rule> = HashMap::new();

        for rule in rules {
            let key = self.rule_key(&rule);

            if let Some(existing) = merged.get_mut(&key) {
                // Combine weights and update confidence
                existing.weight += rule.weight;
                existing.confidence = (existing.confidence + rule.confidence) / 2.0;
            } else {
                merged.insert(key, rule);
            }
        }

        merged.into_values().collect()
    }

    /// Generate a key for a rule based on its conditions.
    fn rule_key(&self, rule: &Rule) -> String {
        let mut parts: Vec<String> = rule
            .conditions
            .iter()
            .map(|c| format!("{}:{}:{}", c.feature_idx, c.operator, c.threshold as i64))
            .collect();
        parts.sort();
        parts.push(format!("pred:{}", rule.prediction));
        parts.join("|")
    }
}

impl Default for RuleConsolidator {
    fn default() -> Self {
        Self::new()
    }
}

/// TREPAN-like rule extraction from black-box models.
///
/// This extracts rules by training a decision tree to mimic
/// the black-box model's predictions.
pub struct TrepanExtractor {
    /// Feature names
    feature_names: Vec<String>,
    /// Maximum tree depth
    max_depth: usize,
    /// Minimum samples per leaf
    min_samples_leaf: usize,
}

impl TrepanExtractor {
    /// Create a new TREPAN extractor.
    pub fn new(feature_names: Vec<String>) -> Self {
        Self {
            feature_names,
            max_depth: 6,
            min_samples_leaf: 50,
        }
    }

    /// Set maximum tree depth.
    pub fn with_max_depth(mut self, max_depth: usize) -> Self {
        self.max_depth = max_depth;
        self
    }

    /// Set minimum samples per leaf.
    pub fn with_min_samples_leaf(mut self, min_samples_leaf: usize) -> Self {
        self.min_samples_leaf = min_samples_leaf;
        self
    }

    /// Build a simple decision tree from data.
    ///
    /// This is a simplified implementation for demonstration.
    /// In production, use a proper tree-building algorithm.
    pub fn build_tree(&self, features: &[Vec<f64>], labels: &[i32]) -> DecisionTree {
        let mut nodes = Vec::new();

        // Build tree recursively
        self.build_node(features, labels, 0, &mut nodes);

        DecisionTree::new(nodes, self.feature_names.clone())
    }

    /// Build a tree node recursively.
    fn build_node(
        &self,
        features: &[Vec<f64>],
        labels: &[i32],
        depth: usize,
        nodes: &mut Vec<TreeNode>,
    ) -> i32 {
        let node_idx = nodes.len() as i32;

        // Check stopping conditions
        if depth >= self.max_depth
            || features.len() < self.min_samples_leaf
            || labels.iter().all(|&l| l == labels[0])
        {
            // Create leaf node
            let mut value = vec![0.0, 0.0];
            for &label in labels {
                if label == -1 {
                    value[0] += 1.0;
                } else if label == 1 {
                    value[1] += 1.0;
                }
            }

            nodes.push(TreeNode {
                feature: -1,
                threshold: 0.0,
                left_child: -1,
                right_child: -1,
                value,
                n_samples: features.len(),
            });

            return node_idx;
        }

        // Find best split
        let (best_feature, best_threshold, best_gain) =
            self.find_best_split(features, labels);

        if best_gain <= 0.0 {
            // No good split found, create leaf
            let mut value = vec![0.0, 0.0];
            for &label in labels {
                if label == -1 {
                    value[0] += 1.0;
                } else if label == 1 {
                    value[1] += 1.0;
                }
            }

            nodes.push(TreeNode {
                feature: -1,
                threshold: 0.0,
                left_child: -1,
                right_child: -1,
                value,
                n_samples: features.len(),
            });

            return node_idx;
        }

        // Create internal node (placeholder for children)
        nodes.push(TreeNode {
            feature: best_feature as i32,
            threshold: best_threshold,
            left_child: -1,
            right_child: -1,
            value: vec![],
            n_samples: features.len(),
        });

        // Split data
        let mut left_features = Vec::new();
        let mut left_labels = Vec::new();
        let mut right_features = Vec::new();
        let mut right_labels = Vec::new();

        for (feat, &label) in features.iter().zip(labels.iter()) {
            if feat[best_feature] <= best_threshold {
                left_features.push(feat.clone());
                left_labels.push(label);
            } else {
                right_features.push(feat.clone());
                right_labels.push(label);
            }
        }

        // Build children
        let left_child = self.build_node(&left_features, &left_labels, depth + 1, nodes);
        let right_child = self.build_node(&right_features, &right_labels, depth + 1, nodes);

        // Update parent with children
        nodes[node_idx as usize].left_child = left_child;
        nodes[node_idx as usize].right_child = right_child;

        node_idx
    }

    /// Find the best split for a node.
    fn find_best_split(
        &self,
        features: &[Vec<f64>],
        labels: &[i32],
    ) -> (usize, f64, f64) {
        let n_features = features.first().map(|f| f.len()).unwrap_or(0);
        let mut best_feature = 0;
        let mut best_threshold = 0.0;
        let mut best_gain = f64::NEG_INFINITY;

        let parent_impurity = self.gini_impurity(labels);

        for feat_idx in 0..n_features {
            // Get unique values for this feature
            let mut values: Vec<f64> = features.iter().map(|f| f[feat_idx]).collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap());
            values.dedup();

            // Try each threshold
            for i in 0..values.len().saturating_sub(1) {
                let threshold = (values[i] + values[i + 1]) / 2.0;

                // Split labels
                let mut left_labels = Vec::new();
                let mut right_labels = Vec::new();

                for (feat, &label) in features.iter().zip(labels.iter()) {
                    if feat[feat_idx] <= threshold {
                        left_labels.push(label);
                    } else {
                        right_labels.push(label);
                    }
                }

                if left_labels.is_empty() || right_labels.is_empty() {
                    continue;
                }

                // Calculate information gain
                let left_impurity = self.gini_impurity(&left_labels);
                let right_impurity = self.gini_impurity(&right_labels);

                let n = labels.len() as f64;
                let weighted_impurity = (left_labels.len() as f64 / n) * left_impurity
                    + (right_labels.len() as f64 / n) * right_impurity;

                let gain = parent_impurity - weighted_impurity;

                if gain > best_gain {
                    best_gain = gain;
                    best_feature = feat_idx;
                    best_threshold = threshold;
                }
            }
        }

        (best_feature, best_threshold, best_gain)
    }

    /// Calculate Gini impurity for a set of labels.
    fn gini_impurity(&self, labels: &[i32]) -> f64 {
        if labels.is_empty() {
            return 0.0;
        }

        let n = labels.len() as f64;
        let neg_count = labels.iter().filter(|&&l| l == -1).count() as f64;
        let pos_count = labels.iter().filter(|&&l| l == 1).count() as f64;

        let p_neg = neg_count / n;
        let p_pos = pos_count / n;

        1.0 - p_neg * p_neg - p_pos * p_pos
    }

    /// Extract rules from black-box model predictions.
    pub fn extract_from_predictions(
        &self,
        features: &[Vec<f64>],
        predictions: &[i32],
    ) -> RuleSet {
        // Build surrogate tree
        let tree = self.build_tree(features, predictions);

        // Extract rules
        let extractor = DecisionTreeExtractor::new(self.feature_names.clone())
            .with_max_depth(self.max_depth)
            .with_min_samples(self.min_samples_leaf);

        extractor.extract_ruleset(&tree)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tree_extraction() {
        let feature_names = vec!["RSI".to_string(), "MACD".to_string()];

        // Create a simple tree: IF RSI > 50 THEN BUY ELSE SELL
        let nodes = vec![
            TreeNode {
                feature: 0,
                threshold: 50.0,
                left_child: 1,
                right_child: 2,
                value: vec![],
                n_samples: 100,
            },
            TreeNode {
                feature: -1,
                threshold: 0.0,
                left_child: -1,
                right_child: -1,
                value: vec![80.0, 20.0], // SELL
                n_samples: 50,
            },
            TreeNode {
                feature: -1,
                threshold: 0.0,
                left_child: -1,
                right_child: -1,
                value: vec![20.0, 80.0], // BUY
                n_samples: 50,
            },
        ];

        let tree = DecisionTree::new(nodes, feature_names.clone());
        let extractor = DecisionTreeExtractor::new(feature_names);
        let rules = extractor.extract_rules(&tree);

        assert_eq!(rules.len(), 2);
    }

    #[test]
    fn test_trepan_extraction() {
        let feature_names = vec!["RSI".to_string()];
        let extractor = TrepanExtractor::new(feature_names).with_max_depth(3);

        let features = vec![
            vec![30.0],
            vec![35.0],
            vec![40.0],
            vec![65.0],
            vec![70.0],
            vec![75.0],
        ];
        let labels = vec![1, 1, 1, -1, -1, -1]; // Low RSI -> BUY, High RSI -> SELL

        let ruleset = extractor.extract_from_predictions(&features, &labels);

        // Should create rules that separate low and high RSI
        assert!(!ruleset.rules.is_empty());
    }
}
