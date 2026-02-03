//! Simple Random Forest classifier implementation.
//!
//! This is a basic implementation for demonstration purposes.
//! For production use, consider using a dedicated ML library.

use ndarray::{Array1, Array2};
use rand::seq::SliceRandom;
use rand::Rng;

/// Decision tree node
#[derive(Debug, Clone)]
enum TreeNode {
    Leaf { class_proba: [f64; 2] },
    Split {
        feature_index: usize,
        threshold: f64,
        left: Box<TreeNode>,
        right: Box<TreeNode>,
    },
}

/// Decision tree classifier
#[derive(Debug, Clone)]
pub struct DecisionTree {
    root: Option<TreeNode>,
    max_depth: usize,
    min_samples_split: usize,
}

impl DecisionTree {
    pub fn new(max_depth: usize, min_samples_split: usize) -> Self {
        Self {
            root: None,
            max_depth,
            min_samples_split,
        }
    }

    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) {
        let indices: Vec<usize> = (0..x.nrows()).collect();
        self.root = Some(self.build_tree(x, y, &indices, 0));
    }

    fn build_tree(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        indices: &[usize],
        depth: usize,
    ) -> TreeNode {
        // Count classes
        let (count_0, count_1) = self.count_classes(y, indices);
        let total = (count_0 + count_1) as f64;

        // Check stopping conditions
        if depth >= self.max_depth
            || indices.len() < self.min_samples_split
            || count_0 == 0
            || count_1 == 0
        {
            return TreeNode::Leaf {
                class_proba: [count_0 as f64 / total, count_1 as f64 / total],
            };
        }

        // Find best split
        if let Some((feature_index, threshold, left_indices, right_indices)) =
            self.find_best_split(x, y, indices)
        {
            if left_indices.is_empty() || right_indices.is_empty() {
                return TreeNode::Leaf {
                    class_proba: [count_0 as f64 / total, count_1 as f64 / total],
                };
            }

            let left = self.build_tree(x, y, &left_indices, depth + 1);
            let right = self.build_tree(x, y, &right_indices, depth + 1);

            TreeNode::Split {
                feature_index,
                threshold,
                left: Box::new(left),
                right: Box::new(right),
            }
        } else {
            TreeNode::Leaf {
                class_proba: [count_0 as f64 / total, count_1 as f64 / total],
            }
        }
    }

    fn count_classes(&self, y: &Array1<f64>, indices: &[usize]) -> (usize, usize) {
        let mut count_0 = 0;
        let mut count_1 = 0;
        for &i in indices {
            if y[i] > 0.5 {
                count_1 += 1;
            } else {
                count_0 += 1;
            }
        }
        (count_0, count_1)
    }

    fn find_best_split(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        indices: &[usize],
    ) -> Option<(usize, f64, Vec<usize>, Vec<usize>)> {
        let mut best_gain = 0.0;
        let mut best_split = None;

        let n_features = x.ncols();
        let mut rng = rand::thread_rng();

        // Sample sqrt(n_features) features
        let n_try = (n_features as f64).sqrt().ceil() as usize;
        let mut features: Vec<usize> = (0..n_features).collect();
        features.shuffle(&mut rng);
        features.truncate(n_try);

        for &feature_idx in &features {
            // Get unique values for this feature
            let mut values: Vec<f64> = indices.iter().map(|&i| x[[i, feature_idx]]).collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap());
            values.dedup();

            // Try different thresholds
            for window in values.windows(2) {
                let threshold = (window[0] + window[1]) / 2.0;

                let (left_indices, right_indices): (Vec<usize>, Vec<usize>) = indices
                    .iter()
                    .partition(|&&i| x[[i, feature_idx]] <= threshold);

                if left_indices.is_empty() || right_indices.is_empty() {
                    continue;
                }

                let gain = self.information_gain(y, indices, &left_indices, &right_indices);

                if gain > best_gain {
                    best_gain = gain;
                    best_split = Some((feature_idx, threshold, left_indices, right_indices));
                }
            }
        }

        best_split
    }

    fn information_gain(
        &self,
        y: &Array1<f64>,
        parent: &[usize],
        left: &[usize],
        right: &[usize],
    ) -> f64 {
        let parent_entropy = self.entropy(y, parent);
        let left_entropy = self.entropy(y, left);
        let right_entropy = self.entropy(y, right);

        let n = parent.len() as f64;
        let n_left = left.len() as f64;
        let n_right = right.len() as f64;

        parent_entropy - (n_left / n) * left_entropy - (n_right / n) * right_entropy
    }

    fn entropy(&self, y: &Array1<f64>, indices: &[usize]) -> f64 {
        if indices.is_empty() {
            return 0.0;
        }

        let (count_0, count_1) = self.count_classes(y, indices);
        let total = (count_0 + count_1) as f64;

        let p0 = count_0 as f64 / total;
        let p1 = count_1 as f64 / total;

        let mut entropy = 0.0;
        if p0 > 0.0 {
            entropy -= p0 * p0.log2();
        }
        if p1 > 0.0 {
            entropy -= p1 * p1.log2();
        }

        entropy
    }

    pub fn predict_proba(&self, x: &Array1<f64>) -> [f64; 2] {
        match &self.root {
            Some(node) => self.predict_node(node, x),
            None => [0.5, 0.5],
        }
    }

    fn predict_node(&self, node: &TreeNode, x: &Array1<f64>) -> [f64; 2] {
        match node {
            TreeNode::Leaf { class_proba } => *class_proba,
            TreeNode::Split {
                feature_index,
                threshold,
                left,
                right,
            } => {
                if x[*feature_index] <= *threshold {
                    self.predict_node(left, x)
                } else {
                    self.predict_node(right, x)
                }
            }
        }
    }
}

/// Random Forest classifier
pub struct RandomForestClassifier {
    trees: Vec<DecisionTree>,
    n_estimators: usize,
    max_depth: usize,
    min_samples_split: usize,
}

impl RandomForestClassifier {
    /// Create a new Random Forest classifier
    pub fn new(n_estimators: usize, max_depth: usize, min_samples_split: usize) -> Self {
        Self {
            trees: Vec::new(),
            n_estimators,
            max_depth,
            min_samples_split,
        }
    }

    /// Train the Random Forest
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) {
        let n_samples = x.nrows();
        let mut rng = rand::thread_rng();

        self.trees.clear();

        for _ in 0..self.n_estimators {
            // Bootstrap sampling
            let bootstrap_indices: Vec<usize> =
                (0..n_samples).map(|_| rng.gen_range(0..n_samples)).collect();

            let x_bootstrap = bootstrap_indices
                .iter()
                .map(|&i| x.row(i).to_vec())
                .collect::<Vec<_>>();
            let x_bootstrap =
                Array2::from_shape_vec((n_samples, x.ncols()), x_bootstrap.into_iter().flatten().collect())
                    .unwrap();

            let y_bootstrap = Array1::from_vec(bootstrap_indices.iter().map(|&i| y[i]).collect());

            // Train a tree
            let mut tree = DecisionTree::new(self.max_depth, self.min_samples_split);
            tree.fit(&x_bootstrap, &y_bootstrap);
            self.trees.push(tree);
        }
    }

    /// Predict class probabilities
    pub fn predict_proba(&self, x: &Array2<f64>) -> Array2<f64> {
        let n_samples = x.nrows();
        let mut proba = Array2::zeros((n_samples, 2));

        for i in 0..n_samples {
            let row = x.row(i).to_owned();
            let mut sum = [0.0, 0.0];

            for tree in &self.trees {
                let tree_proba = tree.predict_proba(&row);
                sum[0] += tree_proba[0];
                sum[1] += tree_proba[1];
            }

            let n_trees = self.trees.len() as f64;
            proba[[i, 0]] = sum[0] / n_trees;
            proba[[i, 1]] = sum[1] / n_trees;
        }

        proba
    }

    /// Predict class labels
    pub fn predict(&self, x: &Array2<f64>) -> Array1<usize> {
        let proba = self.predict_proba(x);
        proba
            .rows()
            .into_iter()
            .map(|row| if row[1] > row[0] { 1 } else { 0 })
            .collect()
    }

    /// Get feature importance (based on frequency of feature usage)
    pub fn feature_importance(&self, n_features: usize) -> Array1<f64> {
        let mut importance = Array1::zeros(n_features);
        let mut count = Array1::zeros(n_features);

        fn count_feature_usage(node: &TreeNode, importance: &mut Array1<f64>, count: &mut Array1<f64>) {
            match node {
                TreeNode::Leaf { .. } => {}
                TreeNode::Split {
                    feature_index,
                    left,
                    right,
                    ..
                } => {
                    importance[*feature_index] += 1.0;
                    count[*feature_index] += 1.0;
                    count_feature_usage(left, importance, count);
                    count_feature_usage(right, importance, count);
                }
            }
        }

        for tree in &self.trees {
            if let Some(root) = &tree.root {
                count_feature_usage(root, &mut importance, &mut count);
            }
        }

        // Normalize
        let sum = importance.sum();
        if sum > 0.0 {
            importance /= sum;
        }

        importance
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_random_forest_training() {
        let x = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [5.0, 6.0],
            [6.0, 7.0],
            [7.0, 8.0],
            [8.0, 9.0],
        ];
        let y = array![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];

        let mut rf = RandomForestClassifier::new(10, 3, 2);
        rf.fit(&x, &y);

        let predictions = rf.predict(&x);
        assert_eq!(predictions.len(), 8);
    }

    #[test]
    fn test_random_forest_predict_proba() {
        let x = array![
            [1.0, 1.0],
            [2.0, 2.0],
            [8.0, 8.0],
            [9.0, 9.0],
        ];
        let y = array![0.0, 0.0, 1.0, 1.0];

        let mut rf = RandomForestClassifier::new(20, 5, 1);
        rf.fit(&x, &y);

        let proba = rf.predict_proba(&x);

        // Low values should have higher probability of class 0
        assert!(proba[[0, 0]] > proba[[0, 1]]);
        // High values should have higher probability of class 1
        assert!(proba[[3, 1]] > proba[[3, 0]]);
    }
}
