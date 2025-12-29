//! Random Forest implementation

use super::decision_tree::{DecisionTree, TaskType, TreeConfig};
use crate::data::Dataset;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Random Forest configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForestConfig {
    /// Number of trees in the forest
    pub n_trees: usize,
    /// Maximum depth of each tree
    pub max_depth: usize,
    /// Minimum samples to split
    pub min_samples_split: usize,
    /// Minimum samples in leaf
    pub min_samples_leaf: usize,
    /// Max features per tree (sqrt of total if None)
    pub max_features: Option<usize>,
    /// Bootstrap sampling
    pub bootstrap: bool,
    /// Random seed
    pub seed: u64,
    /// Task type
    pub task: TaskType,
    /// Out-of-bag score calculation
    pub oob_score: bool,
}

impl Default for ForestConfig {
    fn default() -> Self {
        Self {
            n_trees: 100,
            max_depth: 10,
            min_samples_split: 5,
            min_samples_leaf: 2,
            max_features: None,
            bootstrap: true,
            seed: 42,
            task: TaskType::Regression,
            oob_score: true,
        }
    }
}

/// Random Forest model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RandomForest {
    config: ForestConfig,
    trees: Vec<DecisionTree>,
    feature_names: Vec<String>,
    feature_importances: Vec<f64>,
    oob_score_value: Option<f64>,
}

impl RandomForest {
    /// Create a new random forest
    pub fn new(config: ForestConfig) -> Self {
        Self {
            config,
            trees: Vec::new(),
            feature_names: Vec::new(),
            feature_importances: Vec::new(),
            oob_score_value: None,
        }
    }

    /// Create with default regression config
    pub fn default_regression() -> Self {
        Self::new(ForestConfig {
            task: TaskType::Regression,
            ..Default::default()
        })
    }

    /// Create with default classification config
    pub fn default_classification() -> Self {
        Self::new(ForestConfig {
            task: TaskType::Classification,
            ..Default::default()
        })
    }

    /// Train the random forest
    pub fn fit(&mut self, dataset: &Dataset) {
        self.feature_names = dataset.feature_names.clone();
        let n_features = dataset.n_features();

        // Calculate max_features (default: sqrt for classification, n/3 for regression)
        let max_features = self.config.max_features.unwrap_or_else(|| {
            match self.config.task {
                TaskType::Classification => (n_features as f64).sqrt().ceil() as usize,
                TaskType::Regression => (n_features / 3).max(1),
            }
        });

        // Build trees in parallel
        let trees: Vec<DecisionTree> = (0..self.config.n_trees)
            .into_par_iter()
            .map(|i| {
                let tree_config = TreeConfig {
                    max_depth: self.config.max_depth,
                    min_samples_split: self.config.min_samples_split,
                    min_samples_leaf: self.config.min_samples_leaf,
                    max_features: Some(max_features),
                    seed: self.config.seed.wrapping_add(i as u64),
                    task: self.config.task,
                };

                let mut tree = DecisionTree::new(tree_config);

                // Bootstrap sample or use full dataset
                if self.config.bootstrap {
                    let bootstrap_data = dataset.bootstrap_sample(self.config.seed + i as u64);
                    tree.fit(&bootstrap_data);
                } else {
                    tree.fit(dataset);
                }

                tree
            })
            .collect();

        self.trees = trees;

        // Aggregate feature importances
        self.feature_importances = vec![0.0; n_features];
        for tree in &self.trees {
            for (i, &imp) in tree.feature_importances().iter().enumerate() {
                self.feature_importances[i] += imp;
            }
        }

        // Normalize
        let sum: f64 = self.feature_importances.iter().sum();
        if sum > 0.0 {
            for imp in &mut self.feature_importances {
                *imp /= sum;
            }
        }

        // Calculate OOB score if enabled
        if self.config.oob_score && self.config.bootstrap {
            self.calculate_oob_score(dataset);
        }
    }

    /// Calculate out-of-bag score
    fn calculate_oob_score(&mut self, dataset: &Dataset) {
        let n_samples = dataset.n_samples();
        let mut oob_predictions: Vec<Vec<f64>> = vec![Vec::new(); n_samples];

        // For each tree, predict on samples not in its bootstrap
        for (tree_idx, tree) in self.trees.iter().enumerate() {
            let seed = self.config.seed + tree_idx as u64;
            let in_bag = self.get_bootstrap_indices(n_samples, seed);

            for i in 0..n_samples {
                if !in_bag.contains(&i) {
                    let pred = tree.predict_one(&dataset.features[i]);
                    oob_predictions[i].push(pred);
                }
            }
        }

        // Calculate OOB score
        let mut correct = 0.0;
        let mut total = 0.0;

        for (i, preds) in oob_predictions.iter().enumerate() {
            if !preds.is_empty() {
                let avg_pred: f64 = preds.iter().sum::<f64>() / preds.len() as f64;

                match self.config.task {
                    TaskType::Classification => {
                        let pred_class = if avg_pred > 0.5 { 1.0 } else { 0.0 };
                        let actual_class = if dataset.labels[i] > 0.0 { 1.0 } else { 0.0 };
                        if pred_class == actual_class {
                            correct += 1.0;
                        }
                    }
                    TaskType::Regression => {
                        correct += (avg_pred - dataset.labels[i]).powi(2);
                    }
                }
                total += 1.0;
            }
        }

        self.oob_score_value = if total > 0.0 {
            match self.config.task {
                TaskType::Classification => Some(correct / total),
                TaskType::Regression => {
                    // Return negative MSE (higher is better)
                    let mean_label = dataset.labels.iter().sum::<f64>() / n_samples as f64;
                    let ss_tot: f64 = dataset.labels.iter().map(|l| (l - mean_label).powi(2)).sum();
                    let r2 = 1.0 - (correct / ss_tot);
                    Some(r2)
                }
            }
        } else {
            None
        };
    }

    fn get_bootstrap_indices(&self, n: usize, seed: u64) -> Vec<usize> {
        use rand::Rng;
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        (0..n).map(|_| rng.gen_range(0..n)).collect()
    }

    /// Predict for a single sample
    pub fn predict_one(&self, features: &[f64]) -> f64 {
        if self.trees.is_empty() {
            return 0.0;
        }

        let predictions: Vec<f64> = self.trees.iter().map(|t| t.predict_one(features)).collect();

        match self.config.task {
            TaskType::Regression => predictions.iter().sum::<f64>() / predictions.len() as f64,
            TaskType::Classification => {
                let pos_votes = predictions.iter().filter(|&&p| p > 0.5).count();
                if pos_votes > predictions.len() / 2 {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }

    /// Predict probabilities (for classification)
    pub fn predict_proba_one(&self, features: &[f64]) -> Vec<f64> {
        if self.trees.is_empty() {
            return vec![0.5, 0.5];
        }

        let predictions: Vec<f64> = self.trees.iter().map(|t| t.predict_one(features)).collect();

        let pos_ratio = predictions.iter().filter(|&&p| p > 0.5).count() as f64
            / predictions.len() as f64;

        vec![1.0 - pos_ratio, pos_ratio]
    }

    /// Predict for multiple samples
    pub fn predict(&self, dataset: &Dataset) -> Vec<f64> {
        dataset
            .features
            .par_iter()
            .map(|f| self.predict_one(f))
            .collect()
    }

    /// Predict probabilities
    pub fn predict_proba(&self, dataset: &Dataset) -> Vec<Vec<f64>> {
        dataset
            .features
            .par_iter()
            .map(|f| self.predict_proba_one(f))
            .collect()
    }

    /// Get feature importances
    pub fn feature_importances(&self) -> &[f64] {
        &self.feature_importances
    }

    /// Get feature names with importances, sorted by importance
    pub fn feature_importance_ranking(&self) -> Vec<(&str, f64)> {
        let mut ranking: Vec<(&str, f64)> = self
            .feature_names
            .iter()
            .zip(self.feature_importances.iter())
            .map(|(n, &i)| (n.as_str(), i))
            .collect();

        ranking.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        ranking
    }

    /// Get OOB score
    pub fn oob_score(&self) -> Option<f64> {
        self.oob_score_value
    }

    /// Calculate accuracy
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

    /// Calculate MSE
    pub fn mse(&self, dataset: &Dataset) -> f64 {
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

    /// Number of trees
    pub fn n_trees(&self) -> usize {
        self.trees.len()
    }

    /// Print summary
    pub fn summary(&self) {
        println!("Random Forest Summary");
        println!("=====================");
        println!("Number of trees: {}", self.n_trees());
        println!("Max depth: {}", self.config.max_depth);
        println!(
            "Task: {:?}",
            self.config.task
        );

        if let Some(oob) = self.oob_score_value {
            println!("OOB Score: {:.4}", oob);
        }

        println!("\nTop 10 Feature Importances:");
        for (name, importance) in self.feature_importance_ranking().iter().take(10) {
            println!("  {}: {:.4}", name, importance);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_forest_regression() {
        let mut dataset = Dataset::new(vec!["x1".to_string(), "x2".to_string()]);

        for i in 0..200 {
            let x1 = (i as f64) / 20.0;
            let x2 = ((i as f64) / 10.0).sin();
            let y = x1 + x2 * 2.0 + 0.1 * (i as f64 % 5.0);
            dataset.add_sample(vec![x1, x2], y, i as i64);
        }

        let mut forest = RandomForest::new(ForestConfig {
            n_trees: 10,
            max_depth: 5,
            task: TaskType::Regression,
            ..Default::default()
        });

        forest.fit(&dataset);

        assert_eq!(forest.n_trees(), 10);
        assert!(forest.feature_importances().len() == 2);
    }

    #[test]
    fn test_random_forest_classification() {
        let mut dataset = Dataset::new(vec!["x".to_string()]);

        for i in 0..200 {
            let x = i as f64 / 20.0;
            let y = if x > 5.0 { 1.0 } else { 0.0 };
            dataset.add_sample(vec![x], y, i as i64);
        }

        let mut forest = RandomForest::new(ForestConfig {
            n_trees: 20,
            max_depth: 5,
            task: TaskType::Classification,
            ..Default::default()
        });

        forest.fit(&dataset);
        let accuracy = forest.accuracy(&dataset);

        assert!(accuracy > 0.9);
    }
}
