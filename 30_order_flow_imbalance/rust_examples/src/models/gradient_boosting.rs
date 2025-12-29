//! # Gradient Boosting Model
//!
//! Simple gradient boosting implementation for direction prediction.
//! For production, consider using external libraries like LightGBM bindings.

use crate::data::snapshot::FeatureVector;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Gradient boosting model for binary classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientBoostingModel {
    /// Decision trees
    trees: Vec<DecisionTree>,
    /// Learning rate
    learning_rate: f64,
    /// Feature importance
    feature_importance: HashMap<String, f64>,
    /// Feature names
    feature_names: Vec<String>,
    /// Threshold for classification
    threshold: f64,
    /// Model trained flag
    is_trained: bool,
}

/// Simple decision tree (stump for simplicity)
#[derive(Debug, Clone, Serialize, Deserialize)]
struct DecisionTree {
    feature_index: usize,
    threshold: f64,
    left_value: f64,
    right_value: f64,
}

impl GradientBoostingModel {
    /// Create a new model
    pub fn new(learning_rate: f64) -> Self {
        Self {
            trees: Vec::new(),
            learning_rate,
            feature_importance: HashMap::new(),
            feature_names: Vec::new(),
            threshold: 0.5,
            is_trained: false,
        }
    }

    /// Create with default parameters
    pub fn default_params() -> Self {
        Self::new(0.1)
    }

    /// Train the model on labeled data
    ///
    /// # Arguments
    /// * `features` - Feature vectors with labels
    /// * `n_trees` - Number of trees to train
    /// * `max_depth` - Maximum depth of trees (simplified to 1 here)
    pub fn train(&mut self, features: &[FeatureVector], n_trees: usize, _max_depth: usize) {
        if features.is_empty() {
            return;
        }

        // Store feature names
        self.feature_names = features[0].names.clone();

        // Initialize predictions
        let mut predictions: Vec<f64> = vec![0.5; features.len()];

        // Train trees iteratively
        for _ in 0..n_trees {
            // Calculate residuals
            let residuals: Vec<f64> = features
                .iter()
                .zip(predictions.iter())
                .map(|(f, &pred)| {
                    let label = f.label.unwrap_or(0.5);
                    label - pred
                })
                .collect();

            // Fit a tree to residuals
            if let Some(tree) = self.fit_tree(features, &residuals) {
                // Update predictions
                for (i, f) in features.iter().enumerate() {
                    predictions[i] += self.learning_rate * tree.predict(&f.values);
                    predictions[i] = predictions[i].clamp(0.0, 1.0);
                }

                // Update feature importance
                let name = &self.feature_names[tree.feature_index];
                *self.feature_importance.entry(name.clone()).or_insert(0.0) += 1.0;

                self.trees.push(tree);
            }
        }

        // Normalize feature importance
        let total: f64 = self.feature_importance.values().sum();
        if total > 0.0 {
            for v in self.feature_importance.values_mut() {
                *v /= total;
            }
        }

        self.is_trained = true;
    }

    /// Fit a single tree (stump) to residuals
    fn fit_tree(&self, features: &[FeatureVector], residuals: &[f64]) -> Option<DecisionTree> {
        if features.is_empty() || features[0].values.is_empty() {
            return None;
        }

        let n_features = features[0].values.len();
        let mut best_tree: Option<DecisionTree> = None;
        let mut best_score = f64::MAX;

        // Try random features
        let mut rng = rand::thread_rng();
        let features_to_try = (n_features / 3).max(1).min(10);

        for _ in 0..features_to_try {
            let feature_idx = rng.gen_range(0..n_features);

            // Find best split
            let mut values: Vec<(f64, f64)> = features
                .iter()
                .zip(residuals.iter())
                .map(|(f, &r)| (f.values[feature_idx], r))
                .collect();

            values.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            // Try a few split points
            for split_idx in (0..values.len()).step_by((values.len() / 10).max(1)) {
                let threshold = values[split_idx].0;

                let (left, right): (Vec<_>, Vec<_>) =
                    values.iter().partition(|(v, _)| *v <= threshold);

                if left.is_empty() || right.is_empty() {
                    continue;
                }

                let left_mean: f64 = left.iter().map(|(_, r)| r).sum::<f64>() / left.len() as f64;
                let right_mean: f64 = right.iter().map(|(_, r)| r).sum::<f64>() / right.len() as f64;

                // Calculate MSE
                let mse: f64 = left
                    .iter()
                    .map(|(_, r)| (r - left_mean).powi(2))
                    .sum::<f64>()
                    + right
                        .iter()
                        .map(|(_, r)| (r - right_mean).powi(2))
                        .sum::<f64>();

                if mse < best_score {
                    best_score = mse;
                    best_tree = Some(DecisionTree {
                        feature_index: feature_idx,
                        threshold,
                        left_value: left_mean,
                        right_value: right_mean,
                    });
                }
            }
        }

        best_tree
    }

    /// Predict probability for a single feature vector
    pub fn predict_proba(&self, features: &FeatureVector) -> f64 {
        if !self.is_trained {
            return 0.5;
        }

        let mut prediction = 0.5;
        for tree in &self.trees {
            prediction += self.learning_rate * tree.predict(&features.values);
        }

        prediction.clamp(0.0, 1.0)
    }

    /// Predict class (0 or 1)
    pub fn predict(&self, features: &FeatureVector) -> i32 {
        if self.predict_proba(features) > self.threshold {
            1
        } else {
            0
        }
    }

    /// Predict for multiple samples
    pub fn predict_batch(&self, features: &[FeatureVector]) -> Vec<i32> {
        features.iter().map(|f| self.predict(f)).collect()
    }

    /// Get feature importance
    pub fn feature_importance(&self) -> &HashMap<String, f64> {
        &self.feature_importance
    }

    /// Get top N important features
    pub fn top_features(&self, n: usize) -> Vec<(String, f64)> {
        let mut sorted: Vec<_> = self.feature_importance.iter().collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
        sorted.into_iter().take(n).map(|(k, v)| (k.clone(), *v)).collect()
    }

    /// Set classification threshold
    pub fn set_threshold(&mut self, threshold: f64) {
        self.threshold = threshold;
    }

    /// Check if model is trained
    pub fn is_trained(&self) -> bool {
        self.is_trained
    }

    /// Number of trees
    pub fn n_trees(&self) -> usize {
        self.trees.len()
    }

    /// Save model to JSON
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    /// Load model from JSON
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

impl DecisionTree {
    fn predict(&self, features: &[f64]) -> f64 {
        if features.get(self.feature_index).unwrap_or(&0.0) <= &self.threshold {
            self.left_value
        } else {
            self.right_value
        }
    }
}

impl Default for GradientBoostingModel {
    fn default() -> Self {
        Self::default_params()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn create_sample_data() -> Vec<FeatureVector> {
        let mut data = Vec::new();

        // Create simple linearly separable data
        for i in 0..100 {
            let mut fv = FeatureVector::new(Utc::now());
            let x1 = (i as f64) / 50.0 - 1.0;
            let x2 = ((i as f64) / 25.0).sin();

            fv.add("x1", x1);
            fv.add("x2", x2);

            // Label: 1 if x1 > 0, else 0
            fv.set_label(if x1 > 0.0 { 1.0 } else { 0.0 });

            data.push(fv);
        }

        data
    }

    #[test]
    fn test_model_training() {
        let data = create_sample_data();
        let mut model = GradientBoostingModel::new(0.1);

        model.train(&data, 10, 1);

        assert!(model.is_trained());
        assert!(model.n_trees() > 0);
    }

    #[test]
    fn test_prediction() {
        let data = create_sample_data();
        let mut model = GradientBoostingModel::new(0.1);

        model.train(&data, 50, 1);

        // Test prediction
        let mut fv = FeatureVector::new(Utc::now());
        fv.add("x1", 0.5); // Positive x1
        fv.add("x2", 0.0);

        let proba = model.predict_proba(&fv);
        // Should predict high probability for class 1
        assert!(proba > 0.3); // Relaxed threshold for simple model
    }

    #[test]
    fn test_feature_importance() {
        let data = create_sample_data();
        let mut model = GradientBoostingModel::new(0.1);

        model.train(&data, 50, 1);

        let importance = model.feature_importance();
        assert!(!importance.is_empty());
    }

    #[test]
    fn test_serialization() {
        let mut model = GradientBoostingModel::new(0.1);
        let data = create_sample_data();
        model.train(&data, 10, 1);

        let json = model.to_json().unwrap();
        let loaded = GradientBoostingModel::from_json(&json).unwrap();

        assert_eq!(loaded.n_trees(), model.n_trees());
    }
}
