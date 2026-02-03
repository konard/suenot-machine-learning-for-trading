//! SHAP (SHapley Additive exPlanations) implementation.
//!
//! This module provides SHAP value computation for explaining model predictions.
//! Based on the paper "A Unified Approach to Interpreting Model Predictions"
//! by Lundberg & Lee (2017).

use rand::prelude::*;
use std::collections::HashMap;

/// Container for SHAP values.
#[derive(Debug, Clone)]
pub struct ShapValues {
    /// SHAP values for each feature.
    pub values: Vec<f64>,
    /// Base value (expected prediction).
    pub base_value: f64,
    /// Feature names.
    pub feature_names: Vec<String>,
}

impl ShapValues {
    /// Create new SHAP values.
    pub fn new(values: Vec<f64>, base_value: f64, feature_names: Vec<String>) -> Self {
        Self {
            values,
            base_value,
            feature_names,
        }
    }

    /// Get top N contributing features by absolute value.
    pub fn top_contributors(&self, n: usize) -> Vec<(String, f64)> {
        let mut indexed: Vec<(usize, f64)> = self
            .values
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v.abs()))
            .collect();

        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        indexed
            .into_iter()
            .take(n)
            .map(|(i, _)| (self.feature_names[i].clone(), self.values[i]))
            .collect()
    }

    /// Convert to HashMap.
    pub fn to_map(&self) -> HashMap<String, f64> {
        self.feature_names
            .iter()
            .cloned()
            .zip(self.values.iter().cloned())
            .collect()
    }

    /// Sum of all SHAP values (should equal prediction - base_value).
    pub fn sum(&self) -> f64 {
        self.values.iter().sum()
    }
}

/// SHAP explanation for a single prediction.
#[derive(Debug, Clone)]
pub struct ShapExplanation {
    /// SHAP values.
    pub shap_values: ShapValues,
    /// Model prediction.
    pub prediction: f64,
    /// Input features.
    pub features: Vec<f64>,
}

impl ShapExplanation {
    /// Verify that SHAP values sum to prediction - base_value (efficiency property).
    pub fn verify_efficiency(&self, tolerance: f64) -> bool {
        let sum = self.shap_values.sum();
        let expected = self.prediction - self.shap_values.base_value;
        (sum - expected).abs() < tolerance
    }
}

/// SHAP explainer for trading models.
///
/// Implements KernelSHAP approximation for model-agnostic explanations.
pub struct ShapExplainer {
    /// Feature names.
    pub feature_names: Vec<String>,
    /// Base value (expected prediction on background data).
    pub base_value: f64,
    /// Background data for computing expected values.
    background_data: Vec<Vec<f64>>,
    /// Number of samples for KernelSHAP.
    n_samples: usize,
}

impl ShapExplainer {
    /// Create a new SHAP explainer.
    ///
    /// # Arguments
    ///
    /// * `feature_names` - Names of input features.
    /// * `base_value` - Expected prediction on background data.
    /// * `background_data` - Background dataset for computing expected values.
    /// * `n_samples` - Number of samples for KernelSHAP approximation.
    pub fn new(
        feature_names: Vec<String>,
        base_value: f64,
        background_data: Vec<Vec<f64>>,
        n_samples: usize,
    ) -> Self {
        Self {
            feature_names,
            base_value,
            background_data,
            n_samples,
        }
    }

    /// Compute SHAP values using KernelSHAP approximation.
    ///
    /// # Arguments
    ///
    /// * `predict_fn` - Function that takes features and returns prediction.
    /// * `x` - Input features to explain.
    ///
    /// # Returns
    ///
    /// ShapExplanation with SHAP values and metadata.
    pub fn explain<F>(&self, predict_fn: F, x: &[f64]) -> ShapExplanation
    where
        F: Fn(&[f64]) -> f64,
    {
        let n_features = x.len();
        let prediction = predict_fn(x);

        // Sample coalitions and compute SHAP values
        let shap_values = self.kernel_shap(&predict_fn, x);

        ShapExplanation {
            shap_values: ShapValues::new(
                shap_values,
                self.base_value,
                self.feature_names.clone(),
            ),
            prediction,
            features: x.to_vec(),
        }
    }

    /// KernelSHAP approximation.
    fn kernel_shap<F>(&self, predict_fn: &F, x: &[f64]) -> Vec<f64>
    where
        F: Fn(&[f64]) -> f64,
    {
        let n_features = x.len();
        let mut rng = rand::thread_rng();

        // Initialize SHAP values
        let mut shap_values = vec![0.0; n_features];
        let mut counts = vec![0usize; n_features];

        // Sample coalitions
        for _ in 0..self.n_samples {
            // Random permutation of features
            let mut perm: Vec<usize> = (0..n_features).collect();
            perm.shuffle(&mut rng);

            // Sample background instance
            let bg_idx = rng.gen_range(0..self.background_data.len());
            let bg = &self.background_data[bg_idx];

            // Compute marginal contributions
            let mut current_features = bg.clone();
            let mut prev_pred = predict_fn(&current_features);

            for &feature_idx in &perm {
                current_features[feature_idx] = x[feature_idx];
                let new_pred = predict_fn(&current_features);

                // Marginal contribution
                let contribution = new_pred - prev_pred;
                shap_values[feature_idx] += contribution;
                counts[feature_idx] += 1;

                prev_pred = new_pred;
            }
        }

        // Average the contributions
        for i in 0..n_features {
            if counts[i] > 0 {
                shap_values[i] /= counts[i] as f64;
            }
        }

        shap_values
    }

    /// Compute global feature importance as mean(|SHAP|).
    pub fn global_importance<F>(&self, predict_fn: &F, data: &[Vec<f64>]) -> HashMap<String, f64>
    where
        F: Fn(&[f64]) -> f64,
    {
        let n_features = self.feature_names.len();
        let mut importance = vec![0.0; n_features];

        for x in data {
            let explanation = self.explain(predict_fn, x);
            for (i, v) in explanation.shap_values.values.iter().enumerate() {
                importance[i] += v.abs();
            }
        }

        // Average
        for v in &mut importance {
            *v /= data.len() as f64;
        }

        self.feature_names
            .iter()
            .cloned()
            .zip(importance.into_iter())
            .collect()
    }
}

/// Simple linear model for demonstration.
pub struct LinearModel {
    pub weights: Vec<f64>,
    pub bias: f64,
}

impl LinearModel {
    /// Create a new linear model.
    pub fn new(weights: Vec<f64>, bias: f64) -> Self {
        Self { weights, bias }
    }

    /// Predict for a single input.
    pub fn predict(&self, x: &[f64]) -> f64 {
        let mut result = self.bias;
        for (w, xi) in self.weights.iter().zip(x.iter()) {
            result += w * xi;
        }
        // Sigmoid for probability output
        1.0 / (1.0 + (-result).exp())
    }

    /// For linear models, exact SHAP values are simply: β_i * (x_i - E[x_i])
    pub fn exact_shap_values(&self, x: &[f64], expected_x: &[f64]) -> Vec<f64> {
        self.weights
            .iter()
            .zip(x.iter())
            .zip(expected_x.iter())
            .map(|((w, xi), ex)| w * (xi - ex))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_model() {
        let model = LinearModel::new(vec![0.5, -0.3, 0.2], 0.0);
        let x = vec![1.0, 2.0, 3.0];
        let pred = model.predict(&x);

        // 0.5*1 + (-0.3)*2 + 0.2*3 = 0.5 - 0.6 + 0.6 = 0.5
        // sigmoid(0.5) ≈ 0.622
        assert!((pred - 0.622).abs() < 0.01);
    }

    #[test]
    fn test_shap_values_top_contributors() {
        let values = ShapValues::new(
            vec![0.1, -0.5, 0.3, -0.2],
            0.5,
            vec!["a".into(), "b".into(), "c".into(), "d".into()],
        );

        let top = values.top_contributors(2);
        assert_eq!(top[0].0, "b");
        assert_eq!(top[1].0, "c");
    }
}
