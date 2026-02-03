//! LIME (Local Interpretable Model-agnostic Explanations) implementation.
//!
//! This module provides LIME explanations for trading model predictions.

use ndarray::{Array1, Array2};
use rand::Rng;
use std::collections::HashMap;

/// LIME explanation result
#[derive(Debug, Clone)]
pub struct LimeExplanation {
    /// Original instance being explained
    pub instance: Array1<f64>,
    /// Model prediction (0 or 1)
    pub prediction: usize,
    /// Prediction probability
    pub prediction_proba: f64,
    /// Feature contributions (feature_name -> contribution)
    pub feature_contributions: HashMap<String, f64>,
    /// Intercept of local linear model
    pub intercept: f64,
    /// R² score of local linear model
    pub local_model_score: f64,
    /// Feature names
    pub feature_names: Vec<String>,
}

impl LimeExplanation {
    /// Get top N features by absolute contribution
    pub fn get_top_features(&self, n: usize) -> Vec<(String, f64)> {
        let mut features: Vec<_> = self.feature_contributions.iter().collect();
        features.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
        features
            .into_iter()
            .take(n)
            .map(|(k, v)| (k.clone(), *v))
            .collect()
    }

    /// Format explanation as string
    pub fn to_string_pretty(&self) -> String {
        let mut lines = Vec::new();

        let direction = if self.prediction == 1 { "UP" } else { "DOWN" };
        lines.push(format!(
            "Prediction: {} (probability: {:.1}%)",
            direction,
            self.prediction_proba * 100.0
        ));
        lines.push(String::new());
        lines.push("Feature Contributions:".to_string());

        let mut contributions: Vec<_> = self.feature_contributions.iter().collect();
        contributions.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());

        for (name, contrib) in contributions {
            let sign = if *contrib >= 0.0 { "+" } else { "" };
            lines.push(format!("  {}{:.4}  {}", sign, contrib, name));
        }

        lines.push(String::new());
        lines.push(format!("Local model R²: {:.4}", self.local_model_score));

        lines.join("\n")
    }
}

impl std::fmt::Display for LimeExplanation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_string_pretty())
    }
}

/// LIME explainer for trading models
pub struct LimeExplainer {
    /// Function to get predictions from the model
    predict_fn: Box<dyn Fn(&Array2<f64>) -> Array2<f64>>,
    /// Feature names
    feature_names: Vec<String>,
    /// Number of features
    num_features: usize,
    /// Number of perturbed samples to generate
    num_samples: usize,
    /// Kernel width for proximity weighting
    kernel_width: f64,
    /// Feature means (from training data)
    feature_means: Option<Array1<f64>>,
    /// Feature standard deviations (from training data)
    feature_stds: Option<Array1<f64>>,
}

impl LimeExplainer {
    /// Create a new LIME explainer
    ///
    /// # Arguments
    /// * `predict_fn` - Function that takes features and returns probabilities [P(down), P(up)]
    /// * `feature_names` - Names of features
    /// * `training_data` - Optional training data for computing statistics
    /// * `num_samples` - Number of perturbed samples (default: 5000)
    pub fn new(
        predict_fn: impl Fn(&Array2<f64>) -> Array2<f64> + 'static,
        feature_names: Vec<String>,
        training_data: Option<&Array2<f64>>,
        num_samples: Option<usize>,
    ) -> Self {
        let num_features = feature_names.len();
        let num_samples = num_samples.unwrap_or(5000);
        let kernel_width = (num_features as f64).sqrt() * 0.75;

        let (feature_means, feature_stds) = if let Some(data) = training_data {
            let means = data
                .mean_axis(ndarray::Axis(0))
                .unwrap_or_else(|| Array1::zeros(num_features));
            let mut stds = Array1::zeros(num_features);

            for j in 0..num_features {
                let col = data.column(j);
                let mean = means[j];
                let variance = col.mapv(|x| (x - mean).powi(2)).mean().unwrap_or(1.0);
                stds[j] = variance.sqrt().max(1e-10);
            }

            (Some(means), Some(stds))
        } else {
            (None, None)
        };

        Self {
            predict_fn: Box::new(predict_fn),
            feature_names,
            num_features,
            num_samples,
            kernel_width,
            feature_means,
            feature_stds,
        }
    }

    /// Generate perturbed samples around an instance
    fn generate_perturbations(&self, instance: &Array1<f64>) -> Array2<f64> {
        let mut rng = rand::thread_rng();
        let mut perturbations = Array2::zeros((self.num_samples, self.num_features));

        // First sample is the original instance
        perturbations.row_mut(0).assign(instance);

        // Generate perturbed samples
        for i in 1..self.num_samples {
            for j in 0..self.num_features {
                let std = if let Some(ref stds) = self.feature_stds {
                    stds[j] * 0.5
                } else {
                    (instance[j].abs() * 0.1).max(0.01)
                };

                let noise: f64 = rng.gen::<f64>() * 2.0 - 1.0; // [-1, 1]
                perturbations[[i, j]] = instance[j] + noise * std * 2.0;
            }
        }

        perturbations
    }

    /// Compute distances between instance and perturbations
    fn compute_distances(&self, instance: &Array1<f64>, perturbations: &Array2<f64>) -> Array1<f64> {
        let mut distances = Array1::zeros(self.num_samples);

        let scale = self.feature_stds.as_ref().map(|s| s.clone()).unwrap_or_else(|| {
            Array1::from_elem(self.num_features, 1.0)
        });

        for i in 0..self.num_samples {
            let mut dist_sq = 0.0;
            for j in 0..self.num_features {
                let diff = (perturbations[[i, j]] - instance[j]) / scale[j];
                dist_sq += diff * diff;
            }
            distances[i] = dist_sq.sqrt();
        }

        distances
    }

    /// Compute kernel weights from distances
    fn compute_weights(&self, distances: &Array1<f64>) -> Array1<f64> {
        distances.mapv(|d| (-d * d / (self.kernel_width * self.kernel_width)).exp())
    }

    /// Fit a weighted linear regression model
    fn fit_local_model(
        &self,
        perturbations: &Array2<f64>,
        predictions: &Array1<f64>,
        weights: &Array1<f64>,
    ) -> (Array1<f64>, f64, f64) {
        // Weighted least squares: (X'WX)^-1 X'Wy
        let n = perturbations.nrows();
        let p = perturbations.ncols();

        // Add bias term
        let mut x_with_bias = Array2::zeros((n, p + 1));
        for i in 0..n {
            x_with_bias[[i, 0]] = 1.0; // Bias
            for j in 0..p {
                x_with_bias[[i, j + 1]] = perturbations[[i, j]];
            }
        }

        // Compute weighted X'X and X'y using simple approach
        let mut xtx = Array2::zeros((p + 1, p + 1));
        let mut xty = Array1::zeros(p + 1);

        for i in 0..n {
            let w = weights[i];
            for j in 0..=p {
                xty[j] += w * x_with_bias[[i, j]] * predictions[i];
                for k in 0..=p {
                    xtx[[j, k]] += w * x_with_bias[[i, j]] * x_with_bias[[i, k]];
                }
            }
        }

        // Add regularization for numerical stability
        for j in 0..=p {
            xtx[[j, j]] += 1e-6;
        }

        // Solve using Cholesky decomposition (simple implementation)
        let coefficients = self.solve_linear_system(&xtx, &xty);

        let intercept = coefficients[0];
        let coefs = coefficients.slice(ndarray::s![1..]).to_owned();

        // Compute R² score
        let predicted: Array1<f64> = perturbations
            .rows()
            .into_iter()
            .map(|row| intercept + row.dot(&coefs))
            .collect();

        let weighted_mean =
            predictions.iter().zip(weights.iter()).map(|(p, w)| p * w).sum::<f64>()
                / weights.sum();

        let ss_tot: f64 = predictions
            .iter()
            .zip(weights.iter())
            .map(|(p, w)| w * (p - weighted_mean).powi(2))
            .sum();

        let ss_res: f64 = predictions
            .iter()
            .zip(predicted.iter())
            .zip(weights.iter())
            .map(|((actual, pred), w)| w * (actual - pred).powi(2))
            .sum();

        let r2 = if ss_tot > 0.0 {
            1.0 - ss_res / ss_tot
        } else {
            0.0
        };

        (coefs, intercept, r2)
    }

    /// Simple linear system solver using Gauss elimination
    fn solve_linear_system(&self, a: &Array2<f64>, b: &Array1<f64>) -> Array1<f64> {
        let n = b.len();
        let mut aug = Array2::zeros((n, n + 1));

        // Create augmented matrix
        for i in 0..n {
            for j in 0..n {
                aug[[i, j]] = a[[i, j]];
            }
            aug[[i, n]] = b[i];
        }

        // Forward elimination
        for i in 0..n {
            // Find pivot
            let mut max_row = i;
            for k in (i + 1)..n {
                if aug[[k, i]].abs() > aug[[max_row, i]].abs() {
                    max_row = k;
                }
            }

            // Swap rows
            for j in 0..=n {
                let temp = aug[[i, j]];
                aug[[i, j]] = aug[[max_row, j]];
                aug[[max_row, j]] = temp;
            }

            // Eliminate column
            for k in (i + 1)..n {
                if aug[[i, i]].abs() > 1e-10 {
                    let factor = aug[[k, i]] / aug[[i, i]];
                    for j in i..=n {
                        aug[[k, j]] -= factor * aug[[i, j]];
                    }
                }
            }
        }

        // Back substitution
        let mut x = Array1::zeros(n);
        for i in (0..n).rev() {
            x[i] = aug[[i, n]];
            for j in (i + 1)..n {
                x[i] -= aug[[i, j]] * x[j];
            }
            if aug[[i, i]].abs() > 1e-10 {
                x[i] /= aug[[i, i]];
            }
        }

        x
    }

    /// Generate a LIME explanation for a single instance
    ///
    /// # Arguments
    /// * `instance` - Feature values for the instance to explain
    /// * `class_to_explain` - Which class probability to explain (default: 1 = UP)
    pub fn explain(&self, instance: &Array1<f64>, class_to_explain: usize) -> LimeExplanation {
        // Generate perturbations
        let perturbations = self.generate_perturbations(instance);

        // Get predictions from the model
        let proba = (self.predict_fn)(&perturbations);
        let predictions: Array1<f64> = proba.column(class_to_explain).to_owned();

        // Compute distances and weights
        let distances = self.compute_distances(instance, &perturbations);
        let weights = self.compute_weights(&distances);

        // Fit local linear model
        let (coefficients, intercept, r2_score) =
            self.fit_local_model(&perturbations, &predictions, &weights);

        // Compute feature contributions
        let mut feature_contributions = HashMap::new();
        for (i, name) in self.feature_names.iter().enumerate() {
            feature_contributions.insert(name.clone(), coefficients[i] * instance[i]);
        }

        // Get original prediction
        let original_instance = instance.clone().insert_axis(ndarray::Axis(0));
        let original_proba = (self.predict_fn)(&original_instance);
        let prediction_proba = original_proba[[0, class_to_explain]];
        let prediction = if original_proba[[0, 1]] > 0.5 { 1 } else { 0 };

        LimeExplanation {
            instance: instance.clone(),
            prediction,
            prediction_proba,
            feature_contributions,
            intercept,
            local_model_score: r2_score,
            feature_names: self.feature_names.clone(),
        }
    }

    /// Generate explanations for multiple instances
    pub fn explain_batch(
        &self,
        instances: &Array2<f64>,
        class_to_explain: usize,
    ) -> Vec<LimeExplanation> {
        instances
            .rows()
            .into_iter()
            .map(|row| self.explain(&row.to_owned(), class_to_explain))
            .collect()
    }
}

/// Analyze multiple LIME explanations
pub struct LimeAnalyzer;

impl LimeAnalyzer {
    /// Aggregate feature contributions across multiple explanations
    pub fn aggregate_explanations(explanations: &[LimeExplanation]) -> HashMap<String, f64> {
        if explanations.is_empty() {
            return HashMap::new();
        }

        let mut aggregated: HashMap<String, Vec<f64>> = HashMap::new();

        for exp in explanations {
            for (name, contrib) in &exp.feature_contributions {
                aggregated
                    .entry(name.clone())
                    .or_insert_with(Vec::new)
                    .push(contrib.abs());
            }
        }

        aggregated
            .into_iter()
            .map(|(name, contribs)| {
                let mean = contribs.iter().sum::<f64>() / contribs.len() as f64;
                (name, mean)
            })
            .collect()
    }

    /// Measure consistency of top features across explanations
    pub fn explanation_consistency(explanations: &[LimeExplanation], top_n: usize) -> f64 {
        if explanations.len() < 2 {
            return 1.0;
        }

        let top_features_sets: Vec<std::collections::HashSet<String>> = explanations
            .iter()
            .map(|exp| {
                exp.get_top_features(top_n)
                    .into_iter()
                    .map(|(name, _)| name)
                    .collect()
            })
            .collect();

        let mut total_overlap = 0.0;
        let mut comparisons = 0;

        for i in 0..top_features_sets.len() {
            for j in (i + 1)..top_features_sets.len() {
                let overlap = top_features_sets[i]
                    .intersection(&top_features_sets[j])
                    .count();
                total_overlap += overlap as f64 / top_n as f64;
                comparisons += 1;
            }
        }

        if comparisons == 0 {
            1.0
        } else {
            total_overlap / comparisons as f64
        }
    }

    /// Filter explanations by local model quality
    pub fn filter_by_confidence(
        explanations: Vec<LimeExplanation>,
        min_r2: f64,
    ) -> Vec<LimeExplanation> {
        explanations
            .into_iter()
            .filter(|exp| exp.local_model_score >= min_r2)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_lime_explainer_creation() {
        let predict_fn = |x: &Array2<f64>| {
            let n = x.nrows();
            let mut proba = Array2::zeros((n, 2));
            for i in 0..n {
                proba[[i, 1]] = 0.6;
                proba[[i, 0]] = 0.4;
            }
            proba
        };

        let feature_names = vec!["f1".to_string(), "f2".to_string()];
        let explainer = LimeExplainer::new(predict_fn, feature_names.clone(), None, Some(100));

        assert_eq!(explainer.num_features, 2);
        assert_eq!(explainer.num_samples, 100);
    }

    #[test]
    fn test_lime_explanation() {
        let predict_fn = |x: &Array2<f64>| {
            let n = x.nrows();
            let mut proba = Array2::zeros((n, 2));
            for i in 0..n {
                // Simple model: probability increases with first feature
                let p = (x[[i, 0]] / 10.0).clamp(0.0, 1.0);
                proba[[i, 1]] = p;
                proba[[i, 0]] = 1.0 - p;
            }
            proba
        };

        let feature_names = vec!["f1".to_string(), "f2".to_string()];
        let explainer = LimeExplainer::new(predict_fn, feature_names, None, Some(500));

        let instance = array![5.0, 3.0];
        let explanation = explainer.explain(&instance, 1);

        assert!(explanation.prediction_proba > 0.0);
        assert!(explanation.prediction_proba <= 1.0);
    }
}
