//! Feature Attribution Methods for Trading Models
//!
//! This module implements various methods for explaining which features
//! contribute most to trading model predictions.
//!
//! ## Methods Implemented
//!
//! - **Permutation Importance**: Measure feature importance by shuffling feature values
//! - **Gradient Attribution**: Use model gradients to compute feature contributions
//! - **Shapley Approximation**: Approximate Shapley values for feature contributions
//! - **Integrated Gradients**: Path-based attribution method

use ndarray::{Array1, Array2, Axis};
use rand::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::NeuralNetwork;

/// Error types for attribution methods
#[derive(Debug, thiserror::Error)]
pub enum AttributionError {
    #[error("Invalid input dimension: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("Empty input data")]
    EmptyData,

    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    #[error("Computation error: {0}")]
    ComputationError(String),
}

/// Trait for feature attribution methods
pub trait FeatureAttributor: Send + Sync {
    /// Compute feature attributions for a single input
    fn explain<M: NeuralNetwork>(
        &self,
        model: &M,
        input: &Array1<f64>,
        baseline: Option<&Array1<f64>>,
    ) -> Array1<f64>;

    /// Compute feature attributions for a batch of inputs
    fn explain_batch<M: NeuralNetwork + Sync>(
        &self,
        model: &M,
        inputs: &Array2<f64>,
    ) -> Array2<f64>;

    /// Get method name
    fn name(&self) -> &'static str;
}

/// Feature importance result with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureImportance {
    /// Feature index
    pub index: usize,
    /// Feature name (optional)
    pub name: Option<String>,
    /// Importance score
    pub importance: f64,
    /// Standard deviation (if computed from multiple samples)
    pub std_dev: Option<f64>,
    /// Direction of effect (positive = bullish, negative = bearish)
    pub direction: f64,
}

/// Complete attribution result with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttributionResult {
    /// Per-feature attributions
    pub attributions: Vec<f64>,
    /// Model prediction for the input
    pub prediction: f64,
    /// Baseline prediction (if applicable)
    pub baseline_prediction: Option<f64>,
    /// Convergence delta (completeness check)
    pub delta: Option<f64>,
    /// Method used
    pub method: String,
    /// Additional metadata
    pub metadata: HashMap<String, f64>,
}

impl AttributionResult {
    /// Get top k features by absolute attribution
    pub fn top_k(&self, k: usize) -> Vec<(usize, f64)> {
        let mut indexed: Vec<(usize, f64)> = self
            .attributions
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();

        indexed.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap_or(std::cmp::Ordering::Equal));
        indexed.truncate(k);
        indexed
    }

    /// Get features with attribution above threshold
    pub fn above_threshold(&self, threshold: f64) -> Vec<(usize, f64)> {
        self.attributions
            .iter()
            .enumerate()
            .filter(|(_, &v)| v.abs() > threshold)
            .map(|(i, &v)| (i, v))
            .collect()
    }

    /// Get sum of attributions (should approximate prediction - baseline for some methods)
    pub fn attribution_sum(&self) -> f64 {
        self.attributions.iter().sum()
    }

    /// Normalize attributions to sum to 1 (absolute values)
    pub fn normalized(&self) -> Vec<f64> {
        let total: f64 = self.attributions.iter().map(|x| x.abs()).sum();
        if total > 1e-10 {
            self.attributions.iter().map(|x| x.abs() / total).collect()
        } else {
            vec![0.0; self.attributions.len()]
        }
    }
}

// ============================================================================
// Permutation Importance
// ============================================================================

/// Permutation Importance for feature attribution
///
/// Measures feature importance by randomly permuting feature values and
/// observing the change in model predictions.
#[derive(Debug, Clone)]
pub struct PermutationImportance {
    /// Number of permutation iterations
    n_iterations: usize,
    /// Random seed for reproducibility
    seed: Option<u64>,
}

impl PermutationImportance {
    /// Create a new Permutation Importance attributor
    ///
    /// # Arguments
    ///
    /// * `n_iterations` - Number of permutation iterations per feature
    pub fn new(n_iterations: usize) -> Self {
        Self {
            n_iterations,
            seed: None,
        }
    }

    /// Set random seed for reproducibility
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Compute importance scores for all features on a dataset
    pub fn compute_importance<M: NeuralNetwork>(
        &self,
        model: &M,
        data: &Array2<f64>,
        targets: &[f64],
    ) -> Vec<FeatureImportance> {
        let n_features = data.ncols();
        let n_samples = data.nrows();

        // Compute baseline predictions
        let baseline_preds: Vec<f64> = (0..n_samples)
            .map(|i| model.forward(&data.row(i).to_owned())[0])
            .collect();

        let baseline_mse = compute_mse(&baseline_preds, targets);

        let mut importances = Vec::with_capacity(n_features);

        for feature_idx in 0..n_features {
            let mut mse_increases = Vec::with_capacity(self.n_iterations);

            for iter in 0..self.n_iterations {
                // Create permuted data
                let mut permuted_data = data.clone();
                let seed = self.seed.unwrap_or(42) + iter as u64;
                let mut rng = StdRng::seed_from_u64(seed);

                // Permute the feature column
                let mut indices: Vec<usize> = (0..n_samples).collect();
                indices.shuffle(&mut rng);

                for (i, &new_idx) in indices.iter().enumerate() {
                    permuted_data[[i, feature_idx]] = data[[new_idx, feature_idx]];
                }

                // Compute predictions with permuted feature
                let permuted_preds: Vec<f64> = (0..n_samples)
                    .map(|i| model.forward(&permuted_data.row(i).to_owned())[0])
                    .collect();

                let permuted_mse = compute_mse(&permuted_preds, targets);
                mse_increases.push(permuted_mse - baseline_mse);
            }

            // Compute mean and std of importance
            let mean_importance = mse_increases.iter().sum::<f64>() / mse_increases.len() as f64;
            let variance = mse_increases
                .iter()
                .map(|x| (x - mean_importance).powi(2))
                .sum::<f64>()
                / mse_increases.len() as f64;
            let std_dev = variance.sqrt();

            // Determine direction based on correlation with feature values
            let direction = compute_feature_direction(data, feature_idx, &baseline_preds);

            importances.push(FeatureImportance {
                index: feature_idx,
                name: None,
                importance: mean_importance,
                std_dev: Some(std_dev),
                direction,
            });
        }

        // Sort by importance (descending)
        importances.sort_by(|a, b| {
            b.importance.abs().partial_cmp(&a.importance.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        importances
    }
}

impl FeatureAttributor for PermutationImportance {
    fn explain<M: NeuralNetwork>(
        &self,
        model: &M,
        input: &Array1<f64>,
        baseline: Option<&Array1<f64>>,
    ) -> Array1<f64> {
        let n_features = input.len();
        let baseline_val = baseline.cloned().unwrap_or_else(|| Array1::zeros(n_features));

        let original_pred = model.forward(input)[0];
        let mut attributions = Array1::zeros(n_features);

        let mut rng = self.seed.map(StdRng::seed_from_u64)
            .unwrap_or_else(StdRng::from_entropy);

        for feature_idx in 0..n_features {
            let mut diff_sum = 0.0;

            for _ in 0..self.n_iterations {
                // Replace feature with baseline value (or random perturbation)
                let mut perturbed = input.clone();
                let perturbation = if baseline.is_some() {
                    baseline_val[feature_idx]
                } else {
                    // Random perturbation
                    input[feature_idx] * (1.0 + rng.gen::<f64>() * 0.2 - 0.1)
                };

                perturbed[feature_idx] = perturbation;
                let perturbed_pred = model.forward(&perturbed)[0];
                diff_sum += (original_pred - perturbed_pred).abs();
            }

            attributions[feature_idx] = diff_sum / self.n_iterations as f64;
        }

        // Normalize to sum to prediction difference
        let attr_sum: f64 = attributions.sum();
        if attr_sum > 1e-10 {
            attributions = &attributions / attr_sum * (original_pred - model.forward(&baseline_val)[0]).abs();
        }

        attributions
    }

    fn explain_batch<M: NeuralNetwork + Sync>(
        &self,
        model: &M,
        inputs: &Array2<f64>,
    ) -> Array2<f64> {
        let n_samples = inputs.nrows();
        let n_features = inputs.ncols();

        let attributions: Vec<Array1<f64>> = (0..n_samples)
            .into_par_iter()
            .map(|i| {
                let input = inputs.row(i).to_owned();
                self.explain(model, &input, None)
            })
            .collect();

        // Stack into 2D array
        let mut result = Array2::zeros((n_samples, n_features));
        for (i, attr) in attributions.into_iter().enumerate() {
            result.row_mut(i).assign(&attr);
        }

        result
    }

    fn name(&self) -> &'static str {
        "Permutation Importance"
    }
}

// ============================================================================
// Gradient Attribution
// ============================================================================

/// Gradient-based feature attribution
///
/// Uses the gradient of the model output with respect to input features
/// to compute attributions.
#[derive(Debug, Clone)]
pub struct GradientAttribution {
    /// Number of integration steps for path methods
    n_steps: usize,
    /// Type of baseline
    baseline_type: BaselineType,
    /// Mean values for mean baseline
    mean_baseline: Option<Array1<f64>>,
    /// Use absolute gradients
    use_absolute: bool,
}

/// Type of baseline for gradient-based methods
#[derive(Debug, Clone, Copy)]
pub enum BaselineType {
    /// Zero baseline (default)
    Zero,
    /// Mean of training data
    Mean,
    /// Random baseline
    Random,
}

impl GradientAttribution {
    /// Create a new Gradient Attribution with specified integration steps
    pub fn new(n_steps: usize) -> Self {
        Self {
            n_steps,
            baseline_type: BaselineType::Zero,
            mean_baseline: None,
            use_absolute: false,
        }
    }

    /// Set baseline type
    pub fn with_baseline_type(mut self, baseline_type: BaselineType) -> Self {
        self.baseline_type = baseline_type;
        self
    }

    /// Set mean baseline values
    pub fn with_mean_baseline(mut self, mean: Array1<f64>) -> Self {
        self.mean_baseline = Some(mean);
        self.baseline_type = BaselineType::Mean;
        self
    }

    /// Use absolute values of gradients
    pub fn with_absolute(mut self, use_absolute: bool) -> Self {
        self.use_absolute = use_absolute;
        self
    }

    /// Get baseline for input
    fn get_baseline(&self, input: &Array1<f64>) -> Array1<f64> {
        match self.baseline_type {
            BaselineType::Zero => Array1::zeros(input.len()),
            BaselineType::Mean => {
                self.mean_baseline.clone().unwrap_or_else(|| Array1::zeros(input.len()))
            }
            BaselineType::Random => {
                let mut rng = rand::thread_rng();
                Array1::from_iter((0..input.len()).map(|_| rng.gen::<f64>() * 0.01))
            }
        }
    }

    /// Compute simple gradient attribution (Input * Gradient)
    pub fn input_gradient<M: NeuralNetwork>(
        &self,
        model: &M,
        input: &Array1<f64>,
    ) -> Array1<f64> {
        let gradient = model.gradient(input);

        if self.use_absolute {
            input * &gradient.mapv(|x| x.abs())
        } else {
            input * &gradient
        }
    }

    /// Compute integrated gradients
    pub fn integrated_gradients<M: NeuralNetwork>(
        &self,
        model: &M,
        input: &Array1<f64>,
        baseline: Option<&Array1<f64>>,
    ) -> Array1<f64> {
        let baseline = baseline.cloned().unwrap_or_else(|| self.get_baseline(input));
        let diff = input - &baseline;

        // Generate interpolation alphas
        let alphas: Vec<f64> = (1..=self.n_steps)
            .map(|k| k as f64 / self.n_steps as f64)
            .collect();

        // Compute gradients along the path
        let gradients: Vec<Array1<f64>> = alphas
            .iter()
            .map(|&alpha| {
                let interpolated = &baseline + &(&diff * alpha);
                model.gradient(&interpolated)
            })
            .collect();

        // Riemann sum approximation of integral
        let n = gradients.len() as f64;
        let avg_gradient = gradients
            .into_iter()
            .fold(Array1::zeros(input.len()), |acc, g| acc + g)
            / n;

        // Attribution = (input - baseline) * average_gradient
        &diff * &avg_gradient
    }

    /// Compute smooth gradients (average over noisy inputs)
    pub fn smooth_gradients<M: NeuralNetwork>(
        &self,
        model: &M,
        input: &Array1<f64>,
        noise_std: f64,
    ) -> Array1<f64> {
        let mut rng = rand::thread_rng();
        let n_samples = self.n_steps;

        let gradients: Vec<Array1<f64>> = (0..n_samples)
            .map(|_| {
                // Add Gaussian noise
                let noise: Array1<f64> = Array1::from_iter(
                    (0..input.len()).map(|_| rng.gen::<f64>() * noise_std * 2.0 - noise_std)
                );
                let noisy_input = input + &noise;
                model.gradient(&noisy_input)
            })
            .collect();

        // Average gradients
        let avg_gradient = gradients
            .into_iter()
            .fold(Array1::zeros(input.len()), |acc, g| acc + g)
            / n_samples as f64;

        if self.use_absolute {
            input * &avg_gradient.mapv(|x| x.abs())
        } else {
            input * &avg_gradient
        }
    }

    /// Compute convergence delta (completeness check for integrated gradients)
    pub fn convergence_delta<M: NeuralNetwork>(
        &self,
        model: &M,
        input: &Array1<f64>,
        baseline: Option<&Array1<f64>>,
    ) -> f64 {
        let baseline = baseline.cloned().unwrap_or_else(|| self.get_baseline(input));
        let attributions = self.integrated_gradients(model, input, Some(&baseline));

        let pred_input = model.forward(input)[0];
        let pred_baseline = model.forward(&baseline)[0];

        (attributions.sum() - (pred_input - pred_baseline)).abs()
    }
}

impl FeatureAttributor for GradientAttribution {
    fn explain<M: NeuralNetwork>(
        &self,
        model: &M,
        input: &Array1<f64>,
        baseline: Option<&Array1<f64>>,
    ) -> Array1<f64> {
        self.integrated_gradients(model, input, baseline)
    }

    fn explain_batch<M: NeuralNetwork + Sync>(
        &self,
        model: &M,
        inputs: &Array2<f64>,
    ) -> Array2<f64> {
        let n_samples = inputs.nrows();
        let n_features = inputs.ncols();

        let attributions: Vec<Array1<f64>> = (0..n_samples)
            .into_par_iter()
            .map(|i| {
                let input = inputs.row(i).to_owned();
                self.explain(model, &input, None)
            })
            .collect();

        let mut result = Array2::zeros((n_samples, n_features));
        for (i, attr) in attributions.into_iter().enumerate() {
            result.row_mut(i).assign(&attr);
        }

        result
    }

    fn name(&self) -> &'static str {
        "Gradient Attribution (Integrated Gradients)"
    }
}

// ============================================================================
// Shapley Value Approximation
// ============================================================================

/// Approximate Shapley values for feature attribution
///
/// Uses sampling-based approximation to compute SHAP-like values.
#[derive(Debug, Clone)]
pub struct ShapleyApproximation {
    /// Number of permutation samples
    n_samples: usize,
    /// Random seed
    seed: Option<u64>,
}

impl ShapleyApproximation {
    /// Create a new Shapley approximation
    pub fn new(n_samples: usize) -> Self {
        Self {
            n_samples,
            seed: None,
        }
    }

    /// Set random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Compute kernel SHAP weights
    fn shapley_kernel_weight(n_features: usize, coalition_size: usize) -> f64 {
        if coalition_size == 0 || coalition_size == n_features {
            // Edge cases: infinite weight, but we handle them specially
            1e6
        } else {
            let n = n_features as f64;
            let k = coalition_size as f64;
            (n - 1.0) / (binomial(n_features, coalition_size) * k * (n - k))
        }
    }
}

impl FeatureAttributor for ShapleyApproximation {
    fn explain<M: NeuralNetwork>(
        &self,
        model: &M,
        input: &Array1<f64>,
        baseline: Option<&Array1<f64>>,
    ) -> Array1<f64> {
        let n_features = input.len();
        let baseline = baseline.cloned().unwrap_or_else(|| Array1::zeros(n_features));

        let mut rng = self.seed.map(StdRng::seed_from_u64)
            .unwrap_or_else(StdRng::from_entropy);

        // Initialize attributions
        let mut attributions = Array1::zeros(n_features);
        let mut counts = vec![0usize; n_features];

        // Sample permutations
        for _ in 0..self.n_samples {
            // Random permutation of features
            let mut perm: Vec<usize> = (0..n_features).collect();
            perm.shuffle(&mut rng);

            // Track cumulative prediction as we add features
            let mut current = baseline.clone();
            let mut prev_pred = model.forward(&current)[0];

            for &feature_idx in &perm {
                // Add this feature to the coalition
                current[feature_idx] = input[feature_idx];
                let new_pred = model.forward(&current)[0];

                // Marginal contribution
                let marginal = new_pred - prev_pred;
                attributions[feature_idx] += marginal;
                counts[feature_idx] += 1;

                prev_pred = new_pred;
            }
        }

        // Average over samples
        for i in 0..n_features {
            if counts[i] > 0 {
                attributions[i] /= counts[i] as f64;
            }
        }

        attributions
    }

    fn explain_batch<M: NeuralNetwork + Sync>(
        &self,
        model: &M,
        inputs: &Array2<f64>,
    ) -> Array2<f64> {
        let n_samples = inputs.nrows();
        let n_features = inputs.ncols();

        let attributions: Vec<Array1<f64>> = (0..n_samples)
            .into_par_iter()
            .map(|i| {
                let input = inputs.row(i).to_owned();
                self.explain(model, &input, None)
            })
            .collect();

        let mut result = Array2::zeros((n_samples, n_features));
        for (i, attr) in attributions.into_iter().enumerate() {
            result.row_mut(i).assign(&attr);
        }

        result
    }

    fn name(&self) -> &'static str {
        "Shapley Value Approximation"
    }
}

// ============================================================================
// Attribution Analysis
// ============================================================================

/// Comprehensive attribution analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttributionAnalysis {
    /// Mean attribution per feature
    pub mean_attributions: Vec<f64>,
    /// Standard deviation of attributions
    pub std_attributions: Vec<f64>,
    /// Feature rankings by importance
    pub feature_rankings: Vec<usize>,
    /// Correlation between features and predictions
    pub feature_correlations: Vec<f64>,
    /// Attribution stability (consistency across samples)
    pub stability_score: f64,
}

impl AttributionAnalysis {
    /// Analyze attributions from multiple samples
    pub fn from_batch(attributions: &Array2<f64>) -> Self {
        let n_features = attributions.ncols();
        let n_samples = attributions.nrows();

        // Compute mean and std for each feature
        let mut mean_attributions = Vec::with_capacity(n_features);
        let mut std_attributions = Vec::with_capacity(n_features);

        for j in 0..n_features {
            let col = attributions.column(j);
            let mean = col.mean().unwrap_or(0.0);
            let std = col.std(0.0);

            mean_attributions.push(mean);
            std_attributions.push(std);
        }

        // Compute feature rankings
        let mut indexed: Vec<(usize, f64)> = mean_attributions
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v.abs()))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let feature_rankings: Vec<usize> = indexed.iter().map(|(i, _)| *i).collect();

        // Placeholder for correlations (would need predictions)
        let feature_correlations = vec![0.0; n_features];

        // Compute stability score (how consistent are the rankings across samples)
        let stability_score = if n_samples > 1 {
            let mut rank_variance = 0.0;
            for j in 0..n_features {
                let col = attributions.column(j);
                let mean = col.mean().unwrap_or(0.0);
                let var = col.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n_samples as f64;
                rank_variance += var;
            }
            1.0 / (1.0 + rank_variance / n_features as f64)
        } else {
            1.0
        };

        Self {
            mean_attributions,
            std_attributions,
            feature_rankings,
            feature_correlations,
            stability_score,
        }
    }

    /// Get top N most important features
    pub fn top_features(&self, n: usize) -> Vec<(usize, f64)> {
        self.feature_rankings
            .iter()
            .take(n)
            .map(|&idx| (idx, self.mean_attributions[idx]))
            .collect()
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Compute MSE between predictions and targets
fn compute_mse(predictions: &[f64], targets: &[f64]) -> f64 {
    predictions
        .iter()
        .zip(targets.iter())
        .map(|(p, t)| (p - t).powi(2))
        .sum::<f64>()
        / predictions.len() as f64
}

/// Compute feature direction based on correlation with predictions
fn compute_feature_direction(data: &Array2<f64>, feature_idx: usize, predictions: &[f64]) -> f64 {
    let n = data.nrows();
    if n < 2 {
        return 0.0;
    }

    let feature: Vec<f64> = (0..n).map(|i| data[[i, feature_idx]]).collect();

    let mean_f: f64 = feature.iter().sum::<f64>() / n as f64;
    let mean_p: f64 = predictions.iter().sum::<f64>() / n as f64;

    let covariance: f64 = feature
        .iter()
        .zip(predictions.iter())
        .map(|(f, p)| (f - mean_f) * (p - mean_p))
        .sum::<f64>();

    if covariance > 0.0 {
        1.0
    } else if covariance < 0.0 {
        -1.0
    } else {
        0.0
    }
}

/// Compute binomial coefficient
fn binomial(n: usize, k: usize) -> f64 {
    if k > n {
        return 0.0;
    }
    if k == 0 || k == n {
        return 1.0;
    }

    let k = k.min(n - k);
    let mut result = 1.0;

    for i in 0..k {
        result = result * (n - i) as f64 / (i + 1) as f64;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::TradingModel;

    #[test]
    fn test_permutation_importance() {
        let model = TradingModel::new(5, vec![10], 1);
        let perm = PermutationImportance::new(10).with_seed(42);

        let input = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5]);
        let attrs = perm.explain(&model, &input, None);

        assert_eq!(attrs.len(), 5);
    }

    #[test]
    fn test_gradient_attribution() {
        let model = TradingModel::new(5, vec![10], 1);
        let grad_attr = GradientAttribution::new(100);

        let input = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5]);
        let attrs = grad_attr.explain(&model, &input, None);

        assert_eq!(attrs.len(), 5);
    }

    #[test]
    fn test_integrated_gradients_completeness() {
        let model = TradingModel::new(5, vec![10], 1);
        let grad_attr = GradientAttribution::new(200);

        let input = Array1::from_vec(vec![0.5, 0.5, 0.5, 0.5, 0.5]);
        let delta = grad_attr.convergence_delta(&model, &input, None);

        // Delta should be small for sufficient steps
        assert!(delta < 0.5, "Delta too large: {}", delta);
    }

    #[test]
    fn test_shapley_approximation() {
        let model = TradingModel::new(5, vec![10], 1);
        let shap = ShapleyApproximation::new(50).with_seed(42);

        let input = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5]);
        let attrs = shap.explain(&model, &input, None);

        assert_eq!(attrs.len(), 5);

        // SHAP values should approximately sum to prediction - baseline
        let baseline = Array1::zeros(5);
        let pred_diff = model.forward(&input)[0] - model.forward(&baseline)[0];
        let attr_sum = attrs.sum();

        // Allow some tolerance for approximation
        assert!((attr_sum - pred_diff).abs() < 0.5,
            "SHAP sum {} differs from prediction diff {}", attr_sum, pred_diff);
    }

    #[test]
    fn test_batch_attribution() {
        let model = TradingModel::new(5, vec![10], 1);
        let grad_attr = GradientAttribution::new(50);

        let inputs = Array2::from_shape_fn((3, 5), |(i, j)| (i + j) as f64 * 0.1);
        let attrs = grad_attr.explain_batch(&model, &inputs);

        assert_eq!(attrs.shape(), &[3, 5]);
    }

    #[test]
    fn test_attribution_result() {
        let result = AttributionResult {
            attributions: vec![0.1, -0.3, 0.5, -0.1, 0.2],
            prediction: 0.7,
            baseline_prediction: Some(0.3),
            delta: Some(0.01),
            method: "Test".to_string(),
            metadata: HashMap::new(),
        };

        let top_3 = result.top_k(3);
        assert_eq!(top_3.len(), 3);
        assert_eq!(top_3[0].0, 2); // Feature 2 has highest absolute attribution (0.5)

        let normalized = result.normalized();
        let sum: f64 = normalized.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_attribution_analysis() {
        let attributions = Array2::from_shape_fn((10, 5), |(i, j)| {
            ((i as f64 + 1.0) * (j as f64 + 1.0) * 0.1).sin()
        });

        let analysis = AttributionAnalysis::from_batch(&attributions);

        assert_eq!(analysis.mean_attributions.len(), 5);
        assert_eq!(analysis.feature_rankings.len(), 5);
        assert!(analysis.stability_score > 0.0);
    }

    #[test]
    fn test_binomial() {
        assert_eq!(binomial(5, 0), 1.0);
        assert_eq!(binomial(5, 5), 1.0);
        assert_eq!(binomial(5, 2), 10.0);
        assert_eq!(binomial(10, 3), 120.0);
    }
}
