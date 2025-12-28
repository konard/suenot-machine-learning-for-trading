//! Gradient Boosting Machine implementation
//!
//! This module provides a wrapper around the smartcore GBM implementation
//! with additional utilities for training, prediction, and evaluation.

use crate::data::Dataset;
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use smartcore::ensemble::gradient_boosting_classifier::GradientBoostingClassifier;
use smartcore::ensemble::gradient_boosting_regressor::GradientBoostingRegressor;
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::metrics::{accuracy, mean_squared_error, r2};
use std::collections::HashMap;
use thiserror::Error;
use tracing::info;

/// Errors that can occur with the model
#[derive(Error, Debug)]
pub enum ModelError {
    #[error("Training failed: {0}")]
    TrainingFailed(String),

    #[error("Prediction failed: {0}")]
    PredictionFailed(String),

    #[error("Invalid data: {0}")]
    InvalidData(String),

    #[error("Model not trained")]
    NotTrained,

    #[error("Serialization failed: {0}")]
    SerializationFailed(String),
}

/// GBM hyperparameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GbmParams {
    /// Number of boosting iterations (trees)
    pub n_estimators: usize,
    /// Maximum depth of each tree
    pub max_depth: u16,
    /// Learning rate (shrinkage)
    pub learning_rate: f64,
    /// Minimum samples required to split a node
    pub min_samples_split: usize,
    /// Minimum samples required in a leaf node
    pub min_samples_leaf: usize,
    /// Subsample ratio of the training instances
    pub subsample: f64,
}

impl Default for GbmParams {
    fn default() -> Self {
        Self {
            n_estimators: 100,
            max_depth: 5,
            learning_rate: 0.1,
            min_samples_split: 2,
            min_samples_leaf: 1,
            subsample: 1.0,
        }
    }
}

/// Model evaluation metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetrics {
    /// Mean squared error (for regression)
    pub mse: Option<f64>,
    /// Root mean squared error
    pub rmse: Option<f64>,
    /// R-squared score
    pub r2: Option<f64>,
    /// Mean absolute error
    pub mae: Option<f64>,
    /// Accuracy (for classification)
    pub accuracy: Option<f64>,
    /// Directional accuracy (% of correct direction predictions)
    pub directional_accuracy: Option<f64>,
}

impl ModelMetrics {
    /// Calculate regression metrics
    pub fn regression(y_true: &[f64], y_pred: &[f64]) -> Self {
        let n = y_true.len();
        if n == 0 || n != y_pred.len() {
            return Self {
                mse: None,
                rmse: None,
                r2: None,
                mae: None,
                accuracy: None,
                directional_accuracy: None,
            };
        }

        let mse_val: f64 = y_true
            .iter()
            .zip(y_pred.iter())
            .map(|(t, p)| (t - p).powi(2))
            .sum::<f64>()
            / n as f64;

        let mae_val: f64 = y_true
            .iter()
            .zip(y_pred.iter())
            .map(|(t, p)| (t - p).abs())
            .sum::<f64>()
            / n as f64;

        let mean_true: f64 = y_true.iter().sum::<f64>() / n as f64;
        let ss_tot: f64 = y_true.iter().map(|t| (t - mean_true).powi(2)).sum();
        let ss_res: f64 = y_true
            .iter()
            .zip(y_pred.iter())
            .map(|(t, p)| (t - p).powi(2))
            .sum();

        let r2_val = if ss_tot != 0.0 {
            1.0 - ss_res / ss_tot
        } else {
            0.0
        };

        // Directional accuracy
        let mut correct_direction = 0;
        let mut total_direction = 0;
        for (t, p) in y_true.iter().zip(y_pred.iter()) {
            if *t != 0.0 {
                total_direction += 1;
                if (*t > 0.0 && *p > 0.0) || (*t < 0.0 && *p < 0.0) {
                    correct_direction += 1;
                }
            }
        }
        let dir_acc = if total_direction > 0 {
            Some(correct_direction as f64 / total_direction as f64 * 100.0)
        } else {
            None
        };

        Self {
            mse: Some(mse_val),
            rmse: Some(mse_val.sqrt()),
            r2: Some(r2_val),
            mae: Some(mae_val),
            accuracy: None,
            directional_accuracy: dir_acc,
        }
    }

    /// Calculate classification metrics
    pub fn classification(y_true: &[f64], y_pred: &[f64]) -> Self {
        let n = y_true.len();
        if n == 0 || n != y_pred.len() {
            return Self {
                mse: None,
                rmse: None,
                r2: None,
                mae: None,
                accuracy: None,
                directional_accuracy: None,
            };
        }

        let correct = y_true
            .iter()
            .zip(y_pred.iter())
            .filter(|(t, p)| (*t - *p).abs() < 1e-10)
            .count();

        Self {
            mse: None,
            rmse: None,
            r2: None,
            mae: None,
            accuracy: Some(correct as f64 / n as f64 * 100.0),
            directional_accuracy: None,
        }
    }
}

/// Gradient Boosting Regressor wrapper
#[derive(Debug)]
pub struct GbmRegressor {
    params: GbmParams,
    model: Option<GradientBoostingRegressor<f64, i32, DenseMatrix<f64>, Vec<f64>>>,
    feature_names: Vec<String>,
    feature_importance: HashMap<String, f64>,
}

impl GbmRegressor {
    /// Create a new GBM regressor with default parameters
    pub fn new() -> Self {
        Self::with_params(GbmParams::default())
    }

    /// Create a new GBM regressor with custom parameters
    pub fn with_params(params: GbmParams) -> Self {
        Self {
            params,
            model: None,
            feature_names: Vec::new(),
            feature_importance: HashMap::new(),
        }
    }

    /// Train the model on a dataset
    pub fn fit(&mut self, dataset: &Dataset) -> Result<(), ModelError> {
        if dataset.is_empty() {
            return Err(ModelError::InvalidData("Empty dataset".to_string()));
        }

        let n_samples = dataset.len();
        let n_features = dataset.num_features();

        // Flatten features into a single vector (row-major order)
        let x_data: Vec<f64> = dataset.features.iter().flatten().copied().collect();
        let x = DenseMatrix::from_2d_vec(&dataset.features)
            .map_err(|e| ModelError::InvalidData(format!("Failed to create feature matrix: {:?}", e)))?;

        let y = dataset.targets.clone();

        info!(
            "Training GBM regressor with {} samples and {} features",
            n_samples, n_features
        );
        info!("Parameters: {:?}", self.params);

        let model = GradientBoostingRegressor::fit(
            &x,
            &y,
            smartcore::ensemble::gradient_boosting_regressor::GradientBoostingRegressorParameters::default()
                .with_n_trees(self.params.n_estimators)
                .with_max_depth(self.params.max_depth)
                .with_learning_rate(self.params.learning_rate)
                .with_min_samples_split(self.params.min_samples_split)
                .with_min_samples_leaf(self.params.min_samples_leaf),
        )
        .map_err(|e| ModelError::TrainingFailed(format!("{:?}", e)))?;

        self.model = Some(model);
        self.feature_names = dataset.feature_names.clone();

        info!("Model training completed successfully");

        Ok(())
    }

    /// Make predictions on new data
    pub fn predict(&self, features: &[Vec<f64>]) -> Result<Vec<f64>, ModelError> {
        let model = self.model.as_ref().ok_or(ModelError::NotTrained)?;

        let x = DenseMatrix::from_2d_vec(features)
            .map_err(|e| ModelError::PredictionFailed(format!("Failed to create feature matrix: {:?}", e)))?;

        let predictions = model
            .predict(&x)
            .map_err(|e| ModelError::PredictionFailed(format!("{:?}", e)))?;

        Ok(predictions)
    }

    /// Predict on a dataset
    pub fn predict_dataset(&self, dataset: &Dataset) -> Result<Vec<f64>, ModelError> {
        self.predict(&dataset.features)
    }

    /// Evaluate the model on a test dataset
    pub fn evaluate(&self, dataset: &Dataset) -> Result<ModelMetrics, ModelError> {
        let predictions = self.predict_dataset(dataset)?;
        Ok(ModelMetrics::regression(&dataset.targets, &predictions))
    }

    /// Get model parameters
    pub fn params(&self) -> &GbmParams {
        &self.params
    }

    /// Check if the model is trained
    pub fn is_trained(&self) -> bool {
        self.model.is_some()
    }
}

impl Default for GbmRegressor {
    fn default() -> Self {
        Self::new()
    }
}

/// Gradient Boosting Classifier wrapper for direction prediction
#[derive(Debug)]
pub struct GbmClassifier {
    params: GbmParams,
    model: Option<GradientBoostingClassifier<f64, i32, DenseMatrix<f64>, Vec<i32>>>,
    feature_names: Vec<String>,
}

impl GbmClassifier {
    /// Create a new GBM classifier with default parameters
    pub fn new() -> Self {
        Self::with_params(GbmParams::default())
    }

    /// Create a new GBM classifier with custom parameters
    pub fn with_params(params: GbmParams) -> Self {
        Self {
            params,
            model: None,
            feature_names: Vec::new(),
        }
    }

    /// Convert regression targets to classification labels
    /// Returns 1 for positive returns, 0 for negative/zero returns
    pub fn to_labels(targets: &[f64]) -> Vec<i32> {
        targets
            .iter()
            .map(|t| if *t > 0.0 { 1 } else { 0 })
            .collect()
    }

    /// Train the model on a dataset
    pub fn fit(&mut self, dataset: &Dataset) -> Result<(), ModelError> {
        if dataset.is_empty() {
            return Err(ModelError::InvalidData("Empty dataset".to_string()));
        }

        let n_samples = dataset.len();
        let n_features = dataset.num_features();

        let x = DenseMatrix::from_2d_vec(&dataset.features)
            .map_err(|e| ModelError::InvalidData(format!("Failed to create feature matrix: {:?}", e)))?;

        let y: Vec<i32> = Self::to_labels(&dataset.targets);

        info!(
            "Training GBM classifier with {} samples and {} features",
            n_samples, n_features
        );

        let model = GradientBoostingClassifier::fit(
            &x,
            &y,
            smartcore::ensemble::gradient_boosting_classifier::GradientBoostingClassifierParameters::default()
                .with_n_trees(self.params.n_estimators)
                .with_max_depth(self.params.max_depth)
                .with_learning_rate(self.params.learning_rate)
                .with_min_samples_split(self.params.min_samples_split)
                .with_min_samples_leaf(self.params.min_samples_leaf),
        )
        .map_err(|e| ModelError::TrainingFailed(format!("{:?}", e)))?;

        self.model = Some(model);
        self.feature_names = dataset.feature_names.clone();

        info!("Classifier training completed successfully");

        Ok(())
    }

    /// Make predictions on new data
    pub fn predict(&self, features: &[Vec<f64>]) -> Result<Vec<i32>, ModelError> {
        let model = self.model.as_ref().ok_or(ModelError::NotTrained)?;

        let x = DenseMatrix::from_2d_vec(features)
            .map_err(|e| ModelError::PredictionFailed(format!("Failed to create feature matrix: {:?}", e)))?;

        let predictions = model
            .predict(&x)
            .map_err(|e| ModelError::PredictionFailed(format!("{:?}", e)))?;

        Ok(predictions)
    }

    /// Predict on a dataset
    pub fn predict_dataset(&self, dataset: &Dataset) -> Result<Vec<i32>, ModelError> {
        self.predict(&dataset.features)
    }

    /// Evaluate the model on a test dataset
    pub fn evaluate(&self, dataset: &Dataset) -> Result<ModelMetrics, ModelError> {
        let predictions = self.predict_dataset(dataset)?;
        let y_true: Vec<f64> = Self::to_labels(&dataset.targets)
            .iter()
            .map(|&x| x as f64)
            .collect();
        let y_pred: Vec<f64> = predictions.iter().map(|&x| x as f64).collect();

        Ok(ModelMetrics::classification(&y_true, &y_pred))
    }

    /// Check if the model is trained
    pub fn is_trained(&self) -> bool {
        self.model.is_some()
    }
}

impl Default for GbmClassifier {
    fn default() -> Self {
        Self::new()
    }
}

/// Cross-validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossValidationResult {
    /// Metrics for each fold
    pub fold_metrics: Vec<ModelMetrics>,
    /// Mean metrics across all folds
    pub mean_metrics: ModelMetrics,
    /// Standard deviation of metrics
    pub std_metrics: HashMap<String, f64>,
}

/// Perform time-series cross-validation
pub fn time_series_cv(
    dataset: &Dataset,
    params: &GbmParams,
    n_splits: usize,
) -> Result<CrossValidationResult, ModelError> {
    if dataset.len() < n_splits * 2 {
        return Err(ModelError::InvalidData(
            "Dataset too small for cross-validation".to_string(),
        ));
    }

    let fold_size = dataset.len() / (n_splits + 1);
    let mut fold_metrics = Vec::new();

    info!(
        "Performing time-series cross-validation with {} splits",
        n_splits
    );

    for i in 0..n_splits {
        let train_end = (i + 1) * fold_size;
        let test_start = train_end;
        let test_end = test_start + fold_size;

        if test_end > dataset.len() {
            break;
        }

        // Create train/test datasets
        let train = Dataset {
            feature_names: dataset.feature_names.clone(),
            features: dataset.features[..train_end].to_vec(),
            targets: dataset.targets[..train_end].to_vec(),
            timestamps: dataset.timestamps[..train_end].to_vec(),
            symbol: dataset.symbol.clone(),
        };

        let test = Dataset {
            feature_names: dataset.feature_names.clone(),
            features: dataset.features[test_start..test_end].to_vec(),
            targets: dataset.targets[test_start..test_end].to_vec(),
            timestamps: dataset.timestamps[test_start..test_end].to_vec(),
            symbol: dataset.symbol.clone(),
        };

        // Train and evaluate
        let mut model = GbmRegressor::with_params(params.clone());
        model.fit(&train)?;
        let metrics = model.evaluate(&test)?;

        info!("Fold {}: RMSE={:.4}, R2={:.4}", i + 1,
              metrics.rmse.unwrap_or(f64::NAN),
              metrics.r2.unwrap_or(f64::NAN));

        fold_metrics.push(metrics);
    }

    // Calculate mean metrics
    let n_folds = fold_metrics.len() as f64;
    let mean_rmse: f64 = fold_metrics.iter().filter_map(|m| m.rmse).sum::<f64>() / n_folds;
    let mean_r2: f64 = fold_metrics.iter().filter_map(|m| m.r2).sum::<f64>() / n_folds;
    let mean_mae: f64 = fold_metrics.iter().filter_map(|m| m.mae).sum::<f64>() / n_folds;
    let mean_dir_acc: f64 = fold_metrics
        .iter()
        .filter_map(|m| m.directional_accuracy)
        .sum::<f64>()
        / n_folds;

    let mean_metrics = ModelMetrics {
        mse: Some(mean_rmse.powi(2)),
        rmse: Some(mean_rmse),
        r2: Some(mean_r2),
        mae: Some(mean_mae),
        accuracy: None,
        directional_accuracy: Some(mean_dir_acc),
    };

    // Calculate standard deviations
    let mut std_metrics = HashMap::new();
    let rmse_std: f64 = (fold_metrics
        .iter()
        .filter_map(|m| m.rmse)
        .map(|x| (x - mean_rmse).powi(2))
        .sum::<f64>()
        / n_folds)
        .sqrt();
    std_metrics.insert("rmse".to_string(), rmse_std);

    Ok(CrossValidationResult {
        fold_metrics,
        mean_metrics,
        std_metrics,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn create_test_dataset(n: usize) -> Dataset {
        let mut dataset = Dataset::new(
            "TEST".to_string(),
            vec!["feature1".to_string(), "feature2".to_string()],
        );

        for i in 0..n {
            let x1 = i as f64;
            let x2 = (i as f64 * 0.5).sin();
            let target = x1 * 0.5 + x2 * 2.0 + 0.1;
            dataset.add_sample(vec![x1, x2], target, Utc::now());
        }

        dataset
    }

    #[test]
    fn test_gbm_regressor() {
        let dataset = create_test_dataset(200);
        let (train, test) = dataset.train_test_split(0.8);

        let mut model = GbmRegressor::new();
        model.fit(&train).unwrap();

        let metrics = model.evaluate(&test).unwrap();
        assert!(metrics.rmse.is_some());
        assert!(metrics.r2.is_some());
    }
}
