//! Traits for ensemble models.

use anyhow::Result;
use ndarray::{Array1, Array2};

/// Trait for ensemble models with uncertainty quantification
pub trait EnsembleModel {
    /// Fit the model to training data
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()>;

    /// Make predictions
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>>;

    /// Make predictions with uncertainty estimates
    fn predict_with_uncertainty(&self, x: &Array2<f64>) -> Result<(Array1<f64>, Array1<f64>)>;

    /// Get number of estimators
    fn n_estimators(&self) -> usize;

    /// Check if model is fitted
    fn is_fitted(&self) -> bool;
}

/// Trait for models that support feature importance
pub trait FeatureImportance {
    /// Get feature importances
    fn feature_importances(&self) -> Result<Array1<f64>>;
}

/// Trait for models that support quantile predictions
pub trait QuantilePrediction {
    /// Predict quantiles
    fn predict_quantiles(&self, x: &Array2<f64>, quantiles: &[f64]) -> Result<Array2<f64>>;
}

/// Configuration for ensemble models
#[derive(Debug, Clone)]
pub struct EnsembleConfig {
    pub n_estimators: usize,
    pub max_depth: Option<usize>,
    pub min_samples_split: usize,
    pub min_samples_leaf: usize,
    pub max_features: MaxFeatures,
    pub bootstrap: bool,
    pub random_state: Option<u64>,
}

impl Default for EnsembleConfig {
    fn default() -> Self {
        Self {
            n_estimators: 100,
            max_depth: None,
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_features: MaxFeatures::Sqrt,
            bootstrap: true,
            random_state: Some(42),
        }
    }
}

/// Maximum features to consider at each split
#[derive(Debug, Clone, Copy)]
pub enum MaxFeatures {
    All,
    Sqrt,
    Log2,
    Fraction(f64),
    Fixed(usize),
}

impl MaxFeatures {
    /// Calculate number of features to use
    pub fn calculate(&self, n_features: usize) -> usize {
        match self {
            MaxFeatures::All => n_features,
            MaxFeatures::Sqrt => (n_features as f64).sqrt().ceil() as usize,
            MaxFeatures::Log2 => (n_features as f64).log2().ceil() as usize,
            MaxFeatures::Fraction(f) => ((n_features as f64) * f).ceil() as usize,
            MaxFeatures::Fixed(n) => (*n).min(n_features),
        }
    }
}
