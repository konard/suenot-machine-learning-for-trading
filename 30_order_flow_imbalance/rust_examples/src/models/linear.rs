//! # Linear Model
//!
//! Logistic regression for binary classification.

use crate::data::snapshot::FeatureVector;
use serde::{Deserialize, Serialize};

/// Logistic regression model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearModel {
    /// Weights
    weights: Vec<f64>,
    /// Bias
    bias: f64,
    /// Feature names
    feature_names: Vec<String>,
    /// Learning rate
    learning_rate: f64,
    /// Trained flag
    is_trained: bool,
}

impl LinearModel {
    /// Create a new model
    pub fn new(learning_rate: f64) -> Self {
        Self {
            weights: Vec::new(),
            bias: 0.0,
            feature_names: Vec::new(),
            learning_rate,
            is_trained: false,
        }
    }

    /// Train the model using gradient descent
    pub fn train(&mut self, data: &[FeatureVector], epochs: usize) {
        if data.is_empty() || data[0].values.is_empty() {
            return;
        }

        let n_features = data[0].values.len();
        self.feature_names = data[0].names.clone();

        // Initialize weights
        self.weights = vec![0.0; n_features];
        self.bias = 0.0;

        // Gradient descent
        for _ in 0..epochs {
            let mut weight_grads = vec![0.0; n_features];
            let mut bias_grad = 0.0;

            for sample in data {
                let label = sample.label.unwrap_or(0.5);
                let pred = self.predict_proba_internal(&sample.values);
                let error = pred - label;

                for (i, &x) in sample.values.iter().enumerate() {
                    weight_grads[i] += error * x;
                }
                bias_grad += error;
            }

            let n = data.len() as f64;

            // Update weights
            for (i, grad) in weight_grads.iter().enumerate() {
                self.weights[i] -= self.learning_rate * grad / n;
            }
            self.bias -= self.learning_rate * bias_grad / n;
        }

        self.is_trained = true;
    }

    /// Sigmoid function
    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    /// Internal prediction
    fn predict_proba_internal(&self, features: &[f64]) -> f64 {
        let z: f64 = self
            .weights
            .iter()
            .zip(features.iter())
            .map(|(w, x)| w * x)
            .sum::<f64>()
            + self.bias;

        Self::sigmoid(z)
    }

    /// Predict probability
    pub fn predict_proba(&self, features: &FeatureVector) -> f64 {
        if !self.is_trained {
            return 0.5;
        }
        self.predict_proba_internal(&features.values)
    }

    /// Predict class
    pub fn predict(&self, features: &FeatureVector, threshold: f64) -> i32 {
        if self.predict_proba(features) > threshold {
            1
        } else {
            0
        }
    }

    /// Get weights
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// Check if trained
    pub fn is_trained(&self) -> bool {
        self.is_trained
    }
}

impl Default for LinearModel {
    fn default() -> Self {
        Self::new(0.01)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    #[test]
    fn test_linear_model() {
        let mut data = Vec::new();

        for i in 0..100 {
            let mut fv = FeatureVector::new(Utc::now());
            let x = (i as f64) / 50.0 - 1.0;
            fv.add("x", x);
            fv.set_label(if x > 0.0 { 1.0 } else { 0.0 });
            data.push(fv);
        }

        let mut model = LinearModel::new(0.1);
        model.train(&data, 100);

        assert!(model.is_trained());

        let mut fv = FeatureVector::new(Utc::now());
        fv.add("x", 0.5);
        let proba = model.predict_proba(&fv);
        assert!(proba > 0.5);
    }
}
