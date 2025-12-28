//! Logistic Regression for binary classification
//!
//! This module provides logistic regression for predicting price direction
//! (up or down) in cryptocurrency markets.

use ndarray::{Array1, Array2, Axis};
use thiserror::Error;

/// Errors for logistic regression
#[derive(Error, Debug)]
pub enum LogisticRegressionError {
    #[error("Model has not been fitted yet")]
    NotFitted,

    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("Invalid regularization parameter")]
    InvalidParameter,

    #[error("Convergence failed after {0} iterations")]
    ConvergenceFailed(usize),
}

/// Regularization type for logistic regression
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Regularization {
    /// No regularization
    None,
    /// L1 regularization (Lasso)
    L1(f64),
    /// L2 regularization (Ridge)
    L2(f64),
    /// Elastic Net (L1 + L2)
    ElasticNet { l1: f64, l2: f64 },
}

/// Logistic Regression classifier
#[derive(Debug, Clone)]
pub struct LogisticRegression {
    /// Fitted coefficients
    pub coefficients: Option<Array1<f64>>,
    /// Intercept term
    pub intercept: Option<f64>,
    /// Learning rate
    learning_rate: f64,
    /// Maximum iterations
    max_iter: usize,
    /// Convergence tolerance
    tolerance: f64,
    /// Whether to fit intercept
    fit_intercept: bool,
    /// Regularization type
    regularization: Regularization,
    /// Cost history during training
    pub cost_history: Vec<f64>,
}

impl Default for LogisticRegression {
    fn default() -> Self {
        Self::new(0.01, 1000, 1e-6, true, Regularization::None)
    }
}

impl LogisticRegression {
    /// Create a new Logistic Regression model
    pub fn new(
        learning_rate: f64,
        max_iter: usize,
        tolerance: f64,
        fit_intercept: bool,
        regularization: Regularization,
    ) -> Self {
        Self {
            coefficients: None,
            intercept: None,
            learning_rate,
            max_iter,
            tolerance,
            fit_intercept,
            regularization,
            cost_history: Vec::new(),
        }
    }

    /// Create with L2 regularization
    pub fn with_l2(c: f64) -> Self {
        Self::new(0.01, 1000, 1e-6, true, Regularization::L2(1.0 / c))
    }

    /// Create with L1 regularization
    pub fn with_l1(c: f64) -> Self {
        Self::new(0.01, 1000, 1e-6, true, Regularization::L1(1.0 / c))
    }

    /// Sigmoid activation function
    fn sigmoid(z: f64) -> f64 {
        if z >= 0.0 {
            1.0 / (1.0 + (-z).exp())
        } else {
            let exp_z = z.exp();
            exp_z / (1.0 + exp_z)
        }
    }

    /// Compute sigmoid for array
    fn sigmoid_array(z: &Array1<f64>) -> Array1<f64> {
        z.mapv(Self::sigmoid)
    }

    /// Compute log loss (binary cross-entropy)
    fn log_loss(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
        let eps = 1e-15;
        let n = y_true.len() as f64;

        -y_true
            .iter()
            .zip(y_pred.iter())
            .map(|(&y, &p)| {
                let p_clipped = p.clamp(eps, 1.0 - eps);
                y * p_clipped.ln() + (1.0 - y) * (1.0 - p_clipped).ln()
            })
            .sum::<f64>()
            / n
    }

    /// Fit using gradient descent
    pub fn fit(
        &mut self,
        x: &Array2<f64>,
        y: &Array1<f64>,
    ) -> Result<(), LogisticRegressionError> {
        let n_samples = x.nrows() as f64;
        let n_features = x.ncols();

        // Initialize weights
        let mut weights = Array1::<f64>::zeros(n_features);
        let mut bias = 0.0;

        self.cost_history.clear();

        for iter in 0..self.max_iter {
            // Forward pass
            let linear = x.dot(&weights) + bias;
            let predictions = Self::sigmoid_array(&linear);

            // Compute gradients
            let errors = &predictions - y;
            let mut dw = x.t().dot(&errors) / n_samples;
            let db = errors.sum() / n_samples;

            // Add regularization gradient
            match self.regularization {
                Regularization::L2(alpha) => {
                    dw = &dw + &(&weights * alpha);
                }
                Regularization::L1(alpha) => {
                    let sign = weights.mapv(|w| {
                        if w > 0.0 {
                            1.0
                        } else if w < 0.0 {
                            -1.0
                        } else {
                            0.0
                        }
                    });
                    dw = &dw + &(&sign * alpha);
                }
                Regularization::ElasticNet { l1, l2 } => {
                    let sign = weights.mapv(|w| w.signum());
                    dw = &dw + &(&weights * l2) + &(&sign * l1);
                }
                Regularization::None => {}
            }

            // Update weights
            weights = &weights - &(&dw * self.learning_rate);
            if self.fit_intercept {
                bias -= self.learning_rate * db;
            }

            // Compute cost
            let cost = Self::log_loss(y, &predictions);
            self.cost_history.push(cost);

            // Check convergence
            if iter > 0 {
                let cost_diff = (self.cost_history[iter - 1] - cost).abs();
                if cost_diff < self.tolerance {
                    log::debug!("Converged at iteration {}", iter);
                    break;
                }
            }
        }

        self.coefficients = Some(weights);
        self.intercept = Some(bias);

        Ok(())
    }

    /// Predict probabilities
    pub fn predict_proba(
        &self,
        x: &Array2<f64>,
    ) -> Result<Array1<f64>, LogisticRegressionError> {
        let weights = self
            .coefficients
            .as_ref()
            .ok_or(LogisticRegressionError::NotFitted)?;
        let bias = self.intercept.ok_or(LogisticRegressionError::NotFitted)?;

        let linear = x.dot(weights) + bias;
        Ok(Self::sigmoid_array(&linear))
    }

    /// Predict class labels (0 or 1)
    pub fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>, LogisticRegressionError> {
        let proba = self.predict_proba(x)?;
        Ok(proba.mapv(|p| if p >= 0.5 { 1.0 } else { 0.0 }))
    }

    /// Predict with custom threshold
    pub fn predict_with_threshold(
        &self,
        x: &Array2<f64>,
        threshold: f64,
    ) -> Result<Array1<f64>, LogisticRegressionError> {
        let proba = self.predict_proba(x)?;
        Ok(proba.mapv(|p| if p >= threshold { 1.0 } else { 0.0 }))
    }

    /// Get decision function values (log-odds)
    pub fn decision_function(
        &self,
        x: &Array2<f64>,
    ) -> Result<Array1<f64>, LogisticRegressionError> {
        let weights = self
            .coefficients
            .as_ref()
            .ok_or(LogisticRegressionError::NotFitted)?;
        let bias = self.intercept.ok_or(LogisticRegressionError::NotFitted)?;

        Ok(x.dot(weights) + bias)
    }

    /// Get model summary
    pub fn summary(&self, feature_names: Option<&[String]>) -> String {
        let mut s = String::new();
        s.push_str("Logistic Regression Summary\n");
        s.push_str("===========================\n\n");

        if let Some(ref coef) = self.coefficients {
            s.push_str(&format!(
                "Intercept: {:.6}\n\n",
                self.intercept.unwrap_or(0.0)
            ));
            s.push_str("Coefficients (log-odds):\n");

            if let Some(names) = feature_names {
                for (i, (name, &c)) in names.iter().zip(coef.iter()).enumerate() {
                    let odds_ratio = c.exp();
                    s.push_str(&format!(
                        "  {:3}. {:20}: {:>10.6} (OR: {:.4})\n",
                        i + 1,
                        name,
                        c,
                        odds_ratio
                    ));
                }
            } else {
                for (i, &c) in coef.iter().enumerate() {
                    let odds_ratio = c.exp();
                    s.push_str(&format!(
                        "  {:3}. Feature {:2}: {:>10.6} (OR: {:.4})\n",
                        i + 1,
                        i,
                        c,
                        odds_ratio
                    ));
                }
            }

            s.push_str(&format!(
                "\nFinal cost: {:.6}\n",
                self.cost_history.last().unwrap_or(&0.0)
            ));
        } else {
            s.push_str("Model not fitted yet.\n");
        }

        s
    }
}

/// Multinomial Logistic Regression (Softmax)
#[derive(Debug, Clone)]
pub struct MultinomialLogisticRegression {
    /// Coefficients matrix (n_classes x n_features)
    pub coefficients: Option<Array2<f64>>,
    /// Intercepts (n_classes)
    pub intercepts: Option<Array1<f64>>,
    /// Number of classes
    n_classes: usize,
    /// Learning rate
    learning_rate: f64,
    /// Maximum iterations
    max_iter: usize,
    /// Convergence tolerance
    tolerance: f64,
}

impl MultinomialLogisticRegression {
    /// Create a new multinomial logistic regression
    pub fn new(n_classes: usize, learning_rate: f64, max_iter: usize, tolerance: f64) -> Self {
        Self {
            coefficients: None,
            intercepts: None,
            n_classes,
            learning_rate,
            max_iter,
            tolerance,
        }
    }

    /// Softmax function
    fn softmax(z: &Array1<f64>) -> Array1<f64> {
        let max_z = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_z = z.mapv(|x| (x - max_z).exp());
        let sum = exp_z.sum();
        exp_z / sum
    }

    /// Fit the model
    pub fn fit(
        &mut self,
        x: &Array2<f64>,
        y: &Array1<usize>,
    ) -> Result<(), LogisticRegressionError> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Initialize weights
        let mut weights = Array2::<f64>::zeros((self.n_classes, n_features));
        let mut biases = Array1::<f64>::zeros(self.n_classes);

        // One-hot encode y
        let mut y_onehot = Array2::<f64>::zeros((n_samples, self.n_classes));
        for (i, &class) in y.iter().enumerate() {
            if class < self.n_classes {
                y_onehot[[i, class]] = 1.0;
            }
        }

        for _iter in 0..self.max_iter {
            let weights_old = weights.clone();

            // Forward pass
            let mut proba = Array2::<f64>::zeros((n_samples, self.n_classes));
            for i in 0..n_samples {
                let linear = weights.dot(&x.row(i).to_owned()) + &biases;
                proba.row_mut(i).assign(&Self::softmax(&linear));
            }

            // Compute gradients
            let errors = &proba - &y_onehot;

            for c in 0..self.n_classes {
                let dw = x.t().dot(&errors.column(c)) / n_samples as f64;
                let db = errors.column(c).sum() / n_samples as f64;

                for j in 0..n_features {
                    weights[[c, j]] -= self.learning_rate * dw[j];
                }
                biases[c] -= self.learning_rate * db;
            }

            // Check convergence
            let diff: f64 = weights
                .iter()
                .zip(weights_old.iter())
                .map(|(&a, &b)| (a - b).abs())
                .sum();

            if diff < self.tolerance {
                break;
            }
        }

        self.coefficients = Some(weights);
        self.intercepts = Some(biases);

        Ok(())
    }

    /// Predict class probabilities
    pub fn predict_proba(
        &self,
        x: &Array2<f64>,
    ) -> Result<Array2<f64>, LogisticRegressionError> {
        let weights = self
            .coefficients
            .as_ref()
            .ok_or(LogisticRegressionError::NotFitted)?;
        let biases = self
            .intercepts
            .as_ref()
            .ok_or(LogisticRegressionError::NotFitted)?;

        let n_samples = x.nrows();
        let mut proba = Array2::<f64>::zeros((n_samples, self.n_classes));

        for i in 0..n_samples {
            let linear = weights.dot(&x.row(i).to_owned()) + biases;
            proba.row_mut(i).assign(&Self::softmax(&linear));
        }

        Ok(proba)
    }

    /// Predict class labels
    pub fn predict(&self, x: &Array2<f64>) -> Result<Array1<usize>, LogisticRegressionError> {
        let proba = self.predict_proba(x)?;

        let predictions: Vec<usize> = proba
            .rows()
            .into_iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            })
            .collect();

        Ok(Array1::from_vec(predictions))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid() {
        assert!((LogisticRegression::sigmoid(0.0) - 0.5).abs() < 1e-10);
        assert!(LogisticRegression::sigmoid(100.0) > 0.99);
        assert!(LogisticRegression::sigmoid(-100.0) < 0.01);
    }

    #[test]
    fn test_logistic_regression_fit() {
        // Simple linearly separable data
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 5.0, 5.0, 5.5, 5.5, 6.0, 6.0],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

        let mut model = LogisticRegression::new(0.5, 1000, 1e-6, true, Regularization::None);
        model.fit(&x, &y).unwrap();

        let predictions = model.predict(&x).unwrap();

        // Should correctly classify most points
        let accuracy: f64 = predictions
            .iter()
            .zip(y.iter())
            .filter(|(&p, &a)| (p - a).abs() < 0.5)
            .count() as f64
            / y.len() as f64;

        assert!(accuracy >= 0.8);
    }

    #[test]
    fn test_logistic_with_regularization() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 5.0, 5.0, 5.5, 5.5, 6.0, 6.0],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

        let mut model = LogisticRegression::with_l2(1.0);
        model.fit(&x, &y).unwrap();

        assert!(model.coefficients.is_some());

        // L2 regularization should shrink coefficients
        let coef = model.coefficients.as_ref().unwrap();
        let coef_norm: f64 = coef.iter().map(|c| c * c).sum::<f64>().sqrt();
        assert!(coef_norm < 10.0); // Reasonably bounded
    }
}
