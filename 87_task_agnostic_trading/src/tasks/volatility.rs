//! Volatility estimation head for predicting future price volatility

use super::{TaskHead, TaskType};
use ndarray::{Array1, Array2, Axis};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use serde::{Deserialize, Serialize};

/// Volatility level classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VolatilityLevel {
    Low,
    Medium,
    High,
    Extreme,
}

impl VolatilityLevel {
    /// Classify volatility from percentage value
    pub fn from_percentage(vol_pct: f64) -> Self {
        if vol_pct < 1.0 {
            VolatilityLevel::Low
        } else if vol_pct < 3.0 {
            VolatilityLevel::Medium
        } else if vol_pct < 5.0 {
            VolatilityLevel::High
        } else {
            VolatilityLevel::Extreme
        }
    }
}

impl std::fmt::Display for VolatilityLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VolatilityLevel::Low => write!(f, "Low"),
            VolatilityLevel::Medium => write!(f, "Medium"),
            VolatilityLevel::High => write!(f, "High"),
            VolatilityLevel::Extreme => write!(f, "Extreme"),
        }
    }
}

/// Volatility prediction result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolatilityPrediction {
    /// Predicted volatility as percentage
    pub volatility_pct: f64,
    /// Volatility level classification
    pub level: VolatilityLevel,
    /// Prediction confidence (based on uncertainty estimation)
    pub confidence: f64,
    /// Lower bound of prediction interval
    pub lower_bound: f64,
    /// Upper bound of prediction interval
    pub upper_bound: f64,
}

impl VolatilityPrediction {
    /// Create from raw output (mean and log_var)
    pub fn from_output(mean: f64, log_var: f64) -> Self {
        let volatility_pct = mean.abs() * 100.0;
        let std = (log_var.exp()).sqrt();
        let confidence = 1.0 / (1.0 + std);

        // Compute bounds properly ensuring lower <= upper
        let lower_raw = (mean.abs() - 2.0 * std).max(0.0) * 100.0;
        let upper_raw = (mean.abs() + 2.0 * std) * 100.0;

        Self {
            volatility_pct,
            level: VolatilityLevel::from_percentage(volatility_pct),
            confidence,
            lower_bound: lower_raw.min(upper_raw),
            upper_bound: lower_raw.max(upper_raw),
        }
    }

    /// Check if volatility is actionable (not extreme uncertainty)
    pub fn is_actionable(&self) -> bool {
        self.confidence > 0.3 && (self.upper_bound - self.lower_bound) < 10.0
    }
}

/// Volatility head configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolatilityConfig {
    /// Input embedding dimension
    pub embedding_dim: usize,
    /// Hidden layer dimensions
    pub hidden_dims: Vec<usize>,
    /// Whether to output uncertainty (log variance)
    pub with_uncertainty: bool,
    /// Dropout rate
    pub dropout: f64,
}

impl Default for VolatilityConfig {
    fn default() -> Self {
        Self {
            embedding_dim: 64,
            hidden_dims: vec![32, 16],
            with_uncertainty: true,
            dropout: 0.1,
        }
    }
}

/// Volatility estimation head
pub struct VolatilityHead {
    config: VolatilityConfig,
    layers: Vec<Array2<f64>>,
    output_mean: Array2<f64>,
    output_var: Array2<f64>,
}

impl VolatilityHead {
    /// Create a new volatility head
    pub fn new(config: VolatilityConfig) -> Self {
        let mut layers = Vec::new();
        let mut prev_dim = config.embedding_dim;

        for &hidden_dim in &config.hidden_dims {
            let scale = (2.0 / (prev_dim + hidden_dim) as f64).sqrt();
            layers.push(Array2::random(
                (prev_dim, hidden_dim),
                Uniform::new(-scale, scale),
            ));
            prev_dim = hidden_dim;
        }

        let last_dim = *config.hidden_dims.last().unwrap_or(&config.embedding_dim);
        let scale_out = (2.0 / (last_dim + 1) as f64).sqrt();

        let output_mean = Array2::random((last_dim, 1), Uniform::new(-scale_out, scale_out));
        let output_var = Array2::random((last_dim, 1), Uniform::new(-scale_out, scale_out));

        Self {
            config,
            layers,
            output_mean,
            output_var,
        }
    }

    /// Predict volatility from embedding
    pub fn predict(&self, embedding: &Array1<f64>) -> VolatilityPrediction {
        let output = self.forward(embedding);
        let mean = output[0];
        let log_var = if self.config.with_uncertainty && output.len() > 1 {
            output[1]
        } else {
            -2.0 // default low uncertainty
        };

        VolatilityPrediction::from_output(mean, log_var)
    }

    /// Get configuration
    pub fn config(&self) -> &VolatilityConfig {
        &self.config
    }
}

impl TaskHead for VolatilityHead {
    fn task_type(&self) -> TaskType {
        TaskType::Volatility
    }

    fn forward(&self, embedding: &Array1<f64>) -> Array1<f64> {
        let mut x = embedding.clone();

        // Hidden layers with ReLU
        for layer in &self.layers {
            x = x.dot(layer).mapv(|v| v.max(0.0));
        }

        // Output: mean and log_variance
        let mean = x.dot(&self.output_mean)[[0]];

        if self.config.with_uncertainty {
            let log_var = x.dot(&self.output_var)[[0]];
            Array1::from_vec(vec![mean, log_var])
        } else {
            Array1::from_vec(vec![mean])
        }
    }

    fn forward_batch(&self, embeddings: &Array2<f64>) -> Array2<f64> {
        let output_dim = if self.config.with_uncertainty { 2 } else { 1 };
        let mut outputs = Vec::with_capacity(embeddings.nrows() * output_dim);

        for row in embeddings.axis_iter(Axis(0)) {
            let out = self.forward(&row.to_owned());
            outputs.extend(out.to_vec());
        }

        Array2::from_shape_vec((embeddings.nrows(), output_dim), outputs).expect("Shape mismatch")
    }

    fn parameters(&self) -> Vec<Array2<f64>> {
        let mut params = self.layers.clone();
        params.push(self.output_mean.clone());
        params.push(self.output_var.clone());
        params
    }

    fn update_parameters(&mut self, gradients: &[Array2<f64>], learning_rate: f64) {
        let n_layers = self.layers.len();
        for (i, layer) in self.layers.iter_mut().enumerate() {
            if i < gradients.len() {
                *layer = &*layer - &(&gradients[i] * learning_rate);
            }
        }

        if gradients.len() > n_layers {
            self.output_mean = &self.output_mean - &(&gradients[n_layers] * learning_rate);
        }
        if gradients.len() > n_layers + 1 {
            self.output_var = &self.output_var - &(&gradients[n_layers + 1] * learning_rate);
        }
    }

    fn compute_loss(&self, predictions: &Array2<f64>, targets: &Array2<f64>) -> f64 {
        // Gaussian negative log-likelihood if with_uncertainty, else MSE
        if self.config.with_uncertainty && predictions.ncols() >= 2 {
            let mut total_loss = 0.0;
            let n = predictions.nrows() as f64;

            for (pred_row, target_row) in
                predictions.axis_iter(Axis(0)).zip(targets.axis_iter(Axis(0)))
            {
                let mean = pred_row[0];
                let log_var = pred_row[1];
                let target = target_row[0];

                // NLL: 0.5 * (log_var + (target - mean)^2 / var)
                let var = log_var.exp();
                total_loss += 0.5 * (log_var + (target - mean).powi(2) / var);
            }

            total_loss / n
        } else {
            // MSE loss
            let mut total_loss = 0.0;
            let n = predictions.nrows() as f64;

            for (pred_row, target_row) in
                predictions.axis_iter(Axis(0)).zip(targets.axis_iter(Axis(0)))
            {
                total_loss += (pred_row[0] - target_row[0]).powi(2);
            }

            total_loss / n
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;

    #[test]
    fn test_volatility_level() {
        assert_eq!(VolatilityLevel::from_percentage(0.5), VolatilityLevel::Low);
        assert_eq!(
            VolatilityLevel::from_percentage(2.0),
            VolatilityLevel::Medium
        );
        assert_eq!(VolatilityLevel::from_percentage(4.0), VolatilityLevel::High);
        assert_eq!(
            VolatilityLevel::from_percentage(6.0),
            VolatilityLevel::Extreme
        );
    }

    #[test]
    fn test_volatility_head() {
        let config = VolatilityConfig {
            embedding_dim: 32,
            hidden_dims: vec![16, 8],
            with_uncertainty: true,
            dropout: 0.1,
        };

        let head = VolatilityHead::new(config);
        let embedding = Array::random(32, Uniform::new(-1.0, 1.0));
        let prediction = head.predict(&embedding);

        assert!(prediction.confidence >= 0.0 && prediction.confidence <= 1.0);
        assert!(prediction.lower_bound <= prediction.upper_bound);
    }

    #[test]
    fn test_prediction_actionable() {
        let pred = VolatilityPrediction {
            volatility_pct: 2.5,
            level: VolatilityLevel::Medium,
            confidence: 0.6,
            lower_bound: 1.5,
            upper_bound: 3.5,
        };

        assert!(pred.is_actionable());

        let pred_uncertain = VolatilityPrediction {
            volatility_pct: 5.0,
            level: VolatilityLevel::High,
            confidence: 0.2,
            lower_bound: 0.0,
            upper_bound: 15.0,
        };

        assert!(!pred_uncertain.is_actionable());
    }
}
