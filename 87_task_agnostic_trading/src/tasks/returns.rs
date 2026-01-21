//! Return prediction head for forecasting expected returns

use super::{TaskHead, TaskType};
use ndarray::{Array1, Array2, Axis};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use serde::{Deserialize, Serialize};

/// Return prediction result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReturnsPrediction {
    /// Predicted return as percentage
    pub return_pct: f64,
    /// Prediction confidence (based on uncertainty)
    pub confidence: f64,
    /// Lower bound (95% interval)
    pub lower_bound: f64,
    /// Upper bound (95% interval)
    pub upper_bound: f64,
    /// Risk-adjusted return (Sharpe-like ratio)
    pub risk_adjusted: f64,
}

impl ReturnsPrediction {
    /// Create from output (mean and log_variance)
    pub fn from_output(mean: f64, log_var: f64) -> Self {
        let return_pct = mean * 100.0;
        let std = (log_var.exp()).sqrt();
        let confidence = 1.0 / (1.0 + std);

        // Risk-adjusted: return / volatility (Sharpe-like)
        let risk_adjusted = if std > 1e-6 { mean / std } else { 0.0 };

        Self {
            return_pct,
            confidence,
            lower_bound: (mean - 1.96 * std) * 100.0,
            upper_bound: (mean + 1.96 * std) * 100.0,
            risk_adjusted,
        }
    }

    /// Check if return is significantly positive
    pub fn is_bullish(&self) -> bool {
        self.lower_bound > 0.0
    }

    /// Check if return is significantly negative
    pub fn is_bearish(&self) -> bool {
        self.upper_bound < 0.0
    }

    /// Check if prediction interval is tight enough to be actionable
    pub fn is_actionable(&self, max_spread_pct: f64) -> bool {
        (self.upper_bound - self.lower_bound) < max_spread_pct
    }
}

/// Returns head configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReturnsConfig {
    /// Input embedding dimension
    pub embedding_dim: usize,
    /// Hidden layer dimensions
    pub hidden_dims: Vec<usize>,
    /// Prediction horizon (in periods)
    pub horizon: usize,
    /// Whether to output uncertainty
    pub with_uncertainty: bool,
    /// Dropout rate
    pub dropout: f64,
}

impl Default for ReturnsConfig {
    fn default() -> Self {
        Self {
            embedding_dim: 64,
            hidden_dims: vec![32, 16],
            horizon: 1,
            with_uncertainty: true,
            dropout: 0.1,
        }
    }
}

/// Returns prediction head
pub struct ReturnsHead {
    config: ReturnsConfig,
    layers: Vec<Array2<f64>>,
    output_mean: Array2<f64>,
    output_var: Array2<f64>,
}

impl ReturnsHead {
    /// Create a new returns head
    pub fn new(config: ReturnsConfig) -> Self {
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

    /// Predict returns from embedding
    pub fn predict(&self, embedding: &Array1<f64>) -> ReturnsPrediction {
        let output = self.forward(embedding);
        let mean = output[0];
        let log_var = if self.config.with_uncertainty && output.len() > 1 {
            output[1]
        } else {
            -2.0 // default low uncertainty
        };

        ReturnsPrediction::from_output(mean, log_var)
    }

    /// Get configuration
    pub fn config(&self) -> &ReturnsConfig {
        &self.config
    }
}

impl TaskHead for ReturnsHead {
    fn task_type(&self) -> TaskType {
        TaskType::Returns
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
        // Gaussian NLL if with_uncertainty, else MSE
        if self.config.with_uncertainty && predictions.ncols() >= 2 {
            let mut total_loss = 0.0;
            let n = predictions.nrows() as f64;

            for (pred_row, target_row) in
                predictions.axis_iter(Axis(0)).zip(targets.axis_iter(Axis(0)))
            {
                let mean = pred_row[0];
                let log_var = pred_row[1];
                let target = target_row[0];

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
    fn test_returns_prediction() {
        let pred = ReturnsPrediction::from_output(0.02, -3.0); // 2% return, low variance

        assert!((pred.return_pct - 2.0).abs() < 0.01);
        assert!(pred.confidence > 0.5);
        assert!(pred.lower_bound < pred.return_pct);
        assert!(pred.upper_bound > pred.return_pct);
    }

    #[test]
    fn test_returns_head() {
        let config = ReturnsConfig {
            embedding_dim: 32,
            hidden_dims: vec![16, 8],
            horizon: 1,
            with_uncertainty: true,
            dropout: 0.1,
        };

        let head = ReturnsHead::new(config);
        let embedding = Array::random(32, Uniform::new(-1.0, 1.0));
        let prediction = head.predict(&embedding);

        assert!(prediction.confidence >= 0.0 && prediction.confidence <= 1.0);
        assert!(prediction.lower_bound < prediction.upper_bound);
    }

    #[test]
    fn test_bullish_bearish() {
        let bullish_pred = ReturnsPrediction {
            return_pct: 3.0,
            confidence: 0.8,
            lower_bound: 1.0,
            upper_bound: 5.0,
            risk_adjusted: 1.5,
        };
        assert!(bullish_pred.is_bullish());
        assert!(!bullish_pred.is_bearish());

        let bearish_pred = ReturnsPrediction {
            return_pct: -3.0,
            confidence: 0.8,
            lower_bound: -5.0,
            upper_bound: -1.0,
            risk_adjusted: -1.5,
        };
        assert!(!bearish_pred.is_bullish());
        assert!(bearish_pred.is_bearish());
    }
}
