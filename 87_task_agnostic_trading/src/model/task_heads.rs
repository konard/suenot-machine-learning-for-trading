//! Task-specific heads for multi-task learning
//!
//! Each head takes the shared representation from the universal encoder
//! and produces task-specific predictions.

use ndarray::{Array1, Array2};
use rand::Rng;
use serde::{Deserialize, Serialize};

/// Types of trading tasks
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TaskType {
    /// Predict price direction (up/sideways/down)
    TrendPrediction,
    /// Forecast future volatility
    VolatilityForecast,
    /// Detect market regime (trending/mean-reverting/volatile/calm)
    RegimeDetection,
    /// Assess risk level (low/medium/high)
    RiskAssessment,
}

impl TaskType {
    /// Human-readable name
    pub fn name(&self) -> &str {
        match self {
            TaskType::TrendPrediction => "Trend Prediction",
            TaskType::VolatilityForecast => "Volatility Forecast",
            TaskType::RegimeDetection => "Regime Detection",
            TaskType::RiskAssessment => "Risk Assessment",
        }
    }

    /// Whether this task is a classification task
    pub fn is_classification(&self) -> bool {
        match self {
            TaskType::TrendPrediction => true,
            TaskType::VolatilityForecast => false,
            TaskType::RegimeDetection => true,
            TaskType::RiskAssessment => true,
        }
    }

    /// Class labels for classification tasks
    pub fn class_labels(&self) -> Vec<&str> {
        match self {
            TaskType::TrendPrediction => vec!["Uptrend", "Sideways", "Downtrend"],
            TaskType::RegimeDetection => vec!["Trending", "Mean-Reverting", "Volatile", "Calm"],
            TaskType::RiskAssessment => vec!["Low", "Medium", "High"],
            TaskType::VolatilityForecast => vec!["Value"],
        }
    }
}

/// Configuration for a task head
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskHeadConfig {
    /// Type of task
    pub task_type: TaskType,
    /// Input dimension (from encoder)
    pub input_dim: usize,
    /// Hidden dimension within the head
    pub hidden_dim: usize,
    /// Output dimension
    pub output_dim: usize,
    /// Task weight in the multi-task loss
    pub task_weight: f64,
}

impl TaskHeadConfig {
    /// Create a new task head configuration
    pub fn new(task_type: TaskType, output_dim: usize) -> Self {
        Self {
            task_type,
            input_dim: 16, // default from encoder repr_dim
            hidden_dim: 32,
            output_dim,
            task_weight: 1.0,
        }
    }

    /// Set input dimension
    pub fn with_input_dim(mut self, dim: usize) -> Self {
        self.input_dim = dim;
        self
    }

    /// Set task weight
    pub fn with_weight(mut self, weight: f64) -> Self {
        self.task_weight = weight;
        self
    }
}

/// Dense layer for task heads
#[derive(Debug, Clone)]
struct HeadLayer {
    weights: Array2<f64>,
    bias: Array1<f64>,
}

impl HeadLayer {
    fn new(input_dim: usize, output_dim: usize, seed: u64) -> Self {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let scale = (2.0 / (input_dim + output_dim) as f64).sqrt();

        let weights = Array2::from_shape_fn((input_dim, output_dim), |_| {
            rng.gen_range(-scale..scale)
        });
        let bias = Array1::zeros(output_dim);

        Self { weights, bias }
    }

    fn forward(&self, input: &Array2<f64>) -> Array2<f64> {
        input.dot(&self.weights) + &self.bias
    }
}

/// A task-specific head that maps representations to task outputs
pub struct TaskHead {
    /// Configuration
    pub config: TaskHeadConfig,
    /// Hidden layer
    hidden: HeadLayer,
    /// Output layer
    output: HeadLayer,
}

impl TaskHead {
    /// Create a new task head
    pub fn new(config: TaskHeadConfig) -> Self {
        let seed_base = match config.task_type {
            TaskType::TrendPrediction => 100,
            TaskType::VolatilityForecast => 200,
            TaskType::RegimeDetection => 300,
            TaskType::RiskAssessment => 400,
        };

        let hidden = HeadLayer::new(config.input_dim, config.hidden_dim, seed_base);
        let output = HeadLayer::new(config.hidden_dim, config.output_dim, seed_base + 1);

        Self {
            config,
            hidden,
            output,
        }
    }

    /// Forward pass through the task head
    pub fn forward(&self, representations: &Array2<f64>) -> Array2<f64> {
        // Hidden layer with ReLU
        let mut h = self.hidden.forward(representations);
        h.mapv_inplace(|v| v.max(0.0));

        // Output layer
        let out = self.output.forward(&h);

        // Apply softmax for classification, sigmoid for regression
        if self.config.task_type.is_classification() {
            self.softmax(&out)
        } else {
            out.mapv(|v| 1.0 / (1.0 + (-v).exp())) // Sigmoid for bounded output
        }
    }

    /// Softmax activation
    fn softmax(&self, input: &Array2<f64>) -> Array2<f64> {
        let mut output = input.clone();
        for mut row in output.rows_mut() {
            let max_val = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            row.mapv_inplace(|v| (v - max_val).exp());
            let sum: f64 = row.iter().sum();
            row.mapv_inplace(|v| v / sum.max(1e-10));
        }
        output
    }

    /// Get predicted class for classification tasks
    pub fn predict_class(&self, representations: &Array2<f64>) -> Vec<usize> {
        let probs = self.forward(representations);
        probs
            .rows()
            .into_iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap_or(0)
            })
            .collect()
    }

    /// Get prediction confidence
    pub fn predict_confidence(&self, representations: &Array2<f64>) -> Vec<f64> {
        let probs = self.forward(representations);
        probs
            .rows()
            .into_iter()
            .map(|row| {
                row.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
            })
            .collect()
    }
}

use rand::SeedableRng;

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_task_head_classification() {
        let config = TaskHeadConfig::new(TaskType::TrendPrediction, 3)
            .with_input_dim(8);
        let head = TaskHead::new(config);
        let input = Array2::ones((4, 8));
        let output = head.forward(&input);

        assert_eq!(output.shape(), &[4, 3]);

        // Check softmax: each row sums to 1
        for row in output.rows() {
            let sum: f64 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_task_head_regression() {
        let config = TaskHeadConfig::new(TaskType::VolatilityForecast, 1)
            .with_input_dim(8);
        let head = TaskHead::new(config);
        let input = Array2::ones((4, 8));
        let output = head.forward(&input);

        assert_eq!(output.shape(), &[4, 1]);

        // Check sigmoid: all values between 0 and 1
        for &val in output.iter() {
            assert!(val >= 0.0 && val <= 1.0);
        }
    }
}
