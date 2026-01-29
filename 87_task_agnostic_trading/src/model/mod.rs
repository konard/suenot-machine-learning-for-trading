//! Model components for task-agnostic representation learning
//!
//! This module provides:
//! - Universal encoder for shared feature extraction
//! - Task-specific heads for different trading objectives
//! - Decision fusion for combining multi-task outputs

mod encoder;
mod task_heads;
mod fusion;

pub use encoder::{UniversalEncoder, EncoderConfig};
pub use task_heads::{TaskHead, TaskHeadConfig, TaskType};
pub use fusion::{DecisionFusion, FusionConfig, FusionMethod};

use ndarray::Array2;
use serde::{Deserialize, Serialize};

/// Configuration for the full task-agnostic model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskAgnosticConfig {
    /// Encoder configuration
    pub encoder: EncoderConfig,
    /// Task head configurations
    pub task_heads: Vec<TaskHeadConfig>,
    /// Fusion configuration
    pub fusion: FusionConfig,
}

impl Default for TaskAgnosticConfig {
    fn default() -> Self {
        Self {
            encoder: EncoderConfig::default(),
            task_heads: vec![
                TaskHeadConfig::new(TaskType::TrendPrediction, 3),
                TaskHeadConfig::new(TaskType::VolatilityForecast, 1),
                TaskHeadConfig::new(TaskType::RegimeDetection, 4),
                TaskHeadConfig::new(TaskType::RiskAssessment, 3),
            ],
            fusion: FusionConfig::default(),
        }
    }
}

impl TaskAgnosticConfig {
    /// Set the input dimension
    pub fn with_input_dim(mut self, dim: usize) -> Self {
        self.encoder.input_dim = dim;
        self
    }

    /// Set the representation dimension
    pub fn with_repr_dim(mut self, dim: usize) -> Self {
        self.encoder.repr_dim = dim;
        for head in &mut self.task_heads {
            head.input_dim = dim;
        }
        self
    }

    /// Set fusion method
    pub fn with_fusion_method(mut self, method: FusionMethod) -> Self {
        self.fusion.method = method;
        self
    }
}

/// Complete task-agnostic model combining encoder, heads, and fusion
pub struct TaskAgnosticModel {
    /// Configuration
    pub config: TaskAgnosticConfig,
    /// Universal encoder
    pub encoder: UniversalEncoder,
    /// Task-specific heads
    pub heads: Vec<TaskHead>,
    /// Decision fusion module
    pub fusion: DecisionFusion,
}

impl TaskAgnosticModel {
    /// Create a new task-agnostic model
    pub fn new(config: TaskAgnosticConfig) -> Self {
        let encoder = UniversalEncoder::new(config.encoder.clone());
        let heads: Vec<TaskHead> = config
            .task_heads
            .iter()
            .map(|tc| TaskHead::new(tc.clone()))
            .collect();
        let fusion = DecisionFusion::new(config.fusion.clone());

        Self {
            config,
            encoder,
            heads,
            fusion,
        }
    }

    /// Encode input features into task-agnostic representations
    pub fn encode(&self, features: &Array2<f64>) -> Array2<f64> {
        self.encoder.forward(features)
    }

    /// Run all task heads on encoded representations
    pub fn predict_all_tasks(&self, features: &Array2<f64>) -> Vec<TaskOutput> {
        let representations = self.encode(features);
        self.heads
            .iter()
            .map(|head| {
                let output = head.forward(&representations);
                TaskOutput {
                    task_type: head.config.task_type.clone(),
                    predictions: output,
                    task_name: head.config.task_type.name().to_string(),
                }
            })
            .collect()
    }

    /// Generate fused trading decision from all task outputs
    pub fn predict(&self, features: &Array2<f64>) -> FusedPrediction {
        let task_outputs = self.predict_all_tasks(features);
        self.fusion.fuse(&task_outputs)
    }

    /// Get the number of task heads
    pub fn num_tasks(&self) -> usize {
        self.heads.len()
    }
}

/// Output from a single task head
#[derive(Debug, Clone)]
pub struct TaskOutput {
    /// Type of task
    pub task_type: TaskType,
    /// Raw predictions (n_samples x n_outputs)
    pub predictions: Array2<f64>,
    /// Human-readable task name
    pub task_name: String,
}

/// Fused prediction combining all task outputs
#[derive(Debug, Clone)]
pub struct FusedPrediction {
    /// Overall trading signal: -1.0 (strong sell) to 1.0 (strong buy)
    pub signal: Vec<f64>,
    /// Confidence in the signal (0.0 to 1.0)
    pub confidence: Vec<f64>,
    /// Predicted regime for each sample
    pub regime: Vec<String>,
    /// Risk level (0.0 to 1.0)
    pub risk_level: Vec<f64>,
    /// Individual task contributions
    pub task_contributions: Vec<(String, f64)>,
}
