//! Direction prediction head for market movement classification

use super::{TaskHead, TaskType};
use ndarray::{Array1, Array2, Axis};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use serde::{Deserialize, Serialize};

/// Direction prediction: Up, Down, or Sideways
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Direction {
    Up,
    Down,
    Sideways,
}

impl Direction {
    /// Get direction from class index
    pub fn from_index(idx: usize) -> Self {
        match idx {
            0 => Direction::Up,
            1 => Direction::Down,
            _ => Direction::Sideways,
        }
    }

    /// Get class index
    pub fn to_index(&self) -> usize {
        match self {
            Direction::Up => 0,
            Direction::Down => 1,
            Direction::Sideways => 2,
        }
    }
}

impl std::fmt::Display for Direction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Direction::Up => write!(f, "Up"),
            Direction::Down => write!(f, "Down"),
            Direction::Sideways => write!(f, "Sideways"),
        }
    }
}

/// Direction prediction result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectionPrediction {
    /// Predicted direction
    pub direction: Direction,
    /// Prediction confidence (0-1)
    pub confidence: f64,
    /// Class probabilities [up, down, sideways]
    pub probabilities: [f64; 3],
}

impl DirectionPrediction {
    /// Create from class probabilities
    pub fn from_probabilities(probs: &[f64; 3]) -> Self {
        let (max_idx, &max_prob) = probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();

        Self {
            direction: Direction::from_index(max_idx),
            confidence: max_prob,
            probabilities: *probs,
        }
    }

    /// Check if prediction is reliable (high confidence)
    pub fn is_reliable(&self, threshold: f64) -> bool {
        self.confidence >= threshold
    }
}

/// Direction head configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectionConfig {
    /// Input embedding dimension
    pub embedding_dim: usize,
    /// Hidden layer dimension
    pub hidden_dim: usize,
    /// Number of output classes (typically 3: up, down, sideways)
    pub num_classes: usize,
    /// Dropout rate
    pub dropout: f64,
}

impl Default for DirectionConfig {
    fn default() -> Self {
        Self {
            embedding_dim: 64,
            hidden_dim: 32,
            num_classes: 3,
            dropout: 0.1,
        }
    }
}

/// Softmax function
fn softmax(x: &Array1<f64>) -> Array1<f64> {
    let max_val = x.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let exp_x = x.mapv(|v| (v - max_val).exp());
    let sum = exp_x.sum();
    exp_x / sum
}

/// Direction prediction head
pub struct DirectionHead {
    config: DirectionConfig,
    w1: Array2<f64>,
    w2: Array2<f64>,
}

impl DirectionHead {
    /// Create a new direction head
    pub fn new(config: DirectionConfig) -> Self {
        let scale1 = (2.0 / (config.embedding_dim + config.hidden_dim) as f64).sqrt();
        let scale2 = (2.0 / (config.hidden_dim + config.num_classes) as f64).sqrt();

        let w1 = Array2::random(
            (config.embedding_dim, config.hidden_dim),
            Uniform::new(-scale1, scale1),
        );
        let w2 = Array2::random(
            (config.hidden_dim, config.num_classes),
            Uniform::new(-scale2, scale2),
        );

        Self { config, w1, w2 }
    }

    /// Predict direction from embedding
    pub fn predict(&self, embedding: &Array1<f64>) -> DirectionPrediction {
        let logits = self.forward(embedding);
        let probs = softmax(&logits);

        let probs_arr: [f64; 3] = [probs[0], probs[1], probs[2]];
        DirectionPrediction::from_probabilities(&probs_arr)
    }

    /// Get configuration
    pub fn config(&self) -> &DirectionConfig {
        &self.config
    }
}

impl TaskHead for DirectionHead {
    fn task_type(&self) -> TaskType {
        TaskType::Direction
    }

    fn forward(&self, embedding: &Array1<f64>) -> Array1<f64> {
        // Two-layer MLP with ReLU
        let hidden = embedding.dot(&self.w1).mapv(|x| x.max(0.0));
        hidden.dot(&self.w2)
    }

    fn forward_batch(&self, embeddings: &Array2<f64>) -> Array2<f64> {
        let mut outputs = Vec::with_capacity(embeddings.nrows());
        for row in embeddings.axis_iter(Axis(0)) {
            let logits = self.forward(&row.to_owned());
            outputs.push(logits);
        }

        let flat: Vec<f64> = outputs.iter().flat_map(|o| o.to_vec()).collect();
        Array2::from_shape_vec((embeddings.nrows(), self.config.num_classes), flat)
            .expect("Shape mismatch")
    }

    fn parameters(&self) -> Vec<Array2<f64>> {
        vec![self.w1.clone(), self.w2.clone()]
    }

    fn update_parameters(&mut self, gradients: &[Array2<f64>], learning_rate: f64) {
        if gradients.len() >= 2 {
            self.w1 = &self.w1 - &(&gradients[0] * learning_rate);
            self.w2 = &self.w2 - &(&gradients[1] * learning_rate);
        }
    }

    fn compute_loss(&self, predictions: &Array2<f64>, targets: &Array2<f64>) -> f64 {
        // Cross-entropy loss
        let mut total_loss = 0.0;
        let n = predictions.nrows() as f64;

        for (pred_row, target_row) in predictions.axis_iter(Axis(0)).zip(targets.axis_iter(Axis(0)))
        {
            let probs = softmax(&pred_row.to_owned());
            for (p, t) in probs.iter().zip(target_row.iter()) {
                if *t > 0.0 {
                    total_loss -= t * (p + 1e-10).ln();
                }
            }
        }

        total_loss / n
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;

    #[test]
    fn test_direction_from_index() {
        assert_eq!(Direction::from_index(0), Direction::Up);
        assert_eq!(Direction::from_index(1), Direction::Down);
        assert_eq!(Direction::from_index(2), Direction::Sideways);
    }

    #[test]
    fn test_direction_head() {
        let config = DirectionConfig {
            embedding_dim: 32,
            hidden_dim: 16,
            num_classes: 3,
            dropout: 0.1,
        };

        let head = DirectionHead::new(config);
        let embedding = Array::random(32, Uniform::new(-1.0, 1.0));
        let prediction = head.predict(&embedding);

        assert!(prediction.confidence >= 0.0 && prediction.confidence <= 1.0);
        let prob_sum: f64 = prediction.probabilities.iter().sum();
        assert!((prob_sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_prediction_reliability() {
        let pred = DirectionPrediction {
            direction: Direction::Up,
            confidence: 0.85,
            probabilities: [0.85, 0.10, 0.05],
        };

        assert!(pred.is_reliable(0.8));
        assert!(!pred.is_reliable(0.9));
    }
}
