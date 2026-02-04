//! Model inference module
//!
//! This module provides interfaces for model inference.
//! Actual inference requires integration with ONNX Runtime or LibTorch.

use ndarray::Array3;

use super::config::ModelConfig;
use crate::strategy::signal::{Signal, SignalType};

/// Prediction result from the model
#[derive(Debug, Clone)]
pub struct Prediction {
    /// Probability of price going down
    pub prob_down: f64,
    /// Probability of price staying neutral
    pub prob_neutral: f64,
    /// Probability of price going up
    pub prob_up: f64,
    /// Expected return magnitude (if predicted)
    pub magnitude: Option<f64>,
}

impl Prediction {
    /// Create a new prediction
    pub fn new(prob_down: f64, prob_neutral: f64, prob_up: f64, magnitude: Option<f64>) -> Self {
        Self {
            prob_down,
            prob_neutral,
            prob_up,
            magnitude,
        }
    }

    /// Get the predicted class (0=down, 1=neutral, 2=up)
    pub fn predicted_class(&self) -> usize {
        if self.prob_up > self.prob_down && self.prob_up > self.prob_neutral {
            2
        } else if self.prob_down > self.prob_neutral {
            0
        } else {
            1
        }
    }

    /// Get the confidence of the prediction
    pub fn confidence(&self) -> f64 {
        self.prob_down.max(self.prob_neutral).max(self.prob_up)
    }

    /// Convert to trading signal
    pub fn to_signal(&self, buy_threshold: f64, sell_threshold: f64) -> Signal {
        if self.prob_up > buy_threshold {
            Signal::new(SignalType::Buy, self.prob_up)
        } else if self.prob_down > sell_threshold {
            Signal::new(SignalType::Sell, self.prob_down)
        } else {
            Signal::new(SignalType::Hold, 0.0)
        }
    }
}

/// Model inference interface
///
/// This is a placeholder that can be implemented with actual
/// inference backends (ONNX Runtime, LibTorch, etc.)
pub trait ModelInference {
    /// Run inference on a single image
    fn predict(&self, image: &Array3<f64>) -> anyhow::Result<Prediction>;

    /// Run inference on a batch of images
    fn predict_batch(&self, images: &[Array3<f64>]) -> anyhow::Result<Vec<Prediction>>;
}

/// Dummy model for testing (returns random predictions)
pub struct DummyModel {
    pub config: ModelConfig,
}

impl DummyModel {
    /// Create a new dummy model
    pub fn new(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelInference for DummyModel {
    fn predict(&self, _image: &Array3<f64>) -> anyhow::Result<Prediction> {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        // Generate random probabilities
        let probs: Vec<f64> = (0..3).map(|_| rng.gen::<f64>()).collect();
        let sum: f64 = probs.iter().sum();
        let normalized: Vec<f64> = probs.iter().map(|p| p / sum).collect();

        let magnitude = if self.config.predict_magnitude {
            Some(rng.gen_range(-0.05..0.05))
        } else {
            None
        };

        Ok(Prediction::new(
            normalized[0],
            normalized[1],
            normalized[2],
            magnitude,
        ))
    }

    fn predict_batch(&self, images: &[Array3<f64>]) -> anyhow::Result<Vec<Prediction>> {
        images.iter().map(|img| self.predict(img)).collect()
    }
}

/// Rule-based model using simple price patterns
pub struct RuleBasedModel {
    pub config: ModelConfig,
    /// Threshold for trend detection
    pub trend_threshold: f64,
}

impl RuleBasedModel {
    /// Create a new rule-based model
    pub fn new(config: ModelConfig) -> Self {
        Self {
            config,
            trend_threshold: 0.02,
        }
    }

    /// Analyze price trend from image
    fn analyze_trend(&self, image: &Array3<f64>) -> (f64, f64, f64) {
        // Simple heuristic: analyze the diagonal of the image
        // which represents autocorrelation in GAF images
        let (h, w, _) = image.dim();
        let diagonal_sum: f64 = (0..h.min(w)).map(|i| image[[i, i, 0]]).sum();
        let avg = diagonal_sum / h.min(w) as f64;

        // Higher diagonal values in GASF often indicate trending behavior
        if avg > 0.6 {
            (0.2, 0.2, 0.6) // Likely uptrend
        } else if avg < 0.4 {
            (0.6, 0.2, 0.2) // Likely downtrend
        } else {
            (0.3, 0.4, 0.3) // Neutral
        }
    }
}

impl ModelInference for RuleBasedModel {
    fn predict(&self, image: &Array3<f64>) -> anyhow::Result<Prediction> {
        let (prob_down, prob_neutral, prob_up) = self.analyze_trend(image);

        let magnitude = if self.config.predict_magnitude {
            Some((prob_up - prob_down) * 0.02)
        } else {
            None
        };

        Ok(Prediction::new(prob_down, prob_neutral, prob_up, magnitude))
    }

    fn predict_batch(&self, images: &[Array3<f64>]) -> anyhow::Result<Vec<Prediction>> {
        images.iter().map(|img| self.predict(img)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prediction_class() {
        let pred = Prediction::new(0.1, 0.2, 0.7, None);
        assert_eq!(pred.predicted_class(), 2);

        let pred = Prediction::new(0.7, 0.2, 0.1, None);
        assert_eq!(pred.predicted_class(), 0);

        let pred = Prediction::new(0.2, 0.6, 0.2, None);
        assert_eq!(pred.predicted_class(), 1);
    }

    #[test]
    fn test_prediction_to_signal() {
        let pred = Prediction::new(0.1, 0.2, 0.7, None);
        let signal = pred.to_signal(0.6, 0.6);
        assert_eq!(signal.signal_type, SignalType::Buy);

        let pred = Prediction::new(0.7, 0.2, 0.1, None);
        let signal = pred.to_signal(0.6, 0.6);
        assert_eq!(signal.signal_type, SignalType::Sell);
    }

    #[test]
    fn test_dummy_model() {
        let config = ModelConfig::default();
        let model = DummyModel::new(config);

        let image = Array3::zeros((64, 64, 3));
        let pred = model.predict(&image).unwrap();

        // Probabilities should sum to approximately 1
        let sum = pred.prob_down + pred.prob_neutral + pred.prob_up;
        assert!((sum - 1.0).abs() < 0.01);
    }
}
