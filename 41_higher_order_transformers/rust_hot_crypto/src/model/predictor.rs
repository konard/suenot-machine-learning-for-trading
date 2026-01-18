//! Price movement predictor using HOT model
//!
//! Wraps the HOT model with classification head for trading predictions

use ndarray::{Array1, Array2, Axis};
use rand::Rng;
use rand_distr::StandardNormal;
use serde::{Deserialize, Serialize};

use super::HOTModel;
use crate::data::{Features, PriceSeries};

/// Classification of price movement
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MovementClass {
    /// Price expected to go down
    Down = 0,
    /// Price expected to stay neutral
    Neutral = 1,
    /// Price expected to go up
    Up = 2,
}

impl MovementClass {
    /// Get the class index
    pub fn index(&self) -> usize {
        *self as usize
    }

    /// Create from index
    pub fn from_index(idx: usize) -> Self {
        match idx {
            0 => Self::Down,
            1 => Self::Neutral,
            _ => Self::Up,
        }
    }

    /// Get a display string
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Down => "DOWN",
            Self::Neutral => "NEUTRAL",
            Self::Up => "UP",
        }
    }
}

/// Prediction result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionResult {
    /// Symbol
    pub symbol: String,
    /// Predicted class
    pub prediction: MovementClass,
    /// Confidence (probability)
    pub confidence: f64,
    /// All class probabilities
    pub probabilities: [f64; 3],
}

impl PredictionResult {
    /// Check if prediction is bullish
    pub fn is_bullish(&self) -> bool {
        self.prediction == MovementClass::Up
    }

    /// Check if prediction is bearish
    pub fn is_bearish(&self) -> bool {
        self.prediction == MovementClass::Down
    }

    /// Check if prediction is strong (high confidence)
    pub fn is_strong(&self, threshold: f64) -> bool {
        self.confidence >= threshold
    }
}

/// HOT-based price movement predictor
#[derive(Debug, Clone)]
pub struct HOTPredictor {
    /// The underlying HOT model
    pub model: HOTModel,
    /// Classification head weights
    pub classifier_w: Array2<f64>,
    /// Classification head bias
    pub classifier_b: Array1<f64>,
    /// Lookback period for features
    pub lookback: usize,
    /// Movement threshold (percentage)
    pub movement_threshold: f64,
}

impl HOTPredictor {
    /// Create a new predictor
    pub fn new(input_dim: usize, lookback: usize) -> Self {
        let model = HOTModel::for_crypto(input_dim);
        let d_model = model.d_model;

        let mut rng = rand::thread_rng();

        // Classification head: d_model -> 3 classes
        let std = (2.0 / (d_model + 3) as f64).sqrt();
        let classifier_w = Array2::from_shape_fn((d_model, 3), |_| {
            rng.sample::<f64, _>(StandardNormal) * std
        });
        let classifier_b = Array1::zeros(3);

        Self {
            model,
            classifier_w,
            classifier_b,
            lookback,
            movement_threshold: 0.005, // 0.5% threshold
        }
    }

    /// Create with default settings for crypto
    pub fn default_crypto() -> Self {
        Self::new(7, 60) // 7 features, 60 period lookback
    }

    /// Make a prediction from raw features
    pub fn predict(&self, features: &Array2<f64>) -> (MovementClass, f64, [f64; 3]) {
        // Forward through HOT model
        let h = self.model.forward(features);

        // Get last representation
        let last = h.row(h.nrows() - 1);

        // Classification head
        let logits: Array1<f64> = last.dot(&self.classifier_w) + &self.classifier_b;

        // Softmax
        let probs = softmax(&logits);

        // Find argmax
        let (max_idx, max_prob) = probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();

        let prediction = MovementClass::from_index(max_idx);
        let probabilities = [probs[0], probs[1], probs[2]];

        (prediction, *max_prob, probabilities)
    }

    /// Predict from price series
    pub fn predict_from_series(&self, series: &PriceSeries) -> PredictionResult {
        let features = Features::from_price_series(series, self.lookback);

        // Take last `lookback` rows
        let n_rows = features.data.nrows();
        let start = n_rows.saturating_sub(self.lookback);
        let input = features.data.slice(ndarray::s![start.., ..]).to_owned();

        let (prediction, confidence, probabilities) = self.predict(&input);

        PredictionResult {
            symbol: series.symbol.clone(),
            prediction,
            confidence,
            probabilities,
        }
    }

    /// Predict multiple series (batch prediction)
    pub fn predict_batch(&self, series_list: &[PriceSeries]) -> Vec<PredictionResult> {
        series_list
            .iter()
            .map(|s| self.predict_from_series(s))
            .collect()
    }

    /// Get attention weights for interpretability
    pub fn get_attention_weights(&self, features: &Array2<f64>) -> Vec<Array2<f64>> {
        // Project input
        let h = features.dot(&self.model.input_proj);

        // Get attention weights from each block
        self.model
            .blocks
            .iter()
            .map(|block| block.attention.get_attention_weights(&h))
            .collect()
    }

    /// Determine the label for a given return
    pub fn get_label(&self, future_return: f64) -> MovementClass {
        if future_return > self.movement_threshold {
            MovementClass::Up
        } else if future_return < -self.movement_threshold {
            MovementClass::Down
        } else {
            MovementClass::Neutral
        }
    }
}

/// Softmax function for 1D array
fn softmax(x: &Array1<f64>) -> Array1<f64> {
    let max = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_x: Array1<f64> = x.mapv(|v| (v - max).exp());
    let sum = exp_x.sum();
    &exp_x / sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_movement_class() {
        assert_eq!(MovementClass::Down.index(), 0);
        assert_eq!(MovementClass::Neutral.index(), 1);
        assert_eq!(MovementClass::Up.index(), 2);

        assert_eq!(MovementClass::from_index(0), MovementClass::Down);
        assert_eq!(MovementClass::from_index(2), MovementClass::Up);
    }

    #[test]
    fn test_predictor_output_shape() {
        let predictor = HOTPredictor::new(7, 30);
        let features = Array2::from_shape_fn((30, 7), |_| rand::random::<f64>());

        let (prediction, confidence, probs) = predictor.predict(&features);

        // Probabilities should sum to 1
        let prob_sum: f64 = probs.iter().sum();
        assert!((prob_sum - 1.0).abs() < 1e-6);

        // Confidence should be between 0 and 1
        assert!(confidence >= 0.0 && confidence <= 1.0);
    }

    #[test]
    fn test_softmax() {
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let probs = softmax(&x);

        // Should sum to 1
        let sum: f64 = probs.sum();
        assert!((sum - 1.0).abs() < 1e-10);

        // Should be ordered
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }
}
