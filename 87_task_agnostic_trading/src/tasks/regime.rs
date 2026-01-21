//! Market regime classification head

use super::{TaskHead, TaskType};
use ndarray::{Array1, Array2, Axis};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use serde::{Deserialize, Serialize};

/// Market regime types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MarketRegime {
    /// Strong upward trend
    Trending,
    /// Sideways movement within range
    Ranging,
    /// High volatility, erratic movement
    Volatile,
    /// Rapid price decline
    Crash,
    /// Recovery from crash/bottom
    Recovery,
}

impl MarketRegime {
    /// Get regime from class index
    pub fn from_index(idx: usize) -> Self {
        match idx {
            0 => MarketRegime::Trending,
            1 => MarketRegime::Ranging,
            2 => MarketRegime::Volatile,
            3 => MarketRegime::Crash,
            _ => MarketRegime::Recovery,
        }
    }

    /// Get class index
    pub fn to_index(&self) -> usize {
        match self {
            MarketRegime::Trending => 0,
            MarketRegime::Ranging => 1,
            MarketRegime::Volatile => 2,
            MarketRegime::Crash => 3,
            MarketRegime::Recovery => 4,
        }
    }

    /// Get recommended trading approach for this regime
    pub fn trading_recommendation(&self) -> &'static str {
        match self {
            MarketRegime::Trending => "Follow the trend with tight stops",
            MarketRegime::Ranging => "Trade range bounds, avoid trend-following",
            MarketRegime::Volatile => "Reduce position size, widen stops",
            MarketRegime::Crash => "Stay out or hedge positions",
            MarketRegime::Recovery => "Scale in carefully, watch for false recoveries",
        }
    }

    /// Get risk level for this regime (1-5)
    pub fn risk_level(&self) -> u8 {
        match self {
            MarketRegime::Trending => 2,
            MarketRegime::Ranging => 1,
            MarketRegime::Volatile => 4,
            MarketRegime::Crash => 5,
            MarketRegime::Recovery => 3,
        }
    }
}

impl std::fmt::Display for MarketRegime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MarketRegime::Trending => write!(f, "Trending"),
            MarketRegime::Ranging => write!(f, "Ranging"),
            MarketRegime::Volatile => write!(f, "Volatile"),
            MarketRegime::Crash => write!(f, "Crash"),
            MarketRegime::Recovery => write!(f, "Recovery"),
        }
    }
}

/// Regime prediction result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimePrediction {
    /// Predicted regime
    pub regime: MarketRegime,
    /// Prediction confidence (0-1)
    pub confidence: f64,
    /// Class probabilities
    pub probabilities: Vec<f64>,
    /// Risk level (1-5)
    pub risk_level: u8,
    /// Trading recommendation
    pub recommendation: String,
}

impl RegimePrediction {
    /// Create from class probabilities
    pub fn from_probabilities(probs: &[f64]) -> Self {
        let (max_idx, &max_prob) = probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();

        let regime = MarketRegime::from_index(max_idx);

        Self {
            regime,
            confidence: max_prob,
            probabilities: probs.to_vec(),
            risk_level: regime.risk_level(),
            recommendation: regime.trading_recommendation().to_string(),
        }
    }

    /// Check if regime is high risk
    pub fn is_high_risk(&self) -> bool {
        self.risk_level >= 4
    }

    /// Check if prediction is reliable
    pub fn is_reliable(&self, threshold: f64) -> bool {
        self.confidence >= threshold
    }
}

/// Regime head configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeConfig {
    /// Input embedding dimension
    pub embedding_dim: usize,
    /// Hidden layer dimensions
    pub hidden_dims: Vec<usize>,
    /// Number of regime classes
    pub num_regimes: usize,
    /// Dropout rate
    pub dropout: f64,
}

impl Default for RegimeConfig {
    fn default() -> Self {
        Self {
            embedding_dim: 64,
            hidden_dims: vec![32, 16],
            num_regimes: 5,
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

/// Regime classification head
pub struct RegimeHead {
    config: RegimeConfig,
    layers: Vec<Array2<f64>>,
    output_layer: Array2<f64>,
}

impl RegimeHead {
    /// Create a new regime head
    pub fn new(config: RegimeConfig) -> Self {
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
        let scale_out = (2.0 / (last_dim + config.num_regimes) as f64).sqrt();
        let output_layer = Array2::random(
            (last_dim, config.num_regimes),
            Uniform::new(-scale_out, scale_out),
        );

        Self {
            config,
            layers,
            output_layer,
        }
    }

    /// Predict regime from embedding
    pub fn predict(&self, embedding: &Array1<f64>) -> RegimePrediction {
        let logits = self.forward(embedding);
        let probs = softmax(&logits);
        RegimePrediction::from_probabilities(&probs.to_vec())
    }

    /// Get configuration
    pub fn config(&self) -> &RegimeConfig {
        &self.config
    }
}

impl TaskHead for RegimeHead {
    fn task_type(&self) -> TaskType {
        TaskType::Regime
    }

    fn forward(&self, embedding: &Array1<f64>) -> Array1<f64> {
        let mut x = embedding.clone();

        // Hidden layers with ReLU
        for layer in &self.layers {
            x = x.dot(layer).mapv(|v| v.max(0.0));
        }

        // Output logits
        x.dot(&self.output_layer)
    }

    fn forward_batch(&self, embeddings: &Array2<f64>) -> Array2<f64> {
        let mut outputs = Vec::with_capacity(embeddings.nrows());
        for row in embeddings.axis_iter(Axis(0)) {
            let logits = self.forward(&row.to_owned());
            outputs.push(logits);
        }

        let flat: Vec<f64> = outputs.iter().flat_map(|o| o.to_vec()).collect();
        Array2::from_shape_vec((embeddings.nrows(), self.config.num_regimes), flat)
            .expect("Shape mismatch")
    }

    fn parameters(&self) -> Vec<Array2<f64>> {
        let mut params = self.layers.clone();
        params.push(self.output_layer.clone());
        params
    }

    fn update_parameters(&mut self, gradients: &[Array2<f64>], learning_rate: f64) {
        for (i, layer) in self.layers.iter_mut().enumerate() {
            if i < gradients.len() {
                *layer = &*layer - &(&gradients[i] * learning_rate);
            }
        }

        let n_layers = self.layers.len();
        if gradients.len() > n_layers {
            self.output_layer = &self.output_layer - &(&gradients[n_layers] * learning_rate);
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
    fn test_regime_from_index() {
        assert_eq!(MarketRegime::from_index(0), MarketRegime::Trending);
        assert_eq!(MarketRegime::from_index(1), MarketRegime::Ranging);
        assert_eq!(MarketRegime::from_index(2), MarketRegime::Volatile);
        assert_eq!(MarketRegime::from_index(3), MarketRegime::Crash);
        assert_eq!(MarketRegime::from_index(4), MarketRegime::Recovery);
    }

    #[test]
    fn test_regime_head() {
        let config = RegimeConfig {
            embedding_dim: 32,
            hidden_dims: vec![16],
            num_regimes: 5,
            dropout: 0.1,
        };

        let head = RegimeHead::new(config);
        let embedding = Array::random(32, Uniform::new(-1.0, 1.0));
        let prediction = head.predict(&embedding);

        assert!(prediction.confidence >= 0.0 && prediction.confidence <= 1.0);
        assert_eq!(prediction.probabilities.len(), 5);
        let prob_sum: f64 = prediction.probabilities.iter().sum();
        assert!((prob_sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_risk_levels() {
        assert_eq!(MarketRegime::Ranging.risk_level(), 1);
        assert_eq!(MarketRegime::Crash.risk_level(), 5);
    }
}
