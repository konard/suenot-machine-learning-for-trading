//! TimeParticle SSM model implementation.

use crate::model::particle::Particle;
use ndarray::{Array1, Array2, Axis};
use rand::prelude::*;
use rand_distr::Normal;

/// Trading signal types.
#[derive(Debug, Clone, PartialEq)]
pub enum TradingSignal {
    /// Buy signal with confidence
    Buy(f64),
    /// Sell signal with confidence
    Sell(f64),
    /// Hold signal with confidence
    Hold(f64),
}

impl TradingSignal {
    /// Get the signal type as a string.
    pub fn signal_type(&self) -> &str {
        match self {
            TradingSignal::Buy(_) => "BUY",
            TradingSignal::Sell(_) => "SELL",
            TradingSignal::Hold(_) => "HOLD",
        }
    }

    /// Get the confidence value.
    pub fn confidence(&self) -> f64 {
        match self {
            TradingSignal::Buy(c) | TradingSignal::Sell(c) | TradingSignal::Hold(c) => *c,
        }
    }
}

/// TimeParticle State Space Model for time series forecasting.
///
/// Combines multiple particles operating at different time scales with
/// cross-scale aggregation for comprehensive temporal pattern capture.
pub struct TimeParticleSSM {
    /// Number of input features
    pub n_features: usize,
    /// Hidden dimension
    pub d_hidden: usize,
    /// State dimension for each particle
    pub d_state: usize,
    /// Number of particles (time scales)
    pub n_particles: usize,
    /// Number of processing layers
    pub n_layers: usize,
    /// Input projection weights [d_hidden x n_features]
    pub input_weights: Array2<f64>,
    /// Particles organized by layer
    pub particles: Vec<Vec<Particle>>,
    /// Aggregation weights [d_hidden x (n_particles * d_hidden)]
    pub aggregation_weights: Array2<f64>,
    /// Output weights [3 x d_hidden] for (Sell, Hold, Buy)
    pub output_weights: Array2<f64>,
    /// Output bias [3]
    pub output_bias: Array1<f64>,
    /// Confidence threshold for signals
    pub confidence_threshold: f64,
}

impl TimeParticleSSM {
    /// Create a new TimeParticle model with random initialization.
    ///
    /// # Arguments
    ///
    /// * `n_features` - Number of input features
    /// * `d_hidden` - Hidden dimension
    /// * `d_state` - State dimension for each particle
    /// * `n_particles` - Number of particles (time scales)
    /// * `n_layers` - Number of processing layers
    pub fn new(
        n_features: usize,
        d_hidden: usize,
        d_state: usize,
        n_particles: usize,
        n_layers: usize,
    ) -> Self {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 0.1).unwrap();

        // Create particles for each layer
        let particles: Vec<Vec<Particle>> = (0..n_layers)
            .map(|_| {
                (0..n_particles)
                    .map(|k| Particle::new(d_hidden, d_state, k + 1))
                    .collect()
            })
            .collect();

        Self {
            n_features,
            d_hidden,
            d_state,
            n_particles,
            n_layers,
            input_weights: Array2::from_shape_fn((d_hidden, n_features), |_| {
                normal.sample(&mut rng)
            }),
            particles,
            aggregation_weights: Array2::from_shape_fn(
                (d_hidden, n_particles * d_hidden),
                |_| normal.sample(&mut rng),
            ),
            output_weights: Array2::from_shape_fn((3, d_hidden), |_| normal.sample(&mut rng)),
            output_bias: Array1::zeros(3),
            confidence_threshold: 0.6,
        }
    }

    /// Set confidence threshold for trading signals.
    pub fn set_confidence_threshold(&mut self, threshold: f64) {
        self.confidence_threshold = threshold;
    }

    /// Forward pass through the model.
    ///
    /// # Arguments
    ///
    /// * `x` - Input features [seq_len x n_features]
    ///
    /// # Returns
    ///
    /// Probability distribution over (Sell, Hold, Buy)
    pub fn forward(&self, x: &Array2<f64>) -> Array1<f64> {
        let seq_len = x.nrows();

        // Input projection
        let mut h = x.dot(&self.input_weights.t());

        // Initialize particle states
        let mut states: Vec<Vec<Array1<f64>>> = self
            .particles
            .iter()
            .map(|layer| layer.iter().map(|p| p.initial_state()).collect())
            .collect();

        // Process through layers
        let mut particle_outputs: Vec<Array2<f64>> = Vec::new();

        for (layer_idx, layer_particles) in self.particles.iter().enumerate() {
            let mut layer_outputs: Vec<Array2<f64>> = Vec::new();

            // Evolve each particle
            for (k, particle) in layer_particles.iter().enumerate() {
                let (output, new_state) = particle.forward(&h, &states[layer_idx][k]);
                states[layer_idx][k] = new_state;
                layer_outputs.push(output);
            }

            particle_outputs = layer_outputs.clone();

            // Aggregate for next layer
            let concatenated = concatenate_cols(&layer_outputs);
            h = concatenated.dot(&self.aggregation_weights.t());
        }

        // Final aggregation from last timestep
        let final_concat = concatenate_cols(&particle_outputs);
        let final_row = final_concat.row(seq_len - 1).to_owned();
        let final_features = self.aggregation_weights.dot(&final_row);

        // Output logits
        let logits = self.output_weights.dot(&final_features) + &self.output_bias;

        // Softmax
        softmax(&logits)
    }

    /// Generate trading signal from input features.
    ///
    /// # Arguments
    ///
    /// * `x` - Input features [seq_len x n_features]
    ///
    /// # Returns
    ///
    /// Trading signal (Buy, Sell, or Hold) with confidence
    pub fn predict(&self, x: &Array2<f64>) -> TradingSignal {
        let probs = self.forward(x);

        let sell_prob = probs[0];
        let hold_prob = probs[1];
        let buy_prob = probs[2];

        if buy_prob > self.confidence_threshold && buy_prob > sell_prob {
            TradingSignal::Buy(buy_prob)
        } else if sell_prob > self.confidence_threshold && sell_prob > buy_prob {
            TradingSignal::Sell(sell_prob)
        } else {
            TradingSignal::Hold(hold_prob)
        }
    }

    /// Get probabilities for all classes.
    ///
    /// # Arguments
    ///
    /// * `x` - Input features [seq_len x n_features]
    ///
    /// # Returns
    ///
    /// Array of probabilities [sell_prob, hold_prob, buy_prob]
    pub fn predict_proba(&self, x: &Array2<f64>) -> Array1<f64> {
        self.forward(x)
    }

    /// Batch prediction for multiple sequences.
    ///
    /// # Arguments
    ///
    /// * `sequences` - Vector of input sequences
    ///
    /// # Returns
    ///
    /// Vector of trading signals
    pub fn predict_batch(&self, sequences: &[Array2<f64>]) -> Vec<TradingSignal> {
        sequences.iter().map(|x| self.predict(x)).collect()
    }

    /// Get analysis at each time scale.
    ///
    /// # Arguments
    ///
    /// * `x` - Input features [seq_len x n_features]
    ///
    /// # Returns
    ///
    /// Vector of (scale_name, momentum, signal) for each particle
    pub fn multiscale_analysis(&self, x: &Array2<f64>) -> Vec<(String, f64, &str)> {
        let seq_len = x.nrows();

        // Input projection
        let mut h = x.dot(&self.input_weights.t());

        // Initialize particle states
        let mut states: Vec<Vec<Array1<f64>>> = self
            .particles
            .iter()
            .map(|layer| layer.iter().map(|p| p.initial_state()).collect())
            .collect();

        // Process through last layer only for analysis
        let layer_idx = self.n_layers - 1;
        let layer_particles = &self.particles[layer_idx];
        let mut layer_outputs: Vec<Array2<f64>> = Vec::new();

        for (k, particle) in layer_particles.iter().enumerate() {
            let (output, _) = particle.forward(&h, &states[layer_idx][k]);
            layer_outputs.push(output);
        }

        // Analyze each particle's output
        let scale_names = ["fast", "medium", "slow", "trend"];
        let mut analysis = Vec::new();

        for (k, output) in layer_outputs.iter().enumerate() {
            let scale_name = if k < scale_names.len() {
                scale_names[k].to_string()
            } else {
                format!("scale_{}", k)
            };

            // Calculate momentum from output
            let recent_end = seq_len;
            let recent_start = recent_end.saturating_sub(10);
            let older_end = recent_start;
            let older_start = older_end.saturating_sub(10);

            let recent_mean: f64 = output
                .slice(ndarray::s![recent_start..recent_end, ..])
                .mean()
                .unwrap_or(0.0);
            let older_mean: f64 = if older_start < older_end {
                output
                    .slice(ndarray::s![older_start..older_end, ..])
                    .mean()
                    .unwrap_or(0.0)
            } else {
                recent_mean
            };

            let momentum = recent_mean - older_mean;
            let signal = if momentum > 0.0 { "bullish" } else { "bearish" };

            analysis.push((scale_name, momentum, signal));
        }

        analysis
    }
}

/// Concatenate arrays column-wise.
fn concatenate_cols(arrays: &[Array2<f64>]) -> Array2<f64> {
    if arrays.is_empty() {
        return Array2::zeros((0, 0));
    }

    let nrows = arrays[0].nrows();
    let total_cols: usize = arrays.iter().map(|a| a.ncols()).sum();

    let mut result = Array2::zeros((nrows, total_cols));
    let mut col_offset = 0;

    for arr in arrays {
        for j in 0..arr.ncols() {
            for i in 0..nrows {
                result[[i, col_offset + j]] = arr[[i, j]];
            }
        }
        col_offset += arr.ncols();
    }

    result
}

/// Softmax function.
fn softmax(x: &Array1<f64>) -> Array1<f64> {
    let max_val = x.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let exp_x = x.mapv(|v| (v - max_val).exp());
    let sum = exp_x.sum();
    exp_x / sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_creation() {
        let model = TimeParticleSSM::new(20, 64, 32, 4, 3);

        assert_eq!(model.n_features, 20);
        assert_eq!(model.d_hidden, 64);
        assert_eq!(model.d_state, 32);
        assert_eq!(model.n_particles, 4);
        assert_eq!(model.n_layers, 3);
    }

    #[test]
    fn test_forward() {
        let model = TimeParticleSSM::new(10, 16, 8, 2, 2);
        let x = Array2::zeros((50, 10));

        let probs = model.forward(&x);

        assert_eq!(probs.len(), 3);
        assert!((probs.sum() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_predict() {
        let model = TimeParticleSSM::new(10, 16, 8, 2, 2);
        let x = Array2::zeros((50, 10));

        let signal = model.predict(&x);

        match signal {
            TradingSignal::Buy(c) | TradingSignal::Sell(c) | TradingSignal::Hold(c) => {
                assert!(c >= 0.0 && c <= 1.0);
            }
        }
    }

    #[test]
    fn test_trading_signal() {
        let buy = TradingSignal::Buy(0.75);
        assert_eq!(buy.signal_type(), "BUY");
        assert_eq!(buy.confidence(), 0.75);

        let sell = TradingSignal::Sell(0.65);
        assert_eq!(sell.signal_type(), "SELL");

        let hold = TradingSignal::Hold(0.5);
        assert_eq!(hold.signal_type(), "HOLD");
    }

    #[test]
    fn test_softmax() {
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let y = softmax(&x);

        assert!((y.sum() - 1.0).abs() < 1e-6);
        assert!(y[2] > y[1] && y[1] > y[0]);
    }

    #[test]
    fn test_multiscale_analysis() {
        let model = TimeParticleSSM::new(10, 16, 8, 4, 2);
        let x = Array2::zeros((100, 10));

        let analysis = model.multiscale_analysis(&x);

        assert_eq!(analysis.len(), 4);
        for (name, momentum, signal) in &analysis {
            assert!(!name.is_empty());
            assert!(signal == &"bullish" || signal == &"bearish");
        }
    }
}
