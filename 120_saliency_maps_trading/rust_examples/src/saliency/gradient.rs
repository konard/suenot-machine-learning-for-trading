//! Saliency map computation using gradient-based methods
//!
//! This module provides methods to compute saliency maps that show
//! which input features most influence the model's predictions.

use crate::models::network::TradingNetwork;
use ndarray::Array2;

/// Saliency computation methods
pub struct SaliencyComputer {
    /// The network to compute saliency for
    network: TradingNetwork,
    /// Epsilon for numerical gradient computation
    epsilon: f64,
}

impl SaliencyComputer {
    /// Create a new saliency computer
    pub fn new(network: TradingNetwork) -> Self {
        Self {
            network,
            epsilon: 1e-5,
        }
    }

    /// Compute vanilla gradient saliency using numerical differentiation
    ///
    /// Returns a saliency map with the same shape as the input
    pub fn vanilla_gradient(&self, input: &Array2<f64>) -> Array2<f64> {
        let (rows, cols) = input.dim();
        let mut saliency = Array2::zeros((rows, cols));

        for i in 0..rows {
            for j in 0..cols {
                let mut input_plus = input.clone();
                let mut input_minus = input.clone();

                input_plus[[i, j]] += self.epsilon;
                input_minus[[i, j]] -= self.epsilon;

                let output_plus = self.network.forward(&input_plus);
                let output_minus = self.network.forward(&input_minus);

                let grad = (output_plus - output_minus) / (2.0 * self.epsilon);
                saliency[[i, j]] = grad.abs();
            }
        }

        saliency
    }

    /// Compute gradient × input saliency
    ///
    /// This method multiplies the gradient by the input value,
    /// providing better attribution for features with non-zero values.
    pub fn gradient_x_input(&self, input: &Array2<f64>) -> Array2<f64> {
        let gradient = self.vanilla_gradient(input);
        &gradient * input.mapv(|x| x.abs())
    }

    /// Compute integrated gradients
    ///
    /// Integrates gradients along a path from baseline to input
    pub fn integrated_gradients(&self, input: &Array2<f64>, steps: usize) -> Array2<f64> {
        let (rows, cols) = input.dim();
        let baseline = Array2::zeros((rows, cols));

        let mut integrated = Array2::zeros((rows, cols));

        for step in 0..=steps {
            let alpha = step as f64 / steps as f64;
            let interpolated = &baseline + alpha * (input - &baseline);
            let gradient = self.compute_gradient(&interpolated);
            integrated = integrated + gradient;
        }

        // Average gradients and multiply by (input - baseline)
        let avg_gradient = integrated / (steps + 1) as f64;
        (input - &baseline) * avg_gradient
    }

    /// Compute gradient at a point
    fn compute_gradient(&self, input: &Array2<f64>) -> Array2<f64> {
        let (rows, cols) = input.dim();
        let mut gradient = Array2::zeros((rows, cols));

        for i in 0..rows {
            for j in 0..cols {
                let mut input_plus = input.clone();
                let mut input_minus = input.clone();

                input_plus[[i, j]] += self.epsilon;
                input_minus[[i, j]] -= self.epsilon;

                let output_plus = self.network.forward(&input_plus);
                let output_minus = self.network.forward(&input_minus);

                gradient[[i, j]] = (output_plus - output_minus) / (2.0 * self.epsilon);
            }
        }

        gradient
    }

    /// Compute SmoothGrad saliency
    ///
    /// Averages gradients over noisy versions of the input
    pub fn smoothgrad(&self, input: &Array2<f64>, noise_level: f64, num_samples: usize) -> Array2<f64> {
        let (rows, cols) = input.dim();
        let mut total_saliency = Array2::zeros((rows, cols));

        let mut rng = rand::thread_rng();

        for _ in 0..num_samples {
            // Add noise to input
            let noise = Array2::from_shape_fn((rows, cols), |_| {
                use rand::Rng;
                rng.gen_range(-noise_level..noise_level)
            });
            let noisy_input = input + &noise;

            // Compute gradient for noisy input
            let gradient = self.vanilla_gradient(&noisy_input);
            total_saliency = total_saliency + gradient;
        }

        total_saliency / num_samples as f64
    }

    /// Aggregate saliency across features to get temporal importance
    pub fn temporal_importance(&self, saliency: &Array2<f64>) -> Vec<f64> {
        saliency
            .rows()
            .into_iter()
            .map(|row| row.iter().sum::<f64>() / row.len() as f64)
            .collect()
    }

    /// Aggregate saliency across time to get feature importance
    pub fn feature_importance(&self, saliency: &Array2<f64>) -> Vec<f64> {
        let num_features = saliency.ncols();
        let mut importance = vec![0.0; num_features];

        for col_idx in 0..num_features {
            let col_sum: f64 = saliency.column(col_idx).iter().sum();
            importance[col_idx] = col_sum / saliency.nrows() as f64;
        }

        importance
    }

    /// Compute saliency concentration (inverse entropy)
    ///
    /// High concentration means the model focuses on few features
    pub fn concentration(&self, saliency: &Array2<f64>) -> f64 {
        let flat: Vec<f64> = saliency.iter().cloned().collect();
        let sum: f64 = flat.iter().sum();

        if sum < 1e-10 {
            return 0.0;
        }

        // Compute normalized distribution
        let probs: Vec<f64> = flat.iter().map(|&x| x / sum).collect();

        // Compute entropy
        let entropy: f64 = probs
            .iter()
            .filter(|&&p| p > 1e-10)
            .map(|&p| -p * p.ln())
            .sum();

        // Max entropy for uniform distribution
        let max_entropy = (flat.len() as f64).ln();

        // Concentration = 1 - normalized entropy
        1.0 - entropy / max_entropy
    }

    /// Check if top salient features are in interpretable set
    pub fn check_interpretability(
        &self,
        saliency: &Array2<f64>,
        interpretable_features: &[usize],
        top_k: usize,
    ) -> bool {
        let feature_imp = self.feature_importance(saliency);

        // Get indices of top-k features
        let mut indexed: Vec<(usize, f64)> = feature_imp.iter().cloned().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let top_features: Vec<usize> = indexed.iter().take(top_k).map(|(i, _)| *i).collect();

        // Count overlap
        let overlap = top_features
            .iter()
            .filter(|f| interpretable_features.contains(f))
            .count();

        overlap >= (top_k / 2 + 1)
    }
}

/// Saliency-based trading signal generator
pub struct SaliencyTrader {
    /// Saliency computer
    computer: SaliencyComputer,
    /// Minimum confidence threshold
    min_confidence: f64,
    /// Minimum concentration threshold
    min_concentration: f64,
    /// Interpretable feature indices
    interpretable_features: Vec<usize>,
}

impl SaliencyTrader {
    /// Create a new saliency-based trader
    pub fn new(
        network: TradingNetwork,
        min_confidence: f64,
        min_concentration: f64,
        interpretable_features: Vec<usize>,
    ) -> Self {
        Self {
            computer: SaliencyComputer::new(network),
            min_confidence,
            min_concentration,
            interpretable_features,
        }
    }

    /// Generate trading signal
    ///
    /// Returns (direction, position_size) where direction is -1, 0, or 1
    pub fn generate_signal(&self, input: &Array2<f64>) -> (i32, f64) {
        let (direction, confidence) = self.computer.network.predict(input);

        // Check confidence
        if confidence < self.min_confidence - 0.5 {
            return (0, 0.0);
        }

        // Compute saliency
        let saliency = self.computer.gradient_x_input(input);
        let concentration = self.computer.concentration(&saliency);

        // Check concentration
        if concentration < self.min_concentration {
            return (0, 0.0);
        }

        // Check interpretability
        if !self.interpretable_features.is_empty() {
            if !self.computer.check_interpretability(&saliency, &self.interpretable_features, 3) {
                return (0, 0.0);
            }
        }

        // Return signal with position size based on concentration
        (direction, concentration)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vanilla_gradient() {
        let network = TradingNetwork::new(30, 5, &[32, 16]);
        let computer = SaliencyComputer::new(network);

        let input = Array2::from_shape_fn((30, 5), |_| 0.5);
        let saliency = computer.vanilla_gradient(&input);

        assert_eq!(saliency.dim(), input.dim());
        assert!(saliency.iter().all(|&x| x >= 0.0));
    }

    #[test]
    fn test_concentration() {
        let network = TradingNetwork::new(30, 5, &[32, 16]);
        let computer = SaliencyComputer::new(network);

        // Uniform saliency should have low concentration
        let uniform = Array2::from_elem((30, 5), 1.0);
        let conc_uniform = computer.concentration(&uniform);

        // Concentrated saliency should have high concentration
        let mut concentrated = Array2::zeros((30, 5));
        concentrated[[0, 0]] = 100.0;
        let conc_focused = computer.concentration(&concentrated);

        assert!(conc_focused > conc_uniform);
    }
}
