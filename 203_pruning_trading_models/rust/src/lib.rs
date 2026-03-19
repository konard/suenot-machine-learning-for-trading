use anyhow::Result;
use ndarray::{Array1, Array2};
use rand::Rng;
use serde::Deserialize;

// =============================================================================
// Bybit API types
// =============================================================================

#[derive(Debug, Deserialize)]
pub struct BybitResponse {
    #[serde(rename = "retCode")]
    pub ret_code: i32,
    #[serde(rename = "retMsg")]
    pub ret_msg: String,
    pub result: BybitResult,
}

#[derive(Debug, Deserialize)]
pub struct BybitResult {
    pub list: Vec<Vec<String>>,
}

#[derive(Debug, Clone)]
pub struct Candle {
    pub timestamp: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

// =============================================================================
// Neural network with pruning support
// =============================================================================

/// A single layer in a prunable neural network.
/// The mask has the same shape as the weights: 1.0 means active, 0.0 means pruned.
#[derive(Debug, Clone)]
pub struct PrunableLayer {
    pub weights: Array2<f64>, // shape: (input_size, output_size)
    pub biases: Array1<f64>,  // shape: (output_size,)
    pub mask: Array2<f64>,    // shape: (input_size, output_size)
}

/// A feedforward neural network with pruning mask support.
#[derive(Debug, Clone)]
pub struct PrunableNetwork {
    pub layers: Vec<PrunableLayer>,
}

/// Sparsity report for a network.
#[derive(Debug, Clone)]
pub struct SparsityReport {
    pub total_weights: usize,
    pub nonzero_weights: usize,
    pub sparsity: f64,
    pub layer_sparsities: Vec<f64>,
    pub compression_ratio: f64,
}

/// Result of evaluating accuracy vs sparsity.
#[derive(Debug, Clone)]
pub struct AccuracySparsityPoint {
    pub sparsity: f64,
    pub accuracy: f64,
    pub total_weights: usize,
    pub active_weights: usize,
}

// =============================================================================
// Activation functions
// =============================================================================

fn relu(x: f64) -> f64 {
    x.max(0.0)
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

// =============================================================================
// PrunableLayer implementation
// =============================================================================

impl PrunableLayer {
    /// Create a new layer with Xavier initialization.
    pub fn new(input_size: usize, output_size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (2.0 / (input_size + output_size) as f64).sqrt();

        let weights = Array2::from_shape_fn((input_size, output_size), |_| {
            rng.gen_range(-scale..scale)
        });
        let biases = Array1::zeros(output_size);
        let mask = Array2::ones((input_size, output_size));

        PrunableLayer {
            weights,
            biases,
            mask,
        }
    }

    /// Forward pass: output = activation(input @ (weights * mask) + bias)
    pub fn forward(&self, input: &Array2<f64>, use_relu: bool) -> Array2<f64> {
        let effective_weights = &self.weights * &self.mask;
        let mut output = input.dot(&effective_weights);
        // Add biases to each row
        for mut row in output.rows_mut() {
            row += &self.biases;
        }
        if use_relu {
            output.mapv_inplace(relu);
        } else {
            output.mapv_inplace(sigmoid);
        }
        output
    }

    /// Count non-zero (active) weights in this layer.
    pub fn active_weight_count(&self) -> usize {
        self.mask.iter().filter(|&&v| v > 0.5).count()
    }

    /// Total number of weights in this layer.
    pub fn total_weight_count(&self) -> usize {
        self.mask.len()
    }

    /// Layer sparsity.
    pub fn sparsity(&self) -> f64 {
        1.0 - (self.active_weight_count() as f64 / self.total_weight_count() as f64)
    }
}

// =============================================================================
// PrunableNetwork implementation
// =============================================================================

impl PrunableNetwork {
    /// Create a new network with the given layer sizes.
    /// layer_sizes: [input, hidden1, hidden2, ..., output]
    pub fn new(layer_sizes: &[usize]) -> Self {
        let mut layers = Vec::new();
        for i in 0..layer_sizes.len() - 1 {
            layers.push(PrunableLayer::new(layer_sizes[i], layer_sizes[i + 1]));
        }
        PrunableNetwork { layers }
    }

    /// Forward pass through all layers.
    /// Hidden layers use ReLU, output layer uses sigmoid.
    pub fn forward(&self, input: &Array2<f64>) -> Array2<f64> {
        let num_layers = self.layers.len();
        let mut current = input.clone();
        for (i, layer) in self.layers.iter().enumerate() {
            let use_relu = i < num_layers - 1;
            current = layer.forward(&current, use_relu);
        }
        current
    }

    /// Predict class labels (threshold = 0.5).
    pub fn predict(&self, input: &Array2<f64>) -> Vec<f64> {
        let output = self.forward(input);
        output
            .rows()
            .into_iter()
            .map(|row| if row[0] > 0.5 { 1.0 } else { 0.0 })
            .collect()
    }

    /// Compute binary cross-entropy loss.
    pub fn loss(&self, inputs: &Array2<f64>, targets: &Array1<f64>) -> f64 {
        let predictions = self.forward(inputs);
        let n = targets.len() as f64;
        let eps = 1e-7;
        let mut total_loss = 0.0;
        for (i, &t) in targets.iter().enumerate() {
            let p = predictions[[i, 0]].clamp(eps, 1.0 - eps);
            total_loss += -(t * p.ln() + (1.0 - t) * (1.0 - p).ln());
        }
        total_loss / n
    }

    /// Compute accuracy.
    pub fn accuracy(&self, inputs: &Array2<f64>, targets: &Array1<f64>) -> f64 {
        let predictions = self.predict(inputs);
        let correct = predictions
            .iter()
            .zip(targets.iter())
            .filter(|(&p, &t)| (p - t).abs() < 0.5)
            .count();
        correct as f64 / targets.len() as f64
    }

    /// Train the network using gradient-free random direction descent.
    /// This is a simplified training procedure for demonstration purposes.
    pub fn train(
        &mut self,
        inputs: &Array2<f64>,
        targets: &Array1<f64>,
        epochs: usize,
        learning_rate: f64,
    ) {
        let mut rng = rand::thread_rng();

        for _epoch in 0..epochs {
            let current_loss = self.loss(inputs, targets);

            // Random direction descent: try a random perturbation per layer
            for layer_idx in 0..self.layers.len() {
                let shape = self.layers[layer_idx].weights.raw_dim();
                let direction = Array2::from_shape_fn(shape, |_| {
                    rng.gen_range(-1.0_f64..1.0)
                }) * &self.layers[layer_idx].mask;

                // Try positive direction
                self.layers[layer_idx].weights =
                    &self.layers[layer_idx].weights + &(&direction * learning_rate);
                let new_loss = self.loss(inputs, targets);
                if new_loss < current_loss {
                    continue; // Keep the change
                }

                // Revert and try negative direction
                self.layers[layer_idx].weights =
                    &self.layers[layer_idx].weights - &(&direction * (2.0 * learning_rate));
                let new_loss2 = self.loss(inputs, targets);
                if new_loss2 < current_loss {
                    continue; // Keep the negative direction
                }

                // Neither worked, revert
                self.layers[layer_idx].weights =
                    &self.layers[layer_idx].weights + &(&direction * learning_rate);
            }
        }
    }

    /// Get a sparsity report for the network.
    pub fn sparsity_report(&self) -> SparsityReport {
        let mut total = 0usize;
        let mut nonzero = 0usize;
        let mut layer_sparsities = Vec::new();

        for layer in &self.layers {
            let lt = layer.total_weight_count();
            let la = layer.active_weight_count();
            total += lt;
            nonzero += la;
            layer_sparsities.push(1.0 - la as f64 / lt as f64);
        }

        let sparsity = 1.0 - nonzero as f64 / total as f64;
        let compression_ratio = if nonzero > 0 {
            total as f64 / nonzero as f64
        } else {
            f64::INFINITY
        };

        SparsityReport {
            total_weights: total,
            nonzero_weights: nonzero,
            sparsity,
            layer_sparsities,
            compression_ratio,
        }
    }

    /// Apply global magnitude pruning at the given target sparsity.
    /// Collects all active weights, finds the threshold at the target percentile,
    /// and masks out weights below that threshold.
    pub fn magnitude_prune(&mut self, target_sparsity: f64) {
        // Collect absolute values of all currently active weights
        let mut all_weights: Vec<f64> = Vec::new();
        for layer in &self.layers {
            for (w, &m) in layer.weights.iter().zip(layer.mask.iter()) {
                if m > 0.5 {
                    all_weights.push(w.abs());
                }
            }
        }

        if all_weights.is_empty() {
            return;
        }

        all_weights.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Find the threshold at the target sparsity percentile of active weights
        // We want to keep (1 - target_sparsity) fraction of TOTAL weights
        let total_weights: usize = self.layers.iter().map(|l| l.total_weight_count()).sum();
        let target_active = ((1.0 - target_sparsity) * total_weights as f64).round() as usize;
        let weights_to_prune = if all_weights.len() > target_active {
            all_weights.len() - target_active
        } else {
            0
        };

        if weights_to_prune == 0 {
            return;
        }

        let threshold_idx = (weights_to_prune - 1).min(all_weights.len() - 1);
        let threshold = all_weights[threshold_idx];

        // Apply the mask
        for layer in &mut self.layers {
            let shape = layer.mask.raw_dim();
            for i in 0..shape[0] {
                for j in 0..shape[1] {
                    if layer.mask[[i, j]] > 0.5 && layer.weights[[i, j]].abs() <= threshold {
                        layer.mask[[i, j]] = 0.0;
                    }
                }
            }
        }
    }

    /// Apply layer-wise magnitude pruning: each layer gets pruned independently
    /// to the target sparsity level.
    pub fn layerwise_magnitude_prune(&mut self, target_sparsity: f64) {
        for layer in &mut self.layers {
            let mut active_weights: Vec<f64> = Vec::new();
            for (w, &m) in layer.weights.iter().zip(layer.mask.iter()) {
                if m > 0.5 {
                    active_weights.push(w.abs());
                }
            }

            if active_weights.is_empty() {
                continue;
            }

            active_weights.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let total = layer.total_weight_count();
            let target_active = ((1.0 - target_sparsity) * total as f64).round() as usize;
            let to_prune = if active_weights.len() > target_active {
                active_weights.len() - target_active
            } else {
                0
            };

            if to_prune == 0 {
                continue;
            }

            let threshold_idx = (to_prune - 1).min(active_weights.len() - 1);
            let threshold = active_weights[threshold_idx];

            let shape = layer.mask.raw_dim();
            for i in 0..shape[0] {
                for j in 0..shape[1] {
                    if layer.mask[[i, j]] > 0.5 && layer.weights[[i, j]].abs() <= threshold {
                        layer.mask[[i, j]] = 0.0;
                    }
                }
            }
        }
    }

    /// Iterative magnitude pruning with retraining.
    /// Prunes `prune_fraction` of remaining weights at each step,
    /// retrains for `retrain_epochs` epochs, and repeats for `num_steps` steps.
    pub fn iterative_prune(
        &mut self,
        inputs: &Array2<f64>,
        targets: &Array1<f64>,
        num_steps: usize,
        prune_fraction: f64,
        retrain_epochs: usize,
        learning_rate: f64,
    ) -> Vec<AccuracySparsityPoint> {
        let mut results = Vec::new();

        // Record initial state
        let report = self.sparsity_report();
        let acc = self.accuracy(inputs, targets);
        results.push(AccuracySparsityPoint {
            sparsity: report.sparsity,
            accuracy: acc,
            total_weights: report.total_weights,
            active_weights: report.nonzero_weights,
        });

        for step in 0..num_steps {
            // Compute target sparsity for this step
            let current_sparsity = self.sparsity_report().sparsity;
            let remaining = 1.0 - current_sparsity;
            let new_remaining = remaining * (1.0 - prune_fraction);
            let new_sparsity = 1.0 - new_remaining;

            println!(
                "  IMP step {}/{}: pruning to {:.1}% sparsity...",
                step + 1,
                num_steps,
                new_sparsity * 100.0
            );

            // Prune
            self.magnitude_prune(new_sparsity);

            // Retrain
            self.train(inputs, targets, retrain_epochs, learning_rate);

            // Record
            let report = self.sparsity_report();
            let acc = self.accuracy(inputs, targets);
            results.push(AccuracySparsityPoint {
                sparsity: report.sparsity,
                accuracy: acc,
                total_weights: report.total_weights,
                active_weights: report.nonzero_weights,
            });

            println!(
                "    Sparsity: {:.1}%, Accuracy: {:.1}%, Active weights: {}",
                report.sparsity * 100.0,
                acc * 100.0,
                report.nonzero_weights
            );
        }

        results
    }

    /// Structured pruning: remove entire neurons from a hidden layer.
    /// Ranks neurons by the L2-norm of their incoming weights and removes
    /// the least important ones.
    pub fn structured_prune(&mut self, layer_idx: usize, neurons_to_remove: usize) {
        if layer_idx >= self.layers.len() - 1 {
            println!("Warning: Cannot structurally prune the output layer.");
            return;
        }

        let output_size = self.layers[layer_idx].weights.ncols();
        if neurons_to_remove >= output_size {
            println!("Warning: Cannot remove all neurons from a layer.");
            return;
        }

        // Compute L2-norm of incoming weights for each neuron (column)
        let mut neuron_importance: Vec<(usize, f64)> = Vec::new();
        for j in 0..output_size {
            let col = self.layers[layer_idx].weights.column(j);
            let mask_col = self.layers[layer_idx].mask.column(j);
            let norm: f64 = col
                .iter()
                .zip(mask_col.iter())
                .map(|(w, m)| (w * m).powi(2))
                .sum::<f64>()
                .sqrt();
            neuron_importance.push((j, norm));
        }

        // Sort by importance (ascending) and select neurons to remove
        neuron_importance.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let neurons_to_prune: Vec<usize> = neuron_importance
            .iter()
            .take(neurons_to_remove)
            .map(|(idx, _)| *idx)
            .collect();

        // Zero out the mask for pruned neurons
        // In the current layer: zero out the corresponding columns
        for &neuron_idx in &neurons_to_prune {
            for i in 0..self.layers[layer_idx].weights.nrows() {
                self.layers[layer_idx].mask[[i, neuron_idx]] = 0.0;
            }
            self.layers[layer_idx].biases[neuron_idx] = 0.0;
        }

        // In the next layer: zero out the corresponding rows (incoming connections)
        if layer_idx + 1 < self.layers.len() {
            for &neuron_idx in &neurons_to_prune {
                for j in 0..self.layers[layer_idx + 1].weights.ncols() {
                    self.layers[layer_idx + 1].mask[[neuron_idx, j]] = 0.0;
                }
            }
        }
    }

    /// Evaluate accuracy at multiple sparsity levels using one-shot magnitude pruning.
    /// Returns a clone of the model for each sparsity level, preserving the original.
    pub fn evaluate_sparsity_levels(
        &self,
        inputs: &Array2<f64>,
        targets: &Array1<f64>,
        sparsity_levels: &[f64],
    ) -> Vec<AccuracySparsityPoint> {
        let mut results = Vec::new();

        for &sparsity in sparsity_levels {
            let mut model_clone = self.clone();
            model_clone.magnitude_prune(sparsity);

            let report = model_clone.sparsity_report();
            let acc = model_clone.accuracy(inputs, targets);

            results.push(AccuracySparsityPoint {
                sparsity: report.sparsity,
                accuracy: acc,
                total_weights: report.total_weights,
                active_weights: report.nonzero_weights,
            });
        }

        results
    }

    /// Measure inference time for a batch of inputs (in microseconds).
    pub fn measure_inference_us(&self, input: &Array2<f64>, iterations: usize) -> f64 {
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let _ = self.forward(input);
        }
        let elapsed = start.elapsed();
        elapsed.as_micros() as f64 / iterations as f64
    }
}

// =============================================================================
// Bybit data fetching
// =============================================================================

/// Fetch OHLCV candle data from Bybit API.
pub fn fetch_bybit_candles(
    symbol: &str,
    interval: &str,
    limit: usize,
) -> Result<Vec<Candle>> {
    let url = format!(
        "https://api.bybit.com/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
        symbol, interval, limit
    );

    let client = reqwest::blocking::Client::new();
    let response: BybitResponse = client.get(&url).send()?.json()?;

    if response.ret_code != 0 {
        anyhow::bail!("Bybit API error: {}", response.ret_msg);
    }

    let mut candles: Vec<Candle> = response
        .result
        .list
        .iter()
        .filter_map(|item| {
            if item.len() >= 6 {
                Some(Candle {
                    timestamp: item[0].parse().ok()?,
                    open: item[1].parse().ok()?,
                    high: item[2].parse().ok()?,
                    low: item[3].parse().ok()?,
                    close: item[4].parse().ok()?,
                    volume: item[5].parse().ok()?,
                })
            } else {
                None
            }
        })
        .collect();

    // Bybit returns newest first, reverse to chronological order
    candles.reverse();
    Ok(candles)
}

/// Prepare features and labels from candle data.
/// Features: [return, normalized_volume, high_low_range, open_close_diff]
/// Labels: 1.0 if next close > current close, else 0.0
pub fn prepare_features(candles: &[Candle]) -> (Array2<f64>, Array1<f64>) {
    let n = candles.len();
    if n < 3 {
        return (Array2::zeros((0, 4)), Array1::zeros(0));
    }

    let mean_volume: f64 = candles.iter().map(|c| c.volume).sum::<f64>() / n as f64;
    let mean_volume = if mean_volume == 0.0 { 1.0 } else { mean_volume };

    let num_samples = n - 2; // Need previous candle for return, next candle for label
    let num_features = 4;

    let mut features = Array2::zeros((num_samples, num_features));
    let mut labels = Array1::zeros(num_samples);

    for i in 0..num_samples {
        let idx = i + 1; // Current candle index (skip first for return computation)
        let prev = &candles[idx - 1];
        let curr = &candles[idx];
        let next = &candles[idx + 1];

        // Feature 1: price return
        let ret = if prev.close != 0.0 {
            (curr.close - prev.close) / prev.close
        } else {
            0.0
        };

        // Feature 2: normalized volume
        let norm_vol = curr.volume / mean_volume;

        // Feature 3: high-low range
        let hl_range = if curr.close != 0.0 {
            (curr.high - curr.low) / curr.close
        } else {
            0.0
        };

        // Feature 4: open-close difference
        let oc_diff = if curr.open != 0.0 {
            (curr.close - curr.open) / curr.open
        } else {
            0.0
        };

        features[[i, 0]] = ret;
        features[[i, 1]] = norm_vol;
        features[[i, 2]] = hl_range;
        features[[i, 3]] = oc_diff;

        // Label: 1.0 if next close > current close
        labels[i] = if next.close > curr.close { 1.0 } else { 0.0 };
    }

    (features, labels)
}

/// Generate synthetic candle data for testing when the API is unavailable.
pub fn generate_synthetic_candles(n: usize) -> Vec<Candle> {
    let mut rng = rand::thread_rng();
    let mut candles = Vec::with_capacity(n);
    let mut price = 50000.0;

    for i in 0..n {
        let ret: f64 = rng.gen_range(-0.02..0.02);
        let open: f64 = price;
        let close: f64 = price * (1.0 + ret);
        let high = open.max(close) * (1.0 + rng.gen_range(0.0..0.01));
        let low = open.min(close) * (1.0 - rng.gen_range(0.0..0.01));
        let volume = rng.gen_range(100.0..10000.0);

        candles.push(Candle {
            timestamp: 1700000000000 + (i as u64 * 300000),
            open,
            high,
            low,
            close,
            volume,
        });

        price = close;
    }

    candles
}

// =============================================================================
// Display helpers
// =============================================================================

impl std::fmt::Display for SparsityReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "=== Sparsity Report ===")?;
        writeln!(
            f,
            "Total weights:    {}",
            self.total_weights
        )?;
        writeln!(
            f,
            "Non-zero weights: {}",
            self.nonzero_weights
        )?;
        writeln!(
            f,
            "Overall sparsity: {:.2}%",
            self.sparsity * 100.0
        )?;
        writeln!(
            f,
            "Compression ratio: {:.2}x",
            self.compression_ratio
        )?;
        for (i, &s) in self.layer_sparsities.iter().enumerate() {
            writeln!(f, "  Layer {} sparsity: {:.2}%", i, s * 100.0)?;
        }
        Ok(())
    }
}

impl std::fmt::Display for AccuracySparsityPoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Sparsity: {:.1}% | Accuracy: {:.1}% | Weights: {}/{}",
            self.sparsity * 100.0,
            self.accuracy * 100.0,
            self.active_weights,
            self.total_weights
        )
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_creation() {
        let net = PrunableNetwork::new(&[4, 8, 4, 1]);
        assert_eq!(net.layers.len(), 3);
        assert_eq!(net.layers[0].weights.nrows(), 4);
        assert_eq!(net.layers[0].weights.ncols(), 8);
        assert_eq!(net.layers[1].weights.nrows(), 8);
        assert_eq!(net.layers[1].weights.ncols(), 4);
        assert_eq!(net.layers[2].weights.nrows(), 4);
        assert_eq!(net.layers[2].weights.ncols(), 1);
    }

    #[test]
    fn test_forward_pass() {
        let net = PrunableNetwork::new(&[4, 8, 1]);
        let input = Array2::zeros((2, 4));
        let output = net.forward(&input);
        assert_eq!(output.shape(), &[2, 1]);
        // Sigmoid output should be in (0, 1)
        for &v in output.iter() {
            assert!(v > 0.0 && v < 1.0);
        }
    }

    #[test]
    fn test_initial_sparsity_is_zero() {
        let net = PrunableNetwork::new(&[4, 8, 1]);
        let report = net.sparsity_report();
        assert_eq!(report.sparsity, 0.0);
        assert_eq!(report.nonzero_weights, report.total_weights);
        assert_eq!(report.total_weights, 4 * 8 + 8 * 1);
    }

    #[test]
    fn test_magnitude_pruning_50_percent() {
        let mut net = PrunableNetwork::new(&[4, 16, 1]);
        net.magnitude_prune(0.5);
        let report = net.sparsity_report();
        // Allow small tolerance due to rounding
        assert!(
            (report.sparsity - 0.5).abs() < 0.05,
            "Expected ~50% sparsity, got {:.1}%",
            report.sparsity * 100.0
        );
    }

    #[test]
    fn test_magnitude_pruning_90_percent() {
        let mut net = PrunableNetwork::new(&[4, 32, 16, 1]);
        net.magnitude_prune(0.9);
        let report = net.sparsity_report();
        assert!(
            (report.sparsity - 0.9).abs() < 0.05,
            "Expected ~90% sparsity, got {:.1}%",
            report.sparsity * 100.0
        );
    }

    #[test]
    fn test_layerwise_pruning() {
        let mut net = PrunableNetwork::new(&[4, 16, 8, 1]);
        net.layerwise_magnitude_prune(0.5);
        let report = net.sparsity_report();
        // Each layer should be approximately 50% sparse
        for (i, &s) in report.layer_sparsities.iter().enumerate() {
            assert!(
                (s - 0.5).abs() < 0.1,
                "Layer {} expected ~50% sparsity, got {:.1}%",
                i,
                s * 100.0
            );
        }
    }

    #[test]
    fn test_structured_pruning() {
        let mut net = PrunableNetwork::new(&[4, 8, 1]);
        net.structured_prune(0, 4); // Remove 4 of 8 neurons

        // Check that 4 columns in layer 0 are fully masked out
        let masked_columns: usize = (0..8)
            .filter(|&j| {
                (0..4).all(|i| net.layers[0].mask[[i, j]] < 0.5)
            })
            .count();
        assert_eq!(masked_columns, 4);

        // Check that corresponding rows in layer 1 are also masked out
        let masked_rows: usize = (0..8)
            .filter(|&i| {
                (0..1).all(|j| net.layers[1].mask[[i, j]] < 0.5)
            })
            .count();
        assert_eq!(masked_rows, 4);
    }

    #[test]
    fn test_pruning_preserves_output_shape() {
        let net = PrunableNetwork::new(&[4, 16, 8, 1]);
        let input = Array2::from_shape_fn((5, 4), |_| rand::thread_rng().gen_range(-1.0..1.0));
        let output_before = net.forward(&input);

        let mut pruned = net.clone();
        pruned.magnitude_prune(0.8);
        let output_after = pruned.forward(&input);

        assert_eq!(output_before.shape(), output_after.shape());
    }

    #[test]
    fn test_synthetic_candles() {
        let candles = generate_synthetic_candles(100);
        assert_eq!(candles.len(), 100);
        for c in &candles {
            assert!(c.high >= c.low);
            assert!(c.volume > 0.0);
        }
    }

    #[test]
    fn test_prepare_features() {
        let candles = generate_synthetic_candles(50);
        let (features, labels) = prepare_features(&candles);
        assert_eq!(features.nrows(), 48); // n - 2
        assert_eq!(features.ncols(), 4);
        assert_eq!(labels.len(), 48);
        // Labels should be 0.0 or 1.0
        for &l in labels.iter() {
            assert!(l == 0.0 || l == 1.0);
        }
    }

    #[test]
    fn test_accuracy_computation() {
        let net = PrunableNetwork::new(&[4, 8, 1]);
        let candles = generate_synthetic_candles(50);
        let (features, labels) = prepare_features(&candles);
        let acc = net.accuracy(&features, &labels);
        assert!(acc >= 0.0 && acc <= 1.0);
    }

    #[test]
    fn test_evaluate_sparsity_levels() {
        let net = PrunableNetwork::new(&[4, 16, 1]);
        let candles = generate_synthetic_candles(50);
        let (features, labels) = prepare_features(&candles);

        let levels = vec![0.0, 0.25, 0.5, 0.75];
        let results = net.evaluate_sparsity_levels(&features, &labels, &levels);
        assert_eq!(results.len(), 4);

        // Sparsity should be approximately as requested
        for (result, &target) in results.iter().zip(levels.iter()) {
            assert!(
                (result.sparsity - target).abs() < 0.1,
                "Expected ~{:.0}% sparsity, got {:.1}%",
                target * 100.0,
                result.sparsity * 100.0
            );
        }
    }

    #[test]
    fn test_compression_ratio() {
        let mut net = PrunableNetwork::new(&[4, 16, 1]);
        net.magnitude_prune(0.75);
        let report = net.sparsity_report();
        assert!(
            report.compression_ratio > 3.0,
            "Expected compression ratio > 3x, got {:.2}x",
            report.compression_ratio
        );
    }

    #[test]
    fn test_inference_measurement() {
        let net = PrunableNetwork::new(&[4, 8, 1]);
        let input = Array2::zeros((1, 4));
        let us = net.measure_inference_us(&input, 100);
        assert!(us > 0.0);
    }
}
