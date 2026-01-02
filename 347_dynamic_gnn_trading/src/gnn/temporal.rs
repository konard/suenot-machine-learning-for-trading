//! Temporal components for Dynamic GNN

use ndarray::{Array1, Array2};
use rand::Rng;
use std::collections::VecDeque;
use std::f64::consts::PI;

/// Time encoder using Fourier features
#[derive(Debug, Clone)]
pub struct TimeEncoder {
    /// Output dimension
    pub dim: usize,
    /// Frequency parameters
    frequencies: Array1<f64>,
    /// Phase offsets
    phases: Array1<f64>,
}

impl TimeEncoder {
    /// Create a new time encoder
    pub fn new(dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let half_dim = dim / 2;

        // Log-spaced frequencies for multi-scale time encoding
        let frequencies = Array1::from_shape_fn(half_dim, |i| {
            10f64.powf(-(4.0 * i as f64) / half_dim as f64)
        });

        let phases = Array1::from_shape_fn(half_dim, |_| rng.gen_range(0.0..2.0 * PI));

        Self {
            dim,
            frequencies,
            phases,
        }
    }

    /// Encode timestamps
    pub fn encode(&self, timestamps: &Array1<f64>) -> Array2<f64> {
        let n = timestamps.len();
        let half_dim = self.dim / 2;
        let mut encoded = Array2::zeros((n, self.dim));

        for i in 0..n {
            let t = timestamps[i];
            for j in 0..half_dim {
                let angle = t * self.frequencies[j] + self.phases[j];
                encoded[[i, 2 * j]] = angle.cos();
                encoded[[i, 2 * j + 1]] = angle.sin();
            }
        }

        encoded
    }

    /// Encode a single timestamp
    pub fn encode_single(&self, timestamp: f64) -> Array1<f64> {
        let half_dim = self.dim / 2;
        let mut encoded = Array1::zeros(self.dim);

        for j in 0..half_dim {
            let angle = timestamp * self.frequencies[j] + self.phases[j];
            encoded[2 * j] = angle.cos();
            encoded[2 * j + 1] = angle.sin();
        }

        encoded
    }

    /// Encode time difference
    pub fn encode_delta(&self, delta_t: f64) -> Array1<f64> {
        self.encode_single(delta_t)
    }
}

/// Temporal memory module for remembering past states
#[derive(Debug)]
pub struct TemporalMemory {
    /// Hidden dimension
    pub dim: usize,
    /// Memory capacity
    pub capacity: usize,
    /// Memory storage: node_id -> history of embeddings
    memory: Vec<VecDeque<Array1<f64>>>,
    /// Number of nodes being tracked
    num_nodes: usize,
    /// Forget gate weights
    forget_weights: Array2<f64>,
    /// Input gate weights
    input_weights: Array2<f64>,
    /// Output gate weights
    output_weights: Array2<f64>,
    /// Cell state weights
    cell_weights: Array2<f64>,
}

impl TemporalMemory {
    /// Create a new temporal memory
    pub fn new(dim: usize, capacity: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (2.0 / (2.0 * dim as f64)).sqrt();

        let forget_weights =
            Array2::from_shape_fn((2 * dim, dim), |_| rng.gen_range(-scale..scale));
        let input_weights =
            Array2::from_shape_fn((2 * dim, dim), |_| rng.gen_range(-scale..scale));
        let output_weights =
            Array2::from_shape_fn((2 * dim, dim), |_| rng.gen_range(-scale..scale));
        let cell_weights =
            Array2::from_shape_fn((2 * dim, dim), |_| rng.gen_range(-scale..scale));

        Self {
            dim,
            capacity,
            memory: Vec::new(),
            num_nodes: 0,
            forget_weights,
            input_weights,
            output_weights,
            cell_weights,
        }
    }

    /// Initialize memory for a number of nodes
    pub fn initialize(&mut self, num_nodes: usize) {
        self.num_nodes = num_nodes;
        self.memory = (0..num_nodes)
            .map(|_| VecDeque::with_capacity(self.capacity))
            .collect();
    }

    /// Update memory with new embeddings and return updated embeddings
    pub fn update(&mut self, embeddings: &Array2<f64>) -> Array2<f64> {
        let n = embeddings.nrows();

        // Initialize if needed
        if self.num_nodes != n {
            self.initialize(n);
        }

        let mut output = Array2::zeros((n, self.dim));

        for i in 0..n {
            let current = embeddings.row(i).to_owned();

            // Get previous state or zeros
            let prev = if let Some(history) = self.memory.get(i) {
                history.back().cloned().unwrap_or_else(|| Array1::zeros(self.dim))
            } else {
                Array1::zeros(self.dim)
            };

            // LSTM-like update
            let updated = self.lstm_update(&current, &prev);

            // Store in memory
            if let Some(history) = self.memory.get_mut(i) {
                history.push_back(updated.clone());
                if history.len() > self.capacity {
                    history.pop_front();
                }
            }

            // Set output
            for j in 0..self.dim {
                output[[i, j]] = updated[j];
            }
        }

        output
    }

    /// LSTM-style update
    fn lstm_update(&self, current: &Array1<f64>, previous: &Array1<f64>) -> Array1<f64> {
        // Concatenate current and previous: [h_t || h_{t-1}]
        let mut concat = Array1::zeros(2 * self.dim);
        for i in 0..self.dim {
            concat[i] = current[i];
            concat[self.dim + i] = previous[i];
        }

        // Gate computations
        let forget_gate = self.gate(&concat, &self.forget_weights);
        let input_gate = self.gate(&concat, &self.input_weights);
        let output_gate = self.gate(&concat, &self.output_weights);
        let cell_candidate = self.tanh_transform(&concat, &self.cell_weights);

        // Update cell state
        let mut cell = Array1::zeros(self.dim);
        for i in 0..self.dim {
            cell[i] = forget_gate[i] * previous[i] + input_gate[i] * cell_candidate[i];
        }

        // Compute output
        let mut output = Array1::zeros(self.dim);
        for i in 0..self.dim {
            output[i] = output_gate[i] * cell[i].tanh();
        }

        output
    }

    /// Sigmoid gate
    fn gate(&self, input: &Array1<f64>, weights: &Array2<f64>) -> Array1<f64> {
        let linear: Array1<f64> = weights.t().dot(input);
        linear.mapv(|x| 1.0 / (1.0 + (-x).exp()))
    }

    /// Tanh transformation
    fn tanh_transform(&self, input: &Array1<f64>, weights: &Array2<f64>) -> Array1<f64> {
        let linear: Array1<f64> = weights.t().dot(input);
        linear.mapv(|x| x.tanh())
    }

    /// Get memory for a specific node
    pub fn get_node_memory(&self, node_idx: usize) -> Option<&VecDeque<Array1<f64>>> {
        self.memory.get(node_idx)
    }

    /// Aggregate historical embeddings for a node
    pub fn aggregate_history(&self, node_idx: usize) -> Option<Array1<f64>> {
        self.memory.get(node_idx).map(|history| {
            if history.is_empty() {
                Array1::zeros(self.dim)
            } else {
                // Weighted average with exponential decay
                let mut result = Array1::zeros(self.dim);
                let mut weight_sum = 0.0;
                let decay = 0.9;

                for (i, embedding) in history.iter().rev().enumerate() {
                    let weight = decay.powi(i as i32);
                    result = result + embedding * weight;
                    weight_sum += weight;
                }

                result / weight_sum
            }
        })
    }

    /// Clear all memory
    pub fn clear(&mut self) {
        for history in &mut self.memory {
            history.clear();
        }
    }

    /// Get parameter count
    pub fn param_count(&self) -> usize {
        self.forget_weights.len()
            + self.input_weights.len()
            + self.output_weights.len()
            + self.cell_weights.len()
    }
}

/// Temporal Graph Attention - attention over time
#[derive(Debug, Clone)]
pub struct TemporalAttention {
    /// Hidden dimension
    pub dim: usize,
    /// Number of time steps to attend over
    pub window_size: usize,
    /// Query projection
    w_q: Array2<f64>,
    /// Key projection
    w_k: Array2<f64>,
    /// Value projection
    w_v: Array2<f64>,
    /// Time encoder
    time_encoder: TimeEncoder,
}

impl TemporalAttention {
    /// Create new temporal attention
    pub fn new(dim: usize, window_size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (2.0 / dim as f64).sqrt();

        let w_q = Array2::from_shape_fn((dim, dim), |_| rng.gen_range(-scale..scale));
        let w_k = Array2::from_shape_fn((dim, dim), |_| rng.gen_range(-scale..scale));
        let w_v = Array2::from_shape_fn((dim, dim), |_| rng.gen_range(-scale..scale));

        let time_encoder = TimeEncoder::new(dim);

        Self {
            dim,
            window_size,
            w_q,
            w_k,
            w_v,
            time_encoder,
        }
    }

    /// Attend over temporal sequence
    pub fn forward(&self, history: &[Array1<f64>], timestamps: &[f64]) -> Array1<f64> {
        let seq_len = history.len().min(self.window_size);
        if seq_len == 0 {
            return Array1::zeros(self.dim);
        }

        // Get the most recent entries
        let start = history.len().saturating_sub(self.window_size);
        let history = &history[start..];
        let timestamps = &timestamps[start..];

        // Current time (last timestamp)
        let current_time = *timestamps.last().unwrap_or(&0.0);

        // Project query from current embedding
        let query = self.project_vector(history.last().unwrap(), &self.w_q);

        // Compute attention scores
        let mut scores = Vec::with_capacity(seq_len);
        let mut keys = Vec::with_capacity(seq_len);
        let mut values = Vec::with_capacity(seq_len);

        for (i, (emb, &ts)) in history.iter().zip(timestamps.iter()).enumerate() {
            // Add time encoding to key
            let time_delta = current_time - ts;
            let time_enc = self.time_encoder.encode_delta(time_delta);

            let key = self.project_vector(emb, &self.w_k);
            let key_with_time: Array1<f64> = &key + &time_enc;

            let value = self.project_vector(emb, &self.w_v);

            // Compute attention score
            let score: f64 = query.iter().zip(key_with_time.iter()).map(|(q, k)| q * k).sum();
            let score = score / (self.dim as f64).sqrt();

            scores.push(score);
            keys.push(key_with_time);
            values.push(value);
        }

        // Softmax
        let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_scores: Vec<f64> = scores.iter().map(|s| (s - max_score).exp()).collect();
        let sum: f64 = exp_scores.iter().sum();
        let attention: Vec<f64> = exp_scores.iter().map(|e| e / sum).collect();

        // Weighted sum of values
        let mut output = Array1::zeros(self.dim);
        for (attn, value) in attention.iter().zip(values.iter()) {
            for j in 0..self.dim {
                output[j] += attn * value[j];
            }
        }

        output
    }

    /// Project a vector using weight matrix
    fn project_vector(&self, v: &Array1<f64>, w: &Array2<f64>) -> Array1<f64> {
        w.t().dot(v)
    }

    /// Get parameter count
    pub fn param_count(&self) -> usize {
        self.w_q.len() + self.w_k.len() + self.w_v.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_time_encoder() {
        let encoder = TimeEncoder::new(8);
        let timestamps = Array1::from_vec(vec![0.0, 1.0, 2.0]);

        let encoded = encoder.encode(&timestamps);
        assert_eq!(encoded.shape(), &[3, 8]);

        // Different timestamps should give different encodings
        assert!(encoded.row(0) != encoded.row(1));
    }

    #[test]
    fn test_temporal_memory() {
        let mut memory = TemporalMemory::new(4, 10);

        let embeddings = Array2::from_shape_fn((3, 4), |_| rand::random::<f64>());
        let output1 = memory.update(&embeddings);
        assert_eq!(output1.shape(), &[3, 4]);

        // Second update should produce different results
        let output2 = memory.update(&embeddings);
        assert!(output1 != output2);
    }

    #[test]
    fn test_temporal_attention() {
        let attention = TemporalAttention::new(8, 5);

        let history: Vec<Array1<f64>> = (0..3)
            .map(|_| Array1::from_shape_fn(8, |_| rand::random::<f64>()))
            .collect();
        let timestamps = vec![0.0, 1.0, 2.0];

        let output = attention.forward(&history, &timestamps);
        assert_eq!(output.len(), 8);
    }
}
