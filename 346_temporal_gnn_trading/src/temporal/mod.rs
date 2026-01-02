//! Temporal module for TGN
//!
//! This module implements time encoding and temporal attention
//! mechanisms for capturing temporal patterns.

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// Time2Vec encoding that captures both periodic and non-periodic patterns
#[derive(Debug, Clone)]
pub struct Time2Vec {
    /// Dimension of the encoding
    dim: usize,
    /// Learnable frequencies
    frequencies: Array1<f64>,
    /// Learnable phases
    phases: Array1<f64>,
    /// Linear coefficient for non-periodic component
    linear_coef: f64,
}

impl Time2Vec {
    /// Create a new Time2Vec encoder
    pub fn new(dim: usize) -> Self {
        let frequencies = Array1::from_shape_fn(dim - 1, |i| {
            // Initialize with different frequencies for multi-scale patterns
            2.0 * PI / (10.0_f64.powi(i as i32 / 4))
        });

        let phases = Array1::zeros(dim - 1);
        let linear_coef = 0.001;

        Self {
            dim,
            frequencies,
            phases,
            linear_coef,
        }
    }

    /// Encode a time delta
    pub fn encode(&self, t: f64) -> Array1<f64> {
        let mut encoding = Array1::zeros(self.dim);

        // First component is linear
        encoding[0] = self.linear_coef * t;

        // Remaining components are periodic
        for i in 1..self.dim {
            let idx = i - 1;
            encoding[i] = (self.frequencies[idx] * t + self.phases[idx]).sin();
        }

        encoding
    }

    /// Batch encode multiple time deltas
    pub fn encode_batch(&self, times: &[f64]) -> Array2<f64> {
        let n = times.len();
        let mut encodings = Array2::zeros((n, self.dim));

        for (i, &t) in times.iter().enumerate() {
            let enc = self.encode(t);
            encodings.row_mut(i).assign(&enc);
        }

        encodings
    }
}

/// Functional time encoder using sinusoidal functions
#[derive(Debug, Clone)]
pub struct TimeEncoder {
    /// Dimension of the encoding
    dim: usize,
    /// Base frequencies for different time scales
    frequencies: Array1<f64>,
}

impl TimeEncoder {
    /// Create a new time encoder
    pub fn new(dim: usize) -> Self {
        // Create frequencies for different time scales:
        // milliseconds, seconds, minutes, hours, days
        let frequencies = Array1::from_shape_fn(dim / 2, |i| {
            let scale = match i % 5 {
                0 => 1.0,          // milliseconds
                1 => 1000.0,       // seconds
                2 => 60000.0,      // minutes
                3 => 3600000.0,    // hours
                4 => 86400000.0,   // days
                _ => 1.0,
            };
            2.0 * PI / scale * (1.0 + i as f64 / 5.0)
        });

        Self { dim, frequencies }
    }

    /// Encode a time delta (in milliseconds)
    pub fn encode(&self, delta_t: f64) -> Array1<f64> {
        let mut encoding = Array1::zeros(self.dim);

        for (i, &freq) in self.frequencies.iter().enumerate() {
            let angle = freq * delta_t;
            encoding[i * 2] = angle.cos();
            if i * 2 + 1 < self.dim {
                encoding[i * 2 + 1] = angle.sin();
            }
        }

        // Normalize
        let norm = encoding.dot(&encoding).sqrt();
        if norm > 0.0 {
            encoding /= norm;
        }

        encoding
    }

    /// Get encoding dimension
    pub fn dim(&self) -> usize {
        self.dim
    }
}

/// Temporal attention mechanism for aggregating neighbor information
#[derive(Debug)]
pub struct TemporalAttention {
    /// Hidden dimension
    dim: usize,
    /// Number of attention heads
    num_heads: usize,
    /// Head dimension
    head_dim: usize,
    /// Query projection
    w_query: Array2<f64>,
    /// Key projection
    w_key: Array2<f64>,
    /// Value projection
    w_value: Array2<f64>,
    /// Output projection
    w_out: Array2<f64>,
}

impl TemporalAttention {
    /// Create new temporal attention
    pub fn new(dim: usize, num_heads: usize) -> Self {
        let head_dim = dim / num_heads;

        Self {
            dim,
            num_heads,
            head_dim,
            w_query: Array2::from_shape_fn((dim, dim), |_| {
                rand::random::<f64>() * 0.1 - 0.05
            }),
            w_key: Array2::from_shape_fn((dim, dim), |_| {
                rand::random::<f64>() * 0.1 - 0.05
            }),
            w_value: Array2::from_shape_fn((dim, dim), |_| {
                rand::random::<f64>() * 0.1 - 0.05
            }),
            w_out: Array2::from_shape_fn((dim, dim), |_| {
                rand::random::<f64>() * 0.1 - 0.05
            }),
        }
    }

    /// Aggregate neighbor memories using attention
    pub fn aggregate(&self, query: &Array1<f64>, neighbors: &[Array1<f64>]) -> Array1<f64> {
        if neighbors.is_empty() {
            return query.clone();
        }

        // Project query
        let q = self.project(query, &self.w_query);

        // Project keys and values
        let keys: Vec<Array1<f64>> = neighbors
            .iter()
            .map(|n| self.project(n, &self.w_key))
            .collect();

        let values: Vec<Array1<f64>> = neighbors
            .iter()
            .map(|n| self.project(n, &self.w_value))
            .collect();

        // Compute attention scores
        let scores: Vec<f64> = keys
            .iter()
            .map(|k| q.dot(k) / (self.head_dim as f64).sqrt())
            .collect();

        // Softmax
        let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_scores: Vec<f64> = scores.iter().map(|&s| (s - max_score).exp()).collect();
        let sum_exp: f64 = exp_scores.iter().sum();
        let weights: Vec<f64> = exp_scores.iter().map(|&e| e / sum_exp).collect();

        // Weighted sum of values
        let mut aggregated = Array1::zeros(self.dim);
        for (value, &weight) in values.iter().zip(weights.iter()) {
            aggregated = aggregated + value * weight;
        }

        // Output projection and residual connection
        let output = aggregated.dot(&self.w_out);
        query + &output
    }

    /// Project a vector using weight matrix
    fn project(&self, x: &Array1<f64>, w: &Array2<f64>) -> Array1<f64> {
        // Handle dimension mismatch
        if x.len() != w.nrows() {
            let mut padded = Array1::zeros(w.nrows());
            let copy_len = x.len().min(w.nrows());
            padded.slice_mut(ndarray::s![..copy_len]).assign(&x.slice(ndarray::s![..copy_len]));
            padded.dot(w)
        } else {
            x.dot(w)
        }
    }

    /// Get number of attention heads
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }
}

/// Multi-scale temporal encoder for different time resolutions
#[derive(Debug)]
pub struct MultiScaleTimeEncoder {
    /// Individual encoders for each scale
    encoders: Vec<TimeEncoder>,
    /// Total output dimension
    total_dim: usize,
}

impl MultiScaleTimeEncoder {
    /// Create multi-scale time encoder
    pub fn new(dim_per_scale: usize, num_scales: usize) -> Self {
        let encoders: Vec<TimeEncoder> = (0..num_scales)
            .map(|_| TimeEncoder::new(dim_per_scale))
            .collect();

        Self {
            total_dim: dim_per_scale * num_scales,
            encoders,
        }
    }

    /// Encode time delta at multiple scales
    pub fn encode(&self, delta_t: f64) -> Array1<f64> {
        let mut encoding = Array1::zeros(self.total_dim);
        let dim_per_scale = self.total_dim / self.encoders.len();

        for (i, encoder) in self.encoders.iter().enumerate() {
            // Scale the time differently for each encoder
            let scale = 10.0_f64.powi(i as i32);
            let scaled_t = delta_t / scale;
            let enc = encoder.encode(scaled_t);

            let start = i * dim_per_scale;
            let end = start + dim_per_scale;
            encoding.slice_mut(ndarray::s![start..end]).assign(&enc);
        }

        encoding
    }

    /// Get total encoding dimension
    pub fn dim(&self) -> usize {
        self.total_dim
    }
}

/// Temporal feature extractor combining multiple temporal aspects
#[derive(Debug)]
pub struct TemporalFeatures {
    /// Time since last event
    pub time_delta: f64,
    /// Hour of day (0-23)
    pub hour_of_day: u32,
    /// Day of week (0-6)
    pub day_of_week: u32,
    /// Is weekend
    pub is_weekend: bool,
    /// Event rate (events per minute)
    pub event_rate: f64,
}

impl TemporalFeatures {
    /// Create from timestamp
    pub fn from_timestamp(timestamp: u64, last_timestamp: u64, recent_event_count: usize) -> Self {
        let time_delta = (timestamp.saturating_sub(last_timestamp)) as f64 / 1000.0;

        // Extract hour and day from timestamp (simplified - assuming UTC)
        let secs = (timestamp / 1000) % 86400;
        let hour_of_day = ((secs / 3600) % 24) as u32;

        let days = (timestamp / 1000) / 86400;
        let day_of_week = (days % 7) as u32;

        let is_weekend = day_of_week >= 5;

        // Calculate event rate (events per minute in last window)
        let event_rate = if time_delta > 0.0 {
            recent_event_count as f64 / (time_delta / 60.0)
        } else {
            0.0
        };

        Self {
            time_delta,
            hour_of_day,
            day_of_week,
            is_weekend,
            event_rate,
        }
    }

    /// Convert to feature vector
    pub fn to_vector(&self) -> Array1<f64> {
        let mut features = Array1::zeros(8);

        // Normalized time delta (log scale)
        features[0] = (1.0 + self.time_delta).ln();

        // Cyclical hour encoding
        let hour_angle = 2.0 * PI * self.hour_of_day as f64 / 24.0;
        features[1] = hour_angle.sin();
        features[2] = hour_angle.cos();

        // Cyclical day encoding
        let day_angle = 2.0 * PI * self.day_of_week as f64 / 7.0;
        features[3] = day_angle.sin();
        features[4] = day_angle.cos();

        // Weekend flag
        features[5] = if self.is_weekend { 1.0 } else { 0.0 };

        // Event rate (normalized)
        features[6] = (1.0 + self.event_rate).ln();

        // Combined temporal signal
        features[7] = features[0] * features[6];

        features
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_time2vec() {
        let encoder = Time2Vec::new(8);
        let encoding = encoder.encode(1000.0);
        assert_eq!(encoding.len(), 8);
    }

    #[test]
    fn test_time_encoder() {
        let encoder = TimeEncoder::new(16);
        let encoding = encoder.encode(5000.0);
        assert_eq!(encoding.len(), 16);

        // Check normalization
        let norm = encoding.dot(&encoding).sqrt();
        assert!((norm - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_temporal_attention() {
        let attention = TemporalAttention::new(16, 4);

        let query = Array1::from_shape_fn(16, |_| rand::random::<f64>());
        let neighbors: Vec<Array1<f64>> = (0..5)
            .map(|_| Array1::from_shape_fn(16, |_| rand::random::<f64>()))
            .collect();

        let result = attention.aggregate(&query, &neighbors);
        assert_eq!(result.len(), 16);
    }

    #[test]
    fn test_multi_scale_encoder() {
        let encoder = MultiScaleTimeEncoder::new(8, 3);
        let encoding = encoder.encode(10000.0);
        assert_eq!(encoding.len(), 24);
    }

    #[test]
    fn test_temporal_features() {
        let features = TemporalFeatures::from_timestamp(1704110400000, 1704110395000, 10);
        let vector = features.to_vector();
        assert_eq!(vector.len(), 8);
    }
}
