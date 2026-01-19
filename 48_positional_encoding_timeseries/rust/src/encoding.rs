//! Positional Encoding Implementations
//!
//! This module provides various positional encoding algorithms
//! for transformer models processing time series data.

use ndarray::{Array1, Array2, ArrayView1};

/// Trait for positional encoding implementations
pub trait PositionalEncoding {
    /// Encode a sequence of positions
    fn encode(&self, positions: &[usize]) -> Array2<f64>;

    /// Get the encoding dimension
    fn dim(&self) -> usize;

    /// Encode a single position
    fn encode_single(&self, position: usize) -> Array1<f64> {
        self.encode(&[position]).row(0).to_owned()
    }
}

/// Sinusoidal Positional Encoding
///
/// Uses sine and cosine functions at different frequencies to encode positions.
/// This is the original encoding from "Attention is All You Need".
///
/// # Formula
///
/// PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
/// PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
///
/// # Example
///
/// ```rust
/// use positional_encoding::{SinusoidalEncoding, PositionalEncoding};
///
/// let encoding = SinusoidalEncoding::new(64, 100);
/// let result = encoding.encode(&[0, 1, 2]);
/// assert_eq!(result.shape(), &[3, 64]);
/// ```
#[derive(Debug, Clone)]
pub struct SinusoidalEncoding {
    d_model: usize,
    max_len: usize,
    base: f64,
    /// Precomputed encoding matrix for efficiency
    cache: Array2<f64>,
}

impl SinusoidalEncoding {
    /// Create a new sinusoidal encoding
    ///
    /// # Arguments
    ///
    /// * `d_model` - Dimension of the encoding (must be even)
    /// * `max_len` - Maximum sequence length to precompute
    pub fn new(d_model: usize, max_len: usize) -> Self {
        Self::with_base(d_model, max_len, 10000.0)
    }

    /// Create with custom base frequency
    pub fn with_base(d_model: usize, max_len: usize, base: f64) -> Self {
        let cache = Self::compute_encoding(d_model, max_len, base);
        Self {
            d_model,
            max_len,
            base,
            cache,
        }
    }

    fn compute_encoding(d_model: usize, max_len: usize, base: f64) -> Array2<f64> {
        let mut encoding = Array2::zeros((max_len, d_model));

        for pos in 0..max_len {
            for i in 0..(d_model / 2) {
                let angle = (pos as f64) / base.powf((2.0 * i as f64) / d_model as f64);
                encoding[[pos, 2 * i]] = angle.sin();
                encoding[[pos, 2 * i + 1]] = angle.cos();
            }
        }

        encoding
    }
}

impl PositionalEncoding for SinusoidalEncoding {
    fn encode(&self, positions: &[usize]) -> Array2<f64> {
        let mut result = Array2::zeros((positions.len(), self.d_model));
        for (i, &pos) in positions.iter().enumerate() {
            if pos < self.max_len {
                result.row_mut(i).assign(&self.cache.row(pos));
            } else {
                // Compute on-the-fly for positions beyond cache
                for j in 0..(self.d_model / 2) {
                    let angle = (pos as f64) / self.base.powf((2.0 * j as f64) / self.d_model as f64);
                    result[[i, 2 * j]] = angle.sin();
                    result[[i, 2 * j + 1]] = angle.cos();
                }
            }
        }
        result
    }

    fn dim(&self) -> usize {
        self.d_model
    }
}

/// Time Series Sinusoidal Encoding
///
/// Extended sinusoidal encoding with multiple frequency scales
/// for capturing different temporal patterns.
#[derive(Debug, Clone)]
pub struct TimeSeriesSinusoidalEncoding {
    d_model: usize,
    scales: Vec<f64>,
}

impl TimeSeriesSinusoidalEncoding {
    /// Create with default scales for financial time series
    ///
    /// Scales are optimized for hourly cryptocurrency data:
    /// - 1.0: Fine-grained position
    /// - 24.0: Daily patterns
    /// - 168.0: Weekly patterns (24 * 7)
    /// - 720.0: Monthly patterns (24 * 30)
    pub fn new(d_model: usize) -> Self {
        Self::with_scales(d_model, vec![1.0, 24.0, 168.0, 720.0])
    }

    /// Create with custom frequency scales
    pub fn with_scales(d_model: usize, scales: Vec<f64>) -> Self {
        Self { d_model, scales }
    }

    /// Encode positions with scale information
    pub fn encode_with_scales(&self, positions: &[usize]) -> Array2<f64> {
        let n_positions = positions.len();
        let dims_per_scale = self.d_model / self.scales.len();
        let mut result = Array2::zeros((n_positions, self.d_model));

        for (scale_idx, &scale) in self.scales.iter().enumerate() {
            let offset = scale_idx * dims_per_scale;
            for (pos_idx, &pos) in positions.iter().enumerate() {
                let scaled_pos = (pos as f64) / scale;
                for i in 0..(dims_per_scale / 2) {
                    let freq = 1.0 / 10000_f64.powf((2.0 * i as f64) / dims_per_scale as f64);
                    let angle = scaled_pos * freq;
                    result[[pos_idx, offset + 2 * i]] = angle.sin();
                    result[[pos_idx, offset + 2 * i + 1]] = angle.cos();
                }
            }
        }

        result
    }
}

impl PositionalEncoding for TimeSeriesSinusoidalEncoding {
    fn encode(&self, positions: &[usize]) -> Array2<f64> {
        self.encode_with_scales(positions)
    }

    fn dim(&self) -> usize {
        self.d_model
    }
}

/// Learned Positional Encoding
///
/// Position embeddings that can be trained/updated.
/// In this implementation, we initialize with random values.
#[derive(Debug, Clone)]
pub struct LearnedEncoding {
    embeddings: Array2<f64>,
}

impl LearnedEncoding {
    /// Create with random initialization
    pub fn new(d_model: usize, max_len: usize) -> Self {
        use ndarray_rand::RandomExt;
        use ndarray_rand::rand_distr::Uniform;

        let scale = 1.0 / (d_model as f64).sqrt();
        let embeddings = Array2::random((max_len, d_model), Uniform::new(-scale, scale));

        Self { embeddings }
    }

    /// Create from existing embeddings
    pub fn from_embeddings(embeddings: Array2<f64>) -> Self {
        Self { embeddings }
    }

    /// Get mutable access to embeddings for training
    pub fn embeddings_mut(&mut self) -> &mut Array2<f64> {
        &mut self.embeddings
    }

    /// Get immutable access to embeddings
    pub fn embeddings(&self) -> &Array2<f64> {
        &self.embeddings
    }
}

impl PositionalEncoding for LearnedEncoding {
    fn encode(&self, positions: &[usize]) -> Array2<f64> {
        let d_model = self.embeddings.ncols();
        let mut result = Array2::zeros((positions.len(), d_model));

        for (i, &pos) in positions.iter().enumerate() {
            if pos < self.embeddings.nrows() {
                result.row_mut(i).assign(&self.embeddings.row(pos));
            }
        }

        result
    }

    fn dim(&self) -> usize {
        self.embeddings.ncols()
    }
}

/// Relative Positional Encoding
///
/// Encodes the relative distance between positions rather than
/// absolute positions. Useful for variable-length sequences.
#[derive(Debug, Clone)]
pub struct RelativeEncoding {
    d_model: usize,
    max_distance: usize,
    /// Encoding for positive distances (to > from)
    positive_encodings: Array2<f64>,
    /// Encoding for negative distances (to < from)
    negative_encodings: Array2<f64>,
}

impl RelativeEncoding {
    /// Create a new relative encoding
    pub fn new(d_model: usize, max_distance: usize) -> Self {
        // Compute separate encodings for positive and negative distances
        // to preserve directionality information
        let positive_encodings = Self::compute_directional_encoding(d_model, max_distance, 1.0);
        let negative_encodings = Self::compute_directional_encoding(d_model, max_distance, -1.0);

        Self {
            d_model,
            max_distance,
            positive_encodings,
            negative_encodings,
        }
    }

    /// Compute directional encoding with a sign multiplier
    /// This ensures positive and negative distances have distinct encodings
    fn compute_directional_encoding(d_model: usize, max_distance: usize, sign: f64) -> Array2<f64> {
        let mut encoding = Array2::zeros((max_distance + 1, d_model));
        let base = 10000.0_f64;

        for dist in 0..=max_distance {
            // Apply sign to the position to create directional encoding
            let signed_dist = sign * (dist as f64);
            for i in 0..(d_model / 2) {
                let angle = signed_dist / base.powf((2.0 * i as f64) / d_model as f64);
                encoding[[dist, 2 * i]] = angle.sin();
                encoding[[dist, 2 * i + 1]] = angle.cos();
            }
        }

        encoding
    }

    /// Get relative position encoding between two positions
    pub fn encode_relative(&self, from: usize, to: usize) -> Array1<f64> {
        let distance = (to as i64 - from as i64).unsigned_abs() as usize;
        let clamped = distance.min(self.max_distance);

        if to >= from {
            self.positive_encodings.row(clamped).to_owned()
        } else {
            self.negative_encodings.row(clamped).to_owned()
        }
    }

    /// Compute relative encoding matrix for a sequence
    ///
    /// Returns a 3D tensor (seq_len, seq_len, d_model) containing
    /// the relative encoding between each pair of positions.
    pub fn encode_matrix(&self, seq_len: usize) -> Vec<Array2<f64>> {
        let mut result = Vec::with_capacity(seq_len);

        for i in 0..seq_len {
            let mut row = Array2::zeros((seq_len, self.d_model));
            for j in 0..seq_len {
                row.row_mut(j).assign(&self.encode_relative(i, j));
            }
            result.push(row);
        }

        result
    }
}

impl PositionalEncoding for RelativeEncoding {
    fn encode(&self, positions: &[usize]) -> Array2<f64> {
        // For relative encoding, we return the encoding relative to position 0
        let mut result = Array2::zeros((positions.len(), self.d_model));
        for (i, &pos) in positions.iter().enumerate() {
            result.row_mut(i).assign(&self.encode_relative(0, pos));
        }
        result
    }

    fn dim(&self) -> usize {
        self.d_model
    }
}

/// Rotary Positional Encoding (RoPE)
///
/// Implements the rotation-based position encoding from the RoFormer paper.
/// This encoding rotates query and key vectors based on their positions.
///
/// # Key Features
///
/// - Naturally decays with distance (no clipping needed)
/// - Works well with long sequences
/// - Position is encoded through rotation, not addition
#[derive(Debug, Clone)]
pub struct RotaryEncoding {
    d_model: usize,
    base: f64,
    /// Precomputed sin values for rotation
    sin_cache: Array2<f64>,
    /// Precomputed cos values for rotation
    cos_cache: Array2<f64>,
}

impl RotaryEncoding {
    /// Create a new rotary encoding
    pub fn new(d_model: usize, max_len: usize) -> Self {
        Self::with_base(d_model, max_len, 10000.0)
    }

    /// Create with custom base frequency
    pub fn with_base(d_model: usize, max_len: usize, base: f64) -> Self {
        let (sin_cache, cos_cache) = Self::compute_caches(d_model, max_len, base);
        Self {
            d_model,
            base,
            sin_cache,
            cos_cache,
        }
    }

    fn compute_caches(d_model: usize, max_len: usize, base: f64) -> (Array2<f64>, Array2<f64>) {
        let half_d = d_model / 2;
        let mut sin_cache = Array2::zeros((max_len, half_d));
        let mut cos_cache = Array2::zeros((max_len, half_d));

        for pos in 0..max_len {
            for i in 0..half_d {
                let theta = (pos as f64) / base.powf((2.0 * i as f64) / d_model as f64);
                sin_cache[[pos, i]] = theta.sin();
                cos_cache[[pos, i]] = theta.cos();
            }
        }

        (sin_cache, cos_cache)
    }

    /// Apply rotary encoding to a vector
    ///
    /// Rotates the input vector based on position, where pairs of
    /// dimensions are rotated together. For positions beyond the cache,
    /// values are computed on-the-fly.
    pub fn apply_rotation(&self, x: &Array1<f64>, position: usize) -> Array1<f64> {
        let half_d = self.d_model / 2;
        let mut result = Array1::zeros(self.d_model);

        // Check if position is within cache bounds
        let use_cache = position < self.sin_cache.nrows();

        for i in 0..half_d {
            let (sin_val, cos_val) = if use_cache {
                (self.sin_cache[[position, i]], self.cos_cache[[position, i]])
            } else {
                // Compute on-the-fly for out-of-range positions
                let theta = (position as f64) / self.base.powf((2.0 * i as f64) / self.d_model as f64);
                (theta.sin(), theta.cos())
            };

            // Rotate pairs: [x0, x1] -> [x0*cos - x1*sin, x0*sin + x1*cos]
            result[2 * i] = x[2 * i] * cos_val - x[2 * i + 1] * sin_val;
            result[2 * i + 1] = x[2 * i] * sin_val + x[2 * i + 1] * cos_val;
        }

        result
    }

    /// Apply rotary encoding to a batch of vectors
    pub fn apply_rotation_batch(&self, x: &Array2<f64>, positions: &[usize]) -> Array2<f64> {
        let mut result = Array2::zeros((x.nrows(), x.ncols()));
        for (i, &pos) in positions.iter().enumerate() {
            let rotated = self.apply_rotation(&x.row(i).to_owned(), pos);
            result.row_mut(i).assign(&rotated);
        }
        result
    }

    /// Get the maximum cached position
    pub fn max_len(&self) -> usize {
        self.sin_cache.nrows()
    }

    /// Get sin cache for a position
    ///
    /// Returns None if position is out of range
    pub fn get_sin(&self, position: usize) -> Option<ArrayView1<'_, f64>> {
        if position < self.sin_cache.nrows() {
            Some(self.sin_cache.row(position))
        } else {
            None
        }
    }

    /// Get cos cache for a position
    ///
    /// Returns None if position is out of range
    pub fn get_cos(&self, position: usize) -> Option<ArrayView1<'_, f64>> {
        if position < self.cos_cache.nrows() {
            Some(self.cos_cache.row(position))
        } else {
            None
        }
    }

    /// Get sin value for a position, computing on-the-fly if out of cache range
    pub fn get_sin_or_compute(&self, position: usize) -> Array1<f64> {
        if position < self.sin_cache.nrows() {
            self.sin_cache.row(position).to_owned()
        } else {
            let half_d = self.d_model / 2;
            let mut result = Array1::zeros(half_d);
            for i in 0..half_d {
                let theta = (position as f64) / self.base.powf((2.0 * i as f64) / self.d_model as f64);
                result[i] = theta.sin();
            }
            result
        }
    }

    /// Get cos value for a position, computing on-the-fly if out of cache range
    pub fn get_cos_or_compute(&self, position: usize) -> Array1<f64> {
        if position < self.cos_cache.nrows() {
            self.cos_cache.row(position).to_owned()
        } else {
            let half_d = self.d_model / 2;
            let mut result = Array1::zeros(half_d);
            for i in 0..half_d {
                let theta = (position as f64) / self.base.powf((2.0 * i as f64) / self.d_model as f64);
                result[i] = theta.cos();
            }
            result
        }
    }
}

impl PositionalEncoding for RotaryEncoding {
    fn encode(&self, positions: &[usize]) -> Array2<f64> {
        // For RoPE, we return the interleaved sin/cos values
        let half_d = self.d_model / 2;
        let mut result = Array2::zeros((positions.len(), self.d_model));
        let max_cached = self.sin_cache.nrows();

        for (i, &pos) in positions.iter().enumerate() {
            if pos < max_cached {
                // Use cached values
                for j in 0..half_d {
                    result[[i, 2 * j]] = self.sin_cache[[pos, j]];
                    result[[i, 2 * j + 1]] = self.cos_cache[[pos, j]];
                }
            } else {
                // Compute on-the-fly for out-of-range positions
                for j in 0..half_d {
                    let theta = (pos as f64) / self.base.powf((2.0 * j as f64) / self.d_model as f64);
                    result[[i, 2 * j]] = theta.sin();
                    result[[i, 2 * j + 1]] = theta.cos();
                }
            }
        }

        result
    }

    fn dim(&self) -> usize {
        self.d_model
    }
}

/// Alibi Positional Encoding (Attention with Linear Biases)
///
/// Adds linear biases based on distance to attention scores.
/// Simple and effective for long sequences.
#[derive(Debug, Clone)]
pub struct AlibiEncoding {
    n_heads: usize,
    slopes: Vec<f64>,
}

impl AlibiEncoding {
    /// Create a new Alibi encoding
    pub fn new(n_heads: usize) -> Self {
        let slopes = Self::compute_slopes(n_heads);
        Self { n_heads, slopes }
    }

    fn compute_slopes(n_heads: usize) -> Vec<f64> {
        // Slopes follow geometric sequence starting from 2^(-8/n_heads)
        let base = 2.0_f64.powf(-8.0 / n_heads as f64);
        (0..n_heads)
            .map(|i| base.powf((i + 1) as f64))
            .collect()
    }

    /// Compute attention bias for a sequence
    ///
    /// Returns (n_heads, seq_len, seq_len) bias matrix to add to attention scores
    pub fn compute_bias(&self, seq_len: usize) -> Vec<Array2<f64>> {
        let mut biases = Vec::with_capacity(self.n_heads);

        for &slope in &self.slopes {
            let mut bias = Array2::zeros((seq_len, seq_len));
            for i in 0..seq_len {
                for j in 0..seq_len {
                    // Negative bias based on distance
                    bias[[i, j]] = -slope * (i as f64 - j as f64).abs();
                }
            }
            biases.push(bias);
        }

        biases
    }

    /// Get slopes for each attention head
    pub fn slopes(&self) -> &[f64] {
        &self.slopes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_sinusoidal_encoding_shape() {
        let encoding = SinusoidalEncoding::new(64, 100);
        let result = encoding.encode(&[0, 1, 2, 3]);
        assert_eq!(result.shape(), &[4, 64]);
    }

    #[test]
    fn test_sinusoidal_encoding_position_zero() {
        let encoding = SinusoidalEncoding::new(4, 10);
        let result = encoding.encode(&[0]);

        // At position 0, sin(0) = 0 and cos(0) = 1
        assert_relative_eq!(result[[0, 0]], 0.0, epsilon = 1e-10);
        assert_relative_eq!(result[[0, 1]], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_learned_encoding_shape() {
        let encoding = LearnedEncoding::new(64, 100);
        let result = encoding.encode(&[5, 10, 15]);
        assert_eq!(result.shape(), &[3, 64]);
    }

    #[test]
    fn test_relative_encoding_symmetry() {
        let encoding = RelativeEncoding::new(32, 50);
        let forward = encoding.encode_relative(5, 10);
        let backward = encoding.encode_relative(10, 5);

        // Same distance, but may have different signs
        assert_eq!(forward.len(), backward.len());
    }

    #[test]
    fn test_rotary_encoding_rotation() {
        let encoding = RotaryEncoding::new(4, 10);
        let x = Array1::from_vec(vec![1.0, 0.0, 0.0, 1.0]);

        let rotated = encoding.apply_rotation(&x, 1);
        assert_eq!(rotated.len(), 4);

        // Rotation should preserve magnitude
        let orig_norm: f64 = x.iter().map(|v| v * v).sum::<f64>().sqrt();
        let new_norm: f64 = rotated.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert_relative_eq!(orig_norm, new_norm, epsilon = 1e-10);
    }

    #[test]
    fn test_alibi_slopes() {
        let encoding = AlibiEncoding::new(8);
        assert_eq!(encoding.slopes().len(), 8);

        // Slopes should be decreasing (more negative bias with more heads)
        for i in 1..8 {
            assert!(encoding.slopes()[i] < encoding.slopes()[i - 1]);
        }
    }

    #[test]
    fn test_time_series_encoding() {
        let encoding = TimeSeriesSinusoidalEncoding::new(64);
        let result = encoding.encode(&[0, 24, 168]);
        assert_eq!(result.shape(), &[3, 64]);
    }
}
