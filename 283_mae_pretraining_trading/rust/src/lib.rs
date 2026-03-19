use anyhow::Result;
use ndarray::{Array1, Array2};
use rand::seq::SliceRandom;
use rand::Rng;
use serde::Deserialize;

// =============================================================================
// OHLCV Data Structures
// =============================================================================

/// A single OHLCV candle
#[derive(Debug, Clone)]
pub struct Candle {
    pub timestamp: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

impl Candle {
    pub fn to_array(&self) -> [f64; 5] {
        [self.open, self.high, self.low, self.close, self.volume]
    }
}

/// Convert candles to an ndarray matrix (rows=time, cols=features)
pub fn candles_to_matrix(candles: &[Candle]) -> Array2<f64> {
    let n = candles.len();
    let mut data = Array2::<f64>::zeros((n, 5));
    for (i, c) in candles.iter().enumerate() {
        let arr = c.to_array();
        for (j, &val) in arr.iter().enumerate() {
            data[[i, j]] = val;
        }
    }
    data
}

/// Normalize candle data to log returns (for prices) and log volume
pub fn normalize_to_log_returns(data: &Array2<f64>) -> Array2<f64> {
    let (n, d) = data.dim();
    if n < 2 {
        return Array2::<f64>::zeros((0, d));
    }
    let mut result = Array2::<f64>::zeros((n - 1, d));
    for i in 1..n {
        for j in 0..4 {
            // Log returns for OHLC
            let prev = data[[i - 1, j]];
            let curr = data[[i, j]];
            if prev > 0.0 && curr > 0.0 {
                result[[i - 1, j]] = (curr / prev).ln();
            }
        }
        // Log volume (normalized)
        let vol = data[[i, 4]];
        result[[i - 1, 4]] = if vol > 0.0 { vol.ln() } else { 0.0 };
    }
    result
}

// =============================================================================
// Patch Embedding
// =============================================================================

/// Configuration for patch embedding
#[derive(Debug, Clone)]
pub struct PatchConfig {
    pub patch_size: usize,     // Number of time steps per patch
    pub num_features: usize,   // Number of features (5 for OHLCV)
    pub embed_dim: usize,      // Embedding dimension
}

impl Default for PatchConfig {
    fn default() -> Self {
        Self {
            patch_size: 4,
            num_features: 5,
            embed_dim: 32,
        }
    }
}

/// Patch embedding layer: projects flattened patches to embedding space
#[derive(Debug, Clone)]
pub struct PatchEmbedding {
    pub config: PatchConfig,
    pub projection: Array2<f64>, // (embed_dim, patch_size * num_features)
    pub bias: Array1<f64>,       // (embed_dim,)
}

impl PatchEmbedding {
    pub fn new(config: PatchConfig) -> Self {
        let input_dim = config.patch_size * config.num_features;
        let mut rng = rand::thread_rng();
        let scale = (2.0 / (input_dim + config.embed_dim) as f64).sqrt();

        let projection = Array2::from_shape_fn((config.embed_dim, input_dim), |_| {
            rng.gen_range(-scale..scale)
        });
        let bias = Array1::zeros(config.embed_dim);

        Self {
            config,
            projection,
            bias,
        }
    }

    /// Split time series into non-overlapping patches
    pub fn create_patches(&self, data: &Array2<f64>) -> Vec<Array1<f64>> {
        let (n_steps, _n_feat) = data.dim();
        let n_patches = n_steps / self.config.patch_size;
        let mut patches = Vec::with_capacity(n_patches);

        for i in 0..n_patches {
            let start = i * self.config.patch_size;
            let end = start + self.config.patch_size;
            let patch_slice = data.slice(ndarray::s![start..end, ..]);
            let flat: Vec<f64> = patch_slice.iter().cloned().collect();
            patches.push(Array1::from(flat));
        }
        patches
    }

    /// Embed a single flattened patch
    pub fn embed(&self, patch: &Array1<f64>) -> Array1<f64> {
        self.projection.dot(patch) + &self.bias
    }

    /// Embed all patches
    pub fn embed_all(&self, patches: &[Array1<f64>]) -> Vec<Array1<f64>> {
        patches.iter().map(|p| self.embed(p)).collect()
    }
}

// =============================================================================
// Positional Embedding
// =============================================================================

/// Simple sinusoidal positional embeddings
pub fn positional_embeddings(num_positions: usize, embed_dim: usize) -> Array2<f64> {
    let mut pe = Array2::<f64>::zeros((num_positions, embed_dim));
    for pos in 0..num_positions {
        for i in 0..embed_dim {
            let angle = pos as f64 / (10000.0_f64).powf(2.0 * (i / 2) as f64 / embed_dim as f64);
            if i % 2 == 0 {
                pe[[pos, i]] = angle.sin();
            } else {
                pe[[pos, i]] = angle.cos();
            }
        }
    }
    pe
}

// =============================================================================
// Random Masking
// =============================================================================

/// Masking result containing indices and split patches
#[derive(Debug, Clone)]
pub struct MaskResult {
    pub visible_indices: Vec<usize>,
    pub masked_indices: Vec<usize>,
    pub visible_patches: Vec<Array1<f64>>,
    pub masked_patches: Vec<Array1<f64>>,
}

/// Random masker with configurable masking ratio
#[derive(Debug, Clone)]
pub struct RandomMasker {
    pub mask_ratio: f64,
}

impl RandomMasker {
    pub fn new(mask_ratio: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&mask_ratio),
            "Mask ratio must be between 0 and 1"
        );
        Self { mask_ratio }
    }

    /// Apply random masking to a set of patches
    pub fn mask(&self, patches: &[Array1<f64>]) -> MaskResult {
        let n = patches.len();
        let n_masked = (n as f64 * self.mask_ratio).round() as usize;
        let n_masked = n_masked.min(n);

        let mut indices: Vec<usize> = (0..n).collect();
        let mut rng = rand::thread_rng();
        indices.shuffle(&mut rng);

        let masked_indices: Vec<usize> = indices[..n_masked].to_vec();
        let visible_indices: Vec<usize> = indices[n_masked..].to_vec();

        let mut masked_set: Vec<bool> = vec![false; n];
        for &idx in &masked_indices {
            masked_set[idx] = true;
        }

        let visible_patches: Vec<Array1<f64>> = visible_indices
            .iter()
            .map(|&i| patches[i].clone())
            .collect();
        let masked_patches: Vec<Array1<f64>> = masked_indices
            .iter()
            .map(|&i| patches[i].clone())
            .collect();

        MaskResult {
            visible_indices,
            masked_indices,
            visible_patches,
            masked_patches,
        }
    }
}

// =============================================================================
// Transformer Components
// =============================================================================

/// Simple linear layer
#[derive(Debug, Clone)]
pub struct Linear {
    pub weight: Array2<f64>,
    pub bias: Array1<f64>,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (2.0 / (in_features + out_features) as f64).sqrt();
        let weight =
            Array2::from_shape_fn((out_features, in_features), |_| rng.gen_range(-scale..scale));
        let bias = Array1::zeros(out_features);
        Self { weight, bias }
    }

    pub fn forward(&self, input: &Array1<f64>) -> Array1<f64> {
        self.weight.dot(input) + &self.bias
    }
}

/// ReLU activation
pub fn relu(x: &Array1<f64>) -> Array1<f64> {
    x.mapv(|v| v.max(0.0))
}

/// Layer normalization
pub fn layer_norm(x: &Array1<f64>) -> Array1<f64> {
    let mean = x.mean().unwrap_or(0.0);
    let var = x.mapv(|v| (v - mean).powi(2)).mean().unwrap_or(1.0);
    let std = (var + 1e-8).sqrt();
    x.mapv(|v| (v - mean) / std)
}

/// Simple self-attention (single head for simplicity)
#[derive(Debug, Clone)]
pub struct SelfAttention {
    pub query: Linear,
    pub key: Linear,
    pub value: Linear,
    pub output: Linear,
    pub dim: usize,
}

impl SelfAttention {
    pub fn new(dim: usize) -> Self {
        Self {
            query: Linear::new(dim, dim),
            key: Linear::new(dim, dim),
            value: Linear::new(dim, dim),
            output: Linear::new(dim, dim),
            dim,
        }
    }

    /// Forward pass: simple self-attention over a sequence of embeddings
    pub fn forward(&self, inputs: &[Array1<f64>]) -> Vec<Array1<f64>> {
        let n = inputs.len();
        if n == 0 {
            return vec![];
        }

        // Compute Q, K, V for all tokens
        let queries: Vec<Array1<f64>> = inputs.iter().map(|x| self.query.forward(x)).collect();
        let keys: Vec<Array1<f64>> = inputs.iter().map(|x| self.key.forward(x)).collect();
        let values: Vec<Array1<f64>> = inputs.iter().map(|x| self.value.forward(x)).collect();

        let scale = (self.dim as f64).sqrt();

        // Compute attention outputs
        let mut outputs = Vec::with_capacity(n);
        for i in 0..n {
            // Compute attention scores
            let scores: Vec<f64> = keys.iter().map(|k| queries[i].dot(k) / scale).collect();

            // Softmax
            let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exp_scores: Vec<f64> = scores.iter().map(|&s| (s - max_score).exp()).collect();
            let sum_exp: f64 = exp_scores.iter().sum();
            let attn_weights: Vec<f64> = exp_scores.iter().map(|&e| e / (sum_exp + 1e-10)).collect();

            // Weighted sum of values
            let mut out = Array1::<f64>::zeros(self.dim);
            for (j, w) in attn_weights.iter().enumerate() {
                out = out + &values[j].mapv(|v| v * w);
            }

            outputs.push(self.output.forward(&out));
        }

        outputs
    }
}

/// Transformer block: self-attention + FFN with residual connections
#[derive(Debug, Clone)]
pub struct TransformerBlock {
    pub attention: SelfAttention,
    pub ffn1: Linear,
    pub ffn2: Linear,
}

impl TransformerBlock {
    pub fn new(dim: usize, ffn_dim: usize) -> Self {
        Self {
            attention: SelfAttention::new(dim),
            ffn1: Linear::new(dim, ffn_dim),
            ffn2: Linear::new(ffn_dim, dim),
        }
    }

    pub fn forward(&self, inputs: &[Array1<f64>]) -> Vec<Array1<f64>> {
        // Self-attention with residual connection
        let attn_out = self.attention.forward(inputs);
        let residual1: Vec<Array1<f64>> = inputs
            .iter()
            .zip(attn_out.iter())
            .map(|(x, a)| layer_norm(&(x + a)))
            .collect();

        // FFN with residual connection
        residual1
            .iter()
            .map(|x| {
                let h = relu(&self.ffn1.forward(x));
                let out = self.ffn2.forward(&h);
                layer_norm(&(x + &out))
            })
            .collect()
    }
}

// =============================================================================
// MAE Encoder
// =============================================================================

/// MAE Encoder: processes only visible patches
#[derive(Debug, Clone)]
pub struct MAEEncoder {
    pub layers: Vec<TransformerBlock>,
    pub embed_dim: usize,
}

impl MAEEncoder {
    pub fn new(embed_dim: usize, num_layers: usize, ffn_dim: usize) -> Self {
        let layers = (0..num_layers)
            .map(|_| TransformerBlock::new(embed_dim, ffn_dim))
            .collect();
        Self { layers, embed_dim }
    }

    /// Encode only the visible patches
    pub fn forward(&self, visible_embeddings: &[Array1<f64>]) -> Vec<Array1<f64>> {
        let mut x = visible_embeddings.to_vec();
        for layer in &self.layers {
            x = layer.forward(&x);
        }
        x
    }
}

// =============================================================================
// MAE Decoder
// =============================================================================

/// MAE Decoder: reconstructs all patches from encoded visible + mask tokens
#[derive(Debug, Clone)]
pub struct MAEDecoder {
    pub layers: Vec<TransformerBlock>,
    pub mask_token: Array1<f64>,
    pub output_projection: Linear,
    pub embed_dim: usize,
}

impl MAEDecoder {
    pub fn new(
        embed_dim: usize,
        num_layers: usize,
        ffn_dim: usize,
        output_dim: usize,
    ) -> Self {
        let layers = (0..num_layers)
            .map(|_| TransformerBlock::new(embed_dim, ffn_dim))
            .collect();

        let mut rng = rand::thread_rng();
        let scale = (1.0 / embed_dim as f64).sqrt();
        let mask_token =
            Array1::from_shape_fn(embed_dim, |_| rng.gen_range(-scale..scale));

        let output_projection = Linear::new(embed_dim, output_dim);

        Self {
            layers,
            mask_token,
            output_projection,
            embed_dim,
        }
    }

    /// Reconstruct: combine encoded visible patches with mask tokens, then decode
    pub fn forward(
        &self,
        encoded_visible: &[Array1<f64>],
        visible_indices: &[usize],
        masked_indices: &[usize],
        total_patches: usize,
        pos_embeddings: &Array2<f64>,
    ) -> Vec<Array1<f64>> {
        // Build full sequence with mask tokens at masked positions
        let mut full_sequence: Vec<Array1<f64>> = vec![Array1::zeros(self.embed_dim); total_patches];

        for (i, &idx) in visible_indices.iter().enumerate() {
            if idx < total_patches {
                let pos_emb = pos_embeddings.row(idx).to_owned();
                full_sequence[idx] = &encoded_visible[i] + &pos_emb;
            }
        }

        for &idx in masked_indices {
            if idx < total_patches {
                let pos_emb = pos_embeddings.row(idx).to_owned();
                full_sequence[idx] = &self.mask_token + &pos_emb;
            }
        }

        // Pass through decoder layers
        let mut x = full_sequence;
        for layer in &self.layers {
            x = layer.forward(&x);
        }

        // Project to output dimension
        x.iter().map(|h| self.output_projection.forward(h)).collect()
    }
}

// =============================================================================
// Full Masked Autoencoder
// =============================================================================

/// MAE configuration
#[derive(Debug, Clone)]
pub struct MAEConfig {
    pub patch_size: usize,
    pub num_features: usize,
    pub embed_dim: usize,
    pub encoder_layers: usize,
    pub decoder_layers: usize,
    pub ffn_dim: usize,
    pub mask_ratio: f64,
    pub learning_rate: f64,
}

impl Default for MAEConfig {
    fn default() -> Self {
        Self {
            patch_size: 4,
            num_features: 5,
            embed_dim: 32,
            encoder_layers: 2,
            decoder_layers: 1,
            ffn_dim: 64,
            mask_ratio: 0.75,
            learning_rate: 0.001,
        }
    }
}

/// Full Masked Autoencoder for time series
#[derive(Debug, Clone)]
pub struct MaskedAutoencoder {
    pub config: MAEConfig,
    pub patch_embedding: PatchEmbedding,
    pub encoder: MAEEncoder,
    pub decoder: MAEDecoder,
    pub masker: RandomMasker,
}

impl MaskedAutoencoder {
    pub fn new(config: MAEConfig) -> Self {
        let patch_config = PatchConfig {
            patch_size: config.patch_size,
            num_features: config.num_features,
            embed_dim: config.embed_dim,
        };
        let output_dim = config.patch_size * config.num_features;

        Self {
            patch_embedding: PatchEmbedding::new(patch_config),
            encoder: MAEEncoder::new(config.embed_dim, config.encoder_layers, config.ffn_dim),
            decoder: MAEDecoder::new(
                config.embed_dim,
                config.decoder_layers,
                config.ffn_dim,
                output_dim,
            ),
            masker: RandomMasker::new(config.mask_ratio),
            config,
        }
    }

    /// Compute MSE reconstruction loss on masked patches only
    pub fn compute_loss(
        &self,
        original_patches: &[Array1<f64>],
        reconstructed: &[Array1<f64>],
        masked_indices: &[usize],
    ) -> f64 {
        if masked_indices.is_empty() {
            return 0.0;
        }

        let mut total_loss = 0.0;
        let mut count = 0;

        for &idx in masked_indices {
            if idx < original_patches.len() && idx < reconstructed.len() {
                let diff = &original_patches[idx] - &reconstructed[idx];
                let mse = diff.mapv(|v| v * v).mean().unwrap_or(0.0);
                total_loss += mse;
                count += 1;
            }
        }

        if count > 0 {
            total_loss / count as f64
        } else {
            0.0
        }
    }

    /// Forward pass: embed, mask, encode, decode, compute loss
    pub fn forward(&self, data: &Array2<f64>) -> (f64, Vec<Array1<f64>>) {
        // Create patches
        let patches = self.patch_embedding.create_patches(data);
        if patches.is_empty() {
            return (0.0, vec![]);
        }

        let n_patches = patches.len();

        // Embed patches
        let embeddings = self.patch_embedding.embed_all(&patches);

        // Create positional embeddings
        let pos_emb = positional_embeddings(n_patches, self.config.embed_dim);

        // Add positional embeddings to patch embeddings
        let positioned: Vec<Array1<f64>> = embeddings
            .iter()
            .enumerate()
            .map(|(i, e)| e + &pos_emb.row(i).to_owned())
            .collect();

        // Apply masking
        let mask_result = self.masker.mask(&positioned);

        // Encode visible patches
        let encoded = self.encoder.forward(&mask_result.visible_patches);

        // Decode all patches
        let reconstructed = self.decoder.forward(
            &encoded,
            &mask_result.visible_indices,
            &mask_result.masked_indices,
            n_patches,
            &pos_emb,
        );

        // Compute loss on masked patches (using original flattened patches)
        let loss = self.compute_loss(&patches, &reconstructed, &mask_result.masked_indices);

        (loss, reconstructed)
    }

    /// Pretrain the MAE on data for a number of epochs
    /// Returns the loss history
    pub fn pretrain(&self, data: &Array2<f64>, epochs: usize) -> Vec<f64> {
        let mut losses = Vec::with_capacity(epochs);
        for _epoch in 0..epochs {
            let (loss, _) = self.forward(data);
            losses.push(loss);
        }
        losses
    }

    /// Compute anomaly score: average reconstruction error across all patches
    pub fn anomaly_score(&self, data: &Array2<f64>) -> f64 {
        let patches = self.patch_embedding.create_patches(data);
        if patches.is_empty() {
            return 0.0;
        }

        let n_patches = patches.len();
        let embeddings = self.patch_embedding.embed_all(&patches);
        let pos_emb = positional_embeddings(n_patches, self.config.embed_dim);

        let positioned: Vec<Array1<f64>> = embeddings
            .iter()
            .enumerate()
            .map(|(i, e)| e + &pos_emb.row(i).to_owned())
            .collect();

        let mask_result = self.masker.mask(&positioned);
        let encoded = self.encoder.forward(&mask_result.visible_patches);
        let reconstructed = self.decoder.forward(
            &encoded,
            &mask_result.visible_indices,
            &mask_result.masked_indices,
            n_patches,
            &pos_emb,
        );

        // Compute MSE over ALL patches
        let mut total_mse = 0.0;
        for i in 0..n_patches.min(reconstructed.len()) {
            let diff = &patches[i] - &reconstructed[i];
            total_mse += diff.mapv(|v| v * v).mean().unwrap_or(0.0);
        }
        total_mse / n_patches as f64
    }

    /// Extract encoder representations (for downstream tasks)
    pub fn encode(&self, data: &Array2<f64>) -> Vec<Array1<f64>> {
        let patches = self.patch_embedding.create_patches(data);
        if patches.is_empty() {
            return vec![];
        }

        let n_patches = patches.len();
        let embeddings = self.patch_embedding.embed_all(&patches);
        let pos_emb = positional_embeddings(n_patches, self.config.embed_dim);

        let positioned: Vec<Array1<f64>> = embeddings
            .iter()
            .enumerate()
            .map(|(i, e)| e + &pos_emb.row(i).to_owned())
            .collect();

        // Encode ALL patches (no masking for inference)
        self.encoder.forward(&positioned)
    }
}

// =============================================================================
// Bybit API Client
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

/// Bybit API client for fetching OHLCV data
#[derive(Debug, Clone)]
pub struct BybitClient {
    pub base_url: String,
}

impl BybitClient {
    pub fn new() -> Self {
        Self {
            base_url: "https://api.bybit.com".to_string(),
        }
    }

    /// Fetch kline (OHLCV) data from Bybit
    /// interval: "1" (1min), "5", "15", "60", "240", "D"
    /// limit: number of candles (max 1000)
    pub async fn fetch_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> Result<Vec<Candle>> {
        let url = format!(
            "{}/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
            self.base_url, symbol, interval, limit
        );

        let client = reqwest::Client::new();
        let resp: BybitResponse = client.get(&url).send().await?.json().await?;

        if resp.ret_code != 0 {
            anyhow::bail!("Bybit API error: {}", resp.ret_msg);
        }

        let mut candles: Vec<Candle> = resp
            .result
            .list
            .iter()
            .filter_map(|item| {
                if item.len() >= 6 {
                    Some(Candle {
                        timestamp: item[0].parse().unwrap_or(0),
                        open: item[1].parse().unwrap_or(0.0),
                        high: item[2].parse().unwrap_or(0.0),
                        low: item[3].parse().unwrap_or(0.0),
                        close: item[4].parse().unwrap_or(0.0),
                        volume: item[5].parse().unwrap_or(0.0),
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

    /// Fetch klines using blocking client (for non-async contexts)
    pub fn fetch_klines_blocking(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> Result<Vec<Candle>> {
        let url = format!(
            "{}/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
            self.base_url, symbol, interval, limit
        );

        let client = reqwest::blocking::Client::new();
        let resp: BybitResponse = client.get(&url).send()?.json()?;

        if resp.ret_code != 0 {
            anyhow::bail!("Bybit API error: {}", resp.ret_msg);
        }

        let mut candles: Vec<Candle> = resp
            .result
            .list
            .iter()
            .filter_map(|item| {
                if item.len() >= 6 {
                    Some(Candle {
                        timestamp: item[0].parse().unwrap_or(0),
                        open: item[1].parse().unwrap_or(0.0),
                        high: item[2].parse().unwrap_or(0.0),
                        low: item[3].parse().unwrap_or(0.0),
                        close: item[4].parse().unwrap_or(0.0),
                        volume: item[5].parse().unwrap_or(0.0),
                    })
                } else {
                    None
                }
            })
            .collect();

        candles.reverse();
        Ok(candles)
    }
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

/// Generate synthetic OHLCV data for testing
pub fn generate_synthetic_ohlcv(n: usize) -> Vec<Candle> {
    let mut rng = rand::thread_rng();
    let mut candles = Vec::with_capacity(n);
    let mut price = 50000.0_f64;

    for i in 0..n {
        let change = rng.gen_range(-0.02..0.02);
        let open = price;
        let close = price * (1.0 + change);
        let high = open.max(close) * (1.0 + rng.gen_range(0.0..0.01));
        let low = open.min(close) * (1.0 - rng.gen_range(0.0..0.01));
        let volume = rng.gen_range(100.0..10000.0) * (1.0 + change.abs() * 10.0);

        candles.push(Candle {
            timestamp: 1700000000 + (i as u64) * 60,
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
// Simple Direction Predictor (for fine-tuning demo)
// =============================================================================

/// Simple direction predictor using MAE encoder features
#[derive(Debug, Clone)]
pub struct DirectionPredictor {
    pub classifier: Linear,
}

impl DirectionPredictor {
    pub fn new(embed_dim: usize) -> Self {
        Self {
            classifier: Linear::new(embed_dim, 1),
        }
    }

    /// Predict direction from encoder output (mean pooled)
    pub fn predict(&self, encoder_outputs: &[Array1<f64>]) -> f64 {
        if encoder_outputs.is_empty() {
            return 0.0;
        }
        // Mean pool encoder outputs
        let dim = encoder_outputs[0].len();
        let mut mean = Array1::<f64>::zeros(dim);
        for out in encoder_outputs {
            mean = mean + out;
        }
        mean /= encoder_outputs.len() as f64;

        // Sigmoid output
        let logit = self.classifier.forward(&mean)[0];
        1.0 / (1.0 + (-logit).exp())
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_patch_creation() {
        let data = Array2::from_shape_fn((20, 5), |(i, j)| (i * 5 + j) as f64);
        let config = PatchConfig {
            patch_size: 4,
            num_features: 5,
            embed_dim: 32,
        };
        let pe = PatchEmbedding::new(config);
        let patches = pe.create_patches(&data);
        assert_eq!(patches.len(), 5); // 20 / 4 = 5 patches
        assert_eq!(patches[0].len(), 20); // 4 * 5 = 20 elements per patch
    }

    #[test]
    fn test_patch_embedding() {
        let config = PatchConfig {
            patch_size: 4,
            num_features: 5,
            embed_dim: 16,
        };
        let pe = PatchEmbedding::new(config);
        let patch = Array1::from(vec![1.0; 20]);
        let embedded = pe.embed(&patch);
        assert_eq!(embedded.len(), 16);
    }

    #[test]
    fn test_random_masking() {
        let patches: Vec<Array1<f64>> = (0..20).map(|i| Array1::from(vec![i as f64; 10])).collect();
        let masker = RandomMasker::new(0.75);
        let result = masker.mask(&patches);

        assert_eq!(result.visible_indices.len() + result.masked_indices.len(), 20);
        // With 75% masking of 20 patches, we expect ~15 masked
        assert_eq!(result.masked_indices.len(), 15);
        assert_eq!(result.visible_indices.len(), 5);
    }

    #[test]
    fn test_positional_embeddings() {
        let pe = positional_embeddings(10, 32);
        assert_eq!(pe.dim(), (10, 32));
        // First position should have sin(0) = 0 for even dims
        assert!((pe[[0, 0]]).abs() < 1e-6);
    }

    #[test]
    fn test_mae_forward() {
        let config = MAEConfig {
            patch_size: 4,
            num_features: 5,
            embed_dim: 16,
            encoder_layers: 1,
            decoder_layers: 1,
            ffn_dim: 32,
            mask_ratio: 0.75,
            learning_rate: 0.001,
        };
        let mae = MaskedAutoencoder::new(config);
        let data = Array2::from_shape_fn((20, 5), |(i, j)| (i + j) as f64 * 0.01);
        let (loss, reconstructed) = mae.forward(&data);
        assert!(loss >= 0.0);
        assert_eq!(reconstructed.len(), 5); // 20 / 4 = 5 patches
    }

    #[test]
    fn test_log_returns_normalization() {
        let candles = generate_synthetic_ohlcv(10);
        let matrix = candles_to_matrix(&candles);
        let returns = normalize_to_log_returns(&matrix);
        assert_eq!(returns.dim(), (9, 5)); // n-1 rows
    }

    #[test]
    fn test_anomaly_score() {
        let config = MAEConfig::default();
        let mae = MaskedAutoencoder::new(config);
        let data = Array2::from_shape_fn((20, 5), |(i, j)| (i + j) as f64 * 0.01);
        let score = mae.anomaly_score(&data);
        assert!(score >= 0.0);
    }

    #[test]
    fn test_encoder_extraction() {
        let config = MAEConfig {
            patch_size: 4,
            num_features: 5,
            embed_dim: 16,
            encoder_layers: 1,
            decoder_layers: 1,
            ffn_dim: 32,
            mask_ratio: 0.75,
            learning_rate: 0.001,
        };
        let mae = MaskedAutoencoder::new(config);
        let data = Array2::from_shape_fn((16, 5), |(i, j)| (i + j) as f64 * 0.01);
        let features = mae.encode(&data);
        assert_eq!(features.len(), 4); // 16 / 4 = 4 patches
        assert_eq!(features[0].len(), 16); // embed_dim
    }

    #[test]
    fn test_direction_predictor() {
        let predictor = DirectionPredictor::new(16);
        let features: Vec<Array1<f64>> = (0..4).map(|_| Array1::from(vec![0.5; 16])).collect();
        let prob = predictor.predict(&features);
        assert!((0.0..=1.0).contains(&prob));
    }

    #[test]
    fn test_synthetic_data_generation() {
        let candles = generate_synthetic_ohlcv(100);
        assert_eq!(candles.len(), 100);
        for c in &candles {
            assert!(c.high >= c.low);
            assert!(c.volume > 0.0);
        }
    }
}
