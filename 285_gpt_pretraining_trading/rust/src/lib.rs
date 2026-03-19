use ndarray::{Array1, Array2};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

// =============================================================================
// Price Tokenizer
// =============================================================================

/// Tokenizer that discretizes continuous returns into a fixed vocabulary.
/// Uses percentile-based bin boundaries for balanced token frequencies.
#[derive(Debug, Clone)]
pub struct PriceTokenizer {
    /// Number of bins in the vocabulary (excluding special tokens)
    pub vocab_size: usize,
    /// Bin boundaries (sorted, length = vocab_size - 1)
    pub boundaries: Vec<f64>,
    /// Bin centers for decoding tokens back to approximate returns
    pub centers: Vec<f64>,
}

/// Special token IDs (offset after vocab_size)
pub const BOS_TOKEN: usize = 0;
pub const EOS_TOKEN: usize = 1;
pub const SEP_TOKEN: usize = 2;
pub const SPECIAL_TOKEN_COUNT: usize = 3;

impl PriceTokenizer {
    /// Create a new tokenizer by fitting bin boundaries from training returns.
    pub fn fit(returns: &[f64], vocab_size: usize) -> Self {
        assert!(vocab_size >= 4, "Vocabulary size must be at least 4");
        assert!(!returns.is_empty(), "Returns must not be empty");

        let mut sorted = returns.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = sorted.len();
        let num_boundaries = vocab_size - 1;
        let mut boundaries = Vec::with_capacity(num_boundaries);

        for i in 1..=num_boundaries {
            let idx = (i as f64 / vocab_size as f64 * n as f64) as usize;
            let idx = idx.min(n - 1);
            boundaries.push(sorted[idx]);
        }

        // Compute bin centers
        let mut centers = Vec::with_capacity(vocab_size);
        // First bin: from -inf to boundaries[0]
        centers.push(if n > 0 {
            (sorted[0] + boundaries[0]) / 2.0
        } else {
            boundaries[0] - 0.01
        });
        // Middle bins
        for i in 0..num_boundaries.saturating_sub(1) {
            centers.push((boundaries[i] + boundaries[i + 1]) / 2.0);
        }
        // Last bin: from boundaries[-1] to +inf
        centers.push(if n > 0 {
            (boundaries[num_boundaries - 1] + sorted[n - 1]) / 2.0
        } else {
            boundaries[num_boundaries - 1] + 0.01
        });

        PriceTokenizer {
            vocab_size,
            boundaries,
            centers,
        }
    }

    /// Encode a single return value into a token index.
    /// Returns token in range [SPECIAL_TOKEN_COUNT, SPECIAL_TOKEN_COUNT + vocab_size).
    pub fn encode(&self, ret: f64) -> usize {
        let mut bin = 0;
        for &b in &self.boundaries {
            if ret > b {
                bin += 1;
            } else {
                break;
            }
        }
        bin + SPECIAL_TOKEN_COUNT
    }

    /// Encode a sequence of returns into token indices.
    pub fn encode_sequence(&self, returns: &[f64]) -> Vec<usize> {
        let mut tokens = Vec::with_capacity(returns.len() + 2);
        tokens.push(BOS_TOKEN);
        for &r in returns {
            tokens.push(self.encode(r));
        }
        tokens.push(EOS_TOKEN);
        tokens
    }

    /// Decode a token index back to an approximate return value.
    pub fn decode(&self, token: usize) -> Option<f64> {
        if token < SPECIAL_TOKEN_COUNT {
            return None; // Special token
        }
        let bin = token - SPECIAL_TOKEN_COUNT;
        self.centers.get(bin).copied()
    }

    /// Total vocabulary size including special tokens.
    pub fn total_vocab_size(&self) -> usize {
        self.vocab_size + SPECIAL_TOKEN_COUNT
    }
}

/// Compute log-returns from a price series.
pub fn compute_log_returns(prices: &[f64]) -> Vec<f64> {
    if prices.len() < 2 {
        return vec![];
    }
    prices
        .windows(2)
        .map(|w| (w[1] / w[0]).ln())
        .collect()
}

// =============================================================================
// Positional Encoding
// =============================================================================

/// Generate sinusoidal positional encodings.
pub fn sinusoidal_positional_encoding(max_len: usize, d_model: usize) -> Array2<f64> {
    let mut pe = Array2::zeros((max_len, d_model));
    for pos in 0..max_len {
        for i in 0..d_model / 2 {
            let angle = pos as f64 / (10000.0_f64).powf(2.0 * i as f64 / d_model as f64);
            pe[[pos, 2 * i]] = angle.sin();
            if 2 * i + 1 < d_model {
                pe[[pos, 2 * i + 1]] = angle.cos();
            }
        }
    }
    pe
}

// =============================================================================
// GPT Decoder
// =============================================================================

/// Configuration for the GPT decoder.
#[derive(Debug, Clone)]
pub struct GptConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub max_seq_len: usize,
    pub learning_rate: f64,
}

impl Default for GptConfig {
    fn default() -> Self {
        GptConfig {
            vocab_size: 64 + SPECIAL_TOKEN_COUNT,
            d_model: 32,
            n_heads: 4,
            n_layers: 2,
            max_seq_len: 128,
            learning_rate: 0.001,
        }
    }
}

/// A simplified GPT decoder with causal self-attention.
#[derive(Debug)]
pub struct GptDecoder {
    pub config: GptConfig,
    /// Token embedding matrix: (vocab_size, d_model)
    pub token_embeddings: Array2<f64>,
    /// Positional encoding: (max_seq_len, d_model)
    pub positional_encoding: Array2<f64>,
    /// Attention weights per layer: (n_layers, d_model, 3*d_model) for Q,K,V
    pub attention_weights: Vec<Array2<f64>>,
    /// Attention output projection per layer: (n_layers, d_model, d_model)
    pub attention_output: Vec<Array2<f64>>,
    /// FFN weights per layer: (n_layers) - W1: (d_model, 4*d_model), W2: (4*d_model, d_model)
    pub ffn_w1: Vec<Array2<f64>>,
    pub ffn_w2: Vec<Array2<f64>>,
    /// Output projection: (d_model, vocab_size)
    pub output_projection: Array2<f64>,
}

impl GptDecoder {
    /// Create a new GPT decoder with randomly initialized weights.
    pub fn new(config: GptConfig) -> Self {
        let mut rng = rand::thread_rng();
        let d = config.d_model;
        let v = config.vocab_size;
        let scale = (2.0 / d as f64).sqrt();

        let token_embeddings = random_matrix(&mut rng, v, d, scale);
        let positional_encoding = sinusoidal_positional_encoding(config.max_seq_len, d);

        let mut attention_weights = Vec::with_capacity(config.n_layers);
        let mut attention_output = Vec::with_capacity(config.n_layers);
        let mut ffn_w1 = Vec::with_capacity(config.n_layers);
        let mut ffn_w2 = Vec::with_capacity(config.n_layers);

        for _ in 0..config.n_layers {
            attention_weights.push(random_matrix(&mut rng, d, 3 * d, scale));
            attention_output.push(random_matrix(&mut rng, d, d, scale));
            ffn_w1.push(random_matrix(&mut rng, d, 4 * d, scale));
            ffn_w2.push(random_matrix(&mut rng, 4 * d, d, scale));
        }

        let output_projection = random_matrix(&mut rng, d, v, scale);

        GptDecoder {
            config,
            token_embeddings,
            positional_encoding,
            attention_weights,
            attention_output,
            ffn_w1,
            ffn_w2,
            output_projection,
        }
    }

    /// Forward pass: compute logits for each position in the sequence.
    /// Input: token indices of shape (seq_len,)
    /// Output: logits of shape (seq_len, vocab_size)
    pub fn forward(&self, tokens: &[usize]) -> Array2<f64> {
        let seq_len = tokens.len();
        let d = self.config.d_model;

        // Token embeddings + positional encoding
        let mut hidden = Array2::zeros((seq_len, d));
        for (t, &tok) in tokens.iter().enumerate() {
            let tok_idx = tok.min(self.config.vocab_size - 1);
            for j in 0..d {
                hidden[[t, j]] =
                    self.token_embeddings[[tok_idx, j]] + self.positional_encoding[[t, j]];
            }
        }

        // Apply transformer layers
        for layer in 0..self.config.n_layers {
            hidden = self.transformer_layer(&hidden, layer);
        }

        // Output projection to logits
        matmul(&hidden, &self.output_projection)
    }

    /// Single transformer decoder layer with causal self-attention.
    fn transformer_layer(&self, input: &Array2<f64>, layer: usize) -> Array2<f64> {
        let seq_len = input.nrows();
        let d = self.config.d_model;
        let n_heads = self.config.n_heads;
        let head_dim = d / n_heads;

        // Compute Q, K, V
        let qkv = matmul(input, &self.attention_weights[layer]);
        let q = qkv.slice(ndarray::s![.., ..d]).to_owned();
        let k = qkv.slice(ndarray::s![.., d..2 * d]).to_owned();
        let v = qkv.slice(ndarray::s![.., 2 * d..3 * d]).to_owned();

        // Multi-head causal attention
        let mut attn_out = Array2::zeros((seq_len, d));
        let scale = (head_dim as f64).sqrt();

        for h in 0..n_heads {
            let h_start = h * head_dim;
            let h_end = (h + 1) * head_dim;

            // Extract head slices
            let q_h = q.slice(ndarray::s![.., h_start..h_end]);
            let k_h = k.slice(ndarray::s![.., h_start..h_end]);
            let v_h = v.slice(ndarray::s![.., h_start..h_end]);

            // Compute attention scores with causal mask
            for i in 0..seq_len {
                let mut scores = Vec::with_capacity(i + 1);
                let mut max_score = f64::NEG_INFINITY;

                for j in 0..=i {
                    let mut dot = 0.0;
                    for dd in 0..head_dim {
                        dot += q_h[[i, dd]] * k_h[[j, dd]];
                    }
                    let s = dot / scale;
                    if s > max_score {
                        max_score = s;
                    }
                    scores.push(s);
                }

                // Softmax
                let mut sum_exp = 0.0;
                let mut weights = Vec::with_capacity(scores.len());
                for &s in &scores {
                    let e = (s - max_score).exp();
                    weights.push(e);
                    sum_exp += e;
                }
                for w in weights.iter_mut() {
                    *w /= sum_exp + 1e-10;
                }

                // Weighted sum of values
                for dd in 0..head_dim {
                    let mut val = 0.0;
                    for (j, &w) in weights.iter().enumerate() {
                        val += w * v_h[[j, dd]];
                    }
                    attn_out[[i, h_start + dd]] = val;
                }
            }
        }

        // Output projection
        let projected = matmul(&attn_out, &self.attention_output[layer]);

        // Residual connection + layer norm
        let normed = layer_norm(&add_matrices(input, &projected));

        // Feed-forward network
        let ff_hidden = matmul(&normed, &self.ffn_w1[layer]);
        let ff_activated = gelu_matrix(&ff_hidden);
        let ff_out = matmul(&ff_activated, &self.ffn_w2[layer]);

        // Residual connection + layer norm
        layer_norm(&add_matrices(&normed, &ff_out))
    }

    /// Compute cross-entropy loss for next-token prediction.
    /// Tokens: full sequence including BOS.
    /// Returns average loss over all valid prediction positions.
    pub fn compute_loss(&self, tokens: &[usize]) -> f64 {
        if tokens.len() < 2 {
            return 0.0;
        }

        let logits = self.forward(tokens);
        let seq_len = tokens.len();
        let mut total_loss = 0.0;

        for t in 0..seq_len - 1 {
            let target = tokens[t + 1];
            let logit_row = logits.row(t);

            // Log-softmax
            let max_logit = logit_row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let log_sum_exp: f64 = logit_row.iter().map(|&x| (x - max_logit).exp()).sum::<f64>().ln() + max_logit;
            let target_idx = target.min(self.config.vocab_size - 1);
            let log_prob = logit_row[target_idx] - log_sum_exp;
            total_loss -= log_prob;
        }

        total_loss / (seq_len - 1) as f64
    }

    /// Train the model on a batch of token sequences using simple SGD.
    pub fn train_step(&mut self, sequences: &[Vec<usize>], learning_rate: f64) -> f64 {
        let mut total_loss = 0.0;
        let epsilon = 1e-5;

        for seq in sequences {
            if seq.len() < 2 {
                continue;
            }
            let base_loss = self.compute_loss(seq);
            total_loss += base_loss;

            // Numerical gradient descent on output projection (simplified training)
            let (rows, cols) = (self.output_projection.nrows(), self.output_projection.ncols());
            let sample_rate = 4; // Sample subset of parameters for efficiency
            for i in (0..rows).step_by(sample_rate) {
                for j in (0..cols).step_by(sample_rate) {
                    self.output_projection[[i, j]] += epsilon;
                    let loss_plus = self.compute_loss(seq);
                    self.output_projection[[i, j]] -= epsilon;

                    let grad = (loss_plus - base_loss) / epsilon;
                    self.output_projection[[i, j]] -= learning_rate * grad;
                }
            }
        }

        total_loss / sequences.len().max(1) as f64
    }

    /// Autoregressively generate the next `n_tokens` given a context.
    pub fn generate(&self, context: &[usize], n_tokens: usize, temperature: f64) -> Vec<usize> {
        let mut rng = rand::thread_rng();
        let mut sequence = context.to_vec();

        for _ in 0..n_tokens {
            let max_ctx = self.config.max_seq_len.min(sequence.len());
            let ctx_start = sequence.len().saturating_sub(max_ctx);
            let ctx = &sequence[ctx_start..];

            let logits = self.forward(ctx);
            let last_logits = logits.row(logits.nrows() - 1);

            // Temperature scaling
            let temp = if temperature < 1e-10 { 1e-10 } else { temperature };
            let scaled: Vec<f64> = last_logits.iter().map(|&x| x / temp).collect();

            // Softmax sampling
            let max_val = scaled.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exp_vals: Vec<f64> = scaled.iter().map(|&x| (x - max_val).exp()).collect();
            let sum: f64 = exp_vals.iter().sum();
            let probs: Vec<f64> = exp_vals.iter().map(|&x| x / sum).collect();

            // Sample from distribution
            let r: f64 = rng.gen();
            let mut cumsum = 0.0;
            let mut chosen = probs.len() - 1;
            for (idx, &p) in probs.iter().enumerate() {
                cumsum += p;
                if r <= cumsum {
                    chosen = idx;
                    break;
                }
            }

            sequence.push(chosen);
        }

        sequence[context.len()..].to_vec()
    }
}

// =============================================================================
// Fine-Tuning Head
// =============================================================================

/// Trading signal head that maps GPT hidden states to trading signals.
#[derive(Debug)]
pub struct TradingHead {
    /// Weights: (d_model, 1) for signal generation
    pub weights: Array1<f64>,
    pub bias: f64,
}

impl TradingHead {
    pub fn new(d_model: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (2.0 / d_model as f64).sqrt();
        let weights = Array1::from_vec(
            (0..d_model)
                .map(|_| rng.gen_range(-scale..scale))
                .collect(),
        );
        TradingHead {
            weights,
            bias: 0.0,
        }
    }

    /// Generate trading signal in [-1, 1] from a hidden state vector.
    pub fn predict_signal(&self, hidden_state: &Array1<f64>) -> f64 {
        let dot: f64 = self
            .weights
            .iter()
            .zip(hidden_state.iter())
            .map(|(w, h)| w * h)
            .sum();
        (dot + self.bias).tanh()
    }

    /// Predict signals for a sequence of hidden states.
    pub fn predict_signals(&self, hidden_states: &Array2<f64>) -> Vec<f64> {
        (0..hidden_states.nrows())
            .map(|i| {
                let row = hidden_states.row(i).to_owned();
                self.predict_signal(&row)
            })
            .collect()
    }
}

// =============================================================================
// Bybit API Integration
// =============================================================================

#[derive(Debug, Deserialize, Serialize)]
pub struct BybitKlineResponse {
    #[serde(rename = "retCode")]
    pub ret_code: i32,
    #[serde(rename = "retMsg")]
    pub ret_msg: String,
    pub result: BybitKlineResult,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct BybitKlineResult {
    pub symbol: String,
    pub category: String,
    pub list: Vec<Vec<String>>,
}

/// OHLCV candle data.
#[derive(Debug, Clone)]
pub struct Candle {
    pub timestamp: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// Fetch kline data from Bybit API.
pub fn fetch_bybit_klines(
    symbol: &str,
    interval: &str,
    limit: usize,
) -> anyhow::Result<Vec<Candle>> {
    let url = format!(
        "https://api.bybit.com/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
        symbol, interval, limit
    );

    let response: BybitKlineResponse = reqwest::blocking::get(&url)?.json()?;

    if response.ret_code != 0 {
        anyhow::bail!("Bybit API error: {}", response.ret_msg);
    }

    let mut candles: Vec<Candle> = response
        .result
        .list
        .iter()
        .filter_map(|row| {
            if row.len() >= 6 {
                Some(Candle {
                    timestamp: row[0].parse().ok()?,
                    open: row[1].parse().ok()?,
                    high: row[2].parse().ok()?,
                    low: row[3].parse().ok()?,
                    close: row[4].parse().ok()?,
                    volume: row[5].parse().ok()?,
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

/// Extract close prices from candles.
pub fn extract_close_prices(candles: &[Candle]) -> Vec<f64> {
    candles.iter().map(|c| c.close).collect()
}

// =============================================================================
// Utility Functions
// =============================================================================

fn random_matrix(rng: &mut impl Rng, rows: usize, cols: usize, scale: f64) -> Array2<f64> {
    Array2::from_shape_fn((rows, cols), |_| rng.gen_range(-scale..scale))
}

fn matmul(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let (m, k1) = (a.nrows(), a.ncols());
    let (k2, n) = (b.nrows(), b.ncols());
    assert_eq!(k1, k2, "Matrix dimension mismatch: {} vs {}", k1, k2);
    let mut result = Array2::zeros((m, n));
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..k1 {
                sum += a[[i, k]] * b[[k, j]];
            }
            result[[i, j]] = sum;
        }
    }
    result
}

fn add_matrices(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    a + b
}

fn layer_norm(x: &Array2<f64>) -> Array2<f64> {
    let mut result = x.clone();
    let eps = 1e-5;
    for i in 0..x.nrows() {
        let row = x.row(i);
        let mean = row.mean().unwrap_or(0.0);
        let var = row.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / row.len() as f64;
        let std = (var + eps).sqrt();
        for j in 0..x.ncols() {
            result[[i, j]] = (x[[i, j]] - mean) / std;
        }
    }
    result
}

fn gelu(x: f64) -> f64 {
    0.5 * x * (1.0 + ((2.0 / PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
}

fn gelu_matrix(x: &Array2<f64>) -> Array2<f64> {
    x.mapv(gelu)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_price_tokenizer_fit() {
        let returns: Vec<f64> = (-50..=50).map(|i| i as f64 * 0.001).collect();
        let tokenizer = PriceTokenizer::fit(&returns, 16);
        assert_eq!(tokenizer.vocab_size, 16);
        assert_eq!(tokenizer.boundaries.len(), 15);
        assert_eq!(tokenizer.centers.len(), 16);
    }

    #[test]
    fn test_price_tokenizer_encode_decode() {
        let returns: Vec<f64> = (-100..=100).map(|i| i as f64 * 0.001).collect();
        let tokenizer = PriceTokenizer::fit(&returns, 32);

        // Encode and decode should give approximate values
        let test_val = 0.005;
        let token = tokenizer.encode(test_val);
        assert!(token >= SPECIAL_TOKEN_COUNT);
        assert!(token < tokenizer.total_vocab_size());

        let decoded = tokenizer.decode(token).unwrap();
        assert!((decoded - test_val).abs() < 0.02);
    }

    #[test]
    fn test_encode_sequence() {
        let returns: Vec<f64> = (-50..=50).map(|i| i as f64 * 0.001).collect();
        let tokenizer = PriceTokenizer::fit(&returns, 16);

        let seq = tokenizer.encode_sequence(&[0.01, -0.02, 0.005]);
        assert_eq!(seq[0], BOS_TOKEN);
        assert_eq!(*seq.last().unwrap(), EOS_TOKEN);
        assert_eq!(seq.len(), 5); // BOS + 3 returns + EOS
    }

    #[test]
    fn test_compute_log_returns() {
        let prices = vec![100.0, 101.0, 99.0, 102.0];
        let returns = compute_log_returns(&prices);
        assert_eq!(returns.len(), 3);
        assert!((returns[0] - (101.0_f64 / 100.0).ln()).abs() < 1e-10);
        assert!((returns[1] - (99.0_f64 / 101.0).ln()).abs() < 1e-10);
    }

    #[test]
    fn test_gpt_forward() {
        let config = GptConfig {
            vocab_size: 16 + SPECIAL_TOKEN_COUNT,
            d_model: 16,
            n_heads: 2,
            n_layers: 1,
            max_seq_len: 32,
            learning_rate: 0.001,
        };
        let model = GptDecoder::new(config);
        let tokens = vec![BOS_TOKEN, 5, 8, 3, 10];
        let logits = model.forward(&tokens);
        assert_eq!(logits.nrows(), 5);
        assert_eq!(logits.ncols(), 16 + SPECIAL_TOKEN_COUNT);
    }

    #[test]
    fn test_gpt_loss() {
        let config = GptConfig {
            vocab_size: 16 + SPECIAL_TOKEN_COUNT,
            d_model: 16,
            n_heads: 2,
            n_layers: 1,
            max_seq_len: 32,
            learning_rate: 0.001,
        };
        let model = GptDecoder::new(config);
        let tokens = vec![BOS_TOKEN, 5, 8, 3, 10, EOS_TOKEN];
        let loss = model.compute_loss(&tokens);
        assert!(loss > 0.0, "Loss should be positive");
        // Random init should have high loss (close to -ln(1/vocab_size))
        assert!(loss < 100.0, "Loss should be bounded");
    }

    #[test]
    fn test_gpt_generate() {
        let config = GptConfig {
            vocab_size: 16 + SPECIAL_TOKEN_COUNT,
            d_model: 16,
            n_heads: 2,
            n_layers: 1,
            max_seq_len: 32,
            learning_rate: 0.001,
        };
        let model = GptDecoder::new(config);
        let context = vec![BOS_TOKEN, 5, 8];
        let generated = model.generate(&context, 3, 1.0);
        assert_eq!(generated.len(), 3);
    }

    #[test]
    fn test_trading_head() {
        let d_model = 16;
        let head = TradingHead::new(d_model);
        let hidden = Array1::from_vec(vec![0.1; d_model]);
        let signal = head.predict_signal(&hidden);
        assert!(signal >= -1.0 && signal <= 1.0, "Signal must be in [-1, 1]");
    }

    #[test]
    fn test_sinusoidal_positional_encoding() {
        let pe = sinusoidal_positional_encoding(10, 8);
        assert_eq!(pe.nrows(), 10);
        assert_eq!(pe.ncols(), 8);
        // Position 0 should have sin(0)=0 at even indices
        assert!((pe[[0, 0]]).abs() < 1e-10);
    }

    #[test]
    fn test_layer_norm() {
        let x = Array2::from_shape_vec((2, 4), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .unwrap();
        let normed = layer_norm(&x);
        // Each row should have mean ~0 and std ~1
        for i in 0..normed.nrows() {
            let row = normed.row(i);
            let mean: f64 = row.iter().sum::<f64>() / row.len() as f64;
            assert!(mean.abs() < 1e-5, "Mean should be ~0, got {}", mean);
        }
    }
}
