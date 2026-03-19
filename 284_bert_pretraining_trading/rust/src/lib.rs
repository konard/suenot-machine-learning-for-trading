use ndarray::{Array1, Array2};
use rand::Rng;
use serde::{Deserialize, Serialize};

// ============================================================
// Special token IDs
// ============================================================
pub const PAD_TOKEN: usize = 0;
pub const MASK_TOKEN: usize = 1;
pub const CLS_TOKEN: usize = 2;
pub const SEP_TOKEN: usize = 3;
pub const SPECIAL_TOKENS: usize = 4;

// ============================================================
// Configuration
// ============================================================

#[derive(Debug, Clone)]
pub struct BertConfig {
    pub vocab_size: usize,
    pub hidden_dim: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub intermediate_dim: usize,
    pub max_seq_len: usize,
    pub mask_ratio: f64,
}

impl Default for BertConfig {
    fn default() -> Self {
        Self {
            vocab_size: 64,
            hidden_dim: 32,
            num_heads: 4,
            num_layers: 2,
            intermediate_dim: 64,
            max_seq_len: 128,
            mask_ratio: 0.15,
        }
    }
}

// ============================================================
// Price Tokenizer
// ============================================================

#[derive(Debug, Clone)]
pub struct PriceTokenizer {
    pub num_bins: usize,
    pub mask_ratio: f64,
    pub bin_edges: Vec<f64>,
}

impl PriceTokenizer {
    /// Create a new price tokenizer with the given number of bins and mask ratio.
    pub fn new(num_bins: usize, mask_ratio: f64) -> Self {
        Self {
            num_bins,
            mask_ratio,
            bin_edges: Vec::new(),
        }
    }

    /// Fit bin edges from historical price data (compute returns, then quantiles).
    pub fn fit(&mut self, prices: &[f64]) {
        let returns = self.compute_returns(prices);
        self.bin_edges = self.compute_quantile_edges(&returns, self.num_bins);
    }

    /// Compute log-returns from a price series.
    pub fn compute_returns(&self, prices: &[f64]) -> Vec<f64> {
        if prices.len() < 2 {
            return Vec::new();
        }
        prices
            .windows(2)
            .map(|w| {
                if w[0] > 0.0 {
                    (w[1] / w[0]).ln()
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// Compute quantile-based bin edges.
    fn compute_quantile_edges(&self, data: &[f64], num_bins: usize) -> Vec<f64> {
        if data.is_empty() || num_bins == 0 {
            return Vec::new();
        }
        let mut sorted = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mut edges = Vec::with_capacity(num_bins + 1);
        edges.push(f64::NEG_INFINITY);
        for i in 1..num_bins {
            let idx = (i as f64 / num_bins as f64 * sorted.len() as f64) as usize;
            let idx = idx.min(sorted.len() - 1);
            edges.push(sorted[idx]);
        }
        edges.push(f64::INFINITY);
        edges
    }

    /// Tokenize a sequence of returns into bin indices (offset by SPECIAL_TOKENS).
    pub fn tokenize(&self, returns: &[f64]) -> Vec<usize> {
        returns
            .iter()
            .map(|&r| {
                let bin = self
                    .bin_edges
                    .windows(2)
                    .position(|w| r >= w[0] && r < w[1])
                    .unwrap_or(self.num_bins - 1);
                bin + SPECIAL_TOKENS
            })
            .collect()
    }

    /// Apply MLM masking: returns (masked_tokens, mask_positions, original_tokens_at_mask).
    pub fn mask_tokens(
        &self,
        tokens: &[usize],
    ) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
        let mut rng = rand::thread_rng();
        let mut masked = tokens.to_vec();
        let mut mask_positions = Vec::new();
        let mut original_tokens = Vec::new();

        for i in 0..tokens.len() {
            if rng.gen::<f64>() < self.mask_ratio {
                mask_positions.push(i);
                original_tokens.push(tokens[i]);

                let r: f64 = rng.gen();
                if r < 0.8 {
                    masked[i] = MASK_TOKEN;
                } else if r < 0.9 {
                    // Replace with random token
                    masked[i] = rng.gen_range(SPECIAL_TOKENS..SPECIAL_TOKENS + self.num_bins);
                }
                // else keep original (10%)
            }
        }

        (masked, mask_positions, original_tokens)
    }

    /// Build a full input sequence with [CLS] prefix and [SEP] suffix.
    pub fn build_input_sequence(&self, tokens: &[usize]) -> Vec<usize> {
        let mut seq = Vec::with_capacity(tokens.len() + 2);
        seq.push(CLS_TOKEN);
        seq.extend_from_slice(tokens);
        seq.push(SEP_TOKEN);
        seq
    }
}

// ============================================================
// Utility math functions
// ============================================================

fn softmax(x: &Array1<f64>) -> Array1<f64> {
    let max_val = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_x = x.mapv(|v| (v - max_val).exp());
    let sum = exp_x.sum();
    if sum > 0.0 {
        exp_x / sum
    } else {
        Array1::ones(x.len()) / x.len() as f64
    }
}

fn softmax_2d(x: &Array2<f64>) -> Array2<f64> {
    let mut result = x.clone();
    for mut row in result.rows_mut() {
        let max_val = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        row.mapv_inplace(|v| (v - max_val).exp());
        let sum = row.sum();
        if sum > 0.0 {
            row /= sum;
        }
    }
    result
}

fn layer_norm(x: &Array1<f64>) -> Array1<f64> {
    let mean = x.mean().unwrap_or(0.0);
    let var = x.mapv(|v| (v - mean).powi(2)).mean().unwrap_or(1.0);
    let std = (var + 1e-8).sqrt();
    x.mapv(|v| (v - mean) / std)
}

fn layer_norm_2d(x: &Array2<f64>) -> Array2<f64> {
    let mut result = Array2::zeros(x.raw_dim());
    for (i, row) in x.rows().into_iter().enumerate() {
        let normed = layer_norm(&row.to_owned());
        result.row_mut(i).assign(&normed);
    }
    result
}

fn random_matrix(rows: usize, cols: usize, scale: f64) -> Array2<f64> {
    let mut rng = rand::thread_rng();
    Array2::from_shape_fn((rows, cols), |_| rng.gen::<f64>() * scale - scale / 2.0)
}

fn _random_vector(size: usize, scale: f64) -> Array1<f64> {
    let mut rng = rand::thread_rng();
    Array1::from_shape_fn(size, |_| rng.gen::<f64>() * scale - scale / 2.0)
}

// ============================================================
// Attention Head
// ============================================================

#[derive(Debug, Clone)]
pub struct AttentionHead {
    pub w_q: Array2<f64>,
    pub w_k: Array2<f64>,
    pub w_v: Array2<f64>,
    pub head_dim: usize,
}

impl AttentionHead {
    pub fn new(hidden_dim: usize, head_dim: usize) -> Self {
        let scale = (2.0 / (hidden_dim + head_dim) as f64).sqrt();
        Self {
            w_q: random_matrix(hidden_dim, head_dim, scale),
            w_k: random_matrix(hidden_dim, head_dim, scale),
            w_v: random_matrix(hidden_dim, head_dim, scale),
            head_dim,
        }
    }

    /// Compute scaled dot-product attention: softmax(QK^T / sqrt(d_k)) V
    pub fn forward(&self, x: &Array2<f64>) -> Array2<f64> {
        let q = x.dot(&self.w_q);
        let k = x.dot(&self.w_k);
        let v = x.dot(&self.w_v);

        let scale = (self.head_dim as f64).sqrt();
        let scores = q.dot(&k.t()) / scale;
        let attn_weights = softmax_2d(&scores);
        attn_weights.dot(&v)
    }
}

// ============================================================
// Multi-Head Attention
// ============================================================

#[derive(Debug, Clone)]
pub struct MultiHeadAttention {
    pub heads: Vec<AttentionHead>,
    pub w_o: Array2<f64>,
}

impl MultiHeadAttention {
    pub fn new(hidden_dim: usize, num_heads: usize) -> Self {
        let head_dim = hidden_dim / num_heads;
        let heads: Vec<AttentionHead> = (0..num_heads)
            .map(|_| AttentionHead::new(hidden_dim, head_dim))
            .collect();
        let total_head_dim = head_dim * num_heads;
        let scale = (2.0 / (total_head_dim + hidden_dim) as f64).sqrt();
        Self {
            heads,
            w_o: random_matrix(total_head_dim, hidden_dim, scale),
        }
    }

    pub fn forward(&self, x: &Array2<f64>) -> Array2<f64> {
        let head_outputs: Vec<Array2<f64>> =
            self.heads.iter().map(|head| head.forward(x)).collect();

        // Concatenate along feature dimension
        let seq_len = x.nrows();
        let total_dim: usize = head_outputs.iter().map(|h| h.ncols()).sum();
        let mut concat = Array2::zeros((seq_len, total_dim));
        let mut col_offset = 0;
        for h_out in &head_outputs {
            let cols = h_out.ncols();
            concat
                .slice_mut(ndarray::s![.., col_offset..col_offset + cols])
                .assign(h_out);
            col_offset += cols;
        }

        concat.dot(&self.w_o)
    }
}

// ============================================================
// Feed-Forward Network
// ============================================================

#[derive(Debug, Clone)]
pub struct FeedForward {
    pub w1: Array2<f64>,
    pub b1: Array1<f64>,
    pub w2: Array2<f64>,
    pub b2: Array1<f64>,
}

impl FeedForward {
    pub fn new(hidden_dim: usize, intermediate_dim: usize) -> Self {
        let scale1 = (2.0 / (hidden_dim + intermediate_dim) as f64).sqrt();
        let scale2 = (2.0 / (intermediate_dim + hidden_dim) as f64).sqrt();
        Self {
            w1: random_matrix(hidden_dim, intermediate_dim, scale1),
            b1: Array1::zeros(intermediate_dim),
            w2: random_matrix(intermediate_dim, hidden_dim, scale2),
            b2: Array1::zeros(hidden_dim),
        }
    }

    /// GELU-approximated forward pass.
    pub fn forward(&self, x: &Array2<f64>) -> Array2<f64> {
        let h = x.dot(&self.w1) + &self.b1;
        // GELU approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        let activated = h.mapv(|v| {
            let inner = (2.0_f64 / std::f64::consts::PI).sqrt() * (v + 0.044715 * v.powi(3));
            v * 0.5 * (1.0 + inner.tanh())
        });
        activated.dot(&self.w2) + &self.b2
    }
}

// ============================================================
// Transformer Layer
// ============================================================

#[derive(Debug, Clone)]
pub struct TransformerLayer {
    pub attention: MultiHeadAttention,
    pub ffn: FeedForward,
}

impl TransformerLayer {
    pub fn new(hidden_dim: usize, num_heads: usize, intermediate_dim: usize) -> Self {
        Self {
            attention: MultiHeadAttention::new(hidden_dim, num_heads),
            ffn: FeedForward::new(hidden_dim, intermediate_dim),
        }
    }

    pub fn forward(&self, x: &Array2<f64>) -> Array2<f64> {
        // Self-attention with residual + layer norm
        let attn_out = self.attention.forward(x);
        let normed1 = layer_norm_2d(&(x + &attn_out));

        // FFN with residual + layer norm
        let ffn_out = self.ffn.forward(&normed1);
        layer_norm_2d(&(&normed1 + &ffn_out))
    }
}

// ============================================================
// BERT Encoder
// ============================================================

#[derive(Debug, Clone)]
pub struct BertEncoder {
    pub config: BertConfig,
    pub token_embeddings: Array2<f64>,
    pub position_embeddings: Array2<f64>,
    pub layers: Vec<TransformerLayer>,
}

impl BertEncoder {
    pub fn new(config: BertConfig) -> Self {
        let scale = (2.0 / config.hidden_dim as f64).sqrt();
        let token_embeddings =
            random_matrix(config.vocab_size, config.hidden_dim, scale);
        let position_embeddings =
            random_matrix(config.max_seq_len, config.hidden_dim, scale);

        let layers: Vec<TransformerLayer> = (0..config.num_layers)
            .map(|_| {
                TransformerLayer::new(
                    config.hidden_dim,
                    config.num_heads,
                    config.intermediate_dim,
                )
            })
            .collect();

        Self {
            config,
            token_embeddings,
            position_embeddings,
            layers,
        }
    }

    /// Embed token IDs into continuous vectors, adding positional embeddings.
    pub fn embed(&self, token_ids: &[usize]) -> Array2<f64> {
        let seq_len = token_ids.len();
        let hidden_dim = self.config.hidden_dim;
        let mut embeddings = Array2::zeros((seq_len, hidden_dim));

        for (i, &tid) in token_ids.iter().enumerate() {
            let tid_clamped = tid.min(self.config.vocab_size - 1);
            let pos_clamped = i.min(self.config.max_seq_len - 1);
            let tok_emb = self.token_embeddings.row(tid_clamped);
            let pos_emb = self.position_embeddings.row(pos_clamped);
            for j in 0..hidden_dim {
                embeddings[[i, j]] = tok_emb[j] + pos_emb[j];
            }
        }

        embeddings
    }

    /// Forward pass: embed tokens, then pass through all transformer layers.
    pub fn forward(&self, token_ids: &[usize]) -> Array2<f64> {
        let mut hidden = self.embed(token_ids);
        for layer in &self.layers {
            hidden = layer.forward(&hidden);
        }
        hidden
    }

    /// Get the [CLS] token representation (first token).
    pub fn cls_representation(&self, token_ids: &[usize]) -> Array1<f64> {
        let hidden = self.forward(token_ids);
        hidden.row(0).to_owned()
    }
}

// ============================================================
// MLM Head
// ============================================================

#[derive(Debug, Clone)]
pub struct MLMHead {
    pub projection: Array2<f64>,
    pub bias: Array1<f64>,
}

impl MLMHead {
    pub fn new(hidden_dim: usize, vocab_size: usize) -> Self {
        let scale = (2.0 / (hidden_dim + vocab_size) as f64).sqrt();
        Self {
            projection: random_matrix(hidden_dim, vocab_size, scale),
            bias: Array1::zeros(vocab_size),
        }
    }

    /// Predict token probabilities at masked positions.
    pub fn forward(
        &self,
        hidden_states: &Array2<f64>,
        mask_positions: &[usize],
    ) -> Vec<Array1<f64>> {
        mask_positions
            .iter()
            .map(|&pos| {
                let h = hidden_states.row(pos).to_owned();
                let logits = h.dot(&self.projection) + &self.bias;
                softmax(&logits)
            })
            .collect()
    }

    /// Compute MLM cross-entropy loss.
    pub fn compute_loss(
        &self,
        hidden_states: &Array2<f64>,
        mask_positions: &[usize],
        target_tokens: &[usize],
    ) -> f64 {
        let predictions = self.forward(hidden_states, mask_positions);
        if predictions.is_empty() {
            return 0.0;
        }
        let mut total_loss = 0.0;
        for (pred, &target) in predictions.iter().zip(target_tokens.iter()) {
            let target_clamped = target.min(pred.len() - 1);
            let prob = pred[target_clamped].max(1e-10);
            total_loss -= prob.ln();
        }
        total_loss / predictions.len() as f64
    }
}

// ============================================================
// Classification Head (for fine-tuning: buy/sell/hold)
// ============================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TradingSignal {
    Buy,
    Sell,
    Hold,
}

impl std::fmt::Display for TradingSignal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TradingSignal::Buy => write!(f, "BUY"),
            TradingSignal::Sell => write!(f, "SELL"),
            TradingSignal::Hold => write!(f, "HOLD"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ClassificationHead {
    pub w1: Array2<f64>,
    pub b1: Array1<f64>,
    pub w2: Array2<f64>,
    pub b2: Array1<f64>,
    pub num_classes: usize,
}

impl ClassificationHead {
    pub fn new(hidden_dim: usize, num_classes: usize) -> Self {
        let intermediate = hidden_dim / 2;
        let scale1 = (2.0 / (hidden_dim + intermediate) as f64).sqrt();
        let scale2 = (2.0 / (intermediate + num_classes) as f64).sqrt();
        Self {
            w1: random_matrix(hidden_dim, intermediate, scale1),
            b1: Array1::zeros(intermediate),
            w2: random_matrix(intermediate, num_classes, scale2),
            b2: Array1::zeros(num_classes),
            num_classes,
        }
    }

    /// Forward pass: hidden -> probabilities over classes.
    pub fn forward(&self, cls_repr: &Array1<f64>) -> Array1<f64> {
        let h = cls_repr.dot(&self.w1) + &self.b1;
        let activated = h.mapv(|v| v.max(0.0)); // ReLU
        let logits = activated.dot(&self.w2) + &self.b2;
        softmax(&logits)
    }

    /// Predict a trading signal from class probabilities.
    pub fn predict(&self, cls_repr: &Array1<f64>) -> TradingSignal {
        let probs = self.forward(cls_repr);
        let max_idx = probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(2);
        match max_idx {
            0 => TradingSignal::Buy,
            1 => TradingSignal::Sell,
            _ => TradingSignal::Hold,
        }
    }

    /// Compute cross-entropy loss for classification.
    pub fn compute_loss(&self, cls_repr: &Array1<f64>, target_class: usize) -> f64 {
        let probs = self.forward(cls_repr);
        let target = target_class.min(self.num_classes - 1);
        let prob = probs[target].max(1e-10);
        -prob.ln()
    }
}

// ============================================================
// Bybit API Integration
// ============================================================

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

#[derive(Debug, Clone)]
pub struct KlineData {
    pub timestamp: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// Fetch kline data from Bybit API (blocking).
pub fn fetch_bybit_klines(
    symbol: &str,
    interval: &str,
    limit: usize,
) -> anyhow::Result<Vec<KlineData>> {
    let url = format!(
        "https://api.bybit.com/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
        symbol, interval, limit
    );

    let client = reqwest::blocking::Client::new();
    let resp: BybitKlineResponse = client.get(&url).send()?.json()?;

    if resp.ret_code != 0 {
        anyhow::bail!("Bybit API error: {}", resp.ret_msg);
    }

    let mut klines: Vec<KlineData> = resp
        .result
        .list
        .iter()
        .filter_map(|row| {
            if row.len() >= 6 {
                Some(KlineData {
                    timestamp: row[0].parse().unwrap_or(0),
                    open: row[1].parse().unwrap_or(0.0),
                    high: row[2].parse().unwrap_or(0.0),
                    low: row[3].parse().unwrap_or(0.0),
                    close: row[4].parse().unwrap_or(0.0),
                    volume: row[5].parse().unwrap_or(0.0),
                })
            } else {
                None
            }
        })
        .collect();

    // Bybit returns newest first; reverse to chronological order.
    klines.reverse();
    Ok(klines)
}

/// Extract close prices from kline data.
pub fn extract_close_prices(klines: &[KlineData]) -> Vec<f64> {
    klines.iter().map(|k| k.close).collect()
}

// ============================================================
// BERT Trading Model (convenience wrapper)
// ============================================================

#[derive(Debug)]
pub struct BertTradingModel {
    pub encoder: BertEncoder,
    pub mlm_head: MLMHead,
    pub classification_head: ClassificationHead,
    pub tokenizer: PriceTokenizer,
}

impl BertTradingModel {
    pub fn new(config: BertConfig) -> Self {
        let vocab_size = config.vocab_size;
        let hidden_dim = config.hidden_dim;
        let mask_ratio = config.mask_ratio;
        let num_bins = vocab_size - SPECIAL_TOKENS;

        Self {
            encoder: BertEncoder::new(config),
            mlm_head: MLMHead::new(hidden_dim, vocab_size),
            classification_head: ClassificationHead::new(hidden_dim, 3),
            tokenizer: PriceTokenizer::new(num_bins, mask_ratio),
        }
    }

    /// Fit the tokenizer on historical prices.
    pub fn fit_tokenizer(&mut self, prices: &[f64]) {
        self.tokenizer.fit(prices);
    }

    /// Run MLM pretraining step on a price sequence; returns loss.
    pub fn pretrain_step(&self, prices: &[f64]) -> f64 {
        let returns = self.tokenizer.compute_returns(prices);
        if returns.is_empty() {
            return 0.0;
        }
        let tokens = self.tokenizer.tokenize(&returns);
        let (masked, mask_pos, originals) = self.tokenizer.mask_tokens(&tokens);
        if mask_pos.is_empty() {
            return 0.0;
        }
        let input = self.tokenizer.build_input_sequence(&masked);
        let hidden = self.encoder.forward(&input);

        // Adjust mask positions by +1 for the prepended [CLS] token
        let adjusted_positions: Vec<usize> = mask_pos.iter().map(|&p| p + 1).collect();
        self.mlm_head
            .compute_loss(&hidden, &adjusted_positions, &originals)
    }

    /// Predict trading signal from a price sequence.
    pub fn predict_signal(&self, prices: &[f64]) -> TradingSignal {
        let returns = self.tokenizer.compute_returns(prices);
        if returns.is_empty() {
            return TradingSignal::Hold;
        }
        let tokens = self.tokenizer.tokenize(&returns);
        let input = self.tokenizer.build_input_sequence(&tokens);
        let cls_repr = self.encoder.cls_representation(&input);
        self.classification_head.predict(&cls_repr)
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_price_tokenizer_fit_and_tokenize() {
        let prices = vec![100.0, 101.0, 99.5, 102.0, 98.0, 103.0, 100.5, 101.5];
        let mut tokenizer = PriceTokenizer::new(8, 0.15);
        tokenizer.fit(&prices);

        assert_eq!(tokenizer.bin_edges.len(), 9); // 8 bins => 9 edges
        let returns = tokenizer.compute_returns(&prices);
        assert_eq!(returns.len(), prices.len() - 1);

        let tokens = tokenizer.tokenize(&returns);
        assert_eq!(tokens.len(), returns.len());
        for &t in &tokens {
            assert!(t >= SPECIAL_TOKENS);
            assert!(t < SPECIAL_TOKENS + 8);
        }
    }

    #[test]
    fn test_masking_produces_valid_output() {
        let mut tokenizer = PriceTokenizer::new(16, 0.5);
        let prices: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64) * 0.1).collect();
        tokenizer.fit(&prices);
        let returns = tokenizer.compute_returns(&prices);
        let tokens = tokenizer.tokenize(&returns);

        let (masked, mask_pos, originals) = tokenizer.mask_tokens(&tokens);
        assert_eq!(masked.len(), tokens.len());
        assert_eq!(mask_pos.len(), originals.len());
        // With 50% mask ratio and 99 tokens, we expect some masked tokens
        assert!(!mask_pos.is_empty());
    }

    #[test]
    fn test_bert_encoder_forward_shape() {
        let config = BertConfig {
            vocab_size: 32,
            hidden_dim: 16,
            num_heads: 2,
            num_layers: 1,
            intermediate_dim: 32,
            max_seq_len: 64,
            mask_ratio: 0.15,
        };
        let encoder = BertEncoder::new(config);
        let input = vec![CLS_TOKEN, 5, 10, 15, SEP_TOKEN];
        let output = encoder.forward(&input);
        assert_eq!(output.nrows(), 5);
        assert_eq!(output.ncols(), 16);
    }

    #[test]
    fn test_mlm_head_loss() {
        let hidden_dim = 16;
        let vocab_size = 32;
        let mlm_head = MLMHead::new(hidden_dim, vocab_size);

        let hidden = Array2::from_shape_fn((5, hidden_dim), |(i, j)| {
            ((i * hidden_dim + j) as f64) * 0.01
        });
        let mask_pos = vec![1, 3];
        let targets = vec![5, 10];
        let loss = mlm_head.compute_loss(&hidden, &mask_pos, &targets);
        assert!(loss > 0.0, "Loss should be positive");
        assert!(loss.is_finite(), "Loss should be finite");
    }

    #[test]
    fn test_classification_head_predict() {
        let hidden_dim = 16;
        let head = ClassificationHead::new(hidden_dim, 3);
        let cls_repr = Array1::from_shape_fn(hidden_dim, |i| i as f64 * 0.1);

        let probs = head.forward(&cls_repr);
        assert_eq!(probs.len(), 3);

        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "Probabilities must sum to 1");

        let signal = head.predict(&cls_repr);
        assert!(
            matches!(signal, TradingSignal::Buy | TradingSignal::Sell | TradingSignal::Hold),
            "Signal must be a valid TradingSignal"
        );
    }

    #[test]
    fn test_bert_trading_model_pretrain_step() {
        let config = BertConfig::default();
        let mut model = BertTradingModel::new(config);
        let prices: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64).sin() * 5.0).collect();
        model.fit_tokenizer(&prices);

        let loss = model.pretrain_step(&prices);
        assert!(loss >= 0.0, "Pretrain loss should be non-negative");
        assert!(loss.is_finite(), "Pretrain loss should be finite");
    }

    #[test]
    fn test_bert_trading_model_predict() {
        let config = BertConfig::default();
        let mut model = BertTradingModel::new(config);
        let prices: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64).sin() * 5.0).collect();
        model.fit_tokenizer(&prices);

        let signal = model.predict_signal(&prices);
        assert!(
            matches!(signal, TradingSignal::Buy | TradingSignal::Sell | TradingSignal::Hold),
            "Must produce a valid trading signal"
        );
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let x = Array1::from(vec![1.0, 2.0, 3.0, 4.0]);
        let result = softmax(&x);
        let sum: f64 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(result.iter().all(|&v| v > 0.0));
    }

    #[test]
    fn test_build_input_sequence() {
        let tokenizer = PriceTokenizer::new(8, 0.15);
        let tokens = vec![4, 5, 6, 7];
        let seq = tokenizer.build_input_sequence(&tokens);
        assert_eq!(seq[0], CLS_TOKEN);
        assert_eq!(seq[seq.len() - 1], SEP_TOKEN);
        assert_eq!(seq.len(), tokens.len() + 2);
    }
}
