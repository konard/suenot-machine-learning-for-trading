use ndarray::{Array1, Array2};
use rand::Rng;
use serde::Deserialize;

// ─── ALBERT Tokenizer ─────────────────────────────────────────────

/// A simplified tokenizer for financial text.
///
/// Handles lowercasing, whitespace splitting, vocabulary lookup,
/// and insertion of special tokens ([CLS], [SEP]).
#[derive(Debug, Clone)]
pub struct ALBERTTokenizer {
    vocab: Vec<String>,
}

impl ALBERTTokenizer {
    /// Create a new tokenizer with a default financial vocabulary.
    pub fn new() -> Self {
        let vocab: Vec<String> = vec![
            "[PAD]", "[UNK]", "[CLS]", "[SEP]",
            // common financial terms
            "revenue", "earnings", "profit", "loss", "growth",
            "decline", "beat", "miss", "exceeded", "fell",
            "strong", "weak", "bullish", "bearish", "neutral",
            "market", "stock", "price", "shares", "trading",
            "crypto", "bitcoin", "btc", "eth", "blockchain",
            "increase", "decrease", "rose", "dropped", "surged",
            "plunged", "gains", "losses", "volatility", "momentum",
            "sentiment", "forecast", "guidance", "outlook", "target",
            "buy", "sell", "hold", "upgrade", "downgrade",
            "quarterly", "annual", "report", "results", "announced",
            "company", "sector", "industry", "analyst", "investors",
            "above", "below", "expectations", "consensus", "estimate",
            "margin", "eps", "ebitda", "debt", "cash",
            "the", "a", "an", "is", "are", "was", "were",
            "and", "or", "but", "in", "of", "to", "for",
            "on", "with", "by", "at", "from", "up", "down",
            "has", "had", "have", "not", "no", "its", "this",
            "that", "than", "more", "less", "high", "low",
            "new", "after", "before", "over", "under", "per",
            "year", "quarter", "month", "percent", "million", "billion",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();

        Self { vocab }
    }

    /// Tokenize text into token IDs.
    /// Prepends [CLS] and appends [SEP].
    pub fn tokenize(&self, text: &str) -> Vec<usize> {
        let cls_id = self.token_to_id("[CLS]");
        let sep_id = self.token_to_id("[SEP]");
        let unk_id = self.token_to_id("[UNK]");

        let mut ids = vec![cls_id];

        let lower = text.to_lowercase();
        for word in lower.split_whitespace() {
            let clean: String = word.chars().filter(|c| c.is_alphanumeric()).collect();
            if clean.is_empty() {
                continue;
            }
            let id = self
                .vocab
                .iter()
                .position(|v| v == &clean)
                .unwrap_or(unk_id);
            ids.push(id);
        }

        ids.push(sep_id);
        ids
    }

    fn token_to_id(&self, token: &str) -> usize {
        self.vocab.iter().position(|v| v == token).unwrap_or(1)
    }

    /// Return vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}

impl Default for ALBERTTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Factorized Embedding Layer ───────────────────────────────────

/// ALBERT-style factorized embedding: token → low-dim (E) → hidden-dim (H).
///
/// Standard BERT uses V×H parameters.  ALBERT uses V×E + E×H, where E << H.
#[derive(Debug)]
pub struct EmbeddingLayer {
    vocab_embed: Array2<f64>,  // V × E
    projection: Array2<f64>,   // E × H
    pub embed_dim: usize,      // E
    pub hidden_dim: usize,     // H
}

impl EmbeddingLayer {
    /// Create with random initialization.
    pub fn new(vocab_size: usize, embed_dim: usize, hidden_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (1.0 / embed_dim as f64).sqrt();

        let vocab_embed = Array2::from_shape_fn((vocab_size, embed_dim), |_| {
            rng.gen_range(-scale..scale)
        });
        let projection = Array2::from_shape_fn((embed_dim, hidden_dim), |_| {
            rng.gen_range(-scale..scale)
        });

        Self {
            vocab_embed,
            projection,
            embed_dim,
            hidden_dim,
        }
    }

    /// Embed a sequence of token IDs into hidden-dimensional vectors.
    /// Returns shape (seq_len, hidden_dim).
    pub fn forward(&self, token_ids: &[usize]) -> Array2<f64> {
        let seq_len = token_ids.len();
        let mut output = Array2::zeros((seq_len, self.hidden_dim));

        for (i, &tid) in token_ids.iter().enumerate() {
            let idx = tid.min(self.vocab_embed.nrows() - 1);
            let low_dim = self.vocab_embed.row(idx);
            // Project: (1, E) × (E, H) → (1, H)
            for h in 0..self.hidden_dim {
                let mut val = 0.0;
                for e in 0..self.embed_dim {
                    val += low_dim[e] * self.projection[[e, h]];
                }
                output[[i, h]] = val;
            }
        }
        output
    }

    /// Total parameter count.
    pub fn param_count(&self) -> usize {
        self.vocab_embed.len() + self.projection.len()
    }
}

// ─── Transformer Block (shared across layers) ─────────────────────

/// A single transformer block with self-attention and feed-forward layers.
///
/// In ALBERT, this block is shared across all L layers — the same weights
/// are applied recursively.
#[derive(Debug)]
pub struct TransformerBlock {
    w_q: Array2<f64>,
    w_k: Array2<f64>,
    w_v: Array2<f64>,
    w_o: Array2<f64>,
    w_ff1: Array2<f64>,
    w_ff2: Array2<f64>,
    hidden_dim: usize,
    _ff_dim: usize,
    num_heads: usize,
}

impl TransformerBlock {
    pub fn new(hidden_dim: usize, num_heads: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (1.0 / hidden_dim as f64).sqrt();
        let ff_dim = hidden_dim * 4;

        let mut rand_matrix = |rows: usize, cols: usize| -> Array2<f64> {
            Array2::from_shape_fn((rows, cols), |_| rng.gen_range(-scale..scale))
        };

        Self {
            w_q: rand_matrix(hidden_dim, hidden_dim),
            w_k: rand_matrix(hidden_dim, hidden_dim),
            w_v: rand_matrix(hidden_dim, hidden_dim),
            w_o: rand_matrix(hidden_dim, hidden_dim),
            w_ff1: rand_matrix(hidden_dim, ff_dim),
            w_ff2: rand_matrix(ff_dim, hidden_dim),
            hidden_dim,
            _ff_dim: ff_dim,
            num_heads,
        }
    }

    /// Apply self-attention + feed-forward with residual connections.
    /// Input and output shape: (seq_len, hidden_dim).
    pub fn forward(&self, input: &Array2<f64>) -> Array2<f64> {
        let seq_len = input.nrows();

        // Self-attention
        let q = matmul(input, &self.w_q);
        let k = matmul(input, &self.w_k);
        let v = matmul(input, &self.w_v);

        // Scaled dot-product attention (simplified single-head for efficiency)
        let dk = (self.hidden_dim / self.num_heads) as f64;
        let scores = matmul(&q, &k.t().to_owned());
        let mut scaled = scores / dk.sqrt();

        // Softmax over last dimension
        for i in 0..seq_len {
            let max_val = scaled.row(i).iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let mut sum = 0.0;
            for j in 0..seq_len {
                scaled[[i, j]] = (scaled[[i, j]] - max_val).exp();
                sum += scaled[[i, j]];
            }
            if sum > 0.0 {
                for j in 0..seq_len {
                    scaled[[i, j]] /= sum;
                }
            }
        }

        let attn_out = matmul(&scaled, &v);
        let projected = matmul(&attn_out, &self.w_o);

        // Residual connection + layer norm (simplified)
        let normed = layer_norm(&(input + &projected));

        // Feed-forward network
        let ff_hidden = matmul(&normed, &self.w_ff1);
        let ff_relu = ff_hidden.mapv(|x| x.max(0.0)); // ReLU (GELU approximation)
        let ff_out = matmul(&ff_relu, &self.w_ff2);

        // Residual connection + layer norm
        layer_norm(&(&normed + &ff_out))
    }

    /// Total parameter count.
    pub fn param_count(&self) -> usize {
        self.w_q.len()
            + self.w_k.len()
            + self.w_v.len()
            + self.w_o.len()
            + self.w_ff1.len()
            + self.w_ff2.len()
    }
}

// ─── ALBERT Classifier ────────────────────────────────────────────

/// ALBERT-based text classifier for financial sentiment.
///
/// Classes: 0 = bearish, 1 = neutral, 2 = bullish.
#[derive(Debug)]
pub struct ALBERTClassifier {
    tokenizer: ALBERTTokenizer,
    embedding: EmbeddingLayer,
    transformer: TransformerBlock,
    classifier_weights: Array2<f64>,  // H × num_classes
    classifier_bias: Array1<f64>,
    num_layers: usize,
    num_classes: usize,
    learning_rate: f64,
}

impl ALBERTClassifier {
    /// Create a new ALBERT classifier.
    ///
    /// - `embed_dim`: factorized embedding dimension (E), default 128
    /// - `hidden_dim`: transformer hidden dimension (H), default 256
    /// - `num_layers`: number of shared transformer applications, default 6
    /// - `num_heads`: attention heads, default 4
    /// - `num_classes`: output classes, default 3 (bearish/neutral/bullish)
    pub fn new(
        embed_dim: usize,
        hidden_dim: usize,
        num_layers: usize,
        num_heads: usize,
        num_classes: usize,
        learning_rate: f64,
    ) -> Self {
        let tokenizer = ALBERTTokenizer::new();
        let vocab_size = tokenizer.vocab_size();
        let embedding = EmbeddingLayer::new(vocab_size, embed_dim, hidden_dim);
        let transformer = TransformerBlock::new(hidden_dim, num_heads);

        let mut rng = rand::thread_rng();
        let scale = (1.0 / hidden_dim as f64).sqrt();
        let classifier_weights = Array2::from_shape_fn((hidden_dim, num_classes), |_| {
            rng.gen_range(-scale..scale)
        });
        let classifier_bias = Array1::zeros(num_classes);

        Self {
            tokenizer,
            embedding,
            transformer,
            classifier_weights,
            classifier_bias,
            num_layers,
            num_classes,
            learning_rate,
        }
    }

    /// Classify text and return class probabilities.
    pub fn predict_proba(&self, text: &str) -> Vec<f64> {
        let token_ids = self.tokenizer.tokenize(text);
        let mut hidden = self.embedding.forward(&token_ids);

        // Apply shared transformer block L times
        for _ in 0..self.num_layers {
            hidden = self.transformer.forward(&hidden);
        }

        // Use [CLS] token representation (first token)
        let cls_repr: Vec<f64> = hidden.row(0).to_vec();
        let cls = Array1::from_vec(cls_repr);

        // Classification head: softmax(W * cls + b)
        let logits = cls.dot(&self.classifier_weights) + &self.classifier_bias;
        softmax(&logits)
    }

    /// Predict the most likely class and its confidence.
    pub fn predict(&self, text: &str) -> (usize, f64) {
        let probs = self.predict_proba(text);
        let (idx, &max_prob) = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();
        (idx, max_prob)
    }

    /// Return the label name for a class index.
    pub fn label_name(class: usize) -> &'static str {
        match class {
            0 => "BEARISH",
            1 => "NEUTRAL",
            2 => "BULLISH",
            _ => "UNKNOWN",
        }
    }

    /// Train on labeled data: Vec<(text, class_index)>.
    pub fn train(&mut self, data: &[(String, usize)], epochs: usize) {
        for _ in 0..epochs {
            for (text, label) in data {
                let probs = self.predict_proba(text);

                // Compute gradient of cross-entropy loss w.r.t. classifier weights
                let token_ids = self.tokenizer.tokenize(text);
                let mut hidden = self.embedding.forward(&token_ids);
                for _ in 0..self.num_layers {
                    hidden = self.transformer.forward(&hidden);
                }
                let cls_repr: Vec<f64> = hidden.row(0).to_vec();

                // Gradient: d_loss/d_logit = prob - one_hot
                let mut grad = Array1::from_vec(probs.clone());
                grad[*label] -= 1.0;

                // Update classifier weights and bias
                for h in 0..self.classifier_weights.nrows() {
                    for c in 0..self.num_classes {
                        self.classifier_weights[[h, c]] -=
                            self.learning_rate * grad[c] * cls_repr[h];
                    }
                }
                for c in 0..self.num_classes {
                    self.classifier_bias[c] -= self.learning_rate * grad[c];
                }
            }
        }
    }

    /// Evaluate accuracy on test data.
    pub fn accuracy(&self, data: &[(String, usize)]) -> f64 {
        let correct = data
            .iter()
            .filter(|(text, label)| {
                let (pred, _) = self.predict(text);
                pred == *label
            })
            .count();
        correct as f64 / data.len() as f64
    }

    /// Total parameter count for the model.
    pub fn param_count(&self) -> usize {
        self.embedding.param_count()
            + self.transformer.param_count()
            + self.classifier_weights.len()
            + self.classifier_bias.len()
    }

    /// Return the tokenizer reference.
    pub fn tokenizer(&self) -> &ALBERTTokenizer {
        &self.tokenizer
    }
}

// ─── Sentiment Aggregator ─────────────────────────────────────────

/// Aggregates individual sentiment predictions into trading signals.
///
/// Maintains a rolling window of sentiment scores and produces
/// exponentially-weighted signals for position sizing.
#[derive(Debug)]
pub struct SentimentAggregator {
    alpha: f64,
    signal: f64,
    history: Vec<f64>,
    bullish_threshold: f64,
    bearish_threshold: f64,
}

impl SentimentAggregator {
    /// Create a new aggregator.
    ///
    /// - `alpha`: smoothing factor (0..1), higher = more reactive
    /// - `bullish_threshold`: signal above this triggers long
    /// - `bearish_threshold`: signal below this triggers short
    pub fn new(alpha: f64, bullish_threshold: f64, bearish_threshold: f64) -> Self {
        Self {
            alpha,
            signal: 0.0,
            history: Vec::new(),
            bullish_threshold,
            bearish_threshold,
        }
    }

    /// Add a sentiment score (-1.0 bearish to +1.0 bullish).
    pub fn add_score(&mut self, score: f64) {
        self.signal = self.alpha * score + (1.0 - self.alpha) * self.signal;
        self.history.push(self.signal);
    }

    /// Current smoothed signal value.
    pub fn signal(&self) -> f64 {
        self.signal
    }

    /// Generate a trading position signal: +1 (long), -1 (short), 0 (flat).
    pub fn position_signal(&self) -> i32 {
        if self.signal > self.bullish_threshold {
            1
        } else if self.signal < self.bearish_threshold {
            -1
        } else {
            0
        }
    }

    /// Signal history.
    pub fn history(&self) -> &[f64] {
        &self.history
    }

    /// Convert class probabilities to a sentiment score in [-1, 1].
    /// Expects [bearish_prob, neutral_prob, bullish_prob].
    pub fn probs_to_score(probs: &[f64]) -> f64 {
        if probs.len() < 3 {
            return 0.0;
        }
        // score = bullish_prob - bearish_prob, in [-1, 1]
        probs[2] - probs[0]
    }
}

// ─── Bybit Client ──────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct BybitResponse<T> {
    #[serde(rename = "retCode")]
    pub ret_code: i32,
    #[serde(rename = "retMsg")]
    pub ret_msg: String,
    pub result: T,
}

#[derive(Debug, Deserialize)]
pub struct KlineResult {
    pub list: Vec<Vec<String>>,
}

/// A parsed kline bar.
#[derive(Debug, Clone)]
pub struct Kline {
    pub timestamp: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// Async client for Bybit V5 API.
pub struct BybitClient {
    base_url: String,
    client: reqwest::Client,
}

impl BybitClient {
    pub fn new() -> Self {
        Self {
            base_url: "https://api.bybit.com".to_string(),
            client: reqwest::Client::new(),
        }
    }

    /// Fetch kline (candlestick) data.
    pub async fn get_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: u32,
    ) -> anyhow::Result<Vec<Kline>> {
        let url = format!(
            "{}/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
            self.base_url, symbol, interval, limit
        );
        let resp: BybitResponse<KlineResult> = self.client.get(&url).send().await?.json().await?;

        let mut klines = Vec::new();
        for item in &resp.result.list {
            if item.len() >= 6 {
                klines.push(Kline {
                    timestamp: item[0].parse().unwrap_or(0),
                    open: item[1].parse().unwrap_or(0.0),
                    high: item[2].parse().unwrap_or(0.0),
                    low: item[3].parse().unwrap_or(0.0),
                    close: item[4].parse().unwrap_or(0.0),
                    volume: item[5].parse().unwrap_or(0.0),
                });
            }
        }
        klines.reverse(); // Bybit returns newest first
        Ok(klines)
    }
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Synthetic Data Generation ─────────────────────────────────────

/// Sample financial headlines for training and testing.
pub fn sample_financial_headlines() -> Vec<(String, usize)> {
    vec![
        // Bullish (class 2)
        ("Revenue beat expectations with strong growth".into(), 2),
        ("Earnings exceeded consensus estimate by a wide margin".into(), 2),
        ("Company announced bullish guidance for the quarter".into(), 2),
        ("Stock surged on strong quarterly results".into(), 2),
        ("Bitcoin price rose above new high on momentum".into(), 2),
        ("Crypto market gains on bullish sentiment".into(), 2),
        ("Shares rose after earnings beat forecast".into(), 2),
        ("Strong revenue growth above market estimate".into(), 2),
        ("Profit exceeded expectations with gains".into(), 2),
        ("BTC surged to new high with strong momentum".into(), 2),
        ("Company reported strong growth in revenue and profit".into(), 2),
        ("Analyst upgrade on strong results and guidance".into(), 2),
        ("Market sentiment bullish after strong report".into(), 2),
        ("ETH gains above target on bullish outlook".into(), 2),
        ("Trading volume surged on bullish momentum".into(), 2),
        // Bearish (class 0)
        ("Revenue fell below expectations with decline".into(), 0),
        ("Company reported loss and weak guidance".into(), 0),
        ("Stock dropped on earnings miss and bearish outlook".into(), 0),
        ("Bitcoin price plunged on bearish sentiment".into(), 0),
        ("Crypto market losses on weak trading volume".into(), 0),
        ("Shares declined after downgrade from analyst".into(), 0),
        ("Profit fell below consensus with decline".into(), 0),
        ("Company missed earnings estimate with losses".into(), 0),
        ("BTC dropped below target on weak sentiment".into(), 0),
        ("Revenue decline and weak forecast".into(), 0),
        ("Market losses on bearish outlook and volatility".into(), 0),
        ("Stock plunged after weak quarterly results".into(), 0),
        ("Earnings miss and bearish guidance from company".into(), 0),
        ("ETH dropped on weak momentum and losses".into(), 0),
        ("Trading volume decline on bearish sentiment".into(), 0),
        // Neutral (class 1)
        ("Company announced quarterly results in report".into(), 1),
        ("Market trading volume was at forecast".into(), 1),
        ("Stock price held at consensus estimate".into(), 1),
        ("Bitcoin trading was neutral for the quarter".into(), 1),
        ("Industry sector report announced by analyst".into(), 1),
        ("Company results were in line with estimate".into(), 1),
        ("Shares trading at target by analyst consensus".into(), 1),
        ("Crypto market held by momentum and trading".into(), 1),
        ("Company announced new report for the quarter".into(), 1),
        ("BTC and ETH trading at market price".into(), 1),
        ("Annual report results announced by company".into(), 1),
        ("Revenue and earnings in line with forecast".into(), 1),
        ("Quarterly results from sector company report".into(), 1),
        ("Market outlook for the year announced".into(), 1),
        ("Company held annual industry report".into(), 1),
    ]
}

/// Generate synthetic labeled data with controlled sentiment patterns.
pub fn generate_training_data(n: usize) -> Vec<(String, usize)> {
    let mut rng = rand::thread_rng();
    let base = sample_financial_headlines();
    let mut data = Vec::with_capacity(n);

    for _ in 0..n {
        let idx = rng.gen_range(0..base.len());
        data.push(base[idx].clone());
    }
    data
}

/// Compute simple returns from kline data.
pub fn compute_returns(klines: &[Kline]) -> Vec<f64> {
    klines
        .windows(2)
        .map(|w| (w[1].close - w[0].close) / w[0].close)
        .collect()
}

// ─── Helper Functions ──────────────────────────────────────────────

/// Matrix multiply: (m, k) × (k, n) → (m, n).
fn matmul(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let m = a.nrows();
    let k = a.ncols();
    let n = b.ncols();
    assert_eq!(k, b.nrows());

    let mut c = Array2::zeros((m, n));
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..k {
                sum += a[[i, p]] * b[[p, j]];
            }
            c[[i, j]] = sum;
        }
    }
    c
}

/// Layer normalization across the last dimension.
fn layer_norm(x: &Array2<f64>) -> Array2<f64> {
    let mut out = x.clone();
    let eps = 1e-12;

    for i in 0..x.nrows() {
        let row = x.row(i);
        let mean = row.mean().unwrap_or(0.0);
        let var: f64 = row.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / row.len() as f64;
        let std = (var + eps).sqrt();
        for j in 0..x.ncols() {
            out[[i, j]] = (x[[i, j]] - mean) / std;
        }
    }
    out
}

/// Softmax over a 1-D array.
fn softmax(logits: &Array1<f64>) -> Vec<f64> {
    let max_val = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|x| (x - max_val).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

// ─── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer_basic() {
        let tok = ALBERTTokenizer::new();
        let ids = tok.tokenize("Revenue beat expectations");
        // Should start with [CLS]=2 and end with [SEP]=3
        assert_eq!(ids[0], 2);
        assert_eq!(*ids.last().unwrap(), 3);
        assert!(ids.len() >= 5); // CLS + 3 words + SEP
    }

    #[test]
    fn test_tokenizer_unknown_word() {
        let tok = ALBERTTokenizer::new();
        let ids = tok.tokenize("xyznotaword");
        // Unknown word maps to [UNK]=1
        assert_eq!(ids[1], 1);
    }

    #[test]
    fn test_tokenizer_case_insensitive() {
        let tok = ALBERTTokenizer::new();
        let ids1 = tok.tokenize("Revenue");
        let ids2 = tok.tokenize("revenue");
        assert_eq!(ids1, ids2);
    }

    #[test]
    fn test_embedding_shape() {
        let emb = EmbeddingLayer::new(100, 32, 64);
        let token_ids = vec![0, 5, 10, 3];
        let out = emb.forward(&token_ids);
        assert_eq!(out.nrows(), 4);
        assert_eq!(out.ncols(), 64);
    }

    #[test]
    fn test_embedding_param_reduction() {
        // V=100, E=32, H=256: factorized = 100*32 + 32*256 = 11,392
        // Non-factorized = 100*256 = 25,600
        let emb = EmbeddingLayer::new(100, 32, 256);
        let factorized = emb.param_count();
        let non_factorized = 100 * 256;
        assert!(factorized < non_factorized);
    }

    #[test]
    fn test_transformer_output_shape() {
        let block = TransformerBlock::new(64, 4);
        let input = Array2::from_shape_fn((5, 64), |_| rand::thread_rng().gen_range(-1.0..1.0));
        let output = block.forward(&input);
        assert_eq!(output.nrows(), 5);
        assert_eq!(output.ncols(), 64);
    }

    #[test]
    fn test_classifier_predict() {
        let clf = ALBERTClassifier::new(32, 64, 2, 4, 3, 0.01);
        let probs = clf.predict_proba("Revenue beat expectations");
        assert_eq!(probs.len(), 3);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "probabilities must sum to 1, got {}", sum);
    }

    #[test]
    fn test_classifier_predict_class() {
        let clf = ALBERTClassifier::new(32, 64, 2, 4, 3, 0.01);
        let (class, confidence) = clf.predict("Stock surged on strong earnings");
        assert!(class < 3);
        assert!(confidence > 0.0 && confidence <= 1.0);
    }

    #[test]
    fn test_classifier_train() {
        let data = sample_financial_headlines();
        let (train, test) = data.split_at(35);

        let mut clf = ALBERTClassifier::new(32, 64, 2, 4, 3, 0.05);
        let acc_before = clf.accuracy(test);

        clf.train(&train.to_vec(), 10);
        let acc_after = clf.accuracy(test);

        // After training, accuracy should generally not be zero
        assert!(acc_after > 0.0, "accuracy after training: {}", acc_after);
        // Model should be at least as good as before (or not catastrophically worse)
        assert!(
            acc_after >= acc_before * 0.5,
            "before: {}, after: {}",
            acc_before,
            acc_after
        );
    }

    #[test]
    fn test_sentiment_aggregator() {
        let mut agg = SentimentAggregator::new(0.3, 0.2, -0.2);
        agg.add_score(0.8);
        agg.add_score(0.6);
        agg.add_score(0.7);
        assert!(agg.signal() > 0.0);
        assert_eq!(agg.position_signal(), 1); // should be long
    }

    #[test]
    fn test_sentiment_aggregator_bearish() {
        let mut agg = SentimentAggregator::new(0.5, 0.2, -0.2);
        agg.add_score(-0.8);
        agg.add_score(-0.9);
        assert!(agg.signal() < 0.0);
        assert_eq!(agg.position_signal(), -1); // should be short
    }

    #[test]
    fn test_sentiment_aggregator_neutral() {
        let mut agg = SentimentAggregator::new(0.3, 0.5, -0.5);
        agg.add_score(0.1);
        agg.add_score(-0.1);
        assert_eq!(agg.position_signal(), 0); // should be flat
    }

    #[test]
    fn test_probs_to_score() {
        // All bearish
        assert!((SentimentAggregator::probs_to_score(&[1.0, 0.0, 0.0]) - (-1.0)).abs() < 1e-9);
        // All bullish
        assert!((SentimentAggregator::probs_to_score(&[0.0, 0.0, 1.0]) - 1.0).abs() < 1e-9);
        // Balanced
        assert!(
            (SentimentAggregator::probs_to_score(&[0.33, 0.34, 0.33]) - 0.0).abs() < 0.01
        );
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let logits = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let probs = softmax(&logits);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);
        // Probabilities should be monotonically increasing
        for i in 0..probs.len() - 1 {
            assert!(probs[i] < probs[i + 1]);
        }
    }

    #[test]
    fn test_layer_norm() {
        let x = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let normed = layer_norm(&x);
        // After layer norm, each row should have approximately zero mean
        for i in 0..normed.nrows() {
            let mean = normed.row(i).mean().unwrap();
            assert!(mean.abs() < 1e-6, "row {} mean = {}", i, mean);
        }
    }

    #[test]
    fn test_sample_headlines() {
        let data = sample_financial_headlines();
        assert_eq!(data.len(), 45); // 15 bullish + 15 bearish + 15 neutral
        for (_, label) in &data {
            assert!(*label <= 2);
        }
    }

    #[test]
    fn test_compute_returns() {
        let klines = vec![
            Kline { timestamp: 0, open: 100.0, high: 105.0, low: 99.0, close: 100.0, volume: 10.0 },
            Kline { timestamp: 1, open: 100.0, high: 110.0, low: 99.0, close: 110.0, volume: 15.0 },
            Kline { timestamp: 2, open: 110.0, high: 115.0, low: 105.0, close: 105.0, volume: 12.0 },
        ];
        let returns = compute_returns(&klines);
        assert_eq!(returns.len(), 2);
        assert!((returns[0] - 0.1).abs() < 1e-9); // 10% gain
        assert!((returns[1] - (-5.0 / 110.0)).abs() < 1e-9); // ~-4.5% loss
    }

    #[test]
    fn test_label_names() {
        assert_eq!(ALBERTClassifier::label_name(0), "BEARISH");
        assert_eq!(ALBERTClassifier::label_name(1), "NEUTRAL");
        assert_eq!(ALBERTClassifier::label_name(2), "BULLISH");
    }
}
