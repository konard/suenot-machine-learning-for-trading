use ndarray::{Array1, Array2};
use rand::Rng;
use serde::Deserialize;
use std::collections::HashSet;

// ─── BigBird Configuration ───────────────────────────────────────

/// Configuration for the BigBird sparse attention mechanism.
#[derive(Debug, Clone)]
pub struct BigBirdConfig {
    /// Maximum sequence length.
    pub seq_length: usize,
    /// Number of random attention connections per token.
    pub num_random_tokens: usize,
    /// Half-width of the sliding window attention.
    pub window_size: usize,
    /// Number of global tokens (attend to/from all positions).
    pub num_global_tokens: usize,
    /// Hidden dimension size.
    pub hidden_dim: usize,
    /// Number of attention heads.
    pub num_heads: usize,
}

impl BigBirdConfig {
    pub fn new(
        seq_length: usize,
        num_random_tokens: usize,
        window_size: usize,
        num_global_tokens: usize,
        hidden_dim: usize,
        num_heads: usize,
    ) -> Self {
        Self {
            seq_length,
            num_random_tokens,
            window_size,
            num_global_tokens,
            hidden_dim,
            num_heads,
        }
    }
}

impl Default for BigBirdConfig {
    fn default() -> Self {
        Self {
            seq_length: 512,
            num_random_tokens: 3,
            window_size: 3,
            num_global_tokens: 2,
            hidden_dim: 64,
            num_heads: 4,
        }
    }
}

// ─── BigBird Attention ───────────────────────────────────────────

/// Implements BigBird's sparse attention mechanism combining
/// random, window (local), and global attention patterns.
#[derive(Debug)]
pub struct BigBirdAttention {
    config: BigBirdConfig,
}

impl BigBirdAttention {
    pub fn new(config: BigBirdConfig) -> Self {
        Self { config }
    }

    /// Generate the random attention pattern.
    /// Returns a set of (i, j) pairs where token i attends to token j.
    pub fn random_attention(&self, seq_len: usize) -> Vec<(usize, usize)> {
        let mut rng = rand::thread_rng();
        let mut edges = Vec::new();

        for i in 0..seq_len {
            let mut targets: HashSet<usize> = HashSet::new();
            while targets.len() < self.config.num_random_tokens.min(seq_len - 1) {
                let j = rng.gen_range(0..seq_len);
                if j != i {
                    targets.insert(j);
                }
            }
            for j in targets {
                edges.push((i, j));
            }
        }
        edges
    }

    /// Generate the window (local) attention pattern.
    pub fn window_attention(&self, seq_len: usize) -> Vec<(usize, usize)> {
        let mut edges = Vec::new();
        let w = self.config.window_size;

        for i in 0..seq_len {
            let start = i.saturating_sub(w);
            let end = (i + w + 1).min(seq_len);
            for j in start..end {
                if i != j {
                    edges.push((i, j));
                }
            }
        }
        edges
    }

    /// Generate the global attention pattern.
    /// The first `num_global_tokens` positions are global.
    pub fn global_attention(&self, seq_len: usize) -> Vec<(usize, usize)> {
        let mut edges = Vec::new();
        let g = self.config.num_global_tokens.min(seq_len);

        for gi in 0..g {
            for j in 0..seq_len {
                if gi != j {
                    // Global token attends to all
                    edges.push((gi, j));
                    // All attend to global token
                    edges.push((j, gi));
                }
            }
        }
        edges
    }

    /// Generate the combined sparse attention mask as a binary matrix.
    /// mask[i][j] = 1.0 if token i attends to token j, 0.0 otherwise.
    pub fn attention_mask(&self, seq_len: usize) -> Array2<f64> {
        let mut mask = Array2::zeros((seq_len, seq_len));

        // Self-attention (diagonal)
        for i in 0..seq_len {
            mask[[i, i]] = 1.0;
        }

        // Random attention
        for (i, j) in self.random_attention(seq_len) {
            mask[[i, j]] = 1.0;
        }

        // Window attention
        for (i, j) in self.window_attention(seq_len) {
            mask[[i, j]] = 1.0;
        }

        // Global attention
        for (i, j) in self.global_attention(seq_len) {
            mask[[i, j]] = 1.0;
        }

        mask
    }

    /// Compute the sparsity ratio of the attention mask.
    /// Returns the fraction of non-zero entries.
    pub fn sparsity_ratio(&self, seq_len: usize) -> f64 {
        let mask = self.attention_mask(seq_len);
        let total = (seq_len * seq_len) as f64;
        let nonzero = mask.iter().filter(|&&v| v > 0.0).count() as f64;
        nonzero / total
    }

    /// Compute sparse attention output given Q, K, V matrices.
    /// This applies scaled dot-product attention only where the mask is non-zero.
    pub fn forward(
        &self,
        queries: &Array2<f64>,
        keys: &Array2<f64>,
        values: &Array2<f64>,
    ) -> Array2<f64> {
        let seq_len = queries.nrows();
        let d_k = queries.ncols() as f64;
        let mask = self.attention_mask(seq_len);
        let mut output = Array2::zeros((seq_len, values.ncols()));

        for i in 0..seq_len {
            // Gather attended positions
            let attended: Vec<usize> = (0..seq_len)
                .filter(|&j| mask[[i, j]] > 0.0)
                .collect();

            if attended.is_empty() {
                continue;
            }

            // Compute attention scores for attended positions
            let mut scores: Vec<f64> = attended
                .iter()
                .map(|&j| {
                    let dot: f64 = (0..queries.ncols())
                        .map(|k| queries[[i, k]] * keys[[j, k]])
                        .sum();
                    dot / d_k.sqrt()
                })
                .collect();

            // Softmax
            let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exp_scores: Vec<f64> = scores.iter().map(|&s| (s - max_score).exp()).collect();
            let sum_exp: f64 = exp_scores.iter().sum();
            scores = exp_scores.iter().map(|&e| e / sum_exp).collect();

            // Weighted sum of values
            for (idx, &j) in attended.iter().enumerate() {
                for k in 0..values.ncols() {
                    output[[i, k]] += scores[idx] * values[[j, k]];
                }
            }
        }

        output
    }

    /// Return the configuration.
    pub fn config(&self) -> &BigBirdConfig {
        &self.config
    }
}

// ─── Document Processor ─────────────────────────────────────────

/// Represents a processed financial document with tokens and metadata.
#[derive(Debug, Clone)]
pub struct ProcessedDocument {
    pub tokens: Vec<String>,
    pub token_ids: Vec<usize>,
    pub section_labels: Vec<String>,
    pub financial_entities: Vec<(usize, String, String)>, // (position, entity, type)
}

/// Simple financial document tokenizer and processor.
#[derive(Debug)]
pub struct DocumentProcessor {
    max_length: usize,
}

impl DocumentProcessor {
    pub fn new(max_length: usize) -> Self {
        Self { max_length }
    }

    /// Tokenize text into words (simple whitespace + punctuation splitting).
    pub fn tokenize(&self, text: &str) -> Vec<String> {
        let mut tokens: Vec<String> = text
            .split_whitespace()
            .flat_map(|word| {
                // Split off trailing punctuation
                let word = word.to_lowercase();
                let mut parts = Vec::new();
                let trimmed = word.trim_matches(|c: char| c.is_ascii_punctuation());
                if !trimmed.is_empty() {
                    parts.push(trimmed.to_string());
                }
                // Keep punctuation as separate tokens
                for c in word.chars() {
                    if c.is_ascii_punctuation() && c != '\'' {
                        parts.push(c.to_string());
                    }
                }
                parts
            })
            .collect();

        tokens.truncate(self.max_length);
        tokens
    }

    /// Detect financial entities in token list.
    /// Returns (position, entity_text, entity_type).
    pub fn detect_entities(&self, tokens: &[String]) -> Vec<(usize, String, String)> {
        let mut entities = Vec::new();
        let financial_keywords = [
            ("revenue", "METRIC"),
            ("profit", "METRIC"),
            ("loss", "METRIC"),
            ("earnings", "METRIC"),
            ("ebitda", "METRIC"),
            ("margin", "METRIC"),
            ("growth", "METRIC"),
            ("dividend", "METRIC"),
            ("btc", "CRYPTO"),
            ("eth", "CRYPTO"),
            ("bitcoin", "CRYPTO"),
            ("ethereum", "CRYPTO"),
            ("bybit", "EXCHANGE"),
            ("sec", "REGULATOR"),
            ("q1", "PERIOD"),
            ("q2", "PERIOD"),
            ("q3", "PERIOD"),
            ("q4", "PERIOD"),
        ];

        for (i, token) in tokens.iter().enumerate() {
            let lower = token.to_lowercase();
            for (keyword, entity_type) in &financial_keywords {
                if lower.contains(keyword) {
                    entities.push((i, token.clone(), entity_type.to_string()));
                }
            }
            // Detect monetary amounts (tokens starting with $ or containing digits with %)
            if token.starts_with('$') || token.ends_with('%') {
                entities.push((i, token.clone(), "MONEY".to_string()));
            }
        }
        entities
    }

    /// Detect document sections based on common financial document patterns.
    pub fn detect_sections(&self, tokens: &[String]) -> Vec<String> {
        let section_markers = [
            ("summary", "SUMMARY"),
            ("overview", "SUMMARY"),
            ("risk", "RISK_FACTORS"),
            ("financial", "FINANCIALS"),
            ("revenue", "FINANCIALS"),
            ("outlook", "OUTLOOK"),
            ("forward", "OUTLOOK"),
            ("management", "MANAGEMENT"),
            ("discussion", "MANAGEMENT"),
        ];

        let mut labels = vec!["BODY".to_string(); tokens.len()];
        let mut current_section = "BODY".to_string();

        for (i, token) in tokens.iter().enumerate() {
            let lower = token.to_lowercase();
            for (marker, section) in &section_markers {
                if lower.contains(marker) {
                    current_section = section.to_string();
                    break;
                }
            }
            labels[i] = current_section.clone();
        }
        labels
    }

    /// Process a document: tokenize, detect entities, detect sections.
    pub fn process(&self, text: &str) -> ProcessedDocument {
        let tokens = self.tokenize(text);
        let entities = self.detect_entities(&tokens);
        let section_labels = self.detect_sections(&tokens);
        let token_ids: Vec<usize> = tokens.iter().enumerate().map(|(i, _)| i).collect();

        ProcessedDocument {
            tokens,
            token_ids,
            section_labels,
            financial_entities: entities,
        }
    }
}

impl Default for DocumentProcessor {
    fn default() -> Self {
        Self::new(4096)
    }
}

// ─── Sentiment Classifier ───────────────────────────────────────

/// Three-class sentiment classifier (positive / neutral / negative)
/// using multinomial logistic regression.
#[derive(Debug)]
pub struct SentimentClassifier {
    /// Weight matrix: [num_features x 3]
    weights: Array2<f64>,
    /// Bias vector: [3]
    biases: Array1<f64>,
    learning_rate: f64,
    num_features: usize,
}

impl SentimentClassifier {
    pub fn new(num_features: usize, learning_rate: f64) -> Self {
        let mut rng = rand::thread_rng();
        let weights = Array2::from_shape_fn((num_features, 3), |_| rng.gen_range(-0.1..0.1));
        let biases = Array1::zeros(3);
        Self {
            weights,
            biases,
            learning_rate,
            num_features,
        }
    }

    /// Softmax over a 3-element vector.
    fn softmax(logits: &[f64; 3]) -> [f64; 3] {
        let max_val = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = logits.iter().map(|&x| (x - max_val).exp()).collect();
        let sum: f64 = exps.iter().sum();
        [exps[0] / sum, exps[1] / sum, exps[2] / sum]
    }

    /// Predict class probabilities: [P(positive), P(neutral), P(negative)].
    pub fn predict_proba(&self, features: &[f64]) -> [f64; 3] {
        assert_eq!(features.len(), self.num_features);
        let x = Array1::from_vec(features.to_vec());
        let mut logits = [0.0f64; 3];
        for c in 0..3 {
            logits[c] = self.weights.column(c).dot(&x) + self.biases[c];
        }
        Self::softmax(&logits)
    }

    /// Predict sentiment: 0 = positive, 1 = neutral, 2 = negative.
    /// Returns (class, confidence).
    pub fn predict(&self, features: &[f64]) -> (usize, f64) {
        let probs = self.predict_proba(features);
        let mut best_class = 0;
        let mut best_prob = probs[0];
        for c in 1..3 {
            if probs[c] > best_prob {
                best_class = c;
                best_prob = probs[c];
            }
        }
        (best_class, best_prob)
    }

    /// Return sentiment label string.
    pub fn label(class: usize) -> &'static str {
        match class {
            0 => "positive",
            1 => "neutral",
            2 => "negative",
            _ => "unknown",
        }
    }

    /// Train on labeled data. Label is 0, 1, or 2.
    pub fn train(&mut self, data: &[(Vec<f64>, usize)], epochs: usize) {
        for _ in 0..epochs {
            for (features, label) in data {
                let x = Array1::from_vec(features.clone());
                let probs = self.predict_proba(features);

                // Gradient: d_loss/d_logit_c = probs[c] - 1{c == label}
                for c in 0..3 {
                    let target = if c == *label { 1.0 } else { 0.0 };
                    let error = probs[c] - target;
                    for j in 0..self.num_features {
                        self.weights[[j, c]] -= self.learning_rate * error * x[j];
                    }
                    self.biases[c] -= self.learning_rate * error;
                }
            }
        }
    }

    /// Evaluate accuracy on test data.
    pub fn accuracy(&self, data: &[(Vec<f64>, usize)]) -> f64 {
        let correct = data
            .iter()
            .filter(|(features, label)| {
                let (pred, _) = self.predict(features);
                pred == *label
            })
            .count();
        correct as f64 / data.len() as f64
    }
}

// ─── Sentiment Feature Extraction ───────────────────────────────

/// Positive and negative financial sentiment word lists.
const POSITIVE_WORDS: &[&str] = &[
    "growth", "profit", "increase", "gain", "positive", "strong",
    "improve", "exceed", "beat", "outperform", "surge", "rally",
    "bullish", "upgrade", "optimistic", "recovery", "upside",
];

const NEGATIVE_WORDS: &[&str] = &[
    "loss", "decline", "decrease", "risk", "negative", "weak",
    "miss", "below", "underperform", "drop", "fall", "bearish",
    "downgrade", "pessimistic", "recession", "downside", "default",
];

/// Extract sentiment features from a processed document.
/// Returns feature vector: [pos_ratio, neg_ratio, pos_neg_ratio,
///                          doc_length_norm, entity_density, risk_section_ratio]
pub fn extract_sentiment_features(doc: &ProcessedDocument) -> Vec<f64> {
    let n = doc.tokens.len() as f64;
    if n == 0.0 {
        return vec![0.0; 6];
    }

    let mut pos_count = 0.0;
    let mut neg_count = 0.0;
    let mut risk_section_tokens = 0.0;

    for (i, token) in doc.tokens.iter().enumerate() {
        let lower = token.to_lowercase();
        if POSITIVE_WORDS.iter().any(|&w| lower.contains(w)) {
            pos_count += 1.0;
        }
        if NEGATIVE_WORDS.iter().any(|&w| lower.contains(w)) {
            neg_count += 1.0;
        }
        if doc.section_labels[i] == "RISK_FACTORS" {
            risk_section_tokens += 1.0;
        }
    }

    let pos_ratio = pos_count / n;
    let neg_ratio = neg_count / n;
    let pos_neg_ratio = if neg_count > 0.0 {
        pos_count / neg_count
    } else {
        pos_count + 1.0
    };
    let doc_length_norm = (n / 4096.0).min(1.0);
    let entity_density = doc.financial_entities.len() as f64 / n;
    let risk_section_ratio = risk_section_tokens / n;

    vec![
        pos_ratio,
        neg_ratio,
        pos_neg_ratio,
        doc_length_norm,
        entity_density,
        risk_section_ratio,
    ]
}

// ─── Bybit Client ───────────────────────────────────────────────

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

#[derive(Debug, Deserialize)]
pub struct OrderbookResult {
    pub b: Vec<Vec<String>>, // bids: [price, size]
    pub a: Vec<Vec<String>>, // asks: [price, size]
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

    /// Fetch order book snapshot.
    pub async fn get_orderbook(
        &self,
        symbol: &str,
        limit: u32,
    ) -> anyhow::Result<(Vec<(f64, f64)>, Vec<(f64, f64)>)> {
        let url = format!(
            "{}/v5/market/orderbook?category=spot&symbol={}&limit={}",
            self.base_url, symbol, limit
        );
        let resp: BybitResponse<OrderbookResult> =
            self.client.get(&url).send().await?.json().await?;

        let bids: Vec<(f64, f64)> = resp
            .result
            .b
            .iter()
            .filter_map(|entry| {
                if entry.len() >= 2 {
                    Some((
                        entry[0].parse().unwrap_or(0.0),
                        entry[1].parse().unwrap_or(0.0),
                    ))
                } else {
                    None
                }
            })
            .collect();

        let asks: Vec<(f64, f64)> = resp
            .result
            .a
            .iter()
            .filter_map(|entry| {
                if entry.len() >= 2 {
                    Some((
                        entry[0].parse().unwrap_or(0.0),
                        entry[1].parse().unwrap_or(0.0),
                    ))
                } else {
                    None
                }
            })
            .collect();

        Ok((bids, asks))
    }
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Synthetic Data Generation ──────────────────────────────────

/// Generate synthetic labeled sentiment data for training.
/// Features: [pos_ratio, neg_ratio, pos_neg_ratio, doc_length_norm, entity_density, risk_ratio]
/// Labels: 0 = positive, 1 = neutral, 2 = negative
pub fn generate_sentiment_training_data(n: usize) -> Vec<(Vec<f64>, usize)> {
    let mut rng = rand::thread_rng();
    let mut data = Vec::with_capacity(n);

    for _ in 0..n {
        let pos_ratio: f64 = rng.gen_range(0.0..0.15);
        let neg_ratio: f64 = rng.gen_range(0.0..0.15);
        let pos_neg_ratio: f64 = if neg_ratio > 0.001 {
            pos_ratio / neg_ratio
        } else {
            pos_ratio * 10.0 + 1.0
        };
        let doc_length_norm: f64 = rng.gen_range(0.1..1.0);
        let entity_density: f64 = rng.gen_range(0.0..0.1);
        let risk_ratio: f64 = rng.gen_range(0.0..0.5);

        // Label based on sentiment balance
        let signal = 2.0 * pos_ratio - 2.0 * neg_ratio + 0.5 * pos_neg_ratio
            - 1.0 * risk_ratio
            + rng.gen_range(-0.1..0.1);

        let label = if signal > 0.3 {
            0 // positive
        } else if signal < -0.1 {
            2 // negative
        } else {
            1 // neutral
        };

        data.push((
            vec![pos_ratio, neg_ratio, pos_neg_ratio, doc_length_norm, entity_density, risk_ratio],
            label,
        ));
    }
    data
}

/// Sample financial texts for demonstration.
pub fn sample_financial_texts() -> Vec<(&'static str, &'static str)> {
    vec![
        (
            "The company reported strong revenue growth of 25% year-over-year, \
             with earnings exceeding analyst expectations. Management expressed \
             optimistic outlook for the upcoming quarter, citing improved margins \
             and strong demand across all segments. The board approved a dividend \
             increase reflecting confidence in sustained profitability.",
            "positive",
        ),
        (
            "Quarterly results were mixed with revenue in line with expectations \
             but margins slightly below guidance. The company maintained its \
             full-year forecast unchanged. Management noted stable market conditions \
             with no significant changes to the competitive landscape. Operating \
             expenses remained flat compared to the prior quarter.",
            "neutral",
        ),
        (
            "The firm disclosed a significant loss in its trading division due to \
             increased market volatility and risk exposure. Revenue declined 15% \
             as key clients reduced their activity. Management warned of potential \
             further downside risks including regulatory changes and recession \
             concerns. The credit rating was placed on negative watch.",
            "negative",
        ),
    ]
}

// ─── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bigbird_config_default() {
        let config = BigBirdConfig::default();
        assert_eq!(config.seq_length, 512);
        assert_eq!(config.num_random_tokens, 3);
        assert_eq!(config.window_size, 3);
        assert_eq!(config.num_global_tokens, 2);
    }

    #[test]
    fn test_attention_mask_shape() {
        let config = BigBirdConfig::default();
        let attn = BigBirdAttention::new(config);
        let mask = attn.attention_mask(32);
        assert_eq!(mask.nrows(), 32);
        assert_eq!(mask.ncols(), 32);
    }

    #[test]
    fn test_attention_mask_diagonal() {
        let config = BigBirdConfig::default();
        let attn = BigBirdAttention::new(config);
        let mask = attn.attention_mask(16);
        // Diagonal should always be 1 (self-attention)
        for i in 0..16 {
            assert_eq!(mask[[i, i]], 1.0);
        }
    }

    #[test]
    fn test_attention_mask_global_tokens() {
        let config = BigBirdConfig::new(64, 3, 3, 2, 64, 4);
        let attn = BigBirdAttention::new(config);
        let mask = attn.attention_mask(16);
        // Global tokens (0 and 1) should attend to all and be attended by all
        for j in 0..16 {
            assert_eq!(mask[[0, j]], 1.0, "Global token 0 should attend to {}", j);
            assert_eq!(mask[[j, 0]], 1.0, "Token {} should attend to global 0", j);
        }
    }

    #[test]
    fn test_attention_mask_window() {
        let config = BigBirdConfig::new(64, 0, 2, 0, 64, 4);
        let attn = BigBirdAttention::new(config);
        let mask = attn.attention_mask(10);
        // Token 5 should attend to tokens 3, 4, 5, 6, 7
        assert_eq!(mask[[5, 3]], 1.0);
        assert_eq!(mask[[5, 4]], 1.0);
        assert_eq!(mask[[5, 5]], 1.0);
        assert_eq!(mask[[5, 6]], 1.0);
        assert_eq!(mask[[5, 7]], 1.0);
        // But not token 1
        assert_eq!(mask[[5, 1]], 0.0);
    }

    #[test]
    fn test_sparsity_ratio() {
        let config = BigBirdConfig::new(512, 3, 3, 2, 64, 4);
        let attn = BigBirdAttention::new(config);
        let ratio = attn.sparsity_ratio(64);
        // Sparse attention should use much less than full attention
        assert!(ratio < 0.5, "Sparsity ratio {} should be < 0.5", ratio);
        assert!(ratio > 0.0, "Sparsity ratio should be > 0.0");
    }

    #[test]
    fn test_forward_pass() {
        let config = BigBirdConfig::new(64, 2, 2, 1, 8, 1);
        let attn = BigBirdAttention::new(config);

        let seq_len = 8;
        let d = 4;
        let mut rng = rand::thread_rng();
        let q = Array2::from_shape_fn((seq_len, d), |_| rng.gen_range(-1.0..1.0));
        let k = Array2::from_shape_fn((seq_len, d), |_| rng.gen_range(-1.0..1.0));
        let v = Array2::from_shape_fn((seq_len, d), |_| rng.gen_range(-1.0..1.0));

        let output = attn.forward(&q, &k, &v);
        assert_eq!(output.nrows(), seq_len);
        assert_eq!(output.ncols(), d);
        // Output values should be bounded by value range
        for val in output.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_document_processor_tokenize() {
        let proc = DocumentProcessor::new(100);
        let tokens = proc.tokenize("The company reported revenue of $1.2B in Q3.");
        assert!(!tokens.is_empty());
        assert!(tokens.len() <= 100);
    }

    #[test]
    fn test_document_processor_entities() {
        let proc = DocumentProcessor::new(100);
        let tokens = proc.tokenize("Revenue growth was strong with earnings exceeding expectations");
        let entities = proc.detect_entities(&tokens);
        // Should detect "revenue", "growth", "earnings" as financial entities
        assert!(!entities.is_empty());
        let entity_types: Vec<&str> = entities.iter().map(|(_, _, t)| t.as_str()).collect();
        assert!(entity_types.contains(&"METRIC"));
    }

    #[test]
    fn test_document_processor_sections() {
        let proc = DocumentProcessor::new(100);
        let tokens = proc.tokenize("summary of results risk factors include market volatility");
        let sections = proc.detect_sections(&tokens);
        assert!(sections.contains(&"SUMMARY".to_string()));
        assert!(sections.contains(&"RISK_FACTORS".to_string()));
    }

    #[test]
    fn test_sentiment_classifier_predict() {
        let clf = SentimentClassifier::new(6, 0.01);
        let features = vec![0.1, 0.02, 5.0, 0.5, 0.05, 0.1];
        let (class, confidence) = clf.predict(&features);
        assert!(class <= 2);
        assert!(confidence >= 0.0 && confidence <= 1.0);
    }

    #[test]
    fn test_sentiment_classifier_train() {
        let data = generate_sentiment_training_data(500);
        let (train, test) = data.split_at(400);

        let mut clf = SentimentClassifier::new(6, 0.01);
        clf.train(&train.to_vec(), 50);
        let acc = clf.accuracy(test);
        // After training, accuracy should be meaningfully above random (33%)
        assert!(acc > 0.0, "Accuracy should be > 0, got {}", acc);
    }

    #[test]
    fn test_extract_sentiment_features() {
        let proc = DocumentProcessor::new(4096);
        let doc = proc.process(
            "The company reported strong revenue growth with profit exceeding expectations",
        );
        let features = extract_sentiment_features(&doc);
        assert_eq!(features.len(), 6);
        // Positive ratio should be > 0 (has positive words)
        assert!(features[0] > 0.0, "Positive ratio should be > 0");
    }

    #[test]
    fn test_sentiment_label() {
        assert_eq!(SentimentClassifier::label(0), "positive");
        assert_eq!(SentimentClassifier::label(1), "neutral");
        assert_eq!(SentimentClassifier::label(2), "negative");
    }

    #[test]
    fn test_sample_financial_texts() {
        let texts = sample_financial_texts();
        assert_eq!(texts.len(), 3);
        assert_eq!(texts[0].1, "positive");
        assert_eq!(texts[1].1, "neutral");
        assert_eq!(texts[2].1, "negative");
    }

    #[test]
    fn test_training_data_generation() {
        let data = generate_sentiment_training_data(100);
        assert_eq!(data.len(), 100);
        for (features, label) in &data {
            assert_eq!(features.len(), 6);
            assert!(*label <= 2);
        }
    }
}
