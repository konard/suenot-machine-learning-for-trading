use ndarray::Array1;
use rand::Rng;
use serde::Deserialize;

// ─── Sliding Window Attention ─────────────────────────────────────

/// Implements sliding window (local) attention for long sequences.
///
/// Each token attends only to `window_size` neighbors on each side,
/// reducing the O(n^2) complexity to O(n * w).
#[derive(Debug)]
pub struct SlidingWindowAttention {
    window_size: usize,
    scaling_factor: f64,
}

impl SlidingWindowAttention {
    /// Create a new sliding window attention layer.
    /// - `window_size`: number of tokens to attend to on each side
    /// - `d_k`: dimension of the key vectors (used for scaling)
    pub fn new(window_size: usize, d_k: usize) -> Self {
        Self {
            window_size,
            scaling_factor: (d_k as f64).sqrt(),
        }
    }

    /// Compute local attention scores for a sequence.
    ///
    /// `query` and `key` are vectors of per-token representations.
    /// Returns attention-weighted values for each position.
    pub fn forward(&self, query: &[Vec<f64>], key: &[Vec<f64>], value: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let seq_len = query.len();
        let d_v = if value.is_empty() { 0 } else { value[0].len() };
        let mut output = vec![vec![0.0; d_v]; seq_len];

        for i in 0..seq_len {
            let start = i.saturating_sub(self.window_size);
            let end = (i + self.window_size + 1).min(seq_len);

            // Compute attention scores within the window
            let mut scores: Vec<f64> = Vec::new();
            for j in start..end {
                let dot: f64 = query[i].iter().zip(key[j].iter()).map(|(a, b)| a * b).sum();
                scores.push(dot / self.scaling_factor);
            }

            // Softmax
            let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exp_scores: Vec<f64> = scores.iter().map(|s| (s - max_score).exp()).collect();
            let sum_exp: f64 = exp_scores.iter().sum();
            let weights: Vec<f64> = exp_scores.iter().map(|e| e / sum_exp).collect();

            // Weighted sum of values
            for (w_idx, j) in (start..end).enumerate() {
                for (d, v) in value[j].iter().enumerate() {
                    output[i][d] += weights[w_idx] * v;
                }
            }
        }

        output
    }

    pub fn window_size(&self) -> usize {
        self.window_size
    }
}

// ─── Global Attention Mask ────────────────────────────────────────

/// Manages which token positions receive global attention.
///
/// Global attention tokens attend to all positions in the sequence,
/// and all positions attend to them. This enables information flow
/// across the entire document for designated tokens (e.g., [CLS],
/// section headers, entity tokens).
#[derive(Debug)]
pub struct GlobalAttentionMask {
    seq_len: usize,
    global_positions: Vec<bool>,
}

impl GlobalAttentionMask {
    /// Create a mask with no global positions.
    pub fn new(seq_len: usize) -> Self {
        Self {
            seq_len,
            global_positions: vec![false; seq_len],
        }
    }

    /// Designate a position as having global attention.
    pub fn set_global(&mut self, pos: usize) {
        if pos < self.seq_len {
            self.global_positions[pos] = true;
        }
    }

    /// Set the [CLS] token (position 0) as global.
    pub fn set_cls_global(&mut self) {
        self.set_global(0);
    }

    /// Set multiple positions as global (e.g., section headers).
    pub fn set_positions_global(&mut self, positions: &[usize]) {
        for &pos in positions {
            self.set_global(pos);
        }
    }

    /// Check whether a position has global attention.
    pub fn is_global(&self, pos: usize) -> bool {
        pos < self.seq_len && self.global_positions[pos]
    }

    /// Return indices of all global positions.
    pub fn global_indices(&self) -> Vec<usize> {
        self.global_positions
            .iter()
            .enumerate()
            .filter_map(|(i, &g)| if g { Some(i) } else { None })
            .collect()
    }

    /// Count of global attention positions.
    pub fn num_global(&self) -> usize {
        self.global_positions.iter().filter(|&&g| g).count()
    }
}

// ─── Longformer Encoder ───────────────────────────────────────────

/// Combines sliding window and global attention into a complete
/// encoding layer, producing contextualized representations.
#[derive(Debug)]
pub struct LongformerEncoder {
    sliding_attn: SlidingWindowAttention,
    d_model: usize,
}

impl LongformerEncoder {
    /// Create a new encoder layer.
    /// - `window_size`: local attention window
    /// - `d_model`: model dimension
    pub fn new(window_size: usize, d_model: usize) -> Self {
        Self {
            sliding_attn: SlidingWindowAttention::new(window_size, d_model),
            d_model,
        }
    }

    /// Encode a sequence using mixed local + global attention.
    ///
    /// `embeddings`: per-token embedding vectors
    /// `global_mask`: which positions have global attention
    ///
    /// Returns contextualized representations for each token.
    pub fn encode(
        &self,
        embeddings: &[Vec<f64>],
        global_mask: &GlobalAttentionMask,
    ) -> Vec<Vec<f64>> {
        let seq_len = embeddings.len();
        if seq_len == 0 {
            return Vec::new();
        }

        // Step 1: Local attention (sliding window)
        let local_output = self.sliding_attn.forward(embeddings, embeddings, embeddings);

        // Step 2: Global attention contribution
        let global_indices = global_mask.global_indices();

        if global_indices.is_empty() {
            // No global tokens, return local output with layer norm
            return self.layer_norm(&local_output);
        }

        // Compute global attention: global tokens attend to all, all attend to global
        let mut output = local_output;

        for &g_idx in &global_indices {
            // Global token attends to all positions
            let mut scores: Vec<f64> = Vec::with_capacity(seq_len);
            for j in 0..seq_len {
                let dot: f64 = embeddings[g_idx]
                    .iter()
                    .zip(embeddings[j].iter())
                    .map(|(a, b)| a * b)
                    .sum();
                scores.push(dot / (self.d_model as f64).sqrt());
            }

            // Softmax for global token
            let max_s = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exp_s: Vec<f64> = scores.iter().map(|s| (s - max_s).exp()).collect();
            let sum_s: f64 = exp_s.iter().sum();
            let weights: Vec<f64> = exp_s.iter().map(|e| e / sum_s).collect();

            // Update global token representation
            let mut global_repr = vec![0.0; self.d_model];
            for j in 0..seq_len {
                for d in 0..self.d_model.min(embeddings[j].len()) {
                    global_repr[d] += weights[j] * embeddings[j][d];
                }
            }
            output[g_idx] = global_repr;

            // All tokens receive information from global token (additive)
            for i in 0..seq_len {
                let dot: f64 = embeddings[i]
                    .iter()
                    .zip(embeddings[g_idx].iter())
                    .map(|(a, b)| a * b)
                    .sum();
                let weight = 1.0 / (1.0 + (-dot / (self.d_model as f64).sqrt()).exp()); // sigmoid gate
                for d in 0..self.d_model.min(output[i].len()) {
                    output[i][d] += weight * embeddings[g_idx][d] * 0.1; // scaled contribution
                }
            }
        }

        self.layer_norm(&output)
    }

    /// Simple layer normalization.
    fn layer_norm(&self, vectors: &[Vec<f64>]) -> Vec<Vec<f64>> {
        vectors
            .iter()
            .map(|v| {
                let mean: f64 = v.iter().sum::<f64>() / v.len() as f64;
                let var: f64 = v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / v.len() as f64;
                let std = (var + 1e-6).sqrt();
                v.iter().map(|x| (x - mean) / std).collect()
            })
            .collect()
    }

    pub fn d_model(&self) -> usize {
        self.d_model
    }
}

// ─── Financial Sentiment Lexicon ──────────────────────────────────

/// A simple financial sentiment lexicon for scoring text.
///
/// Uses the Loughran-McDonald approach: domain-specific word lists
/// calibrated for financial text (where "liability" is not negative).
#[derive(Debug)]
pub struct FinancialSentimentLexicon {
    positive_words: Vec<String>,
    negative_words: Vec<String>,
}

impl FinancialSentimentLexicon {
    /// Create a lexicon with common financial sentiment words.
    pub fn new() -> Self {
        Self {
            positive_words: vec![
                "growth", "profit", "increase", "gain", "positive", "strong",
                "exceeded", "improvement", "robust", "optimistic", "bullish",
                "upgrade", "outperform", "beat", "recovery", "opportunity",
                "innovative", "efficient", "expansion", "momentum", "upside",
                "dividend", "surplus", "breakthrough", "accelerate", "success",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
            negative_words: vec![
                "loss", "decline", "decrease", "risk", "negative", "weak",
                "missed", "deterioration", "concern", "pessimistic", "bearish",
                "downgrade", "underperform", "fail", "recession", "threat",
                "impairment", "default", "bankruptcy", "litigation", "downside",
                "deficit", "writedown", "restructuring", "uncertainty", "volatile",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
        }
    }

    /// Score a text based on the sentiment lexicon.
    /// Returns a value in [-1, 1].
    pub fn score(&self, text: &str) -> f64 {
        let lower = text.to_lowercase();
        let words: Vec<&str> = lower.split_whitespace().collect();
        let total = words.len() as f64;
        if total == 0.0 {
            return 0.0;
        }

        let pos_count = words
            .iter()
            .filter(|w| {
                let clean: String = w.chars().filter(|c| c.is_alphanumeric()).collect();
                self.positive_words.iter().any(|p| p == &clean)
            })
            .count() as f64;

        let neg_count = words
            .iter()
            .filter(|w| {
                let clean: String = w.chars().filter(|c| c.is_alphanumeric()).collect();
                self.negative_words.iter().any(|n| n == &clean)
            })
            .count() as f64;

        (pos_count - neg_count) / total
    }

    /// Count positive and negative words separately.
    pub fn word_counts(&self, text: &str) -> (usize, usize) {
        let lower = text.to_lowercase();
        let words: Vec<&str> = lower.split_whitespace().collect();

        let pos = words
            .iter()
            .filter(|w| {
                let clean: String = w.chars().filter(|c| c.is_alphanumeric()).collect();
                self.positive_words.iter().any(|p| p == &clean)
            })
            .count();

        let neg = words
            .iter()
            .filter(|w| {
                let clean: String = w.chars().filter(|c| c.is_alphanumeric()).collect();
                self.negative_words.iter().any(|n| n == &clean)
            })
            .count();

        (pos, neg)
    }
}

impl Default for FinancialSentimentLexicon {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Sentiment Classifier (Logistic Regression) ───────────────────

/// Binary sentiment classifier using logistic regression.
///
/// Features: [lexicon_score, doc_length_norm, positive_ratio,
///            negative_ratio, sentence_count_norm]
#[derive(Debug)]
pub struct SentimentClassifier {
    weights: Array1<f64>,
    bias: f64,
    learning_rate: f64,
    num_features: usize,
}

impl SentimentClassifier {
    pub fn new(num_features: usize, learning_rate: f64) -> Self {
        let mut rng = rand::thread_rng();
        let weights = Array1::from_vec(
            (0..num_features)
                .map(|_| rng.gen_range(-0.1..0.1))
                .collect(),
        );
        Self {
            weights,
            bias: 0.0,
            learning_rate,
            num_features,
        }
    }

    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    /// Predict probability of positive sentiment (label = 1).
    pub fn predict_proba(&self, features: &[f64]) -> f64 {
        assert_eq!(features.len(), self.num_features);
        let x = Array1::from_vec(features.to_vec());
        let z = self.weights.dot(&x) + self.bias;
        Self::sigmoid(z)
    }

    /// Predict sentiment: true = positive, false = negative.
    /// Returns (is_positive, confidence).
    pub fn predict(&self, features: &[f64]) -> (bool, f64) {
        let prob = self.predict_proba(features);
        if prob >= 0.5 {
            (true, prob)
        } else {
            (false, 1.0 - prob)
        }
    }

    /// Train on labeled data for `epochs` iterations.
    pub fn train(&mut self, data: &[(Vec<f64>, f64)], epochs: usize) {
        for _ in 0..epochs {
            for (features, label) in data {
                let x = Array1::from_vec(features.clone());
                let z = self.weights.dot(&x) + self.bias;
                let pred = Self::sigmoid(z);
                let error = pred - label;

                for j in 0..self.num_features {
                    self.weights[j] -= self.learning_rate * error * x[j];
                }
                self.bias -= self.learning_rate * error;
            }
        }
    }

    /// Evaluate accuracy on a test set.
    pub fn accuracy(&self, data: &[(Vec<f64>, f64)]) -> f64 {
        let correct = data
            .iter()
            .filter(|(features, label)| {
                let (pred, _) = self.predict(features);
                let label_bool = *label >= 0.5;
                pred == label_bool
            })
            .count();
        correct as f64 / data.len() as f64
    }

    pub fn weights(&self) -> &Array1<f64> {
        &self.weights
    }

    pub fn bias(&self) -> f64 {
        self.bias
    }
}

// ─── Text Feature Extractor ───────────────────────────────────────

/// Extracts numerical features from text for the sentiment classifier.
pub struct TextFeatureExtractor {
    lexicon: FinancialSentimentLexicon,
}

impl TextFeatureExtractor {
    pub fn new() -> Self {
        Self {
            lexicon: FinancialSentimentLexicon::new(),
        }
    }

    /// Extract features from a text document.
    ///
    /// Returns: [lexicon_score, doc_length_norm, positive_ratio,
    ///           negative_ratio, sentence_count_norm]
    pub fn extract(&self, text: &str) -> Vec<f64> {
        let lexicon_score = self.lexicon.score(text);
        let (pos_count, neg_count) = self.lexicon.word_counts(text);
        let word_count = text.split_whitespace().count() as f64;
        let sentence_count = text
            .chars()
            .filter(|&c| c == '.' || c == '!' || c == '?')
            .count() as f64;

        let doc_length_norm = (word_count / 1000.0).min(1.0);
        let total_sentiment = (pos_count + neg_count) as f64;
        let positive_ratio = if total_sentiment > 0.0 {
            pos_count as f64 / total_sentiment
        } else {
            0.5
        };
        let negative_ratio = if total_sentiment > 0.0 {
            neg_count as f64 / total_sentiment
        } else {
            0.5
        };
        let sentence_count_norm = (sentence_count / 50.0).min(1.0);

        vec![
            lexicon_score,
            doc_length_norm,
            positive_ratio,
            negative_ratio,
            sentence_count_norm,
        ]
    }
}

impl Default for TextFeatureExtractor {
    fn default() -> Self {
        Self::new()
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

// ─── Synthetic Data Generation ─────────────────────────────────────

/// Generate synthetic financial texts for testing.
pub fn generate_synthetic_texts(n: usize) -> Vec<(String, f64)> {
    let mut rng = rand::thread_rng();
    let positive_phrases = [
        "The company reported strong growth in revenue and profit margins exceeded expectations.",
        "Robust demand and expansion into new markets drove significant gains this quarter.",
        "Management expressed optimistic outlook with momentum in key business segments.",
        "The recovery in sales has been strong with positive trends across all divisions.",
        "Innovation and efficiency improvements led to breakthrough performance and upside.",
    ];
    let negative_phrases = [
        "Revenue decline and ongoing litigation risk continue to concern investors.",
        "The company faces restructuring costs and uncertainty in key markets.",
        "Weak demand and deterioration in margins led to a significant loss this period.",
        "Management warned of recession risk and potential impairment of assets.",
        "Default risk increased as the company reported deficit and volatile earnings.",
    ];

    let mut data = Vec::with_capacity(n);
    for _ in 0..n {
        let is_positive = rng.gen_bool(0.5);
        let phrases = if is_positive {
            &positive_phrases[..]
        } else {
            &negative_phrases[..]
        };
        let idx = rng.gen_range(0..phrases.len());
        let text = phrases[idx].to_string();
        let label = if is_positive { 1.0 } else { 0.0 };
        data.push((text, label));
    }
    data
}

/// Generate training data from synthetic texts.
pub fn generate_training_data(n: usize) -> Vec<(Vec<f64>, f64)> {
    let extractor = TextFeatureExtractor::new();
    let texts = generate_synthetic_texts(n);
    texts
        .iter()
        .map(|(text, label)| (extractor.extract(text), *label))
        .collect()
}

/// Generate synthetic token embeddings for testing attention.
pub fn generate_synthetic_embeddings(seq_len: usize, d_model: usize) -> Vec<Vec<f64>> {
    let mut rng = rand::thread_rng();
    (0..seq_len)
        .map(|_| (0..d_model).map(|_| rng.gen_range(-1.0..1.0)).collect())
        .collect()
}

// ─── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sliding_window_attention_basic() {
        let attn = SlidingWindowAttention::new(2, 4);
        let seq_len = 6;
        let d = 4;
        let embeddings = generate_synthetic_embeddings(seq_len, d);
        let output = attn.forward(&embeddings, &embeddings, &embeddings);
        assert_eq!(output.len(), seq_len);
        for v in &output {
            assert_eq!(v.len(), d);
        }
    }

    #[test]
    fn test_sliding_window_short_sequence() {
        let attn = SlidingWindowAttention::new(10, 4);
        let embeddings = generate_synthetic_embeddings(3, 4);
        let output = attn.forward(&embeddings, &embeddings, &embeddings);
        assert_eq!(output.len(), 3);
    }

    #[test]
    fn test_global_attention_mask() {
        let mut mask = GlobalAttentionMask::new(100);
        assert_eq!(mask.num_global(), 0);

        mask.set_cls_global();
        assert!(mask.is_global(0));
        assert!(!mask.is_global(1));
        assert_eq!(mask.num_global(), 1);

        mask.set_positions_global(&[10, 20, 30]);
        assert_eq!(mask.num_global(), 4);
        assert!(mask.is_global(10));
        assert!(mask.is_global(20));
        assert!(mask.is_global(30));

        let indices = mask.global_indices();
        assert_eq!(indices, vec![0, 10, 20, 30]);
    }

    #[test]
    fn test_global_mask_out_of_bounds() {
        let mut mask = GlobalAttentionMask::new(5);
        mask.set_global(10); // out of bounds, should not panic
        assert!(!mask.is_global(10));
        assert_eq!(mask.num_global(), 0);
    }

    #[test]
    fn test_longformer_encoder_local_only() {
        let encoder = LongformerEncoder::new(2, 4);
        let embeddings = generate_synthetic_embeddings(8, 4);
        let mask = GlobalAttentionMask::new(8); // no global tokens
        let output = encoder.encode(&embeddings, &mask);
        assert_eq!(output.len(), 8);
        for v in &output {
            assert_eq!(v.len(), 4);
        }
    }

    #[test]
    fn test_longformer_encoder_with_global() {
        let encoder = LongformerEncoder::new(2, 4);
        let embeddings = generate_synthetic_embeddings(10, 4);
        let mut mask = GlobalAttentionMask::new(10);
        mask.set_cls_global();
        mask.set_global(5);
        let output = encoder.encode(&embeddings, &mask);
        assert_eq!(output.len(), 10);
    }

    #[test]
    fn test_longformer_encoder_empty() {
        let encoder = LongformerEncoder::new(2, 4);
        let mask = GlobalAttentionMask::new(0);
        let output = encoder.encode(&[], &mask);
        assert!(output.is_empty());
    }

    #[test]
    fn test_financial_lexicon_positive() {
        let lex = FinancialSentimentLexicon::new();
        let text = "Strong growth and robust profit improvement exceeded expectations";
        let score = lex.score(text);
        assert!(score > 0.0, "Expected positive score, got {}", score);
    }

    #[test]
    fn test_financial_lexicon_negative() {
        let lex = FinancialSentimentLexicon::new();
        let text = "Significant loss due to decline risk uncertainty and weak demand";
        let score = lex.score(text);
        assert!(score < 0.0, "Expected negative score, got {}", score);
    }

    #[test]
    fn test_financial_lexicon_neutral() {
        let lex = FinancialSentimentLexicon::new();
        let score = lex.score("The company filed its quarterly report today");
        assert!(
            score.abs() < 0.1,
            "Expected near-zero score, got {}",
            score
        );
    }

    #[test]
    fn test_financial_lexicon_empty() {
        let lex = FinancialSentimentLexicon::new();
        assert_eq!(lex.score(""), 0.0);
    }

    #[test]
    fn test_financial_lexicon_word_counts() {
        let lex = FinancialSentimentLexicon::new();
        let (pos, neg) = lex.word_counts("growth and loss with profit");
        assert_eq!(pos, 2); // growth, profit
        assert_eq!(neg, 1); // loss
    }

    #[test]
    fn test_text_feature_extractor() {
        let extractor = TextFeatureExtractor::new();
        let features = extractor.extract("Strong growth exceeded expectations with robust momentum");
        assert_eq!(features.len(), 5);
        // lexicon_score should be positive
        assert!(features[0] > 0.0);
        // doc_length_norm should be small (short text)
        assert!(features[1] < 0.1);
        // positive_ratio should be high
        assert!(features[2] > 0.5);
    }

    #[test]
    fn test_sentiment_classifier_predict() {
        let clf = SentimentClassifier::new(5, 0.01);
        let features = vec![0.5, 0.3, 0.8, 0.2, 0.1];
        let (_, confidence) = clf.predict(&features);
        assert!(confidence >= 0.5 && confidence <= 1.0);
    }

    #[test]
    fn test_sentiment_classifier_train() {
        let data = generate_training_data(500);
        let (train, test) = data.split_at(400);

        let mut clf = SentimentClassifier::new(5, 0.05);
        clf.train(&train.to_vec(), 100);
        let acc = clf.accuracy(test);

        assert!(acc > 0.0);
        assert!(acc >= 0.4, "accuracy after training: {}", acc);
    }

    #[test]
    fn test_synthetic_text_generation() {
        let texts = generate_synthetic_texts(100);
        assert_eq!(texts.len(), 100);
        for (text, label) in &texts {
            assert!(!text.is_empty());
            assert!(*label == 0.0 || *label == 1.0);
        }
    }

    #[test]
    fn test_training_data_generation() {
        let data = generate_training_data(50);
        assert_eq!(data.len(), 50);
        for (features, label) in &data {
            assert_eq!(features.len(), 5);
            assert!(*label == 0.0 || *label == 1.0);
        }
    }

    #[test]
    fn test_sliding_window_size() {
        let attn = SlidingWindowAttention::new(3, 8);
        assert_eq!(attn.window_size(), 3);
    }

    #[test]
    fn test_encoder_d_model() {
        let encoder = LongformerEncoder::new(4, 16);
        assert_eq!(encoder.d_model(), 16);
    }
}
