use rand::Rng;
use serde::Deserialize;
use std::collections::HashMap;

// ─── Utility Functions ──────────────────────────────────────────────

/// Compute softmax over a slice of f64 values.
pub fn softmax(x: &[f64]) -> Vec<f64> {
    let max_val = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = x.iter().map(|v| (v - max_val).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

/// Sigmoid activation function.
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// ReLU activation function.
pub fn relu(x: f64) -> f64 {
    x.max(0.0)
}

/// Compute the mean of a slice.
pub fn mean(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    data.iter().sum::<f64>() / data.len() as f64
}

/// Compute the standard deviation of a slice.
pub fn std_dev(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return 0.0;
    }
    let m = mean(data);
    let var = data.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (data.len() - 1) as f64;
    var.sqrt()
}

/// Dot product of two slices.
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Mean Absolute Error.
pub fn mae(actual: &[f64], predicted: &[f64]) -> f64 {
    if actual.is_empty() {
        return 0.0;
    }
    actual
        .iter()
        .zip(predicted.iter())
        .map(|(a, p)| (a - p).abs())
        .sum::<f64>()
        / actual.len() as f64
}

/// Root Mean Squared Error.
pub fn rmse(actual: &[f64], predicted: &[f64]) -> f64 {
    if actual.is_empty() {
        return 0.0;
    }
    let mse = actual
        .iter()
        .zip(predicted.iter())
        .map(|(a, p)| (a - p).powi(2))
        .sum::<f64>()
        / actual.len() as f64;
    mse.sqrt()
}

// ─── Financial Tokenizer ────────────────────────────────────────────

/// A simple financial text tokenizer that maps words to indices
/// and provides embeddings for financial domain vocabulary.
#[derive(Debug, Clone)]
pub struct FinancialTokenizer {
    vocab: HashMap<String, usize>,
    embeddings: Vec<Vec<f64>>,
    embed_dim: usize,
}

impl FinancialTokenizer {
    /// Create a new tokenizer with a predefined financial vocabulary.
    pub fn new(embed_dim: usize) -> Self {
        let mut vocab = HashMap::new();
        let financial_words = vec![
            // Sentiment words
            "bullish", "bearish", "positive", "negative", "neutral",
            "growth", "decline", "increase", "decrease", "stable",
            "strong", "weak", "exceeded", "missed", "beat",
            "optimistic", "pessimistic", "cautious", "confident", "uncertain",
            // Financial terms
            "revenue", "profit", "loss", "earnings", "margin",
            "dividend", "guidance", "forecast", "outlook", "target",
            "acquisition", "merger", "ipo", "buyback", "restructuring",
            "inflation", "interest", "rate", "yield", "spread",
            // Market terms
            "buy", "sell", "hold", "long", "short",
            "volume", "volatility", "momentum", "trend", "breakout",
            "support", "resistance", "overbought", "oversold", "consolidation",
            "rally", "correction", "crash", "recovery", "dip",
            // General
            "the", "a", "is", "was", "and", "or", "to", "of", "in", "for",
            "company", "market", "stock", "price", "trading",
            "quarter", "year", "annual", "report", "analyst",
            // Unknown token
            "<UNK>",
        ];

        for (i, word) in financial_words.iter().enumerate() {
            vocab.insert(word.to_string(), i);
        }

        // Initialize random embeddings (in practice these would be pre-trained)
        let mut rng = rand::thread_rng();
        let embeddings: Vec<Vec<f64>> = (0..financial_words.len())
            .map(|_| {
                (0..embed_dim)
                    .map(|_| rng.gen_range(-0.5..0.5))
                    .collect()
            })
            .collect();

        Self {
            vocab,
            embeddings,
            embed_dim,
        }
    }

    /// Tokenize a text string into token indices.
    pub fn tokenize(&self, text: &str) -> Vec<usize> {
        let unk_idx = self.vocab.get("<UNK>").copied().unwrap_or(0);
        text.to_lowercase()
            .split_whitespace()
            .map(|word| {
                let cleaned: String = word.chars().filter(|c| c.is_alphanumeric()).collect();
                *self.vocab.get(&cleaned).unwrap_or(&unk_idx)
            })
            .collect()
    }

    /// Get the embedding for a token index.
    pub fn get_embedding(&self, idx: usize) -> &[f64] {
        if idx < self.embeddings.len() {
            &self.embeddings[idx]
        } else {
            &self.embeddings[0]
        }
    }

    /// Get embeddings for a sequence of token indices.
    pub fn get_sequence_embeddings(&self, indices: &[usize]) -> Vec<Vec<f64>> {
        indices.iter().map(|&i| self.get_embedding(i).to_vec()).collect()
    }

    /// Vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Embedding dimension.
    pub fn embed_dim(&self) -> usize {
        self.embed_dim
    }
}

// ─── Positional Encoding ────────────────────────────────────────────

/// Generates sinusoidal positional encodings for transformer models.
#[derive(Debug, Clone)]
pub struct PositionalEncoding {
    max_len: usize,
    d_model: usize,
    encodings: Vec<Vec<f64>>,
}

impl PositionalEncoding {
    /// Create positional encodings using sine and cosine functions.
    pub fn new(max_len: usize, d_model: usize) -> Self {
        let mut encodings = vec![vec![0.0; d_model]; max_len];
        for pos in 0..max_len {
            for i in 0..d_model {
                let angle = pos as f64 / (10000.0_f64).powf(2.0 * (i / 2) as f64 / d_model as f64);
                encodings[pos][i] = if i % 2 == 0 { angle.sin() } else { angle.cos() };
            }
        }
        Self {
            max_len,
            d_model,
            encodings,
        }
    }

    /// Add positional encoding to a sequence of embeddings.
    pub fn encode(&self, embeddings: &[Vec<f64>]) -> Vec<Vec<f64>> {
        embeddings
            .iter()
            .enumerate()
            .map(|(pos, emb)| {
                if pos < self.max_len {
                    emb.iter()
                        .zip(self.encodings[pos].iter())
                        .map(|(e, p)| e + p)
                        .collect()
                } else {
                    emb.clone()
                }
            })
            .collect()
    }

    /// Maximum sequence length supported.
    pub fn max_len(&self) -> usize {
        self.max_len
    }

    /// Model dimension.
    pub fn d_model(&self) -> usize {
        self.d_model
    }
}

// ─── Scaled Dot-Product Attention ───────────────────────────────────

/// Implements scaled dot-product attention, the core building block
/// of transformer models.
#[derive(Debug, Clone)]
pub struct AttentionLayer {
    w_query: Vec<Vec<f64>>,
    w_key: Vec<Vec<f64>>,
    w_value: Vec<Vec<f64>>,
    d_k: usize,
}

impl AttentionLayer {
    /// Create a new attention layer with random weight initialization.
    pub fn new(d_model: usize, d_k: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (1.0 / d_model as f64).sqrt();

        let mut init_matrix = |rows: usize, cols: usize| -> Vec<Vec<f64>> {
            (0..rows)
                .map(|_| (0..cols).map(|_| rng.gen_range(-scale..scale)).collect())
                .collect()
        };

        Self {
            w_query: init_matrix(d_model, d_k),
            w_key: init_matrix(d_model, d_k),
            w_value: init_matrix(d_model, d_k),
            d_k,
        }
    }

    /// Project input through a weight matrix.
    fn project(&self, input: &[f64], weights: &[Vec<f64>]) -> Vec<f64> {
        let out_dim = weights[0].len();
        let mut output = vec![0.0; out_dim];
        for (i, w_row) in weights.iter().enumerate() {
            if i < input.len() {
                for (j, &w) in w_row.iter().enumerate() {
                    output[j] += input[i] * w;
                }
            }
        }
        output
    }

    /// Compute attention over a sequence of embeddings.
    /// Returns (attended output for each position, attention weights).
    pub fn forward(&self, sequence: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let seq_len = sequence.len();
        if seq_len == 0 {
            return (vec![], vec![]);
        }

        // Compute Q, K, V projections
        let queries: Vec<Vec<f64>> = sequence.iter().map(|s| self.project(s, &self.w_query)).collect();
        let keys: Vec<Vec<f64>> = sequence.iter().map(|s| self.project(s, &self.w_key)).collect();
        let values: Vec<Vec<f64>> = sequence.iter().map(|s| self.project(s, &self.w_value)).collect();

        let scale = (self.d_k as f64).sqrt();
        let mut outputs = Vec::with_capacity(seq_len);
        let mut all_weights = Vec::with_capacity(seq_len);

        for i in 0..seq_len {
            // Compute attention scores (with causal mask: only attend to positions <= i)
            let mut scores = Vec::with_capacity(i + 1);
            for j in 0..=i {
                let score = dot(&queries[i], &keys[j]) / scale;
                scores.push(score);
            }

            // Apply softmax to get attention weights
            let weights = softmax(&scores);

            // Compute weighted sum of values
            let d_v = values[0].len();
            let mut output = vec![0.0; d_v];
            for (j, &w) in weights.iter().enumerate() {
                for (k, &v) in values[j].iter().enumerate() {
                    output[k] += w * v;
                }
            }

            // Pad weights to full sequence length for visualization
            let mut full_weights = vec![0.0; seq_len];
            for (j, &w) in weights.iter().enumerate() {
                full_weights[j] = w;
            }

            outputs.push(output);
            all_weights.push(full_weights);
        }

        (outputs, all_weights)
    }

    /// Get the attention dimension.
    pub fn d_k(&self) -> usize {
        self.d_k
    }
}

// ─── Feed-Forward Network ───────────────────────────────────────────

/// A two-layer feed-forward network with ReLU activation,
/// as used in each transformer block.
#[derive(Debug, Clone)]
pub struct FeedForwardNetwork {
    w1: Vec<Vec<f64>>,
    b1: Vec<f64>,
    w2: Vec<Vec<f64>>,
    b2: Vec<f64>,
}

impl FeedForwardNetwork {
    /// Create a new FFN with given input and hidden dimensions.
    pub fn new(d_model: usize, d_ff: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale1 = (1.0 / d_model as f64).sqrt();
        let scale2 = (1.0 / d_ff as f64).sqrt();

        let w1: Vec<Vec<f64>> = (0..d_model)
            .map(|_| (0..d_ff).map(|_| rng.gen_range(-scale1..scale1)).collect())
            .collect();
        let b1: Vec<f64> = vec![0.0; d_ff];
        let w2: Vec<Vec<f64>> = (0..d_ff)
            .map(|_| (0..d_model).map(|_| rng.gen_range(-scale2..scale2)).collect())
            .collect();
        let b2: Vec<f64> = vec![0.0; d_model];

        Self { w1, b1, w2, b2 }
    }

    /// Forward pass: Linear -> ReLU -> Linear.
    pub fn forward(&self, input: &[f64]) -> Vec<f64> {
        // First linear layer
        let d_ff = self.b1.len();
        let mut hidden = vec![0.0; d_ff];
        for (i, w_row) in self.w1.iter().enumerate() {
            if i < input.len() {
                for (j, &w) in w_row.iter().enumerate() {
                    hidden[j] += input[i] * w;
                }
            }
        }
        for (j, h) in hidden.iter_mut().enumerate() {
            *h = relu(*h + self.b1[j]);
        }

        // Second linear layer
        let d_model = self.b2.len();
        let mut output = vec![0.0; d_model];
        for (i, w_row) in self.w2.iter().enumerate() {
            if i < hidden.len() {
                for (j, &w) in w_row.iter().enumerate() {
                    output[j] += hidden[i] * w;
                }
            }
        }
        for (j, o) in output.iter_mut().enumerate() {
            *o += self.b2[j];
        }

        output
    }
}

// ─── Sentiment Classifier ───────────────────────────────────────────

/// Classifies financial text into sentiment categories:
/// bullish (0), neutral (1), bearish (2).
#[derive(Debug, Clone)]
pub struct SentimentClassifier {
    weights: Vec<Vec<f64>>,
    bias: Vec<f64>,
    num_classes: usize,
}

/// Sentiment classification result.
#[derive(Debug, Clone)]
pub struct SentimentResult {
    pub label: String,
    pub confidence: f64,
    pub probabilities: Vec<f64>,
}

impl SentimentClassifier {
    /// Create a new sentiment classifier.
    pub fn new(input_dim: usize, num_classes: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (1.0 / input_dim as f64).sqrt();

        let weights: Vec<Vec<f64>> = (0..input_dim)
            .map(|_| (0..num_classes).map(|_| rng.gen_range(-scale..scale)).collect())
            .collect();
        let bias = vec![0.0; num_classes];

        Self {
            weights,
            bias,
            num_classes,
        }
    }

    /// Initialize with sentiment-aware weights using a keyword-based approach.
    pub fn new_with_financial_bias(input_dim: usize) -> Self {
        let num_classes = 3; // bullish, neutral, bearish
        let mut rng = rand::thread_rng();
        let scale = (1.0 / input_dim as f64).sqrt();

        let weights: Vec<Vec<f64>> = (0..input_dim)
            .map(|_| (0..num_classes).map(|_| rng.gen_range(-scale..scale)).collect())
            .collect();
        // Slight bias toward neutral for stability
        let bias = vec![0.0, 0.1, 0.0];

        Self {
            weights,
            bias,
            num_classes,
        }
    }

    /// Classify the attended representation into sentiment.
    pub fn classify(&self, representation: &[f64]) -> SentimentResult {
        let mut logits = vec![0.0; self.num_classes];
        for (i, w_row) in self.weights.iter().enumerate() {
            if i < representation.len() {
                for (j, &w) in w_row.iter().enumerate() {
                    logits[j] += representation[i] * w;
                }
            }
        }
        for (j, l) in logits.iter_mut().enumerate() {
            *l += self.bias[j];
        }

        let probabilities = softmax(&logits);
        let (max_idx, &max_prob) = probabilities
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        let label = match max_idx {
            0 => "bullish".to_string(),
            1 => "neutral".to_string(),
            2 => "bearish".to_string(),
            _ => "unknown".to_string(),
        };

        SentimentResult {
            label,
            confidence: max_prob,
            probabilities,
        }
    }

    /// Number of output classes.
    pub fn num_classes(&self) -> usize {
        self.num_classes
    }
}

// ─── GPT Financial Analyzer ────────────────────────────────────────

/// The main GPT-inspired financial text analyzer.
/// Combines tokenization, positional encoding, attention, and sentiment
/// classification into a single pipeline.
#[derive(Debug)]
pub struct GptFinancialAnalyzer {
    tokenizer: FinancialTokenizer,
    positional_encoding: PositionalEncoding,
    attention: AttentionLayer,
    ffn: FeedForwardNetwork,
    classifier: SentimentClassifier,
}

impl GptFinancialAnalyzer {
    /// Create a new GPT financial analyzer with given dimensions.
    pub fn new(embed_dim: usize, attention_dim: usize, ff_dim: usize, max_seq_len: usize) -> Self {
        Self {
            tokenizer: FinancialTokenizer::new(embed_dim),
            positional_encoding: PositionalEncoding::new(max_seq_len, embed_dim),
            attention: AttentionLayer::new(embed_dim, attention_dim),
            ffn: FeedForwardNetwork::new(attention_dim, ff_dim),
            classifier: SentimentClassifier::new_with_financial_bias(attention_dim),
        }
    }

    /// Analyze a financial text and return a sentiment result.
    pub fn analyze(&self, text: &str) -> SentimentResult {
        // Step 1: Tokenize
        let token_ids = self.tokenizer.tokenize(text);
        if token_ids.is_empty() {
            return SentimentResult {
                label: "neutral".to_string(),
                confidence: 1.0,
                probabilities: vec![0.0, 1.0, 0.0],
            };
        }

        // Step 2: Get embeddings
        let embeddings = self.tokenizer.get_sequence_embeddings(&token_ids);

        // Step 3: Add positional encoding
        let pos_embeddings = self.positional_encoding.encode(&embeddings);

        // Step 4: Apply attention
        let (attended, _weights) = self.attention.forward(&pos_embeddings);

        // Step 5: Use the last position's output (causal model)
        let last_output = attended.last().unwrap();

        // Step 6: Apply feed-forward network
        let ff_output = self.ffn.forward(last_output);

        // Step 7: Classify sentiment
        self.classifier.classify(&ff_output)
    }

    /// Analyze text and return attention weights for visualization.
    pub fn analyze_with_attention(&self, text: &str) -> (SentimentResult, Vec<Vec<f64>>, Vec<String>) {
        let words: Vec<String> = text
            .to_lowercase()
            .split_whitespace()
            .map(|s| s.to_string())
            .collect();

        let token_ids = self.tokenizer.tokenize(text);
        if token_ids.is_empty() {
            let result = SentimentResult {
                label: "neutral".to_string(),
                confidence: 1.0,
                probabilities: vec![0.0, 1.0, 0.0],
            };
            return (result, vec![], words);
        }

        let embeddings = self.tokenizer.get_sequence_embeddings(&token_ids);
        let pos_embeddings = self.positional_encoding.encode(&embeddings);
        let (attended, weights) = self.attention.forward(&pos_embeddings);
        let last_output = attended.last().unwrap();
        let ff_output = self.ffn.forward(last_output);
        let result = self.classifier.classify(&ff_output);

        (result, weights, words)
    }

    /// Get the tokenizer reference.
    pub fn tokenizer(&self) -> &FinancialTokenizer {
        &self.tokenizer
    }
}

// ─── Trading Signal Generator ───────────────────────────────────────

/// Generates trading signals by combining GPT sentiment analysis
/// with technical price/volume features.
#[derive(Debug, Clone)]
pub struct TradingSignalGenerator {
    sentiment_weight: f64,
    momentum_weight: f64,
    volatility_weight: f64,
    threshold: f64,
}

/// A trading signal with direction and strength.
#[derive(Debug, Clone)]
pub struct TradingSignal {
    pub direction: String,
    pub strength: f64,
    pub sentiment_score: f64,
    pub momentum_score: f64,
    pub volatility_score: f64,
    pub combined_score: f64,
}

impl TradingSignalGenerator {
    /// Create a new signal generator with configurable weights.
    pub fn new(sentiment_weight: f64, momentum_weight: f64, volatility_weight: f64, threshold: f64) -> Self {
        Self {
            sentiment_weight,
            momentum_weight,
            volatility_weight,
            threshold,
        }
    }

    /// Create a default signal generator with balanced weights.
    pub fn default_config() -> Self {
        Self {
            sentiment_weight: 0.4,
            momentum_weight: 0.35,
            volatility_weight: 0.25,
            threshold: 0.15,
        }
    }

    /// Convert a sentiment result to a numerical score in [-1, 1].
    pub fn sentiment_to_score(sentiment: &SentimentResult) -> f64 {
        // bullish=+1, neutral=0, bearish=-1, weighted by probabilities
        sentiment.probabilities[0] - sentiment.probabilities[2]
    }

    /// Compute momentum score from price history.
    pub fn compute_momentum(prices: &[f64], window: usize) -> f64 {
        if prices.len() < window + 1 {
            return 0.0;
        }
        let current = prices[prices.len() - 1];
        let past = prices[prices.len() - 1 - window];
        if past == 0.0 {
            return 0.0;
        }
        ((current - past) / past).clamp(-1.0, 1.0)
    }

    /// Compute volatility-adjusted score from price history.
    pub fn compute_volatility_score(prices: &[f64], window: usize) -> f64 {
        if prices.len() < window {
            return 0.0;
        }
        let recent = &prices[prices.len() - window..];
        let returns: Vec<f64> = recent
            .windows(2)
            .map(|w| {
                if w[0] == 0.0 {
                    0.0
                } else {
                    (w[1] - w[0]) / w[0]
                }
            })
            .collect();

        let vol = std_dev(&returns);
        // Higher volatility -> lower confidence in momentum signal
        // Return inverse volatility as a score, normalized
        if vol == 0.0 {
            0.5
        } else {
            (1.0 / (1.0 + vol * 100.0)).clamp(0.0, 1.0)
        }
    }

    /// Generate a trading signal from sentiment and price data.
    pub fn generate_signal(
        &self,
        sentiment: &SentimentResult,
        prices: &[f64],
        momentum_window: usize,
        volatility_window: usize,
    ) -> TradingSignal {
        let sentiment_score = Self::sentiment_to_score(sentiment);
        let momentum_score = Self::compute_momentum(prices, momentum_window);
        let volatility_score = Self::compute_volatility_score(prices, volatility_window);

        let combined = self.sentiment_weight * sentiment_score
            + self.momentum_weight * momentum_score
            + self.volatility_weight * (2.0 * volatility_score - 1.0);

        let direction = if combined > self.threshold {
            "BUY".to_string()
        } else if combined < -self.threshold {
            "SELL".to_string()
        } else {
            "HOLD".to_string()
        };

        let strength = combined.abs().clamp(0.0, 1.0);

        TradingSignal {
            direction,
            strength,
            sentiment_score,
            momentum_score,
            volatility_score,
            combined_score: combined,
        }
    }
}

// ─── Backtester ─────────────────────────────────────────────────────

/// Simple backtesting engine for evaluating GPT-based trading signals.
#[derive(Debug, Clone)]
pub struct Backtester {
    initial_capital: f64,
    position_size: f64,
}

/// Result of a backtest run.
#[derive(Debug, Clone)]
pub struct BacktestResult {
    pub total_return: f64,
    pub num_trades: usize,
    pub winning_trades: usize,
    pub losing_trades: usize,
    pub sharpe_ratio: f64,
    pub max_drawdown: f64,
    pub final_equity: f64,
}

impl Backtester {
    /// Create a new backtester with initial capital and position sizing.
    pub fn new(initial_capital: f64, position_size: f64) -> Self {
        Self {
            initial_capital,
            position_size,
        }
    }

    /// Run a backtest given signals and corresponding price data.
    /// `signals` contains direction strings ("BUY", "SELL", "HOLD").
    /// `prices` are closing prices aligned with the signals.
    pub fn run(&self, signals: &[String], prices: &[f64]) -> BacktestResult {
        let n = signals.len().min(prices.len());
        if n < 2 {
            return BacktestResult {
                total_return: 0.0,
                num_trades: 0,
                winning_trades: 0,
                losing_trades: 0,
                sharpe_ratio: 0.0,
                max_drawdown: 0.0,
                final_equity: self.initial_capital,
            };
        }

        let mut equity = self.initial_capital;
        let mut peak_equity = equity;
        let mut max_drawdown = 0.0_f64;
        let mut position: f64 = 0.0; // positive = long, negative = short
        let mut entry_price = 0.0_f64;
        let mut num_trades = 0;
        let mut winning = 0;
        let mut losing = 0;
        let mut returns = Vec::new();

        for i in 0..n - 1 {
            let signal = &signals[i];
            let current_price = prices[i];
            let next_price = prices[i + 1];

            // Close existing position if signal changes
            if position != 0.0
                && ((position > 0.0 && signal != "BUY") || (position < 0.0 && signal != "SELL"))
            {
                let pnl = position * (current_price - entry_price);
                equity += pnl;
                if pnl > 0.0 {
                    winning += 1;
                } else if pnl < 0.0 {
                    losing += 1;
                }
                position = 0.0;
            }

            // Open new position
            if position == 0.0 {
                if signal == "BUY" {
                    position = self.position_size / current_price;
                    entry_price = current_price;
                    num_trades += 1;
                } else if signal == "SELL" {
                    position = -(self.position_size / current_price);
                    entry_price = current_price;
                    num_trades += 1;
                }
            }

            // Track equity curve
            let unrealized = position * (next_price - entry_price);
            let current_equity = equity + unrealized;
            if current_equity > peak_equity {
                peak_equity = current_equity;
            }
            let drawdown = if peak_equity > 0.0 {
                (peak_equity - current_equity) / peak_equity
            } else {
                0.0
            };
            max_drawdown = max_drawdown.max(drawdown);

            // Track returns
            if equity > 0.0 {
                returns.push((next_price - current_price) / current_price * position.signum());
            }
        }

        // Close final position
        if position != 0.0 {
            let pnl = position * (prices[n - 1] - entry_price);
            equity += pnl;
            if pnl > 0.0 {
                winning += 1;
            } else if pnl < 0.0 {
                losing += 1;
            }
        }

        let total_return = (equity - self.initial_capital) / self.initial_capital;
        let sharpe_ratio = if !returns.is_empty() && std_dev(&returns) > 0.0 {
            mean(&returns) / std_dev(&returns) * (252.0_f64).sqrt()
        } else {
            0.0
        };

        BacktestResult {
            total_return,
            num_trades,
            winning_trades: winning,
            losing_trades: losing,
            sharpe_ratio,
            max_drawdown,
            final_equity: equity,
        }
    }
}

// ─── Bybit API Client ──────────────────────────────────────────────

/// Kline (candlestick) data from Bybit.
#[derive(Debug, Clone)]
pub struct Kline {
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub timestamp: u64,
}

/// Bybit V5 API client for fetching market data.
#[derive(Debug)]
pub struct BybitClient {
    base_url: String,
    client: reqwest::Client,
}

#[derive(Deserialize)]
struct BybitResponse {
    result: BybitResult,
}

#[derive(Deserialize)]
struct BybitResult {
    list: Vec<serde_json::Value>,
}

#[derive(Deserialize)]
struct BybitOrderbookResponse {
    result: BybitOrderbookResult,
}

#[derive(Deserialize)]
struct BybitOrderbookResult {
    #[serde(rename = "b")]
    bids: Vec<Vec<String>>,
    #[serde(rename = "a")]
    asks: Vec<Vec<String>>,
}

impl BybitClient {
    /// Create a new Bybit V5 API client.
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
        limit: usize,
    ) -> anyhow::Result<Vec<Kline>> {
        let url = format!(
            "{}/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
            self.base_url, symbol, interval, limit
        );

        let resp: BybitResponse = self.client.get(&url).send().await?.json().await?;

        let klines = resp
            .result
            .list
            .iter()
            .filter_map(|item| {
                let arr = item.as_array()?;
                if arr.len() >= 6 {
                    Some(Kline {
                        timestamp: arr[0].as_str()?.parse().ok()?,
                        open: arr[1].as_str()?.parse().ok()?,
                        high: arr[2].as_str()?.parse().ok()?,
                        low: arr[3].as_str()?.parse().ok()?,
                        close: arr[4].as_str()?.parse().ok()?,
                        volume: arr[5].as_str()?.parse().ok()?,
                    })
                } else {
                    None
                }
            })
            .collect();

        Ok(klines)
    }

    /// Fetch order book data. Returns (bids, asks) as Vec<(price, qty)>.
    pub async fn get_orderbook(
        &self,
        symbol: &str,
        limit: usize,
    ) -> anyhow::Result<(Vec<(f64, f64)>, Vec<(f64, f64)>)> {
        let url = format!(
            "{}/v5/market/orderbook?category=spot&symbol={}&limit={}",
            self.base_url, symbol, limit
        );

        let resp: BybitOrderbookResponse = self.client.get(&url).send().await?.json().await?;

        let parse_levels = |levels: &[Vec<String>]| -> Vec<(f64, f64)> {
            levels
                .iter()
                .filter_map(|level| {
                    if level.len() >= 2 {
                        Some((level[0].parse().ok()?, level[1].parse().ok()?))
                    } else {
                        None
                    }
                })
                .collect()
        };

        Ok((
            parse_levels(&resp.result.bids),
            parse_levels(&resp.result.asks),
        ))
    }
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Synthetic Data Generation ──────────────────────────────────────

/// Generate synthetic price data for testing.
pub fn generate_synthetic_prices(n: usize, start_price: f64, volatility: f64) -> Vec<f64> {
    let mut rng = rand::thread_rng();
    let mut prices = Vec::with_capacity(n);
    let mut price = start_price;

    for _ in 0..n {
        prices.push(price);
        let change = rng.gen_range(-volatility..volatility);
        price *= 1.0 + change;
        price = price.max(start_price * 0.5); // Floor at 50% of start
    }

    prices
}

/// Generate synthetic financial news texts for testing.
pub fn generate_synthetic_news() -> Vec<(&'static str, &'static str)> {
    vec![
        ("Company reported strong earnings growth exceeding analyst expectations with revenue increase", "bullish"),
        ("The stock market rally continued as positive guidance boosted investor confidence", "bullish"),
        ("Revenue exceeded forecast targets with strong margin growth and optimistic outlook", "bullish"),
        ("Earnings beat expectations with strong revenue growth and raised annual guidance", "bullish"),
        ("The company announced a major acquisition to strengthen market position", "bullish"),
        ("Trading volume increased significantly as buy interest in the stock surged", "bullish"),
        ("Market conditions remain stable with neutral trading volume and steady prices", "neutral"),
        ("The company maintained its quarterly dividend and held guidance unchanged", "neutral"),
        ("Analyst report indicates the stock is fairly valued at current price levels", "neutral"),
        ("Trading was in consolidation range with volume near the annual average", "neutral"),
        ("The company reported a significant decline in quarterly earnings and revenue", "bearish"),
        ("Weak guidance and missed targets led to a sharp sell-off in the stock", "bearish"),
        ("Revenue missed expectations with declining margin and pessimistic outlook", "bearish"),
        ("The market correction deepened as negative sentiment spread across the trading floor", "bearish"),
        ("Loss widened as the company struggled with weak demand and restructuring costs", "bearish"),
        ("Short interest increased sharply as bearish sentiment dominated the stock", "bearish"),
    ]
}

// ─── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax() {
        let input = vec![1.0, 2.0, 3.0];
        let result = softmax(&input);
        let sum: f64 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        assert!(result[2] > result[1]);
        assert!(result[1] > result[0]);
    }

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-10);
        assert!(sigmoid(10.0) > 0.99);
        assert!(sigmoid(-10.0) < 0.01);
    }

    #[test]
    fn test_relu() {
        assert_eq!(relu(5.0), 5.0);
        assert_eq!(relu(-3.0), 0.0);
        assert_eq!(relu(0.0), 0.0);
    }

    #[test]
    fn test_mean_and_std() {
        let data = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        assert!((mean(&data) - 6.0).abs() < 1e-10);
        assert!(std_dev(&data) > 0.0);
    }

    #[test]
    fn test_tokenizer() {
        let tokenizer = FinancialTokenizer::new(16);
        let tokens = tokenizer.tokenize("revenue growth strong bullish");
        assert_eq!(tokens.len(), 4);
        assert!(tokenizer.vocab_size() > 0);
        assert_eq!(tokenizer.embed_dim(), 16);
    }

    #[test]
    fn test_positional_encoding() {
        let pe = PositionalEncoding::new(100, 16);
        let embeddings = vec![vec![0.0; 16]; 5];
        let encoded = pe.encode(&embeddings);
        assert_eq!(encoded.len(), 5);
        assert_eq!(encoded[0].len(), 16);
        // Positional encoding should make each position different
        assert_ne!(encoded[0], encoded[1]);
    }

    #[test]
    fn test_attention_layer() {
        let attention = AttentionLayer::new(16, 8);
        let sequence = vec![vec![0.1; 16]; 3];
        let (outputs, weights) = attention.forward(&sequence);
        assert_eq!(outputs.len(), 3);
        assert_eq!(weights.len(), 3);
        // Attention weights should sum to ~1 for each position
        for w in &weights {
            let sum: f64 = w.iter().sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_ffn() {
        let ffn = FeedForwardNetwork::new(16, 32);
        let input = vec![0.5; 16];
        let output = ffn.forward(&input);
        assert_eq!(output.len(), 16);
    }

    #[test]
    fn test_sentiment_classifier() {
        let classifier = SentimentClassifier::new(8, 3);
        let input = vec![0.5; 8];
        let result = classifier.classify(&input);
        assert!(result.confidence > 0.0);
        assert_eq!(result.probabilities.len(), 3);
        let prob_sum: f64 = result.probabilities.iter().sum();
        assert!((prob_sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_gpt_analyzer() {
        let analyzer = GptFinancialAnalyzer::new(16, 8, 32, 100);
        let result = analyzer.analyze("strong revenue growth exceeded expectations");
        assert!(!result.label.is_empty());
        assert!(result.confidence > 0.0);
    }

    #[test]
    fn test_gpt_analyzer_with_attention() {
        let analyzer = GptFinancialAnalyzer::new(16, 8, 32, 100);
        let (result, weights, words) =
            analyzer.analyze_with_attention("earnings beat strong guidance");
        assert!(!result.label.is_empty());
        assert!(!weights.is_empty());
        assert!(!words.is_empty());
    }

    #[test]
    fn test_trading_signal_generator() {
        let generator = TradingSignalGenerator::default_config();
        let sentiment = SentimentResult {
            label: "bullish".to_string(),
            confidence: 0.8,
            probabilities: vec![0.8, 0.15, 0.05],
        };
        let prices = generate_synthetic_prices(50, 100.0, 0.02);
        let signal = generator.generate_signal(&sentiment, &prices, 10, 20);
        assert!(!signal.direction.is_empty());
        assert!(signal.strength >= 0.0 && signal.strength <= 1.0);
    }

    #[test]
    fn test_momentum_calculation() {
        let prices = vec![100.0, 101.0, 102.0, 103.0, 104.0, 105.0];
        let momentum = TradingSignalGenerator::compute_momentum(&prices, 5);
        assert!(momentum > 0.0); // prices are rising
    }

    #[test]
    fn test_volatility_score() {
        let stable_prices = vec![100.0, 100.1, 100.0, 100.1, 100.0];
        let volatile_prices = vec![100.0, 110.0, 90.0, 115.0, 85.0];
        let stable_vol = TradingSignalGenerator::compute_volatility_score(&stable_prices, 5);
        let volatile_vol = TradingSignalGenerator::compute_volatility_score(&volatile_prices, 5);
        assert!(stable_vol > volatile_vol);
    }

    #[test]
    fn test_backtester() {
        let backtester = Backtester::new(10000.0, 1000.0);
        let signals = vec![
            "BUY".to_string(),
            "BUY".to_string(),
            "SELL".to_string(),
            "SELL".to_string(),
            "HOLD".to_string(),
        ];
        let prices = vec![100.0, 105.0, 103.0, 98.0, 100.0];
        let result = backtester.run(&signals, &prices);
        assert!(result.num_trades > 0);
    }

    #[test]
    fn test_synthetic_data_generation() {
        let prices = generate_synthetic_prices(100, 50000.0, 0.01);
        assert_eq!(prices.len(), 100);
        assert!(prices[0] > 0.0);
    }

    #[test]
    fn test_synthetic_news() {
        let news = generate_synthetic_news();
        assert!(!news.is_empty());
        let bullish_count = news.iter().filter(|(_, l)| *l == "bullish").count();
        let bearish_count = news.iter().filter(|(_, l)| *l == "bearish").count();
        assert!(bullish_count > 0);
        assert!(bearish_count > 0);
    }

    #[test]
    fn test_mae_and_rmse() {
        let actual = vec![1.0, 2.0, 3.0];
        let predicted = vec![1.1, 2.2, 2.8];
        assert!(mae(&actual, &predicted) > 0.0);
        assert!(rmse(&actual, &predicted) > 0.0);
        assert!(rmse(&actual, &predicted) >= mae(&actual, &predicted));
    }

    #[test]
    fn test_empty_inputs() {
        let analyzer = GptFinancialAnalyzer::new(16, 8, 32, 100);
        let result = analyzer.analyze("");
        assert_eq!(result.label, "neutral");

        assert_eq!(mean(&[]), 0.0);
        assert_eq!(std_dev(&[]), 0.0);
        assert_eq!(mae(&[], &[]), 0.0);
    }
}
