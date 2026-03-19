use ndarray::Array1;
use rand::Rng;
use serde::Deserialize;
use std::collections::HashMap;

// ─── Aspect Categories ────────────────────────────────────────────

/// Financial aspect categories used for classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AspectCategory {
    Revenue,
    Profitability,
    Growth,
    Risk,
    Operations,
    Valuation,
    Macro,
    Technology,
    Adoption,
    Regulation,
    Tokenomics,
}

impl AspectCategory {
    pub fn as_str(&self) -> &'static str {
        match self {
            AspectCategory::Revenue => "revenue",
            AspectCategory::Profitability => "profitability",
            AspectCategory::Growth => "growth",
            AspectCategory::Risk => "risk",
            AspectCategory::Operations => "operations",
            AspectCategory::Valuation => "valuation",
            AspectCategory::Macro => "macro",
            AspectCategory::Technology => "technology",
            AspectCategory::Adoption => "adoption",
            AspectCategory::Regulation => "regulation",
            AspectCategory::Tokenomics => "tokenomics",
        }
    }
}

impl std::fmt::Display for AspectCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

// ─── Aspect Extraction ────────────────────────────────────────────

/// A detected aspect within a text.
#[derive(Debug, Clone)]
pub struct DetectedAspect {
    pub category: AspectCategory,
    pub keyword: String,
    pub position: usize,
}

/// Dictionary-based aspect extractor for financial text.
///
/// Maps keywords to aspect categories and identifies them in input text.
#[derive(Debug)]
pub struct AspectExtractor {
    dictionary: HashMap<String, AspectCategory>,
}

impl AspectExtractor {
    /// Create a new extractor with the default financial aspect dictionary.
    pub fn new() -> Self {
        let mut dictionary = HashMap::new();

        // Revenue / Sales
        for kw in &[
            "revenue", "sales", "income", "turnover", "top line", "bookings",
        ] {
            dictionary.insert(kw.to_string(), AspectCategory::Revenue);
        }

        // Profitability
        for kw in &[
            "margin",
            "margins",
            "profit",
            "earnings",
            "eps",
            "ebitda",
            "net income",
            "operating income",
            "bottom line",
            "cost",
            "costs",
            "expense",
            "expenses",
        ] {
            dictionary.insert(kw.to_string(), AspectCategory::Profitability);
        }

        // Growth
        for kw in &[
            "growth",
            "expansion",
            "market share",
            "user growth",
            "subscriber",
            "subscribers",
            "acquisition",
            "scale",
            "scaling",
        ] {
            dictionary.insert(kw.to_string(), AspectCategory::Growth);
        }

        // Risk
        for kw in &[
            "debt",
            "leverage",
            "risk",
            "volatility",
            "default",
            "downgrade",
            "covenant",
            "liability",
            "liabilities",
        ] {
            dictionary.insert(kw.to_string(), AspectCategory::Risk);
        }

        // Operations
        for kw in &[
            "supply chain",
            "operations",
            "efficiency",
            "capex",
            "workforce",
            "headcount",
            "production",
            "logistics",
            "inventory",
        ] {
            dictionary.insert(kw.to_string(), AspectCategory::Operations);
        }

        // Valuation
        for kw in &[
            "valuation",
            "price target",
            "fair value",
            "multiple",
            "pe ratio",
            "overvalued",
            "undervalued",
        ] {
            dictionary.insert(kw.to_string(), AspectCategory::Valuation);
        }

        // Macro
        for kw in &[
            "interest rate",
            "interest rates",
            "inflation",
            "gdp",
            "recession",
            "monetary policy",
            "fiscal policy",
            "unemployment",
            "fed",
            "central bank",
        ] {
            dictionary.insert(kw.to_string(), AspectCategory::Macro);
        }

        // Technology (crypto-relevant)
        for kw in &[
            "protocol",
            "upgrade",
            "blockchain",
            "smart contract",
            "scalability",
            "throughput",
            "latency",
            "consensus",
            "layer 2",
            "security audit",
        ] {
            dictionary.insert(kw.to_string(), AspectCategory::Technology);
        }

        // Adoption (crypto-relevant)
        for kw in &[
            "adoption",
            "institutional",
            "partnership",
            "integration",
            "merchant",
            "payment",
            "wallet",
            "active users",
        ] {
            dictionary.insert(kw.to_string(), AspectCategory::Adoption);
        }

        // Regulation (crypto-relevant)
        for kw in &[
            "regulation",
            "regulatory",
            "compliance",
            "sec",
            "ban",
            "legal",
            "lawsuit",
            "enforcement",
            "license",
        ] {
            dictionary.insert(kw.to_string(), AspectCategory::Regulation);
        }

        // Tokenomics (crypto-relevant)
        for kw in &[
            "tokenomics",
            "supply",
            "staking",
            "burn",
            "halving",
            "emission",
            "yield",
            "apy",
            "tvl",
        ] {
            dictionary.insert(kw.to_string(), AspectCategory::Tokenomics);
        }

        Self { dictionary }
    }

    /// Create an extractor with a custom dictionary.
    pub fn with_dictionary(dictionary: HashMap<String, AspectCategory>) -> Self {
        Self { dictionary }
    }

    /// Extract aspects from the given text.
    ///
    /// Performs case-insensitive matching against the aspect dictionary.
    pub fn extract(&self, text: &str) -> Vec<DetectedAspect> {
        let lower = text.to_lowercase();
        let mut aspects = Vec::new();

        // Sort keywords by length (longest first) to prefer multi-word matches
        let mut keywords: Vec<(&String, &AspectCategory)> = self.dictionary.iter().collect();
        keywords.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

        let mut matched_positions: Vec<(usize, usize)> = Vec::new();

        for (keyword, category) in &keywords {
            let kw_lower = keyword.to_lowercase();
            let mut search_start = 0;
            while let Some(pos) = lower[search_start..].find(&kw_lower) {
                let abs_pos = search_start + pos;
                let end_pos = abs_pos + kw_lower.len();

                // Check that this position does not overlap with an already matched span
                let overlaps = matched_positions
                    .iter()
                    .any(|&(s, e)| abs_pos < e && end_pos > s);

                if !overlaps {
                    aspects.push(DetectedAspect {
                        category: **category,
                        keyword: keyword.to_string(),
                        position: abs_pos,
                    });
                    matched_positions.push((abs_pos, end_pos));
                }

                search_start = abs_pos + 1;
                if search_start >= lower.len() {
                    break;
                }
            }
        }

        // Sort by position for natural ordering
        aspects.sort_by_key(|a| a.position);
        aspects
    }

    /// Return the number of keywords in the dictionary.
    pub fn dictionary_size(&self) -> usize {
        self.dictionary.len()
    }
}

impl Default for AspectExtractor {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Sentiment Scoring ────────────────────────────────────────────

/// Lexicon-based sentiment scorer with negation and intensity handling.
///
/// Scores the sentiment expressed toward a detected aspect by examining
/// words in a context window around the aspect keyword.
#[derive(Debug)]
pub struct SentimentScorer {
    positive_words: HashMap<String, f64>,
    negative_words: HashMap<String, f64>,
    negation_words: Vec<String>,
    intensifiers: HashMap<String, f64>,
    context_window: usize,
}

impl SentimentScorer {
    /// Create a new scorer with the default financial sentiment lexicon.
    pub fn new() -> Self {
        let mut positive_words = HashMap::new();
        for (word, score) in &[
            ("record", 0.8),
            ("beat", 0.7),
            ("exceeded", 0.75),
            ("strong", 0.6),
            ("grew", 0.6),
            ("growth", 0.5),
            ("improved", 0.65),
            ("positive", 0.5),
            ("surpassed", 0.7),
            ("outperformed", 0.7),
            ("robust", 0.6),
            ("accelerating", 0.7),
            ("raised", 0.6),
            ("upgraded", 0.65),
            ("bullish", 0.7),
            ("gains", 0.5),
            ("higher", 0.4),
            ("increased", 0.5),
            ("momentum", 0.5),
            ("optimistic", 0.6),
            ("resilient", 0.5),
            ("upbeat", 0.6),
            ("expanded", 0.55),
            ("innovative", 0.5),
            ("attractive", 0.5),
            ("solid", 0.5),
        ] {
            positive_words.insert(word.to_string(), *score);
        }

        let mut negative_words = HashMap::new();
        for (word, score) in &[
            ("decline", -0.7),
            ("declined", -0.7),
            ("missed", -0.7),
            ("weak", -0.6),
            ("fell", -0.6),
            ("below", -0.4),
            ("loss", -0.7),
            ("losses", -0.7),
            ("negative", -0.5),
            ("deteriorated", -0.75),
            ("underperformed", -0.7),
            ("downgraded", -0.65),
            ("bearish", -0.7),
            ("lower", -0.4),
            ("decreased", -0.5),
            ("contraction", -0.6),
            ("compressed", -0.5),
            ("concern", -0.4),
            ("concerning", -0.5),
            ("risk", -0.3),
            ("disappointing", -0.7),
            ("slowed", -0.5),
            ("headwinds", -0.5),
            ("volatile", -0.4),
            ("breach", -0.8),
            ("penalty", -0.7),
            ("lawsuit", -0.6),
            ("ban", -0.8),
            ("hack", -0.9),
            ("exploit", -0.85),
            ("warning", -0.5),
        ] {
            negative_words.insert(word.to_string(), *score);
        }

        let negation_words = vec![
            "not".to_string(),
            "no".to_string(),
            "never".to_string(),
            "neither".to_string(),
            "nor".to_string(),
            "hardly".to_string(),
            "barely".to_string(),
            "didn't".to_string(),
            "doesn't".to_string(),
            "wasn't".to_string(),
            "weren't".to_string(),
            "won't".to_string(),
            "couldn't".to_string(),
            "shouldn't".to_string(),
            "failed".to_string(),
        ];

        let mut intensifiers = HashMap::new();
        for (word, mult) in &[
            ("significantly", 1.5),
            ("substantially", 1.4),
            ("dramatically", 1.6),
            ("sharply", 1.5),
            ("very", 1.3),
            ("extremely", 1.6),
            ("slightly", 0.5),
            ("marginally", 0.4),
            ("modestly", 0.6),
            ("somewhat", 0.7),
        ] {
            intensifiers.insert(word.to_string(), *mult);
        }

        Self {
            positive_words,
            negative_words,
            negation_words,
            intensifiers,
            context_window: 10,
        }
    }

    /// Set the context window size (number of tokens on each side of the aspect).
    pub fn with_context_window(mut self, window: usize) -> Self {
        self.context_window = window;
        self
    }

    /// Score the sentiment toward a specific aspect in the text.
    ///
    /// Returns a score in [-1.0, 1.0].
    pub fn score_aspect(&self, text: &str, aspect_position: usize) -> f64 {
        let tokens: Vec<&str> = text.split_whitespace().collect();

        // Find token index closest to the character position
        let mut char_pos = 0;
        let mut aspect_token_idx = 0;
        for (i, token) in tokens.iter().enumerate() {
            if char_pos >= aspect_position {
                aspect_token_idx = i;
                break;
            }
            char_pos += token.len() + 1; // +1 for space
            aspect_token_idx = i;
        }

        // Define context window
        let start = aspect_token_idx.saturating_sub(self.context_window);
        let end = (aspect_token_idx + self.context_window + 1).min(tokens.len());

        let mut score = 0.0;
        let mut has_negation = false;
        let mut intensity_mult = 1.0;
        let mut word_count = 0;

        for i in start..end {
            let word = tokens[i].to_lowercase();
            // Remove punctuation for matching
            let clean: String = word.chars().filter(|c| c.is_alphanumeric() || *c == '\'').collect();

            if self.negation_words.contains(&clean) {
                has_negation = true;
                continue;
            }

            if let Some(&mult) = self.intensifiers.get(&clean) {
                intensity_mult = mult;
                continue;
            }

            if let Some(&s) = self.positive_words.get(&clean) {
                score += s * intensity_mult;
                word_count += 1;
                intensity_mult = 1.0; // Reset after use
            } else if let Some(&s) = self.negative_words.get(&clean) {
                score += s * intensity_mult;
                word_count += 1;
                intensity_mult = 1.0;
            }
        }

        // Normalize
        if word_count > 0 {
            score /= word_count as f64;
        }

        // Apply negation
        if has_negation {
            score = -score;
        }

        // Clamp to [-1, 1]
        score.clamp(-1.0, 1.0)
    }
}

impl Default for SentimentScorer {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Aspect Sentiment Result ──────────────────────────────────────

/// The sentiment result for a single aspect.
#[derive(Debug, Clone)]
pub struct AspectSentiment {
    pub category: AspectCategory,
    pub keyword: String,
    pub score: f64, // [-1.0, 1.0]
}

impl AspectSentiment {
    /// Human-readable polarity label.
    pub fn polarity(&self) -> &'static str {
        if self.score > 0.1 {
            "positive"
        } else if self.score < -0.1 {
            "negative"
        } else {
            "neutral"
        }
    }
}

// ─── Aspect Sentiment Analyzer ────────────────────────────────────

/// Combined pipeline that extracts aspects and scores their sentiment.
#[derive(Debug)]
pub struct AspectSentimentAnalyzer {
    extractor: AspectExtractor,
    scorer: SentimentScorer,
}

impl AspectSentimentAnalyzer {
    pub fn new() -> Self {
        Self {
            extractor: AspectExtractor::new(),
            scorer: SentimentScorer::new(),
        }
    }

    pub fn with_components(extractor: AspectExtractor, scorer: SentimentScorer) -> Self {
        Self { extractor, scorer }
    }

    /// Analyze a single text and return aspect-sentiment pairs.
    pub fn analyze(&self, text: &str) -> Vec<AspectSentiment> {
        let aspects = self.extractor.extract(text);
        aspects
            .into_iter()
            .map(|a| {
                let score = self.scorer.score_aspect(text, a.position);
                AspectSentiment {
                    category: a.category,
                    keyword: a.keyword,
                    score,
                }
            })
            .collect()
    }

    /// Analyze multiple documents and aggregate sentiment by aspect category.
    ///
    /// Returns a map from aspect category to average sentiment score.
    pub fn analyze_batch(&self, texts: &[&str]) -> HashMap<AspectCategory, f64> {
        let mut category_scores: HashMap<AspectCategory, Vec<f64>> = HashMap::new();

        for text in texts {
            let results = self.analyze(text);
            for result in results {
                category_scores
                    .entry(result.category)
                    .or_default()
                    .push(result.score);
            }
        }

        category_scores
            .into_iter()
            .map(|(cat, scores)| {
                let avg = scores.iter().sum::<f64>() / scores.len() as f64;
                (cat, avg)
            })
            .collect()
    }
}

impl Default for AspectSentimentAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Trading Signal Generator ─────────────────────────────────────

/// Generates trading signals from aspect-level sentiment scores.
///
/// Supports configurable aspect weights, exponential smoothing,
/// and threshold-based signal generation.
#[derive(Debug)]
pub struct TradingSignalGenerator {
    aspect_weights: HashMap<AspectCategory, f64>,
    smoothing_factor: f64, // lambda for EMA
    prev_signal: f64,
    signal_history: Vec<f64>,
}

impl TradingSignalGenerator {
    /// Create a new signal generator with default equal weights.
    pub fn new(smoothing_factor: f64) -> Self {
        let mut aspect_weights = HashMap::new();
        aspect_weights.insert(AspectCategory::Revenue, 1.0);
        aspect_weights.insert(AspectCategory::Profitability, 1.0);
        aspect_weights.insert(AspectCategory::Growth, 1.0);
        aspect_weights.insert(AspectCategory::Risk, 1.0);
        aspect_weights.insert(AspectCategory::Operations, 0.7);
        aspect_weights.insert(AspectCategory::Valuation, 0.8);
        aspect_weights.insert(AspectCategory::Macro, 0.6);
        aspect_weights.insert(AspectCategory::Technology, 0.9);
        aspect_weights.insert(AspectCategory::Adoption, 0.9);
        aspect_weights.insert(AspectCategory::Regulation, 0.8);
        aspect_weights.insert(AspectCategory::Tokenomics, 0.7);

        Self {
            aspect_weights,
            smoothing_factor,
            prev_signal: 0.0,
            signal_history: Vec::new(),
        }
    }

    /// Set custom aspect weights.
    pub fn with_weights(mut self, weights: HashMap<AspectCategory, f64>) -> Self {
        self.aspect_weights = weights;
        self
    }

    /// Generate a trading signal from aspect-level sentiment scores.
    ///
    /// Returns a smoothed signal in [-1, 1] where positive = bullish.
    pub fn generate_signal(
        &mut self,
        aspect_scores: &HashMap<AspectCategory, f64>,
    ) -> f64 {
        let mut weighted_sum = 0.0;
        let mut total_weight = 0.0;

        for (category, score) in aspect_scores {
            let weight = self.aspect_weights.get(category).copied().unwrap_or(0.5);
            weighted_sum += weight * score;
            total_weight += weight;
        }

        let raw_signal = if total_weight > 0.0 {
            weighted_sum / total_weight
        } else {
            0.0
        };

        // Exponential smoothing
        let smoothed = self.smoothing_factor * raw_signal
            + (1.0 - self.smoothing_factor) * self.prev_signal;
        self.prev_signal = smoothed;
        self.signal_history.push(smoothed);

        smoothed
    }

    /// Interpret the signal as a trading action.
    pub fn signal_to_action(signal: f64, buy_threshold: f64, sell_threshold: f64) -> &'static str {
        if signal > buy_threshold {
            "BUY"
        } else if signal < sell_threshold {
            "SELL"
        } else {
            "HOLD"
        }
    }

    /// Return the signal history.
    pub fn history(&self) -> &[f64] {
        &self.signal_history
    }

    /// Reset the signal generator state.
    pub fn reset(&mut self) {
        self.prev_signal = 0.0;
        self.signal_history.clear();
    }
}

// ─── Aspect Sentiment Classifier (Logistic Regression) ────────────

/// Multi-class logistic regression for aspect sentiment classification.
///
/// Predicts sentiment polarity (positive / neutral / negative) given
/// a feature vector derived from the text around an aspect.
///
/// Features: [positive_word_count, negative_word_count, negation_present,
///            intensity_mean, aspect_position_ratio]
#[derive(Debug)]
pub struct AspectSentimentClassifier {
    weights_pos: Array1<f64>,
    weights_neg: Array1<f64>,
    bias_pos: f64,
    bias_neg: f64,
    learning_rate: f64,
    num_features: usize,
}

impl AspectSentimentClassifier {
    pub fn new(num_features: usize, learning_rate: f64) -> Self {
        let mut rng = rand::thread_rng();
        let weights_pos =
            Array1::from_vec((0..num_features).map(|_| rng.gen_range(-0.1..0.1)).collect());
        let weights_neg =
            Array1::from_vec((0..num_features).map(|_| rng.gen_range(-0.1..0.1)).collect());
        Self {
            weights_pos,
            weights_neg,
            bias_pos: 0.0,
            bias_neg: 0.0,
            learning_rate,
            num_features,
        }
    }

    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    /// Predict sentiment probabilities: (P(positive), P(negative)).
    /// P(neutral) = 1 - P(positive) - P(negative).
    pub fn predict_proba(&self, features: &[f64]) -> (f64, f64, f64) {
        assert_eq!(features.len(), self.num_features);
        let x = Array1::from_vec(features.to_vec());
        let p_pos = Self::sigmoid(self.weights_pos.dot(&x) + self.bias_pos);
        let p_neg = Self::sigmoid(self.weights_neg.dot(&x) + self.bias_neg);

        // Normalize so they sum to 1
        let total = p_pos + p_neg + 0.5; // 0.5 as neutral base
        (p_pos / total, p_neg / total, 0.5 / total)
    }

    /// Predict sentiment label: "positive", "negative", or "neutral".
    pub fn predict(&self, features: &[f64]) -> (&'static str, f64) {
        let (p_pos, p_neg, p_neu) = self.predict_proba(features);
        if p_pos >= p_neg && p_pos >= p_neu {
            ("positive", p_pos)
        } else if p_neg >= p_pos && p_neg >= p_neu {
            ("negative", p_neg)
        } else {
            ("neutral", p_neu)
        }
    }

    /// Compute the continuous sentiment score: P(positive) - P(negative).
    pub fn score(&self, features: &[f64]) -> f64 {
        let (p_pos, p_neg, _) = self.predict_proba(features);
        p_pos - p_neg
    }

    /// Train on labeled data.
    /// Labels: 1.0 = positive, 0.0 = negative, 0.5 = neutral.
    pub fn train(&mut self, data: &[(Vec<f64>, f64)], epochs: usize) {
        for _ in 0..epochs {
            for (features, label) in data {
                let x = Array1::from_vec(features.clone());

                // Target for positive head
                let target_pos = if *label > 0.7 { 1.0 } else { 0.0 };
                let pred_pos = Self::sigmoid(self.weights_pos.dot(&x) + self.bias_pos);
                let error_pos = pred_pos - target_pos;
                for j in 0..self.num_features {
                    self.weights_pos[j] -= self.learning_rate * error_pos * x[j];
                }
                self.bias_pos -= self.learning_rate * error_pos;

                // Target for negative head
                let target_neg = if *label < 0.3 { 1.0 } else { 0.0 };
                let pred_neg = Self::sigmoid(self.weights_neg.dot(&x) + self.bias_neg);
                let error_neg = pred_neg - target_neg;
                for j in 0..self.num_features {
                    self.weights_neg[j] -= self.learning_rate * error_neg * x[j];
                }
                self.bias_neg -= self.learning_rate * error_neg;
            }
        }
    }

    /// Evaluate accuracy on test data.
    pub fn accuracy(&self, data: &[(Vec<f64>, f64)]) -> f64 {
        let correct = data
            .iter()
            .filter(|(features, label)| {
                let (pred_label, _) = self.predict(features);
                let true_label = if *label > 0.7 {
                    "positive"
                } else if *label < 0.3 {
                    "negative"
                } else {
                    "neutral"
                };
                pred_label == true_label
            })
            .count();
        correct as f64 / data.len() as f64
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
pub struct TickerResult {
    pub list: Vec<TickerItem>,
}

#[derive(Debug, Deserialize)]
pub struct TickerItem {
    pub symbol: String,
    #[serde(rename = "lastPrice")]
    pub last_price: String,
    #[serde(rename = "volume24h")]
    pub volume_24h: String,
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

    /// Fetch current ticker data.
    pub async fn get_ticker(&self, symbol: &str) -> anyhow::Result<(f64, f64)> {
        let url = format!(
            "{}/v5/market/tickers?category=spot&symbol={}",
            self.base_url, symbol
        );
        let resp: BybitResponse<TickerResult> = self.client.get(&url).send().await?.json().await?;

        if let Some(item) = resp.result.list.first() {
            let price = item.last_price.parse().unwrap_or(0.0);
            let volume = item.volume_24h.parse().unwrap_or(0.0);
            Ok((price, volume))
        } else {
            anyhow::bail!("No ticker data returned for {}", symbol)
        }
    }
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Synthetic Data Generation ─────────────────────────────────────

/// Generate synthetic financial text samples for testing.
pub fn generate_synthetic_texts(n: usize) -> Vec<String> {
    let mut rng = rand::thread_rng();
    let templates = vec![
        "Company reported {adj} revenue {dir} driven by {reason}.",
        "Profit margins {dir} as {reason} impacted the bottom line.",
        "The {adj} growth in market share {verb} expectations.",
        "Supply chain {adj2} led to {adj} operational efficiency.",
        "Debt levels are {adj2} following the recent {event}.",
        "Interest rates {dir} putting {adj2} pressure on valuations.",
        "The protocol upgrade {verb} blockchain throughput by 50 percent.",
        "Institutional adoption {dir} with major banks {verb} integration.",
        "Regulatory compliance {adj2} remain a {adj} concern for the sector.",
        "Staking yield {dir} as tokenomics changes {verb} supply dynamics.",
    ];

    let positive_adj = ["record", "strong", "robust", "impressive", "solid"];
    let negative_adj = ["weak", "disappointing", "poor", "concerning", "volatile"];
    let positive_dir = ["grew", "surpassed", "increased", "accelerated", "expanded"];
    let negative_dir = ["declined", "missed", "fell", "slowed", "contracted"];
    let positive_verb = ["exceeded", "surpassed", "beat", "outperformed"];
    let negative_verb = ["missed", "underperformed", "failed", "disappointed"];
    let adj2_pos = ["improvements", "stabilized", "normalized", "favorable"];
    let adj2_neg = ["challenges", "concerning", "elevated", "persistent"];
    let reasons = [
        "strong demand",
        "cost optimization",
        "market expansion",
        "new product launches",
        "rising costs",
        "supply disruptions",
        "competitive pressure",
        "currency headwinds",
    ];
    let events = [
        "restructuring",
        "acquisition",
        "divestiture",
        "refinancing",
        "issuance",
    ];

    let mut texts = Vec::with_capacity(n);
    for _ in 0..n {
        let template = templates[rng.gen_range(0..templates.len())];
        let is_positive = rng.gen_bool(0.5);

        let text = template
            .replace(
                "{adj}",
                if is_positive {
                    positive_adj[rng.gen_range(0..positive_adj.len())]
                } else {
                    negative_adj[rng.gen_range(0..negative_adj.len())]
                },
            )
            .replace(
                "{dir}",
                if is_positive {
                    positive_dir[rng.gen_range(0..positive_dir.len())]
                } else {
                    negative_dir[rng.gen_range(0..negative_dir.len())]
                },
            )
            .replace(
                "{verb}",
                if is_positive {
                    positive_verb[rng.gen_range(0..positive_verb.len())]
                } else {
                    negative_verb[rng.gen_range(0..negative_verb.len())]
                },
            )
            .replace(
                "{adj2}",
                if is_positive {
                    adj2_pos[rng.gen_range(0..adj2_pos.len())]
                } else {
                    adj2_neg[rng.gen_range(0..adj2_neg.len())]
                },
            )
            .replace("{reason}", reasons[rng.gen_range(0..reasons.len())])
            .replace("{event}", events[rng.gen_range(0..events.len())]);

        texts.push(text);
    }
    texts
}

/// Generate labeled training data for the aspect sentiment classifier.
///
/// Features: [positive_word_count, negative_word_count, negation_present,
///            intensity_mean, aspect_position_ratio]
/// Label: 1.0 = positive, 0.0 = negative, 0.5 = neutral
pub fn generate_training_data(n: usize) -> Vec<(Vec<f64>, f64)> {
    let mut rng = rand::thread_rng();
    let mut data = Vec::with_capacity(n);

    for _ in 0..n {
        let pos_count: f64 = rng.gen_range(0.0..5.0);
        let neg_count: f64 = rng.gen_range(0.0..5.0);
        let has_negation: f64 = if rng.gen_bool(0.2) { 1.0 } else { 0.0 };
        let intensity: f64 = rng.gen_range(0.5..1.5);
        let position_ratio: f64 = rng.gen_range(0.0..1.0);

        // Generate label based on features
        let signal = (pos_count - neg_count) * intensity * (if has_negation > 0.5 { -1.0 } else { 1.0 });
        let label = if signal > 1.0 {
            1.0 // positive
        } else if signal < -1.0 {
            0.0 // negative
        } else {
            0.5 // neutral
        };
        // Add some noise
        let label = if rng.gen_bool(0.1) {
            match rng.gen_range(0..3) {
                0 => 0.0,
                1 => 0.5,
                _ => 1.0,
            }
        } else {
            label
        };

        data.push((
            vec![pos_count, neg_count, has_negation, intensity, position_ratio],
            label,
        ));
    }
    data
}

// ─── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aspect_extraction_basic() {
        let extractor = AspectExtractor::new();
        let text = "Company reported record revenue and strong growth in market share.";
        let aspects = extractor.extract(text);

        let categories: Vec<AspectCategory> = aspects.iter().map(|a| a.category).collect();
        assert!(categories.contains(&AspectCategory::Revenue));
        assert!(categories.contains(&AspectCategory::Growth));
    }

    #[test]
    fn test_aspect_extraction_crypto() {
        let extractor = AspectExtractor::new();
        let text = "The protocol upgrade improved blockchain scalability while regulatory compliance remains uncertain.";
        let aspects = extractor.extract(text);

        let categories: Vec<AspectCategory> = aspects.iter().map(|a| a.category).collect();
        assert!(categories.contains(&AspectCategory::Technology));
        assert!(categories.contains(&AspectCategory::Regulation));
    }

    #[test]
    fn test_aspect_extraction_empty() {
        let extractor = AspectExtractor::new();
        let text = "The weather today is sunny and warm.";
        let aspects = extractor.extract(text);
        assert!(aspects.is_empty());
    }

    #[test]
    fn test_sentiment_scorer_positive() {
        let scorer = SentimentScorer::new();
        let text = "Revenue achieved a record high with strong performance.";
        let score = scorer.score_aspect(text, 0);
        assert!(score > 0.0, "Expected positive score, got {}", score);
    }

    #[test]
    fn test_sentiment_scorer_negative() {
        let scorer = SentimentScorer::new();
        let text = "Margins declined sharply due to disappointing cost control.";
        let score = scorer.score_aspect(text, 0);
        assert!(score < 0.0, "Expected negative score, got {}", score);
    }

    #[test]
    fn test_sentiment_scorer_negation() {
        let scorer = SentimentScorer::new();
        let text_negated = "Revenue did not decline this quarter.";
        let score_negated = scorer.score_aspect(text_negated, 0);
        // Negation should flip the negative word "decline" to positive
        assert!(
            score_negated > 0.0,
            "Expected positive score with negation, got {}",
            score_negated
        );
    }

    #[test]
    fn test_aspect_sentiment_analyzer() {
        let analyzer = AspectSentimentAnalyzer::new();
        let text = "Revenue grew significantly while margins contracted due to rising costs.";
        let results = analyzer.analyze(text);

        assert!(!results.is_empty());
        // Revenue should be positive
        let revenue_results: Vec<&AspectSentiment> = results
            .iter()
            .filter(|r| r.category == AspectCategory::Revenue)
            .collect();
        if !revenue_results.is_empty() {
            assert!(
                revenue_results[0].score > 0.0,
                "Revenue sentiment should be positive"
            );
        }
    }

    #[test]
    fn test_analyzer_batch() {
        let analyzer = AspectSentimentAnalyzer::new();
        let texts = vec![
            "Revenue grew strongly this quarter.",
            "Revenue exceeded expectations by a wide margin.",
            "Revenue declined due to weak demand.",
        ];
        let results = analyzer.analyze_batch(&texts);
        assert!(results.contains_key(&AspectCategory::Revenue));
    }

    #[test]
    fn test_trading_signal_generator() {
        let mut gen = TradingSignalGenerator::new(0.3);

        let mut scores = HashMap::new();
        scores.insert(AspectCategory::Revenue, 0.8);
        scores.insert(AspectCategory::Profitability, 0.5);
        scores.insert(AspectCategory::Growth, 0.7);

        let signal = gen.generate_signal(&scores);
        assert!(signal > 0.0, "Expected positive signal, got {}", signal);

        let action = TradingSignalGenerator::signal_to_action(signal, 0.1, -0.1);
        assert_eq!(action, "BUY");
    }

    #[test]
    fn test_trading_signal_smoothing() {
        let mut gen = TradingSignalGenerator::new(0.2);

        // Send a strong positive signal
        let mut positive_scores = HashMap::new();
        positive_scores.insert(AspectCategory::Revenue, 1.0);
        gen.generate_signal(&positive_scores);

        // Send a strong negative signal — smoothing should dampen it
        let mut negative_scores = HashMap::new();
        negative_scores.insert(AspectCategory::Revenue, -1.0);
        let signal = gen.generate_signal(&negative_scores);

        // Should not fully reach -1.0 due to smoothing
        assert!(
            signal > -1.0,
            "Signal should be dampened by smoothing, got {}",
            signal
        );
    }

    #[test]
    fn test_signal_to_action() {
        assert_eq!(
            TradingSignalGenerator::signal_to_action(0.5, 0.2, -0.2),
            "BUY"
        );
        assert_eq!(
            TradingSignalGenerator::signal_to_action(-0.5, 0.2, -0.2),
            "SELL"
        );
        assert_eq!(
            TradingSignalGenerator::signal_to_action(0.0, 0.2, -0.2),
            "HOLD"
        );
    }

    #[test]
    fn test_classifier_predict() {
        let clf = AspectSentimentClassifier::new(5, 0.01);
        let features = vec![3.0, 0.5, 0.0, 1.0, 0.5];
        let (label, confidence) = clf.predict(&features);
        assert!(
            label == "positive" || label == "negative" || label == "neutral"
        );
        assert!(confidence > 0.0 && confidence <= 1.0);
    }

    #[test]
    fn test_classifier_train_and_improve() {
        let data = generate_training_data(500);
        let (train, test) = data.split_at(400);

        let mut clf = AspectSentimentClassifier::new(5, 0.01);
        clf.train(&train.to_vec(), 50);
        let accuracy = clf.accuracy(test);

        assert!(accuracy > 0.0, "Accuracy should be above 0: {}", accuracy);
        assert!(
            accuracy >= 0.3,
            "Trained accuracy should be reasonable: {}",
            accuracy
        );
    }

    #[test]
    fn test_classifier_score() {
        let mut clf = AspectSentimentClassifier::new(5, 0.01);
        let data = generate_training_data(300);
        clf.train(&data, 30);

        // High positive word count, low negative
        let pos_features = vec![4.0, 0.5, 0.0, 1.2, 0.5];
        let pos_score = clf.score(&pos_features);

        // Low positive word count, high negative
        let neg_features = vec![0.5, 4.0, 0.0, 1.2, 0.5];
        let neg_score = clf.score(&neg_features);

        assert!(
            pos_score > neg_score,
            "Positive features should score higher: pos={}, neg={}",
            pos_score,
            neg_score
        );
    }

    #[test]
    fn test_synthetic_text_generation() {
        let texts = generate_synthetic_texts(20);
        assert_eq!(texts.len(), 20);
        for text in &texts {
            assert!(!text.is_empty());
            assert!(!text.contains('{'));
        }
    }

    #[test]
    fn test_synthetic_training_data() {
        let data = generate_training_data(100);
        assert_eq!(data.len(), 100);
        for (features, label) in &data {
            assert_eq!(features.len(), 5);
            assert!(*label == 0.0 || *label == 0.5 || *label == 1.0);
        }
    }

    #[test]
    fn test_aspect_polarity_labels() {
        let pos = AspectSentiment {
            category: AspectCategory::Revenue,
            keyword: "revenue".to_string(),
            score: 0.5,
        };
        assert_eq!(pos.polarity(), "positive");

        let neg = AspectSentiment {
            category: AspectCategory::Risk,
            keyword: "debt".to_string(),
            score: -0.5,
        };
        assert_eq!(neg.polarity(), "negative");

        let neu = AspectSentiment {
            category: AspectCategory::Operations,
            keyword: "operations".to_string(),
            score: 0.0,
        };
        assert_eq!(neu.polarity(), "neutral");
    }

    #[test]
    fn test_extractor_dictionary_size() {
        let extractor = AspectExtractor::new();
        assert!(
            extractor.dictionary_size() > 50,
            "Dictionary should have many entries"
        );
    }

    #[test]
    fn test_full_pipeline() {
        let analyzer = AspectSentimentAnalyzer::new();
        let mut signal_gen = TradingSignalGenerator::new(0.3);

        let texts = vec![
            "Revenue exceeded expectations with strong growth.",
            "Margins improved and costs were well controlled.",
            "The company announced a record earnings per share.",
        ];

        let aspect_scores = analyzer.analyze_batch(&texts);
        let signal = signal_gen.generate_signal(&aspect_scores);

        // All texts are positive, signal should be positive
        assert!(
            signal > 0.0,
            "Pipeline should produce positive signal for positive texts, got {}",
            signal
        );
    }
}
