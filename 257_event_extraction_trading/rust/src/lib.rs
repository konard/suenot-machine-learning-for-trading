use ndarray::Array1;
use rand::Rng;
use serde::Deserialize;
use std::collections::HashMap;

// ─── Event Types ──────────────────────────────────────────────────

/// Supported financial event categories.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EventType {
    Earnings,
    MergersAcquisitions,
    Regulatory,
    ExchangeListing,
    SecurityBreach,
    TokenBurn,
    Partnership,
    ProductLaunch,
    ManagementChange,
    ProtocolUpgrade,
    WhaleMovement,
    Airdrop,
    Other,
}

impl EventType {
    /// Return all known event types (excluding Other).
    pub fn all() -> &'static [EventType] {
        &[
            EventType::Earnings,
            EventType::MergersAcquisitions,
            EventType::Regulatory,
            EventType::ExchangeListing,
            EventType::SecurityBreach,
            EventType::TokenBurn,
            EventType::Partnership,
            EventType::ProductLaunch,
            EventType::ManagementChange,
            EventType::ProtocolUpgrade,
            EventType::WhaleMovement,
            EventType::Airdrop,
        ]
    }

    /// Index for classification vectors.
    pub fn index(&self) -> usize {
        match self {
            EventType::Earnings => 0,
            EventType::MergersAcquisitions => 1,
            EventType::Regulatory => 2,
            EventType::ExchangeListing => 3,
            EventType::SecurityBreach => 4,
            EventType::TokenBurn => 5,
            EventType::Partnership => 6,
            EventType::ProductLaunch => 7,
            EventType::ManagementChange => 8,
            EventType::ProtocolUpgrade => 9,
            EventType::WhaleMovement => 10,
            EventType::Airdrop => 11,
            EventType::Other => 12,
        }
    }

    /// Number of event types (including Other).
    pub fn count() -> usize {
        13
    }

    /// Create from index.
    pub fn from_index(idx: usize) -> Self {
        match idx {
            0 => EventType::Earnings,
            1 => EventType::MergersAcquisitions,
            2 => EventType::Regulatory,
            3 => EventType::ExchangeListing,
            4 => EventType::SecurityBreach,
            5 => EventType::TokenBurn,
            6 => EventType::Partnership,
            7 => EventType::ProductLaunch,
            8 => EventType::ManagementChange,
            9 => EventType::ProtocolUpgrade,
            10 => EventType::WhaleMovement,
            11 => EventType::Airdrop,
            _ => EventType::Other,
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            EventType::Earnings => "earnings",
            EventType::MergersAcquisitions => "m&a",
            EventType::Regulatory => "regulatory",
            EventType::ExchangeListing => "listing",
            EventType::SecurityBreach => "security_breach",
            EventType::TokenBurn => "token_burn",
            EventType::Partnership => "partnership",
            EventType::ProductLaunch => "product_launch",
            EventType::ManagementChange => "management_change",
            EventType::ProtocolUpgrade => "protocol_upgrade",
            EventType::WhaleMovement => "whale_movement",
            EventType::Airdrop => "airdrop",
            EventType::Other => "other",
        }
    }
}

impl std::fmt::Display for EventType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label())
    }
}

// ─── Extracted Event ──────────────────────────────────────────────

/// A structured financial event extracted from text.
#[derive(Debug, Clone)]
pub struct ExtractedEvent {
    pub event_type: EventType,
    pub trigger_word: String,
    pub text: String,
    pub confidence: f64,
    pub sentiment: f64,
    pub entities: Vec<String>,
}

// ─── Event Extractor ──────────────────────────────────────────────

/// Extracts financial events from text using keyword-based trigger detection.
///
/// Maintains a configurable dictionary of trigger words mapped to event types.
/// Applies pattern matching to classify events from news headlines and snippets.
#[derive(Debug)]
pub struct EventExtractor {
    trigger_map: HashMap<String, EventType>,
}

impl EventExtractor {
    pub fn new() -> Self {
        let mut trigger_map = HashMap::new();

        // Earnings triggers
        for w in &["earnings", "revenue", "profit", "loss", "eps", "guidance",
                    "quarterly", "annual report", "beat estimates", "missed estimates",
                    "revenue growth", "net income"] {
            trigger_map.insert(w.to_string(), EventType::Earnings);
        }

        // M&A triggers
        for w in &["acquired", "acquisition", "merger", "merged", "takeover",
                    "buyout", "deal", "purchase agreement", "tender offer"] {
            trigger_map.insert(w.to_string(), EventType::MergersAcquisitions);
        }

        // Regulatory triggers
        for w in &["sec", "fined", "penalty", "regulation", "compliance",
                    "approved", "rejected", "banned", "sanctioned", "lawsuit",
                    "investigation", "subpoena", "enforcement"] {
            trigger_map.insert(w.to_string(), EventType::Regulatory);
        }

        // Exchange listing triggers
        for w in &["listed", "listing", "delisted", "delisting", "ipo",
                    "trading pair", "perpetual contract", "spot listing"] {
            trigger_map.insert(w.to_string(), EventType::ExchangeListing);
        }

        // Security breach triggers
        for w in &["hacked", "hack", "breach", "exploit", "vulnerability",
                    "stolen", "compromised", "attack", "drained"] {
            trigger_map.insert(w.to_string(), EventType::SecurityBreach);
        }

        // Token burn triggers
        for w in &["burned", "burn", "token burn", "deflationary",
                    "supply reduction", "destroyed"] {
            trigger_map.insert(w.to_string(), EventType::TokenBurn);
        }

        // Partnership triggers
        for w in &["partnership", "partnered", "collaboration", "alliance",
                    "joint venture", "integration", "strategic agreement"] {
            trigger_map.insert(w.to_string(), EventType::Partnership);
        }

        // Product launch triggers
        for w in &["launched", "launch", "released", "release", "unveiled",
                    "introduced", "rollout", "beta", "mainnet"] {
            trigger_map.insert(w.to_string(), EventType::ProductLaunch);
        }

        // Management change triggers
        for w in &["ceo", "cfo", "appointed", "resigned", "fired",
                    "stepped down", "new leadership", "board member"] {
            trigger_map.insert(w.to_string(), EventType::ManagementChange);
        }

        // Protocol upgrade triggers
        for w in &["upgrade", "hard fork", "soft fork", "update", "v2",
                    "migration", "protocol change", "eip"] {
            trigger_map.insert(w.to_string(), EventType::ProtocolUpgrade);
        }

        // Whale movement triggers
        for w in &["whale", "large transfer", "moved", "transferred",
                    "wallet", "dormant"] {
            trigger_map.insert(w.to_string(), EventType::WhaleMovement);
        }

        // Airdrop triggers
        for w in &["airdrop", "token distribution", "free tokens",
                    "claim", "snapshot"] {
            trigger_map.insert(w.to_string(), EventType::Airdrop);
        }

        Self { trigger_map }
    }

    /// Extract events from a text string.
    /// Returns a list of extracted events with their types and confidence.
    pub fn extract(&self, text: &str) -> Vec<ExtractedEvent> {
        let lower = text.to_lowercase();
        let mut events = Vec::new();
        let mut found_types: HashMap<EventType, (String, usize)> = HashMap::new();

        // Find all trigger words present in the text
        for (trigger, event_type) in &self.trigger_map {
            if let Some(pos) = lower.find(trigger.as_str()) {
                let entry = found_types.entry(*event_type).or_insert_with(|| {
                    (trigger.clone(), pos)
                });
                // Keep the earliest trigger for each type
                if pos < entry.1 {
                    *entry = (trigger.clone(), pos);
                }
            }
        }

        // Create events for each unique type found
        for (event_type, (trigger_word, _)) in found_types {
            let confidence = self.compute_trigger_confidence(&lower, &trigger_word);
            events.push(ExtractedEvent {
                event_type,
                trigger_word,
                text: text.to_string(),
                confidence,
                sentiment: 0.0, // Filled by SentimentScorer
                entities: extract_entities(text),
            });
        }

        // Sort by confidence descending
        events.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
        events
    }

    /// Compute confidence for a trigger match based on context.
    fn compute_trigger_confidence(&self, text: &str, trigger: &str) -> f64 {
        let base: f64 = 0.6;
        let mut score: f64 = base;

        // Boost if trigger appears early in text (likely in headline)
        if let Some(pos) = text.find(trigger) {
            if pos < 50 {
                score += 0.15;
            }
        }

        // Boost for longer trigger phrases (more specific)
        if trigger.contains(' ') {
            score += 0.1;
        }

        // Boost if text is short (headline-like, more focused)
        if text.len() < 200 {
            score += 0.1;
        }

        score.min(1.0)
    }

    /// Add a custom trigger word to the extractor.
    pub fn add_trigger(&mut self, word: &str, event_type: EventType) {
        self.trigger_map.insert(word.to_lowercase(), event_type);
    }

    /// Number of registered trigger words.
    pub fn trigger_count(&self) -> usize {
        self.trigger_map.len()
    }
}

impl Default for EventExtractor {
    fn default() -> Self {
        Self::new()
    }
}

/// Extract entity-like tokens from text (capitalized words, ticker symbols).
fn extract_entities(text: &str) -> Vec<String> {
    let mut entities = Vec::new();
    for word in text.split_whitespace() {
        let clean = word.trim_matches(|c: char| !c.is_alphanumeric());
        if clean.is_empty() {
            continue;
        }
        // All-caps words of 2-6 chars are likely tickers
        if clean.len() >= 2
            && clean.len() <= 6
            && clean.chars().all(|c| c.is_ascii_uppercase())
        {
            entities.push(clean.to_string());
        }
        // Capitalized words longer than 1 char (potential entity names)
        else if clean.len() > 1
            && clean.chars().next().map_or(false, |c| c.is_uppercase())
            && clean.chars().skip(1).any(|c| c.is_lowercase())
        {
            entities.push(clean.to_string());
        }
    }
    entities.dedup();
    entities
}

// ─── Sentiment Scorer ─────────────────────────────────────────────

/// Assigns sentiment polarity to event descriptions using a lexicon-based approach.
///
/// Maintains dictionaries of positive and negative financial terms, computes
/// the net sentiment as the normalized difference between positive and negative
/// word counts, and adjusts for negation patterns.
#[derive(Debug)]
pub struct SentimentScorer {
    positive_words: Vec<String>,
    negative_words: Vec<String>,
    negation_words: Vec<String>,
    intensifiers: Vec<String>,
}

impl SentimentScorer {
    pub fn new() -> Self {
        let positive_words: Vec<String> = vec![
            "beat", "beats", "exceeded", "surpassed", "record", "growth",
            "surge", "surged", "soared", "rally", "bullish", "gain", "gains",
            "profit", "profitable", "upgrade", "upgraded", "outperform",
            "strong", "positive", "optimistic", "boost", "boosted",
            "breakthrough", "innovation", "partnership", "approved",
            "success", "successful", "impressive", "excellent", "robust",
            "momentum", "recovery", "recovered", "expansion", "expanded",
        ]
        .into_iter()
        .map(String::from)
        .collect();

        let negative_words: Vec<String> = vec![
            "miss", "missed", "missed", "decline", "declined", "loss",
            "losses", "crash", "crashed", "plunge", "plunged", "bearish",
            "downgrade", "downgraded", "underperform", "weak", "negative",
            "pessimistic", "risk", "crisis", "hack", "hacked", "breach",
            "exploit", "vulnerability", "fine", "fined", "penalty",
            "lawsuit", "fraud", "scandal", "bankruptcy", "default",
            "collapse", "collapsed", "warning", "concern", "concerns",
            "threat", "delay", "delayed", "failure", "failed", "recall",
        ]
        .into_iter()
        .map(String::from)
        .collect();

        let negation_words: Vec<String> = vec![
            "not", "no", "never", "neither", "nor", "barely", "hardly",
            "without", "don't", "doesn't", "didn't", "won't", "wouldn't",
            "couldn't", "shouldn't",
        ]
        .into_iter()
        .map(String::from)
        .collect();

        let intensifiers: Vec<String> = vec![
            "very", "extremely", "significantly", "substantially",
            "dramatically", "sharply", "massive", "huge", "enormous",
        ]
        .into_iter()
        .map(String::from)
        .collect();

        Self {
            positive_words,
            negative_words,
            negation_words,
            intensifiers,
        }
    }

    /// Score sentiment of text. Returns a value in [-1.0, 1.0].
    pub fn score(&self, text: &str) -> f64 {
        let words: Vec<String> = text
            .to_lowercase()
            .split_whitespace()
            .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
            .filter(|w| !w.is_empty())
            .collect();

        if words.is_empty() {
            return 0.0;
        }

        let mut pos_count = 0.0_f64;
        let mut neg_count = 0.0_f64;

        for (i, word) in words.iter().enumerate() {
            let is_negated = self.is_negated(&words, i);
            let intensity = self.get_intensity(&words, i);

            if self.positive_words.contains(word) {
                if is_negated {
                    neg_count += intensity;
                } else {
                    pos_count += intensity;
                }
            } else if self.negative_words.contains(word) {
                if is_negated {
                    pos_count += intensity;
                } else {
                    neg_count += intensity;
                }
            }
        }

        let total = pos_count + neg_count;
        if total == 0.0 {
            return 0.0;
        }
        (pos_count - neg_count) / total
    }

    /// Check if word at position `idx` is preceded by a negation word
    /// within a window of 3 words.
    fn is_negated(&self, words: &[String], idx: usize) -> bool {
        let start = idx.saturating_sub(3);
        for i in start..idx {
            if self.negation_words.contains(&words[i]) {
                return true;
            }
        }
        false
    }

    /// Get intensity multiplier based on preceding intensifier words.
    fn get_intensity(&self, words: &[String], idx: usize) -> f64 {
        let start = idx.saturating_sub(2);
        for i in start..idx {
            if self.intensifiers.contains(&words[i]) {
                return 1.5;
            }
        }
        1.0
    }
}

impl Default for SentimentScorer {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Event Classifier (Logistic Regression) ───────────────────────

/// Multi-class logistic regression classifier for event type prediction.
///
/// Given a feature vector (derived from text), outputs probabilities
/// over predefined event categories.
#[derive(Debug)]
pub struct EventClassifier {
    weights: Vec<Array1<f64>>,
    biases: Vec<f64>,
    learning_rate: f64,
    num_features: usize,
    num_classes: usize,
}

impl EventClassifier {
    pub fn new(num_features: usize, learning_rate: f64) -> Self {
        let num_classes = EventType::count();
        let mut rng = rand::thread_rng();

        let weights: Vec<Array1<f64>> = (0..num_classes)
            .map(|_| {
                Array1::from_vec(
                    (0..num_features)
                        .map(|_| rng.gen_range(-0.1..0.1))
                        .collect(),
                )
            })
            .collect();

        let biases = vec![0.0; num_classes];

        Self {
            weights,
            biases,
            learning_rate,
            num_features,
            num_classes,
        }
    }

    /// Softmax function over logits.
    fn softmax(logits: &[f64]) -> Vec<f64> {
        let max_val = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = logits.iter().map(|&l| (l - max_val).exp()).collect();
        let sum: f64 = exps.iter().sum();
        exps.iter().map(|&e| e / sum).collect()
    }

    /// Predict class probabilities for a feature vector.
    pub fn predict_proba(&self, features: &[f64]) -> Vec<f64> {
        assert_eq!(features.len(), self.num_features);
        let x = Array1::from_vec(features.to_vec());
        let logits: Vec<f64> = (0..self.num_classes)
            .map(|k| self.weights[k].dot(&x) + self.biases[k])
            .collect();
        Self::softmax(&logits)
    }

    /// Predict the most likely event type. Returns (EventType, confidence).
    pub fn predict(&self, features: &[f64]) -> (EventType, f64) {
        let probs = self.predict_proba(features);
        let (best_idx, &best_prob) = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();
        (EventType::from_index(best_idx), best_prob)
    }

    /// Train on a dataset of (features, class_index) pairs.
    pub fn train(&mut self, data: &[(Vec<f64>, usize)], epochs: usize) {
        for _ in 0..epochs {
            for (features, label) in data {
                let x = Array1::from_vec(features.clone());
                let probs = self.predict_proba(features);

                // Gradient descent for softmax cross-entropy
                for k in 0..self.num_classes {
                    let target = if k == *label { 1.0 } else { 0.0 };
                    let error = probs[k] - target;
                    for j in 0..self.num_features {
                        self.weights[k][j] -= self.learning_rate * error * x[j];
                    }
                    self.biases[k] -= self.learning_rate * error;
                }
            }
        }
    }

    /// Evaluate accuracy on a test set.
    pub fn accuracy(&self, data: &[(Vec<f64>, usize)]) -> f64 {
        let correct = data
            .iter()
            .filter(|(features, label)| {
                let (pred, _) = self.predict(features);
                pred.index() == *label
            })
            .count();
        correct as f64 / data.len() as f64
    }
}

// ─── Impact Predictor ─────────────────────────────────────────────

/// Predicts the expected price impact of an extracted event.
///
/// Uses a linear model mapping event features (type, sentiment, magnitude,
/// market context) to a signed percentage price change estimate.
#[derive(Debug)]
pub struct ImpactPredictor {
    weights: Array1<f64>,
    bias: f64,
    learning_rate: f64,
    num_features: usize,
}

impl ImpactPredictor {
    /// Create a new predictor. `num_features` is the length of the feature vector.
    pub fn new(num_features: usize, learning_rate: f64) -> Self {
        let mut rng = rand::thread_rng();
        let weights = Array1::from_vec(
            (0..num_features)
                .map(|_| rng.gen_range(-0.05..0.05))
                .collect(),
        );
        Self {
            weights,
            bias: 0.0,
            learning_rate,
            num_features,
        }
    }

    /// Predict price impact (signed percentage change).
    pub fn predict(&self, features: &[f64]) -> f64 {
        assert_eq!(features.len(), self.num_features);
        let x = Array1::from_vec(features.to_vec());
        self.weights.dot(&x) + self.bias
    }

    /// Train on (features, actual_impact) pairs using gradient descent.
    pub fn train(&mut self, data: &[(Vec<f64>, f64)], epochs: usize) {
        for _ in 0..epochs {
            for (features, target) in data {
                let x = Array1::from_vec(features.clone());
                let pred = self.weights.dot(&x) + self.bias;
                let error = pred - target;

                for j in 0..self.num_features {
                    self.weights[j] -= self.learning_rate * error * x[j];
                }
                self.bias -= self.learning_rate * error;
            }
        }
    }

    /// Mean absolute error on a test set.
    pub fn mae(&self, data: &[(Vec<f64>, f64)]) -> f64 {
        let total: f64 = data
            .iter()
            .map(|(features, target)| (self.predict(features) - target).abs())
            .sum();
        total / data.len() as f64
    }
}

// ─── Trading Signal ───────────────────────────────────────────────

/// Trading action derived from event analysis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TradingAction {
    Buy,
    Sell,
    Hold,
}

impl std::fmt::Display for TradingAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TradingAction::Buy => write!(f, "BUY"),
            TradingAction::Sell => write!(f, "SELL"),
            TradingAction::Hold => write!(f, "HOLD"),
        }
    }
}

/// A trading signal produced from event analysis.
#[derive(Debug, Clone)]
pub struct TradingSignal {
    pub action: TradingAction,
    pub confidence: f64,
    pub predicted_impact: f64,
    pub event_type: EventType,
    pub reason: String,
}

// ─── Trading Signal Generator ─────────────────────────────────────

/// Combines event extraction, classification, impact prediction, and
/// sentiment scoring into actionable trading signals.
pub struct TradingSignalGenerator {
    extractor: EventExtractor,
    sentiment_scorer: SentimentScorer,
    impact_predictor: ImpactPredictor,
    confidence_threshold: f64,
    impact_threshold: f64,
}

impl TradingSignalGenerator {
    pub fn new(confidence_threshold: f64, impact_threshold: f64) -> Self {
        // Impact features: [event_type_one_hot(13), sentiment, confidence]
        let num_impact_features = EventType::count() + 2;
        Self {
            extractor: EventExtractor::new(),
            sentiment_scorer: SentimentScorer::new(),
            impact_predictor: ImpactPredictor::new(num_impact_features, 0.001),
            confidence_threshold,
            impact_threshold,
        }
    }

    /// Process a text input and generate trading signals.
    pub fn process(&self, text: &str) -> Vec<TradingSignal> {
        let mut events = self.extractor.extract(text);
        let mut signals = Vec::new();

        for event in &mut events {
            // Score sentiment
            event.sentiment = self.sentiment_scorer.score(&event.text);

            // Build impact feature vector
            let impact_features = self.build_impact_features(event);
            let predicted_impact = self.impact_predictor.predict(&impact_features);

            // Filter by thresholds
            if event.confidence < self.confidence_threshold {
                continue;
            }

            let action = if predicted_impact.abs() < self.impact_threshold {
                TradingAction::Hold
            } else if predicted_impact > 0.0 {
                TradingAction::Buy
            } else {
                TradingAction::Sell
            };

            signals.push(TradingSignal {
                action,
                confidence: event.confidence,
                predicted_impact,
                event_type: event.event_type,
                reason: format!(
                    "{} event detected (trigger: '{}', sentiment: {:.2})",
                    event.event_type, event.trigger_word, event.sentiment
                ),
            });
        }

        signals
    }

    /// Build the feature vector for impact prediction.
    fn build_impact_features(&self, event: &ExtractedEvent) -> Vec<f64> {
        let mut features = vec![0.0; EventType::count() + 2];
        // One-hot encode event type
        features[event.event_type.index()] = 1.0;
        // Sentiment
        features[EventType::count()] = event.sentiment;
        // Confidence
        features[EventType::count() + 1] = event.confidence;
        features
    }

    /// Train the impact predictor on historical event-impact pairs.
    pub fn train_impact_model(&mut self, data: &[(Vec<f64>, f64)], epochs: usize) {
        self.impact_predictor.train(data, epochs);
    }

    /// Get a reference to the extractor (for adding custom triggers).
    pub fn extractor_mut(&mut self) -> &mut EventExtractor {
        &mut self.extractor
    }
}

// ─── Bybit Client ─────────────────────────────────────────────────

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
    pub list: Vec<TickerEntry>,
}

#[derive(Debug, Deserialize)]
pub struct TickerEntry {
    pub symbol: String,
    #[serde(rename = "lastPrice")]
    pub last_price: String,
    #[serde(rename = "volume24h")]
    pub volume_24h: String,
    #[serde(rename = "turnover24h")]
    pub turnover_24h: String,
    #[serde(rename = "highPrice24h")]
    pub high_price_24h: String,
    #[serde(rename = "lowPrice24h")]
    pub low_price_24h: String,
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

/// Parsed ticker data.
#[derive(Debug, Clone)]
pub struct Ticker {
    pub symbol: String,
    pub last_price: f64,
    pub volume_24h: f64,
    pub turnover_24h: f64,
    pub high_24h: f64,
    pub low_24h: f64,
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
        let resp: BybitResponse<KlineResult> =
            self.client.get(&url).send().await?.json().await?;

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

    /// Fetch ticker data for a symbol.
    pub async fn get_ticker(&self, symbol: &str) -> anyhow::Result<Ticker> {
        let url = format!(
            "{}/v5/market/tickers?category=spot&symbol={}",
            self.base_url, symbol
        );
        let resp: BybitResponse<TickerResult> =
            self.client.get(&url).send().await?.json().await?;

        let entry = resp
            .result
            .list
            .first()
            .ok_or_else(|| anyhow::anyhow!("No ticker data for {}", symbol))?;

        Ok(Ticker {
            symbol: entry.symbol.clone(),
            last_price: entry.last_price.parse().unwrap_or(0.0),
            volume_24h: entry.volume_24h.parse().unwrap_or(0.0),
            turnover_24h: entry.turnover_24h.parse().unwrap_or(0.0),
            high_24h: entry.high_price_24h.parse().unwrap_or(0.0),
            low_24h: entry.low_price_24h.parse().unwrap_or(0.0),
        })
    }
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Synthetic Data Generation ────────────────────────────────────

/// Generate synthetic news headlines for testing.
pub fn generate_sample_news() -> Vec<&'static str> {
    vec![
        "Tesla beats earnings estimates with record revenue in Q4",
        "Bybit listed SOL/USDT perpetual contract, trading begins immediately",
        "Major crypto exchange hacked, $100 million stolen from hot wallets",
        "Ethereum protocol upgrade scheduled for next month, improved scalability expected",
        "SEC fined major trading firm $50 million for compliance violations",
        "Apple acquired AI startup for $2 billion to boost machine learning capabilities",
        "Bitcoin whale moved 5000 BTC to Binance, largest transfer this month",
        "New partnership between Chainlink and major DeFi protocol announced",
        "Company CEO resigned amid accounting scandal investigation",
        "Token burn event destroyed 10 million tokens, reducing total supply by 5%",
        "Airdrop announced for early protocol users, snapshot date confirmed",
        "Product launch: new DeFi lending platform goes live on mainnet",
        "Market rally continues as bullish momentum drives prices higher",
        "Crash warning: leveraged positions face liquidation risk",
        "Not impressed with quarterly results, revenue growth declined sharply",
    ]
}

/// Generate synthetic training data for the event classifier.
pub fn generate_classifier_training_data(n: usize) -> Vec<(Vec<f64>, usize)> {
    let mut rng = rand::thread_rng();
    let num_features = 20;
    let num_classes = EventType::count();
    let mut data = Vec::with_capacity(n);

    for _ in 0..n {
        let label = rng.gen_range(0..num_classes);
        let mut features = vec![0.0; num_features];

        // Create features that correlate with the label
        for (i, f) in features.iter_mut().enumerate() {
            *f = rng.gen_range(-1.0..1.0);
            // Add signal for the correct class
            if i % num_classes == label {
                *f += rng.gen_range(0.5..1.5);
            }
        }

        data.push((features, label));
    }
    data
}

/// Generate synthetic training data for the impact predictor.
pub fn generate_impact_training_data(n: usize) -> Vec<(Vec<f64>, f64)> {
    let mut rng = rand::thread_rng();
    let num_features = EventType::count() + 2; // one-hot type + sentiment + confidence
    let mut data = Vec::with_capacity(n);

    // Average impact per event type (signed percentage)
    let type_impacts = [
        0.5,  // Earnings (mild positive on average)
        2.0,  // M&A (positive)
        -1.5, // Regulatory (negative)
        3.0,  // Exchange listing (positive)
        -5.0, // Security breach (very negative)
        1.0,  // Token burn (positive, deflationary)
        1.5,  // Partnership (positive)
        1.0,  // Product launch (positive)
        -0.5, // Management change (mildly negative)
        2.0,  // Protocol upgrade (positive)
        -1.0, // Whale movement (uncertain, slight negative)
        0.5,  // Airdrop (mild positive)
        0.0,  // Other
    ];

    for _ in 0..n {
        let event_type = rng.gen_range(0..EventType::count());
        let sentiment: f64 = rng.gen_range(-1.0..1.0);
        let confidence: f64 = rng.gen_range(0.5..1.0);

        let mut features = vec![0.0; num_features];
        features[event_type] = 1.0;
        features[EventType::count()] = sentiment;
        features[EventType::count() + 1] = confidence;

        // Impact = base type impact + sentiment contribution + noise
        let impact = type_impacts[event_type]
            + sentiment * 1.5
            + rng.gen_range(-0.5..0.5);

        data.push((features, impact));
    }
    data
}

// ─── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_extractor_basic() {
        let extractor = EventExtractor::new();
        let events = extractor.extract("Tesla beats earnings estimates with record revenue");
        assert!(!events.is_empty());
        let has_earnings = events.iter().any(|e| e.event_type == EventType::Earnings);
        assert!(has_earnings, "Should detect earnings event");
    }

    #[test]
    fn test_event_extractor_multiple_events() {
        let extractor = EventExtractor::new();
        let events = extractor.extract(
            "Company acquired a startup and listed new trading pair on the exchange"
        );
        assert!(events.len() >= 2, "Should detect at least 2 events");
        let types: Vec<EventType> = events.iter().map(|e| e.event_type).collect();
        assert!(types.contains(&EventType::MergersAcquisitions));
        assert!(types.contains(&EventType::ExchangeListing));
    }

    #[test]
    fn test_event_extractor_no_events() {
        let extractor = EventExtractor::new();
        let events = extractor.extract("The weather is nice today");
        assert!(events.is_empty(), "Should not detect events in irrelevant text");
    }

    #[test]
    fn test_event_extractor_security_breach() {
        let extractor = EventExtractor::new();
        let events = extractor.extract("Major exchange hacked, millions stolen");
        assert!(!events.is_empty());
        let has_breach = events.iter().any(|e| e.event_type == EventType::SecurityBreach);
        assert!(has_breach, "Should detect security breach event");
    }

    #[test]
    fn test_event_extractor_custom_trigger() {
        let mut extractor = EventExtractor::new();
        extractor.add_trigger("halving", EventType::TokenBurn);
        let events = extractor.extract("Bitcoin halving event is approaching");
        let has_burn = events.iter().any(|e| e.event_type == EventType::TokenBurn);
        assert!(has_burn, "Should detect custom trigger");
    }

    #[test]
    fn test_event_type_roundtrip() {
        for et in EventType::all() {
            let idx = et.index();
            let recovered = EventType::from_index(idx);
            assert_eq!(*et, recovered);
        }
    }

    #[test]
    fn test_sentiment_positive() {
        let scorer = SentimentScorer::new();
        let score = scorer.score("Company beats earnings with record profit and strong growth");
        assert!(score > 0.0, "Should be positive, got {}", score);
    }

    #[test]
    fn test_sentiment_negative() {
        let scorer = SentimentScorer::new();
        let score = scorer.score("Exchange hacked, massive losses and crash expected");
        assert!(score < 0.0, "Should be negative, got {}", score);
    }

    #[test]
    fn test_sentiment_neutral() {
        let scorer = SentimentScorer::new();
        let score = scorer.score("The meeting will be held on Tuesday");
        assert!(
            score.abs() < 0.01,
            "Should be neutral, got {}",
            score
        );
    }

    #[test]
    fn test_sentiment_negation() {
        let scorer = SentimentScorer::new();
        let positive = scorer.score("Company shows strong growth");
        let negated = scorer.score("Company does not show strong growth");
        assert!(
            negated < positive,
            "Negation should reduce sentiment: positive={}, negated={}",
            positive,
            negated
        );
    }

    #[test]
    fn test_sentiment_intensifier() {
        let scorer = SentimentScorer::new();
        let base = scorer.score("Revenue growth reported");
        let intensified = scorer.score("Extremely significant revenue growth reported");
        assert!(
            intensified >= base,
            "Intensifier should increase magnitude: base={}, intensified={}",
            base,
            intensified
        );
    }

    #[test]
    fn test_entity_extraction() {
        let entities = extract_entities("Tesla CEO Elon Musk announced TSLA buyback on NYSE");
        assert!(entities.contains(&"Tesla".to_string()));
        assert!(entities.contains(&"TSLA".to_string()));
        assert!(entities.contains(&"NYSE".to_string()));
    }

    #[test]
    fn test_event_classifier_predict() {
        let clf = EventClassifier::new(20, 0.01);
        let features = vec![0.5; 20];
        let (event_type, confidence) = clf.predict(&features);
        assert!(confidence > 0.0 && confidence <= 1.0);
        assert!(event_type.index() < EventType::count());
    }

    #[test]
    fn test_event_classifier_probabilities_sum_to_one() {
        let clf = EventClassifier::new(20, 0.01);
        let features = vec![0.5; 20];
        let probs = clf.predict_proba(&features);
        let sum: f64 = probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "Probabilities should sum to 1.0, got {}",
            sum
        );
    }

    #[test]
    fn test_event_classifier_train() {
        let data = generate_classifier_training_data(500);
        let (train, test) = data.split_at(400);

        let mut clf = EventClassifier::new(20, 0.01);
        clf.train(&train.to_vec(), 30);
        let acc = clf.accuracy(test);
        assert!(
            acc > 0.0,
            "Trained classifier should have non-zero accuracy, got {}",
            acc
        );
    }

    #[test]
    fn test_impact_predictor() {
        let predictor = ImpactPredictor::new(15, 0.001);
        let features = vec![0.0; 15];
        let impact = predictor.predict(&features);
        // With random init, impact should be near zero for zero features
        assert!(impact.abs() < 1.0, "Impact should be small for zero features");
    }

    #[test]
    fn test_impact_predictor_train() {
        let data = generate_impact_training_data(500);
        let (train, test) = data.split_at(400);

        let mut predictor = ImpactPredictor::new(EventType::count() + 2, 0.001);
        let mae_before = predictor.mae(test);
        predictor.train(&train.to_vec(), 50);
        let mae_after = predictor.mae(test);

        assert!(
            mae_after < mae_before,
            "MAE should decrease after training: before={}, after={}",
            mae_before,
            mae_after
        );
    }

    #[test]
    fn test_trading_signal_generator() {
        let generator = TradingSignalGenerator::new(0.5, 0.01);
        let signals = generator.process("Bybit listed new SOL perpetual contract");
        assert!(!signals.is_empty(), "Should generate at least one signal");
        assert!(signals[0].confidence >= 0.5);
    }

    #[test]
    fn test_trading_signal_generator_no_signal() {
        let generator = TradingSignalGenerator::new(0.99, 10.0);
        let signals = generator.process("The weather is nice today");
        assert!(signals.is_empty(), "Should not generate signals for irrelevant text");
    }

    #[test]
    fn test_sample_news_extraction() {
        let extractor = EventExtractor::new();
        let news = generate_sample_news();
        let mut total_events = 0;
        for headline in &news {
            let events = extractor.extract(headline);
            total_events += events.len();
        }
        assert!(
            total_events >= 10,
            "Should extract events from most sample headlines, got {}",
            total_events
        );
    }

    #[test]
    fn test_synthetic_data_generation() {
        let clf_data = generate_classifier_training_data(100);
        assert_eq!(clf_data.len(), 100);
        for (features, label) in &clf_data {
            assert_eq!(features.len(), 20);
            assert!(*label < EventType::count());
        }

        let impact_data = generate_impact_training_data(100);
        assert_eq!(impact_data.len(), 100);
        for (features, _impact) in &impact_data {
            assert_eq!(features.len(), EventType::count() + 2);
        }
    }

    #[test]
    fn test_trading_action_display() {
        assert_eq!(format!("{}", TradingAction::Buy), "BUY");
        assert_eq!(format!("{}", TradingAction::Sell), "SELL");
        assert_eq!(format!("{}", TradingAction::Hold), "HOLD");
    }

    #[test]
    fn test_event_type_display() {
        assert_eq!(format!("{}", EventType::Earnings), "earnings");
        assert_eq!(format!("{}", EventType::SecurityBreach), "security_breach");
    }
}
