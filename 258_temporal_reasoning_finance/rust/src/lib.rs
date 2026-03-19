use ndarray::Array1;
use rand::Rng;
use regex::Regex;
use serde::Deserialize;
use std::collections::HashMap;

// ─── Temporal Expression Types ────────────────────────────────────

/// The type of a temporal expression found in text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TemporalExpressionType {
    Date,
    Duration,
    FiscalQuarter,
    FiscalYear,
    RelativeReference,
    RecurringEvent,
}

/// A temporal expression extracted from financial text.
#[derive(Debug, Clone)]
pub struct TemporalExpression {
    pub text: String,
    pub expr_type: TemporalExpressionType,
    pub start_offset: usize,
    pub end_offset: usize,
    pub confidence: f64,
    /// Normalized timestamp (seconds since epoch) for the start of the interval.
    pub normalized_start: Option<i64>,
    /// Normalized timestamp (seconds since epoch) for the end of the interval.
    pub normalized_end: Option<i64>,
}

// ─── Temporal Relation (Allen's Interval Algebra) ─────────────────

/// Allen's 13 interval relations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TemporalRelation {
    Before,
    After,
    Meets,
    MetBy,
    Overlaps,
    OverlappedBy,
    Starts,
    StartedBy,
    Finishes,
    FinishedBy,
    During,
    Contains,
    Equals,
}

impl TemporalRelation {
    /// Return the inverse of a relation.
    pub fn inverse(&self) -> Self {
        match self {
            Self::Before => Self::After,
            Self::After => Self::Before,
            Self::Meets => Self::MetBy,
            Self::MetBy => Self::Meets,
            Self::Overlaps => Self::OverlappedBy,
            Self::OverlappedBy => Self::Overlaps,
            Self::Starts => Self::StartedBy,
            Self::StartedBy => Self::Starts,
            Self::Finishes => Self::FinishedBy,
            Self::FinishedBy => Self::Finishes,
            Self::During => Self::Contains,
            Self::Contains => Self::During,
            Self::Equals => Self::Equals,
        }
    }

    /// Determine the Allen relation between two intervals [s1, e1] and [s2, e2].
    pub fn from_intervals(s1: i64, e1: i64, s2: i64, e2: i64) -> Self {
        if e1 < s2 {
            Self::Before
        } else if e1 == s2 {
            Self::Meets
        } else if s1 < s2 && e1 > s2 && e1 < e2 {
            Self::Overlaps
        } else if s1 == s2 && e1 < e2 {
            Self::Starts
        } else if s1 > s2 && e1 < e2 {
            Self::During
        } else if s1 > s2 && e1 == e2 {
            Self::Finishes
        } else if s1 == s2 && e1 == e2 {
            Self::Equals
        } else if s1 == s2 && e1 > e2 {
            Self::StartedBy
        } else if s1 < s2 && e1 > e2 {
            Self::Contains
        } else if s1 > s2 && s1 < e2 && e1 > e2 {
            Self::OverlappedBy
        } else if s1 == e2 {
            Self::MetBy
        } else if s1 > e2 {
            Self::After
        } else if s1 < s2 && e1 == e2 {
            Self::FinishedBy
        } else {
            Self::Equals
        }
    }
}

// ─── Financial Event ──────────────────────────────────────────────

/// Type of financial event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EventType {
    Earnings,
    FomcDecision,
    Halving,
    TokenUnlock,
    RegulatoryDeadline,
    ProductLaunch,
    ContractExpiry,
    ExDividend,
    IpoDate,
    ProtocolUpgrade,
}

/// A financial event with temporal information.
#[derive(Debug, Clone)]
pub struct FinancialEvent {
    pub name: String,
    pub event_type: EventType,
    pub entity: String,
    pub timestamp: i64,
    pub duration_secs: Option<i64>,
    pub impact_score: f64,
}

impl FinancialEvent {
    pub fn end_timestamp(&self) -> i64 {
        self.timestamp + self.duration_secs.unwrap_or(0)
    }
}

// ─── Event Timeline ───────────────────────────────────────────────

/// An ordered timeline of financial events.
#[derive(Debug, Default)]
pub struct EventTimeline {
    events: Vec<FinancialEvent>,
}

impl EventTimeline {
    pub fn new() -> Self {
        Self { events: Vec::new() }
    }

    /// Insert an event maintaining chronological order.
    pub fn insert(&mut self, event: FinancialEvent) {
        let pos = self
            .events
            .binary_search_by_key(&event.timestamp, |e| e.timestamp)
            .unwrap_or_else(|i| i);
        self.events.insert(pos, event);
    }

    /// Query events within a time range [start, end].
    pub fn query_range(&self, start: i64, end: i64) -> Vec<&FinancialEvent> {
        self.events
            .iter()
            .filter(|e| e.timestamp >= start && e.timestamp <= end)
            .collect()
    }

    /// Detect temporal gaps larger than `min_gap_secs` between consecutive events.
    pub fn detect_gaps(&self, min_gap_secs: i64) -> Vec<(i64, i64)> {
        let mut gaps = Vec::new();
        for window in self.events.windows(2) {
            let gap = window[1].timestamp - window[0].end_timestamp();
            if gap > min_gap_secs {
                gaps.push((window[0].end_timestamp(), window[1].timestamp));
            }
        }
        gaps
    }

    /// Compute the Allen relation between two events by index.
    pub fn relation(&self, i: usize, j: usize) -> Option<TemporalRelation> {
        let a = self.events.get(i)?;
        let b = self.events.get(j)?;
        Some(TemporalRelation::from_intervals(
            a.timestamp,
            a.end_timestamp(),
            b.timestamp,
            b.end_timestamp(),
        ))
    }

    pub fn events(&self) -> &[FinancialEvent] {
        &self.events
    }

    pub fn len(&self) -> usize {
        self.events.len()
    }

    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }
}

// ─── Temporal Expression Extractor ────────────────────────────────

/// Extracts temporal expressions from financial text using pattern matching.
pub struct TemporalExpressionExtractor {
    date_pattern: Regex,
    quarter_pattern: Regex,
    fiscal_year_pattern: Regex,
    relative_pattern: Regex,
    duration_pattern: Regex,
}

impl TemporalExpressionExtractor {
    pub fn new() -> Self {
        Self {
            date_pattern: Regex::new(
                r"(?i)(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:,?\s*\d{4})?|\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4}"
            ).unwrap(),
            quarter_pattern: Regex::new(
                r"(?i)Q[1-4]\s*(?:20\d{2})?|(?:first|second|third|fourth)\s+quarter"
            ).unwrap(),
            fiscal_year_pattern: Regex::new(
                r"(?i)FY\s*20\d{2}|fiscal\s+year\s+20\d{2}"
            ).unwrap(),
            relative_pattern: Regex::new(
                r"(?i)(?:next|last|previous|this|coming|past)\s+(?:week|month|quarter|year|fiscal\s+year)|(?:yesterday|today|tomorrow)"
            ).unwrap(),
            duration_pattern: Regex::new(
                r"(?i)(?:over\s+the\s+(?:past|last|next)\s+)?\d+\s+(?:days?|weeks?|months?|quarters?|years?)|T\+[0-9]+"
            ).unwrap(),
        }
    }

    /// Extract all temporal expressions from a text.
    pub fn extract(&self, text: &str) -> Vec<TemporalExpression> {
        let mut expressions = Vec::new();

        for m in self.date_pattern.find_iter(text) {
            expressions.push(TemporalExpression {
                text: m.as_str().to_string(),
                expr_type: TemporalExpressionType::Date,
                start_offset: m.start(),
                end_offset: m.end(),
                confidence: 0.95,
                normalized_start: None,
                normalized_end: None,
            });
        }

        for m in self.quarter_pattern.find_iter(text) {
            expressions.push(TemporalExpression {
                text: m.as_str().to_string(),
                expr_type: TemporalExpressionType::FiscalQuarter,
                start_offset: m.start(),
                end_offset: m.end(),
                confidence: 0.90,
                normalized_start: None,
                normalized_end: None,
            });
        }

        for m in self.fiscal_year_pattern.find_iter(text) {
            expressions.push(TemporalExpression {
                text: m.as_str().to_string(),
                expr_type: TemporalExpressionType::FiscalYear,
                start_offset: m.start(),
                end_offset: m.end(),
                confidence: 0.92,
                normalized_start: None,
                normalized_end: None,
            });
        }

        for m in self.relative_pattern.find_iter(text) {
            expressions.push(TemporalExpression {
                text: m.as_str().to_string(),
                expr_type: TemporalExpressionType::RelativeReference,
                start_offset: m.start(),
                end_offset: m.end(),
                confidence: 0.80,
                normalized_start: None,
                normalized_end: None,
            });
        }

        for m in self.duration_pattern.find_iter(text) {
            expressions.push(TemporalExpression {
                text: m.as_str().to_string(),
                expr_type: TemporalExpressionType::Duration,
                start_offset: m.start(),
                end_offset: m.end(),
                confidence: 0.85,
                normalized_start: None,
                normalized_end: None,
            });
        }

        // Sort by position in text
        expressions.sort_by_key(|e| e.start_offset);
        expressions
    }
}

impl Default for TemporalExpressionExtractor {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Temporal Feature Extractor ───────────────────────────────────

/// Computes temporal features from financial text for ML models.
pub struct TemporalFeatureExtractor {
    extractor: TemporalExpressionExtractor,
    lambda: f64,
    time_scale: f64,
}

impl TemporalFeatureExtractor {
    /// Create a new feature extractor.
    /// - `lambda`: decay parameter for recency scoring
    /// - `time_scale`: normalization factor for time differences (in seconds)
    pub fn new(lambda: f64, time_scale: f64) -> Self {
        Self {
            extractor: TemporalExpressionExtractor::new(),
            lambda,
            time_scale,
        }
    }

    /// Compute temporal density: ratio of temporal tokens to total tokens.
    pub fn temporal_density(&self, text: &str) -> f64 {
        let expressions = self.extractor.extract(text);
        let total_tokens = text.split_whitespace().count();
        if total_tokens == 0 {
            return 0.0;
        }
        let temporal_tokens: usize = expressions
            .iter()
            .map(|e| e.text.split_whitespace().count())
            .sum();
        temporal_tokens as f64 / total_tokens as f64
    }

    /// Compute event recency score.
    pub fn recency_score(&self, event_time: i64, current_time: i64) -> f64 {
        let dt = (current_time - event_time).max(0) as f64;
        (-self.lambda * dt / self.time_scale).exp()
    }

    /// Compute forward-looking ratio from temporal expressions.
    /// `reference_time` is the "now" used to classify expressions as past or future.
    pub fn forward_looking_ratio(&self, expressions: &[TemporalExpression], reference_time: i64) -> f64 {
        let mut future_count = 0usize;
        let mut past_count = 0usize;

        for expr in expressions {
            if let Some(start) = expr.normalized_start {
                if start > reference_time {
                    future_count += 1;
                } else {
                    past_count += 1;
                }
            }
        }

        future_count as f64 / (past_count as f64 + 1e-9)
    }

    /// Compute temporal coherence: fraction of event pairs in chronological order.
    pub fn temporal_coherence(&self, events: &[FinancialEvent]) -> f64 {
        if events.len() < 2 {
            return 1.0;
        }
        let mut concordant = 0usize;
        let mut total = 0usize;

        for i in 0..events.len() {
            for j in (i + 1)..events.len() {
                total += 1;
                if events[i].timestamp <= events[j].timestamp {
                    concordant += 1;
                }
            }
        }

        concordant as f64 / total as f64
    }

    /// Compute a full feature vector from text and events.
    /// Returns [temporal_density, avg_recency, forward_looking_ratio, coherence, event_count].
    pub fn compute_features(
        &self,
        text: &str,
        events: &[FinancialEvent],
        current_time: i64,
    ) -> Vec<f64> {
        let expressions = self.extractor.extract(text);
        let density = self.temporal_density(text);

        let avg_recency = if events.is_empty() {
            0.0
        } else {
            let sum: f64 = events
                .iter()
                .map(|e| self.recency_score(e.timestamp, current_time))
                .sum();
            sum / events.len() as f64
        };

        let flr = self.forward_looking_ratio(&expressions, current_time);
        let coherence = self.temporal_coherence(events);
        let event_count = events.len() as f64;

        vec![density, avg_recency, flr, coherence, event_count]
    }

    /// Access the underlying expression extractor.
    pub fn extractor(&self) -> &TemporalExpressionExtractor {
        &self.extractor
    }
}

// ─── Temporal Trading Strategy ────────────────────────────────────

/// A trading signal generated by temporal analysis.
#[derive(Debug, Clone)]
pub struct TemporalSignal {
    pub timestamp: i64,
    pub direction: f64,
    pub strength: f64,
    pub source_event: String,
}

/// Position tracking for backtesting.
#[derive(Debug, Clone)]
pub struct Position {
    pub entry_price: f64,
    pub entry_time: i64,
    pub size: f64,
    pub direction: f64,
    pub exit_time: Option<i64>,
    pub exit_price: Option<f64>,
    pub pnl: Option<f64>,
}

/// Backtesting results.
#[derive(Debug, Clone)]
pub struct BacktestResult {
    pub total_trades: usize,
    pub winning_trades: usize,
    pub losing_trades: usize,
    pub total_pnl: f64,
    pub max_drawdown: f64,
    pub sharpe_ratio: f64,
    pub win_rate: f64,
}

/// Temporal trading strategy that generates signals from event timelines and features.
pub struct TemporalTradingStrategy {
    feature_extractor: TemporalFeatureExtractor,
    classifier: TemporalClassifier,
    signal_threshold: f64,
    max_holding_period: i64,
}

impl TemporalTradingStrategy {
    pub fn new(signal_threshold: f64, max_holding_period_secs: i64) -> Self {
        Self {
            feature_extractor: TemporalFeatureExtractor::new(0.1, 86400.0),
            classifier: TemporalClassifier::new(5, 0.01),
            signal_threshold,
            max_holding_period: max_holding_period_secs,
        }
    }

    /// Train the strategy's internal classifier on historical data.
    pub fn train(&mut self, data: &[(Vec<f64>, f64)], epochs: usize) {
        self.classifier.train(data, epochs);
    }

    /// Generate a signal from temporal features.
    pub fn generate_signal(
        &self,
        text: &str,
        events: &[FinancialEvent],
        current_time: i64,
    ) -> Option<TemporalSignal> {
        let features = self
            .feature_extractor
            .compute_features(text, events, current_time);
        let (direction, confidence) = self.classifier.predict(&features);

        if confidence >= self.signal_threshold {
            Some(TemporalSignal {
                timestamp: current_time,
                direction: if direction { 1.0 } else { -1.0 },
                strength: confidence,
                source_event: if events.is_empty() {
                    "temporal_features".to_string()
                } else {
                    events.last().unwrap().name.clone()
                },
            })
        } else {
            None
        }
    }

    /// Run a backtest on kline data with temporal signals.
    pub fn backtest(
        &self,
        klines: &[Kline],
        signals: &[TemporalSignal],
    ) -> BacktestResult {
        let mut positions: Vec<Position> = Vec::new();
        let mut equity_curve = vec![0.0f64];
        let mut current_position: Option<Position> = None;

        let kline_map: HashMap<i64, &Kline> = klines
            .iter()
            .map(|k| (k.timestamp as i64, k))
            .collect();

        for signal in signals {
            // Close existing position if holding period expired
            if let Some(ref pos) = current_position {
                if signal.timestamp - pos.entry_time > self.max_holding_period {
                    if let Some(kline) = kline_map.get(&signal.timestamp) {
                        let mut closed = pos.clone();
                        closed.exit_time = Some(signal.timestamp);
                        closed.exit_price = Some(kline.close);
                        closed.pnl = Some(
                            (kline.close - pos.entry_price) * pos.direction * pos.size,
                        );
                        let last_eq = *equity_curve.last().unwrap();
                        equity_curve.push(last_eq + closed.pnl.unwrap());
                        positions.push(closed);
                        current_position = None;
                    }
                }
            }

            // Open new position if no current position
            if current_position.is_none() {
                if let Some(kline) = kline_map.get(&signal.timestamp) {
                    current_position = Some(Position {
                        entry_price: kline.close,
                        entry_time: signal.timestamp,
                        size: signal.strength,
                        direction: signal.direction,
                        exit_time: None,
                        exit_price: None,
                        pnl: None,
                    });
                }
            }
        }

        // Close any remaining position at the last kline
        if let Some(pos) = current_position {
            if let Some(last_kline) = klines.last() {
                let mut closed = pos.clone();
                closed.exit_time = Some(last_kline.timestamp as i64);
                closed.exit_price = Some(last_kline.close);
                closed.pnl = Some(
                    (last_kline.close - pos.entry_price) * pos.direction * pos.size,
                );
                let last_eq = *equity_curve.last().unwrap();
                equity_curve.push(last_eq + closed.pnl.unwrap());
                positions.push(closed);
            }
        }

        // Compute metrics
        let winning = positions.iter().filter(|p| p.pnl.unwrap_or(0.0) > 0.0).count();
        let losing = positions.iter().filter(|p| p.pnl.unwrap_or(0.0) < 0.0).count();
        let total_pnl: f64 = positions.iter().map(|p| p.pnl.unwrap_or(0.0)).sum();

        let max_drawdown = compute_max_drawdown(&equity_curve);
        let sharpe = compute_sharpe_ratio(&equity_curve);
        let win_rate = if positions.is_empty() {
            0.0
        } else {
            winning as f64 / positions.len() as f64
        };

        BacktestResult {
            total_trades: positions.len(),
            winning_trades: winning,
            losing_trades: losing,
            total_pnl,
            max_drawdown,
            sharpe_ratio: sharpe,
            win_rate,
        }
    }

    pub fn feature_extractor(&self) -> &TemporalFeatureExtractor {
        &self.feature_extractor
    }
}

// ─── Temporal Classifier (Logistic Regression) ────────────────────

/// Binary logistic regression classifier for temporal signal prediction.
///
/// Features: [temporal_density, avg_recency, forward_looking_ratio, coherence, event_count]
#[derive(Debug)]
pub struct TemporalClassifier {
    weights: Array1<f64>,
    bias: f64,
    learning_rate: f64,
    num_features: usize,
}

impl TemporalClassifier {
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

    /// Predict probability that signal is bullish (label = 1).
    pub fn predict_proba(&self, features: &[f64]) -> f64 {
        assert_eq!(features.len(), self.num_features);
        let x = Array1::from_vec(features.to_vec());
        let z = self.weights.dot(&x) + self.bias;
        Self::sigmoid(z)
    }

    /// Predict direction: true = bullish, false = bearish. Returns (direction, confidence).
    pub fn predict(&self, features: &[f64]) -> (bool, f64) {
        let prob = self.predict_proba(features);
        if prob >= 0.5 {
            (true, prob)
        } else {
            (false, 1.0 - prob)
        }
    }

    /// Train on a dataset of (features, label) pairs for `epochs` iterations.
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
    pub b: Vec<Vec<String>>,
    pub a: Vec<Vec<String>>,
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

// ─── Utility Functions ────────────────────────────────────────────

/// Compute maximum drawdown from an equity curve.
pub fn compute_max_drawdown(equity: &[f64]) -> f64 {
    let mut max_dd = 0.0f64;
    let mut peak = f64::NEG_INFINITY;

    for &val in equity {
        if val > peak {
            peak = val;
        }
        let dd = peak - val;
        if dd > max_dd {
            max_dd = dd;
        }
    }
    max_dd
}

/// Compute Sharpe ratio from an equity curve (assuming daily returns).
pub fn compute_sharpe_ratio(equity: &[f64]) -> f64 {
    if equity.len() < 2 {
        return 0.0;
    }
    let returns: Vec<f64> = equity
        .windows(2)
        .map(|w| w[1] - w[0])
        .collect();

    let mean = returns.iter().sum::<f64>() / returns.len() as f64;
    let variance = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;
    let std_dev = variance.sqrt();

    if std_dev < 1e-12 {
        return 0.0;
    }
    mean / std_dev
}

// ─── Synthetic Data Generation ────────────────────────────────────

/// Generate synthetic financial events for testing.
pub fn generate_synthetic_events(n: usize, base_timestamp: i64) -> Vec<FinancialEvent> {
    let mut rng = rand::thread_rng();
    let event_types = [
        EventType::Earnings,
        EventType::FomcDecision,
        EventType::Halving,
        EventType::TokenUnlock,
        EventType::RegulatoryDeadline,
        EventType::ProductLaunch,
    ];
    let entities = ["AAPL", "BTCUSD", "ETHUSDT", "GOOGL", "MSFT", "SOLUSDT"];

    let mut events = Vec::with_capacity(n);
    let mut t = base_timestamp;

    for i in 0..n {
        let event_type = event_types[rng.gen_range(0..event_types.len())];
        let entity = entities[rng.gen_range(0..entities.len())];
        let gap = rng.gen_range(3600..86400 * 7); // 1 hour to 7 days
        t += gap;

        events.push(FinancialEvent {
            name: format!("Event_{}", i),
            event_type,
            entity: entity.to_string(),
            timestamp: t,
            duration_secs: Some(rng.gen_range(1800..86400)),
            impact_score: rng.gen_range(0.1..1.0),
        });
    }
    events
}

/// Generate synthetic temporal training data.
///
/// Features: [temporal_density, avg_recency, forward_looking_ratio, coherence, event_count]
/// Label: 1.0 = price up, 0.0 = price down
pub fn generate_training_data(n: usize) -> Vec<(Vec<f64>, f64)> {
    let mut rng = rand::thread_rng();
    let mut data = Vec::with_capacity(n);

    for _ in 0..n {
        let density: f64 = rng.gen_range(0.0..0.3);
        let recency: f64 = rng.gen_range(0.0..1.0);
        let flr: f64 = rng.gen_range(0.0..5.0);
        let coherence: f64 = rng.gen_range(0.5..1.0);
        let event_count: f64 = rng.gen_range(0.0..20.0);

        // Signal: high recency + high FLR + high density = bullish
        let signal = 0.3 * recency + 0.25 * flr + 0.2 * density * 10.0
            + 0.15 * coherence
            + 0.1 * (event_count / 20.0)
            - 0.5;
        let prob = 1.0 / (1.0 + (-signal).exp());
        let label = if rng.gen::<f64>() < prob { 1.0 } else { 0.0 };

        data.push((vec![density, recency, flr, coherence, event_count], label));
    }
    data
}

/// Generate sample financial text for testing temporal extraction.
pub fn generate_sample_financial_text() -> &'static str {
    "Apple Inc. will report Q3 2024 earnings on August 1, 2024. Analysts expect revenue guidance \
     for the holiday quarter. The Federal Reserve's next FOMC decision is scheduled for \
     September 18, 2024, following the July meeting where rates were held steady. \
     Over the past 6 months, Bitcoin has rallied significantly. The next Bitcoin halving \
     occurred in April 2024, reducing block rewards for the next 4 years. \
     Ethereum's Dencun upgrade was completed last quarter. Token unlock schedules show \
     $500 million in SOL tokens unlocking next month. FY2024 guidance suggests \
     revenue growth of 15% compared to fiscal year 2023. The T+1 settlement rule \
     took effect on May 28, 2024, reducing settlement risk."
}

// ─── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temporal_relation_from_intervals() {
        assert_eq!(
            TemporalRelation::from_intervals(1, 3, 5, 7),
            TemporalRelation::Before
        );
        assert_eq!(
            TemporalRelation::from_intervals(1, 5, 5, 7),
            TemporalRelation::Meets
        );
        assert_eq!(
            TemporalRelation::from_intervals(1, 6, 3, 8),
            TemporalRelation::Overlaps
        );
        assert_eq!(
            TemporalRelation::from_intervals(3, 5, 1, 7),
            TemporalRelation::During
        );
        assert_eq!(
            TemporalRelation::from_intervals(1, 7, 1, 7),
            TemporalRelation::Equals
        );
        assert_eq!(
            TemporalRelation::from_intervals(5, 7, 1, 3),
            TemporalRelation::After
        );
    }

    #[test]
    fn test_temporal_relation_inverse() {
        assert_eq!(
            TemporalRelation::Before.inverse(),
            TemporalRelation::After
        );
        assert_eq!(
            TemporalRelation::Meets.inverse(),
            TemporalRelation::MetBy
        );
        assert_eq!(
            TemporalRelation::Overlaps.inverse(),
            TemporalRelation::OverlappedBy
        );
        assert_eq!(
            TemporalRelation::Equals.inverse(),
            TemporalRelation::Equals
        );
    }

    #[test]
    fn test_event_timeline_insert_and_query() {
        let mut timeline = EventTimeline::new();

        timeline.insert(FinancialEvent {
            name: "Earnings".to_string(),
            event_type: EventType::Earnings,
            entity: "AAPL".to_string(),
            timestamp: 100,
            duration_secs: Some(10),
            impact_score: 0.8,
        });

        timeline.insert(FinancialEvent {
            name: "FOMC".to_string(),
            event_type: EventType::FomcDecision,
            entity: "FED".to_string(),
            timestamp: 50,
            duration_secs: Some(20),
            impact_score: 0.9,
        });

        timeline.insert(FinancialEvent {
            name: "Halving".to_string(),
            event_type: EventType::Halving,
            entity: "BTC".to_string(),
            timestamp: 200,
            duration_secs: None,
            impact_score: 0.95,
        });

        assert_eq!(timeline.len(), 3);
        // Should be sorted: FOMC(50), Earnings(100), Halving(200)
        assert_eq!(timeline.events()[0].name, "FOMC");
        assert_eq!(timeline.events()[1].name, "Earnings");
        assert_eq!(timeline.events()[2].name, "Halving");

        // Query range
        let results = timeline.query_range(60, 150);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "Earnings");
    }

    #[test]
    fn test_event_timeline_gaps() {
        let mut timeline = EventTimeline::new();

        timeline.insert(FinancialEvent {
            name: "A".to_string(),
            event_type: EventType::Earnings,
            entity: "X".to_string(),
            timestamp: 100,
            duration_secs: Some(10),
            impact_score: 0.5,
        });

        timeline.insert(FinancialEvent {
            name: "B".to_string(),
            event_type: EventType::Earnings,
            entity: "X".to_string(),
            timestamp: 200,
            duration_secs: Some(10),
            impact_score: 0.5,
        });

        // Gap from 110 to 200 = 90 seconds
        let gaps = timeline.detect_gaps(50);
        assert_eq!(gaps.len(), 1);
        assert_eq!(gaps[0], (110, 200));

        let gaps_large = timeline.detect_gaps(100);
        assert_eq!(gaps_large.len(), 0);
    }

    #[test]
    fn test_event_timeline_relation() {
        let mut timeline = EventTimeline::new();

        timeline.insert(FinancialEvent {
            name: "A".to_string(),
            event_type: EventType::Earnings,
            entity: "X".to_string(),
            timestamp: 10,
            duration_secs: Some(5),
            impact_score: 0.5,
        });

        timeline.insert(FinancialEvent {
            name: "B".to_string(),
            event_type: EventType::FomcDecision,
            entity: "FED".to_string(),
            timestamp: 20,
            duration_secs: Some(5),
            impact_score: 0.5,
        });

        let rel = timeline.relation(0, 1).unwrap();
        assert_eq!(rel, TemporalRelation::Before);
    }

    #[test]
    fn test_temporal_expression_extraction() {
        let extractor = TemporalExpressionExtractor::new();
        let text = generate_sample_financial_text();
        let expressions = extractor.extract(text);

        assert!(!expressions.is_empty());

        // Should find dates, quarters, fiscal years
        let types: Vec<TemporalExpressionType> = expressions.iter().map(|e| e.expr_type).collect();
        assert!(types.contains(&TemporalExpressionType::FiscalQuarter));
        assert!(types.contains(&TemporalExpressionType::Date));
        assert!(types.contains(&TemporalExpressionType::FiscalYear));
    }

    #[test]
    fn test_temporal_expression_specific_patterns() {
        let extractor = TemporalExpressionExtractor::new();

        let text = "Q1 2024 earnings were strong. FY2025 guidance raised.";
        let exprs = extractor.extract(text);
        let texts: Vec<&str> = exprs.iter().map(|e| e.text.as_str()).collect();
        assert!(texts.iter().any(|t| t.contains("Q1")));
        assert!(texts.iter().any(|t| t.contains("FY2025")));
    }

    #[test]
    fn test_temporal_density() {
        let feat = TemporalFeatureExtractor::new(0.1, 86400.0);

        let dense_text = "Q1 2024 report on January 15, 2024. FY2024 guidance for next quarter.";
        let sparse_text = "The company is doing well and growing steadily in the market.";

        let dense_score = feat.temporal_density(dense_text);
        let sparse_score = feat.temporal_density(sparse_text);

        assert!(dense_score > sparse_score);
    }

    #[test]
    fn test_recency_score() {
        let feat = TemporalFeatureExtractor::new(0.1, 86400.0);
        let now = 1000000;

        let recent = feat.recency_score(now - 86400, now);
        let old = feat.recency_score(now - 864000, now);

        assert!(recent > old);
        assert!(recent > 0.0 && recent <= 1.0);
        assert!(old > 0.0 && old <= 1.0);
    }

    #[test]
    fn test_temporal_coherence() {
        let feat = TemporalFeatureExtractor::new(0.1, 86400.0);

        let ordered_events = vec![
            FinancialEvent {
                name: "A".to_string(),
                event_type: EventType::Earnings,
                entity: "X".to_string(),
                timestamp: 100,
                duration_secs: None,
                impact_score: 0.5,
            },
            FinancialEvent {
                name: "B".to_string(),
                event_type: EventType::Earnings,
                entity: "X".to_string(),
                timestamp: 200,
                duration_secs: None,
                impact_score: 0.5,
            },
            FinancialEvent {
                name: "C".to_string(),
                event_type: EventType::Earnings,
                entity: "X".to_string(),
                timestamp: 300,
                duration_secs: None,
                impact_score: 0.5,
            },
        ];

        assert!((feat.temporal_coherence(&ordered_events) - 1.0).abs() < 1e-9);

        let unordered_events = vec![
            FinancialEvent {
                name: "C".to_string(),
                event_type: EventType::Earnings,
                entity: "X".to_string(),
                timestamp: 300,
                duration_secs: None,
                impact_score: 0.5,
            },
            FinancialEvent {
                name: "A".to_string(),
                event_type: EventType::Earnings,
                entity: "X".to_string(),
                timestamp: 100,
                duration_secs: None,
                impact_score: 0.5,
            },
            FinancialEvent {
                name: "B".to_string(),
                event_type: EventType::Earnings,
                entity: "X".to_string(),
                timestamp: 200,
                duration_secs: None,
                impact_score: 0.5,
            },
        ];

        let coherence = feat.temporal_coherence(&unordered_events);
        assert!(coherence < 1.0);
    }

    #[test]
    fn test_classifier_predict() {
        let clf = TemporalClassifier::new(5, 0.01);
        let features = vec![0.1, 0.8, 2.0, 0.9, 5.0];
        let (_, confidence) = clf.predict(&features);
        assert!(confidence >= 0.5 && confidence <= 1.0);
    }

    #[test]
    fn test_classifier_train_and_improve() {
        let data = generate_training_data(500);
        let (train, test) = data.split_at(400);

        let mut clf = TemporalClassifier::new(5, 0.01);
        clf.train(&train.to_vec(), 50);
        let acc = clf.accuracy(test);

        assert!(acc > 0.0);
        assert!(acc >= 0.4, "accuracy after training: {}", acc);
    }

    #[test]
    fn test_compute_max_drawdown() {
        let equity = vec![0.0, 1.0, 2.0, 1.5, 3.0, 2.0, 4.0];
        let dd = compute_max_drawdown(&equity);
        assert!((dd - 1.0).abs() < 1e-9); // peak 3.0 to trough 2.0
    }

    #[test]
    fn test_compute_sharpe_ratio() {
        let equity = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let sharpe = compute_sharpe_ratio(&equity);
        // All returns are 1.0, so std_dev ≈ 0, sharpe should be 0 (or very large if mean > 0 and std tiny)
        // With constant returns, variance = 0, so sharpe = 0
        assert!((sharpe - 0.0).abs() < 1e-6);

        let equity2 = vec![0.0, 1.0, 0.5, 1.5, 1.0];
        let sharpe2 = compute_sharpe_ratio(&equity2);
        assert!(sharpe2.is_finite());
    }

    #[test]
    fn test_synthetic_event_generation() {
        let events = generate_synthetic_events(10, 1700000000);
        assert_eq!(events.len(), 10);
        // Events should be in chronological order
        for window in events.windows(2) {
            assert!(window[0].timestamp < window[1].timestamp);
        }
    }

    #[test]
    fn test_synthetic_training_data() {
        let data = generate_training_data(100);
        assert_eq!(data.len(), 100);
        for (features, label) in &data {
            assert_eq!(features.len(), 5);
            assert!(*label == 0.0 || *label == 1.0);
        }
    }

    #[test]
    fn test_feature_vector_computation() {
        let feat = TemporalFeatureExtractor::new(0.1, 86400.0);
        let text = generate_sample_financial_text();
        let events = generate_synthetic_events(5, 1700000000);
        let current_time = 1700500000;

        let features = feat.compute_features(text, &events, current_time);
        assert_eq!(features.len(), 5);
        assert!(features[0] >= 0.0); // density
        assert!(features[1] >= 0.0 && features[1] <= 1.0); // recency
        assert!(features[3] >= 0.0 && features[3] <= 1.0); // coherence
        assert!(features[4] >= 0.0); // event count
    }
}
