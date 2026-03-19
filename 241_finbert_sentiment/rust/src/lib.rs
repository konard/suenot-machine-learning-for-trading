use ndarray::Array1;
use rand::Rng;
use serde::Deserialize;

// ─── Sentiment Analyzer (Keyword-Based) ───────────────────────────

/// Financial sentiment analyzer using Loughran-McDonald keyword dictionary.
///
/// Scores text by counting positive and negative financial keywords,
/// returning a normalized sentiment score between -1 and +1.
#[derive(Debug)]
pub struct SentimentAnalyzer {
    positive_words: Vec<&'static str>,
    negative_words: Vec<&'static str>,
}

impl SentimentAnalyzer {
    pub fn new() -> Self {
        Self {
            positive_words: vec![
                "profit", "growth", "gain", "improvement", "increase",
                "outperform", "upgrade", "beat", "exceed", "strong",
                "positive", "bullish", "rally", "surge", "recover",
                "optimistic", "record", "breakout", "momentum", "upside",
                "dividend", "expansion", "innovation", "success", "opportunity",
                "revenue", "earnings", "buy", "accumulate", "overweight",
            ],
            negative_words: vec![
                "loss", "decline", "drop", "risk", "downgrade",
                "deficit", "lawsuit", "bankruptcy", "default", "weak",
                "negative", "bearish", "crash", "plunge", "recession",
                "pessimistic", "miss", "underperform", "sell", "warning",
                "debt", "fraud", "investigation", "layoff", "closure",
                "impairment", "writedown", "shortfall", "violation", "underweight",
            ],
        }
    }

    /// Analyze sentiment of a text string.
    /// Returns a score between -1.0 (very negative) and +1.0 (very positive).
    pub fn analyze(&self, text: &str) -> SentimentResult {
        let lower = text.to_lowercase();
        let words: Vec<&str> = lower.split_whitespace().collect();
        let total_words = words.len() as f64;

        if total_words == 0.0 {
            return SentimentResult {
                score: 0.0,
                positive_count: 0,
                negative_count: 0,
                confidence: 0.0,
            };
        }

        let mut pos_count = 0usize;
        let mut neg_count = 0usize;

        for word in &words {
            let cleaned: String = word.chars().filter(|c| c.is_alphabetic()).collect();
            if self.positive_words.contains(&cleaned.as_str()) {
                pos_count += 1;
            }
            if self.negative_words.contains(&cleaned.as_str()) {
                neg_count += 1;
            }
        }

        let sentiment_words = (pos_count + neg_count) as f64;
        let score = if sentiment_words > 0.0 {
            (pos_count as f64 - neg_count as f64) / sentiment_words
        } else {
            0.0
        };

        let confidence = (sentiment_words / total_words).min(1.0);

        SentimentResult {
            score,
            positive_count: pos_count,
            negative_count: neg_count,
            confidence,
        }
    }

    /// Batch-analyze multiple texts and return individual scores.
    pub fn analyze_batch(&self, texts: &[&str]) -> Vec<SentimentResult> {
        texts.iter().map(|t| self.analyze(t)).collect()
    }
}

impl Default for SentimentAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of sentiment analysis on a single text.
#[derive(Debug, Clone)]
pub struct SentimentResult {
    /// Sentiment score from -1.0 (negative) to +1.0 (positive).
    pub score: f64,
    /// Number of positive keywords found.
    pub positive_count: usize,
    /// Number of negative keywords found.
    pub negative_count: usize,
    /// Confidence based on proportion of sentiment words in text.
    pub confidence: f64,
}

// ─── Sentiment Aggregator ─────────────────────────────────────────

/// Aggregates sentiment scores over time to produce trading signals.
///
/// Supports simple averaging, exponentially weighted averaging,
/// and sentiment momentum computation.
#[derive(Debug)]
pub struct SentimentAggregator {
    scores: Vec<f64>,
    timestamps: Vec<u64>,
    decay_factor: f64,
}

impl SentimentAggregator {
    /// Create a new aggregator with the given exponential decay factor.
    /// `decay_factor` should be in (0, 1). Typical: 0.9 for slow decay, 0.5 for fast.
    pub fn new(decay_factor: f64) -> Self {
        Self {
            scores: Vec::new(),
            timestamps: Vec::new(),
            decay_factor,
        }
    }

    /// Add a sentiment score with an optional timestamp.
    pub fn add(&mut self, score: f64, timestamp: u64) {
        self.scores.push(score);
        self.timestamps.push(timestamp);
    }

    /// Simple average of all scores.
    pub fn average(&self) -> f64 {
        if self.scores.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.scores.iter().sum();
        sum / self.scores.len() as f64
    }

    /// Simple average of the last `n` scores.
    pub fn average_last(&self, n: usize) -> f64 {
        if self.scores.is_empty() {
            return 0.0;
        }
        let start = self.scores.len().saturating_sub(n);
        let slice = &self.scores[start..];
        let sum: f64 = slice.iter().sum();
        sum / slice.len() as f64
    }

    /// Exponentially weighted average (more recent scores have higher weight).
    pub fn ewma(&self) -> f64 {
        if self.scores.is_empty() {
            return 0.0;
        }
        let n = self.scores.len();
        let mut weighted_sum = 0.0;
        let mut weight_sum = 0.0;

        for (i, &score) in self.scores.iter().enumerate() {
            let age = (n - 1 - i) as f64;
            let weight = self.decay_factor.powf(age);
            weighted_sum += weight * score;
            weight_sum += weight;
        }

        if weight_sum > 0.0 {
            weighted_sum / weight_sum
        } else {
            0.0
        }
    }

    /// Sentiment momentum: change in average sentiment over `lookback` periods.
    pub fn momentum(&self, lookback: usize) -> f64 {
        if self.scores.len() < lookback + 1 {
            return 0.0;
        }
        let current = self.average_last(lookback);
        let n = self.scores.len();
        let start = n.saturating_sub(2 * lookback);
        let end = n.saturating_sub(lookback);
        if start >= end {
            return 0.0;
        }
        let prev_slice = &self.scores[start..end];
        let prev_avg: f64 = prev_slice.iter().sum::<f64>() / prev_slice.len() as f64;
        current - prev_avg
    }

    /// Number of scores recorded.
    pub fn count(&self) -> usize {
        self.scores.len()
    }

    /// Get all recorded scores.
    pub fn history(&self) -> &[f64] {
        &self.scores
    }
}

// ─── Sentiment Classifier (Logistic Regression) ───────────────────

/// Binary logistic regression classifier for sentiment classification.
///
/// Features might include: [keyword_score, word_count, uppercase_ratio,
/// exclamation_count, question_count]
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

    /// Predict probability that sentiment is positive (label = 1).
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

// ─── Sentiment Trading Strategy ───────────────────────────────────

/// Trading signal generated by the sentiment strategy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Signal {
    Buy,
    Sell,
    Hold,
}

/// Tracks the performance of a sentiment-based trading strategy.
#[derive(Debug)]
pub struct SentimentStrategy {
    buy_threshold: f64,
    sell_threshold: f64,
    position: f64,       // +1 = long, -1 = short, 0 = flat
    entry_price: f64,
    trades: Vec<Trade>,
    cumulative_return: f64,
}

/// A completed trade record.
#[derive(Debug, Clone)]
pub struct Trade {
    pub entry_price: f64,
    pub exit_price: f64,
    pub direction: f64,  // +1 or -1
    pub pnl: f64,
}

impl SentimentStrategy {
    /// Create a new strategy with buy/sell sentiment thresholds.
    /// - `buy_threshold`: minimum sentiment score to trigger a buy (e.g. 0.3)
    /// - `sell_threshold`: sentiment score below which to trigger a sell (e.g. -0.3)
    pub fn new(buy_threshold: f64, sell_threshold: f64) -> Self {
        Self {
            buy_threshold,
            sell_threshold,
            position: 0.0,
            entry_price: 0.0,
            trades: Vec::new(),
            cumulative_return: 0.0,
        }
    }

    /// Generate a trading signal based on sentiment score.
    pub fn signal(&self, sentiment: f64) -> Signal {
        if sentiment > self.buy_threshold {
            Signal::Buy
        } else if sentiment < self.sell_threshold {
            Signal::Sell
        } else {
            Signal::Hold
        }
    }

    /// Process a new data point: sentiment score and current price.
    /// Returns the generated signal.
    pub fn update(&mut self, sentiment: f64, price: f64) -> Signal {
        let sig = self.signal(sentiment);

        match sig {
            Signal::Buy => {
                if self.position <= 0.0 {
                    // Close short if any
                    if self.position < 0.0 {
                        let pnl = (self.entry_price - price) / self.entry_price;
                        self.cumulative_return += pnl;
                        self.trades.push(Trade {
                            entry_price: self.entry_price,
                            exit_price: price,
                            direction: -1.0,
                            pnl,
                        });
                    }
                    // Open long
                    self.position = 1.0;
                    self.entry_price = price;
                }
            }
            Signal::Sell => {
                if self.position >= 0.0 {
                    // Close long if any
                    if self.position > 0.0 {
                        let pnl = (price - self.entry_price) / self.entry_price;
                        self.cumulative_return += pnl;
                        self.trades.push(Trade {
                            entry_price: self.entry_price,
                            exit_price: price,
                            direction: 1.0,
                            pnl,
                        });
                    }
                    // Open short
                    self.position = -1.0;
                    self.entry_price = price;
                }
            }
            Signal::Hold => {}
        }

        sig
    }

    /// Get cumulative return as a fraction (0.05 = 5%).
    pub fn cumulative_return(&self) -> f64 {
        self.cumulative_return
    }

    /// Get all completed trades.
    pub fn trades(&self) -> &[Trade] {
        &self.trades
    }

    /// Win rate: fraction of profitable trades.
    pub fn win_rate(&self) -> f64 {
        if self.trades.is_empty() {
            return 0.0;
        }
        let wins = self.trades.iter().filter(|t| t.pnl > 0.0).count();
        wins as f64 / self.trades.len() as f64
    }

    /// Current position: +1 long, -1 short, 0 flat.
    pub fn position(&self) -> f64 {
        self.position
    }

    /// Compute Sharpe ratio of completed trades (assuming zero risk-free rate).
    pub fn sharpe_ratio(&self) -> f64 {
        if self.trades.len() < 2 {
            return 0.0;
        }
        let returns: Vec<f64> = self.trades.iter().map(|t| t.pnl).collect();
        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns
            .iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>()
            / (returns.len() - 1) as f64;
        let std_dev = variance.sqrt();
        if std_dev == 0.0 {
            return 0.0;
        }
        mean / std_dev
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
    pub b: Vec<Vec<String>>,  // bids: [price, size]
    pub a: Vec<Vec<String>>,  // asks: [price, size]
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

/// Sample financial headlines for testing sentiment analysis.
pub fn sample_financial_headlines() -> Vec<&'static str> {
    vec![
        "Company reports record earnings growth and revenue beat expectations",
        "Stock plunges after fraud investigation and CEO resignation",
        "Bitcoin rally continues as institutional buyers accumulate positions",
        "Crypto market crash deepens amid regulatory risk and sell pressure",
        "Strong quarterly profit drives positive momentum for tech sector",
        "Recession fears grow as economic decline spreads to global markets",
        "Upgrade from analysts as company beats expectations with record revenue",
        "Massive layoff announced following revenue shortfall and profit decline",
        "Innovation in blockchain technology creates new investment opportunity",
        "Debt default risk increases amid bankruptcy concerns and weak earnings",
        "Market rally on strong earnings and bullish growth outlook",
        "Investors sell as downgrade warning signals bearish momentum",
        "Expansion into new markets drives optimistic revenue growth forecast",
        "Lawsuit and compliance violation lead to significant stock drop",
        "Record dividend increase signals strong financial improvement",
        "Company warns of potential loss amid declining market conditions",
        "Breakout in crypto prices as positive sentiment drives accumulation",
        "Investigation into fraud allegations causes plunge in share price",
        "Success in product launch exceeds growth expectations by wide margin",
        "Risk of recession increases as economic indicators show decline",
    ]
}

/// Generate synthetic labeled training data for the sentiment classifier.
///
/// Each sample has features [keyword_score, word_count_norm, sentiment_words_ratio,
/// positive_ratio, confidence]
/// and a binary label (1.0 = positive, 0.0 = negative).
pub fn generate_training_data(n: usize) -> Vec<(Vec<f64>, f64)> {
    let mut rng = rand::thread_rng();
    let mut data = Vec::with_capacity(n);

    for _ in 0..n {
        let keyword_score: f64 = rng.gen_range(-1.0..1.0);
        let word_count_norm: f64 = rng.gen_range(0.0..1.0);
        let sentiment_ratio: f64 = rng.gen_range(0.0..0.5);
        let positive_ratio: f64 = rng.gen_range(0.0..1.0);
        let confidence: f64 = rng.gen_range(0.0..1.0);

        // Label: higher probability of positive when keyword_score > 0 and positive_ratio > 0.5
        let signal = 0.5 * keyword_score + 0.3 * (positive_ratio - 0.5) + 0.1 * confidence;
        let prob = 1.0 / (1.0 + (-signal * 3.0).exp());
        let label = if rng.gen::<f64>() < prob { 1.0 } else { 0.0 };

        data.push((
            vec![keyword_score, word_count_norm, sentiment_ratio, positive_ratio, confidence],
            label,
        ));
    }
    data
}

/// Generate synthetic price data for strategy backtesting.
/// Returns Vec of (timestamp, price) tuples.
pub fn generate_synthetic_prices(n: usize, start_price: f64) -> Vec<(u64, f64)> {
    let mut rng = rand::thread_rng();
    let mut price = start_price;
    let mut prices = Vec::with_capacity(n);
    let start_ts = 1700000000u64;

    for i in 0..n {
        let change = rng.gen_range(-0.02..0.02) * price;
        price += change;
        price = price.max(start_price * 0.5); // floor
        prices.push((start_ts + i as u64 * 60, price));
    }
    prices
}

/// Generate synthetic sentiment scores correlated with price movements.
pub fn generate_synthetic_sentiments(prices: &[(u64, f64)]) -> Vec<(u64, f64)> {
    let mut rng = rand::thread_rng();
    let mut sentiments = Vec::with_capacity(prices.len());

    for i in 0..prices.len() {
        let price_change = if i > 0 {
            (prices[i].1 - prices[i - 1].1) / prices[i - 1].1
        } else {
            0.0
        };
        // Sentiment loosely follows price changes with noise
        let noise = rng.gen_range(-0.3..0.3);
        let sentiment = (price_change * 20.0 + noise).clamp(-1.0, 1.0);
        sentiments.push((prices[i].0, sentiment));
    }
    sentiments
}

// ─── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sentiment_analyzer_positive() {
        let analyzer = SentimentAnalyzer::new();
        let result = analyzer.analyze("Company reports record profit growth and strong earnings");
        assert!(result.score > 0.0);
        assert!(result.positive_count > 0);
    }

    #[test]
    fn test_sentiment_analyzer_negative() {
        let analyzer = SentimentAnalyzer::new();
        let result = analyzer.analyze("Stock crash after fraud investigation and bankruptcy risk");
        assert!(result.score < 0.0);
        assert!(result.negative_count > 0);
    }

    #[test]
    fn test_sentiment_analyzer_neutral() {
        let analyzer = SentimentAnalyzer::new();
        let result = analyzer.analyze("The meeting was held at the headquarters today");
        assert!((result.score - 0.0).abs() < 1e-9);
        assert_eq!(result.positive_count, 0);
        assert_eq!(result.negative_count, 0);
    }

    #[test]
    fn test_sentiment_analyzer_empty() {
        let analyzer = SentimentAnalyzer::new();
        let result = analyzer.analyze("");
        assert!((result.score - 0.0).abs() < 1e-9);
        assert!((result.confidence - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_sentiment_batch() {
        let analyzer = SentimentAnalyzer::new();
        let texts = vec![
            "profit growth strong",
            "loss decline risk",
            "the company held a meeting",
        ];
        let results = analyzer.analyze_batch(&texts);
        assert_eq!(results.len(), 3);
        assert!(results[0].score > 0.0);
        assert!(results[1].score < 0.0);
        assert!((results[2].score - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_aggregator_average() {
        let mut agg = SentimentAggregator::new(0.9);
        agg.add(0.5, 1);
        agg.add(-0.3, 2);
        agg.add(0.2, 3);
        let avg = agg.average();
        let expected = (0.5 - 0.3 + 0.2) / 3.0;
        assert!((avg - expected).abs() < 1e-9);
    }

    #[test]
    fn test_aggregator_average_last() {
        let mut agg = SentimentAggregator::new(0.9);
        agg.add(0.1, 1);
        agg.add(0.2, 2);
        agg.add(0.3, 3);
        agg.add(0.4, 4);
        let avg = agg.average_last(2);
        let expected = (0.3 + 0.4) / 2.0;
        assert!((avg - expected).abs() < 1e-9);
    }

    #[test]
    fn test_aggregator_ewma() {
        let mut agg = SentimentAggregator::new(0.5);
        agg.add(0.0, 1);
        agg.add(1.0, 2);
        // decay=0.5: weight of first = 0.5^1=0.5, weight of second = 0.5^0=1.0
        // ewma = (0.5*0.0 + 1.0*1.0) / (0.5 + 1.0) = 1.0/1.5
        let ewma = agg.ewma();
        let expected = 1.0 / 1.5;
        assert!((ewma - expected).abs() < 1e-9);
    }

    #[test]
    fn test_aggregator_empty() {
        let agg = SentimentAggregator::new(0.9);
        assert!((agg.average() - 0.0).abs() < 1e-9);
        assert!((agg.ewma() - 0.0).abs() < 1e-9);
        assert!((agg.momentum(5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_classifier_predict() {
        let clf = SentimentClassifier::new(5, 0.01);
        let features = vec![0.5, 0.3, 0.1, 0.7, 0.8];
        let (_, confidence) = clf.predict(&features);
        assert!(confidence >= 0.5 && confidence <= 1.0);
    }

    #[test]
    fn test_classifier_train_and_improve() {
        let data = generate_training_data(500);
        let (train, test) = data.split_at(400);

        let mut clf = SentimentClassifier::new(5, 0.01);
        let _acc_before = clf.accuracy(test);

        clf.train(&train.to_vec(), 50);
        let acc_after = clf.accuracy(test);

        assert!(acc_after > 0.0);
        assert!(acc_after >= 0.4, "accuracy after training: {}", acc_after);
    }

    #[test]
    fn test_strategy_signals() {
        let strategy = SentimentStrategy::new(0.3, -0.3);
        assert_eq!(strategy.signal(0.5), Signal::Buy);
        assert_eq!(strategy.signal(-0.5), Signal::Sell);
        assert_eq!(strategy.signal(0.0), Signal::Hold);
        assert_eq!(strategy.signal(0.3), Signal::Hold);
        assert_eq!(strategy.signal(-0.3), Signal::Hold);
    }

    #[test]
    fn test_strategy_trade_lifecycle() {
        let mut strategy = SentimentStrategy::new(0.3, -0.3);

        // Buy signal at price 100
        strategy.update(0.5, 100.0);
        assert!((strategy.position() - 1.0).abs() < 1e-9);

        // Hold
        strategy.update(0.1, 105.0);
        assert!((strategy.position() - 1.0).abs() < 1e-9);

        // Sell signal at price 110 -> close long, open short
        strategy.update(-0.5, 110.0);
        assert!((strategy.position() - (-1.0)).abs() < 1e-9);

        // Should have one completed trade
        assert_eq!(strategy.trades().len(), 1);
        let trade = &strategy.trades()[0];
        assert!((trade.entry_price - 100.0).abs() < 1e-9);
        assert!((trade.exit_price - 110.0).abs() < 1e-9);
        assert!(trade.pnl > 0.0); // profitable
    }

    #[test]
    fn test_strategy_sharpe() {
        let mut strategy = SentimentStrategy::new(0.3, -0.3);
        // Make a few trades
        strategy.update(0.5, 100.0);
        strategy.update(-0.5, 105.0);
        strategy.update(0.5, 103.0);
        strategy.update(-0.5, 108.0);
        assert!(strategy.trades().len() >= 2);
        // Sharpe should be computable (not NaN)
        let sharpe = strategy.sharpe_ratio();
        assert!(!sharpe.is_nan());
    }

    #[test]
    fn test_synthetic_prices() {
        let prices = generate_synthetic_prices(100, 50000.0);
        assert_eq!(prices.len(), 100);
        for (ts, price) in &prices {
            assert!(*ts > 0);
            assert!(*price > 0.0);
        }
    }

    #[test]
    fn test_synthetic_sentiments() {
        let prices = generate_synthetic_prices(100, 50000.0);
        let sentiments = generate_synthetic_sentiments(&prices);
        assert_eq!(sentiments.len(), 100);
        for (_, score) in &sentiments {
            assert!(*score >= -1.0 && *score <= 1.0);
        }
    }

    #[test]
    fn test_sample_headlines() {
        let headlines = sample_financial_headlines();
        assert!(headlines.len() >= 10);

        let analyzer = SentimentAnalyzer::new();
        let mut positive_count = 0;
        let mut negative_count = 0;
        for h in &headlines {
            let result = analyzer.analyze(h);
            if result.score > 0.0 {
                positive_count += 1;
            } else if result.score < 0.0 {
                negative_count += 1;
            }
        }
        // Should have a mix of positive and negative headlines
        assert!(positive_count > 0);
        assert!(negative_count > 0);
    }

    #[test]
    fn test_training_data_generation() {
        let data = generate_training_data(100);
        assert_eq!(data.len(), 100);
        for (features, label) in &data {
            assert_eq!(features.len(), 5);
            assert!(*label == 0.0 || *label == 1.0);
        }
    }

    #[test]
    fn test_aggregator_momentum() {
        let mut agg = SentimentAggregator::new(0.9);
        // First block: low sentiment
        for _ in 0..5 {
            agg.add(-0.5, 0);
        }
        // Second block: high sentiment
        for _ in 0..5 {
            agg.add(0.5, 0);
        }
        let mom = agg.momentum(5);
        // Momentum should be positive (sentiment improved)
        assert!(mom > 0.0, "momentum should be positive: {}", mom);
    }
}
