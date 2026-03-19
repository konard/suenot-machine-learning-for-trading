use ndarray::Array1;
use rand::Rng;
use serde::Deserialize;
use std::collections::HashMap;

// ─── Sentiment Scorer (TF-IDF Logistic Regression) ──────────────────

/// A simple TF-IDF-based sentiment scorer for financial texts.
///
/// Uses a bag-of-words model with learned weights to classify text
/// as bearish (-1), neutral (0), or bullish (+1).
#[derive(Debug)]
pub struct SentimentScorer {
    vocabulary: HashMap<String, usize>,
    weights: Array1<f64>,
    bias: f64,
    learning_rate: f64,
}

impl SentimentScorer {
    /// Create a new sentiment scorer with a predefined financial vocabulary.
    pub fn new(learning_rate: f64) -> Self {
        let financial_words = vec![
            // Positive words
            "bullish", "surge", "rally", "growth", "profit", "gain", "upgrade",
            "beat", "strong", "positive", "optimistic", "record", "breakout",
            "momentum", "outperform", "buy", "accumulate", "upside", "recovery",
            "expansion", "revenue", "earnings", "dividend", "innovation",
            // Negative words
            "bearish", "crash", "decline", "loss", "drop", "downgrade",
            "miss", "weak", "negative", "pessimistic", "correction", "selloff",
            "underperform", "sell", "risk", "downside", "recession",
            "contraction", "debt", "default", "warning", "concern", "fear",
            "volatile",
            // Neutral / contextual words
            "market", "stock", "price", "trade", "volume", "quarter",
            "report", "guidance", "forecast", "analyst", "company", "sector",
            "bitcoin", "ethereum", "crypto", "exchange", "regulation",
            "institutional", "hedge", "portfolio", "asset", "inflation",
            "rate", "fed", "treasury", "bond", "yield", "spread",
        ];

        let mut vocabulary = HashMap::new();
        for (i, word) in financial_words.iter().enumerate() {
            vocabulary.insert(word.to_string(), i);
        }

        let vocab_size = vocabulary.len();
        let mut rng = rand::thread_rng();
        let weights = Array1::from_vec(
            (0..vocab_size)
                .map(|_| rng.gen_range(-0.1..0.1))
                .collect(),
        );

        Self {
            vocabulary,
            weights,
            bias: 0.0,
            learning_rate,
        }
    }

    /// Tokenize text into lowercase words.
    fn tokenize(text: &str) -> Vec<String> {
        text.to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| !w.is_empty())
            .map(|w| w.to_string())
            .collect()
    }

    /// Convert text to a feature vector using term frequency.
    fn text_to_features(&self, text: &str) -> Array1<f64> {
        let tokens = Self::tokenize(text);
        let mut features = Array1::zeros(self.vocabulary.len());
        let token_count = tokens.len().max(1) as f64;

        for token in &tokens {
            if let Some(&idx) = self.vocabulary.get(token) {
                features[idx] += 1.0 / token_count; // normalized TF
            }
        }
        features
    }

    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    /// Score a text's sentiment. Returns a value in [-1, 1].
    /// Positive = bullish, negative = bearish, near zero = neutral.
    pub fn score(&self, text: &str) -> f64 {
        let features = self.text_to_features(text);
        let z = self.weights.dot(&features) + self.bias;
        // Map sigmoid output [0,1] to [-1,1]
        2.0 * Self::sigmoid(z) - 1.0
    }

    /// Classify sentiment into a label.
    pub fn classify(&self, text: &str) -> (&str, f64) {
        let score = self.score(text);
        let label = if score > 0.2 {
            "bullish"
        } else if score < -0.2 {
            "bearish"
        } else {
            "neutral"
        };
        (label, score)
    }

    /// Score a batch of texts.
    pub fn score_batch(&self, texts: &[&str]) -> Vec<f64> {
        texts.iter().map(|t| self.score(t)).collect()
    }

    /// Train on labeled data: (text, label) where label is in [-1, 1].
    pub fn train(&mut self, data: &[(&str, f64)], epochs: usize) {
        for _ in 0..epochs {
            for (text, label) in data {
                let features = self.text_to_features(text);
                let z = self.weights.dot(&features) + self.bias;
                let pred = 2.0 * Self::sigmoid(z) - 1.0;
                let error = pred - label;

                // Gradient descent
                for j in 0..self.weights.len() {
                    self.weights[j] -= self.learning_rate * error * features[j];
                }
                self.bias -= self.learning_rate * error;
            }
        }
    }
}

// ─── Signal Generator ───────────────────────────────────────────────

/// Converts sentiment scores into trading signals using exponential decay.
#[derive(Debug)]
pub struct SignalGenerator {
    decay: f64,
    aggressiveness: f64,
    history: Vec<f64>,
}

impl SignalGenerator {
    pub fn new(decay: f64, aggressiveness: f64) -> Self {
        Self {
            decay,
            aggressiveness,
            history: Vec::new(),
        }
    }

    /// Add a new sentiment score and return the aggregated signal.
    pub fn update(&mut self, sentiment_score: f64) -> f64 {
        self.history.push(sentiment_score);
        let mut signal = 0.0;
        for (i, &score) in self.history.iter().rev().enumerate() {
            signal += self.decay.powi(i as i32) * score;
        }
        signal
    }

    /// Convert aggregated signal to a position in [-1, 1].
    pub fn position(&self, signal: f64) -> f64 {
        2.0 / (1.0 + (-self.aggressiveness * signal).exp()) - 1.0
    }

    /// Generate a trading action from a sentiment score.
    pub fn generate(&mut self, sentiment_score: f64) -> TradeSignal {
        let signal = self.update(sentiment_score);
        let pos = self.position(signal);
        let action = if pos > 0.3 {
            "BUY"
        } else if pos < -0.3 {
            "SELL"
        } else {
            "HOLD"
        };
        TradeSignal {
            signal,
            position: pos,
            action: action.to_string(),
        }
    }

    /// Reset the signal history.
    pub fn reset(&mut self) {
        self.history.clear();
    }

    /// Current number of observations.
    pub fn len(&self) -> usize {
        self.history.len()
    }

    /// Whether the generator has no observations.
    pub fn is_empty(&self) -> bool {
        self.history.is_empty()
    }
}

/// A trading signal with aggregated value, position size, and action.
#[derive(Debug, Clone)]
pub struct TradeSignal {
    pub signal: f64,
    pub position: f64,
    pub action: String,
}

// ─── Backtester ─────────────────────────────────────────────────────

/// Simulates a sentiment-driven trading strategy on historical prices.
#[derive(Debug)]
pub struct Backtester {
    initial_capital: f64,
    capital: f64,
    position: f64,
    trades: Vec<Trade>,
    equity_curve: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct Trade {
    pub price: f64,
    pub units: f64,
    pub direction: String,
}

#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub total_return: f64,
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub max_drawdown: f64,
    pub num_trades: usize,
}

impl Backtester {
    pub fn new(initial_capital: f64) -> Self {
        Self {
            initial_capital,
            capital: initial_capital,
            position: 0.0,
            trades: Vec::new(),
            equity_curve: vec![initial_capital],
        }
    }

    /// Execute one step: rebalance position based on signal.
    pub fn step(&mut self, price: f64, target_position: f64) -> f64 {
        let target_units = target_position * self.capital / price;
        let delta = target_units - self.position;

        if delta.abs() > 0.001 {
            self.trades.push(Trade {
                price,
                units: delta,
                direction: if delta > 0.0 {
                    "buy".to_string()
                } else {
                    "sell".to_string()
                },
            });
        }
        self.position = target_units;
        let equity = self.capital + self.position * price;
        self.equity_curve.push(equity);
        equity
    }

    /// Compute performance metrics.
    pub fn metrics(&self) -> PerformanceMetrics {
        let returns: Vec<f64> = self
            .equity_curve
            .windows(2)
            .filter(|w| w[0] != 0.0)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect();

        if returns.is_empty() {
            return PerformanceMetrics {
                total_return: 0.0,
                sharpe_ratio: 0.0,
                sortino_ratio: 0.0,
                max_drawdown: 0.0,
                num_trades: self.trades.len(),
            };
        }

        let avg: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
        let var: f64 = returns.iter().map(|r| (r - avg).powi(2)).sum::<f64>() / returns.len() as f64;
        let std = var.sqrt();
        let sharpe = if std > 0.0 {
            avg / std * (252.0_f64).sqrt()
        } else {
            0.0
        };

        let downside: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).cloned().collect();
        let downside_std = if !downside.is_empty() {
            (downside.iter().map(|r| r.powi(2)).sum::<f64>() / downside.len() as f64).sqrt()
        } else {
            0.0
        };
        let sortino = if downside_std > 0.0 {
            avg / downside_std * (252.0_f64).sqrt()
        } else {
            0.0
        };

        let mut max_equity = self.equity_curve[0];
        let mut max_dd = 0.0_f64;
        for &eq in &self.equity_curve {
            max_equity = max_equity.max(eq);
            let dd = (max_equity - eq) / max_equity;
            max_dd = max_dd.max(dd);
        }

        let total_return =
            (self.equity_curve.last().unwrap_or(&self.initial_capital) - self.initial_capital)
                / self.initial_capital;

        PerformanceMetrics {
            total_return,
            sharpe_ratio: sharpe,
            sortino_ratio: sortino,
            max_drawdown: max_dd,
            num_trades: self.trades.len(),
        }
    }

    pub fn equity_curve(&self) -> &[f64] {
        &self.equity_curve
    }

    pub fn trades(&self) -> &[Trade] {
        &self.trades
    }
}

// ─── Bybit Client ───────────────────────────────────────────────────

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

/// A parsed kline (candlestick) bar.
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
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Synthetic Data Generation ──────────────────────────────────────

/// Generate synthetic financial news headlines for testing.
pub fn generate_synthetic_news() -> Vec<(&'static str, f64)> {
    vec![
        ("Bitcoin surges past resistance on strong institutional buying", 0.8),
        ("Ethereum upgrade reduces gas fees significantly", 0.6),
        ("Market crashes as regulatory crackdown intensifies", -0.9),
        ("Company reports record quarterly earnings beating estimates", 0.7),
        ("Analysts downgrade stock amid weak guidance and declining revenue", -0.7),
        ("Federal Reserve signals hawkish rate policy causing selloff", -0.6),
        ("Strong economic growth data fuels market rally", 0.8),
        ("Crypto exchange faces security breach concerns", -0.5),
        ("Institutional investors accumulate Bitcoin at record pace", 0.7),
        ("Market correction deepens as fear spreads across sectors", -0.8),
        ("Tech sector shows resilience with positive earnings beat", 0.5),
        ("Oil prices decline on recession fears and weak demand forecast", -0.6),
        ("Central bank maintains neutral stance on monetary policy", 0.0),
        ("Trading volume surges as market momentum builds", 0.4),
        ("Company issues profit warning citing supply chain disruption", -0.7),
        ("Blockchain innovation drives new wave of crypto adoption", 0.5),
        ("Bond yields rise as inflation concerns mount", -0.3),
        ("Portfolio managers shift to risk assets on optimistic outlook", 0.6),
        ("Stock buyback program boosts shareholder value", 0.4),
        ("Hedge fund liquidation causes volatile market conditions", -0.5),
    ]
}

/// Generate synthetic price data for backtesting.
pub fn generate_synthetic_prices(n: usize, start_price: f64) -> Vec<f64> {
    let mut rng = rand::thread_rng();
    let mut prices = Vec::with_capacity(n);
    let mut price = start_price;

    for _ in 0..n {
        price *= 1.0 + rng.gen_range(-0.03..0.03);
        price = price.max(start_price * 0.5); // floor at 50% of start
        prices.push(price);
    }
    prices
}

// ─── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sentiment_scorer_creation() {
        let scorer = SentimentScorer::new(0.01);
        let score = scorer.score("Bitcoin surges on strong buying momentum");
        assert!(score >= -1.0 && score <= 1.0);
    }

    #[test]
    fn test_sentiment_scorer_training() {
        let mut scorer = SentimentScorer::new(0.1);
        let data = generate_synthetic_news();
        let train_data: Vec<(&str, f64)> = data.iter().map(|&(t, l)| (t, l)).collect();

        scorer.train(&train_data, 100);

        // After training, bullish text should score higher than bearish
        let bullish_score = scorer.score("strong rally and growth in market");
        let bearish_score = scorer.score("crash and decline with heavy loss");
        assert!(
            bullish_score > bearish_score,
            "bullish={:.4} should be > bearish={:.4}",
            bullish_score,
            bearish_score
        );
    }

    #[test]
    fn test_sentiment_classify() {
        let scorer = SentimentScorer::new(0.01);
        let (label, score) = scorer.classify("market is stable");
        assert!(["bullish", "bearish", "neutral"].contains(&label));
        assert!(score >= -1.0 && score <= 1.0);
    }

    #[test]
    fn test_sentiment_batch() {
        let scorer = SentimentScorer::new(0.01);
        let texts = vec!["bullish rally", "market crash", "neutral report"];
        let scores = scorer.score_batch(&texts);
        assert_eq!(scores.len(), 3);
        for s in &scores {
            assert!(*s >= -1.0 && *s <= 1.0);
        }
    }

    #[test]
    fn test_signal_generator_basic() {
        let mut gen = SignalGenerator::new(0.9, 2.0);
        assert!(gen.is_empty());

        let sig = gen.generate(0.5);
        assert!(!gen.is_empty());
        assert_eq!(gen.len(), 1);
        assert!(sig.position >= -1.0 && sig.position <= 1.0);
    }

    #[test]
    fn test_signal_generator_decay() {
        let mut gen = SignalGenerator::new(0.9, 2.0);

        // Add positive sentiment repeatedly
        for _ in 0..5 {
            gen.update(0.8);
        }
        let pos_signal = gen.history.iter().rev().enumerate()
            .map(|(i, &s)| 0.9_f64.powi(i as i32) * s)
            .sum::<f64>();
        let pos = gen.position(pos_signal);
        assert!(pos > 0.0, "multiple positive sentiments should yield positive position");
    }

    #[test]
    fn test_signal_generator_actions() {
        let mut gen = SignalGenerator::new(0.5, 5.0);

        let sig = gen.generate(0.9);
        assert_eq!(sig.action, "BUY");

        gen.reset();
        let sig = gen.generate(-0.9);
        assert_eq!(sig.action, "SELL");
    }

    #[test]
    fn test_backtester_basic() {
        let mut bt = Backtester::new(10000.0);
        let prices = generate_synthetic_prices(20, 100.0);

        for &price in &prices {
            bt.step(price, 0.5); // always 50% long
        }

        let metrics = bt.metrics();
        assert!(metrics.max_drawdown >= 0.0 && metrics.max_drawdown <= 1.0);
        assert!(metrics.num_trades > 0);
    }

    #[test]
    fn test_backtester_no_trade() {
        let bt = Backtester::new(10000.0);
        let metrics = bt.metrics();
        assert_eq!(metrics.num_trades, 0);
        assert_eq!(metrics.total_return, 0.0);
    }

    #[test]
    fn test_backtester_equity_curve() {
        let mut bt = Backtester::new(10000.0);
        bt.step(100.0, 1.0);
        bt.step(110.0, 1.0);
        assert_eq!(bt.equity_curve().len(), 3); // initial + 2 steps
    }

    #[test]
    fn test_synthetic_news_generation() {
        let news = generate_synthetic_news();
        assert!(!news.is_empty());
        for (text, label) in &news {
            assert!(!text.is_empty());
            assert!(*label >= -1.0 && *label <= 1.0);
        }
    }

    #[test]
    fn test_synthetic_prices() {
        let prices = generate_synthetic_prices(100, 50000.0);
        assert_eq!(prices.len(), 100);
        for p in &prices {
            assert!(*p > 0.0);
        }
    }

    #[test]
    fn test_end_to_end_pipeline() {
        // Train scorer
        let mut scorer = SentimentScorer::new(0.1);
        let news = generate_synthetic_news();
        let train_data: Vec<(&str, f64)> = news.iter().map(|&(t, l)| (t, l)).collect();
        scorer.train(&train_data, 50);

        // Generate signals
        let mut signal_gen = SignalGenerator::new(0.9, 2.0);
        let test_texts = vec![
            "Bitcoin rally continues with strong momentum",
            "Market decline accelerates on fear",
            "Stable trading with neutral outlook",
        ];

        let mut signals = Vec::new();
        for text in &test_texts {
            let sentiment = scorer.score(text);
            let sig = signal_gen.generate(sentiment);
            signals.push(sig);
        }

        assert_eq!(signals.len(), 3);

        // Backtest
        let mut bt = Backtester::new(10000.0);
        let prices = generate_synthetic_prices(3, 50000.0);
        for (sig, &price) in signals.iter().zip(prices.iter()) {
            bt.step(price, sig.position);
        }

        let metrics = bt.metrics();
        assert!(metrics.max_drawdown <= 1.0);
    }
}
