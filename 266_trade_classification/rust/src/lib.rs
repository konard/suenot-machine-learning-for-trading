use anyhow::Result;
use ndarray::Array1;
use rand::Rng;
use serde::{Deserialize, Serialize};

// ============================================================
// Market Data Structures
// ============================================================

/// A single OHLCV candle from exchange data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candle {
    pub timestamp: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// A single trade record from the exchange
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    pub timestamp: u64,
    pub price: f64,
    pub quantity: f64,
    pub side: Option<TradeSide>,
}

/// Trade side classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TradeSide {
    Buy,
    Sell,
}

impl std::fmt::Display for TradeSide {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TradeSide::Buy => write!(f, "Buy"),
            TradeSide::Sell => write!(f, "Sell"),
        }
    }
}

/// Quote data representing best bid/ask at a point in time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Quote {
    pub timestamp: u64,
    pub bid_price: f64,
    pub ask_price: f64,
    pub bid_size: f64,
    pub ask_size: f64,
}

impl Quote {
    /// Compute the midpoint price
    pub fn midpoint(&self) -> f64 {
        (self.bid_price + self.ask_price) / 2.0
    }

    /// Compute the spread
    pub fn spread(&self) -> f64 {
        self.ask_price - self.bid_price
    }
}

/// Result of a trade classification algorithm
#[derive(Debug, Clone)]
pub struct ClassificationResult {
    pub trade_index: usize,
    pub predicted_side: TradeSide,
    pub confidence: f64,
    pub method: ClassificationMethod,
}

/// Available classification methods
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClassificationMethod {
    TickTest,
    QuoteRule,
    LeeReady,
    BulkVolumeClassification,
    MLEnsemble,
}

impl std::fmt::Display for ClassificationMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ClassificationMethod::TickTest => write!(f, "Tick Test"),
            ClassificationMethod::QuoteRule => write!(f, "Quote Rule"),
            ClassificationMethod::LeeReady => write!(f, "Lee-Ready"),
            ClassificationMethod::BulkVolumeClassification => write!(f, "BVC"),
            ClassificationMethod::MLEnsemble => write!(f, "ML Ensemble"),
        }
    }
}

// ============================================================
// Tick Test Classifier
// ============================================================

/// Implements the tick test (or tick rule) for trade classification.
///
/// A trade is classified as buyer-initiated if its price is higher than
/// the previous trade price (uptick), and seller-initiated if it is lower
/// (downtick). If the price is unchanged, the previous classification is
/// carried forward (zero-tick rule).
pub struct TickTestClassifier;

impl TickTestClassifier {
    /// Classify a sequence of trades using the tick test.
    ///
    /// Returns `None` for the first trade (no prior price to compare).
    pub fn classify(trades: &[Trade]) -> Vec<Option<TradeSide>> {
        if trades.is_empty() {
            return vec![];
        }

        let mut results = Vec::with_capacity(trades.len());
        results.push(None); // first trade is indeterminate

        let mut last_side = TradeSide::Buy; // default

        for i in 1..trades.len() {
            let diff = trades[i].price - trades[i - 1].price;
            if diff > 0.0 {
                last_side = TradeSide::Buy;
            } else if diff < 0.0 {
                last_side = TradeSide::Sell;
            }
            // if diff == 0.0, carry forward last_side
            results.push(Some(last_side));
        }
        results
    }
}

// ============================================================
// Quote Rule Classifier
// ============================================================

/// Implements the quote rule for trade classification.
///
/// A trade is classified as buyer-initiated if the trade price is above
/// the midpoint of the prevailing bid-ask spread, and seller-initiated
/// if below the midpoint.
pub struct QuoteRuleClassifier;

impl QuoteRuleClassifier {
    /// Classify trades given matched quotes.
    ///
    /// `trades` and `quotes` must be the same length (each trade matched
    /// with the prevailing quote at the time of the trade).
    pub fn classify(trades: &[Trade], quotes: &[Quote]) -> Vec<Option<TradeSide>> {
        assert_eq!(trades.len(), quotes.len(), "trades and quotes must match in length");

        trades
            .iter()
            .zip(quotes.iter())
            .map(|(trade, quote)| {
                let mid = quote.midpoint();
                if trade.price > mid {
                    Some(TradeSide::Buy)
                } else if trade.price < mid {
                    Some(TradeSide::Sell)
                } else {
                    None // at midpoint, indeterminate
                }
            })
            .collect()
    }
}

// ============================================================
// Lee-Ready Classifier
// ============================================================

/// Implements the Lee-Ready algorithm for trade classification.
///
/// Lee-Ready combines the quote rule with the tick test as a fallback:
/// 1. If the trade price is above the midpoint -> Buy
/// 2. If the trade price is below the midpoint -> Sell
/// 3. If the trade price equals the midpoint -> use the tick test
pub struct LeeReadyClassifier;

impl LeeReadyClassifier {
    /// Classify trades using the Lee-Ready method.
    pub fn classify(trades: &[Trade], quotes: &[Quote]) -> Vec<Option<TradeSide>> {
        assert_eq!(trades.len(), quotes.len(), "trades and quotes must match in length");

        let tick_results = TickTestClassifier::classify(trades);
        let quote_results = QuoteRuleClassifier::classify(trades, quotes);

        quote_results
            .into_iter()
            .zip(tick_results.into_iter())
            .map(|(quote_class, tick_class)| {
                // Use quote rule first, fall back to tick test
                quote_class.or(tick_class)
            })
            .collect()
    }
}

// ============================================================
// Bulk Volume Classification (BVC)
// ============================================================

/// Implements Bulk Volume Classification (Easley, Lopez de Prado, O'Hara 2012).
///
/// BVC uses the cumulative distribution function of price changes to
/// estimate the probability that a volume bar was buyer- or seller-initiated.
pub struct BulkVolumeClassifier {
    /// Standard deviation parameter for the normal CDF approximation
    pub sigma: f64,
}

impl BulkVolumeClassifier {
    pub fn new(sigma: f64) -> Self {
        Self { sigma }
    }

    /// Approximate the standard normal CDF using a logistic approximation.
    fn normal_cdf_approx(x: f64) -> f64 {
        1.0 / (1.0 + (-1.7 * x).exp())
    }

    /// Classify volume bars and return the buy-volume fraction for each bar.
    ///
    /// `prices` is a vector of closing prices for consecutive volume bars.
    /// Returns a vector of buy-volume fractions in [0, 1].
    pub fn classify_bars(&self, prices: &[f64]) -> Vec<f64> {
        if prices.len() < 2 {
            return vec![];
        }

        let mut buy_fractions = Vec::with_capacity(prices.len() - 1);

        for i in 1..prices.len() {
            let delta_p = prices[i] - prices[i - 1];
            let z = delta_p / self.sigma;
            let buy_frac = Self::normal_cdf_approx(z);
            buy_fractions.push(buy_frac);
        }

        buy_fractions
    }

    /// Convert buy-volume fractions to trade sides using a 0.5 threshold.
    pub fn classify_sides(&self, prices: &[f64]) -> Vec<TradeSide> {
        self.classify_bars(prices)
            .into_iter()
            .map(|frac| {
                if frac >= 0.5 {
                    TradeSide::Buy
                } else {
                    TradeSide::Sell
                }
            })
            .collect()
    }
}

// ============================================================
// ML Ensemble Classifier
// ============================================================

/// Feature vector extracted for ML-based trade classification
#[derive(Debug, Clone)]
pub struct TradeFeatures {
    /// Price relative to midpoint (trade_price - midpoint) / spread
    pub relative_price: f64,
    /// Trade size relative to average
    pub relative_size: f64,
    /// Tick direction: +1, 0, or -1
    pub tick_direction: f64,
    /// Bid-ask spread at trade time
    pub spread: f64,
    /// Quote imbalance: (bid_size - ask_size) / (bid_size + ask_size)
    pub quote_imbalance: f64,
    /// Time since last trade (in milliseconds)
    pub time_delta_ms: f64,
}

impl TradeFeatures {
    /// Convert features to an ndarray Array1
    pub fn to_array(&self) -> Array1<f64> {
        Array1::from(vec![
            self.relative_price,
            self.relative_size,
            self.tick_direction,
            self.spread,
            self.quote_imbalance,
            self.time_delta_ms,
        ])
    }
}

/// Extract features for each trade given associated quotes.
pub fn extract_features(trades: &[Trade], quotes: &[Quote]) -> Vec<TradeFeatures> {
    assert_eq!(trades.len(), quotes.len());

    let avg_size: f64 = if trades.is_empty() {
        1.0
    } else {
        trades.iter().map(|t| t.quantity).sum::<f64>() / trades.len() as f64
    };

    let mut features = Vec::with_capacity(trades.len());

    for i in 0..trades.len() {
        let mid = quotes[i].midpoint();
        let spread = quotes[i].spread();
        let relative_price = if spread > 0.0 {
            (trades[i].price - mid) / spread
        } else {
            0.0
        };

        let relative_size = trades[i].quantity / avg_size;

        let tick_direction = if i == 0 {
            0.0
        } else {
            let diff = trades[i].price - trades[i - 1].price;
            if diff > 0.0 {
                1.0
            } else if diff < 0.0 {
                -1.0
            } else {
                0.0
            }
        };

        let total_size = quotes[i].bid_size + quotes[i].ask_size;
        let quote_imbalance = if total_size > 0.0 {
            (quotes[i].bid_size - quotes[i].ask_size) / total_size
        } else {
            0.0
        };

        let time_delta_ms = if i == 0 {
            0.0
        } else {
            (trades[i].timestamp as f64) - (trades[i - 1].timestamp as f64)
        };

        features.push(TradeFeatures {
            relative_price,
            relative_size,
            tick_direction,
            spread,
            quote_imbalance,
            time_delta_ms,
        });
    }

    features
}

/// Simple logistic regression model for trade classification.
///
/// Uses pre-trained weights (in practice these would come from training).
pub struct LogisticClassifier {
    pub weights: Array1<f64>,
    pub bias: f64,
}

impl LogisticClassifier {
    /// Create a new classifier with given weights.
    pub fn new(weights: Array1<f64>, bias: f64) -> Self {
        Self { weights, bias }
    }

    /// Create a classifier with default weights tuned for trade classification.
    pub fn default_trade_classifier() -> Self {
        // Weights: [relative_price, relative_size, tick_direction, spread, quote_imbalance, time_delta_ms]
        let weights = Array1::from(vec![2.5, 0.1, 1.0, -0.5, 0.8, 0.0001]);
        Self {
            weights,
            bias: 0.0,
        }
    }

    /// Sigmoid function
    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    /// Predict the probability of a trade being buyer-initiated.
    pub fn predict_proba(&self, features: &TradeFeatures) -> f64 {
        let x = features.to_array();
        let logit = x.dot(&self.weights) + self.bias;
        Self::sigmoid(logit)
    }

    /// Classify a trade based on features.
    pub fn classify(&self, features: &TradeFeatures) -> ClassificationResult {
        let proba = self.predict_proba(features);
        let (side, confidence) = if proba >= 0.5 {
            (TradeSide::Buy, proba)
        } else {
            (TradeSide::Sell, 1.0 - proba)
        };

        ClassificationResult {
            trade_index: 0,
            predicted_side: side,
            confidence,
            method: ClassificationMethod::MLEnsemble,
        }
    }
}

/// ML Ensemble that combines multiple classification methods.
pub struct EnsembleClassifier {
    pub logistic: LogisticClassifier,
    pub bvc: BulkVolumeClassifier,
}

impl EnsembleClassifier {
    pub fn new() -> Self {
        Self {
            logistic: LogisticClassifier::default_trade_classifier(),
            bvc: BulkVolumeClassifier::new(0.01),
        }
    }

    /// Classify trades using the full ensemble pipeline.
    pub fn classify(
        &self,
        trades: &[Trade],
        quotes: &[Quote],
    ) -> Vec<ClassificationResult> {
        let features = extract_features(trades, quotes);
        let tick_results = TickTestClassifier::classify(trades);
        let lee_ready_results = LeeReadyClassifier::classify(trades, quotes);
        let prices: Vec<f64> = trades.iter().map(|t| t.price).collect();
        let bvc_fractions = self.bvc.classify_bars(&prices);

        let mut results = Vec::with_capacity(trades.len());

        for i in 0..trades.len() {
            // ML prediction
            let ml_proba = self.logistic.predict_proba(&features[i]);

            // Lee-Ready vote
            let lr_vote = match lee_ready_results[i] {
                Some(TradeSide::Buy) => 1.0,
                Some(TradeSide::Sell) => 0.0,
                None => 0.5,
            };

            // Tick test vote
            let tick_vote = match tick_results[i] {
                Some(TradeSide::Buy) => 1.0,
                Some(TradeSide::Sell) => 0.0,
                None => 0.5,
            };

            // BVC vote (for bars after the first)
            let bvc_vote = if i > 0 && i - 1 < bvc_fractions.len() {
                bvc_fractions[i - 1]
            } else {
                0.5
            };

            // Weighted ensemble: ML 40%, Lee-Ready 30%, Tick 15%, BVC 15%
            let ensemble_proba =
                0.40 * ml_proba + 0.30 * lr_vote + 0.15 * tick_vote + 0.15 * bvc_vote;

            let (side, confidence) = if ensemble_proba >= 0.5 {
                (TradeSide::Buy, ensemble_proba)
            } else {
                (TradeSide::Sell, 1.0 - ensemble_proba)
            };

            results.push(ClassificationResult {
                trade_index: i,
                predicted_side: side,
                confidence,
                method: ClassificationMethod::MLEnsemble,
            });
        }

        results
    }
}

impl Default for EnsembleClassifier {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================
// Accuracy Evaluation
// ============================================================

/// Evaluate classification accuracy against known labels.
pub fn evaluate_accuracy(predictions: &[TradeSide], labels: &[TradeSide]) -> f64 {
    assert_eq!(predictions.len(), labels.len());
    if predictions.is_empty() {
        return 0.0;
    }
    let correct = predictions
        .iter()
        .zip(labels.iter())
        .filter(|(p, l)| p == l)
        .count();
    correct as f64 / predictions.len() as f64
}

// ============================================================
// Bybit API Integration
// ============================================================

/// Bybit API response structures
#[derive(Debug, Deserialize)]
pub struct BybitResponse<T> {
    #[serde(rename = "retCode")]
    pub ret_code: i32,
    #[serde(rename = "retMsg")]
    pub ret_msg: String,
    pub result: T,
}

#[derive(Debug, Deserialize)]
pub struct BybitKlineResult {
    pub list: Vec<Vec<String>>,
}

#[derive(Debug, Deserialize)]
pub struct BybitTradeResult {
    pub list: Vec<BybitTradeRecord>,
}

#[derive(Debug, Deserialize)]
pub struct BybitTradeRecord {
    #[serde(rename = "execId")]
    pub exec_id: String,
    pub symbol: String,
    pub price: String,
    pub size: String,
    pub side: String,
    pub time: String,
}

#[derive(Debug, Deserialize)]
pub struct BybitOrderbookResult {
    pub b: Vec<Vec<String>>, // bids
    pub a: Vec<Vec<String>>, // asks
}

/// Fetch recent trades from Bybit
pub async fn fetch_bybit_trades(symbol: &str, limit: usize) -> Result<Vec<Trade>> {
    let url = format!(
        "https://api.bybit.com/v5/market/recent-trading-records?category=spot&symbol={}&limit={}",
        symbol, limit
    );

    let client = reqwest::Client::new();
    let resp: BybitResponse<BybitTradeResult> = client.get(&url).send().await?.json().await?;

    let trades: Vec<Trade> = resp
        .result
        .list
        .iter()
        .map(|record| {
            let side = match record.side.as_str() {
                "Buy" => Some(TradeSide::Buy),
                "Sell" => Some(TradeSide::Sell),
                _ => None,
            };
            Trade {
                timestamp: record.time.parse().unwrap_or(0),
                price: record.price.parse().unwrap_or(0.0),
                quantity: record.size.parse().unwrap_or(0.0),
                side,
            }
        })
        .collect();

    Ok(trades)
}

/// Fetch current orderbook from Bybit to create quote data
pub async fn fetch_bybit_orderbook(symbol: &str) -> Result<Quote> {
    let url = format!(
        "https://api.bybit.com/v5/market/orderbook?category=spot&symbol={}&limit=1",
        symbol
    );

    let client = reqwest::Client::new();
    let resp: BybitResponse<BybitOrderbookResult> = client.get(&url).send().await?.json().await?;

    let bid_price: f64 = resp
        .result
        .b
        .first()
        .and_then(|b| b.first())
        .and_then(|p| p.parse().ok())
        .unwrap_or(0.0);
    let bid_size: f64 = resp
        .result
        .b
        .first()
        .and_then(|b| b.get(1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0);
    let ask_price: f64 = resp
        .result
        .a
        .first()
        .and_then(|a| a.first())
        .and_then(|p| p.parse().ok())
        .unwrap_or(0.0);
    let ask_size: f64 = resp
        .result
        .a
        .first()
        .and_then(|a| a.get(1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0);

    Ok(Quote {
        timestamp: 0,
        bid_price,
        ask_price,
        bid_size,
        ask_size,
    })
}

/// Fetch klines from Bybit
pub async fn fetch_bybit_klines(
    symbol: &str,
    interval: &str,
    limit: usize,
) -> Result<Vec<Candle>> {
    let url = format!(
        "https://api.bybit.com/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
        symbol, interval, limit
    );

    let client = reqwest::Client::new();
    let resp: BybitResponse<BybitKlineResult> = client.get(&url).send().await?.json().await?;

    let candles: Vec<Candle> = resp
        .result
        .list
        .iter()
        .filter_map(|row| {
            if row.len() >= 6 {
                Some(Candle {
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

    Ok(candles)
}

// ============================================================
// Synthetic Data Generation
// ============================================================

/// Generate synthetic trades for testing
pub fn generate_synthetic_trades(n: usize, base_price: f64) -> (Vec<Trade>, Vec<Quote>, Vec<TradeSide>) {
    let mut rng = rand::thread_rng();
    let mut trades = Vec::with_capacity(n);
    let mut quotes = Vec::with_capacity(n);
    let mut labels = Vec::with_capacity(n);

    let mut price = base_price;
    let spread = base_price * 0.001; // 0.1% spread

    for i in 0..n {
        let is_buy = rng.gen_bool(0.5);
        let side = if is_buy { TradeSide::Buy } else { TradeSide::Sell };

        let bid = price - spread / 2.0;
        let ask = price + spread / 2.0;
        let mid = (bid + ask) / 2.0;

        // Buyer-initiated trades tend to be at or above midpoint
        let trade_price = if is_buy {
            mid + rng.gen_range(0.0..spread * 0.6)
        } else {
            mid - rng.gen_range(0.0..spread * 0.6)
        };

        trades.push(Trade {
            timestamp: 1_000_000 + (i as u64) * 100,
            price: trade_price,
            quantity: rng.gen_range(0.01..10.0),
            side: None, // unknown, to be classified
        });

        quotes.push(Quote {
            timestamp: 1_000_000 + (i as u64) * 100,
            bid_price: bid,
            ask_price: ask,
            bid_size: rng.gen_range(1.0..100.0),
            ask_size: rng.gen_range(1.0..100.0),
        });

        labels.push(side);

        // Random walk for next price
        price += rng.gen_range(-spread..spread);
    }

    (trades, quotes, labels)
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_trades() -> Vec<Trade> {
        vec![
            Trade { timestamp: 1, price: 100.0, quantity: 1.0, side: None },
            Trade { timestamp: 2, price: 100.5, quantity: 2.0, side: None },
            Trade { timestamp: 3, price: 100.5, quantity: 1.5, side: None },
            Trade { timestamp: 4, price: 99.8, quantity: 3.0, side: None },
            Trade { timestamp: 5, price: 100.2, quantity: 1.0, side: None },
        ]
    }

    fn sample_quotes() -> Vec<Quote> {
        vec![
            Quote { timestamp: 1, bid_price: 99.9, ask_price: 100.1, bid_size: 10.0, ask_size: 10.0 },
            Quote { timestamp: 2, bid_price: 100.3, ask_price: 100.7, bid_size: 8.0, ask_size: 12.0 },
            Quote { timestamp: 3, bid_price: 100.3, ask_price: 100.7, bid_size: 15.0, ask_size: 5.0 },
            Quote { timestamp: 4, bid_price: 99.6, ask_price: 100.0, bid_size: 5.0, ask_size: 20.0 },
            Quote { timestamp: 5, bid_price: 100.0, ask_price: 100.4, bid_size: 10.0, ask_size: 10.0 },
        ]
    }

    #[test]
    fn test_tick_test_classifier() {
        let trades = sample_trades();
        let results = TickTestClassifier::classify(&trades);

        assert_eq!(results.len(), 5);
        assert!(results[0].is_none()); // first trade is indeterminate
        assert_eq!(results[1], Some(TradeSide::Buy)); // 100.5 > 100.0 -> uptick
        assert_eq!(results[2], Some(TradeSide::Buy)); // 100.5 == 100.5 -> carry forward Buy
        assert_eq!(results[3], Some(TradeSide::Sell)); // 99.8 < 100.5 -> downtick
        assert_eq!(results[4], Some(TradeSide::Buy)); // 100.2 > 99.8 -> uptick
    }

    #[test]
    fn test_quote_rule_classifier() {
        let trades = sample_trades();
        let quotes = sample_quotes();
        let results = QuoteRuleClassifier::classify(&trades, &quotes);

        assert_eq!(results.len(), 5);
        // Trade 0: price=100.0, mid=100.0 -> at midpoint -> None
        assert!(results[0].is_none());
        // Trade 1: price=100.5, mid=100.5 -> at midpoint -> None
        assert!(results[1].is_none());
        // Trade 2: price=100.5, mid=100.5 -> at midpoint -> None
        assert!(results[2].is_none());
        // Trade 3: price=99.8, mid=99.8 -> at midpoint -> None
        assert!(results[3].is_none());
        // Trade 4: price=100.2, mid=100.2 -> at midpoint -> None
        assert!(results[4].is_none());
    }

    #[test]
    fn test_quote_rule_classifier_clear_cases() {
        let trades = vec![
            Trade { timestamp: 1, price: 100.2, quantity: 1.0, side: None },
            Trade { timestamp: 2, price: 99.8, quantity: 1.0, side: None },
        ];
        let quotes = vec![
            Quote { timestamp: 1, bid_price: 99.9, ask_price: 100.1, bid_size: 10.0, ask_size: 10.0 },
            Quote { timestamp: 2, bid_price: 99.9, ask_price: 100.1, bid_size: 10.0, ask_size: 10.0 },
        ];
        let results = QuoteRuleClassifier::classify(&trades, &quotes);

        assert_eq!(results[0], Some(TradeSide::Buy));  // 100.2 > 100.0
        assert_eq!(results[1], Some(TradeSide::Sell)); // 99.8 < 100.0
    }

    #[test]
    fn test_lee_ready_classifier() {
        // Use trades that are at midpoint so Lee-Ready falls back to tick test
        let trades = sample_trades();
        let quotes = sample_quotes();
        let results = LeeReadyClassifier::classify(&trades, &quotes);

        assert_eq!(results.len(), 5);
        // All at midpoint, so should fall back to tick test
        // Trade 0: tick=None -> None
        assert!(results[0].is_none());
        // Trade 1: tick=Buy
        assert_eq!(results[1], Some(TradeSide::Buy));
    }

    #[test]
    fn test_bvc_classifier() {
        let bvc = BulkVolumeClassifier::new(0.5);
        let prices = vec![100.0, 100.5, 101.0, 100.3, 99.8];

        let fractions = bvc.classify_bars(&prices);
        assert_eq!(fractions.len(), 4);

        // Price up -> fraction > 0.5 (buy)
        assert!(fractions[0] > 0.5);
        assert!(fractions[1] > 0.5);
        // Price down -> fraction < 0.5 (sell)
        assert!(fractions[2] < 0.5);
        assert!(fractions[3] < 0.5);

        let sides = bvc.classify_sides(&prices);
        assert_eq!(sides[0], TradeSide::Buy);
        assert_eq!(sides[1], TradeSide::Buy);
        assert_eq!(sides[2], TradeSide::Sell);
        assert_eq!(sides[3], TradeSide::Sell);
    }

    #[test]
    fn test_ensemble_classifier() {
        let (trades, quotes, _labels) = generate_synthetic_trades(100, 50000.0);
        let ensemble = EnsembleClassifier::new();
        let results = ensemble.classify(&trades, &quotes);

        assert_eq!(results.len(), 100);
        for r in &results {
            assert!(r.confidence >= 0.5 && r.confidence <= 1.0);
        }
    }

    #[test]
    fn test_evaluate_accuracy() {
        let preds = vec![TradeSide::Buy, TradeSide::Sell, TradeSide::Buy, TradeSide::Sell];
        let labels = vec![TradeSide::Buy, TradeSide::Sell, TradeSide::Sell, TradeSide::Sell];
        let acc = evaluate_accuracy(&preds, &labels);
        assert!((acc - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_feature_extraction() {
        let trades = sample_trades();
        let quotes = sample_quotes();
        let features = extract_features(&trades, &quotes);

        assert_eq!(features.len(), 5);
        assert_eq!(features[0].tick_direction, 0.0); // first trade
        assert_eq!(features[1].tick_direction, 1.0); // uptick
        assert_eq!(features[3].tick_direction, -1.0); // downtick
    }

    #[test]
    fn test_logistic_classifier() {
        let classifier = LogisticClassifier::default_trade_classifier();

        // A trade above midpoint should be classified as Buy
        let buy_features = TradeFeatures {
            relative_price: 0.8,
            relative_size: 1.0,
            tick_direction: 1.0,
            spread: 0.01,
            quote_imbalance: 0.3,
            time_delta_ms: 100.0,
        };
        let result = classifier.classify(&buy_features);
        assert_eq!(result.predicted_side, TradeSide::Buy);

        // A trade below midpoint should be classified as Sell
        let sell_features = TradeFeatures {
            relative_price: -0.8,
            relative_size: 1.0,
            tick_direction: -1.0,
            spread: 0.01,
            quote_imbalance: -0.3,
            time_delta_ms: 100.0,
        };
        let result = classifier.classify(&sell_features);
        assert_eq!(result.predicted_side, TradeSide::Sell);
    }
}
