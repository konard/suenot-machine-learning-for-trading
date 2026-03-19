//! # Liquidity Prediction for Trading
//!
//! A from-scratch implementation of liquidity metrics computation, LSTM-based
//! liquidity forecasting, gradient-boosted regime classification, and optimal
//! order sizing, with Bybit exchange data integration.

use anyhow::Result;
use ndarray::{Array1, Array2};
use rand::Rng;
use serde::Deserialize;

// ---------------------------------------------------------------------------
// Activation functions
// ---------------------------------------------------------------------------

/// Sigmoid activation: sigma(x) = 1 / (1 + exp(-x))
fn sigmoid(x: f64) -> f64 {
    let clamped = x.clamp(-15.0, 15.0);
    1.0 / (1.0 + (-clamped).exp())
}

/// Element-wise sigmoid over a 1-D array.
fn sigmoid_vec(v: &Array1<f64>) -> Array1<f64> {
    v.mapv(sigmoid)
}

/// Element-wise tanh over a 1-D array.
fn tanh_vec(v: &Array1<f64>) -> Array1<f64> {
    v.mapv(f64::tanh)
}

// ---------------------------------------------------------------------------
// Xavier weight initialization
// ---------------------------------------------------------------------------

fn xavier_matrix(rows: usize, cols: usize, rng: &mut impl Rng) -> Array2<f64> {
    let limit = (6.0 / (rows + cols) as f64).sqrt();
    Array2::from_shape_fn((rows, cols), |_| rng.gen_range(-limit..limit))
}

fn zeros1(n: usize) -> Array1<f64> {
    Array1::zeros(n)
}

// ---------------------------------------------------------------------------
// Orderbook data structures
// ---------------------------------------------------------------------------

/// A single price level in the order book.
#[derive(Debug, Clone)]
pub struct PriceLevel {
    pub price: f64,
    pub quantity: f64,
}

/// A snapshot of the order book at a point in time.
#[derive(Debug, Clone)]
pub struct OrderbookSnapshot {
    pub bids: Vec<PriceLevel>,
    pub asks: Vec<PriceLevel>,
    pub timestamp: u64,
}

impl OrderbookSnapshot {
    /// Returns the mid-price.
    pub fn mid_price(&self) -> f64 {
        if self.bids.is_empty() || self.asks.is_empty() {
            return 0.0;
        }
        (self.bids[0].price + self.asks[0].price) / 2.0
    }

    /// Returns the best bid price.
    pub fn best_bid(&self) -> f64 {
        self.bids.first().map(|l| l.price).unwrap_or(0.0)
    }

    /// Returns the best ask price.
    pub fn best_ask(&self) -> f64 {
        self.asks.first().map(|l| l.price).unwrap_or(0.0)
    }
}

// ---------------------------------------------------------------------------
// Liquidity Metrics
// ---------------------------------------------------------------------------

/// Computed liquidity metrics from an orderbook snapshot.
#[derive(Debug, Clone)]
pub struct LiquidityMetrics {
    /// Absolute bid-ask spread
    pub spread: f64,
    /// Relative spread (spread / mid_price)
    pub relative_spread: f64,
    /// Mid-price
    pub mid_price: f64,
    /// Cumulative bid depth at N levels
    pub bid_depth: Vec<f64>,
    /// Cumulative ask depth at N levels
    pub ask_depth: Vec<f64>,
    /// Total depth (bid + ask) at N levels
    pub total_depth: Vec<f64>,
    /// Order flow imbalance at N levels
    pub ofi: f64,
    /// Composite liquidity score (higher = more liquid)
    pub liquidity_score: f64,
}

impl LiquidityMetrics {
    /// Compute liquidity metrics from an orderbook snapshot.
    /// `max_levels` controls how many levels of depth to compute.
    pub fn compute(snapshot: &OrderbookSnapshot, max_levels: usize) -> Self {
        let mid = snapshot.mid_price();
        let best_bid = snapshot.best_bid();
        let best_ask = snapshot.best_ask();
        let spread = best_ask - best_bid;
        let relative_spread = if mid > 0.0 { spread / mid } else { 0.0 };

        // Cumulative depth at each level
        let mut bid_depth = Vec::with_capacity(max_levels);
        let mut ask_depth = Vec::with_capacity(max_levels);
        let mut total_depth = Vec::with_capacity(max_levels);

        let mut cum_bid = 0.0;
        let mut cum_ask = 0.0;

        for i in 0..max_levels {
            if i < snapshot.bids.len() {
                cum_bid += snapshot.bids[i].quantity;
            }
            if i < snapshot.asks.len() {
                cum_ask += snapshot.asks[i].quantity;
            }
            bid_depth.push(cum_bid);
            ask_depth.push(cum_ask);
            total_depth.push(cum_bid + cum_ask);
        }

        // OFI at max depth
        let total_bid = *bid_depth.last().unwrap_or(&0.0);
        let total_ask = *ask_depth.last().unwrap_or(&0.0);
        let ofi = if (total_bid + total_ask) > 0.0 {
            (total_bid - total_ask) / (total_bid + total_ask)
        } else {
            0.0
        };

        // Composite liquidity score: weighted combination
        // Higher depth and lower spread = more liquid
        let depth_5 = if max_levels >= 5 {
            total_depth[4]
        } else {
            *total_depth.last().unwrap_or(&0.0)
        };

        let liquidity_score = if relative_spread > 0.0 {
            depth_5 / (relative_spread * 10000.0 + 1.0)
        } else {
            depth_5
        };

        LiquidityMetrics {
            spread,
            relative_spread,
            mid_price: mid,
            bid_depth,
            ask_depth,
            total_depth,
            ofi,
            liquidity_score,
        }
    }

    /// Build a feature vector for ML models.
    /// Features: [relative_spread, depth_1, depth_5, depth_10, ofi, liquidity_score]
    pub fn to_feature_vector(&self) -> Array1<f64> {
        let depth_1 = self.total_depth.first().copied().unwrap_or(0.0);
        let depth_5 = if self.total_depth.len() >= 5 {
            self.total_depth[4]
        } else {
            *self.total_depth.last().unwrap_or(&0.0)
        };
        let depth_10 = if self.total_depth.len() >= 10 {
            self.total_depth[9]
        } else {
            *self.total_depth.last().unwrap_or(&0.0)
        };

        Array1::from(vec![
            self.relative_spread * 10000.0, // in bps
            depth_1,
            depth_5,
            depth_10,
            self.ofi,
            self.liquidity_score,
        ])
    }
}

/// Estimate the resilience proxy given depth before and after an event.
pub fn resilience_proxy(depth_before: f64, depth_after: f64) -> f64 {
    if depth_before > 0.0 {
        depth_after / depth_before
    } else {
        0.0
    }
}

/// Estimate slippage from walking the book.
/// `order_size` is the quantity to execute,
/// `depth` is total available depth,
/// `avg_price_gap` is the average price distance per level.
pub fn estimate_slippage(order_size: f64, depth: f64, avg_price_gap: f64) -> f64 {
    if depth <= 0.0 {
        return f64::INFINITY;
    }
    (order_size * order_size) / (2.0 * depth) * avg_price_gap
}

// ---------------------------------------------------------------------------
// LSTM Cell
// ---------------------------------------------------------------------------

/// A single LSTM cell implementing all four gates.
#[derive(Clone)]
pub struct LstmCell {
    pub input_size: usize,
    pub hidden_size: usize,
    // Forget gate
    pub wf: Array2<f64>,
    pub bf: Array1<f64>,
    // Input gate
    pub wi: Array2<f64>,
    pub bi: Array1<f64>,
    // Candidate
    pub wc: Array2<f64>,
    pub bc: Array1<f64>,
    // Output gate
    pub wo: Array2<f64>,
    pub bo: Array1<f64>,
}

/// Hidden + cell state pair.
#[derive(Clone, Debug)]
pub struct LstmState {
    pub h: Array1<f64>,
    pub c: Array1<f64>,
}

impl LstmCell {
    /// Create a new LSTM cell with Xavier-initialized weights.
    pub fn new(input_size: usize, hidden_size: usize, rng: &mut impl Rng) -> Self {
        let total = input_size + hidden_size;
        LstmCell {
            input_size,
            hidden_size,
            wf: xavier_matrix(hidden_size, total, rng),
            bf: zeros1(hidden_size),
            wi: xavier_matrix(hidden_size, total, rng),
            bi: zeros1(hidden_size),
            wc: xavier_matrix(hidden_size, total, rng),
            bc: zeros1(hidden_size),
            wo: xavier_matrix(hidden_size, total, rng),
            bo: zeros1(hidden_size),
        }
    }

    /// Zero initial state.
    pub fn zero_state(&self) -> LstmState {
        LstmState {
            h: zeros1(self.hidden_size),
            c: zeros1(self.hidden_size),
        }
    }

    /// Forward pass for one timestep.
    pub fn forward(&self, x: &Array1<f64>, state: &LstmState) -> LstmState {
        // Concatenate [h, x]
        let mut hx = Array1::zeros(self.hidden_size + self.input_size);
        hx.slice_mut(ndarray::s![..self.hidden_size])
            .assign(&state.h);
        hx.slice_mut(ndarray::s![self.hidden_size..])
            .assign(x);

        let f = sigmoid_vec(&(self.wf.dot(&hx) + &self.bf));
        let i = sigmoid_vec(&(self.wi.dot(&hx) + &self.bi));
        let c_cand = tanh_vec(&(self.wc.dot(&hx) + &self.bc));
        let o = sigmoid_vec(&(self.wo.dot(&hx) + &self.bo));

        let new_c = &f * &state.c + &i * &c_cand;
        let new_h = &o * &tanh_vec(&new_c);

        LstmState {
            h: new_h,
            c: new_c,
        }
    }
}

// ---------------------------------------------------------------------------
// LSTM Liquidity Forecaster
// ---------------------------------------------------------------------------

/// LSTM-based forecaster that predicts next-period liquidity (depth at 5 levels).
pub struct LstmForecaster {
    pub cell: LstmCell,
    /// Output projection: hidden_size -> 1
    pub w_out: Array1<f64>,
    pub b_out: f64,
}

impl LstmForecaster {
    /// Create a new forecaster.
    pub fn new(input_size: usize, hidden_size: usize, rng: &mut impl Rng) -> Self {
        let limit = (2.0 / hidden_size as f64).sqrt();
        let w_out = Array1::from_shape_fn(hidden_size, |_| rng.gen_range(-limit..limit));
        LstmForecaster {
            cell: LstmCell::new(input_size, hidden_size, rng),
            w_out,
            b_out: 0.0,
        }
    }

    /// Predict next-period depth from a sequence of feature vectors.
    /// Returns the predicted total depth at 5 levels.
    pub fn predict(&self, sequence: &[Array1<f64>]) -> f64 {
        let mut state = self.cell.zero_state();
        for x in sequence {
            state = self.cell.forward(x, &state);
        }
        // Linear projection
        self.w_out.dot(&state.h) + self.b_out
    }

    /// Simple training step (single-sample SGD) on a sequence -> target pair.
    /// Uses numerical gradient approximation for simplicity.
    pub fn train_step(
        &mut self,
        sequence: &[Array1<f64>],
        target: f64,
        learning_rate: f64,
    ) -> f64 {
        let prediction = self.predict(sequence);
        let error = prediction - target;
        let loss = error * error;

        // Update output layer with gradient of MSE
        let mut state = self.cell.zero_state();
        for x in sequence {
            state = self.cell.forward(x, &state);
        }

        // Gradient for w_out: d(loss)/d(w_out) = 2 * error * h
        let grad_w = state.h.mapv(|hi| 2.0 * error * hi);
        self.w_out = &self.w_out - &(grad_w * learning_rate);
        self.b_out -= 2.0 * error * learning_rate;

        loss
    }
}

// ---------------------------------------------------------------------------
// Liquidity Regime Classifier (Gradient-Boosted Stumps)
// ---------------------------------------------------------------------------

/// Liquidity regime categories.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LiquidityRegime {
    High,
    Medium,
    Low,
}

impl LiquidityRegime {
    pub fn as_str(&self) -> &str {
        match self {
            LiquidityRegime::High => "High",
            LiquidityRegime::Medium => "Medium",
            LiquidityRegime::Low => "Low",
        }
    }
}

/// A decision stump: splits on a single feature at a threshold.
#[derive(Debug, Clone)]
struct Stump {
    feature_idx: usize,
    threshold: f64,
    left_value: f64,
    right_value: f64,
    weight: f64,
}

impl Stump {
    fn predict(&self, features: &[f64]) -> f64 {
        let val = features.get(self.feature_idx).copied().unwrap_or(0.0);
        if val <= self.threshold {
            self.left_value * self.weight
        } else {
            self.right_value * self.weight
        }
    }
}

/// Gradient-boosted stump ensemble for regime classification.
pub struct LiquidityRegimeClassifier {
    stumps: Vec<Stump>,
    /// Thresholds for regime assignment (33rd and 67th percentile of depth)
    pub low_threshold: f64,
    pub high_threshold: f64,
}

impl LiquidityRegimeClassifier {
    /// Create a new classifier by fitting stumps to training data.
    /// `features` is a Vec of feature vectors, `depths` is the corresponding total depth.
    pub fn fit(features: &[Vec<f64>], depths: &[f64], n_stumps: usize) -> Self {
        let n = depths.len();
        if n == 0 {
            return LiquidityRegimeClassifier {
                stumps: vec![],
                low_threshold: 0.0,
                high_threshold: 0.0,
            };
        }

        // Compute percentile thresholds
        let mut sorted_depths = depths.to_vec();
        sorted_depths.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let low_threshold = sorted_depths[n / 3];
        let high_threshold = sorted_depths[2 * n / 3];

        // Convert depths to scores: -1 (low), 0 (medium), 1 (high)
        let targets: Vec<f64> = depths
            .iter()
            .map(|&d| {
                if d < low_threshold {
                    -1.0
                } else if d > high_threshold {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();

        // Residuals for boosting
        let mut residuals = targets.clone();
        let mut stumps = Vec::new();
        let num_features = features.first().map(|f| f.len()).unwrap_or(0);
        let learning_rate = 0.1;

        for _ in 0..n_stumps {
            // Find best stump
            let mut best_stump: Option<Stump> = None;
            let mut best_loss = f64::INFINITY;

            for feat_idx in 0..num_features {
                // Try a few thresholds
                let mut vals: Vec<f64> = features.iter().map(|f| f[feat_idx]).collect();
                vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let n_thresholds = 10.min(vals.len());
                for ti in 0..n_thresholds {
                    let threshold = vals[ti * vals.len() / n_thresholds.max(1)];

                    // Compute left/right mean of residuals
                    let (mut left_sum, mut left_cnt) = (0.0, 0);
                    let (mut right_sum, mut right_cnt) = (0.0, 0);
                    for (j, f) in features.iter().enumerate() {
                        if f[feat_idx] <= threshold {
                            left_sum += residuals[j];
                            left_cnt += 1;
                        } else {
                            right_sum += residuals[j];
                            right_cnt += 1;
                        }
                    }
                    let left_val = if left_cnt > 0 {
                        left_sum / left_cnt as f64
                    } else {
                        0.0
                    };
                    let right_val = if right_cnt > 0 {
                        right_sum / right_cnt as f64
                    } else {
                        0.0
                    };

                    // MSE loss
                    let mut loss = 0.0;
                    for (j, f) in features.iter().enumerate() {
                        let pred = if f[feat_idx] <= threshold {
                            left_val
                        } else {
                            right_val
                        };
                        let diff = residuals[j] - pred;
                        loss += diff * diff;
                    }

                    if loss < best_loss {
                        best_loss = loss;
                        best_stump = Some(Stump {
                            feature_idx: feat_idx,
                            threshold,
                            left_value: left_val,
                            right_value: right_val,
                            weight: learning_rate,
                        });
                    }
                }
            }

            if let Some(stump) = best_stump {
                // Update residuals
                for (j, f) in features.iter().enumerate() {
                    let pred = stump.predict(f);
                    residuals[j] -= pred;
                }
                stumps.push(stump);
            }
        }

        LiquidityRegimeClassifier {
            stumps,
            low_threshold,
            high_threshold,
        }
    }

    /// Predict the liquidity regime for a given feature vector.
    pub fn predict(&self, features: &[f64]) -> LiquidityRegime {
        let score: f64 = self.stumps.iter().map(|s| s.predict(features)).sum();
        if score > 0.33 {
            LiquidityRegime::High
        } else if score < -0.33 {
            LiquidityRegime::Low
        } else {
            LiquidityRegime::Medium
        }
    }

    /// Get the raw score for a feature vector.
    pub fn predict_score(&self, features: &[f64]) -> f64 {
        self.stumps.iter().map(|s| s.predict(features)).sum()
    }
}

// ---------------------------------------------------------------------------
// Optimal Order Sizer
// ---------------------------------------------------------------------------

/// Computes optimal child order size given predicted liquidity.
pub struct OptimalOrderSizer {
    /// Maximum participation rate (fraction of predicted depth)
    pub participation_rate: f64,
}

impl OptimalOrderSizer {
    pub fn new(participation_rate: f64) -> Self {
        OptimalOrderSizer {
            participation_rate: participation_rate.clamp(0.01, 1.0),
        }
    }

    /// Compute the optimal child order size.
    /// `remaining_quantity` is how much is left to execute.
    /// `predicted_depth` is the predicted available depth at N levels.
    pub fn optimal_size(&self, remaining_quantity: f64, predicted_depth: f64) -> f64 {
        let max_by_depth = self.participation_rate * predicted_depth;
        remaining_quantity.min(max_by_depth).max(0.0)
    }

    /// Compute the recommended size adjusted by regime.
    pub fn regime_adjusted_size(
        &self,
        remaining_quantity: f64,
        predicted_depth: f64,
        regime: LiquidityRegime,
    ) -> f64 {
        let base = self.optimal_size(remaining_quantity, predicted_depth);
        match regime {
            LiquidityRegime::High => base * 1.5,   // more aggressive
            LiquidityRegime::Medium => base,        // baseline
            LiquidityRegime::Low => base * 0.5,     // more conservative
        }
        .min(remaining_quantity)
        .max(0.0)
    }
}

// ---------------------------------------------------------------------------
// Bybit API Client
// ---------------------------------------------------------------------------

/// Bybit V5 REST API client for orderbook data.
pub struct BybitClient {
    base_url: String,
}

// Bybit API response structures
#[derive(Deserialize, Debug)]
pub struct BybitResponse {
    #[serde(rename = "retCode")]
    pub ret_code: i64,
    #[serde(rename = "retMsg")]
    pub ret_msg: String,
    pub result: BybitOrderbookResult,
}

#[derive(Deserialize, Debug)]
pub struct BybitOrderbookResult {
    #[serde(rename = "s")]
    pub symbol: String,
    #[serde(rename = "b")]
    pub bids: Vec<Vec<String>>,
    #[serde(rename = "a")]
    pub asks: Vec<Vec<String>>,
    #[serde(rename = "ts")]
    pub timestamp: u64,
}

impl BybitClient {
    pub fn new() -> Self {
        BybitClient {
            base_url: "https://api.bybit.com".to_string(),
        }
    }

    /// Fetch orderbook snapshot for a given symbol.
    pub fn fetch_orderbook(&self, symbol: &str, limit: u32) -> Result<OrderbookSnapshot> {
        let url = format!(
            "{}/v5/market/orderbook?category=linear&symbol={}&limit={}",
            self.base_url, symbol, limit
        );

        let resp: BybitResponse = reqwest::blocking::get(&url)?.json()?;

        if resp.ret_code != 0 {
            anyhow::bail!("Bybit API error: {}", resp.ret_msg);
        }

        let bids: Vec<PriceLevel> = resp
            .result
            .bids
            .iter()
            .filter_map(|entry| {
                if entry.len() >= 2 {
                    Some(PriceLevel {
                        price: entry[0].parse().unwrap_or(0.0),
                        quantity: entry[1].parse().unwrap_or(0.0),
                    })
                } else {
                    None
                }
            })
            .collect();

        let asks: Vec<PriceLevel> = resp
            .result
            .asks
            .iter()
            .filter_map(|entry| {
                if entry.len() >= 2 {
                    Some(PriceLevel {
                        price: entry[0].parse().unwrap_or(0.0),
                        quantity: entry[1].parse().unwrap_or(0.0),
                    })
                } else {
                    None
                }
            })
            .collect();

        Ok(OrderbookSnapshot {
            bids,
            asks,
            timestamp: resp.result.timestamp,
        })
    }
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Unit Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_snapshot() -> OrderbookSnapshot {
        OrderbookSnapshot {
            bids: vec![
                PriceLevel { price: 100.0, quantity: 10.0 },
                PriceLevel { price: 99.5, quantity: 15.0 },
                PriceLevel { price: 99.0, quantity: 20.0 },
                PriceLevel { price: 98.5, quantity: 25.0 },
                PriceLevel { price: 98.0, quantity: 30.0 },
            ],
            asks: vec![
                PriceLevel { price: 100.5, quantity: 8.0 },
                PriceLevel { price: 101.0, quantity: 12.0 },
                PriceLevel { price: 101.5, quantity: 18.0 },
                PriceLevel { price: 102.0, quantity: 22.0 },
                PriceLevel { price: 102.5, quantity: 28.0 },
            ],
            timestamp: 1700000000000,
        }
    }

    #[test]
    fn test_mid_price() {
        let snap = make_snapshot();
        let mid = snap.mid_price();
        assert!((mid - 100.25).abs() < 1e-6);
    }

    #[test]
    fn test_liquidity_metrics_spread() {
        let snap = make_snapshot();
        let metrics = LiquidityMetrics::compute(&snap, 5);
        assert!((metrics.spread - 0.5).abs() < 1e-6);
        assert!(metrics.relative_spread > 0.0);
        assert!(metrics.relative_spread < 0.01); // spread is small relative to price
    }

    #[test]
    fn test_liquidity_metrics_depth() {
        let snap = make_snapshot();
        let metrics = LiquidityMetrics::compute(&snap, 5);
        // Bid depth at level 1: 10.0
        assert!((metrics.bid_depth[0] - 10.0).abs() < 1e-6);
        // Bid depth at level 5: 10+15+20+25+30 = 100
        assert!((metrics.bid_depth[4] - 100.0).abs() < 1e-6);
        // Total depth at level 5: 100 + 88 = 188
        assert!((metrics.total_depth[4] - 188.0).abs() < 1e-6);
    }

    #[test]
    fn test_ofi() {
        let snap = make_snapshot();
        let metrics = LiquidityMetrics::compute(&snap, 5);
        // Bid depth 100, ask depth 88, OFI = (100-88)/(100+88) = 12/188
        let expected_ofi = 12.0 / 188.0;
        assert!((metrics.ofi - expected_ofi).abs() < 1e-6);
    }

    #[test]
    fn test_feature_vector_length() {
        let snap = make_snapshot();
        let metrics = LiquidityMetrics::compute(&snap, 10);
        let fv = metrics.to_feature_vector();
        assert_eq!(fv.len(), 6);
    }

    #[test]
    fn test_lstm_cell_forward() {
        let mut rng = rand::thread_rng();
        let cell = LstmCell::new(4, 8, &mut rng);
        let state = cell.zero_state();
        let x = Array1::from(vec![1.0, 2.0, 3.0, 4.0]);
        let new_state = cell.forward(&x, &state);
        assert_eq!(new_state.h.len(), 8);
        assert_eq!(new_state.c.len(), 8);
        // Hidden state should be non-zero after input
        assert!(new_state.h.iter().any(|v| v.abs() > 1e-10));
    }

    #[test]
    fn test_lstm_forecaster_predict() {
        let mut rng = rand::thread_rng();
        let forecaster = LstmForecaster::new(6, 16, &mut rng);
        let sequence: Vec<Array1<f64>> = (0..5)
            .map(|_| Array1::from(vec![0.5, 10.0, 50.0, 100.0, 0.1, 25.0]))
            .collect();
        let pred = forecaster.predict(&sequence);
        // Should return a finite number
        assert!(pred.is_finite());
    }

    #[test]
    fn test_regime_classifier() {
        // Create synthetic training data
        let features: Vec<Vec<f64>> = (0..30)
            .map(|i| vec![i as f64 * 0.1, i as f64 * 2.0, i as f64 * 0.5])
            .collect();
        let depths: Vec<f64> = (0..30).map(|i| i as f64 * 10.0).collect();

        let classifier = LiquidityRegimeClassifier::fit(&features, &depths, 10);

        // Low depth should classify as Low
        let low_pred = classifier.predict(&[0.0, 0.0, 0.0]);
        // High depth features should classify as High
        let high_pred = classifier.predict(&[3.0, 60.0, 15.0]);

        // At minimum, we should get distinct regimes for extreme inputs
        assert_ne!(
            format!("{:?}", low_pred),
            format!("{:?}", high_pred),
            "Extreme inputs should produce different regimes"
        );
    }

    #[test]
    fn test_optimal_order_sizer() {
        let sizer = OptimalOrderSizer::new(0.10);

        // If predicted depth is 100, max order = 10% * 100 = 10
        let size = sizer.optimal_size(50.0, 100.0);
        assert!((size - 10.0).abs() < 1e-6);

        // If remaining is less than max, use remaining
        let size2 = sizer.optimal_size(5.0, 100.0);
        assert!((size2 - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_regime_adjusted_sizing() {
        let sizer = OptimalOrderSizer::new(0.10);
        let depth = 100.0;
        let remaining = 50.0;

        let high_size = sizer.regime_adjusted_size(remaining, depth, LiquidityRegime::High);
        let med_size = sizer.regime_adjusted_size(remaining, depth, LiquidityRegime::Medium);
        let low_size = sizer.regime_adjusted_size(remaining, depth, LiquidityRegime::Low);

        assert!(high_size > med_size);
        assert!(med_size > low_size);
    }

    #[test]
    fn test_resilience_proxy() {
        let r = resilience_proxy(100.0, 95.0);
        assert!((r - 0.95).abs() < 1e-6);

        let r2 = resilience_proxy(100.0, 110.0);
        assert!((r2 - 1.10).abs() < 1e-6);
    }

    #[test]
    fn test_estimate_slippage() {
        let slip = estimate_slippage(10.0, 100.0, 0.5);
        // 10^2 / (2*100) * 0.5 = 100/200 * 0.5 = 0.25
        assert!((slip - 0.25).abs() < 1e-6);
    }
}
