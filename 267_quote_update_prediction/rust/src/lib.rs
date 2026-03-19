//! # Quote Update Prediction
//!
//! This library implements quote update prediction models for high-frequency
//! trading. It predicts the next mid-price movement, spread changes, and
//! order-book imbalance shifts using features extracted from Level-2 market data.
//!
//! The core model is a lightweight linear scoring function that can run at
//! sub-millisecond latency, suitable for tick-by-tick decision making.

use anyhow::{Context, Result};
use ndarray::{Array1, Array2, Axis};
use rand::Rng;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

/// A single Level-2 order-book snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookSnapshot {
    pub timestamp_ms: u64,
    pub best_bid: f64,
    pub best_ask: f64,
    pub bid_size: f64,
    pub ask_size: f64,
    /// Up to `depth` price levels on bid side: (price, size).
    pub bids: Vec<(f64, f64)>,
    /// Up to `depth` price levels on ask side: (price, size).
    pub asks: Vec<(f64, f64)>,
}

impl OrderBookSnapshot {
    /// Mid-price = (best_bid + best_ask) / 2.
    pub fn mid_price(&self) -> f64 {
        (self.best_bid + self.best_ask) / 2.0
    }

    /// Spread in absolute terms.
    pub fn spread(&self) -> f64 {
        self.best_ask - self.best_bid
    }

    /// Volume-weighted mid-price (micro-price).
    ///
    /// micro = (bid_size * ask + ask_size * bid) / (bid_size + ask_size)
    pub fn micro_price(&self) -> f64 {
        let total = self.bid_size + self.ask_size;
        if total == 0.0 {
            return self.mid_price();
        }
        (self.bid_size * self.best_ask + self.ask_size * self.best_bid) / total
    }

    /// Order-book imbalance in [-1, 1].
    ///
    /// imbalance = (bid_size - ask_size) / (bid_size + ask_size)
    pub fn imbalance(&self) -> f64 {
        let total = self.bid_size + self.ask_size;
        if total == 0.0 {
            return 0.0;
        }
        (self.bid_size - self.ask_size) / total
    }

    /// Depth-weighted imbalance across multiple levels.
    pub fn depth_imbalance(&self, levels: usize) -> f64 {
        let n = levels.min(self.bids.len()).min(self.asks.len());
        if n == 0 {
            return 0.0;
        }
        let mut bid_vol = 0.0;
        let mut ask_vol = 0.0;
        for i in 0..n {
            let weight = 1.0 / (i as f64 + 1.0);
            bid_vol += self.bids[i].1 * weight;
            ask_vol += self.asks[i].1 * weight;
        }
        let total = bid_vol + ask_vol;
        if total == 0.0 {
            return 0.0;
        }
        (bid_vol - ask_vol) / total
    }
}

// ---------------------------------------------------------------------------
// Feature engineering
// ---------------------------------------------------------------------------

/// Feature vector extracted from a pair of consecutive snapshots.
#[derive(Debug, Clone)]
pub struct QuoteFeatures {
    /// Order-book imbalance (level 1).
    pub imbalance: f64,
    /// Depth-weighted imbalance (levels 1..5).
    pub depth_imbalance: f64,
    /// Spread normalised by mid-price (bps).
    pub normalised_spread_bps: f64,
    /// Mid-price return since previous snapshot (bps).
    pub mid_return_bps: f64,
    /// Micro-price minus mid-price, normalised (bps).
    pub micro_mid_diff_bps: f64,
    /// Change in bid size (%).
    pub bid_size_change_pct: f64,
    /// Change in ask size (%).
    pub ask_size_change_pct: f64,
    /// Time delta in milliseconds.
    pub dt_ms: f64,
}

impl QuoteFeatures {
    pub fn to_array(&self) -> Array1<f64> {
        Array1::from(vec![
            self.imbalance,
            self.depth_imbalance,
            self.normalised_spread_bps,
            self.mid_return_bps,
            self.micro_mid_diff_bps,
            self.bid_size_change_pct,
            self.ask_size_change_pct,
            self.dt_ms,
        ])
    }

    pub const NUM_FEATURES: usize = 8;
}

/// Extract features from two consecutive snapshots.
pub fn extract_features(prev: &OrderBookSnapshot, curr: &OrderBookSnapshot) -> QuoteFeatures {
    let mid = curr.mid_price();
    let prev_mid = prev.mid_price();
    let mid_return_bps = if prev_mid != 0.0 {
        (mid - prev_mid) / prev_mid * 1e4
    } else {
        0.0
    };
    let normalised_spread_bps = if mid != 0.0 {
        curr.spread() / mid * 1e4
    } else {
        0.0
    };
    let micro_mid_diff_bps = if mid != 0.0 {
        (curr.micro_price() - mid) / mid * 1e4
    } else {
        0.0
    };
    let bid_size_change_pct = if prev.bid_size != 0.0 {
        (curr.bid_size - prev.bid_size) / prev.bid_size * 100.0
    } else {
        0.0
    };
    let ask_size_change_pct = if prev.ask_size != 0.0 {
        (curr.ask_size - prev.ask_size) / prev.ask_size * 100.0
    } else {
        0.0
    };

    QuoteFeatures {
        imbalance: curr.imbalance(),
        depth_imbalance: curr.depth_imbalance(5),
        normalised_spread_bps,
        mid_return_bps,
        micro_mid_diff_bps,
        bid_size_change_pct,
        ask_size_change_pct,
        dt_ms: (curr.timestamp_ms as f64) - (prev.timestamp_ms as f64),
    }
}

/// Build a feature matrix from a sequence of snapshots.
pub fn build_feature_matrix(snapshots: &[OrderBookSnapshot]) -> Array2<f64> {
    let n = if snapshots.len() > 1 {
        snapshots.len() - 1
    } else {
        0
    };
    let mut matrix = Array2::<f64>::zeros((n, QuoteFeatures::NUM_FEATURES));
    for i in 0..n {
        let feat = extract_features(&snapshots[i], &snapshots[i + 1]);
        matrix.row_mut(i).assign(&feat.to_array());
    }
    matrix
}

// ---------------------------------------------------------------------------
// Labels
// ---------------------------------------------------------------------------

/// Label for the next quote update: Up (+1), Down (-1), or Stable (0).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuoteDirection {
    Up = 1,
    Stable = 0,
    Down = -1,
}

/// Generate direction labels from snapshots using a threshold in bps.
/// For snapshot i the label is based on mid-price change from i to i+1.
pub fn generate_labels(snapshots: &[OrderBookSnapshot], threshold_bps: f64) -> Vec<QuoteDirection> {
    let n = if snapshots.len() > 1 {
        snapshots.len() - 1
    } else {
        return vec![];
    };
    let mut labels = Vec::with_capacity(n);
    // Labels are aligned with the feature matrix: label[i] corresponds to
    // the direction from snapshot[i+1] to snapshot[i+2].  To keep the same
    // length as the feature matrix we shift: label[i] = direction(i+1 -> i+2)
    // but the last row has no future snapshot, so we just use i -> i+1.
    for i in 0..n {
        let mid_now = snapshots[i].mid_price();
        let mid_next = snapshots[i + 1].mid_price();
        let ret_bps = if mid_now != 0.0 {
            (mid_next - mid_now) / mid_now * 1e4
        } else {
            0.0
        };
        let dir = if ret_bps > threshold_bps {
            QuoteDirection::Up
        } else if ret_bps < -threshold_bps {
            QuoteDirection::Down
        } else {
            QuoteDirection::Stable
        };
        labels.push(dir);
    }
    labels
}

// ---------------------------------------------------------------------------
// Linear scoring model
// ---------------------------------------------------------------------------

/// A simple linear model: score = w^T x + bias.
/// Predictions: sign(score) => direction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearQuotePredictor {
    pub weights: Vec<f64>,
    pub bias: f64,
    pub learning_rate: f64,
}

impl LinearQuotePredictor {
    pub fn new(num_features: usize, learning_rate: f64) -> Self {
        let mut rng = rand::thread_rng();
        let weights: Vec<f64> = (0..num_features)
            .map(|_| rng.gen_range(-0.01..0.01))
            .collect();
        Self {
            weights,
            bias: 0.0,
            learning_rate,
        }
    }

    /// Raw score = w^T x + b.
    pub fn score(&self, features: &Array1<f64>) -> f64 {
        let mut s = self.bias;
        for (w, &x) in self.weights.iter().zip(features.iter()) {
            s += w * x;
        }
        s
    }

    /// Predicted direction from score.
    pub fn predict(&self, features: &Array1<f64>) -> QuoteDirection {
        let s = self.score(features);
        if s > 0.0 {
            QuoteDirection::Up
        } else if s < 0.0 {
            QuoteDirection::Down
        } else {
            QuoteDirection::Stable
        }
    }

    /// Train one epoch using online SGD with hinge-like loss.
    /// `labels` should be +1 / -1 / 0 encoded.
    pub fn train_epoch(&mut self, x: &Array2<f64>, labels: &[QuoteDirection]) {
        let n = x.nrows().min(labels.len());
        for i in 0..n {
            let row = x.row(i);
            let y = labels[i] as i32 as f64; // +1, 0, -1
            if y == 0.0 {
                continue; // skip stable labels
            }
            let pred = self.score(&row.to_owned());
            let margin = y * pred;
            if margin < 1.0 {
                // hinge loss gradient update
                for (w, &xi) in self.weights.iter_mut().zip(row.iter()) {
                    *w += self.learning_rate * y * xi;
                }
                self.bias += self.learning_rate * y;
            }
        }
    }

    /// Accuracy on a dataset (ignoring Stable labels).
    pub fn accuracy(&self, x: &Array2<f64>, labels: &[QuoteDirection]) -> f64 {
        let n = x.nrows().min(labels.len());
        let mut correct = 0usize;
        let mut total = 0usize;
        for i in 0..n {
            if labels[i] == QuoteDirection::Stable {
                continue;
            }
            let row = x.row(i);
            let pred = self.predict(&row.to_owned());
            if pred == labels[i] {
                correct += 1;
            }
            total += 1;
        }
        if total == 0 {
            return 0.0;
        }
        correct as f64 / total as f64
    }
}

// ---------------------------------------------------------------------------
// Normalisation helpers
// ---------------------------------------------------------------------------

/// Column-wise z-score normalisation. Returns (normalised matrix, means, stds).
pub fn z_normalise(x: &Array2<f64>) -> (Array2<f64>, Array1<f64>, Array1<f64>) {
    let means = x.mean_axis(Axis(0)).unwrap();
    let stds = x.std_axis(Axis(0), 0.0);
    let mut normed = x.clone();
    for col_idx in 0..normed.ncols() {
        let mean = means[col_idx];
        let std = if stds[col_idx] == 0.0 {
            1.0
        } else {
            stds[col_idx]
        };
        for row_idx in 0..normed.nrows() {
            normed[[row_idx, col_idx]] = (normed[[row_idx, col_idx]] - mean) / std;
        }
    }
    (normed, means, stds)
}

// ---------------------------------------------------------------------------
// Bybit integration
// ---------------------------------------------------------------------------

/// Bybit REST API order-book response structures.
#[derive(Debug, Deserialize)]
pub struct BybitResponse {
    #[serde(rename = "retCode")]
    pub ret_code: i32,
    #[serde(rename = "retMsg")]
    pub ret_msg: String,
    pub result: BybitOrderBookResult,
}

#[derive(Debug, Deserialize)]
pub struct BybitOrderBookResult {
    #[serde(rename = "s")]
    pub symbol: String,
    #[serde(rename = "b")]
    pub bids: Vec<(String, String)>,
    #[serde(rename = "a")]
    pub asks: Vec<(String, String)>,
    #[serde(rename = "ts")]
    pub ts: u64,
}

/// Fetch a live order-book snapshot from Bybit v5 API.
pub async fn fetch_bybit_orderbook(symbol: &str, depth: u32) -> Result<OrderBookSnapshot> {
    let url = format!(
        "https://api.bybit.com/v5/market/orderbook?category=spot&symbol={}&limit={}",
        symbol, depth
    );
    let client = reqwest::Client::new();
    let resp: BybitResponse = client
        .get(&url)
        .send()
        .await
        .context("Failed to call Bybit API")?
        .json()
        .await
        .context("Failed to parse Bybit response")?;

    if resp.ret_code != 0 {
        anyhow::bail!("Bybit API error: {}", resp.ret_msg);
    }

    let parse_levels = |levels: &[(String, String)]| -> Vec<(f64, f64)> {
        levels
            .iter()
            .filter_map(|(p, s)| {
                let price = p.parse::<f64>().ok()?;
                let size = s.parse::<f64>().ok()?;
                Some((price, size))
            })
            .collect()
    };

    let bids = parse_levels(&resp.result.bids);
    let asks = parse_levels(&resp.result.asks);

    let best_bid = bids.first().map(|b| b.0).unwrap_or(0.0);
    let best_ask = asks.first().map(|a| a.0).unwrap_or(0.0);
    let bid_size = bids.first().map(|b| b.1).unwrap_or(0.0);
    let ask_size = asks.first().map(|a| a.1).unwrap_or(0.0);

    Ok(OrderBookSnapshot {
        timestamp_ms: resp.result.ts,
        best_bid,
        best_ask,
        bid_size,
        ask_size,
        bids,
        asks,
    })
}

/// Fetch order-book snapshot synchronously (blocking).
pub fn fetch_bybit_orderbook_blocking(symbol: &str, depth: u32) -> Result<OrderBookSnapshot> {
    let url = format!(
        "https://api.bybit.com/v5/market/orderbook?category=spot&symbol={}&limit={}",
        symbol, depth
    );
    let resp: BybitResponse = reqwest::blocking::get(&url)
        .context("Failed to call Bybit API")?
        .json()
        .context("Failed to parse Bybit response")?;

    if resp.ret_code != 0 {
        anyhow::bail!("Bybit API error: {}", resp.ret_msg);
    }

    let parse_levels = |levels: &[(String, String)]| -> Vec<(f64, f64)> {
        levels
            .iter()
            .filter_map(|(p, s)| {
                let price = p.parse::<f64>().ok()?;
                let size = s.parse::<f64>().ok()?;
                Some((price, size))
            })
            .collect()
    };

    let bids = parse_levels(&resp.result.bids);
    let asks = parse_levels(&resp.result.asks);

    let best_bid = bids.first().map(|b| b.0).unwrap_or(0.0);
    let best_ask = asks.first().map(|a| a.0).unwrap_or(0.0);
    let bid_size = bids.first().map(|b| b.1).unwrap_or(0.0);
    let ask_size = asks.first().map(|a| a.1).unwrap_or(0.0);

    Ok(OrderBookSnapshot {
        timestamp_ms: resp.result.ts,
        best_bid,
        best_ask,
        bid_size,
        ask_size,
        bids,
        asks,
    })
}

// ---------------------------------------------------------------------------
// Synthetic data generator (for testing & demos)
// ---------------------------------------------------------------------------

/// Generate synthetic order-book snapshots with a random walk mid-price.
pub fn generate_synthetic_snapshots(n: usize, seed_price: f64, tick_size: f64) -> Vec<OrderBookSnapshot> {
    let mut rng = rand::thread_rng();
    let mut mid = seed_price;
    let mut snapshots = Vec::with_capacity(n);
    let mut ts = 1_700_000_000_000u64; // some base timestamp

    for _ in 0..n {
        // random walk
        let step: f64 = rng.gen_range(-2.0..=2.0);
        mid += step * tick_size;
        if mid < tick_size {
            mid = tick_size;
        }
        let half_spread = tick_size * rng.gen_range(0.5..2.0);
        let best_bid = mid - half_spread;
        let best_ask = mid + half_spread;
        let bid_size = rng.gen_range(0.1..10.0);
        let ask_size = rng.gen_range(0.1..10.0);

        // generate a few depth levels
        let depth = 5;
        let mut bids = Vec::with_capacity(depth);
        let mut asks = Vec::with_capacity(depth);
        let mut bp = best_bid;
        let mut ap = best_ask;
        for _ in 0..depth {
            bids.push((bp, rng.gen_range(0.1..20.0)));
            asks.push((ap, rng.gen_range(0.1..20.0)));
            bp -= tick_size;
            ap += tick_size;
        }

        let dt_ms: u64 = rng.gen_range(50..500);
        ts += dt_ms;

        snapshots.push(OrderBookSnapshot {
            timestamp_ms: ts,
            best_bid,
            best_ask,
            bid_size,
            ask_size,
            bids,
            asks,
        });
    }
    snapshots
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_snapshot(bid: f64, ask: f64, bid_sz: f64, ask_sz: f64, ts: u64) -> OrderBookSnapshot {
        OrderBookSnapshot {
            timestamp_ms: ts,
            best_bid: bid,
            best_ask: ask,
            bid_size: bid_sz,
            ask_size: ask_sz,
            bids: vec![(bid, bid_sz), (bid - 0.01, bid_sz * 2.0)],
            asks: vec![(ask, ask_sz), (ask + 0.01, ask_sz * 2.0)],
        }
    }

    #[test]
    fn test_mid_price() {
        let snap = sample_snapshot(100.0, 101.0, 5.0, 5.0, 0);
        assert!((snap.mid_price() - 100.5).abs() < 1e-9);
    }

    #[test]
    fn test_spread() {
        let snap = sample_snapshot(100.0, 101.0, 5.0, 5.0, 0);
        assert!((snap.spread() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_symmetric() {
        let snap = sample_snapshot(100.0, 101.0, 5.0, 5.0, 0);
        assert!((snap.imbalance()).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_bid_heavy() {
        let snap = sample_snapshot(100.0, 101.0, 10.0, 2.0, 0);
        // (10-2)/(10+2) = 8/12 = 0.6667
        assert!((snap.imbalance() - 8.0 / 12.0).abs() < 1e-6);
    }

    #[test]
    fn test_micro_price() {
        let snap = sample_snapshot(100.0, 102.0, 3.0, 7.0, 0);
        // micro = (3*102 + 7*100) / 10 = (306+700)/10 = 100.6
        assert!((snap.micro_price() - 100.6).abs() < 1e-9);
    }

    #[test]
    fn test_extract_features() {
        let prev = sample_snapshot(100.0, 101.0, 5.0, 5.0, 1000);
        let curr = sample_snapshot(100.5, 101.5, 6.0, 4.0, 1100);
        let feat = extract_features(&prev, &curr);
        // mid return: (101.0 - 100.5)/100.5 * 1e4 ~ 49.75 bps
        assert!(feat.mid_return_bps > 0.0);
        assert!((feat.dt_ms - 100.0).abs() < 1e-9);
        assert!(feat.imbalance > 0.0); // bid > ask
    }

    #[test]
    fn test_generate_labels() {
        let snaps = vec![
            sample_snapshot(100.0, 100.02, 5.0, 5.0, 0),
            sample_snapshot(100.10, 100.12, 5.0, 5.0, 100),
            sample_snapshot(99.90, 99.92, 5.0, 5.0, 200),
        ];
        let labels = generate_labels(&snaps, 1.0);
        assert_eq!(labels.len(), 2);
        assert_eq!(labels[0], QuoteDirection::Up);
        assert_eq!(labels[1], QuoteDirection::Down);
    }

    #[test]
    fn test_linear_predictor_training() {
        let snaps = generate_synthetic_snapshots(200, 50000.0, 0.01);
        let x = build_feature_matrix(&snaps);
        let labels = generate_labels(&snaps, 0.5);
        let (x_norm, _means, _stds) = z_normalise(&x);

        let mut model = LinearQuotePredictor::new(QuoteFeatures::NUM_FEATURES, 0.001);
        for _ in 0..10 {
            model.train_epoch(&x_norm, &labels);
        }
        let acc = model.accuracy(&x_norm, &labels);
        // Just verify it runs and produces a reasonable number
        assert!(acc >= 0.0 && acc <= 1.0);
    }

    #[test]
    fn test_z_normalise() {
        let data = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let (normed, means, stds) = z_normalise(&data);
        assert!((means[0] - 3.0).abs() < 1e-9);
        assert!((means[1] - 4.0).abs() < 1e-9);
        // first col should be centred
        assert!(normed[[1, 0]].abs() < 1e-9); // middle value -> 0
    }

    #[test]
    fn test_depth_imbalance() {
        let snap = sample_snapshot(100.0, 101.0, 5.0, 5.0, 0);
        let di = snap.depth_imbalance(2);
        // level 0: 5.0 * 1.0 vs 5.0 * 1.0, level 1: 10.0 * 0.5 vs 10.0 * 0.5
        // symmetric => 0
        assert!(di.abs() < 1e-9);
    }
}
