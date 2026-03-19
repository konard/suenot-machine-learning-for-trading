use anyhow::Result;
use ndarray::{Array1, Array2, Axis};
use rand::Rng;
use serde::Deserialize;

// ---------------------------------------------------------------------------
// Linear algebra helpers
// ---------------------------------------------------------------------------

/// Softmax along axis-1 (each row independently).
fn softmax_rows(x: &Array2<f64>) -> Array2<f64> {
    let mut out = x.clone();
    for mut row in out.rows_mut() {
        let max = row.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        row.mapv_inplace(|v| (v - max).exp());
        let sum = row.sum();
        if sum > 0.0 {
            row.mapv_inplace(|v| v / sum);
        }
    }
    out
}

/// Softmax for a 1-D vector.
fn softmax_vec(x: &Array1<f64>) -> Array1<f64> {
    let max = x.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let exp = x.mapv(|v| (v - max).exp());
    let sum = exp.sum();
    if sum > 0.0 {
        exp / sum
    } else {
        exp
    }
}

/// ReLU element-wise.
fn relu(x: &Array1<f64>) -> Array1<f64> {
    x.mapv(|v| v.max(0.0))
}

/// Mean-pool rows of a matrix into a single vector.
fn mean_pool(x: &Array2<f64>) -> Array1<f64> {
    x.mean_axis(Axis(0)).unwrap()
}

/// Initialize a random matrix with Xavier-style scaling.
fn random_matrix(rows: usize, cols: usize) -> Array2<f64> {
    let mut rng = rand::thread_rng();
    let scale = (2.0 / (rows + cols) as f64).sqrt();
    Array2::from_shape_fn((rows, cols), |_| rng.gen_range(-scale..scale))
}

/// Initialize a random vector.
fn random_vector(len: usize) -> Array1<f64> {
    let mut rng = rand::thread_rng();
    let scale = (1.0 / len as f64).sqrt();
    Array1::from_shape_fn(len, |_| rng.gen_range(-scale..scale))
}

// ---------------------------------------------------------------------------
// LOB Encoder
// ---------------------------------------------------------------------------

/// Self-attention encoder for Limit Order Book price levels.
///
/// Operates over 2L rows (L bid + L ask levels) with relative positional bias.
pub struct LobEncoder {
    pub d_model: usize,
    pub w_q: Array2<f64>,
    pub w_k: Array2<f64>,
    pub w_v: Array2<f64>,
    pub w_out: Array2<f64>,
    /// Relative positional bias: bias[|i-j|] added to attention logits.
    pub pos_bias: Array1<f64>,
}

impl LobEncoder {
    /// Create a new LOB encoder.
    ///
    /// `d_model` -- dimension of level embeddings.
    /// `max_levels` -- maximum number of levels (2*L) for positional bias.
    pub fn new(d_model: usize, max_levels: usize) -> Self {
        Self {
            d_model,
            w_q: random_matrix(d_model, d_model),
            w_k: random_matrix(d_model, d_model),
            w_v: random_matrix(d_model, d_model),
            w_out: random_matrix(d_model, d_model),
            pos_bias: random_vector(max_levels),
        }
    }

    /// Forward pass: self-attention with relative positional bias.
    ///
    /// `x` has shape (num_levels, d_model).
    pub fn forward(&self, x: &Array2<f64>) -> Array2<f64> {
        let n = x.nrows();
        let dk = self.d_model as f64;

        let q = x.dot(&self.w_q); // (n, d)
        let k = x.dot(&self.w_k); // (n, d)
        let v = x.dot(&self.w_v); // (n, d)

        // Attention logits with relative positional bias.
        let mut logits = q.dot(&k.t()) / dk.sqrt(); // (n, n)
        for i in 0..n {
            for j in 0..n {
                let dist = if i > j { i - j } else { j - i };
                if dist < self.pos_bias.len() {
                    logits[[i, j]] += self.pos_bias[dist];
                }
            }
        }

        let attn = softmax_rows(&logits); // (n, n)
        let out = attn.dot(&v); // (n, d)
        out.dot(&self.w_out)
    }
}

// ---------------------------------------------------------------------------
// Trade Flow Encoder
// ---------------------------------------------------------------------------

/// Causal self-attention encoder for the recent-trade stream.
pub struct TradeFlowEncoder {
    pub d_model: usize,
    pub w_q: Array2<f64>,
    pub w_k: Array2<f64>,
    pub w_v: Array2<f64>,
    pub w_out: Array2<f64>,
}

impl TradeFlowEncoder {
    pub fn new(d_model: usize) -> Self {
        Self {
            d_model,
            w_q: random_matrix(d_model, d_model),
            w_k: random_matrix(d_model, d_model),
            w_v: random_matrix(d_model, d_model),
            w_out: random_matrix(d_model, d_model),
        }
    }

    /// Forward pass with causal masking.
    ///
    /// `x` has shape (num_trades, d_model).
    pub fn forward(&self, x: &Array2<f64>) -> Array2<f64> {
        let n = x.nrows();
        let dk = self.d_model as f64;

        let q = x.dot(&self.w_q);
        let k = x.dot(&self.w_k);
        let v = x.dot(&self.w_v);

        let mut logits = q.dot(&k.t()) / dk.sqrt();

        // Apply causal mask: position i can only attend to j <= i.
        for i in 0..n {
            for j in (i + 1)..n {
                logits[[i, j]] = f64::NEG_INFINITY;
            }
        }

        let attn = softmax_rows(&logits);
        let out = attn.dot(&v);
        out.dot(&self.w_out)
    }
}

// ---------------------------------------------------------------------------
// Fusion Transformer (Cross-Attention)
// ---------------------------------------------------------------------------

/// Bidirectional cross-attention fusion between LOB and trade encodings.
pub struct FusionTransformer {
    pub d_model: usize,
    // Trade queries LOB
    pub wq_t2l: Array2<f64>,
    pub wk_t2l: Array2<f64>,
    pub wv_t2l: Array2<f64>,
    // LOB queries Trade
    pub wq_l2t: Array2<f64>,
    pub wk_l2t: Array2<f64>,
    pub wv_l2t: Array2<f64>,
    // Projection from concatenated pools to d_model
    pub w_proj: Array2<f64>,
}

impl FusionTransformer {
    pub fn new(d_model: usize) -> Self {
        Self {
            d_model,
            wq_t2l: random_matrix(d_model, d_model),
            wk_t2l: random_matrix(d_model, d_model),
            wv_t2l: random_matrix(d_model, d_model),
            wq_l2t: random_matrix(d_model, d_model),
            wk_l2t: random_matrix(d_model, d_model),
            wv_l2t: random_matrix(d_model, d_model),
            w_proj: random_matrix(2 * d_model, d_model),
        }
    }

    /// Forward pass: returns a fused vector of size d_model.
    pub fn forward(&self, h_lob: &Array2<f64>, h_trade: &Array2<f64>) -> Array1<f64> {
        let dk = self.d_model as f64;

        // Cross-attention 1: trade queries LOB
        let q1 = h_trade.dot(&self.wq_t2l);
        let k1 = h_lob.dot(&self.wk_t2l);
        let v1 = h_lob.dot(&self.wv_t2l);
        let logits1 = q1.dot(&k1.t()) / dk.sqrt();
        let attn1 = softmax_rows(&logits1);
        let cross1 = attn1.dot(&v1); // (K, d)
        let pool1 = mean_pool(&cross1); // (d,)

        // Cross-attention 2: LOB queries trade
        let q2 = h_lob.dot(&self.wq_l2t);
        let k2 = h_trade.dot(&self.wk_l2t);
        let v2 = h_trade.dot(&self.wv_l2t);
        let logits2 = q2.dot(&k2.t()) / dk.sqrt();
        let attn2 = softmax_rows(&logits2);
        let cross2 = attn2.dot(&v2); // (2L, d)
        let pool2 = mean_pool(&cross2); // (d,)

        // Concatenate and project
        let mut concat = Array1::zeros(2 * self.d_model);
        concat.slice_mut(ndarray::s![..self.d_model]).assign(&pool1);
        concat
            .slice_mut(ndarray::s![self.d_model..])
            .assign(&pool2);

        // Project: (2d,) x (2d, d) -> (d,)
        concat.dot(&self.w_proj)
    }
}

// ---------------------------------------------------------------------------
// Prediction Head
// ---------------------------------------------------------------------------

/// Two-layer feedforward prediction head for 3-class price direction.
pub struct PredictionHead {
    pub w1: Array2<f64>,
    pub b1: Array1<f64>,
    pub w2: Array2<f64>,
    pub b2: Array1<f64>,
}

impl PredictionHead {
    /// `d_model` -- input dimension, `d_hidden` -- hidden layer size.
    pub fn new(d_model: usize, d_hidden: usize) -> Self {
        Self {
            w1: random_matrix(d_model, d_hidden),
            b1: random_vector(d_hidden),
            w2: random_matrix(d_hidden, 3),
            b2: random_vector(3),
        }
    }

    /// Returns softmax probabilities [P(up), P(down), P(stay)].
    pub fn forward(&self, x: &Array1<f64>) -> Array1<f64> {
        let hidden = relu(&(x.dot(&self.w1) + &self.b1));
        let logits = hidden.dot(&self.w2) + &self.b2;
        softmax_vec(&logits)
    }
}

// ---------------------------------------------------------------------------
// Full LIT Model
// ---------------------------------------------------------------------------

/// End-to-end LIT (Limit-order-book Informed Transformer) model.
pub struct LitModel {
    pub lob_encoder: LobEncoder,
    pub trade_encoder: TradeFlowEncoder,
    pub fusion: FusionTransformer,
    pub head: PredictionHead,
    pub d_model: usize,
}

impl LitModel {
    /// Create a new LIT model.
    ///
    /// * `d_model` -- embedding dimension
    /// * `max_levels` -- max LOB levels (bid+ask combined)
    /// * `d_hidden` -- hidden dimension in prediction head
    pub fn new(d_model: usize, max_levels: usize, d_hidden: usize) -> Self {
        Self {
            lob_encoder: LobEncoder::new(d_model, max_levels),
            trade_encoder: TradeFlowEncoder::new(d_model),
            fusion: FusionTransformer::new(d_model),
            head: PredictionHead::new(d_model, d_hidden),
            d_model,
        }
    }

    /// Run the full LIT pipeline.
    ///
    /// * `lob_features` -- (2L, d_model) level embeddings
    /// * `trade_features` -- (K, d_model) trade embeddings
    ///
    /// Returns [P(up), P(down), P(stay)].
    pub fn forward(
        &self,
        lob_features: &Array2<f64>,
        trade_features: &Array2<f64>,
    ) -> Array1<f64> {
        let h_lob = self.lob_encoder.forward(lob_features);
        let h_trade = self.trade_encoder.forward(trade_features);
        let fused = self.fusion.forward(&h_lob, &h_trade);
        self.head.forward(&fused)
    }
}

// ---------------------------------------------------------------------------
// Feature engineering helpers
// ---------------------------------------------------------------------------

/// Build LOB feature matrix from raw bid/ask levels.
///
/// Each level gets features: [price_distance_from_mid, log_volume, side_indicator, level_position].
/// Returns (2L, 4) matrix zero-padded to d_model via column extension.
pub fn build_lob_features(
    bids: &[(f64, f64)],
    asks: &[(f64, f64)],
    d_model: usize,
) -> Array2<f64> {
    let mid = if !bids.is_empty() && !asks.is_empty() {
        (bids[0].0 + asks[0].0) / 2.0
    } else {
        0.0
    };

    let n = bids.len() + asks.len();
    let mut features = Array2::zeros((n, d_model));

    for (i, (price, qty)) in bids.iter().enumerate() {
        let dist = (price - mid) / mid.max(1e-12);
        let log_vol = (qty + 1.0).ln();
        features[[i, 0]] = dist;
        features[[i, 1]] = log_vol;
        features[[i, 2]] = 0.0; // bid side
        if d_model > 3 {
            features[[i, 3]] = i as f64 / n.max(1) as f64;
        }
    }

    let offset = bids.len();
    for (i, (price, qty)) in asks.iter().enumerate() {
        let dist = (price - mid) / mid.max(1e-12);
        let log_vol = (qty + 1.0).ln();
        features[[offset + i, 0]] = dist;
        features[[offset + i, 1]] = log_vol;
        features[[offset + i, 2]] = 1.0; // ask side
        if d_model > 3 {
            features[[offset + i, 3]] = (offset + i) as f64 / n.max(1) as f64;
        }
    }

    features
}

/// Build trade feature matrix from raw trades.
///
/// Each trade gets features: [price_relative_to_mid, log_size, side, inter_arrival_enc].
pub fn build_trade_features(
    trades: &[TradeRecord],
    mid_price: f64,
    d_model: usize,
) -> Array2<f64> {
    let n = trades.len().max(1);
    let mut features = Array2::zeros((n, d_model));

    if trades.is_empty() {
        return features;
    }

    for (i, trade) in trades.iter().enumerate() {
        let price_rel = (trade.price - mid_price) / mid_price.max(1e-12);
        let log_size = (trade.qty + 1e-12).ln();
        let side_val = if trade.is_buyer_maker { 1.0 } else { -1.0 };

        features[[i, 0]] = price_rel;
        features[[i, 1]] = log_size;
        features[[i, 2]] = side_val;

        // Simple sinusoidal temporal encoding of inter-arrival time
        if i > 0 && d_model > 3 {
            let dt = (trade.timestamp - trades[i - 1].timestamp).max(0) as f64;
            let dt_sec = dt / 1000.0;
            features[[i, 3]] = (dt_sec * 0.1).sin();
            if d_model > 4 {
                features[[i, 4]] = (dt_sec * 0.1).cos();
            }
        }
    }

    features
}

// ---------------------------------------------------------------------------
// Data structures for market data
// ---------------------------------------------------------------------------

/// A single trade record.
#[derive(Debug, Clone)]
pub struct TradeRecord {
    pub price: f64,
    pub qty: f64,
    pub timestamp: i64,
    pub is_buyer_maker: bool,
}

// ---------------------------------------------------------------------------
// Bybit API Client
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct BybitResponse<T> {
    result: T,
}

#[derive(Debug, Deserialize)]
struct OrderbookResult {
    b: Vec<Vec<String>>, // bids: [[price, qty], ...]
    a: Vec<Vec<String>>, // asks: [[price, qty], ...]
}

#[derive(Debug, Deserialize)]
struct TradesResult {
    list: Vec<BybitTrade>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BybitTrade {
    price: String,
    size: String,
    time: String,
    side: String,
    #[allow(dead_code)]
    is_block_trade: bool,
}

/// Client for fetching market data from Bybit public API.
pub struct BybitClient {
    base_url: String,
    client: reqwest::blocking::Client,
}

impl BybitClient {
    /// Create a new client with default Bybit API base URL.
    pub fn new() -> Self {
        Self {
            base_url: "https://api.bybit.com".to_string(),
            client: reqwest::blocking::Client::new(),
        }
    }

    /// Create a client with a custom base URL (useful for testing).
    pub fn with_base_url(base_url: &str) -> Self {
        Self {
            base_url: base_url.to_string(),
            client: reqwest::blocking::Client::new(),
        }
    }

    /// Fetch orderbook snapshot. Returns (bids, asks) as vectors of (price, qty).
    pub fn fetch_orderbook(
        &self,
        symbol: &str,
        limit: u32,
    ) -> Result<(Vec<(f64, f64)>, Vec<(f64, f64)>)> {
        let url = format!(
            "{}/v5/market/orderbook?category=linear&symbol={}&limit={}",
            self.base_url, symbol, limit
        );
        let resp: BybitResponse<OrderbookResult> = self.client.get(&url).send()?.json()?;

        let bids: Vec<(f64, f64)> = resp
            .result
            .b
            .iter()
            .filter_map(|row| {
                if row.len() >= 2 {
                    Some((row[0].parse::<f64>().ok()?, row[1].parse::<f64>().ok()?))
                } else {
                    None
                }
            })
            .collect();

        let asks: Vec<(f64, f64)> = resp
            .result
            .a
            .iter()
            .filter_map(|row| {
                if row.len() >= 2 {
                    Some((row[0].parse::<f64>().ok()?, row[1].parse::<f64>().ok()?))
                } else {
                    None
                }
            })
            .collect();

        Ok((bids, asks))
    }

    /// Fetch recent trades.
    pub fn fetch_recent_trades(
        &self,
        symbol: &str,
        limit: u32,
    ) -> Result<Vec<TradeRecord>> {
        let url = format!(
            "{}/v5/market/recent-trade?category=linear&symbol={}&limit={}",
            self.base_url, symbol, limit
        );
        let resp: BybitResponse<TradesResult> = self.client.get(&url).send()?.json()?;

        let trades: Vec<TradeRecord> = resp
            .result
            .list
            .iter()
            .filter_map(|t| {
                Some(TradeRecord {
                    price: t.price.parse::<f64>().ok()?,
                    qty: t.size.parse::<f64>().ok()?,
                    timestamp: t.time.parse::<i64>().ok()?,
                    is_buyer_maker: t.side == "Sell",
                })
            })
            .collect();

        Ok(trades)
    }
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_softmax_rows() {
        let x = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 1.0, 1.0, 1.0]).unwrap();
        let s = softmax_rows(&x);
        // Each row should sum to 1.
        for row in s.rows() {
            let sum: f64 = row.sum();
            assert!((sum - 1.0).abs() < 1e-6, "Row sum = {}", sum);
        }
    }

    #[test]
    fn test_softmax_vec() {
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let s = softmax_vec(&x);
        assert!((s.sum() - 1.0).abs() < 1e-6);
        // Values should be monotonically increasing.
        assert!(s[0] < s[1]);
        assert!(s[1] < s[2]);
    }

    #[test]
    fn test_lob_encoder_output_shape() {
        let d_model = 8;
        let num_levels = 10; // 5 bid + 5 ask
        let encoder = LobEncoder::new(d_model, num_levels);
        let x = Array2::from_shape_fn((num_levels, d_model), |(i, j)| {
            (i as f64 + j as f64) * 0.1
        });
        let out = encoder.forward(&x);
        assert_eq!(out.shape(), &[num_levels, d_model]);
    }

    #[test]
    fn test_trade_encoder_causal() {
        let d_model = 8;
        let num_trades = 5;
        let encoder = TradeFlowEncoder::new(d_model);
        let x = Array2::from_shape_fn((num_trades, d_model), |(i, j)| {
            (i as f64 * 0.3 + j as f64 * 0.1).sin()
        });
        let out = encoder.forward(&x);
        assert_eq!(out.shape(), &[num_trades, d_model]);
    }

    #[test]
    fn test_fusion_output_shape() {
        let d_model = 8;
        let fusion = FusionTransformer::new(d_model);
        let h_lob = Array2::from_shape_fn((10, d_model), |(i, j)| (i + j) as f64 * 0.01);
        let h_trade = Array2::from_shape_fn((5, d_model), |(i, j)| (i + j) as f64 * 0.02);
        let fused = fusion.forward(&h_lob, &h_trade);
        assert_eq!(fused.len(), d_model);
    }

    #[test]
    fn test_prediction_head_probabilities() {
        let d_model = 8;
        let head = PredictionHead::new(d_model, 16);
        let x = Array1::from_shape_fn(d_model, |i| i as f64 * 0.1);
        let probs = head.forward(&x);
        assert_eq!(probs.len(), 3);
        assert!((probs.sum() - 1.0).abs() < 1e-6);
        for &p in probs.iter() {
            assert!(p >= 0.0 && p <= 1.0);
        }
    }

    #[test]
    fn test_full_lit_model() {
        let d_model = 8;
        let model = LitModel::new(d_model, 20, 16);
        let lob = Array2::from_shape_fn((10, d_model), |(i, j)| (i + j) as f64 * 0.01);
        let trades = Array2::from_shape_fn((5, d_model), |(i, j)| (i + j) as f64 * 0.02);
        let probs = model.forward(&lob, &trades);
        assert_eq!(probs.len(), 3);
        assert!((probs.sum() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_build_lob_features() {
        let bids = vec![(100.0, 5.0), (99.0, 10.0)];
        let asks = vec![(101.0, 3.0), (102.0, 7.0)];
        let features = build_lob_features(&bids, &asks, 8);
        assert_eq!(features.shape(), &[4, 8]);
        // Bid side indicator should be 0, ask should be 1.
        assert_eq!(features[[0, 2]], 0.0);
        assert_eq!(features[[2, 2]], 1.0);
    }

    #[test]
    fn test_build_trade_features() {
        let trades = vec![
            TradeRecord {
                price: 100.0,
                qty: 1.0,
                timestamp: 1000,
                is_buyer_maker: true,
            },
            TradeRecord {
                price: 100.5,
                qty: 2.0,
                timestamp: 1500,
                is_buyer_maker: false,
            },
        ];
        let features = build_trade_features(&trades, 100.0, 8);
        assert_eq!(features.shape(), &[2, 8]);
        // First trade: buyer_maker -> side = 1.0
        assert_eq!(features[[0, 2]], 1.0);
        // Second trade: not buyer_maker -> side = -1.0
        assert_eq!(features[[1, 2]], -1.0);
    }
}
