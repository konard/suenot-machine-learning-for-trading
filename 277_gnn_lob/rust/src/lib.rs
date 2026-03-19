//! # GNN LOB -- Graph Neural Network for Limit Order Book Trading
//!
//! This crate implements Graph Neural Networks (GCN and GAT) for modeling
//! limit order book data. Price levels are represented as graph nodes,
//! with edges connecting nearby levels for message passing.

use ndarray::{Array1, Array2, Axis};
use rand::Rng;
use serde::Deserialize;

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

/// A single price level in the order book.
#[derive(Debug, Clone)]
pub struct PriceLevel {
    pub price: f64,
    pub quantity: f64,
    pub side: Side,
}

/// Bid or Ask side.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Side {
    Bid,
    Ask,
}

/// A graph built from the limit order book.
#[derive(Debug, Clone)]
pub struct LOBGraph {
    /// Node feature matrix (num_nodes x num_features).
    pub features: Array2<f64>,
    /// Adjacency matrix (num_nodes x num_nodes).
    pub adjacency: Array2<f64>,
    /// Number of nodes.
    pub num_nodes: usize,
    /// Number of features per node.
    pub num_features: usize,
    /// Original price levels (for reference).
    pub levels: Vec<PriceLevel>,
}

impl LOBGraph {
    /// Build a graph from bid and ask price levels.
    ///
    /// Each level becomes a node with features: [normalized_price, quantity, side_flag, position].
    /// Edges connect each node to its `k` nearest neighbors by price.
    pub fn from_orderbook(bids: &[PriceLevel], asks: &[PriceLevel], k_neighbors: usize) -> Self {
        let mut levels: Vec<PriceLevel> = Vec::new();
        levels.extend_from_slice(bids);
        levels.extend_from_slice(asks);

        let num_nodes = levels.len();
        let num_features = 4; // normalized_price, quantity, side, position

        // Compute feature matrix
        let mut features = Array2::<f64>::zeros((num_nodes, num_features));
        let max_price = levels
            .iter()
            .map(|l| l.price)
            .fold(f64::NEG_INFINITY, f64::max);
        let min_price = levels
            .iter()
            .map(|l| l.price)
            .fold(f64::INFINITY, f64::min);
        let price_range = (max_price - min_price).max(1e-8);

        let max_qty = levels
            .iter()
            .map(|l| l.quantity)
            .fold(f64::NEG_INFINITY, f64::max)
            .max(1e-8);

        for (i, level) in levels.iter().enumerate() {
            features[[i, 0]] = (level.price - min_price) / price_range;
            features[[i, 1]] = level.quantity / max_qty;
            features[[i, 2]] = match level.side {
                Side::Bid => 0.0,
                Side::Ask => 1.0,
            };
            features[[i, 3]] = i as f64 / num_nodes.max(1) as f64;
        }

        // Build adjacency matrix using k-nearest neighbors by price
        let mut adjacency = Array2::<f64>::zeros((num_nodes, num_nodes));

        for i in 0..num_nodes {
            // Compute distances to all other nodes
            let mut distances: Vec<(usize, f64)> = (0..num_nodes)
                .filter(|&j| j != i)
                .map(|j| {
                    let dist = (levels[i].price - levels[j].price).abs();
                    (j, dist)
                })
                .collect();

            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            // Connect to k nearest neighbors
            let k = k_neighbors.min(distances.len());
            for &(j, _) in distances.iter().take(k) {
                adjacency[[i, j]] = 1.0;
                adjacency[[j, i]] = 1.0; // undirected
            }

            // Self-loop
            adjacency[[i, i]] = 1.0;
        }

        LOBGraph {
            features,
            adjacency,
            num_nodes,
            num_features,
            levels,
        }
    }

    /// Compute the degree matrix for GCN normalization.
    pub fn degree_matrix(&self) -> Array1<f64> {
        self.adjacency.sum_axis(Axis(1))
    }
}

// ---------------------------------------------------------------------------
// GCN Layer
// ---------------------------------------------------------------------------

/// Graph Convolutional Network layer.
///
/// Implements: H' = ReLU(D^{-1/2} A D^{-1/2} H W)
#[derive(Debug, Clone)]
pub struct GCNLayer {
    /// Weight matrix (input_dim x output_dim).
    pub weights: Array2<f64>,
    /// Bias vector (output_dim).
    pub bias: Array1<f64>,
    pub input_dim: usize,
    pub output_dim: usize,
}

impl GCNLayer {
    /// Create a new GCN layer with random initialization.
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (2.0 / (input_dim + output_dim) as f64).sqrt();

        let weights = Array2::from_shape_fn((input_dim, output_dim), |_| {
            rng.gen_range(-scale..scale)
        });
        let bias = Array1::zeros(output_dim);

        GCNLayer {
            weights,
            bias,
            input_dim,
            output_dim,
        }
    }

    /// Forward pass: H' = ReLU(A_norm H W + b)
    pub fn forward(&self, features: &Array2<f64>, adjacency: &Array2<f64>) -> Array2<f64> {
        let num_nodes = features.nrows();

        // Compute D^{-1/2}
        let degrees = adjacency.sum_axis(Axis(1));
        let mut d_inv_sqrt = Array1::<f64>::zeros(num_nodes);
        for i in 0..num_nodes {
            if degrees[i] > 0.0 {
                d_inv_sqrt[i] = 1.0 / degrees[i].sqrt();
            }
        }

        // Compute D^{-1/2} A D^{-1/2}
        let mut a_norm = Array2::<f64>::zeros((num_nodes, num_nodes));
        for i in 0..num_nodes {
            for j in 0..num_nodes {
                a_norm[[i, j]] = d_inv_sqrt[i] * adjacency[[i, j]] * d_inv_sqrt[j];
            }
        }

        // A_norm @ H
        let ah = a_norm.dot(features);

        // AH @ W + b
        let mut output = ah.dot(&self.weights);
        for i in 0..num_nodes {
            for j in 0..self.output_dim {
                output[[i, j]] += self.bias[j];
            }
        }

        // ReLU
        output.mapv(|x| x.max(0.0))
    }
}

// ---------------------------------------------------------------------------
// GAT Layer
// ---------------------------------------------------------------------------

/// Graph Attention Network layer.
///
/// Implements attention-weighted neighbor aggregation.
#[derive(Debug, Clone)]
pub struct GATLayer {
    /// Weight matrix W (input_dim x output_dim).
    pub weights: Array2<f64>,
    /// Attention vector a (2 * output_dim).
    pub attention: Array1<f64>,
    pub input_dim: usize,
    pub output_dim: usize,
    pub num_heads: usize,
    /// Multi-head weight matrices.
    pub head_weights: Vec<Array2<f64>>,
    /// Multi-head attention vectors.
    pub head_attentions: Vec<Array1<f64>>,
}

impl GATLayer {
    /// Create a new GAT layer with random initialization.
    pub fn new(input_dim: usize, output_dim: usize, num_heads: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (2.0 / (input_dim + output_dim) as f64).sqrt();

        let weights = Array2::from_shape_fn((input_dim, output_dim), |_| {
            rng.gen_range(-scale..scale)
        });
        let attention = Array1::from_shape_fn(2 * output_dim, |_| {
            rng.gen_range(-scale..scale)
        });

        let mut head_weights = Vec::with_capacity(num_heads);
        let mut head_attentions = Vec::with_capacity(num_heads);
        for _ in 0..num_heads {
            head_weights.push(Array2::from_shape_fn((input_dim, output_dim), |_| {
                rng.gen_range(-scale..scale)
            }));
            head_attentions.push(Array1::from_shape_fn(2 * output_dim, |_| {
                rng.gen_range(-scale..scale)
            }));
        }

        GATLayer {
            weights,
            attention,
            input_dim,
            output_dim,
            num_heads,
            head_weights,
            head_attentions,
        }
    }

    /// Compute attention coefficients between node i and node j.
    fn attention_coefficient(
        &self,
        wh_i: &Array1<f64>,
        wh_j: &Array1<f64>,
        attn: &Array1<f64>,
    ) -> f64 {
        // e_ij = LeakyReLU(a^T [Wh_i || Wh_j])
        let mut e = 0.0;
        let dim = wh_i.len();
        for k in 0..dim {
            e += attn[k] * wh_i[k];
        }
        for k in 0..dim {
            e += attn[dim + k] * wh_j[k];
        }
        // LeakyReLU with negative slope 0.2
        if e > 0.0 { e } else { 0.2 * e }
    }

    /// Forward pass for a single attention head.
    fn forward_single_head(
        &self,
        features: &Array2<f64>,
        adjacency: &Array2<f64>,
        w: &Array2<f64>,
        attn: &Array1<f64>,
    ) -> Array2<f64> {
        let num_nodes = features.nrows();

        // Compute Wh for all nodes
        let wh = features.dot(w); // (num_nodes x output_dim)

        // Compute attention coefficients
        let mut attention_scores = Array2::<f64>::zeros((num_nodes, num_nodes));
        for i in 0..num_nodes {
            for j in 0..num_nodes {
                if adjacency[[i, j]] > 0.0 {
                    let wh_i = wh.row(i).to_owned();
                    let wh_j = wh.row(j).to_owned();
                    attention_scores[[i, j]] = self.attention_coefficient(&wh_i, &wh_j, attn);
                } else {
                    attention_scores[[i, j]] = f64::NEG_INFINITY;
                }
            }
        }

        // Softmax per row
        let mut alpha = Array2::<f64>::zeros((num_nodes, num_nodes));
        for i in 0..num_nodes {
            let max_val = attention_scores
                .row(i)
                .iter()
                .cloned()
                .filter(|v| v.is_finite())
                .fold(f64::NEG_INFINITY, f64::max);

            let mut sum_exp = 0.0;
            for j in 0..num_nodes {
                if attention_scores[[i, j]].is_finite() {
                    let exp_val = (attention_scores[[i, j]] - max_val).exp();
                    alpha[[i, j]] = exp_val;
                    sum_exp += exp_val;
                }
            }
            if sum_exp > 0.0 {
                for j in 0..num_nodes {
                    alpha[[i, j]] /= sum_exp;
                }
            }
        }

        // Weighted aggregation: h'_i = sum_j alpha_ij * Wh_j
        let output = alpha.dot(&wh);

        // ELU activation
        output.mapv(|x| if x > 0.0 { x } else { x.exp() - 1.0 })
    }

    /// Forward pass with multi-head attention.
    /// Output is concatenation of all heads: (num_nodes x (num_heads * output_dim)).
    pub fn forward(&self, features: &Array2<f64>, adjacency: &Array2<f64>) -> Array2<f64> {
        let num_nodes = features.nrows();
        let total_dim = self.num_heads * self.output_dim;
        let mut output = Array2::<f64>::zeros((num_nodes, total_dim));

        for (head_idx, (w, attn)) in self
            .head_weights
            .iter()
            .zip(self.head_attentions.iter())
            .enumerate()
        {
            let head_out = self.forward_single_head(features, adjacency, w, attn);
            let start_col = head_idx * self.output_dim;
            for i in 0..num_nodes {
                for j in 0..self.output_dim {
                    output[[i, start_col + j]] = head_out[[i, j]];
                }
            }
        }

        output
    }

    /// Forward pass with averaging across heads (for final layer).
    pub fn forward_mean(&self, features: &Array2<f64>, adjacency: &Array2<f64>) -> Array2<f64> {
        let num_nodes = features.nrows();
        let mut output = Array2::<f64>::zeros((num_nodes, self.output_dim));

        for (w, attn) in self.head_weights.iter().zip(self.head_attentions.iter()) {
            let head_out = self.forward_single_head(features, adjacency, w, attn);
            output = output + &head_out;
        }

        output / self.num_heads as f64
    }
}

// ---------------------------------------------------------------------------
// Graph Readout
// ---------------------------------------------------------------------------

/// Graph-level readout (pooling) methods.
#[derive(Debug, Clone, Copy)]
pub enum ReadoutMethod {
    Mean,
    Max,
}

/// Perform graph-level readout to get a fixed-size graph embedding.
pub fn graph_readout(node_embeddings: &Array2<f64>, method: ReadoutMethod) -> Array1<f64> {
    match method {
        ReadoutMethod::Mean => node_embeddings.mean_axis(Axis(0)).unwrap(),
        ReadoutMethod::Max => {
            let num_features = node_embeddings.ncols();
            let mut result = Array1::<f64>::from_elem(num_features, f64::NEG_INFINITY);
            for row in node_embeddings.rows() {
                for (j, &val) in row.iter().enumerate() {
                    if val > result[j] {
                        result[j] = val;
                    }
                }
            }
            result
        }
    }
}

// ---------------------------------------------------------------------------
// MLP Prediction Head
// ---------------------------------------------------------------------------

/// Simple MLP for mid-price direction prediction.
#[derive(Debug, Clone)]
pub struct MLPHead {
    pub w1: Array2<f64>,
    pub b1: Array1<f64>,
    pub w2: Array2<f64>,
    pub b2: Array1<f64>,
}

impl MLPHead {
    /// Create a new MLP head: input_dim -> hidden_dim -> 3 (up/down/stationary).
    pub fn new(input_dim: usize, hidden_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale1 = (2.0 / (input_dim + hidden_dim) as f64).sqrt();
        let scale2 = (2.0 / (hidden_dim + 3) as f64).sqrt();

        let w1 = Array2::from_shape_fn((input_dim, hidden_dim), |_| {
            rng.gen_range(-scale1..scale1)
        });
        let b1 = Array1::zeros(hidden_dim);

        let w2 = Array2::from_shape_fn((hidden_dim, 3), |_| rng.gen_range(-scale2..scale2));
        let b2 = Array1::zeros(3);

        MLPHead { w1, b1, w2, b2 }
    }

    /// Forward pass: returns softmax probabilities [p_down, p_stationary, p_up].
    pub fn forward(&self, x: &Array1<f64>) -> Array1<f64> {
        // Hidden layer with ReLU
        let h = x.dot(&self.w1) + &self.b1;
        let h = h.mapv(|v| v.max(0.0));

        // Output layer
        let logits = h.dot(&self.w2) + &self.b2;

        // Softmax
        softmax(&logits)
    }
}

/// Compute softmax of a 1-D array.
pub fn softmax(x: &Array1<f64>) -> Array1<f64> {
    let max_val = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_vals = x.mapv(|v| (v - max_val).exp());
    let sum_exp = exp_vals.sum();
    exp_vals / sum_exp
}

// ---------------------------------------------------------------------------
// Mid-Price Predictor (full pipeline)
// ---------------------------------------------------------------------------

/// Full GNN pipeline for mid-price direction prediction.
#[derive(Debug)]
pub struct MidPricePredictor {
    pub gcn1: GCNLayer,
    pub gcn2: GCNLayer,
    pub mlp: MLPHead,
    pub readout: ReadoutMethod,
}

impl MidPricePredictor {
    /// Create a new predictor with default architecture.
    pub fn new(input_features: usize, hidden_dim: usize) -> Self {
        let gcn1 = GCNLayer::new(input_features, hidden_dim);
        let gcn2 = GCNLayer::new(hidden_dim, hidden_dim);
        let mlp = MLPHead::new(hidden_dim, hidden_dim / 2);

        MidPricePredictor {
            gcn1,
            gcn2,
            mlp,
            readout: ReadoutMethod::Mean,
        }
    }

    /// Predict mid-price direction from an LOB graph.
    /// Returns [p_down, p_stationary, p_up].
    pub fn predict(&self, graph: &LOBGraph) -> Array1<f64> {
        // Layer 1
        let h1 = self.gcn1.forward(&graph.features, &graph.adjacency);
        // Layer 2
        let h2 = self.gcn2.forward(&h1, &graph.adjacency);
        // Readout
        let graph_emb = graph_readout(&h2, self.readout);
        // Prediction
        self.mlp.forward(&graph_emb)
    }

    /// Get the predicted direction as a string.
    pub fn predict_direction(&self, graph: &LOBGraph) -> (&str, f64) {
        let probs = self.predict(graph);
        let labels = ["DOWN", "STATIONARY", "UP"];
        let mut best_idx = 0;
        let mut best_prob = probs[0];
        for i in 1..3 {
            if probs[i] > best_prob {
                best_prob = probs[i];
                best_idx = i;
            }
        }
        (labels[best_idx], best_prob)
    }
}

// ---------------------------------------------------------------------------
// Bybit API Client
// ---------------------------------------------------------------------------

/// Response from Bybit orderbook endpoint.
#[derive(Debug, Deserialize)]
pub struct BybitResponse {
    #[serde(rename = "retCode")]
    pub ret_code: i32,
    #[serde(rename = "retMsg")]
    pub ret_msg: String,
    pub result: BybitOrderbook,
}

#[derive(Debug, Deserialize)]
pub struct BybitOrderbook {
    /// Bids: [[price, qty], ...]
    pub b: Vec<Vec<String>>,
    /// Asks: [[price, qty], ...]
    pub a: Vec<Vec<String>>,
    /// Timestamp
    pub ts: Option<u64>,
    /// Update ID
    pub u: Option<u64>,
    /// Symbol
    pub s: Option<String>,
}

/// Client for fetching orderbook data from Bybit.
pub struct BybitClient {
    pub base_url: String,
    pub client: reqwest::blocking::Client,
}

impl BybitClient {
    /// Create a new Bybit client.
    pub fn new() -> Self {
        BybitClient {
            base_url: "https://api.bybit.com".to_string(),
            client: reqwest::blocking::Client::new(),
        }
    }

    /// Fetch orderbook for a given symbol.
    /// Returns (bids, asks) as vectors of PriceLevel.
    pub fn fetch_orderbook(
        &self,
        symbol: &str,
        limit: u32,
    ) -> anyhow::Result<(Vec<PriceLevel>, Vec<PriceLevel>)> {
        let url = format!(
            "{}/v5/market/orderbook?category=spot&symbol={}&limit={}",
            self.base_url, symbol, limit
        );

        let resp: BybitResponse = self.client.get(&url).send()?.json()?;

        if resp.ret_code != 0 {
            anyhow::bail!("Bybit API error: {}", resp.ret_msg);
        }

        let bids: Vec<PriceLevel> = resp
            .result
            .b
            .iter()
            .filter_map(|entry| {
                if entry.len() >= 2 {
                    Some(PriceLevel {
                        price: entry[0].parse().ok()?,
                        quantity: entry[1].parse().ok()?,
                        side: Side::Bid,
                    })
                } else {
                    None
                }
            })
            .collect();

        let asks: Vec<PriceLevel> = resp
            .result
            .a
            .iter()
            .filter_map(|entry| {
                if entry.len() >= 2 {
                    Some(PriceLevel {
                        price: entry[0].parse().ok()?,
                        quantity: entry[1].parse().ok()?,
                        side: Side::Ask,
                    })
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_bids() -> Vec<PriceLevel> {
        vec![
            PriceLevel { price: 50000.0, quantity: 1.5, side: Side::Bid },
            PriceLevel { price: 49990.0, quantity: 2.0, side: Side::Bid },
            PriceLevel { price: 49980.0, quantity: 0.8, side: Side::Bid },
            PriceLevel { price: 49970.0, quantity: 3.0, side: Side::Bid },
            PriceLevel { price: 49960.0, quantity: 1.2, side: Side::Bid },
        ]
    }

    fn sample_asks() -> Vec<PriceLevel> {
        vec![
            PriceLevel { price: 50010.0, quantity: 1.0, side: Side::Ask },
            PriceLevel { price: 50020.0, quantity: 2.5, side: Side::Ask },
            PriceLevel { price: 50030.0, quantity: 0.5, side: Side::Ask },
            PriceLevel { price: 50040.0, quantity: 1.8, side: Side::Ask },
            PriceLevel { price: 50050.0, quantity: 2.2, side: Side::Ask },
        ]
    }

    #[test]
    fn test_lob_graph_construction() {
        let bids = sample_bids();
        let asks = sample_asks();
        let graph = LOBGraph::from_orderbook(&bids, &asks, 3);

        assert_eq!(graph.num_nodes, 10);
        assert_eq!(graph.num_features, 4);
        assert_eq!(graph.features.nrows(), 10);
        assert_eq!(graph.features.ncols(), 4);

        // All features should be in [0, 1]
        for val in graph.features.iter() {
            assert!(*val >= 0.0 && *val <= 1.0, "Feature out of range: {}", val);
        }

        // Adjacency should be symmetric with self-loops
        for i in 0..graph.num_nodes {
            assert_eq!(graph.adjacency[[i, i]], 1.0, "Missing self-loop at node {}", i);
            for j in 0..graph.num_nodes {
                assert_eq!(
                    graph.adjacency[[i, j]],
                    graph.adjacency[[j, i]],
                    "Adjacency not symmetric at ({}, {})",
                    i, j
                );
            }
        }
    }

    #[test]
    fn test_gcn_layer_forward() {
        let bids = sample_bids();
        let asks = sample_asks();
        let graph = LOBGraph::from_orderbook(&bids, &asks, 3);

        let gcn = GCNLayer::new(4, 8);
        let output = gcn.forward(&graph.features, &graph.adjacency);

        assert_eq!(output.nrows(), 10);
        assert_eq!(output.ncols(), 8);

        // After ReLU, all values should be >= 0
        for val in output.iter() {
            assert!(*val >= 0.0, "GCN output should be non-negative after ReLU");
        }
    }

    #[test]
    fn test_gat_layer_forward() {
        let bids = sample_bids();
        let asks = sample_asks();
        let graph = LOBGraph::from_orderbook(&bids, &asks, 3);

        let num_heads = 2;
        let output_dim = 4;
        let gat = GATLayer::new(4, output_dim, num_heads);

        // Test concatenated multi-head output
        let output = gat.forward(&graph.features, &graph.adjacency);
        assert_eq!(output.nrows(), 10);
        assert_eq!(output.ncols(), num_heads * output_dim);

        // Test mean multi-head output
        let output_mean = gat.forward_mean(&graph.features, &graph.adjacency);
        assert_eq!(output_mean.nrows(), 10);
        assert_eq!(output_mean.ncols(), output_dim);
    }

    #[test]
    fn test_graph_readout() {
        let embeddings = Array2::from_shape_vec(
            (3, 2),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        )
        .unwrap();

        let mean = graph_readout(&embeddings, ReadoutMethod::Mean);
        assert!((mean[0] - 3.0).abs() < 1e-6);
        assert!((mean[1] - 4.0).abs() < 1e-6);

        let max = graph_readout(&embeddings, ReadoutMethod::Max);
        assert!((max[0] - 5.0).abs() < 1e-6);
        assert!((max[1] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_softmax() {
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let s = softmax(&x);

        // Softmax should sum to 1
        let sum: f64 = s.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "Softmax should sum to 1, got {}", sum);

        // Values should be in (0, 1) and monotonically increasing
        assert!(s[0] < s[1] && s[1] < s[2]);
        for val in s.iter() {
            assert!(*val > 0.0 && *val < 1.0);
        }
    }

    #[test]
    fn test_mid_price_predictor() {
        let bids = sample_bids();
        let asks = sample_asks();
        let graph = LOBGraph::from_orderbook(&bids, &asks, 3);

        let predictor = MidPricePredictor::new(4, 8);
        let probs = predictor.predict(&graph);

        assert_eq!(probs.len(), 3);

        // Probabilities should sum to 1
        let sum: f64 = probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "Probabilities should sum to 1, got {}",
            sum
        );

        // All probabilities should be positive
        for val in probs.iter() {
            assert!(*val > 0.0, "Probability should be positive");
        }

        // predict_direction should return a valid label
        let (direction, confidence) = predictor.predict_direction(&graph);
        assert!(
            direction == "UP" || direction == "DOWN" || direction == "STATIONARY",
            "Invalid direction: {}",
            direction
        );
        assert!(confidence > 0.0 && confidence <= 1.0);
    }

    #[test]
    fn test_mlp_head() {
        let mlp = MLPHead::new(8, 4);
        let input = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]);
        let output = mlp.forward(&input);

        assert_eq!(output.len(), 3);
        let sum: f64 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_degree_matrix() {
        let bids = sample_bids();
        let asks = sample_asks();
        let graph = LOBGraph::from_orderbook(&bids, &asks, 3);
        let degrees = graph.degree_matrix();

        assert_eq!(degrees.len(), 10);
        // Each node has at least a self-loop (degree >= 1)
        for &d in degrees.iter() {
            assert!(d >= 1.0, "Degree should be at least 1 (self-loop)");
        }
    }
}
