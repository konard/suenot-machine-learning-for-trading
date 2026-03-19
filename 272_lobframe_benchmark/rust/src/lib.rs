use ndarray::{Array1, Array2, Axis};
use rand::Rng;
use serde::Deserialize;

// =============================================================================
// LOB Data Structures
// =============================================================================

/// A single level in the order book (price + quantity).
#[derive(Debug, Clone)]
pub struct OrderBookLevel {
    pub price: f64,
    pub quantity: f64,
}

/// A snapshot of the limit order book.
#[derive(Debug, Clone)]
pub struct LobSnapshot {
    pub bids: Vec<OrderBookLevel>,
    pub asks: Vec<OrderBookLevel>,
    pub timestamp: u64,
}

impl LobSnapshot {
    /// Compute mid-price from best bid and ask.
    pub fn mid_price(&self) -> Option<f64> {
        if self.bids.is_empty() || self.asks.is_empty() {
            return None;
        }
        Some((self.bids[0].price + self.asks[0].price) / 2.0)
    }

    /// Compute bid-ask spread.
    pub fn spread(&self) -> Option<f64> {
        if self.bids.is_empty() || self.asks.is_empty() {
            return None;
        }
        Some(self.asks[0].price - self.bids[0].price)
    }

    /// Extract feature vector: [bid_prices, bid_vols, ask_prices, ask_vols] for `levels` levels.
    pub fn to_feature_vector(&self, levels: usize) -> Array1<f64> {
        let mut features = Vec::with_capacity(4 * levels);
        for i in 0..levels {
            if i < self.bids.len() {
                features.push(self.bids[i].price);
            } else {
                features.push(0.0);
            }
        }
        for i in 0..levels {
            if i < self.bids.len() {
                features.push(self.bids[i].quantity);
            } else {
                features.push(0.0);
            }
        }
        for i in 0..levels {
            if i < self.asks.len() {
                features.push(self.asks[i].price);
            } else {
                features.push(0.0);
            }
        }
        for i in 0..levels {
            if i < self.asks.len() {
                features.push(self.asks[i].quantity);
            } else {
                features.push(0.0);
            }
        }
        Array1::from(features)
    }

    /// Compute order imbalance at a given level.
    pub fn order_imbalance(&self, level: usize) -> Option<f64> {
        if level >= self.bids.len() || level >= self.asks.len() {
            return None;
        }
        let bid_vol = self.bids[level].quantity;
        let ask_vol = self.asks[level].quantity;
        let total = bid_vol + ask_vol;
        if total == 0.0 {
            return Some(0.0);
        }
        Some((bid_vol - ask_vol) / total)
    }
}

// =============================================================================
// LOB Normalizer (Z-Score per feature)
// =============================================================================

/// Z-score normalizer that computes statistics from training data.
#[derive(Debug, Clone)]
pub struct LobNormalizer {
    pub mean: Array1<f64>,
    pub std: Array1<f64>,
    pub clip_value: f64,
    pub fitted: bool,
}

impl LobNormalizer {
    pub fn new(clip_value: f64) -> Self {
        Self {
            mean: Array1::zeros(0),
            std: Array1::ones(0),
            clip_value,
            fitted: false,
        }
    }

    /// Fit normalizer on training data. Each row is a sample, each column a feature.
    pub fn fit(&mut self, data: &Array2<f64>) {
        let n = data.nrows() as f64;
        self.mean = data.sum_axis(Axis(0)) / n;

        let diff = data - &self.mean.broadcast(data.raw_dim()).unwrap();
        let sq = &diff * &diff;
        let variance = sq.sum_axis(Axis(0)) / n;
        self.std = variance.mapv(f64::sqrt);

        // Avoid division by zero
        self.std.mapv_inplace(|v| if v < 1e-10 { 1.0 } else { v });
        self.fitted = true;
    }

    /// Transform data using fitted statistics.
    pub fn transform(&self, data: &Array2<f64>) -> Array2<f64> {
        assert!(self.fitted, "Normalizer must be fitted before transform");
        let normalized = (data - &self.mean.broadcast(data.raw_dim()).unwrap())
            / &self.std.broadcast(data.raw_dim()).unwrap();
        normalized.mapv(|v| v.clamp(-self.clip_value, self.clip_value))
    }

    /// Fit and transform in one step.
    pub fn fit_transform(&mut self, data: &Array2<f64>) -> Array2<f64> {
        self.fit(data);
        self.transform(data)
    }
}

// =============================================================================
// Evaluation Metrics
// =============================================================================

/// Direction prediction label.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Down = 0,
    Flat = 1,
    Up = 2,
}

impl Direction {
    pub fn from_index(i: usize) -> Self {
        match i {
            0 => Direction::Down,
            1 => Direction::Flat,
            2 => Direction::Up,
            _ => panic!("Invalid direction index: {}", i),
        }
    }

    pub fn index(&self) -> usize {
        *self as usize
    }
}

/// Confusion matrix for 3-class classification.
#[derive(Debug, Clone)]
pub struct ConfusionMatrix {
    pub matrix: [[usize; 3]; 3],
}

impl ConfusionMatrix {
    pub fn new() -> Self {
        Self {
            matrix: [[0; 3]; 3],
        }
    }

    pub fn from_predictions(true_labels: &[Direction], pred_labels: &[Direction]) -> Self {
        assert_eq!(true_labels.len(), pred_labels.len());
        let mut cm = Self::new();
        for (t, p) in true_labels.iter().zip(pred_labels.iter()) {
            cm.matrix[t.index()][p.index()] += 1;
        }
        cm
    }

    pub fn total(&self) -> usize {
        self.matrix.iter().flat_map(|row| row.iter()).sum()
    }
}

impl Default for ConfusionMatrix {
    fn default() -> Self {
        Self::new()
    }
}

/// Evaluation metrics computed from a confusion matrix.
#[derive(Debug, Clone)]
pub struct EvaluationMetrics {
    pub accuracy: f64,
    pub f1_macro: f64,
    pub mcc: f64,
    pub per_class_f1: [f64; 3],
    pub per_class_precision: [f64; 3],
    pub per_class_recall: [f64; 3],
}

impl EvaluationMetrics {
    /// Compute all metrics from predictions.
    pub fn compute(true_labels: &[Direction], pred_labels: &[Direction]) -> Self {
        let cm = ConfusionMatrix::from_predictions(true_labels, pred_labels);
        Self::from_confusion_matrix(&cm)
    }

    /// Compute all metrics from a confusion matrix.
    pub fn from_confusion_matrix(cm: &ConfusionMatrix) -> Self {
        let total = cm.total() as f64;
        let m = &cm.matrix;

        // Accuracy
        let correct = (0..3).map(|i| m[i][i]).sum::<usize>() as f64;
        let accuracy = if total > 0.0 { correct / total } else { 0.0 };

        // Per-class precision, recall, F1
        let mut precision = [0.0f64; 3];
        let mut recall = [0.0f64; 3];
        let mut f1 = [0.0f64; 3];

        for c in 0..3 {
            let tp = m[c][c] as f64;
            let fp: f64 = (0..3).filter(|&i| i != c).map(|i| m[i][c] as f64).sum();
            let fn_: f64 = (0..3).filter(|&j| j != c).map(|j| m[c][j] as f64).sum();

            precision[c] = if tp + fp > 0.0 {
                tp / (tp + fp)
            } else {
                0.0
            };
            recall[c] = if tp + fn_ > 0.0 {
                tp / (tp + fn_)
            } else {
                0.0
            };
            f1[c] = if precision[c] + recall[c] > 0.0 {
                2.0 * precision[c] * recall[c] / (precision[c] + recall[c])
            } else {
                0.0
            };
        }

        let f1_macro = f1.iter().sum::<f64>() / 3.0;

        // MCC for multiclass
        let mcc = Self::compute_mcc(m);

        EvaluationMetrics {
            accuracy,
            f1_macro,
            mcc,
            per_class_f1: f1,
            per_class_precision: precision,
            per_class_recall: recall,
        }
    }

    /// Compute Matthews Correlation Coefficient for multiclass.
    fn compute_mcc(m: &[[usize; 3]; 3]) -> f64 {
        let k = 3;
        let n: f64 = m.iter().flat_map(|r| r.iter()).map(|&v| v as f64).sum();
        if n == 0.0 {
            return 0.0;
        }

        let mut cov_xy = 0.0;
        let mut cov_xz = 0.0;
        let mut cov_yz = 0.0;

        // Numerator: sum_k(C_kk * N - sum_l(C_kl) * sum_l(C_lk))
        // Using the standard formulation:
        // MCC = (N * sum_k(C_kk) - sum_k(sum_l(C_kl) * sum_l(C_lk)))
        //     / sqrt((N^2 - sum_k(sum_l(C_kl))^2) * (N^2 - sum_k(sum_l(C_lk))^2))

        let correct: f64 = (0..k).map(|i| m[i][i] as f64).sum();

        // Row sums and column sums
        let row_sums: Vec<f64> = (0..k).map(|i| (0..k).map(|j| m[i][j] as f64).sum()).collect();
        let col_sums: Vec<f64> = (0..k).map(|j| (0..k).map(|i| m[i][j] as f64).sum()).collect();

        cov_xy = n * correct - row_sums.iter().zip(col_sums.iter()).map(|(r, c)| r * c).sum::<f64>();

        let row_sq_sum: f64 = row_sums.iter().map(|r| r * r).sum();
        let col_sq_sum: f64 = col_sums.iter().map(|c| c * c).sum();

        cov_xz = (n * n - row_sq_sum).sqrt();
        cov_yz = (n * n - col_sq_sum).sqrt();

        if cov_xz * cov_yz == 0.0 {
            return 0.0;
        }

        cov_xy / (cov_xz * cov_yz)
    }
}

// =============================================================================
// Label Construction
// =============================================================================

/// Construct direction labels from mid-prices using a threshold.
pub fn construct_labels(mid_prices: &[f64], horizon: usize, threshold: f64) -> Vec<Direction> {
    let n = mid_prices.len();
    if n <= horizon {
        return vec![];
    }

    let mut labels = Vec::with_capacity(n - horizon);
    for i in 0..(n - horizon) {
        let current = mid_prices[i];
        if current == 0.0 {
            labels.push(Direction::Flat);
            continue;
        }
        // Smoothed future price
        let future_avg: f64 =
            mid_prices[i + 1..=i + horizon].iter().sum::<f64>() / horizon as f64;
        let change = (future_avg - current) / current;

        if change > threshold {
            labels.push(Direction::Up);
        } else if change < -threshold {
            labels.push(Direction::Down);
        } else {
            labels.push(Direction::Flat);
        }
    }
    labels
}

// =============================================================================
// Benchmark Model Trait
// =============================================================================

/// Trait for all benchmark models.
pub trait BenchmarkModel {
    /// Model name for reporting.
    fn name(&self) -> &str;

    /// Predict direction from a feature matrix (rows = time steps, cols = features).
    fn predict(&self, features: &Array2<f64>) -> Vec<Direction>;
}

// =============================================================================
// Baseline LSTM Model (Simplified)
// =============================================================================

/// Simplified LSTM-like model for benchmarking.
/// Uses learned-like weights to simulate LSTM behavior.
pub struct BaselineLstm {
    hidden_size: usize,
    weights: Array1<f64>,
    bias: [f64; 3],
}

impl BaselineLstm {
    pub fn new(input_size: usize, hidden_size: usize, seed: u64) -> Self {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let weights: Vec<f64> = (0..input_size)
            .map(|_| rng.gen_range(-0.1..0.1))
            .collect();
        let bias = [
            rng.gen_range(-0.05..0.05),
            rng.gen_range(-0.05..0.05),
            rng.gen_range(-0.05..0.05),
        ];
        Self {
            hidden_size,
            weights: Array1::from(weights),
            bias,
        }
    }
}

use rand::SeedableRng;

impl BenchmarkModel for BaselineLstm {
    fn name(&self) -> &str {
        "Baseline LSTM"
    }

    fn predict(&self, features: &Array2<f64>) -> Vec<Direction> {
        let n_samples = features.nrows();
        let mut predictions = Vec::with_capacity(n_samples);

        // Simplified: use weighted sum of last row features as "hidden state"
        // then apply linear layer to get 3-class logits
        let n_features = features.ncols().min(self.weights.len());

        for i in 0..n_samples {
            let row = features.row(i);
            let mut hidden = 0.0f64;
            for j in 0..n_features {
                hidden += row[j] * self.weights[j];
            }
            // Apply tanh activation (LSTM-like)
            hidden = hidden.tanh();

            // Simple linear projection to 3 classes
            let logits = [
                hidden * -1.0 + self.bias[0],
                hidden * 0.1 + self.bias[1],
                hidden * 1.0 + self.bias[2],
            ];

            let max_idx = logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(1);

            predictions.push(Direction::from_index(max_idx));
        }
        predictions
    }
}

// =============================================================================
// Baseline CNN Model (Simplified)
// =============================================================================

/// Simplified CNN-like model for benchmarking.
/// Uses convolution-like operations on feature windows.
pub struct BaselineCnn {
    kernel: Array1<f64>,
    bias: [f64; 3],
}

impl BaselineCnn {
    pub fn new(kernel_size: usize, seed: u64) -> Self {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let kernel: Vec<f64> = (0..kernel_size)
            .map(|_| rng.gen_range(-0.1..0.1))
            .collect();
        let bias = [
            rng.gen_range(-0.05..0.05),
            rng.gen_range(-0.05..0.05),
            rng.gen_range(-0.05..0.05),
        ];
        Self {
            kernel: Array1::from(kernel),
            bias,
        }
    }
}

impl BenchmarkModel for BaselineCnn {
    fn name(&self) -> &str {
        "Baseline CNN"
    }

    fn predict(&self, features: &Array2<f64>) -> Vec<Direction> {
        let n_samples = features.nrows();
        let n_features = features.ncols();
        let k_len = self.kernel.len().min(n_features);
        let mut predictions = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            let row = features.row(i);
            // Convolution-like: slide kernel across features
            let mut conv_sum = 0.0f64;
            let mut count = 0;
            for start in 0..=(n_features.saturating_sub(k_len)) {
                let mut local = 0.0;
                for j in 0..k_len {
                    local += row[start + j] * self.kernel[j];
                }
                conv_sum += local.max(0.0); // ReLU
                count += 1;
            }
            // Global average pooling
            let pooled = if count > 0 {
                conv_sum / count as f64
            } else {
                0.0
            };

            let logits = [
                pooled * -1.0 + self.bias[0],
                pooled * 0.1 + self.bias[1],
                pooled * 1.0 + self.bias[2],
            ];

            let max_idx = logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(1);

            predictions.push(Direction::from_index(max_idx));
        }
        predictions
    }
}

// =============================================================================
// Random Baseline Model
// =============================================================================

/// Random baseline that predicts uniformly at random.
pub struct RandomBaseline {
    seed: u64,
}

impl RandomBaseline {
    pub fn new(seed: u64) -> Self {
        Self { seed }
    }
}

impl BenchmarkModel for RandomBaseline {
    fn name(&self) -> &str {
        "Random Baseline"
    }

    fn predict(&self, features: &Array2<f64>) -> Vec<Direction> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(self.seed);
        (0..features.nrows())
            .map(|_| Direction::from_index(rng.gen_range(0..3)))
            .collect()
    }
}

// =============================================================================
// Benchmark Runner
// =============================================================================

/// Result of benchmarking a single model.
#[derive(Debug)]
pub struct BenchmarkResult {
    pub model_name: String,
    pub metrics: EvaluationMetrics,
    pub num_samples: usize,
}

/// Benchmark runner that evaluates multiple models on the same data.
pub struct BenchmarkRunner {
    pub normalizer: LobNormalizer,
    pub levels: usize,
    pub horizon: usize,
    pub threshold: f64,
}

impl BenchmarkRunner {
    pub fn new(levels: usize, horizon: usize, threshold: f64) -> Self {
        Self {
            normalizer: LobNormalizer::new(5.0),
            levels,
            horizon,
            threshold,
        }
    }

    /// Prepare feature matrix from LOB snapshots.
    pub fn prepare_features(&self, snapshots: &[LobSnapshot]) -> Array2<f64> {
        let n = snapshots.len();
        let feat_size = 4 * self.levels;
        let mut data = Array2::zeros((n, feat_size));
        for (i, snap) in snapshots.iter().enumerate() {
            let feats = snap.to_feature_vector(self.levels);
            for j in 0..feat_size.min(feats.len()) {
                data[[i, j]] = feats[j];
            }
        }
        data
    }

    /// Extract mid-prices from snapshots.
    pub fn extract_mid_prices(&self, snapshots: &[LobSnapshot]) -> Vec<f64> {
        snapshots
            .iter()
            .filter_map(|s| s.mid_price())
            .collect()
    }

    /// Run benchmark for a single model.
    pub fn evaluate(
        &mut self,
        model: &dyn BenchmarkModel,
        train_snapshots: &[LobSnapshot],
        test_snapshots: &[LobSnapshot],
    ) -> BenchmarkResult {
        // Prepare training features and fit normalizer
        let train_features = self.prepare_features(train_snapshots);
        self.normalizer.fit(&train_features);

        // Prepare test features and normalize
        let test_features = self.prepare_features(test_snapshots);
        let test_normalized = self.normalizer.transform(&test_features);

        // Construct labels from mid-prices
        let test_mid_prices = self.extract_mid_prices(test_snapshots);
        let true_labels = construct_labels(&test_mid_prices, self.horizon, self.threshold);

        // Truncate features to match labels
        let n_labels = true_labels.len();
        let test_input = test_normalized.slice(ndarray::s![..n_labels, ..]).to_owned();

        // Get predictions
        let predictions = model.predict(&test_input);

        // Compute metrics
        let metrics = EvaluationMetrics::compute(&true_labels, &predictions);

        BenchmarkResult {
            model_name: model.name().to_string(),
            metrics,
            num_samples: n_labels,
        }
    }

    /// Run benchmark for multiple models and print results.
    pub fn run_all(
        &mut self,
        models: &[&dyn BenchmarkModel],
        train_snapshots: &[LobSnapshot],
        test_snapshots: &[LobSnapshot],
    ) -> Vec<BenchmarkResult> {
        let mut results = Vec::new();
        for model in models {
            let result = self.evaluate(*model, train_snapshots, test_snapshots);
            results.push(result);
        }
        results
    }
}

/// Print benchmark results as a formatted table.
pub fn print_results_table(results: &[BenchmarkResult]) {
    println!("\n{:=<80}", "");
    println!("LOBFrame Benchmark Results");
    println!("{:=<80}", "");
    println!(
        "{:<20} {:>10} {:>10} {:>10} {:>10}",
        "Model", "Accuracy", "F1-Macro", "MCC", "Samples"
    );
    println!("{:-<60}", "");
    for r in results {
        println!(
            "{:<20} {:>10.4} {:>10.4} {:>10.4} {:>10}",
            r.model_name, r.metrics.accuracy, r.metrics.f1_macro, r.metrics.mcc, r.num_samples
        );
    }
    println!("{:=<80}", "");

    // Per-class breakdown
    println!("\nPer-Class F1 Scores:");
    println!(
        "{:<20} {:>10} {:>10} {:>10}",
        "Model", "Down", "Flat", "Up"
    );
    println!("{:-<50}", "");
    for r in results {
        println!(
            "{:<20} {:>10.4} {:>10.4} {:>10.4}",
            r.model_name,
            r.metrics.per_class_f1[0],
            r.metrics.per_class_f1[1],
            r.metrics.per_class_f1[2]
        );
    }
    println!();
}

// =============================================================================
// Bybit Orderbook Integration
// =============================================================================

#[derive(Debug, Deserialize)]
pub struct BybitResponse {
    #[serde(rename = "retCode")]
    pub ret_code: i32,
    #[serde(rename = "retMsg")]
    pub ret_msg: String,
    pub result: BybitOrderbookResult,
}

#[derive(Debug, Deserialize)]
pub struct BybitOrderbookResult {
    pub s: String,
    pub b: Vec<[String; 2]>,
    pub a: Vec<[String; 2]>,
    pub ts: u64,
    pub u: u64,
}

/// Fetch orderbook from Bybit API.
pub fn fetch_bybit_orderbook(symbol: &str, limit: usize) -> anyhow::Result<LobSnapshot> {
    let url = format!(
        "https://api.bybit.com/v5/market/orderbook?category=linear&symbol={}&limit={}",
        symbol, limit
    );

    let client = reqwest::blocking::Client::new();
    let resp: BybitResponse = client.get(&url).send()?.json()?;

    if resp.ret_code != 0 {
        anyhow::bail!("Bybit API error: {}", resp.ret_msg);
    }

    let bids: Vec<OrderBookLevel> = resp
        .result
        .b
        .iter()
        .map(|level| OrderBookLevel {
            price: level[0].parse().unwrap_or(0.0),
            quantity: level[1].parse().unwrap_or(0.0),
        })
        .collect();

    let asks: Vec<OrderBookLevel> = resp
        .result
        .a
        .iter()
        .map(|level| OrderBookLevel {
            price: level[0].parse().unwrap_or(0.0),
            quantity: level[1].parse().unwrap_or(0.0),
        })
        .collect();

    Ok(LobSnapshot {
        bids,
        asks,
        timestamp: resp.result.ts,
    })
}

/// Fetch orderbook asynchronously.
pub async fn fetch_bybit_orderbook_async(
    symbol: &str,
    limit: usize,
) -> anyhow::Result<LobSnapshot> {
    let url = format!(
        "https://api.bybit.com/v5/market/orderbook?category=linear&symbol={}&limit={}",
        symbol, limit
    );

    let client = reqwest::Client::new();
    let resp: BybitResponse = client.get(&url).send().await?.json().await?;

    if resp.ret_code != 0 {
        anyhow::bail!("Bybit API error: {}", resp.ret_msg);
    }

    let bids: Vec<OrderBookLevel> = resp
        .result
        .b
        .iter()
        .map(|level| OrderBookLevel {
            price: level[0].parse().unwrap_or(0.0),
            quantity: level[1].parse().unwrap_or(0.0),
        })
        .collect();

    let asks: Vec<OrderBookLevel> = resp
        .result
        .a
        .iter()
        .map(|level| OrderBookLevel {
            price: level[0].parse().unwrap_or(0.0),
            quantity: level[1].parse().unwrap_or(0.0),
        })
        .collect();

    Ok(LobSnapshot {
        bids,
        asks,
        timestamp: resp.result.ts,
    })
}

// =============================================================================
// Synthetic Data Generation (for testing)
// =============================================================================

/// Generate synthetic LOB snapshots for testing.
pub fn generate_synthetic_snapshots(
    n: usize,
    levels: usize,
    base_price: f64,
    seed: u64,
) -> Vec<LobSnapshot> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut snapshots = Vec::with_capacity(n);
    let mut price = base_price;

    for i in 0..n {
        // Random walk for mid-price
        price += rng.gen_range(-0.5..0.5);

        let spread = rng.gen_range(0.01..0.1);
        let best_bid = price - spread / 2.0;
        let best_ask = price + spread / 2.0;

        let mut bids = Vec::with_capacity(levels);
        let mut asks = Vec::with_capacity(levels);

        for l in 0..levels {
            let bid_price = best_bid - l as f64 * rng.gen_range(0.01..0.05);
            let ask_price = best_ask + l as f64 * rng.gen_range(0.01..0.05);
            let bid_vol = rng.gen_range(0.1..10.0);
            let ask_vol = rng.gen_range(0.1..10.0);
            bids.push(OrderBookLevel {
                price: bid_price,
                quantity: bid_vol,
            });
            asks.push(OrderBookLevel {
                price: ask_price,
                quantity: ask_vol,
            });
        }

        snapshots.push(LobSnapshot {
            bids,
            asks,
            timestamp: 1000000 + i as u64,
        });
    }
    snapshots
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lob_snapshot_mid_price() {
        let snap = LobSnapshot {
            bids: vec![OrderBookLevel {
                price: 99.0,
                quantity: 1.0,
            }],
            asks: vec![OrderBookLevel {
                price: 101.0,
                quantity: 1.0,
            }],
            timestamp: 0,
        };
        assert_eq!(snap.mid_price(), Some(100.0));
        assert_eq!(snap.spread(), Some(2.0));
    }

    #[test]
    fn test_lob_normalizer() {
        let data = Array2::from_shape_vec(
            (4, 2),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        )
        .unwrap();

        let mut normalizer = LobNormalizer::new(5.0);
        let normalized = normalizer.fit_transform(&data);

        // Mean should be [4.0, 5.0], std should be ~2.236
        assert!((normalized[[0, 0]] - (-1.3416)).abs() < 0.01);
        assert!(normalizer.fitted);
    }

    #[test]
    fn test_evaluation_metrics_perfect() {
        let true_labels = vec![Direction::Up, Direction::Down, Direction::Flat];
        let pred_labels = vec![Direction::Up, Direction::Down, Direction::Flat];

        let metrics = EvaluationMetrics::compute(&true_labels, &pred_labels);
        assert!((metrics.accuracy - 1.0).abs() < 1e-10);
        assert!((metrics.f1_macro - 1.0).abs() < 1e-10);
        assert!((metrics.mcc - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_evaluation_metrics_random() {
        let true_labels = vec![Direction::Up, Direction::Up, Direction::Up];
        let pred_labels = vec![Direction::Down, Direction::Down, Direction::Down];

        let metrics = EvaluationMetrics::compute(&true_labels, &pred_labels);
        assert!((metrics.accuracy - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_construct_labels() {
        let prices = vec![100.0, 100.5, 101.0, 101.5, 102.0, 100.0];
        let labels = construct_labels(&prices, 2, 0.005);

        assert!(!labels.is_empty());
        assert_eq!(labels.len(), 4); // n - horizon = 6 - 2 = 4
    }

    #[test]
    fn test_feature_vector() {
        let snap = LobSnapshot {
            bids: vec![
                OrderBookLevel {
                    price: 99.0,
                    quantity: 5.0,
                },
                OrderBookLevel {
                    price: 98.0,
                    quantity: 3.0,
                },
            ],
            asks: vec![
                OrderBookLevel {
                    price: 101.0,
                    quantity: 4.0,
                },
                OrderBookLevel {
                    price: 102.0,
                    quantity: 2.0,
                },
            ],
            timestamp: 0,
        };

        let features = snap.to_feature_vector(2);
        assert_eq!(features.len(), 8); // 4 * 2 levels
        assert_eq!(features[0], 99.0); // bid price 1
        assert_eq!(features[2], 5.0); // bid vol 1
        assert_eq!(features[4], 101.0); // ask price 1
        assert_eq!(features[6], 4.0); // ask vol 1
    }

    #[test]
    fn test_order_imbalance() {
        let snap = LobSnapshot {
            bids: vec![OrderBookLevel {
                price: 99.0,
                quantity: 8.0,
            }],
            asks: vec![OrderBookLevel {
                price: 101.0,
                quantity: 2.0,
            }],
            timestamp: 0,
        };

        let oi = snap.order_imbalance(0).unwrap();
        assert!((oi - 0.6).abs() < 1e-10); // (8-2)/(8+2) = 0.6
    }

    #[test]
    fn test_synthetic_data_generation() {
        let snapshots = generate_synthetic_snapshots(100, 5, 50000.0, 42);
        assert_eq!(snapshots.len(), 100);
        assert_eq!(snapshots[0].bids.len(), 5);
        assert_eq!(snapshots[0].asks.len(), 5);
        assert!(snapshots[0].mid_price().unwrap() > 0.0);
    }

    #[test]
    fn test_benchmark_runner() {
        let train = generate_synthetic_snapshots(50, 5, 50000.0, 42);
        let test = generate_synthetic_snapshots(30, 5, 50000.0, 99);

        let lstm = BaselineLstm::new(20, 64, 42);
        let cnn = BaselineCnn::new(3, 42);
        let random = RandomBaseline::new(42);

        let mut runner = BenchmarkRunner::new(5, 3, 0.0001);
        let results = runner.run_all(
            &[&lstm as &dyn BenchmarkModel, &cnn, &random],
            &train,
            &test,
        );

        assert_eq!(results.len(), 3);
        for r in &results {
            assert!(r.metrics.accuracy >= 0.0 && r.metrics.accuracy <= 1.0);
            assert!(r.metrics.f1_macro >= 0.0 && r.metrics.f1_macro <= 1.0);
        }
    }
}
