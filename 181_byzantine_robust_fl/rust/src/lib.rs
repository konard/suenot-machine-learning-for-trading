use anyhow::{Context, Result};
use ndarray::Array1;
use rand::Rng;
use serde::Deserialize;

// ─── Aggregation Strategy ────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AggregationMethod {
    FedAvg,
    Krum,
    TrimmedMean { beta: usize },
    CoordinateMedian,
}

impl std::fmt::Display for AggregationMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FedAvg => write!(f, "FedAvg"),
            Self::Krum => write!(f, "Krum"),
            Self::TrimmedMean { beta } => write!(f, "TrimmedMean(beta={})", beta),
            Self::CoordinateMedian => write!(f, "CoordinateMedian"),
        }
    }
}

// ─── Client Types ────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ClientType {
    Honest,
    Byzantine,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AttackStrategy {
    /// Add large random noise to the true gradient
    RandomNoise { scale: f64 },
    /// Scale the true gradient by a large negative factor
    SignFlip { scale: f64 },
    /// Replace gradient with a fixed adversarial direction
    FixedDirection { scale: f64 },
}

// ─── Aggregation Functions ───────────────────────────────────────────

/// Simple federated averaging -- vulnerable to Byzantine attacks.
pub fn fed_avg(gradients: &[Array1<f64>]) -> Array1<f64> {
    let n = gradients.len() as f64;
    let dim = gradients[0].len();
    let mut result = Array1::zeros(dim);
    for g in gradients {
        result = result + g;
    }
    result / n
}

/// Krum aggregation: selects the gradient closest to its neighbors.
///
/// For each gradient, computes the sum of squared distances to its
/// (n - f - 2) nearest neighbors and selects the gradient with the
/// minimum score.
pub fn krum(gradients: &[Array1<f64>], num_byzantine: usize) -> Array1<f64> {
    let n = gradients.len();
    assert!(
        n >= 2 * num_byzantine + 3,
        "Krum requires n >= 2f + 3, got n={}, f={}",
        n,
        num_byzantine
    );

    let num_neighbors = n - num_byzantine - 2;

    // Compute pairwise squared distances
    let mut distances = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let diff = &gradients[i] - &gradients[j];
            let dist_sq = diff.dot(&diff);
            distances[i][j] = dist_sq;
            distances[j][i] = dist_sq;
        }
    }

    // For each gradient, compute sum of distances to nearest neighbors
    let mut scores = vec![0.0f64; n];
    for i in 0..n {
        let mut dists: Vec<f64> = (0..n)
            .filter(|&j| j != i)
            .map(|j| distances[i][j])
            .collect();
        dists.sort_by(|a, b| a.partial_cmp(b).unwrap());
        scores[i] = dists[..num_neighbors].iter().sum();
    }

    // Select gradient with minimum score
    let best_idx = scores
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap()
        .0;

    gradients[best_idx].clone()
}

/// Coordinate-wise trimmed mean.
///
/// For each coordinate, sorts all values, removes the top and bottom
/// `beta` values, and averages the rest.
pub fn trimmed_mean(gradients: &[Array1<f64>], beta: usize) -> Array1<f64> {
    let n = gradients.len();
    assert!(
        2 * beta < n,
        "Trimmed mean requires 2*beta < n, got beta={}, n={}",
        beta,
        n
    );

    let dim = gradients[0].len();
    let mut result = Array1::zeros(dim);

    for j in 0..dim {
        let mut values: Vec<f64> = gradients.iter().map(|g| g[j]).collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Trim beta from each end and average
        let trimmed = &values[beta..n - beta];
        let sum: f64 = trimmed.iter().sum();
        result[j] = sum / trimmed.len() as f64;
    }

    result
}

/// Coordinate-wise median aggregation.
///
/// For each coordinate, takes the median of all gradient values.
pub fn coordinate_median(gradients: &[Array1<f64>]) -> Array1<f64> {
    let n = gradients.len();
    let dim = gradients[0].len();
    let mut result = Array1::zeros(dim);

    for j in 0..dim {
        let mut values: Vec<f64> = gradients.iter().map(|g| g[j]).collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        result[j] = if n % 2 == 0 {
            (values[n / 2 - 1] + values[n / 2]) / 2.0
        } else {
            values[n / 2]
        };
    }

    result
}

/// Dispatch to the appropriate aggregation method.
pub fn aggregate(
    gradients: &[Array1<f64>],
    method: AggregationMethod,
    num_byzantine: usize,
) -> Array1<f64> {
    match method {
        AggregationMethod::FedAvg => fed_avg(gradients),
        AggregationMethod::Krum => krum(gradients, num_byzantine),
        AggregationMethod::TrimmedMean { beta } => trimmed_mean(gradients, beta),
        AggregationMethod::CoordinateMedian => coordinate_median(gradients),
    }
}

// ─── Gradient Generation ─────────────────────────────────────────────

/// Generate an honest gradient update: true gradient + small Gaussian noise.
pub fn generate_honest_gradient(
    true_gradient: &Array1<f64>,
    noise_std: f64,
    rng: &mut impl Rng,
) -> Array1<f64> {
    let dim = true_gradient.len();
    let noise: Array1<f64> = Array1::from_shape_fn(dim, |_| {
        // Box-Muller transform for Gaussian noise
        let u1: f64 = rng.gen_range(0.0001..1.0);
        let u2: f64 = rng.gen();
        noise_std * (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    });
    true_gradient + &noise
}

/// Generate a Byzantine (malicious) gradient update.
pub fn generate_byzantine_gradient(
    true_gradient: &Array1<f64>,
    attack: AttackStrategy,
    rng: &mut impl Rng,
) -> Array1<f64> {
    let dim = true_gradient.len();
    match attack {
        AttackStrategy::RandomNoise { scale } => {
            Array1::from_shape_fn(dim, |_| {
                let u1: f64 = rng.gen_range(0.0001..1.0);
                let u2: f64 = rng.gen();
                scale * (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
            })
        }
        AttackStrategy::SignFlip { scale } => true_gradient * (-scale),
        AttackStrategy::FixedDirection { scale } => Array1::from_elem(dim, scale),
    }
}

// ─── Similarity Metrics ──────────────────────────────────────────────

/// Cosine similarity between two vectors.
pub fn cosine_similarity(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    let dot = a.dot(b);
    let norm_a = a.dot(a).sqrt();
    let norm_b = b.dot(b).sqrt();
    if norm_a < 1e-12 || norm_b < 1e-12 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

/// Euclidean distance between two vectors.
pub fn euclidean_distance(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    let diff = a - b;
    diff.dot(&diff).sqrt()
}

// ─── Federated Learning Simulation ───────────────────────────────────

/// Result of a single FL simulation round.
#[derive(Debug, Clone)]
pub struct SimulationResult {
    pub method: String,
    pub cosine_sim_to_true: f64,
    pub euclidean_dist_to_true: f64,
}

/// Run a single round of federated learning simulation.
///
/// Generates honest and byzantine gradients, aggregates them using the
/// specified method, and measures the quality of the result.
pub fn simulate_fl_round(
    true_gradient: &Array1<f64>,
    num_honest: usize,
    num_byzantine: usize,
    honest_noise_std: f64,
    attack: AttackStrategy,
    method: AggregationMethod,
    rng: &mut impl Rng,
) -> SimulationResult {
    let mut gradients = Vec::with_capacity(num_honest + num_byzantine);

    // Generate honest gradients
    for _ in 0..num_honest {
        gradients.push(generate_honest_gradient(true_gradient, honest_noise_std, rng));
    }

    // Generate byzantine gradients
    for _ in 0..num_byzantine {
        gradients.push(generate_byzantine_gradient(true_gradient, attack, rng));
    }

    // Aggregate
    let aggregated = aggregate(&gradients, method, num_byzantine);

    // Measure quality
    let cos_sim = cosine_similarity(&aggregated, true_gradient);
    let euc_dist = euclidean_distance(&aggregated, true_gradient);

    SimulationResult {
        method: method.to_string(),
        cosine_sim_to_true: cos_sim,
        euclidean_dist_to_true: euc_dist,
    }
}

// ─── Bybit API Integration ──────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct BybitResponse {
    #[serde(rename = "retCode")]
    pub ret_code: i32,
    #[serde(rename = "retMsg")]
    pub ret_msg: String,
    pub result: BybitResult,
}

#[derive(Debug, Deserialize)]
pub struct BybitResult {
    pub symbol: Option<String>,
    pub category: Option<String>,
    pub list: Vec<Vec<String>>,
}

/// A single OHLCV candle.
#[derive(Debug, Clone)]
pub struct Candle {
    pub timestamp: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// Client for fetching data from the Bybit V5 API.
pub struct BybitClient {
    base_url: String,
}

impl BybitClient {
    pub fn new() -> Self {
        Self {
            base_url: "https://api.bybit.com".to_string(),
        }
    }

    /// Fetch kline (candlestick) data from Bybit.
    ///
    /// # Arguments
    /// * `symbol` - Trading pair, e.g., "BTCUSDT"
    /// * `interval` - Candle interval, e.g., "60" for 1 hour
    /// * `limit` - Number of candles to fetch (max 200)
    pub fn fetch_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> Result<Vec<Candle>> {
        let url = format!(
            "{}/v5/market/klines?category=linear&symbol={}&interval={}&limit={}",
            self.base_url, symbol, interval, limit
        );

        let response: BybitResponse = reqwest::blocking::get(&url)
            .context("Failed to fetch from Bybit API")?
            .json()
            .context("Failed to parse Bybit response")?;

        if response.ret_code != 0 {
            anyhow::bail!(
                "Bybit API error: {} (code {})",
                response.ret_msg,
                response.ret_code
            );
        }

        let candles: Vec<Candle> = response
            .result
            .list
            .iter()
            .filter_map(|row| {
                if row.len() >= 6 {
                    Some(Candle {
                        timestamp: row[0].parse().ok()?,
                        open: row[1].parse().ok()?,
                        high: row[2].parse().ok()?,
                        low: row[3].parse().ok()?,
                        close: row[4].parse().ok()?,
                        volume: row[5].parse().ok()?,
                    })
                } else {
                    None
                }
            })
            .collect();

        Ok(candles)
    }
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute trading features from candle data.
///
/// Returns a vector of feature vectors. Each feature vector contains:
/// [log_return, normalized_volume, price_to_ma_ratio, volatility_estimate]
pub fn compute_features(candles: &[Candle], ma_period: usize) -> Vec<Array1<f64>> {
    if candles.len() < ma_period + 1 {
        return vec![];
    }

    let avg_volume: f64 = candles.iter().map(|c| c.volume).sum::<f64>() / candles.len() as f64;

    let mut features = Vec::new();

    for i in ma_period..candles.len() {
        // Log return
        let log_return = (candles[i].close / candles[i - 1].close).ln();

        // Normalized volume
        let norm_vol = if avg_volume > 0.0 {
            candles[i].volume / avg_volume
        } else {
            1.0
        };

        // Price to moving average ratio
        let ma: f64 = candles[i - ma_period..i]
            .iter()
            .map(|c| c.close)
            .sum::<f64>()
            / ma_period as f64;
        let price_to_ma = if ma > 0.0 {
            candles[i].close / ma
        } else {
            1.0
        };

        // Volatility estimate from high-low range
        let volatility = if candles[i].close > 0.0 {
            (candles[i].high - candles[i].low) / candles[i].close
        } else {
            0.0
        };

        features.push(Array1::from_vec(vec![
            log_return,
            norm_vol,
            price_to_ma,
            volatility,
        ]));
    }

    features
}

/// Generate a synthetic "true gradient" from market features.
///
/// This simulates what a gradient might look like when training a
/// simple linear model on the market features.
pub fn features_to_gradient(features: &[Array1<f64>]) -> Array1<f64> {
    if features.is_empty() {
        return Array1::zeros(4);
    }

    let dim = features[0].len();
    let n = features.len() as f64;
    let mut mean = Array1::zeros(dim);
    for f in features {
        mean = mean + f;
    }
    mean / n
}

// ─── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn make_rng() -> StdRng {
        StdRng::seed_from_u64(42)
    }

    #[test]
    fn test_fed_avg_no_attack() {
        let g1 = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let g2 = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let g3 = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let result = fed_avg(&[g1, g2, g3]);
        assert!((result[0] - 1.0).abs() < 1e-10);
        assert!((result[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_fed_avg_vulnerable_to_attack() {
        let honest = Array1::from_vec(vec![1.0, 1.0]);
        let malicious = Array1::from_vec(vec![1000.0, 1000.0]);
        let result = fed_avg(&[honest.clone(), honest, malicious]);
        // Average is dominated by the malicious gradient
        assert!(result[0] > 100.0);
    }

    #[test]
    fn test_krum_selects_honest() {
        let g1 = Array1::from_vec(vec![1.0, 1.0]);
        let g2 = Array1::from_vec(vec![1.1, 0.9]);
        let g3 = Array1::from_vec(vec![0.9, 1.1]);
        let g4 = Array1::from_vec(vec![1.05, 0.95]);
        let g5 = Array1::from_vec(vec![100.0, -100.0]); // Byzantine

        let result = krum(&[g1, g2, g3, g4, g5], 1);
        // Result should be close to [1, 1], not [100, -100]
        assert!(result[0] < 2.0);
        assert!(result[1].abs() < 2.0);
    }

    #[test]
    fn test_trimmed_mean_removes_outliers() {
        let gradients: Vec<Array1<f64>> = vec![
            Array1::from_vec(vec![1.0, 1.0]),
            Array1::from_vec(vec![1.1, 0.9]),
            Array1::from_vec(vec![0.9, 1.1]),
            Array1::from_vec(vec![100.0, 100.0]),   // Byzantine
            Array1::from_vec(vec![-100.0, -100.0]),  // Byzantine
        ];
        let result = trimmed_mean(&gradients, 1);
        // After trimming top and bottom 1, we average the 3 middle values
        assert!((result[0] - 1.0).abs() < 0.2);
        assert!((result[1] - 1.0).abs() < 0.2);
    }

    #[test]
    fn test_coordinate_median_robust() {
        let gradients: Vec<Array1<f64>> = vec![
            Array1::from_vec(vec![1.0, 1.0]),
            Array1::from_vec(vec![1.1, 0.9]),
            Array1::from_vec(vec![0.9, 1.1]),
            Array1::from_vec(vec![1000.0, 1000.0]),  // Byzantine
            Array1::from_vec(vec![-1000.0, -1000.0]), // Byzantine
        ];
        let result = coordinate_median(&gradients);
        assert!((result[0] - 1.0).abs() < 0.2);
        assert!((result[1] - 1.0).abs() < 0.2);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = Array1::from_vec(vec![1.0, 0.0]);
        let b = Array1::from_vec(vec![1.0, 0.0]);
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-10);

        let c = Array1::from_vec(vec![-1.0, 0.0]);
        assert!((cosine_similarity(&a, &c) - (-1.0)).abs() < 1e-10);

        let d = Array1::from_vec(vec![0.0, 1.0]);
        assert!(cosine_similarity(&a, &d).abs() < 1e-10);
    }

    #[test]
    fn test_simulation_robustness() {
        let mut rng = make_rng();
        let true_grad = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let attack = AttackStrategy::SignFlip { scale: 10.0 };

        let fedavg_result = simulate_fl_round(
            &true_grad, 8, 2, 0.1, attack, AggregationMethod::FedAvg, &mut rng,
        );
        let krum_result = simulate_fl_round(
            &true_grad, 8, 2, 0.1, attack, AggregationMethod::Krum, &mut rng,
        );

        // Krum should have higher cosine similarity to true gradient
        assert!(krum_result.cosine_sim_to_true > fedavg_result.cosine_sim_to_true);
    }
}
