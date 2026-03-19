use anyhow::Result;
use ndarray::Array1;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

// ─── Data Structures ───────────────────────────────────────────────────────────

/// Represents a single OHLCV candlestick bar.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OhlcvBar {
    pub timestamp: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// Configuration for data augmentation parameters.
#[derive(Debug, Clone)]
pub struct AugmentationConfig {
    /// Number of knot points for time warping
    pub warp_knots: usize,
    /// Maximum displacement for time warp knots (fraction of series length)
    pub warp_sigma: f64,
    /// Standard deviation for magnitude scaling (log-normal)
    pub scale_sigma: f64,
    /// Standard deviation for jittering noise (relative to price)
    pub jitter_sigma: f64,
    /// Window size as fraction of series length for window slicing
    pub window_ratio: f64,
    /// Volatility scale factor for volatility scaling
    pub volatility_scale_factor: f64,
}

impl Default for AugmentationConfig {
    fn default() -> Self {
        Self {
            warp_knots: 4,
            warp_sigma: 0.1,
            scale_sigma: 0.1,
            jitter_sigma: 0.005,
            window_ratio: 0.8,
            volatility_scale_factor: 1.5,
        }
    }
}

/// Main data augmenter that applies various augmentation techniques.
pub struct DataAugmenter {
    pub config: AugmentationConfig,
    rng: StdRng,
}

impl DataAugmenter {
    /// Create a new DataAugmenter with default config and random seed.
    pub fn new() -> Self {
        Self {
            config: AugmentationConfig::default(),
            rng: StdRng::from_entropy(),
        }
    }

    /// Create a new DataAugmenter with specific config and seed.
    pub fn with_config_and_seed(config: AugmentationConfig, seed: u64) -> Self {
        Self {
            config,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Create a new DataAugmenter with a specific seed for reproducibility.
    pub fn with_seed(seed: u64) -> Self {
        Self {
            config: AugmentationConfig::default(),
            rng: StdRng::seed_from_u64(seed),
        }
    }

    // ─── Time Warping ──────────────────────────────────────────────────────

    /// Apply time warping: smooth non-linear distortion of the time axis.
    ///
    /// Generates random knot displacements and interpolates to create a
    /// warped index mapping. The original values are resampled at warped indices.
    pub fn time_warp(&mut self, data: &Array1<f64>) -> Array1<f64> {
        let n = data.len();
        if n < 2 {
            return data.clone();
        }

        let num_knots = self.config.warp_knots.min(n - 1).max(2);
        let sigma = self.config.warp_sigma;

        // Generate knot positions and displacements
        let mut knot_positions: Vec<f64> = Vec::with_capacity(num_knots + 2);
        let mut knot_values: Vec<f64> = Vec::with_capacity(num_knots + 2);

        // First knot fixed at 0
        knot_positions.push(0.0);
        knot_values.push(0.0);

        // Interior knots with random displacement
        for i in 1..=num_knots {
            let pos = (i as f64) / (num_knots as f64 + 1.0) * (n as f64 - 1.0);
            let displacement = self.rng.gen_range(-sigma..sigma) * (n as f64 - 1.0);
            knot_positions.push(pos);
            knot_values.push(pos + displacement);
        }

        // Last knot fixed at n-1
        knot_positions.push((n - 1) as f64);
        knot_values.push((n - 1) as f64);

        // Ensure monotonicity by sorting and clamping
        for i in 1..knot_values.len() {
            if knot_values[i] <= knot_values[i - 1] {
                knot_values[i] = knot_values[i - 1] + 0.01;
            }
        }

        // Interpolate to get warped indices for each position
        let mut result = Array1::zeros(n);
        for i in 0..n {
            let t = i as f64;

            // Find the knot interval
            let mut k = 0;
            for j in 1..knot_positions.len() {
                if knot_positions[j] >= t {
                    k = j - 1;
                    break;
                }
                k = j - 1;
            }

            // Linear interpolation between knots
            let t0 = knot_positions[k];
            let t1 = knot_positions[k + 1];
            let v0 = knot_values[k];
            let v1 = knot_values[k + 1];

            let alpha = if (t1 - t0).abs() < 1e-10 {
                0.0
            } else {
                (t - t0) / (t1 - t0)
            };
            let warped_idx = v0 + alpha * (v1 - v0);

            // Clamp and interpolate the original data
            let warped_idx = warped_idx.max(0.0).min((n - 1) as f64);
            let idx_floor = warped_idx.floor() as usize;
            let idx_ceil = (idx_floor + 1).min(n - 1);
            let frac = warped_idx - idx_floor as f64;

            result[i] = data[idx_floor] * (1.0 - frac) + data[idx_ceil] * frac;
        }

        result
    }

    // ─── Magnitude Scaling ─────────────────────────────────────────────────

    /// Apply magnitude scaling: multiply the series by a random factor.
    ///
    /// The scaling factor is drawn from a log-normal distribution centered at 1.0
    /// to ensure positive values and symmetric scaling on a log scale.
    pub fn magnitude_scale(&mut self, data: &Array1<f64>) -> Array1<f64> {
        let sigma = self.config.scale_sigma;
        // Log-normal: exp(N(0, sigma^2))
        let log_factor: f64 = self.rng.gen_range(-sigma..sigma);
        let factor = log_factor.exp();
        data.mapv(|x| x * factor)
    }

    // ─── Jittering ─────────────────────────────────────────────────────────

    /// Apply jittering: add Gaussian noise scaled to the local price level.
    ///
    /// Noise magnitude is proportional to the absolute price value, ensuring
    /// realistic perturbations across different price levels.
    pub fn jitter(&mut self, data: &Array1<f64>) -> Array1<f64> {
        let sigma = self.config.jitter_sigma;
        let n = data.len();
        let mut result = Array1::zeros(n);

        for i in 0..n {
            let noise = self.rng.gen_range(-3.0..3.0) * sigma * data[i].abs().max(1.0);
            result[i] = data[i] + noise;
        }

        result
    }

    // ─── Window Slicing ────────────────────────────────────────────────────

    /// Apply window slicing: extract a random contiguous subsequence.
    ///
    /// Returns a window of length `floor(n * window_ratio)` starting at a
    /// random position within the series.
    pub fn window_slice(&mut self, data: &Array1<f64>) -> Array1<f64> {
        let n = data.len();
        let window_len = ((n as f64) * self.config.window_ratio).floor() as usize;
        let window_len = window_len.max(1).min(n);

        if window_len >= n {
            return data.clone();
        }

        let max_start = n - window_len;
        let start = self.rng.gen_range(0..=max_start);

        data.slice(ndarray::s![start..start + window_len]).to_owned()
    }

    // ─── Volatility Scaling ────────────────────────────────────────────────

    /// Apply volatility scaling: adjust return volatility while preserving
    /// the directional pattern.
    ///
    /// Computes returns, scales their deviation from the mean by
    /// `volatility_scale_factor`, and reconstructs the price series.
    pub fn volatility_scale(&mut self, data: &Array1<f64>) -> Array1<f64> {
        let n = data.len();
        if n < 2 {
            return data.clone();
        }

        // Compute log returns
        let mut returns = Array1::zeros(n - 1);
        for i in 0..n - 1 {
            if data[i] > 0.0 && data[i + 1] > 0.0 {
                returns[i] = (data[i + 1] / data[i]).ln();
            }
        }

        // Compute mean return
        let mean_return = returns.mean().unwrap_or(0.0);

        // Scale deviation from mean
        let scale = self.config.volatility_scale_factor;
        let scaled_returns = returns.mapv(|r| mean_return + scale * (r - mean_return));

        // Reconstruct prices from scaled returns
        let mut result = Array1::zeros(n);
        result[0] = data[0];
        for i in 0..n - 1 {
            result[i + 1] = result[i] * scaled_returns[i].exp();
        }

        result
    }

    // ─── Combined Augmentation ─────────────────────────────────────────────

    /// Apply all augmentation techniques in sequence: jitter + time warp + magnitude scale.
    pub fn augment_all(&mut self, data: &Array1<f64>) -> Array1<f64> {
        let jittered = self.jitter(data);
        let warped = self.time_warp(&jittered);
        self.magnitude_scale(&warped)
    }
}

// ─── Bybit API Integration ─────────────────────────────────────────────────────

/// Response structure from Bybit v5 kline API.
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

/// Fetch OHLCV kline data from Bybit v5 API.
///
/// # Arguments
/// * `symbol` - Trading pair, e.g., "BTCUSDT"
/// * `interval` - Candle interval: "1", "5", "15", "60", "240", "D", "W"
/// * `limit` - Number of candles to fetch (max 200)
pub fn fetch_bybit_klines(symbol: &str, interval: &str, limit: usize) -> Result<Vec<OhlcvBar>> {
    let url = format!(
        "https://api.bybit.com/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
        symbol, interval, limit
    );

    let client = reqwest::blocking::Client::new();
    let resp: BybitResponse = client
        .get(&url)
        .header("User-Agent", "data-augmentation-trading/0.1.0")
        .send()?
        .json()?;

    if resp.ret_code != 0 {
        anyhow::bail!("Bybit API error: {} (code {})", resp.ret_msg, resp.ret_code);
    }

    let mut bars: Vec<OhlcvBar> = Vec::new();
    for item in &resp.result.list {
        if item.len() >= 6 {
            bars.push(OhlcvBar {
                timestamp: item[0].parse().unwrap_or(0),
                open: item[1].parse().unwrap_or(0.0),
                high: item[2].parse().unwrap_or(0.0),
                low: item[3].parse().unwrap_or(0.0),
                close: item[4].parse().unwrap_or(0.0),
                volume: item[5].parse().unwrap_or(0.0),
            });
        }
    }

    // Bybit returns newest first; reverse to chronological order
    bars.reverse();

    Ok(bars)
}

/// Extract close prices from a vector of OHLCV bars into an ndarray.
pub fn extract_close_prices(bars: &[OhlcvBar]) -> Array1<f64> {
    Array1::from_vec(bars.iter().map(|b| b.close).collect())
}

/// Compute simple returns from a price series.
pub fn compute_returns(prices: &Array1<f64>) -> Array1<f64> {
    let n = prices.len();
    if n < 2 {
        return Array1::zeros(0);
    }
    let mut returns = Array1::zeros(n - 1);
    for i in 0..n - 1 {
        if prices[i].abs() > 1e-10 {
            returns[i] = (prices[i + 1] - prices[i]) / prices[i];
        }
    }
    returns
}

/// Compute basic statistics for a price series.
pub fn compute_stats(data: &Array1<f64>) -> (f64, f64, f64, f64) {
    let n = data.len() as f64;
    if n < 1.0 {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let mean = data.mean().unwrap_or(0.0);
    let variance = data.mapv(|x| (x - mean).powi(2)).sum() / n;
    let std_dev = variance.sqrt();
    let min = data.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    (mean, std_dev, min, max)
}

// ─── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_data() -> Array1<f64> {
        Array1::from_vec(vec![
            100.0, 101.5, 99.8, 102.3, 103.1, 101.0, 104.5, 105.2, 103.8, 106.0,
            107.5, 106.2, 108.0, 109.3, 107.8, 110.5, 111.2, 109.0, 112.3, 113.5,
        ])
    }

    #[test]
    fn test_time_warp_preserves_length() {
        let data = sample_data();
        let mut aug = DataAugmenter::with_seed(42);
        let warped = aug.time_warp(&data);
        assert_eq!(warped.len(), data.len());
    }

    #[test]
    fn test_time_warp_changes_values() {
        let data = sample_data();
        let mut aug = DataAugmenter::with_seed(42);
        let warped = aug.time_warp(&data);
        // At least some values should differ
        let diffs: Vec<bool> = data
            .iter()
            .zip(warped.iter())
            .map(|(a, b)| (a - b).abs() > 1e-10)
            .collect();
        assert!(diffs.iter().any(|&d| d), "Time warp should change some values");
    }

    #[test]
    fn test_magnitude_scale_preserves_length() {
        let data = sample_data();
        let mut aug = DataAugmenter::with_seed(42);
        let scaled = aug.magnitude_scale(&data);
        assert_eq!(scaled.len(), data.len());
    }

    #[test]
    fn test_magnitude_scale_positive() {
        let data = sample_data();
        let mut aug = DataAugmenter::with_seed(42);
        let scaled = aug.magnitude_scale(&data);
        assert!(
            scaled.iter().all(|&x| x > 0.0),
            "Scaled prices should remain positive"
        );
    }

    #[test]
    fn test_jitter_preserves_length() {
        let data = sample_data();
        let mut aug = DataAugmenter::with_seed(42);
        let jittered = aug.jitter(&data);
        assert_eq!(jittered.len(), data.len());
    }

    #[test]
    fn test_jitter_stays_close() {
        let data = sample_data();
        let mut aug = DataAugmenter::with_seed(42);
        let jittered = aug.jitter(&data);
        for (orig, noisy) in data.iter().zip(jittered.iter()) {
            let pct_diff = ((noisy - orig) / orig).abs();
            assert!(
                pct_diff < 0.05,
                "Jitter should produce small perturbations, got {}% change",
                pct_diff * 100.0
            );
        }
    }

    #[test]
    fn test_window_slice_shorter() {
        let data = sample_data();
        let mut aug = DataAugmenter::with_seed(42);
        let sliced = aug.window_slice(&data);
        let expected_len = ((data.len() as f64) * aug.config.window_ratio).floor() as usize;
        assert_eq!(sliced.len(), expected_len);
        assert!(sliced.len() < data.len());
    }

    #[test]
    fn test_volatility_scale_preserves_length() {
        let data = sample_data();
        let mut aug = DataAugmenter::with_seed(42);
        let scaled = aug.volatility_scale(&data);
        assert_eq!(scaled.len(), data.len());
    }

    #[test]
    fn test_volatility_scale_preserves_start() {
        let data = sample_data();
        let mut aug = DataAugmenter::with_seed(42);
        let scaled = aug.volatility_scale(&data);
        assert!(
            (scaled[0] - data[0]).abs() < 1e-10,
            "Volatility scaling should preserve the starting price"
        );
    }

    #[test]
    fn test_volatility_scale_increases_variance() {
        let data = sample_data();
        let mut aug = DataAugmenter::with_config_and_seed(
            AugmentationConfig {
                volatility_scale_factor: 2.0,
                ..AugmentationConfig::default()
            },
            42,
        );
        let scaled = aug.volatility_scale(&data);

        let orig_returns = compute_returns(&data);
        let scaled_returns = compute_returns(&scaled);

        let orig_var = orig_returns.mapv(|x| x.powi(2)).mean().unwrap_or(0.0);
        let scaled_var = scaled_returns.mapv(|x| x.powi(2)).mean().unwrap_or(0.0);

        assert!(
            scaled_var > orig_var,
            "Volatility scaling with factor > 1 should increase return variance"
        );
    }

    #[test]
    fn test_compute_returns_length() {
        let data = sample_data();
        let returns = compute_returns(&data);
        assert_eq!(returns.len(), data.len() - 1);
    }

    #[test]
    fn test_compute_stats() {
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let (mean, std_dev, min, max) = compute_stats(&data);
        assert!((mean - 3.0).abs() < 1e-10);
        assert!((min - 1.0).abs() < 1e-10);
        assert!((max - 5.0).abs() < 1e-10);
        assert!(std_dev > 0.0);
    }
}
