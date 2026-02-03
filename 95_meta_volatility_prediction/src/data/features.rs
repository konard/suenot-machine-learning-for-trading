//! Feature engineering for volatility prediction.
//!
//! Computes features including realized volatility at multiple horizons,
//! RSI, Bollinger Band width, and volume ratios.

use crate::data::bybit::Kline;

/// Feature generator for volatility prediction.
pub struct FeatureGenerator {
    /// Lookback window for RSI.
    rsi_period: usize,
    /// Lookback window for Bollinger Bands.
    bb_period: usize,
}

/// Computed features for a single time step.
#[derive(Debug, Clone)]
pub struct Features {
    pub log_return: f64,
    pub abs_return: f64,
    pub squared_return: f64,
    pub rv_5: f64,
    pub rv_10: f64,
    pub rv_20: f64,
    pub volume_ratio: f64,
    pub spread: f64,
    pub rsi: f64,
    pub bb_width: f64,
}

impl Features {
    /// Convert to a feature vector.
    pub fn to_vec(&self) -> Vec<f64> {
        vec![
            self.log_return,
            self.abs_return,
            self.squared_return,
            self.rv_5,
            self.rv_10,
            self.rv_20,
            self.volume_ratio,
            self.spread,
            self.rsi,
            self.bb_width,
        ]
    }

    /// Number of features.
    pub const NUM_FEATURES: usize = 10;
}

impl FeatureGenerator {
    /// Create a new feature generator with default parameters.
    pub fn new() -> Self {
        Self {
            rsi_period: 14,
            bb_period: 20,
        }
    }

    /// Create with custom parameters.
    pub fn with_params(rsi_period: usize, bb_period: usize) -> Self {
        Self {
            rsi_period,
            bb_period,
        }
    }

    /// Compute features for all time steps from kline data.
    ///
    /// Returns features starting from index `bb_period` (after warmup).
    pub fn compute(&self, klines: &[Kline]) -> Vec<Features> {
        let n = klines.len();
        if n < self.bb_period + 1 {
            return Vec::new();
        }

        // Compute log returns
        let returns: Vec<f64> = klines
            .windows(2)
            .map(|w| (w[1].close / w[0].close).ln())
            .collect();

        let mut features = Vec::new();
        let warmup = self.bb_period;

        for i in warmup..returns.len() {
            let r = returns[i];

            // Realized volatilities
            let rv_5 = rolling_std(&returns, i, 5);
            let rv_10 = rolling_std(&returns, i, 10);
            let rv_20 = rolling_std(&returns, i, 20);

            // Volume ratio
            let vol_mean = rolling_mean_volume(klines, i + 1, 20);
            let volume_ratio = if vol_mean > 1e-10 {
                klines[i + 1].volume / vol_mean
            } else {
                1.0
            };

            // RSI
            let rsi = compute_rsi(&returns, i, self.rsi_period);

            // Bollinger Band width
            let bb_width = compute_bb_width(klines, i + 1, self.bb_period);

            features.push(Features {
                log_return: r,
                abs_return: r.abs(),
                squared_return: r * r,
                rv_5,
                rv_10,
                rv_20,
                volume_ratio,
                spread: 0.0, // Placeholder
                rsi,
                bb_width,
            });
        }

        features
    }
}

impl Default for FeatureGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// Rolling standard deviation of returns.
fn rolling_std(returns: &[f64], end_idx: usize, window: usize) -> f64 {
    if end_idx + 1 < window {
        return 0.0;
    }
    let start = end_idx + 1 - window;
    let slice = &returns[start..=end_idx];
    let mean: f64 = slice.iter().sum::<f64>() / window as f64;
    let variance: f64 = slice.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / window as f64;
    variance.sqrt()
}

/// Rolling mean volume.
fn rolling_mean_volume(klines: &[Kline], end_idx: usize, window: usize) -> f64 {
    if end_idx < window {
        return klines[..end_idx]
            .iter()
            .map(|k| k.volume)
            .sum::<f64>()
            / end_idx.max(1) as f64;
    }
    let start = end_idx - window;
    klines[start..end_idx]
        .iter()
        .map(|k| k.volume)
        .sum::<f64>()
        / window as f64
}

/// Compute RSI (Relative Strength Index).
fn compute_rsi(returns: &[f64], end_idx: usize, period: usize) -> f64 {
    if end_idx + 1 < period {
        return 50.0;
    }
    let start = end_idx + 1 - period;
    let slice = &returns[start..=end_idx];

    let mut avg_gain = 0.0;
    let mut avg_loss = 0.0;

    for &r in slice {
        if r > 0.0 {
            avg_gain += r;
        } else {
            avg_loss += -r;
        }
    }

    avg_gain /= period as f64;
    avg_loss /= period as f64;

    if avg_loss < 1e-10 {
        return 100.0;
    }

    let rs = avg_gain / avg_loss;
    100.0 - (100.0 / (1.0 + rs))
}

/// Compute Bollinger Band width.
fn compute_bb_width(klines: &[Kline], end_idx: usize, period: usize) -> f64 {
    if end_idx < period {
        return 0.0;
    }
    let start = end_idx - period;
    let prices: Vec<f64> = klines[start..end_idx].iter().map(|k| k.close).collect();

    let mean: f64 = prices.iter().sum::<f64>() / period as f64;
    let std: f64 = (prices.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / period as f64).sqrt();

    if mean.abs() < 1e-10 {
        return 0.0;
    }

    (2.0 * std) / mean
}

/// Create volatility prediction tasks from features.
pub fn create_volatility_tasks(
    features: &[Features],
    returns: &[f64],
    forecast_horizon: usize,
    support_size: usize,
    query_size: usize,
) -> Vec<crate::meta::maml::VolatilityTask> {
    let total = support_size + query_size;
    let mut tasks = Vec::new();

    if features.len() < total + forecast_horizon {
        return tasks;
    }

    let mut start = 0;
    while start + total + forecast_horizon <= features.len() {
        let mut support_x = Vec::new();
        let mut support_y = Vec::new();
        let mut query_x = Vec::new();
        let mut query_y = Vec::new();

        for j in 0..total {
            let idx = start + j;
            let x = features[idx].to_vec();

            // Target: realized volatility over forecast horizon
            let target_returns = &returns[idx..std::cmp::min(idx + forecast_horizon, returns.len())];
            let target_vol = if !target_returns.is_empty() {
                let mean: f64 =
                    target_returns.iter().sum::<f64>() / target_returns.len() as f64;
                (target_returns
                    .iter()
                    .map(|r| (r - mean).powi(2))
                    .sum::<f64>()
                    / target_returns.len() as f64)
                    .sqrt()
            } else {
                0.0
            };

            if j < support_size {
                support_x.push(x);
                support_y.push(target_vol);
            } else {
                query_x.push(x);
                query_y.push(target_vol);
            }
        }

        tasks.push(crate::meta::maml::VolatilityTask {
            support_x,
            support_y,
            query_x,
            query_y,
        });

        start += total;
    }

    tasks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rolling_std() {
        let returns = vec![0.01, -0.02, 0.015, -0.005, 0.02];
        let std = rolling_std(&returns, 4, 5);
        assert!(std > 0.0);
        assert!(std < 1.0);
    }

    #[test]
    fn test_compute_rsi() {
        let returns = vec![0.01, -0.02, 0.015, -0.005, 0.02, 0.01, -0.01,
                           0.005, -0.003, 0.008, 0.012, -0.006, 0.003, 0.007];
        let rsi = compute_rsi(&returns, 13, 14);
        assert!(rsi >= 0.0 && rsi <= 100.0);
    }

    #[test]
    fn test_feature_generator() {
        let klines: Vec<Kline> = (0..50)
            .map(|i| Kline {
                timestamp: i as i64 * 3600000,
                open: 100.0 + (i as f64 * 0.1).sin(),
                high: 101.0 + (i as f64 * 0.1).sin(),
                low: 99.0 + (i as f64 * 0.1).sin(),
                close: 100.5 + (i as f64 * 0.1).sin(),
                volume: 1000.0 + i as f64 * 10.0,
                turnover: 100000.0,
            })
            .collect();

        let gen = FeatureGenerator::new();
        let features = gen.compute(&klines);
        assert!(!features.is_empty());

        for f in &features {
            let v = f.to_vec();
            assert_eq!(v.len(), Features::NUM_FEATURES);
        }
    }
}
