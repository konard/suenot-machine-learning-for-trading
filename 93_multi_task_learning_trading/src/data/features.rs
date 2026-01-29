//! Feature engineering pipeline for multi-task learning.
//!
//! Converts raw OHLCV kline data into technical indicators
//! suitable for the multi-task model.

use crate::data::bybit::Kline;
use crate::mtl::trainer::TaskTargets;

/// Feature vector computed from kline data.
#[derive(Debug, Clone)]
pub struct FeatureRow {
    pub features: Vec<f64>,
    pub timestamp: i64,
}

/// Multi-task targets computed from kline data.
#[derive(Debug, Clone)]
pub struct TargetRow {
    pub targets: TaskTargets,
    pub timestamp: i64,
}

/// Feature generator that computes technical indicators from klines.
pub struct FeatureGenerator {
    window: usize,
}

impl FeatureGenerator {
    /// Create a new feature generator.
    ///
    /// # Arguments
    /// * `window` - Lookback window for rolling features
    pub fn new(window: usize) -> Self {
        Self { window }
    }

    /// Generate features from kline data.
    ///
    /// Returns feature vectors with:
    /// - Returns at 1, 5, 10 periods
    /// - SMA ratio, EMA ratio
    /// - Realized volatility
    /// - Momentum
    /// - RSI
    /// - MACD
    /// - Bollinger Band position
    pub fn generate_features(&self, klines: &[Kline]) -> Vec<FeatureRow> {
        if klines.len() < self.window + 26 {
            return Vec::new();
        }

        let closes: Vec<f64> = klines.iter().map(|k| k.close).collect();
        let n = closes.len();
        let mut rows = Vec::new();

        let start = self.window.max(26);

        for i in start..n {
            let mut features = Vec::with_capacity(10);

            // Returns at different horizons
            features.push(if i >= 1 { closes[i] / closes[i - 1] - 1.0 } else { 0.0 });
            features.push(if i >= 5 { closes[i] / closes[i - 5] - 1.0 } else { 0.0 });
            features.push(if i >= 10 { closes[i] / closes[i - 10] - 1.0 } else { 0.0 });

            // SMA ratio
            let sma: f64 = closes[i + 1 - self.window..=i].iter().sum::<f64>() / self.window as f64;
            features.push(closes[i] / sma);

            // EMA ratio (approximate)
            let ema = self.compute_ema(&closes[..=i], self.window);
            features.push(closes[i] / ema);

            // Realized volatility
            let returns: Vec<f64> = (i + 1 - self.window..=i)
                .map(|j| if j > 0 { closes[j] / closes[j - 1] - 1.0 } else { 0.0 })
                .collect();
            let mean_ret: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
            let variance: f64 = returns.iter().map(|r| (r - mean_ret).powi(2)).sum::<f64>()
                / returns.len() as f64;
            features.push(variance.sqrt());

            // Momentum
            features.push(closes[i] / closes[i - self.window] - 1.0);

            // RSI
            features.push(self.compute_rsi(&closes[..=i], self.window));

            // MACD
            let ema12 = self.compute_ema(&closes[..=i], 12);
            let ema26 = self.compute_ema(&closes[..=i], 26);
            features.push((ema12 - ema26) / closes[i]);

            // Bollinger Band position
            let std_dev = variance.sqrt();
            features.push((closes[i] - sma) / (2.0 * std_dev + 1e-10));

            rows.push(FeatureRow {
                features,
                timestamp: klines[i].timestamp,
            });
        }

        rows
    }

    /// Generate multi-task targets from kline data.
    ///
    /// # Arguments
    /// * `klines` - Raw kline data
    /// * `horizon` - Prediction horizon in periods
    pub fn generate_targets(&self, klines: &[Kline], horizon: usize) -> Vec<TargetRow> {
        let closes: Vec<f64> = klines.iter().map(|k| k.close).collect();
        let volumes: Vec<f64> = klines.iter().map(|k| k.volume).collect();
        let n = closes.len();
        let mut rows = Vec::new();

        let start = self.window.max(26);

        for i in start..n.saturating_sub(horizon) {
            let future_return = closes[i + horizon] / closes[i] - 1.0;

            // Future volatility
            let future_returns: Vec<f64> = (i + 1..=i + horizon)
                .map(|j| if j < n && j > 0 { closes[j] / closes[j - 1] - 1.0 } else { 0.0 })
                .collect();
            let mean_r: f64 = future_returns.iter().sum::<f64>() / future_returns.len() as f64;
            let vol: f64 = (future_returns.iter().map(|r| (r - mean_r).powi(2)).sum::<f64>()
                / future_returns.len() as f64).sqrt();

            // Normalized volume
            let vol_mean: f64 = if i >= 20 {
                volumes[i - 19..=i].iter().sum::<f64>() / 20.0
            } else {
                volumes[..=i].iter().sum::<f64>() / (i + 1) as f64
            };
            let norm_volume = if i + horizon < n {
                volumes[i + horizon] / (vol_mean + 1e-10)
            } else {
                1.0
            };

            rows.push(TargetRow {
                targets: TaskTargets {
                    return_target: future_return,
                    volatility_target: vol,
                    direction_target: if future_return > 0.0 { 1.0 } else { 0.0 },
                    volume_target: norm_volume,
                },
                timestamp: klines[i].timestamp,
            });
        }

        rows
    }

    /// Compute exponential moving average.
    fn compute_ema(&self, data: &[f64], span: usize) -> f64 {
        let alpha = 2.0 / (span as f64 + 1.0);
        let mut ema = data[0];
        for &val in &data[1..] {
            ema = alpha * val + (1.0 - alpha) * ema;
        }
        ema
    }

    /// Compute RSI (Relative Strength Index).
    fn compute_rsi(&self, prices: &[f64], window: usize) -> f64 {
        if prices.len() < window + 1 {
            return 50.0;
        }

        let start = prices.len() - window - 1;
        let mut avg_gain = 0.0;
        let mut avg_loss = 0.0;

        for i in start + 1..prices.len() {
            let change = prices[i] - prices[i - 1];
            if change > 0.0 {
                avg_gain += change;
            } else {
                avg_loss -= change;
            }
        }

        avg_gain /= window as f64;
        avg_loss /= window as f64;

        if avg_loss < 1e-10 {
            return 100.0;
        }

        100.0 - (100.0 / (1.0 + avg_gain / avg_loss))
    }
}

impl Default for FeatureGenerator {
    fn default() -> Self {
        Self::new(20)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::bybit::SimulatedDataGenerator;

    #[test]
    fn test_feature_generation() {
        let klines = SimulatedDataGenerator::generate_klines(200, 50000.0, 0.02);
        let gen = FeatureGenerator::new(20);
        let features = gen.generate_features(&klines);

        assert!(!features.is_empty());
        assert_eq!(features[0].features.len(), 10);
    }

    #[test]
    fn test_target_generation() {
        let klines = SimulatedDataGenerator::generate_klines(200, 50000.0, 0.02);
        let gen = FeatureGenerator::new(20);
        let targets = gen.generate_targets(&klines, 5);

        assert!(!targets.is_empty());
        for t in &targets {
            assert!(t.targets.direction_target == 0.0 || t.targets.direction_target == 1.0);
            assert!(t.targets.volatility_target >= 0.0);
        }
    }

    #[test]
    fn test_feature_target_alignment() {
        let klines = SimulatedDataGenerator::generate_klines(200, 50000.0, 0.02);
        let gen = FeatureGenerator::new(20);
        let features = gen.generate_features(&klines);
        let targets = gen.generate_targets(&klines, 5);

        // Both should have similar lengths (targets are shorter due to horizon)
        assert!(features.len() >= targets.len());
    }
}
