//! Feature engineering for trading

use crate::api::Kline;
use ndarray::Array1;

/// Feature engineering pipeline
pub struct FeatureEngineering {
    /// Feature transformations to apply
    transformations: Vec<FeatureTransform>,
    /// Lookback periods
    lookback: usize,
}

/// Feature transformation types
#[derive(Clone)]
pub enum FeatureTransform {
    /// Log returns
    Returns(usize),
    /// Rolling volatility
    Volatility(usize),
    /// RSI
    RSI(usize),
    /// MACD
    MACD(usize, usize, usize),
    /// Bollinger Band position
    BollingerPosition(usize, f64),
    /// Volume ratio
    VolumeRatio(usize),
    /// Price momentum
    Momentum(usize),
    /// ATR (Average True Range)
    ATR(usize),
    /// Rate of Change
    ROC(usize),
    /// Stochastic oscillator
    Stochastic(usize, usize),
}

impl Default for FeatureEngineering {
    fn default() -> Self {
        Self::new()
    }
}

impl FeatureEngineering {
    /// Create new feature engineering pipeline
    pub fn new() -> Self {
        Self {
            transformations: Vec::new(),
            lookback: 0,
        }
    }

    /// Add returns feature
    pub fn add_returns(mut self, period: usize) -> Self {
        self.transformations.push(FeatureTransform::Returns(period));
        self.lookback = self.lookback.max(period);
        self
    }

    /// Add volatility feature
    pub fn add_volatility(mut self, period: usize) -> Self {
        self.transformations.push(FeatureTransform::Volatility(period));
        self.lookback = self.lookback.max(period);
        self
    }

    /// Add RSI feature
    pub fn add_rsi(mut self, period: usize) -> Self {
        self.transformations.push(FeatureTransform::RSI(period));
        self.lookback = self.lookback.max(period + 1);
        self
    }

    /// Add MACD feature
    pub fn add_macd(mut self, fast: usize, slow: usize, signal: usize) -> Self {
        self.transformations.push(FeatureTransform::MACD(fast, slow, signal));
        self.lookback = self.lookback.max(slow + signal);
        self
    }

    /// Add Bollinger Band position
    pub fn add_bollinger(mut self, period: usize, std_dev: f64) -> Self {
        self.transformations.push(FeatureTransform::BollingerPosition(period, std_dev));
        self.lookback = self.lookback.max(period);
        self
    }

    /// Add volume ratio
    pub fn add_volume_ratio(mut self, period: usize) -> Self {
        self.transformations.push(FeatureTransform::VolumeRatio(period));
        self.lookback = self.lookback.max(period);
        self
    }

    /// Add momentum
    pub fn add_momentum(mut self, period: usize) -> Self {
        self.transformations.push(FeatureTransform::Momentum(period));
        self.lookback = self.lookback.max(period);
        self
    }

    /// Add ATR
    pub fn add_atr(mut self, period: usize) -> Self {
        self.transformations.push(FeatureTransform::ATR(period));
        self.lookback = self.lookback.max(period);
        self
    }

    /// Transform klines to features
    pub fn transform(&self, klines: &[Kline]) -> Vec<Array1<f64>> {
        if klines.len() < self.lookback + 1 {
            return Vec::new();
        }

        let n_samples = klines.len() - self.lookback;
        let mut features: Vec<Array1<f64>> = Vec::with_capacity(n_samples);

        for i in self.lookback..klines.len() {
            let window = &klines[i - self.lookback..=i];
            let feature_vec = self.compute_features(window);
            features.push(feature_vec);
        }

        features
    }

    /// Compute features for a single window
    fn compute_features(&self, window: &[Kline]) -> Array1<f64> {
        let mut features = Vec::new();

        for transform in &self.transformations {
            match transform {
                FeatureTransform::Returns(period) => {
                    let ret = self.compute_returns(window, *period);
                    features.push(ret);
                }
                FeatureTransform::Volatility(period) => {
                    let vol = self.compute_volatility(window, *period);
                    features.push(vol);
                }
                FeatureTransform::RSI(period) => {
                    let rsi = self.compute_rsi(window, *period);
                    features.push(rsi);
                }
                FeatureTransform::MACD(fast, slow, signal) => {
                    let (macd, signal_line, hist) = self.compute_macd(window, *fast, *slow, *signal);
                    features.push(macd);
                    features.push(signal_line);
                    features.push(hist);
                }
                FeatureTransform::BollingerPosition(period, std_dev) => {
                    let pos = self.compute_bollinger_position(window, *period, *std_dev);
                    features.push(pos);
                }
                FeatureTransform::VolumeRatio(period) => {
                    let ratio = self.compute_volume_ratio(window, *period);
                    features.push(ratio);
                }
                FeatureTransform::Momentum(period) => {
                    let mom = self.compute_momentum(window, *period);
                    features.push(mom);
                }
                FeatureTransform::ATR(period) => {
                    let atr = self.compute_atr(window, *period);
                    features.push(atr);
                }
                _ => {}
            }
        }

        Array1::from_vec(features)
    }

    fn compute_returns(&self, window: &[Kline], period: usize) -> f64 {
        if window.len() <= period {
            return 0.0;
        }
        let current = window.last().unwrap().close;
        let past = window[window.len() - 1 - period].close;
        if past > 0.0 {
            (current / past).ln()
        } else {
            0.0
        }
    }

    fn compute_volatility(&self, window: &[Kline], period: usize) -> f64 {
        let returns: Vec<f64> = window.windows(2)
            .take(period)
            .map(|w| (w[1].close / w[0].close).ln())
            .collect();

        if returns.is_empty() {
            return 0.0;
        }

        let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance: f64 = returns.iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / returns.len() as f64;

        variance.sqrt()
    }

    fn compute_rsi(&self, window: &[Kline], period: usize) -> f64 {
        let changes: Vec<f64> = window.windows(2)
            .map(|w| w[1].close - w[0].close)
            .collect();

        if changes.len() < period {
            return 50.0;
        }

        let recent = &changes[changes.len() - period..];
        let gains: f64 = recent.iter().filter(|&&x| x > 0.0).sum();
        let losses: f64 = recent.iter().filter(|&&x| x < 0.0).map(|x| x.abs()).sum();

        let avg_gain = gains / period as f64;
        let avg_loss = losses / period as f64;

        if avg_loss == 0.0 {
            return 100.0;
        }

        let rs = avg_gain / avg_loss;
        100.0 - (100.0 / (1.0 + rs))
    }

    fn compute_macd(&self, window: &[Kline], fast: usize, slow: usize, signal: usize) -> (f64, f64, f64) {
        let closes: Vec<f64> = window.iter().map(|k| k.close).collect();

        let fast_ema = self.ema(&closes, fast);
        let slow_ema = self.ema(&closes, slow);
        let macd_line = fast_ema - slow_ema;

        // For signal line, we'd need MACD history
        // Simplified: use last few points
        let signal_line = macd_line * 0.9; // Approximate
        let histogram = macd_line - signal_line;

        // Normalize
        let price = closes.last().unwrap_or(&1.0);
        (macd_line / price, signal_line / price, histogram / price)
    }

    fn compute_bollinger_position(&self, window: &[Kline], period: usize, std_dev: f64) -> f64 {
        let closes: Vec<f64> = window.iter().rev().take(period).map(|k| k.close).collect();

        if closes.is_empty() {
            return 0.0;
        }

        let mean: f64 = closes.iter().sum::<f64>() / closes.len() as f64;
        let variance: f64 = closes.iter()
            .map(|c| (c - mean).powi(2))
            .sum::<f64>() / closes.len() as f64;
        let std = variance.sqrt();

        let current = window.last().unwrap().close;
        let upper = mean + std_dev * std;
        let lower = mean - std_dev * std;

        if upper == lower {
            return 0.0;
        }

        (current - lower) / (upper - lower) * 2.0 - 1.0 // Normalize to [-1, 1]
    }

    fn compute_volume_ratio(&self, window: &[Kline], period: usize) -> f64 {
        let volumes: Vec<f64> = window.iter().map(|k| k.volume).collect();

        if volumes.len() < period {
            return 1.0;
        }

        let current = volumes.last().unwrap_or(&1.0);
        let avg: f64 = volumes.iter().rev().skip(1).take(period).sum::<f64>() / period as f64;

        if avg > 0.0 {
            current / avg
        } else {
            1.0
        }
    }

    fn compute_momentum(&self, window: &[Kline], period: usize) -> f64 {
        if window.len() <= period {
            return 0.0;
        }

        let current = window.last().unwrap().close;
        let past = window[window.len() - 1 - period].close;

        if past > 0.0 {
            (current - past) / past
        } else {
            0.0
        }
    }

    fn compute_atr(&self, window: &[Kline], period: usize) -> f64 {
        if window.len() < 2 {
            return 0.0;
        }

        let tr_values: Vec<f64> = window.windows(2)
            .map(|w| {
                let high_low = w[1].high - w[1].low;
                let high_close = (w[1].high - w[0].close).abs();
                let low_close = (w[1].low - w[0].close).abs();
                high_low.max(high_close).max(low_close)
            })
            .collect();

        let recent: Vec<f64> = tr_values.iter().rev().take(period).cloned().collect();
        if recent.is_empty() {
            return 0.0;
        }

        let atr = recent.iter().sum::<f64>() / recent.len() as f64;
        let price = window.last().unwrap().close;

        if price > 0.0 {
            atr / price // Normalize by price
        } else {
            atr
        }
    }

    fn ema(&self, values: &[f64], period: usize) -> f64 {
        if values.is_empty() || period == 0 {
            return 0.0;
        }

        let multiplier = 2.0 / (period as f64 + 1.0);
        let mut ema = values[0];

        for value in values.iter().skip(1) {
            ema = (value - ema) * multiplier + ema;
        }

        ema
    }

    /// Get number of features
    pub fn num_features(&self) -> usize {
        self.transformations.iter().map(|t| match t {
            FeatureTransform::MACD(_, _, _) => 3,
            _ => 1,
        }).sum()
    }

    /// Get required lookback
    pub fn required_lookback(&self) -> usize {
        self.lookback
    }
}

/// Prepare supervised learning data
pub fn prepare_supervised(
    features: &[Array1<f64>],
    targets: &[f64],
    prediction_horizon: usize,
) -> (Vec<Array1<f64>>, Vec<Array1<f64>>) {
    if features.len() < prediction_horizon + 1 {
        return (Vec::new(), Vec::new());
    }

    let n = features.len() - prediction_horizon;
    let inputs: Vec<Array1<f64>> = features[..n].to_vec();
    let outputs: Vec<Array1<f64>> = targets[prediction_horizon..]
        .iter()
        .take(n)
        .map(|&t| Array1::from_vec(vec![t]))
        .collect();

    (inputs, outputs)
}

/// Normalize features to [-1, 1] range
pub fn normalize_features(features: &mut [Array1<f64>]) {
    if features.is_empty() {
        return;
    }

    let n_features = features[0].len();

    for j in 0..n_features {
        let values: Vec<f64> = features.iter().map(|f| f[j]).collect();
        let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let range = max - min;
        if range > 1e-10 {
            for feature in features.iter_mut() {
                feature[j] = 2.0 * (feature[j] - min) / range - 1.0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_klines(n: usize) -> Vec<Kline> {
        (0..n).map(|i| Kline {
            start_time: i as i64 * 60000,
            open: 100.0 + i as f64,
            high: 102.0 + i as f64,
            low: 98.0 + i as f64,
            close: 101.0 + i as f64,
            volume: 1000.0 + i as f64 * 10.0,
            turnover: 100000.0,
        }).collect()
    }

    #[test]
    fn test_feature_engineering() {
        let klines = create_test_klines(100);

        let features = FeatureEngineering::new()
            .add_returns(10)
            .add_volatility(20)
            .add_rsi(14)
            .transform(&klines);

        assert!(!features.is_empty());
        assert_eq!(features[0].len(), 3); // 3 features
    }

    #[test]
    fn test_rsi_calculation() {
        let klines = create_test_klines(50);
        let fe = FeatureEngineering::new().add_rsi(14);
        let features = fe.transform(&klines);

        for f in &features {
            assert!(f[0] >= 0.0 && f[0] <= 100.0);
        }
    }

    #[test]
    fn test_normalize() {
        let mut features = vec![
            Array1::from_vec(vec![0.0, 100.0]),
            Array1::from_vec(vec![50.0, 200.0]),
            Array1::from_vec(vec![100.0, 300.0]),
        ];

        normalize_features(&mut features);

        // After normalization, should be in [-1, 1]
        for f in &features {
            for &v in f.iter() {
                assert!(v >= -1.0 && v <= 1.0);
            }
        }
    }
}
