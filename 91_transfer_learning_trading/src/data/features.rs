//! Feature extraction for market data.
//!
//! Transforms raw OHLCV data into features suitable for transfer learning models.
//! Features include price-based, volume-based, technical indicators, and
//! cross-asset features.

use ndarray::{Array1, Array2};

/// OHLCV bar representation.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OHLCVBar {
    pub timestamp: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

impl OHLCVBar {
    pub fn new(timestamp: i64, open: f64, high: f64, low: f64, close: f64, volume: f64) -> Self {
        Self { timestamp, open, high, low, close, volume }
    }

    /// Calculate the return from open to close.
    pub fn return_pct(&self) -> f64 {
        if self.open == 0.0 { 0.0 } else { (self.close - self.open) / self.open }
    }

    /// Calculate the range (high - low) as a fraction of open.
    pub fn range_pct(&self) -> f64 {
        if self.open == 0.0 { 0.0 } else { (self.high - self.low) / self.open }
    }

    /// Calculate the upper shadow ratio.
    pub fn upper_shadow_ratio(&self) -> f64 {
        let body_top = self.close.max(self.open);
        let range = self.high - self.low;
        if range == 0.0 { 0.0 } else { (self.high - body_top) / range }
    }

    /// Calculate the lower shadow ratio.
    pub fn lower_shadow_ratio(&self) -> f64 {
        let body_bottom = self.close.min(self.open);
        let range = self.high - self.low;
        if range == 0.0 { 0.0 } else { (body_bottom - self.low) / range }
    }
}

/// Extracted market features for a time window.
#[derive(Debug, Clone)]
pub struct MarketFeatures {
    /// Feature vector
    pub features: Array1<f64>,
    /// Feature names for interpretability
    pub names: Vec<String>,
    /// Timestamp of the observation
    pub timestamp: i64,
}

/// Feature extractor that computes market features from OHLCV bars.
#[derive(Debug, Clone)]
pub struct FeatureExtractor {
    /// Lookback window for computing features
    lookback: usize,
    /// Whether to include volume features
    include_volume: bool,
    /// Whether to include technical indicators
    include_indicators: bool,
}

impl FeatureExtractor {
    /// Create a new feature extractor.
    pub fn new(lookback: usize) -> Self {
        Self {
            lookback,
            include_volume: true,
            include_indicators: true,
        }
    }

    pub fn with_volume(mut self, include: bool) -> Self {
        self.include_volume = include;
        self
    }

    pub fn with_indicators(mut self, include: bool) -> Self {
        self.include_indicators = include;
        self
    }

    /// Extract features from a sequence of OHLCV bars.
    ///
    /// Returns a feature matrix where each row is a feature vector
    /// for one time step (using lookback window).
    pub fn extract_features(&self, bars: &[OHLCVBar]) -> Array2<f64> {
        if bars.len() < self.lookback + 1 {
            return Array2::zeros((0, self.feature_dim()));
        }

        let n_samples = bars.len() - self.lookback;
        let n_features = self.feature_dim();
        let mut features = Array2::zeros((n_samples, n_features));

        for i in 0..n_samples {
            let window = &bars[i..i + self.lookback + 1];
            let feat = self.compute_window_features(window);
            features.row_mut(i).assign(&feat);
        }

        features
    }

    /// Extract features for a single window and return with metadata.
    pub fn extract_single(&self, bars: &[OHLCVBar]) -> Option<MarketFeatures> {
        if bars.len() < self.lookback + 1 {
            return None;
        }

        let window = &bars[bars.len() - self.lookback - 1..];
        let features = self.compute_window_features(window);
        let names = self.feature_names();
        let timestamp = bars.last().map(|b| b.timestamp).unwrap_or(0);

        Some(MarketFeatures { features, names, timestamp })
    }

    /// Compute features for a single lookback window.
    fn compute_window_features(&self, window: &[OHLCVBar]) -> Array1<f64> {
        let mut features = Vec::new();

        let closes: Vec<f64> = window.iter().map(|b| b.close).collect();
        let volumes: Vec<f64> = window.iter().map(|b| b.volume).collect();

        // Price-based features
        let returns = compute_returns(&closes);
        features.push(returns.last().copied().unwrap_or(0.0)); // Latest return
        features.push(mean(&returns)); // Mean return
        features.push(std_dev(&returns)); // Return volatility
        features.push(skewness(&returns)); // Return skewness
        features.push(kurtosis(&returns)); // Return kurtosis

        // Momentum features
        let n = closes.len();
        if n >= 2 {
            features.push((closes[n - 1] - closes[n - 2]) / closes[n - 2].max(1e-10)); // 1-bar momentum
        } else {
            features.push(0.0);
        }
        if n >= 6 {
            features.push((closes[n - 1] - closes[n - 6]) / closes[n - 6].max(1e-10)); // 5-bar momentum
        } else {
            features.push(0.0);
        }

        // Candle features (latest bar)
        let last = &window[window.len() - 1];
        features.push(last.return_pct());
        features.push(last.range_pct());
        features.push(last.upper_shadow_ratio());
        features.push(last.lower_shadow_ratio());

        // Volume features
        if self.include_volume {
            let vol_mean = mean(&volumes);
            let last_vol = volumes.last().copied().unwrap_or(0.0);
            features.push(if vol_mean > 0.0 { last_vol / vol_mean } else { 0.0 }); // Volume ratio
            features.push(std_dev(&volumes) / vol_mean.max(1e-10)); // Volume CV
        }

        // Technical indicators
        if self.include_indicators {
            features.push(compute_rsi(&closes, 14.min(closes.len() - 1)));
            features.push(compute_sma_ratio(&closes, 5.min(closes.len())));
            features.push(compute_sma_ratio(&closes, 10.min(closes.len())));
            features.push(compute_bollinger_position(&closes, 20.min(closes.len())));
            features.push(compute_atr_ratio(window, 14.min(window.len())));
        }

        Array1::from_vec(features)
    }

    /// Number of features extracted.
    pub fn feature_dim(&self) -> usize {
        let mut dim = 11; // Base price features
        if self.include_volume { dim += 2; }
        if self.include_indicators { dim += 5; }
        dim
    }

    /// Feature names for interpretability.
    pub fn feature_names(&self) -> Vec<String> {
        let mut names = vec![
            "return_last".into(),
            "return_mean".into(),
            "return_std".into(),
            "return_skew".into(),
            "return_kurt".into(),
            "momentum_1".into(),
            "momentum_5".into(),
            "candle_return".into(),
            "candle_range".into(),
            "upper_shadow".into(),
            "lower_shadow".into(),
        ];
        if self.include_volume {
            names.push("volume_ratio".into());
            names.push("volume_cv".into());
        }
        if self.include_indicators {
            names.push("rsi_14".into());
            names.push("sma_5_ratio".into());
            names.push("sma_10_ratio".into());
            names.push("bollinger_pos".into());
            names.push("atr_ratio".into());
        }
        names
    }

    /// Generate labels based on future returns.
    /// 0 = Down, 1 = Hold, 2 = Up
    pub fn generate_labels(
        &self,
        bars: &[OHLCVBar],
        forward_periods: usize,
        threshold: f64,
    ) -> Vec<usize> {
        if bars.len() < self.lookback + 1 + forward_periods {
            return Vec::new();
        }

        let n_samples = bars.len() - self.lookback - forward_periods;
        let mut labels = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            let current_idx = i + self.lookback;
            let future_idx = current_idx + forward_periods;
            let current_close = bars[current_idx].close;
            let future_close = bars[future_idx].close;
            let ret = (future_close - current_close) / current_close.max(1e-10);

            if ret > threshold {
                labels.push(2); // Up
            } else if ret < -threshold {
                labels.push(0); // Down
            } else {
                labels.push(1); // Hold
            }
        }

        labels
    }
}

// Helper statistical functions

fn compute_returns(prices: &[f64]) -> Vec<f64> {
    prices
        .windows(2)
        .map(|w| if w[0] == 0.0 { 0.0 } else { (w[1] - w[0]) / w[0] })
        .collect()
}

fn mean(data: &[f64]) -> f64 {
    if data.is_empty() { return 0.0; }
    data.iter().sum::<f64>() / data.len() as f64
}

fn std_dev(data: &[f64]) -> f64 {
    if data.len() < 2 { return 0.0; }
    let m = mean(data);
    let variance = data.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (data.len() - 1) as f64;
    variance.sqrt()
}

fn skewness(data: &[f64]) -> f64 {
    if data.len() < 3 { return 0.0; }
    let m = mean(data);
    let s = std_dev(data);
    if s == 0.0 { return 0.0; }
    let n = data.len() as f64;
    let m3 = data.iter().map(|x| ((x - m) / s).powi(3)).sum::<f64>();
    m3 / n
}

fn kurtosis(data: &[f64]) -> f64 {
    if data.len() < 4 { return 0.0; }
    let m = mean(data);
    let s = std_dev(data);
    if s == 0.0 { return 0.0; }
    let n = data.len() as f64;
    let m4 = data.iter().map(|x| ((x - m) / s).powi(4)).sum::<f64>();
    m4 / n - 3.0 // Excess kurtosis
}

fn compute_rsi(prices: &[f64], period: usize) -> f64 {
    if prices.len() < period + 1 { return 50.0; }
    let returns = compute_returns(prices);
    let recent = &returns[returns.len().saturating_sub(period)..];

    let gains: f64 = recent.iter().filter(|&&r| r > 0.0).sum();
    let losses: f64 = recent.iter().filter(|&&r| r < 0.0).map(|r| r.abs()).sum();

    if losses == 0.0 { return 100.0; }
    let rs = gains / losses;
    100.0 - 100.0 / (1.0 + rs)
}

fn compute_sma_ratio(prices: &[f64], period: usize) -> f64 {
    if prices.len() < period || period == 0 { return 1.0; }
    let sma = mean(&prices[prices.len() - period..]);
    let current = *prices.last().unwrap_or(&1.0);
    if sma == 0.0 { 1.0 } else { current / sma }
}

fn compute_bollinger_position(prices: &[f64], period: usize) -> f64 {
    if prices.len() < period || period == 0 { return 0.5; }
    let window = &prices[prices.len() - period..];
    let sma = mean(window);
    let std = std_dev(window);
    if std == 0.0 { return 0.5; }
    let current = *prices.last().unwrap_or(&0.0);
    // Position within Bollinger Bands (0.0 = lower band, 1.0 = upper band)
    ((current - (sma - 2.0 * std)) / (4.0 * std)).clamp(0.0, 1.0)
}

fn compute_atr_ratio(bars: &[OHLCVBar], period: usize) -> f64 {
    if bars.len() < period + 1 || period == 0 { return 0.0; }
    let recent = &bars[bars.len() - period - 1..];
    let mut true_ranges = Vec::with_capacity(period);

    for i in 1..recent.len() {
        let tr = (recent[i].high - recent[i].low)
            .max((recent[i].high - recent[i - 1].close).abs())
            .max((recent[i].low - recent[i - 1].close).abs());
        true_ranges.push(tr);
    }

    let atr = mean(&true_ranges);
    let current_close = bars.last().map(|b| b.close).unwrap_or(1.0);
    if current_close == 0.0 { 0.0 } else { atr / current_close }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_bars(n: usize) -> Vec<OHLCVBar> {
        (0..n)
            .map(|i| {
                let price = 100.0 + (i as f64 * 0.1).sin() * 10.0;
                OHLCVBar::new(
                    i as i64 * 3600,
                    price - 0.5,
                    price + 1.0,
                    price - 1.0,
                    price + 0.5,
                    1000.0 + (i as f64) * 10.0,
                )
            })
            .collect()
    }

    #[test]
    fn test_ohlcv_bar() {
        let bar = OHLCVBar::new(0, 100.0, 110.0, 90.0, 105.0, 1000.0);
        assert_eq!(bar.return_pct(), 0.05);
        assert_eq!(bar.range_pct(), 0.2);
    }

    #[test]
    fn test_feature_extractor_creation() {
        let extractor = FeatureExtractor::new(20);
        assert_eq!(extractor.feature_dim(), 18); // 11 + 2 vol + 5 indicators
    }

    #[test]
    fn test_extract_features_shape() {
        let extractor = FeatureExtractor::new(20);
        let bars = sample_bars(50);
        let features = extractor.extract_features(&bars);
        assert_eq!(features.shape()[0], 30); // 50 - 20
        assert_eq!(features.shape()[1], 18);
    }

    #[test]
    fn test_extract_features_no_volume() {
        let extractor = FeatureExtractor::new(10).with_volume(false);
        assert_eq!(extractor.feature_dim(), 16); // 11 + 5 indicators
    }

    #[test]
    fn test_extract_features_no_indicators() {
        let extractor = FeatureExtractor::new(10).with_indicators(false);
        assert_eq!(extractor.feature_dim(), 13); // 11 + 2 volume
    }

    #[test]
    fn test_generate_labels() {
        let extractor = FeatureExtractor::new(5);
        let bars = sample_bars(30);
        let labels = extractor.generate_labels(&bars, 3, 0.005);
        assert!(!labels.is_empty());
        for &l in &labels {
            assert!(l <= 2);
        }
    }

    #[test]
    fn test_extract_single() {
        let extractor = FeatureExtractor::new(10);
        let bars = sample_bars(20);
        let result = extractor.extract_single(&bars);
        assert!(result.is_some());
        let feat = result.unwrap();
        assert_eq!(feat.features.len(), extractor.feature_dim());
        assert_eq!(feat.names.len(), extractor.feature_dim());
    }

    #[test]
    fn test_rsi_calculation() {
        let prices: Vec<f64> = (0..20).map(|i| 100.0 + i as f64).collect();
        let rsi = compute_rsi(&prices, 14);
        assert!(rsi > 50.0); // Uptrend should have RSI > 50
    }

    #[test]
    fn test_statistical_functions() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(mean(&data), 3.0);
        assert!((std_dev(&data) - 1.5811).abs() < 0.01);
    }

    #[test]
    fn test_empty_data() {
        assert_eq!(mean(&[]), 0.0);
        assert_eq!(std_dev(&[]), 0.0);
        assert_eq!(skewness(&[]), 0.0);
    }
}
