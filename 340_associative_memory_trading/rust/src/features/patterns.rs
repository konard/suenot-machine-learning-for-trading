//! Pattern construction for associative memory
//!
//! Creates market pattern vectors from OHLCV data for use with
//! associative memory networks.

use crate::data::OHLCVSeries;
use ndarray::{Array1, Array2};

/// Statistics helper functions
mod stats {
    /// Calculate mean of a slice
    pub fn mean(data: &[f64]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }
        data.iter().sum::<f64>() / data.len() as f64
    }

    /// Calculate standard deviation
    pub fn std(data: &[f64]) -> f64 {
        if data.len() < 2 {
            return 0.0;
        }
        let m = mean(data);
        let variance = data.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (data.len() - 1) as f64;
        variance.sqrt()
    }

    /// Calculate skewness
    pub fn skewness(data: &[f64]) -> f64 {
        if data.len() < 3 {
            return 0.0;
        }
        let m = mean(data);
        let s = std(data);
        if s == 0.0 {
            return 0.0;
        }
        let n = data.len() as f64;
        let m3 = data.iter().map(|x| ((x - m) / s).powi(3)).sum::<f64>() / n;
        m3
    }

    /// Calculate kurtosis (excess kurtosis)
    pub fn kurtosis(data: &[f64]) -> f64 {
        if data.len() < 4 {
            return 0.0;
        }
        let m = mean(data);
        let s = std(data);
        if s == 0.0 {
            return 0.0;
        }
        let n = data.len() as f64;
        let m4 = data.iter().map(|x| ((x - m) / s).powi(4)).sum::<f64>() / n;
        m4 - 3.0 // Excess kurtosis
    }

    /// Calculate min of a slice
    pub fn min(data: &[f64]) -> f64 {
        data.iter().cloned().fold(f64::INFINITY, f64::min)
    }

    /// Calculate max of a slice
    pub fn max(data: &[f64]) -> f64 {
        data.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    }
}

/// Market pattern representation
#[derive(Debug, Clone)]
pub struct Pattern {
    /// Feature vector
    pub features: Array1<f64>,
    /// Timestamp of the pattern
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Label (future return)
    pub label: Option<f64>,
}

impl Pattern {
    /// Create a new pattern
    pub fn new(features: Array1<f64>, timestamp: chrono::DateTime<chrono::Utc>) -> Self {
        Self {
            features,
            timestamp,
            label: None,
        }
    }

    /// Create pattern with label
    pub fn with_label(
        features: Array1<f64>,
        timestamp: chrono::DateTime<chrono::Utc>,
        label: f64,
    ) -> Self {
        Self {
            features,
            timestamp,
            label: Some(label),
        }
    }

    /// Get pattern dimension
    pub fn dim(&self) -> usize {
        self.features.len()
    }

    /// Compute cosine similarity with another pattern
    pub fn similarity(&self, other: &Pattern) -> f64 {
        let dot: f64 = self
            .features
            .iter()
            .zip(other.features.iter())
            .map(|(a, b)| a * b)
            .sum();

        let norm_a: f64 = self.features.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
        let norm_b: f64 = other.features.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }

        dot / (norm_a * norm_b)
    }

    /// Normalize the pattern to unit length
    pub fn normalize(&mut self) {
        let norm: f64 = self.features.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
        if norm > 0.0 {
            self.features.mapv_inplace(|x| x / norm);
        }
    }
}

/// Pattern builder configuration
#[derive(Debug, Clone)]
pub struct PatternConfig {
    /// Lookback period for pattern construction
    pub lookback: usize,
    /// Forward period for label calculation
    pub forward: usize,
    /// Whether to normalize patterns
    pub normalize: bool,
    /// Feature set to use
    pub feature_set: FeatureSet,
}

impl Default for PatternConfig {
    fn default() -> Self {
        Self {
            lookback: 20,
            forward: 5,
            normalize: true,
            feature_set: FeatureSet::Full,
        }
    }
}

/// Feature set options
#[derive(Debug, Clone, Copy)]
pub enum FeatureSet {
    /// Basic features only (returns, volatility)
    Basic,
    /// Full feature set
    Full,
    /// Technical indicators focused
    Technical,
}

/// Pattern builder for constructing market patterns
pub struct PatternBuilder {
    config: PatternConfig,
}

impl PatternBuilder {
    /// Create a new pattern builder with default config
    pub fn new(lookback: usize) -> Self {
        Self {
            config: PatternConfig {
                lookback,
                ..Default::default()
            },
        }
    }

    /// Create with custom config
    pub fn with_config(config: PatternConfig) -> Self {
        Self { config }
    }

    /// Get the pattern dimension for this configuration
    pub fn pattern_dim(&self) -> usize {
        match self.config.feature_set {
            FeatureSet::Basic => 6,
            FeatureSet::Full => 15,
            FeatureSet::Technical => 12,
        }
    }

    /// Build patterns from OHLCV series
    pub fn build_patterns(&self, series: &OHLCVSeries) -> Vec<Pattern> {
        let min_required = self.config.lookback + self.config.forward;

        if series.len() < min_required {
            return Vec::new();
        }

        let mut patterns = Vec::new();

        for i in self.config.lookback..(series.len() - self.config.forward) {
            let window = series.slice(i - self.config.lookback, i);
            let features = self.extract_features(&window);

            // Calculate label (future return)
            let current_close = series.data[i - 1].close;
            let future_close = series.data[i + self.config.forward - 1].close;
            let label = (future_close - current_close) / current_close;

            let timestamp = series.data[i - 1].timestamp;
            let mut pattern = Pattern::with_label(features, timestamp, label);

            if self.config.normalize {
                pattern.normalize();
            }

            patterns.push(pattern);
        }

        patterns
    }

    /// Build a single pattern from the most recent data
    pub fn build_current_pattern(&self, series: &OHLCVSeries) -> Option<Pattern> {
        if series.len() < self.config.lookback {
            return None;
        }

        let window = series.tail(self.config.lookback);
        let features = self.extract_features(&window);
        let timestamp = series.data.last()?.timestamp;

        let mut pattern = Pattern::new(features, timestamp);

        if self.config.normalize {
            pattern.normalize();
        }

        Some(pattern)
    }

    /// Extract features from a window of data
    fn extract_features(&self, window: &OHLCVSeries) -> Array1<f64> {
        match self.config.feature_set {
            FeatureSet::Basic => self.extract_basic_features(window),
            FeatureSet::Full => self.extract_full_features(window),
            FeatureSet::Technical => self.extract_technical_features(window),
        }
    }

    /// Extract basic features
    fn extract_basic_features(&self, window: &OHLCVSeries) -> Array1<f64> {
        let returns = window.returns();
        let closes = window.closes();
        let volumes = window.volumes();

        let mut features = Vec::with_capacity(6);

        // Return statistics
        features.push(stats::mean(&returns));
        features.push(stats::std(&returns));

        // Trend
        let first_close = closes.first().copied().unwrap_or(0.0);
        let last_close = closes.last().copied().unwrap_or(0.0);
        features.push(if first_close > 0.0 {
            (last_close - first_close) / first_close
        } else {
            0.0
        });

        // Volume ratio
        let vol_mean = stats::mean(&volumes);
        let last_vol = volumes.last().copied().unwrap_or(0.0);
        features.push(if vol_mean > 0.0 {
            last_vol / vol_mean
        } else {
            1.0
        });

        // Volatility
        features.push(stats::std(&returns) * (252.0_f64).sqrt());

        // Range ratio
        let ranges: Vec<f64> = window.data.iter().map(|c| c.relative_range()).collect();
        features.push(stats::mean(&ranges));

        Array1::from_vec(features)
    }

    /// Extract full feature set
    fn extract_full_features(&self, window: &OHLCVSeries) -> Array1<f64> {
        let returns = window.returns();
        let closes = window.closes();
        let volumes = window.volumes();

        let mut features = Vec::with_capacity(15);

        // Return statistics (4 features)
        features.push(stats::mean(&returns));
        features.push(stats::std(&returns));
        features.push(stats::skewness(&returns));
        features.push(stats::kurtosis(&returns));

        // Trend features (3 features)
        let n = closes.len();
        if n > 0 {
            let first_close = closes[0];
            let last_close = closes[n - 1];
            let mid_close = closes[n / 2];

            features.push((last_close - first_close) / first_close);
            features.push((last_close - mid_close) / mid_close);

            // Linear regression slope approximation
            let x_mean = (n - 1) as f64 / 2.0;
            let y_mean = stats::mean(&closes);
            let mut num = 0.0;
            let mut den = 0.0;
            for (i, &y) in closes.iter().enumerate() {
                let x = i as f64;
                num += (x - x_mean) * (y - y_mean);
                den += (x - x_mean).powi(2);
            }
            let slope = if den > 0.0 { num / den } else { 0.0 };
            features.push(slope / y_mean); // Normalized slope
        } else {
            features.extend([0.0, 0.0, 0.0]);
        }

        // Volatility features (2 features)
        let annualized_vol = stats::std(&returns) * (252.0_f64).sqrt();
        features.push(annualized_vol);

        // Volatility of volatility (if enough data)
        if returns.len() >= 10 {
            let vol_5: Vec<f64> = returns.windows(5).map(|w| stats::std(w)).collect();
            features.push(stats::std(&vol_5));
        } else {
            features.push(0.0);
        }

        // Volume features (3 features)
        let vol_mean = stats::mean(&volumes);
        let last_vol = volumes.last().copied().unwrap_or(0.0);
        features.push(if vol_mean > 0.0 {
            last_vol / vol_mean
        } else {
            1.0
        });
        features.push(stats::std(&volumes) / (vol_mean + 1e-10));

        // Volume trend
        if volumes.len() > 1 {
            let first_vol = volumes[0];
            let last_vol = volumes[volumes.len() - 1];
            features.push(if first_vol > 0.0 {
                (last_vol - first_vol) / first_vol
            } else {
                0.0
            });
        } else {
            features.push(0.0);
        }

        // Price structure features (3 features)
        let ranges: Vec<f64> = window.data.iter().map(|c| c.relative_range()).collect();
        features.push(stats::mean(&ranges));

        // Close position in recent range
        let high_max = stats::max(&window.data.iter().map(|c| c.high).collect::<Vec<_>>());
        let low_min = stats::min(&window.data.iter().map(|c| c.low).collect::<Vec<_>>());
        let last_close = closes.last().copied().unwrap_or(0.0);
        features.push(if high_max > low_min {
            (last_close - low_min) / (high_max - low_min)
        } else {
            0.5
        });

        // Bullish ratio
        let bullish_count = window.data.iter().filter(|c| c.is_bullish()).count() as f64;
        features.push(bullish_count / window.len() as f64);

        Array1::from_vec(features)
    }

    /// Extract technical indicator focused features
    fn extract_technical_features(&self, window: &OHLCVSeries) -> Array1<f64> {
        let closes = window.closes();
        let _returns = window.returns();

        let mut features = Vec::with_capacity(12);

        // SMA ratios (3 features)
        let n = closes.len();
        if n >= 20 {
            let sma_5 = stats::mean(&closes[(n - 5)..]);
            let sma_10 = stats::mean(&closes[(n - 10)..]);
            let sma_20 = stats::mean(&closes);
            let last = closes[n - 1];

            features.push(last / sma_5 - 1.0);
            features.push(last / sma_10 - 1.0);
            features.push(sma_5 / sma_20 - 1.0);
        } else {
            features.extend([0.0, 0.0, 0.0]);
        }

        // RSI approximation (1 feature)
        let returns = window.returns();
        if !returns.is_empty() {
            let gains: f64 = returns.iter().filter(|&&r| r > 0.0).sum();
            let losses: f64 = returns.iter().filter(|&&r| r < 0.0).map(|r| r.abs()).sum();
            let rs = if losses > 0.0 {
                gains / losses
            } else {
                100.0
            };
            features.push(100.0 - 100.0 / (1.0 + rs));
        } else {
            features.push(50.0);
        }

        // Bollinger Band position (2 features)
        let sma = stats::mean(&closes);
        let std = stats::std(&closes);
        let last = closes.last().copied().unwrap_or(0.0);
        if std > 0.0 {
            let bb_position = (last - sma) / (2.0 * std);
            features.push(bb_position);
            features.push((std / sma).min(1.0)); // Bandwidth
        } else {
            features.extend([0.0, 0.0]);
        }

        // ATR ratio (1 feature)
        let atr = window.atr(14);
        if !atr.is_empty() && last > 0.0 {
            features.push(atr.last().copied().unwrap_or(0.0) / last);
        } else {
            features.push(0.0);
        }

        // MACD approximation (2 features)
        if n >= 26 {
            let ema_12 = stats::mean(&closes[(n - 12)..]);
            let ema_26 = stats::mean(&closes[(n - 26)..]);
            let macd = ema_12 - ema_26;
            features.push(macd / last);
            features.push(if macd > 0.0 { 1.0 } else { -1.0 });
        } else {
            features.extend([0.0, 0.0]);
        }

        // Momentum (3 features)
        if n >= 10 {
            features.push((closes[n - 1] - closes[n - 5]) / closes[n - 5]);
            features.push((closes[n - 1] - closes[n - 10]) / closes[n - 10]);
            // Rate of change
            let roc_values: Vec<f64> = (1..n)
                .map(|i| (closes[i] - closes[i - 1]) / closes[i - 1])
                .collect();
            features.push(stats::mean(&roc_values));
        } else {
            features.extend([0.0, 0.0, 0.0]);
        }

        Array1::from_vec(features)
    }
}

/// Convert patterns to matrix format
pub fn patterns_to_matrix(patterns: &[Pattern]) -> (Array2<f64>, Array1<f64>) {
    if patterns.is_empty() {
        return (Array2::zeros((0, 0)), Array1::zeros(0));
    }

    let n = patterns.len();
    let dim = patterns[0].dim();

    let mut features = Array2::zeros((n, dim));
    let mut labels = Array1::zeros(n);

    for (i, pattern) in patterns.iter().enumerate() {
        for (j, &val) in pattern.features.iter().enumerate() {
            features[[i, j]] = val;
        }
        labels[i] = pattern.label.unwrap_or(0.0);
    }

    (features, labels)
}

/// Normalize features using z-score
pub fn normalize_features(features: &mut Array2<f64>) {
    let (n, dim) = features.dim();
    if n == 0 {
        return;
    }

    for j in 0..dim {
        let col: Vec<f64> = (0..n).map(|i| features[[i, j]]).collect();
        let mean = stats::mean(&col);
        let std = stats::std(&col);

        if std > 1e-10 {
            for i in 0..n {
                features[[i, j]] = (features[[i, j]] - mean) / std;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::OHLCV;
    use chrono::Utc;

    fn create_test_series(n: usize) -> OHLCVSeries {
        let mut data = Vec::new();
        let mut price = 100.0;

        for i in 0..n {
            let change = (i as f64 * 0.1).sin() * 2.0;
            price += change;
            data.push(OHLCV::new(
                Utc::now(),
                price,
                price + 1.0,
                price - 1.0,
                price + 0.5,
                1000.0 + (i as f64 * 10.0),
            ));
        }

        OHLCVSeries::with_data("TEST".to_string(), "1h".to_string(), data)
    }

    #[test]
    fn test_pattern_builder() {
        let series = create_test_series(100);
        let builder = PatternBuilder::new(20);
        let patterns = builder.build_patterns(&series);

        assert!(!patterns.is_empty());
        assert_eq!(patterns[0].dim(), builder.pattern_dim());
    }

    #[test]
    fn test_pattern_similarity() {
        let p1 = Pattern::new(Array1::from_vec(vec![1.0, 0.0, 0.0]), Utc::now());
        let p2 = Pattern::new(Array1::from_vec(vec![1.0, 0.0, 0.0]), Utc::now());
        let p3 = Pattern::new(Array1::from_vec(vec![0.0, 1.0, 0.0]), Utc::now());

        assert!((p1.similarity(&p2) - 1.0).abs() < 1e-10);
        assert!(p1.similarity(&p3).abs() < 1e-10);
    }

    #[test]
    fn test_stats() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((stats::mean(&data) - 3.0).abs() < 1e-10);
        assert!((stats::std(&data) - 1.5811388300841898).abs() < 1e-10);
    }
}
