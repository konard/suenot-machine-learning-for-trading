//! Feature extraction from raw market data
//!
//! Extracts task-agnostic features from OHLCV data including:
//! - Price-based features (returns, momentum, volatility)
//! - Volume-based features (volume ratio, VWAP deviation)
//! - Technical indicators (RSI, Bollinger Bands, MACD)
//! - Microstructure features (spread, imbalance)

use super::types::Kline;
use ndarray::Array1;
use serde::{Deserialize, Serialize};

/// Configuration for feature extraction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureConfig {
    /// Lookback window for returns
    pub returns_window: usize,
    /// Window for volatility calculation
    pub volatility_window: usize,
    /// RSI period
    pub rsi_period: usize,
    /// Short MA period
    pub short_ma: usize,
    /// Long MA period
    pub long_ma: usize,
    /// Bollinger Band standard deviations
    pub bb_std: f64,
    /// Whether to normalize features
    pub normalize: bool,
}

impl Default for FeatureConfig {
    fn default() -> Self {
        Self {
            returns_window: 5,
            volatility_window: 20,
            rsi_period: 14,
            short_ma: 10,
            long_ma: 30,
            bb_std: 2.0,
            normalize: true,
        }
    }
}

/// Extracted market features for a single time step
#[derive(Debug, Clone)]
pub struct MarketFeatures {
    /// Feature vector
    pub features: Array1<f64>,
    /// Feature names
    pub names: Vec<String>,
    /// Timestamp index
    pub timestamp_idx: usize,
}

impl MarketFeatures {
    /// Get the number of features
    pub fn dim(&self) -> usize {
        self.features.len()
    }
}

/// Feature extractor that processes raw kline data
pub struct FeatureExtractor {
    config: FeatureConfig,
}

impl FeatureExtractor {
    /// Create a new feature extractor
    pub fn new(config: FeatureConfig) -> Self {
        Self { config }
    }

    /// Extract features from a series of klines for a single time step
    ///
    /// Requires at least `max(long_ma, volatility_window) + 1` klines.
    pub fn extract(&self, klines: &[Kline], idx: usize) -> Option<MarketFeatures> {
        let min_lookback = self.config.long_ma.max(self.config.volatility_window);
        if idx < min_lookback || idx >= klines.len() {
            return None;
        }

        let mut features = Vec::new();
        let mut names = Vec::new();

        // 1. Returns features
        let returns = self.compute_returns(klines, idx);
        features.push(returns.0);
        names.push("return_1d".to_string());
        features.push(returns.1);
        names.push(format!("return_{}d", self.config.returns_window));

        // 2. Momentum features
        let momentum = self.compute_momentum(klines, idx);
        features.push(momentum.0);
        names.push("rsi".to_string());
        features.push(momentum.1);
        names.push("macd_signal".to_string());
        features.push(momentum.2);
        names.push("ma_crossover".to_string());

        // 3. Volatility features
        let volatility = self.compute_volatility(klines, idx);
        features.push(volatility.0);
        names.push("realized_vol".to_string());
        features.push(volatility.1);
        names.push("bb_position".to_string());
        features.push(volatility.2);
        names.push("atr_ratio".to_string());

        // 4. Volume features
        let volume = self.compute_volume_features(klines, idx);
        features.push(volume.0);
        names.push("volume_ratio".to_string());
        features.push(volume.1);
        names.push("volume_trend".to_string());

        // 5. Candle pattern features
        let patterns = self.compute_candle_features(klines, idx);
        features.push(patterns.0);
        names.push("body_ratio".to_string());
        features.push(patterns.1);
        names.push("upper_shadow".to_string());
        features.push(patterns.2);
        names.push("lower_shadow".to_string());

        // 6. Range features
        let range = self.compute_range_features(klines, idx);
        features.push(range.0);
        names.push("range_pct".to_string());
        features.push(range.1);
        names.push("close_position".to_string());

        // 7. Trend strength
        let trend = self.compute_trend_features(klines, idx);
        features.push(trend.0);
        names.push("trend_strength".to_string());
        features.push(trend.1);
        names.push("trend_consistency".to_string());

        // 8. Higher-order features
        features.push(self.compute_skewness(klines, idx));
        names.push("return_skewness".to_string());
        features.push(self.compute_kurtosis(klines, idx));
        names.push("return_kurtosis".to_string());

        let feat_array = if self.config.normalize {
            let arr = Array1::from(features);
            self.normalize_features(&arr)
        } else {
            Array1::from(features)
        };

        Some(MarketFeatures {
            features: feat_array,
            names,
            timestamp_idx: idx,
        })
    }

    /// Extract features for all valid time steps
    pub fn extract_all(&self, klines: &[Kline]) -> Vec<MarketFeatures> {
        let min_lookback = self.config.long_ma.max(self.config.volatility_window);
        (min_lookback..klines.len())
            .filter_map(|idx| self.extract(klines, idx))
            .collect()
    }

    /// Get the expected feature dimension
    pub fn feature_dim(&self) -> usize {
        19 // Total number of features extracted
    }

    // --- Private helper methods ---

    fn compute_returns(&self, klines: &[Kline], idx: usize) -> (f64, f64) {
        let ret_1d = if klines[idx - 1].close > 0.0 {
            (klines[idx].close - klines[idx - 1].close) / klines[idx - 1].close
        } else {
            0.0
        };

        let window = self.config.returns_window.min(idx);
        let ret_nd = if klines[idx - window].close > 0.0 {
            (klines[idx].close - klines[idx - window].close) / klines[idx - window].close
        } else {
            0.0
        };

        (ret_1d, ret_nd)
    }

    fn compute_momentum(&self, klines: &[Kline], idx: usize) -> (f64, f64, f64) {
        // RSI
        let rsi_period = self.config.rsi_period.min(idx);
        let mut gains = 0.0;
        let mut losses = 0.0;
        for i in (idx - rsi_period + 1)..=idx {
            let change = klines[i].close - klines[i - 1].close;
            if change > 0.0 {
                gains += change;
            } else {
                losses += change.abs();
            }
        }
        let avg_gain = gains / rsi_period as f64;
        let avg_loss = losses / rsi_period as f64;
        let rsi = if avg_loss > 0.0 {
            100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
        } else {
            100.0
        };

        // MACD signal (simplified)
        let short_ma = self.moving_average(klines, idx, self.config.short_ma);
        let long_ma = self.moving_average(klines, idx, self.config.long_ma);
        let macd = if long_ma > 0.0 {
            (short_ma - long_ma) / long_ma
        } else {
            0.0
        };

        // MA crossover signal
        let prev_short_ma = if idx > 0 {
            self.moving_average(klines, idx - 1, self.config.short_ma)
        } else {
            short_ma
        };
        let prev_long_ma = if idx > 0 {
            self.moving_average(klines, idx - 1, self.config.long_ma)
        } else {
            long_ma
        };
        let crossover = if short_ma > long_ma && prev_short_ma <= prev_long_ma {
            1.0
        } else if short_ma < long_ma && prev_short_ma >= prev_long_ma {
            -1.0
        } else {
            0.0
        };

        (rsi / 100.0, macd, crossover) // Normalize RSI to [0, 1]
    }

    fn compute_volatility(&self, klines: &[Kline], idx: usize) -> (f64, f64, f64) {
        let window = self.config.volatility_window.min(idx);

        // Realized volatility
        let returns: Vec<f64> = (idx - window + 1..=idx)
            .map(|i| {
                if klines[i - 1].close > 0.0 {
                    (klines[i].close / klines[i - 1].close).ln()
                } else {
                    0.0
                }
            })
            .collect();

        let mean_ret: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance: f64 = returns.iter().map(|r| (r - mean_ret).powi(2)).sum::<f64>()
            / (returns.len() - 1).max(1) as f64;
        let realized_vol = variance.sqrt();

        // Bollinger Band position
        let ma = self.moving_average(klines, idx, window);
        let std_dev = realized_vol * klines[idx].close;
        let bb_upper = ma + self.config.bb_std * std_dev;
        let bb_lower = ma - self.config.bb_std * std_dev;
        let bb_width = bb_upper - bb_lower;
        let bb_position = if bb_width > 0.0 {
            (klines[idx].close - bb_lower) / bb_width
        } else {
            0.5
        };

        // ATR ratio
        let atr: f64 = (idx - window.min(14) + 1..=idx)
            .map(|i| {
                let tr = (klines[i].high - klines[i].low)
                    .max((klines[i].high - klines[i - 1].close).abs())
                    .max((klines[i].low - klines[i - 1].close).abs());
                tr
            })
            .sum::<f64>()
            / window.min(14) as f64;

        let atr_ratio = if klines[idx].close > 0.0 {
            atr / klines[idx].close
        } else {
            0.0
        };

        (realized_vol, bb_position, atr_ratio)
    }

    fn compute_volume_features(&self, klines: &[Kline], idx: usize) -> (f64, f64) {
        let window = 20.min(idx);
        let avg_volume: f64 = klines[idx - window..idx]
            .iter()
            .map(|k| k.volume)
            .sum::<f64>()
            / window as f64;

        let volume_ratio = if avg_volume > 0.0 {
            klines[idx].volume / avg_volume
        } else {
            1.0
        };

        // Volume trend (short-term vs long-term)
        let short_vol: f64 = klines[idx.saturating_sub(5)..idx]
            .iter()
            .map(|k| k.volume)
            .sum::<f64>()
            / 5.0_f64.min(idx as f64).max(1.0);

        let volume_trend = if avg_volume > 0.0 {
            (short_vol - avg_volume) / avg_volume
        } else {
            0.0
        };

        (volume_ratio, volume_trend)
    }

    fn compute_candle_features(&self, klines: &[Kline], idx: usize) -> (f64, f64, f64) {
        let k = &klines[idx];
        let range = k.high - k.low;

        if range <= 0.0 {
            return (0.0, 0.0, 0.0);
        }

        let body = (k.close - k.open).abs();
        let body_ratio = body / range;

        let upper_shadow = if k.close > k.open {
            (k.high - k.close) / range
        } else {
            (k.high - k.open) / range
        };

        let lower_shadow = if k.close > k.open {
            (k.open - k.low) / range
        } else {
            (k.close - k.low) / range
        };

        (body_ratio, upper_shadow, lower_shadow)
    }

    fn compute_range_features(&self, klines: &[Kline], idx: usize) -> (f64, f64) {
        let k = &klines[idx];
        let range_pct = k.range_pct();

        let close_position = if k.high > k.low {
            (k.close - k.low) / (k.high - k.low)
        } else {
            0.5
        };

        (range_pct, close_position)
    }

    fn compute_trend_features(&self, klines: &[Kline], idx: usize) -> (f64, f64) {
        let window = 10.min(idx);

        // Trend strength: linear regression slope
        let prices: Vec<f64> = klines[idx - window..=idx]
            .iter()
            .map(|k| k.close)
            .collect();

        let n = prices.len() as f64;
        let x_mean = (n - 1.0) / 2.0;
        let y_mean: f64 = prices.iter().sum::<f64>() / n;

        let mut numerator = 0.0;
        let mut denominator = 0.0;
        for (i, &p) in prices.iter().enumerate() {
            let x = i as f64 - x_mean;
            let y = p - y_mean;
            numerator += x * y;
            denominator += x * x;
        }

        let slope = if denominator > 0.0 {
            numerator / denominator
        } else {
            0.0
        };

        let trend_strength = if y_mean > 0.0 {
            slope / y_mean
        } else {
            0.0
        };

        // Trend consistency: proportion of positive returns
        let consistency = klines[idx - window + 1..=idx]
            .iter()
            .zip(klines[idx - window..idx].iter())
            .filter(|(curr, prev)| curr.close > prev.close)
            .count() as f64
            / window as f64;

        (trend_strength, consistency)
    }

    fn compute_skewness(&self, klines: &[Kline], idx: usize) -> f64 {
        let window = 20.min(idx);
        let returns: Vec<f64> = (idx - window + 1..=idx)
            .map(|i| {
                if klines[i - 1].close > 0.0 {
                    (klines[i].close - klines[i - 1].close) / klines[i - 1].close
                } else {
                    0.0
                }
            })
            .collect();

        let n = returns.len() as f64;
        let mean: f64 = returns.iter().sum::<f64>() / n;
        let variance: f64 = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;
        let std_dev = variance.sqrt();

        if std_dev > 1e-10 {
            let skew: f64 = returns.iter().map(|r| ((r - mean) / std_dev).powi(3)).sum::<f64>() / n;
            skew
        } else {
            0.0
        }
    }

    fn compute_kurtosis(&self, klines: &[Kline], idx: usize) -> f64 {
        let window = 20.min(idx);
        let returns: Vec<f64> = (idx - window + 1..=idx)
            .map(|i| {
                if klines[i - 1].close > 0.0 {
                    (klines[i].close - klines[i - 1].close) / klines[i - 1].close
                } else {
                    0.0
                }
            })
            .collect();

        let n = returns.len() as f64;
        let mean: f64 = returns.iter().sum::<f64>() / n;
        let variance: f64 = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;
        let std_dev = variance.sqrt();

        if std_dev > 1e-10 {
            let kurt: f64 = returns.iter().map(|r| ((r - mean) / std_dev).powi(4)).sum::<f64>() / n;
            kurt - 3.0 // Excess kurtosis
        } else {
            0.0
        }
    }

    fn moving_average(&self, klines: &[Kline], idx: usize, window: usize) -> f64 {
        let start = idx.saturating_sub(window - 1);
        let slice = &klines[start..=idx];
        slice.iter().map(|k| k.close).sum::<f64>() / slice.len() as f64
    }

    fn normalize_features(&self, features: &Array1<f64>) -> Array1<f64> {
        let mean = features.mean().unwrap_or(0.0);
        let std = features.std(0.0);
        if std > 1e-10 {
            features.mapv(|v| (v - mean) / std)
        } else {
            features.clone()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn make_test_klines(n: usize) -> Vec<Kline> {
        (0..n)
            .map(|i| {
                let base_price = 100.0 + (i as f64) * 0.5;
                Kline {
                    timestamp: Utc::now(),
                    open: base_price,
                    high: base_price + 2.0,
                    low: base_price - 1.0,
                    close: base_price + 1.0,
                    volume: 1000.0 + (i as f64) * 10.0,
                    turnover: base_price * 1000.0,
                }
            })
            .collect()
    }

    #[test]
    fn test_feature_extraction() {
        let klines = make_test_klines(50);
        let extractor = FeatureExtractor::new(FeatureConfig::default());
        let features = extractor.extract(&klines, 35);

        assert!(features.is_some());
        let f = features.unwrap();
        assert_eq!(f.dim(), 19);
        assert_eq!(f.names.len(), 19);
    }

    #[test]
    fn test_extract_all() {
        let klines = make_test_klines(50);
        let extractor = FeatureExtractor::new(FeatureConfig::default());
        let all_features = extractor.extract_all(&klines);

        assert!(!all_features.is_empty());
        assert!(all_features.len() <= 50);
    }
}
