//! Feature engineering engine
//!
//! Combines multiple features for anomaly detection

use crate::data::OHLCVSeries;
use ndarray::Array2;
use super::indicators::*;

/// Configuration for feature engineering
#[derive(Clone, Debug)]
pub struct FeatureConfig {
    /// Window size for rolling calculations
    pub window: usize,
    /// RSI period
    pub rsi_period: usize,
    /// ATR period
    pub atr_period: usize,
    /// Include volume features
    pub include_volume: bool,
    /// Include momentum features
    pub include_momentum: bool,
    /// Include volatility features
    pub include_volatility: bool,
}

impl Default for FeatureConfig {
    fn default() -> Self {
        Self {
            window: 20,
            rsi_period: 14,
            atr_period: 14,
            include_volume: true,
            include_momentum: true,
            include_volatility: true,
        }
    }
}

/// Computed features from OHLCV data
#[derive(Clone, Debug)]
pub struct Features {
    /// Feature names
    pub names: Vec<String>,
    /// Feature matrix (rows = time, columns = features)
    pub data: Array2<f64>,
    /// Number of valid rows (after dropping NaN from rolling calculations)
    pub valid_from: usize,
}

impl Features {
    /// Get the number of features
    pub fn num_features(&self) -> usize {
        self.names.len()
    }

    /// Get the number of time points
    pub fn len(&self) -> usize {
        self.data.nrows()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get a slice of valid data (after warmup period)
    pub fn valid_data(&self) -> Array2<f64> {
        self.data.slice(ndarray::s![self.valid_from.., ..]).to_owned()
    }

    /// Get feature by name
    pub fn get(&self, name: &str) -> Option<Vec<f64>> {
        let idx = self.names.iter().position(|n| n == name)?;
        Some(self.data.column(idx).to_vec())
    }
}

/// Feature engineering engine
pub struct FeatureEngine {
    config: FeatureConfig,
}

impl FeatureEngine {
    /// Create a new feature engine with default config
    pub fn new() -> Self {
        Self {
            config: FeatureConfig::default(),
        }
    }

    /// Create a new feature engine with custom config
    pub fn with_config(config: FeatureConfig) -> Self {
        Self { config }
    }

    /// Compute features from OHLCV series
    pub fn compute(&self, series: &OHLCVSeries) -> Features {
        let n = series.len();
        let closes = series.closes();
        let highs = series.highs();
        let lows = series.lows();
        let volumes = series.volumes();

        let mut feature_names = Vec::new();
        let mut feature_cols: Vec<Vec<f64>> = Vec::new();

        // 1. Returns
        let mut rets = vec![0.0];
        rets.extend(returns(&closes));
        feature_names.push("return".to_string());
        feature_cols.push(rets.clone());

        // 2. Absolute returns
        let abs_rets: Vec<f64> = rets.iter().map(|r| r.abs()).collect();
        feature_names.push("abs_return".to_string());
        feature_cols.push(abs_rets);

        // 3. Log returns
        let mut log_rets = vec![0.0];
        log_rets.extend(log_returns(&closes));
        feature_names.push("log_return".to_string());
        feature_cols.push(log_rets);

        if self.config.include_volatility {
            // 4. Rolling volatility (short)
            let vol_short = volatility(&closes, 5);
            feature_names.push("volatility_5".to_string());
            feature_cols.push(vol_short.clone());

            // 5. Rolling volatility (long)
            let vol_long = volatility(&closes, self.config.window);
            feature_names.push(format!("volatility_{}", self.config.window));
            feature_cols.push(vol_long.clone());

            // 6. Volatility ratio
            let vol_ratio: Vec<f64> = vol_short
                .iter()
                .zip(vol_long.iter())
                .map(|(&s, &l)| {
                    if s.is_nan() || l.is_nan() || l == 0.0 {
                        f64::NAN
                    } else {
                        s / l
                    }
                })
                .collect();
            feature_names.push("volatility_ratio".to_string());
            feature_cols.push(vol_ratio);

            // 7. Bollinger Band %B
            let bb = bollinger_bands(&closes, self.config.window, 2.0);
            feature_names.push("bb_percent_b".to_string());
            feature_cols.push(bb.percent_b);

            // 8. Bollinger Bandwidth
            feature_names.push("bb_bandwidth".to_string());
            feature_cols.push(bb.bandwidth);
        }

        if self.config.include_volume {
            // 9. Volume Z-score
            let vol_mean = sma(&volumes, self.config.window);
            let vol_std = crate::data::rolling_std(&volumes, self.config.window);
            let vol_zscore: Vec<f64> = volumes
                .iter()
                .zip(vol_mean.iter().zip(vol_std.iter()))
                .map(|(&v, (&m, &s))| {
                    if m.is_nan() || s.is_nan() || s == 0.0 {
                        f64::NAN
                    } else {
                        (v - m) / s
                    }
                })
                .collect();
            feature_names.push("volume_zscore".to_string());
            feature_cols.push(vol_zscore);

            // 10. Volume ratio
            let vol_ratio = volume_ratio(&volumes, self.config.window);
            feature_names.push("volume_ratio".to_string());
            feature_cols.push(vol_ratio);
        }

        if self.config.include_momentum {
            // 11. RSI
            let rsi_vals = rsi(&closes, self.config.rsi_period);
            feature_names.push("rsi".to_string());
            feature_cols.push(rsi_vals);

            // 12. MACD histogram
            let macd_vals = macd(&closes, 12, 26, 9);
            feature_names.push("macd_histogram".to_string());
            feature_cols.push(macd_vals.histogram);

            // 13. Price distance from SMA
            let sma_vals = sma(&closes, self.config.window);
            let price_sma_dist: Vec<f64> = closes
                .iter()
                .zip(sma_vals.iter())
                .map(|(&p, &s)| {
                    if s.is_nan() || s == 0.0 {
                        f64::NAN
                    } else {
                        (p - s) / s
                    }
                })
                .collect();
            feature_names.push("price_sma_distance".to_string());
            feature_cols.push(price_sma_dist);
        }

        // 14. Range (High - Low) / Close
        let range: Vec<f64> = highs
            .iter()
            .zip(lows.iter().zip(closes.iter()))
            .map(|(&h, (&l, &c))| if c > 0.0 { (h - l) / c } else { 0.0 })
            .collect();
        feature_names.push("range".to_string());
        feature_cols.push(range.clone());

        // 15. Range ratio
        let range_rat = range_ratio(&highs, &lows, &closes, self.config.window);
        feature_names.push("range_ratio".to_string());
        feature_cols.push(range_rat);

        // 16. Close position within range
        let close_pos: Vec<f64> = series.data.iter().map(|c| c.close_position()).collect();
        feature_names.push("close_position".to_string());
        feature_cols.push(close_pos);

        // 17. Skewness
        let skew = rolling_skewness(&rets, self.config.window);
        feature_names.push("skewness".to_string());
        feature_cols.push(skew);

        // 18. Kurtosis
        let kurt = rolling_kurtosis(&rets, self.config.window);
        feature_names.push("kurtosis".to_string());
        feature_cols.push(kurt);

        // Create feature matrix
        let num_features = feature_cols.len();
        let mut data = Array2::zeros((n, num_features));

        for (j, col) in feature_cols.iter().enumerate() {
            for (i, &val) in col.iter().enumerate() {
                data[[i, j]] = val;
            }
        }

        // Find valid_from (first row with no NaN)
        let valid_from = (0..n)
            .find(|&i| {
                (0..num_features).all(|j| !data[[i, j]].is_nan())
            })
            .unwrap_or(n);

        Features {
            names: feature_names,
            data,
            valid_from,
        }
    }

    /// Compute a minimal set of features for real-time detection
    pub fn compute_realtime(&self, closes: &[f64], volumes: &[f64]) -> Vec<f64> {
        let mut features = Vec::new();

        // Return
        if closes.len() >= 2 {
            let ret = (closes[closes.len() - 1] - closes[closes.len() - 2])
                / closes[closes.len() - 2];
            features.push(ret);
            features.push(ret.abs());
        } else {
            features.push(0.0);
            features.push(0.0);
        }

        // Volatility
        if closes.len() >= 6 {
            let rets: Vec<f64> = closes.windows(2).map(|w| (w[1] - w[0]) / w[0]).collect();
            let last_5: &[f64] = &rets[rets.len().saturating_sub(5)..];
            let mean: f64 = last_5.iter().sum::<f64>() / last_5.len() as f64;
            let variance: f64 = last_5.iter().map(|r| (r - mean).powi(2)).sum::<f64>()
                / last_5.len() as f64;
            features.push(variance.sqrt());
        } else {
            features.push(0.0);
        }

        // Volume ratio
        if volumes.len() >= self.config.window {
            let recent_avg: f64 = volumes[volumes.len() - self.config.window..].iter().sum::<f64>()
                / self.config.window as f64;
            if recent_avg > 0.0 {
                features.push(volumes.last().unwrap_or(&0.0) / recent_avg);
            } else {
                features.push(1.0);
            }
        } else {
            features.push(1.0);
        }

        features
    }
}

impl Default for FeatureEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::OHLCV;
    use chrono::Utc;

    fn create_test_series(n: usize) -> OHLCVSeries {
        let mut data = Vec::with_capacity(n);
        let mut price = 100.0;

        for i in 0..n {
            let change = (i as f64 * 0.1).sin() * 2.0;
            price += change;

            data.push(OHLCV::new(
                Utc::now(),
                price - 1.0,
                price + 1.0,
                price - 1.5,
                price,
                1000.0 + (i as f64 * 10.0),
            ));
        }

        OHLCVSeries::with_data("TEST".to_string(), "60".to_string(), data)
    }

    #[test]
    fn test_feature_engine() {
        let series = create_test_series(100);
        let engine = FeatureEngine::new();
        let features = engine.compute(&series);

        assert!(!features.is_empty());
        assert!(features.num_features() > 10);
        assert!(features.valid_from < features.len());
    }

    #[test]
    fn test_feature_names() {
        let series = create_test_series(50);
        let engine = FeatureEngine::new();
        let features = engine.compute(&series);

        assert!(features.names.contains(&"return".to_string()));
        assert!(features.names.contains(&"volatility_5".to_string()));
        assert!(features.names.contains(&"rsi".to_string()));
    }
}
