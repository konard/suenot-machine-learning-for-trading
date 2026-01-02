//! Feature engineering engine for EBM trading

use ndarray::{Array1, Array2};

use crate::data::Candle;
use super::indicators::*;

/// Feature configuration
#[derive(Debug, Clone)]
pub struct FeatureConfig {
    /// Short-term lookback period
    pub short_period: usize,
    /// Medium-term lookback period
    pub medium_period: usize,
    /// Long-term lookback period
    pub long_period: usize,
    /// Whether to include volume features
    pub include_volume: bool,
    /// Whether to include higher moments (skewness, kurtosis)
    pub include_moments: bool,
    /// Whether to include technical indicators
    pub include_technicals: bool,
}

impl Default for FeatureConfig {
    fn default() -> Self {
        Self {
            short_period: 5,
            medium_period: 20,
            long_period: 50,
            include_volume: true,
            include_moments: true,
            include_technicals: true,
        }
    }
}

/// Feature engine for extracting features from OHLCV data
#[derive(Debug, Clone)]
pub struct FeatureEngine {
    /// Feature configuration
    pub config: FeatureConfig,
    /// Feature names
    pub feature_names: Vec<String>,
}

impl Default for FeatureEngine {
    fn default() -> Self {
        Self::new(FeatureConfig::default())
    }
}

impl FeatureEngine {
    /// Create a new feature engine
    pub fn new(config: FeatureConfig) -> Self {
        let feature_names = Self::generate_feature_names(&config);
        Self {
            config,
            feature_names,
        }
    }

    /// Generate feature names based on configuration
    fn generate_feature_names(config: &FeatureConfig) -> Vec<String> {
        let mut names = Vec::new();

        // Return features
        names.push("return".to_string());
        names.push("log_return".to_string());
        names.push("return_abs".to_string());

        // Volatility features
        names.push(format!("volatility_{}", config.short_period));
        names.push(format!("volatility_{}", config.medium_period));
        names.push(format!("volatility_{}", config.long_period));
        names.push("vol_ratio_short_medium".to_string());
        names.push("vol_ratio_medium_long".to_string());

        // Price features
        names.push("range".to_string());
        names.push("range_ratio".to_string());
        names.push("close_position".to_string());
        names.push("body_ratio".to_string());

        // Momentum features
        names.push(format!("momentum_{}", config.short_period));
        names.push(format!("momentum_{}", config.medium_period));
        names.push(format!("momentum_{}", config.long_period));

        if config.include_volume {
            names.push("volume_ratio".to_string());
            names.push("volume_zscore".to_string());
            names.push("vwap_ratio".to_string());
        }

        if config.include_moments {
            names.push("skewness".to_string());
            names.push("kurtosis".to_string());
        }

        if config.include_technicals {
            names.push("rsi".to_string());
            names.push("bollinger_position".to_string());
            names.push("atr_ratio".to_string());
        }

        names
    }

    /// Get the number of features
    pub fn n_features(&self) -> usize {
        self.feature_names.len()
    }

    /// Compute features from OHLCV candles
    pub fn compute(&self, candles: &[Candle]) -> Array2<f64> {
        let n = candles.len();
        let n_features = self.n_features();

        if n == 0 {
            return Array2::zeros((0, n_features));
        }

        // Extract price and volume arrays
        let closes: Vec<f64> = candles.iter().map(|c| c.close).collect();
        let opens: Vec<f64> = candles.iter().map(|c| c.open).collect();
        let highs: Vec<f64> = candles.iter().map(|c| c.high).collect();
        let lows: Vec<f64> = candles.iter().map(|c| c.low).collect();
        let volumes: Vec<f64> = candles.iter().map(|c| c.volume).collect();

        // Compute all features
        let rets = returns(&closes);
        let log_rets = log_returns(&closes);
        let ret_abs: Vec<f64> = rets.iter().map(|r| r.abs()).collect();

        // Volatility
        let vol_short = rolling_std(&rets, self.config.short_period);
        let vol_medium = rolling_std(&rets, self.config.medium_period);
        let vol_long = rolling_std(&rets, self.config.long_period);

        // Range features
        let ranges: Vec<f64> = candles.iter().map(|c| c.range() / c.close).collect();
        let range_sma = sma(&ranges, self.config.medium_period);

        // Close position
        let close_positions: Vec<f64> = candles.iter().map(|c| c.close_position()).collect();

        // Body ratio
        let body_ratios: Vec<f64> = candles
            .iter()
            .map(|c| {
                let range = c.range();
                if range > 1e-10 {
                    c.body().abs() / range
                } else {
                    0.0
                }
            })
            .collect();

        // Momentum
        let mom_short = momentum(&closes, self.config.short_period);
        let mom_medium = momentum(&closes, self.config.medium_period);
        let mom_long = momentum(&closes, self.config.long_period);

        // Volume features
        let vol_sma = sma(&volumes, self.config.medium_period);
        let vol_zscores = zscore(&volumes, self.config.medium_period);
        let vwap_ratios = vwap_ratio(&closes, &volumes, self.config.medium_period);

        // Moments
        let skewness_vals = rolling_skewness(&rets, self.config.medium_period);
        let kurtosis_vals = rolling_kurtosis(&rets, self.config.medium_period);

        // Technicals
        let rsi_vals = rsi(&closes, self.config.medium_period);
        let bb_positions = bollinger_position(&closes, self.config.medium_period, 2.0);
        let atr_vals = atr(&highs, &lows, &closes, self.config.medium_period);

        // Build feature matrix
        let mut features = Array2::zeros((n, n_features));

        for i in 0..n {
            let mut idx = 0;

            // Returns
            features[[i, idx]] = rets[i];
            idx += 1;
            features[[i, idx]] = log_rets[i];
            idx += 1;
            features[[i, idx]] = ret_abs[i];
            idx += 1;

            // Volatility
            features[[i, idx]] = vol_short[i];
            idx += 1;
            features[[i, idx]] = vol_medium[i];
            idx += 1;
            features[[i, idx]] = vol_long[i];
            idx += 1;

            // Volatility ratios
            let vol_ratio_sm = if vol_medium[i].is_nan() || vol_medium[i] < 1e-10 {
                1.0
            } else {
                vol_short[i] / vol_medium[i]
            };
            features[[i, idx]] = vol_ratio_sm;
            idx += 1;

            let vol_ratio_ml = if vol_long[i].is_nan() || vol_long[i] < 1e-10 {
                1.0
            } else {
                vol_medium[i] / vol_long[i]
            };
            features[[i, idx]] = vol_ratio_ml;
            idx += 1;

            // Range
            features[[i, idx]] = ranges[i];
            idx += 1;

            let range_ratio = if range_sma[i].is_nan() || range_sma[i] < 1e-10 {
                1.0
            } else {
                ranges[i] / range_sma[i]
            };
            features[[i, idx]] = range_ratio;
            idx += 1;

            features[[i, idx]] = close_positions[i];
            idx += 1;
            features[[i, idx]] = body_ratios[i];
            idx += 1;

            // Momentum
            features[[i, idx]] = mom_short[i];
            idx += 1;
            features[[i, idx]] = mom_medium[i];
            idx += 1;
            features[[i, idx]] = mom_long[i];
            idx += 1;

            // Volume features
            if self.config.include_volume {
                let vol_ratio = if vol_sma[i].is_nan() || vol_sma[i] < 1e-10 {
                    1.0
                } else {
                    volumes[i] / vol_sma[i]
                };
                features[[i, idx]] = vol_ratio;
                idx += 1;
                features[[i, idx]] = vol_zscores[i];
                idx += 1;
                features[[i, idx]] = vwap_ratios[i];
                idx += 1;
            }

            // Moments
            if self.config.include_moments {
                features[[i, idx]] = skewness_vals[i];
                idx += 1;
                features[[i, idx]] = kurtosis_vals[i];
                idx += 1;
            }

            // Technicals
            if self.config.include_technicals {
                features[[i, idx]] = (rsi_vals[i] - 50.0) / 50.0; // Normalize to [-1, 1]
                idx += 1;
                features[[i, idx]] = bb_positions[i];
                idx += 1;

                let atr_ratio = if closes[i] > 1e-10 {
                    atr_vals[i] / closes[i]
                } else {
                    0.0
                };
                features[[i, idx]] = atr_ratio;
            }
        }

        // Replace NaN with 0
        features.mapv_inplace(|v| if v.is_nan() { 0.0 } else { v });

        features
    }

    /// Compute features for a single candle (given history)
    pub fn compute_single(&self, candles: &[Candle]) -> Array1<f64> {
        let features = self.compute(candles);
        features.row(features.nrows() - 1).to_owned()
    }

    /// Get feature statistics from computed features
    pub fn feature_stats(&self, features: &Array2<f64>) -> FeatureStats {
        let n = features.nrows();
        let n_features = features.ncols();

        let mut stats = FeatureStats {
            names: self.feature_names.clone(),
            means: vec![0.0; n_features],
            stds: vec![0.0; n_features],
            mins: vec![f64::INFINITY; n_features],
            maxs: vec![f64::NEG_INFINITY; n_features],
        };

        for j in 0..n_features {
            let col: Vec<f64> = features.column(j).iter().cloned().collect();
            let valid: Vec<f64> = col.iter().filter(|v| v.is_finite()).cloned().collect();

            if !valid.is_empty() {
                stats.means[j] = valid.iter().sum::<f64>() / valid.len() as f64;
                stats.stds[j] = (valid
                    .iter()
                    .map(|v| (v - stats.means[j]).powi(2))
                    .sum::<f64>()
                    / valid.len() as f64)
                    .sqrt();
                stats.mins[j] = valid.iter().cloned().fold(f64::INFINITY, f64::min);
                stats.maxs[j] = valid.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            }
        }

        stats
    }
}

/// Feature statistics
#[derive(Debug, Clone)]
pub struct FeatureStats {
    /// Feature names
    pub names: Vec<String>,
    /// Mean values
    pub means: Vec<f64>,
    /// Standard deviations
    pub stds: Vec<f64>,
    /// Minimum values
    pub mins: Vec<f64>,
    /// Maximum values
    pub maxs: Vec<f64>,
}

impl FeatureStats {
    /// Print statistics
    pub fn print(&self) {
        println!("{:<30} {:>12} {:>12} {:>12} {:>12}", "Feature", "Mean", "Std", "Min", "Max");
        println!("{}", "-".repeat(82));

        for i in 0..self.names.len() {
            println!(
                "{:<30} {:>12.6} {:>12.6} {:>12.6} {:>12.6}",
                self.names[i], self.means[i], self.stds[i], self.mins[i], self.maxs[i]
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Candle;

    fn create_test_candles() -> Vec<Candle> {
        (0..100)
            .map(|i| {
                let base = 100.0 + (i as f64 * 0.1).sin() * 10.0;
                Candle::new(
                    i as i64 * 60000,
                    base,
                    base + 1.0,
                    base - 1.0,
                    base + 0.5,
                    1000.0 + (i as f64 * 0.05).cos() * 500.0,
                )
            })
            .collect()
    }

    #[test]
    fn test_feature_engine() {
        let engine = FeatureEngine::default();
        let candles = create_test_candles();
        let features = engine.compute(&candles);

        assert_eq!(features.nrows(), 100);
        assert_eq!(features.ncols(), engine.n_features());

        // Check no NaN values
        assert!(features.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_feature_names() {
        let engine = FeatureEngine::default();
        assert!(engine.feature_names.len() > 10);
        assert!(engine.feature_names.contains(&"return".to_string()));
        assert!(engine.feature_names.contains(&"rsi".to_string()));
    }
}
