//! Feature Engineering Engine
//!
//! Combines multiple technical indicators into a feature matrix for ML

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

use super::indicators::*;
use crate::data::OHLCVSeries;

/// Feature configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureConfig {
    /// SMA periods to use
    pub sma_periods: Vec<usize>,
    /// EMA periods to use
    pub ema_periods: Vec<usize>,
    /// RSI period
    pub rsi_period: usize,
    /// MACD parameters (fast, slow, signal)
    pub macd_params: (usize, usize, usize),
    /// Bollinger Bands period and std dev
    pub bollinger_params: (usize, f64),
    /// ATR period
    pub atr_period: usize,
    /// Stochastic parameters (k, d)
    pub stochastic_params: (usize, usize),
    /// Return periods for lagged returns
    pub return_periods: Vec<usize>,
    /// Rolling volatility period
    pub volatility_period: usize,
    /// Include volume features
    pub include_volume: bool,
}

impl Default for FeatureConfig {
    fn default() -> Self {
        Self {
            sma_periods: vec![5, 10, 20, 50],
            ema_periods: vec![12, 26],
            rsi_period: 14,
            macd_params: (12, 26, 9),
            bollinger_params: (20, 2.0),
            atr_period: 14,
            stochastic_params: (14, 3),
            return_periods: vec![1, 5, 10, 20],
            volatility_period: 20,
            include_volume: true,
        }
    }
}

/// Feature engineering engine
pub struct FeatureEngine {
    pub config: FeatureConfig,
    pub feature_names: Vec<String>,
}

impl FeatureEngine {
    /// Create new feature engine with config
    pub fn new(config: FeatureConfig) -> Self {
        Self {
            config,
            feature_names: Vec::new(),
        }
    }

    /// Create with default configuration
    pub fn default_config() -> Self {
        Self::new(FeatureConfig::default())
    }

    /// Get number of features that will be generated
    pub fn num_features(&self) -> usize {
        let mut count = 0;

        // Price-based
        count += 1; // close/open ratio

        // SMAs
        count += self.config.sma_periods.len() * 2; // SMA and close/SMA ratio

        // EMAs
        count += self.config.ema_periods.len() * 2; // EMA and close/EMA ratio

        // RSI
        count += 1;

        // MACD
        count += 3; // MACD line, signal, histogram

        // Bollinger Bands
        count += 3; // upper, middle, lower ratios

        // ATR
        count += 1;

        // Stochastic
        count += 2; // K and D

        // Returns
        count += self.config.return_periods.len();

        // Volatility
        count += 1;

        // Volume features
        if self.config.include_volume {
            count += 3; // volume change, OBV change, MFI
        }

        count
    }

    /// Extract features from OHLCV series
    /// Returns (features, target, valid_indices)
    pub fn extract_features(
        &mut self,
        series: &OHLCVSeries,
        target_horizon: usize,
    ) -> (Array2<f64>, Array1<f64>, Vec<usize>) {
        let close = series.close_prices();
        let n = close.len();

        // Calculate all indicators
        let smas: Vec<Vec<f64>> = self
            .config
            .sma_periods
            .iter()
            .map(|&p| sma(&close, p))
            .collect();

        let emas: Vec<Vec<f64>> = self
            .config
            .ema_periods
            .iter()
            .map(|&p| ema(&close, p))
            .collect();

        let rsi_vals = rsi(&close, self.config.rsi_period);

        let macd_result = macd(
            &close,
            self.config.macd_params.0,
            self.config.macd_params.1,
            self.config.macd_params.2,
        );

        let bb = bollinger_bands(
            &close,
            self.config.bollinger_params.0,
            self.config.bollinger_params.1,
        );

        let atr_vals = atr(series, self.config.atr_period);
        let stoch = stochastic(series, self.config.stochastic_params.0, self.config.stochastic_params.1);

        let returns_vec: Vec<Vec<f64>> = self
            .config
            .return_periods
            .iter()
            .map(|&p| {
                if n > p {
                    (0..n)
                        .map(|i| {
                            if i >= p && close[i - p] != 0.0 {
                                (close[i] - close[i - p]) / close[i - p]
                            } else {
                                f64::NAN
                            }
                        })
                        .collect()
                } else {
                    vec![f64::NAN; n]
                }
            })
            .collect();

        let volatility = rolling_std(&returns(&close), self.config.volatility_period);

        let obv_vals = if self.config.include_volume {
            obv(series)
        } else {
            vec![]
        };

        let mfi_vals = if self.config.include_volume {
            mfi(series, 14)
        } else {
            vec![]
        };

        // Calculate future returns as target
        let future_returns: Vec<f64> = (0..n)
            .map(|i| {
                if i + target_horizon < n && close[i] != 0.0 {
                    (close[i + target_horizon] - close[i]) / close[i]
                } else {
                    f64::NAN
                }
            })
            .collect();

        // Build feature names
        self.feature_names.clear();
        self.feature_names.push("close_open_ratio".to_string());

        for p in &self.config.sma_periods {
            self.feature_names.push(format!("sma_{}", p));
            self.feature_names.push(format!("close_sma_{}_ratio", p));
        }

        for p in &self.config.ema_periods {
            self.feature_names.push(format!("ema_{}", p));
            self.feature_names.push(format!("close_ema_{}_ratio", p));
        }

        self.feature_names.push("rsi".to_string());
        self.feature_names.push("macd".to_string());
        self.feature_names.push("macd_signal".to_string());
        self.feature_names.push("macd_histogram".to_string());
        self.feature_names.push("bb_upper_ratio".to_string());
        self.feature_names.push("bb_middle_ratio".to_string());
        self.feature_names.push("bb_lower_ratio".to_string());
        self.feature_names.push("atr".to_string());
        self.feature_names.push("stoch_k".to_string());
        self.feature_names.push("stoch_d".to_string());

        for p in &self.config.return_periods {
            self.feature_names.push(format!("return_{}", p));
        }

        self.feature_names.push("volatility".to_string());

        if self.config.include_volume {
            self.feature_names.push("volume_change".to_string());
            self.feature_names.push("obv_change".to_string());
            self.feature_names.push("mfi".to_string());
        }

        // Find valid indices (where all features and target are not NaN)
        let lookback = *self.config.sma_periods.iter().max().unwrap_or(&50).max(&50);
        let valid_start = lookback + self.config.volatility_period;
        let valid_end = n.saturating_sub(target_horizon);

        let valid_indices: Vec<usize> = (valid_start..valid_end)
            .filter(|&i| {
                !future_returns[i].is_nan()
                    && !rsi_vals[i].is_nan()
                    && smas.iter().all(|s| !s[i].is_nan())
            })
            .collect();

        // Build feature matrix
        let num_features = self.num_features();
        let num_samples = valid_indices.len();

        let mut features = Array2::zeros((num_samples, num_features));
        let mut target = Array1::zeros(num_samples);

        for (row_idx, &i) in valid_indices.iter().enumerate() {
            let mut col = 0;

            // Close/Open ratio
            features[[row_idx, col]] = series.data[i].close / series.data[i].open;
            col += 1;

            // SMAs
            for sma_vec in &smas {
                features[[row_idx, col]] = sma_vec[i];
                col += 1;
                features[[row_idx, col]] = close[i] / sma_vec[i];
                col += 1;
            }

            // EMAs
            for ema_vec in &emas {
                features[[row_idx, col]] = ema_vec[i];
                col += 1;
                features[[row_idx, col]] = close[i] / ema_vec[i];
                col += 1;
            }

            // RSI
            features[[row_idx, col]] = rsi_vals[i] / 100.0; // Normalize to [0, 1]
            col += 1;

            // MACD
            features[[row_idx, col]] = macd_result.macd_line[i];
            col += 1;
            features[[row_idx, col]] = macd_result.signal_line[i];
            col += 1;
            features[[row_idx, col]] = macd_result.histogram[i];
            col += 1;

            // Bollinger Bands ratios
            features[[row_idx, col]] = close[i] / bb.upper[i];
            col += 1;
            features[[row_idx, col]] = close[i] / bb.middle[i];
            col += 1;
            features[[row_idx, col]] = close[i] / bb.lower[i];
            col += 1;

            // ATR
            features[[row_idx, col]] = atr_vals[i] / close[i]; // Normalize by price
            col += 1;

            // Stochastic
            features[[row_idx, col]] = stoch.k[i] / 100.0;
            col += 1;
            features[[row_idx, col]] = stoch.d[i] / 100.0;
            col += 1;

            // Returns
            for ret_vec in &returns_vec {
                features[[row_idx, col]] = ret_vec[i];
                col += 1;
            }

            // Volatility
            let vol_idx = i.saturating_sub(1);
            features[[row_idx, col]] = if vol_idx < volatility.len() && !volatility[vol_idx].is_nan() {
                volatility[vol_idx]
            } else {
                0.0
            };
            col += 1;

            // Volume features
            if self.config.include_volume {
                // Volume change
                let vol_change = if i > 0 && series.data[i - 1].volume > 0.0 {
                    (series.data[i].volume - series.data[i - 1].volume) / series.data[i - 1].volume
                } else {
                    0.0
                };
                features[[row_idx, col]] = vol_change;
                col += 1;

                // OBV change
                let obv_change = if i > 0 && obv_vals[i - 1] != 0.0 {
                    (obv_vals[i] - obv_vals[i - 1]) / obv_vals[i - 1].abs().max(1.0)
                } else {
                    0.0
                };
                features[[row_idx, col]] = obv_change;
                col += 1;

                // MFI
                features[[row_idx, col]] = if !mfi_vals[i].is_nan() { mfi_vals[i] / 100.0 } else { 0.5 };
            }

            // Target: future return
            target[row_idx] = future_returns[i];
        }

        (features, target, valid_indices)
    }

    /// Get feature names
    pub fn get_feature_names(&self) -> &[String] {
        &self.feature_names
    }

    /// Create binary classification target (1 if return > threshold, 0 otherwise)
    pub fn to_classification_target(returns: &Array1<f64>, threshold: f64) -> Array1<f64> {
        returns.mapv(|r| if r > threshold { 1.0 } else { 0.0 })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{TimeZone, Utc};
    use crate::data::OHLCV;

    fn create_test_series(n: usize) -> OHLCVSeries {
        let data: Vec<OHLCV> = (0..n)
            .map(|i| {
                let price = 100.0 + (i as f64) * 0.1 + (i as f64 * 0.1).sin() * 5.0;
                OHLCV::new(
                    Utc.with_ymd_and_hms(2024, 1, 1, i as u32 % 24, 0, 0).unwrap(),
                    price - 0.5,
                    price + 1.0,
                    price - 1.0,
                    price,
                    1000.0 + (i as f64) * 10.0,
                )
            })
            .collect();

        OHLCVSeries::with_data("BTCUSDT".to_string(), "1h".to_string(), data)
    }

    #[test]
    fn test_feature_engine() {
        let series = create_test_series(200);
        let mut engine = FeatureEngine::default_config();

        let (features, target, indices) = engine.extract_features(&series, 1);

        assert!(features.nrows() > 0);
        assert_eq!(features.ncols(), engine.num_features());
        assert_eq!(features.nrows(), target.len());
        assert_eq!(features.nrows(), indices.len());
    }

    #[test]
    fn test_feature_names() {
        let series = create_test_series(200);
        let mut engine = FeatureEngine::default_config();

        let (features, _, _) = engine.extract_features(&series, 1);
        let names = engine.get_feature_names();

        assert_eq!(names.len(), features.ncols());
    }
}
