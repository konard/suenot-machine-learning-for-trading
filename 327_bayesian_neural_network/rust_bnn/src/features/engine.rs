//! Feature engineering engine.

use anyhow::Result;
use ndarray::Array2;

use crate::api::types::Kline;
use super::indicators::TechnicalIndicators;

/// Feature engineering result.
#[derive(Debug)]
pub struct Features {
    /// Feature matrix (n_samples, n_features)
    pub x: Array2<f64>,
    /// Classification labels (0=up, 1=neutral, 2=down)
    pub y_class: Vec<i64>,
    /// Continuous return values
    pub y_return: Vec<f64>,
    /// Feature names
    pub feature_names: Vec<String>,
    /// Timestamps
    pub timestamps: Vec<chrono::DateTime<chrono::Utc>>,
}

/// Feature engineering engine.
#[derive(Debug, Clone)]
pub struct FeatureEngine {
    /// Lookback period for rolling calculations
    pub lookback: usize,
    /// Prediction horizon
    pub prediction_horizon: usize,
    /// Return threshold for classification
    pub return_threshold: f64,
}

impl Default for FeatureEngine {
    fn default() -> Self {
        Self {
            lookback: 20,
            prediction_horizon: 5,
            return_threshold: 0.002,
        }
    }
}

impl FeatureEngine {
    /// Create a new feature engine.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set lookback period.
    pub fn lookback(mut self, period: usize) -> Self {
        self.lookback = period;
        self
    }

    /// Set prediction horizon.
    pub fn prediction_horizon(mut self, horizon: usize) -> Self {
        self.prediction_horizon = horizon;
        self
    }

    /// Set return threshold.
    pub fn return_threshold(mut self, threshold: f64) -> Self {
        self.return_threshold = threshold;
        self
    }

    /// Create features from kline data.
    pub fn create_features(&self, klines: &[Kline]) -> Result<Features> {
        if klines.len() < self.lookback + self.prediction_horizon + 1 {
            anyhow::bail!(
                "Not enough data: need at least {} klines, got {}",
                self.lookback + self.prediction_horizon + 1,
                klines.len()
            );
        }

        // Extract price series
        let closes: Vec<f64> = klines.iter().map(|k| k.close).collect();
        let highs: Vec<f64> = klines.iter().map(|k| k.high).collect();
        let lows: Vec<f64> = klines.iter().map(|k| k.low).collect();
        let volumes: Vec<f64> = klines.iter().map(|k| k.volume).collect();

        // Calculate indicators
        let returns_1 = TechnicalIndicators::returns(&closes, 1);
        let returns_5 = TechnicalIndicators::returns(&closes, 5);
        let returns_10 = TechnicalIndicators::returns(&closes, 10);
        let returns_20 = TechnicalIndicators::returns(&closes, 20);

        let log_returns_1 = TechnicalIndicators::log_returns(&closes, 1);

        let volatility_5 = TechnicalIndicators::rolling_std(&log_returns_1, 5);
        let volatility_10 = TechnicalIndicators::rolling_std(&log_returns_1, 10);
        let volatility_20 = TechnicalIndicators::rolling_std(&log_returns_1, 20);

        let rsi = TechnicalIndicators::rsi(&closes, 14);
        let (macd, macd_signal, macd_hist) = TechnicalIndicators::macd(&closes, 12, 26, 9);
        let (bb_upper, bb_middle, bb_lower) = TechnicalIndicators::bollinger_bands(&closes, 20, 2.0);
        let atr = TechnicalIndicators::atr(klines, 14);
        let (stoch_k, stoch_d) = TechnicalIndicators::stochastic(klines, 14, 3);

        let volume_sma = TechnicalIndicators::sma(&volumes, 20);

        // Build feature vectors
        let mut feature_names = vec![
            "return_1".to_string(),
            "return_5".to_string(),
            "return_10".to_string(),
            "return_20".to_string(),
            "volatility_5".to_string(),
            "volatility_10".to_string(),
            "volatility_20".to_string(),
            "rsi".to_string(),
            "rsi_normalized".to_string(),
            "macd_normalized".to_string(),
            "macd_signal_normalized".to_string(),
            "macd_hist_normalized".to_string(),
            "bb_position".to_string(),
            "bb_width".to_string(),
            "atr_normalized".to_string(),
            "stoch_k_normalized".to_string(),
            "stoch_d_normalized".to_string(),
            "volume_ratio".to_string(),
            "range".to_string(),
            "body_ratio".to_string(),
        ];

        // Add lagged features
        for lag in &[1, 2, 3, 5] {
            feature_names.push(format!("return_1_lag_{}", lag));
            feature_names.push(format!("rsi_lag_{}", lag));
        }

        let n_features = feature_names.len();
        let start_idx = self.lookback.max(26); // Start after enough history
        let end_idx = klines.len() - self.prediction_horizon;

        let n_samples = end_idx - start_idx;
        let mut x = Array2::zeros((n_samples, n_features));
        let mut y_class = Vec::with_capacity(n_samples);
        let mut y_return = Vec::with_capacity(n_samples);
        let mut timestamps = Vec::with_capacity(n_samples);

        for (row_idx, i) in (start_idx..end_idx).enumerate() {
            let close = closes[i];

            // Basic features
            x[[row_idx, 0]] = nan_to_zero(returns_1[i]);
            x[[row_idx, 1]] = nan_to_zero(returns_5[i]);
            x[[row_idx, 2]] = nan_to_zero(returns_10[i]);
            x[[row_idx, 3]] = nan_to_zero(returns_20[i]);
            x[[row_idx, 4]] = nan_to_zero(volatility_5[i]);
            x[[row_idx, 5]] = nan_to_zero(volatility_10[i]);
            x[[row_idx, 6]] = nan_to_zero(volatility_20[i]);
            x[[row_idx, 7]] = nan_to_zero(rsi[i]);
            x[[row_idx, 8]] = (nan_to_zero(rsi[i]) - 50.0) / 50.0; // Normalized RSI
            x[[row_idx, 9]] = nan_to_zero(macd[i]) / close; // Normalized MACD
            x[[row_idx, 10]] = nan_to_zero(macd_signal[i]) / close;
            x[[row_idx, 11]] = nan_to_zero(macd_hist[i]) / close;

            // Bollinger position
            let bb_range = bb_upper[i] - bb_lower[i];
            if bb_range > 0.0 && !bb_range.is_nan() {
                x[[row_idx, 12]] = (close - bb_lower[i]) / bb_range;
                x[[row_idx, 13]] = bb_range / bb_middle[i];
            }

            x[[row_idx, 14]] = nan_to_zero(atr[i]) / close; // Normalized ATR
            x[[row_idx, 15]] = (nan_to_zero(stoch_k[i]) - 50.0) / 50.0;
            x[[row_idx, 16]] = (nan_to_zero(stoch_d[i]) - 50.0) / 50.0;

            // Volume ratio
            if !volume_sma[i].is_nan() && volume_sma[i] > 0.0 {
                x[[row_idx, 17]] = volumes[i] / volume_sma[i];
            }

            // Range and body ratio
            let range = highs[i] - lows[i];
            x[[row_idx, 18]] = if close > 0.0 { range / close } else { 0.0 };
            x[[row_idx, 19]] = if range > 0.0 {
                (closes[i] - klines[i].open) / range
            } else {
                0.0
            };

            // Lagged features
            let mut feat_idx = 20;
            for &lag in &[1_usize, 2, 3, 5] {
                if i >= lag {
                    x[[row_idx, feat_idx]] = nan_to_zero(returns_1[i - lag]);
                    x[[row_idx, feat_idx + 1]] = (nan_to_zero(rsi[i - lag]) - 50.0) / 50.0;
                }
                feat_idx += 2;
            }

            // Target: future return
            let future_return =
                (closes[i + self.prediction_horizon] - close) / close;
            y_return.push(future_return);

            // Classification label
            let label = if future_return > self.return_threshold {
                0 // Up
            } else if future_return < -self.return_threshold {
                2 // Down
            } else {
                1 // Neutral
            };
            y_class.push(label);

            timestamps.push(klines[i].timestamp);
        }

        // Normalize features
        let x = normalize_features(x);

        Ok(Features {
            x,
            y_class,
            y_return,
            feature_names,
            timestamps,
        })
    }
}

/// Replace NaN with zero.
fn nan_to_zero(x: f64) -> f64 {
    if x.is_nan() || x.is_infinite() {
        0.0
    } else {
        x
    }
}

/// Normalize features using robust scaling.
fn normalize_features(mut x: Array2<f64>) -> Array2<f64> {
    for mut col in x.columns_mut() {
        let values: Vec<f64> = col.iter().copied().filter(|v| !v.is_nan()).collect();
        if values.is_empty() {
            continue;
        }

        // Compute median
        let mut sorted = values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = sorted[sorted.len() / 2];

        // Compute IQR
        let q25 = sorted[sorted.len() / 4];
        let q75 = sorted[3 * sorted.len() / 4];
        let iqr = q75 - q25;

        if iqr > 1e-10 {
            for v in col.iter_mut() {
                *v = (*v - median) / iqr;
            }
        } else {
            for v in col.iter_mut() {
                *v = *v - median;
            }
        }
    }

    // Clip extreme values
    x.mapv_inplace(|v| v.clamp(-5.0, 5.0));

    x
}
