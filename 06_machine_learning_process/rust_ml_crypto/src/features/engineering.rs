//! Feature engineering for ML models
//!
//! Transforms raw candle data into features suitable for machine learning.
//! Features include:
//! - Price-based features (returns, momentum)
//! - Technical indicators (SMA, RSI, MACD, etc.)
//! - Volume-based features
//! - Volatility measures

use super::indicators::TechnicalIndicators;
use crate::data::{Candle, Dataset};
use ndarray::{Array1, Array2};

/// Feature configuration
#[derive(Debug, Clone)]
pub struct FeatureConfig {
    /// Periods for moving averages
    pub sma_periods: Vec<usize>,
    /// Periods for EMA
    pub ema_periods: Vec<usize>,
    /// Period for RSI
    pub rsi_period: usize,
    /// MACD parameters (fast, slow, signal)
    pub macd_params: (usize, usize, usize),
    /// Bollinger Bands parameters (period, num_std)
    pub bb_params: (usize, f64),
    /// ATR period
    pub atr_period: usize,
    /// Return periods for momentum
    pub return_periods: Vec<usize>,
    /// Rolling volatility periods
    pub volatility_periods: Vec<usize>,
    /// Prediction horizon (how many candles ahead to predict)
    pub prediction_horizon: usize,
    /// Target type: "direction" (binary) or "return" (continuous)
    pub target_type: String,
}

impl Default for FeatureConfig {
    fn default() -> Self {
        Self {
            sma_periods: vec![5, 10, 20, 50],
            ema_periods: vec![12, 26],
            rsi_period: 14,
            macd_params: (12, 26, 9),
            bb_params: (20, 2.0),
            atr_period: 14,
            return_periods: vec![1, 5, 10, 20],
            volatility_periods: vec![5, 10, 20],
            prediction_horizon: 1,
            target_type: "direction".to_string(),
        }
    }
}

/// Feature engineering engine
pub struct FeatureEngine {
    config: FeatureConfig,
}

impl FeatureEngine {
    /// Create a new feature engine with default configuration
    pub fn new() -> Self {
        Self {
            config: FeatureConfig::default(),
        }
    }

    /// Create a feature engine with custom configuration
    pub fn with_config(config: FeatureConfig) -> Self {
        Self { config }
    }

    /// Generate features from candle data
    ///
    /// Returns a Dataset with:
    /// - X: Feature matrix (n_samples x n_features)
    /// - y: Target variable (returns or direction)
    pub fn generate_features(&self, candles: &[Candle]) -> Option<Dataset> {
        if candles.len() < 100 {
            return None; // Need enough data for indicators
        }

        let closes: Vec<f64> = candles.iter().map(|c| c.close).collect();
        let highs: Vec<f64> = candles.iter().map(|c| c.high).collect();
        let lows: Vec<f64> = candles.iter().map(|c| c.low).collect();
        let volumes: Vec<f64> = candles.iter().map(|c| c.volume).collect();

        let mut features: Vec<Vec<f64>> = Vec::new();
        let mut feature_names: Vec<String> = Vec::new();

        // Returns
        let returns = TechnicalIndicators::returns(&closes);
        features.push(returns.clone());
        feature_names.push("return_1".to_string());

        // Log returns
        let log_returns = TechnicalIndicators::log_returns(&closes);
        features.push(log_returns);
        feature_names.push("log_return_1".to_string());

        // Multi-period returns
        for &period in &self.config.return_periods {
            if period > 1 {
                let mut period_returns = vec![f64::NAN; candles.len()];
                for i in period..candles.len() {
                    if closes[i - period] != 0.0 {
                        period_returns[i] = (closes[i] - closes[i - period]) / closes[i - period];
                    }
                }
                features.push(period_returns);
                feature_names.push(format!("return_{}", period));
            }
        }

        // SMAs and price ratios
        for &period in &self.config.sma_periods {
            let sma = TechnicalIndicators::sma(&closes, period);

            // Price / SMA ratio
            let ratio: Vec<f64> = closes
                .iter()
                .zip(sma.iter())
                .map(|(p, s)| if s.is_nan() || *s == 0.0 { f64::NAN } else { p / s })
                .collect();

            features.push(ratio);
            feature_names.push(format!("price_sma{}_ratio", period));
        }

        // EMAs
        for &period in &self.config.ema_periods {
            let ema = TechnicalIndicators::ema(&closes, period);

            let ratio: Vec<f64> = closes
                .iter()
                .zip(ema.iter())
                .map(|(p, e)| if e.is_nan() || *e == 0.0 { f64::NAN } else { p / e })
                .collect();

            features.push(ratio);
            feature_names.push(format!("price_ema{}_ratio", period));
        }

        // RSI
        let rsi = TechnicalIndicators::rsi(&closes, self.config.rsi_period);
        features.push(rsi);
        feature_names.push("rsi".to_string());

        // MACD
        let (macd, signal, hist) = TechnicalIndicators::macd(
            &closes,
            self.config.macd_params.0,
            self.config.macd_params.1,
            self.config.macd_params.2,
        );
        features.push(macd);
        feature_names.push("macd".to_string());
        features.push(signal);
        feature_names.push("macd_signal".to_string());
        features.push(hist);
        feature_names.push("macd_hist".to_string());

        // Bollinger Bands
        let (bb_mid, bb_upper, bb_lower) = TechnicalIndicators::bollinger_bands(
            &closes,
            self.config.bb_params.0,
            self.config.bb_params.1,
        );

        // BB position: (price - lower) / (upper - lower)
        let bb_position: Vec<f64> = closes
            .iter()
            .zip(bb_upper.iter().zip(bb_lower.iter()))
            .map(|(p, (u, l))| {
                if u.is_nan() || l.is_nan() || (u - l) == 0.0 {
                    f64::NAN
                } else {
                    (p - l) / (u - l)
                }
            })
            .collect();
        features.push(bb_position);
        feature_names.push("bb_position".to_string());

        // BB width
        let bb_width: Vec<f64> = bb_upper
            .iter()
            .zip(bb_lower.iter().zip(bb_mid.iter()))
            .map(|(u, (l, m))| {
                if u.is_nan() || l.is_nan() || m.is_nan() || *m == 0.0 {
                    f64::NAN
                } else {
                    (u - l) / m
                }
            })
            .collect();
        features.push(bb_width);
        feature_names.push("bb_width".to_string());

        // ATR and normalized ATR
        let atr = TechnicalIndicators::atr(candles, self.config.atr_period);
        let atr_pct: Vec<f64> = atr
            .iter()
            .zip(closes.iter())
            .map(|(a, c)| if a.is_nan() || *c == 0.0 { f64::NAN } else { a / c })
            .collect();
        features.push(atr_pct);
        feature_names.push("atr_pct".to_string());

        // Volatility (rolling std of returns)
        for &period in &self.config.volatility_periods {
            let vol = TechnicalIndicators::rolling_std(&returns, period);
            features.push(vol);
            feature_names.push(format!("volatility_{}", period));
        }

        // Volume features
        let vol_sma = TechnicalIndicators::sma(&volumes, 20);
        let vol_ratio: Vec<f64> = volumes
            .iter()
            .zip(vol_sma.iter())
            .map(|(v, s)| if s.is_nan() || *s == 0.0 { f64::NAN } else { v / s })
            .collect();
        features.push(vol_ratio);
        feature_names.push("volume_sma20_ratio".to_string());

        // OBV
        let obv = TechnicalIndicators::obv(candles);
        let obv_sma = TechnicalIndicators::sma(&obv, 20);
        let obv_ratio: Vec<f64> = obv
            .iter()
            .zip(obv_sma.iter())
            .map(|(o, s)| {
                if s.is_nan() || *s == 0.0 {
                    f64::NAN
                } else {
                    o / s.abs()
                }
            })
            .collect();
        features.push(obv_ratio);
        feature_names.push("obv_sma20_ratio".to_string());

        // Candle body size
        let body_size: Vec<f64> = candles
            .iter()
            .map(|c| if c.open == 0.0 { f64::NAN } else { (c.close - c.open) / c.open })
            .collect();
        features.push(body_size);
        feature_names.push("candle_body".to_string());

        // Upper shadow
        let upper_shadow: Vec<f64> = candles
            .iter()
            .map(|c| {
                let range = c.high - c.low;
                if range == 0.0 { f64::NAN } else { c.upper_shadow() / range }
            })
            .collect();
        features.push(upper_shadow);
        feature_names.push("upper_shadow_ratio".to_string());

        // Lower shadow
        let lower_shadow: Vec<f64> = candles
            .iter()
            .map(|c| {
                let range = c.high - c.low;
                if range == 0.0 { f64::NAN } else { c.lower_shadow() / range }
            })
            .collect();
        features.push(lower_shadow);
        feature_names.push("lower_shadow_ratio".to_string());

        // High-Low range
        let hl_range: Vec<f64> = candles
            .iter()
            .map(|c| if c.low == 0.0 { f64::NAN } else { (c.high - c.low) / c.low })
            .collect();
        features.push(hl_range);
        feature_names.push("high_low_range".to_string());

        // Create target variable
        let horizon = self.config.prediction_horizon;
        let target: Vec<f64> = if self.config.target_type == "direction" {
            // Binary: 1 if price goes up, 0 if down
            (0..candles.len())
                .map(|i| {
                    if i + horizon >= candles.len() {
                        f64::NAN
                    } else if closes[i + horizon] > closes[i] {
                        1.0
                    } else {
                        0.0
                    }
                })
                .collect()
        } else {
            // Continuous: future return
            (0..candles.len())
                .map(|i| {
                    if i + horizon >= candles.len() || closes[i] == 0.0 {
                        f64::NAN
                    } else {
                        (closes[i + horizon] - closes[i]) / closes[i]
                    }
                })
                .collect()
        };

        // Find valid rows (no NaN in any feature or target)
        let n_samples = candles.len();
        let n_features = features.len();

        let valid_rows: Vec<usize> = (0..n_samples)
            .filter(|&i| {
                !target[i].is_nan()
                    && features.iter().all(|f| !f[i].is_nan() && f[i].is_finite())
            })
            .collect();

        if valid_rows.is_empty() {
            return None;
        }

        // Build final feature matrix
        let mut x_data = Vec::with_capacity(valid_rows.len() * n_features);
        let mut y_data = Vec::with_capacity(valid_rows.len());

        for &row_idx in &valid_rows {
            for feature in &features {
                x_data.push(feature[row_idx]);
            }
            y_data.push(target[row_idx]);
        }

        let x = Array2::from_shape_vec((valid_rows.len(), n_features), x_data).ok()?;
        let y = Array1::from_vec(y_data);

        Some(Dataset::new(
            x,
            y,
            feature_names,
            "target".to_string(),
        ))
    }

    /// Get feature names
    pub fn get_feature_names(&self) -> Vec<String> {
        let mut names = Vec::new();

        names.push("return_1".to_string());
        names.push("log_return_1".to_string());

        for &period in &self.config.return_periods {
            if period > 1 {
                names.push(format!("return_{}", period));
            }
        }

        for &period in &self.config.sma_periods {
            names.push(format!("price_sma{}_ratio", period));
        }

        for &period in &self.config.ema_periods {
            names.push(format!("price_ema{}_ratio", period));
        }

        names.push("rsi".to_string());
        names.push("macd".to_string());
        names.push("macd_signal".to_string());
        names.push("macd_hist".to_string());
        names.push("bb_position".to_string());
        names.push("bb_width".to_string());
        names.push("atr_pct".to_string());

        for &period in &self.config.volatility_periods {
            names.push(format!("volatility_{}", period));
        }

        names.push("volume_sma20_ratio".to_string());
        names.push("obv_sma20_ratio".to_string());
        names.push("candle_body".to_string());
        names.push("upper_shadow_ratio".to_string());
        names.push("lower_shadow_ratio".to_string());
        names.push("high_low_range".to_string());

        names
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

    fn create_test_candles(n: usize) -> Vec<Candle> {
        (0..n)
            .map(|i| {
                let base = 100.0 + (i as f64).sin() * 10.0;
                Candle {
                    timestamp: i as u64 * 60000,
                    open: base,
                    high: base + 2.0,
                    low: base - 1.0,
                    close: base + 1.0 + (i as f64 * 0.1).cos(),
                    volume: 1000.0 + (i as f64 * 100.0),
                    turnover: 100000.0 + (i as f64 * 10000.0),
                }
            })
            .collect()
    }

    #[test]
    fn test_generate_features() {
        let candles = create_test_candles(200);
        let engine = FeatureEngine::new();

        let dataset = engine.generate_features(&candles);
        assert!(dataset.is_some());

        let dataset = dataset.unwrap();
        assert!(dataset.n_samples() > 0);
        assert!(dataset.n_features() > 10);
    }
}
