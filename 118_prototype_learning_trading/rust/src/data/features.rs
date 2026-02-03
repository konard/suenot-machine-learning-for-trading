//! Feature engineering for prototype learning.

use crate::api::bybit::Kline;
use crate::data::processor::DataProcessor;
use ndarray::{Array1, Array2};
use thiserror::Error;

/// Errors that can occur during feature engineering
#[derive(Error, Debug)]
pub enum FeatureError {
    #[error("Insufficient data for feature computation")]
    InsufficientData,

    #[error("Invalid feature configuration")]
    InvalidConfiguration,
}

/// Feature engineering for prototype learning.
///
/// Computes technical indicators from OHLCV data.
pub struct FeatureEngineering {
    /// Lookback window for sequences
    lookback: usize,
}

impl FeatureEngineering {
    /// Create a new feature engineering instance.
    pub fn new(lookback: usize) -> Self {
        Self { lookback }
    }

    /// Compute all features from klines.
    ///
    /// Returns a feature matrix of shape (n_samples, n_features).
    pub fn compute_features(&self, klines: &[Kline]) -> Result<Array2<f64>, FeatureError> {
        if klines.len() < 60 {
            return Err(FeatureError::InsufficientData);
        }

        let n = klines.len();

        // Extract price arrays
        let close: Array1<f64> = klines.iter().map(|k| k.close).collect();
        let high: Array1<f64> = klines.iter().map(|k| k.high).collect();
        let low: Array1<f64> = klines.iter().map(|k| k.low).collect();
        let volume: Array1<f64> = klines.iter().map(|k| k.volume).collect();

        // Calculate features
        let returns_1 = DataProcessor::pct_change(&close, 1);
        let returns_5 = DataProcessor::pct_change(&close, 5);
        let returns_20 = DataProcessor::pct_change(&close, 20);

        let sma_20 = DataProcessor::rolling_mean(&close, 20);
        let sma_50 = DataProcessor::rolling_mean(&close, 50);

        let price_sma20_ratio = self.ratio(&close, &sma_20);
        let price_sma50_ratio = self.ratio(&close, &sma_50);

        let bb_std = DataProcessor::rolling_std(&close, 20);
        let bb_upper = self.add_scaled(&sma_20, &bb_std, 2.0);
        let bb_lower = self.add_scaled(&sma_20, &bb_std, -2.0);
        let bb_position = self.bollinger_position(&close, &bb_lower, &bb_upper);
        let bb_width = self.bollinger_width(&bb_upper, &bb_lower, &sma_20);

        let rsi = self.calculate_rsi(&close, 14);
        let macd_hist = self.calculate_macd_hist(&close);
        let atr_ratio = self.calculate_atr_ratio(&high, &low, &close, 14);

        let volume_sma = DataProcessor::rolling_mean(&volume, 20);
        let volume_ratio = self.ratio(&volume, &volume_sma);

        let volatility = self.calculate_volatility(&returns_1, 20);
        let roc_10 = DataProcessor::pct_change(&close, 10);

        // Stack features into matrix
        let feature_names = [
            "returns_1", "returns_5", "returns_20",
            "price_sma20_ratio", "price_sma50_ratio",
            "bb_position", "bb_width",
            "rsi", "macd_hist",
            "atr_ratio", "volume_ratio",
            "volatility", "roc_10",
        ];
        let n_features = feature_names.len();

        let mut features = Array2::zeros((n, n_features));
        let feature_arrays = [
            returns_1, returns_5, returns_20,
            price_sma20_ratio, price_sma50_ratio,
            bb_position, bb_width,
            rsi, macd_hist,
            atr_ratio, volume_ratio,
            volatility, roc_10,
        ];

        for (j, arr) in feature_arrays.iter().enumerate() {
            for i in 0..n {
                features[[i, j]] = arr[i];
            }
        }

        Ok(features)
    }

    /// Get feature names.
    pub fn get_feature_names(&self) -> Vec<&'static str> {
        vec![
            "returns_1", "returns_5", "returns_20",
            "price_sma20_ratio", "price_sma50_ratio",
            "bb_position", "bb_width",
            "rsi", "macd_hist",
            "atr_ratio", "volume_ratio",
            "volatility", "roc_10",
        ]
    }

    /// Calculate ratio between two arrays.
    fn ratio(&self, a: &Array1<f64>, b: &Array1<f64>) -> Array1<f64> {
        let n = a.len();
        let mut result = Array1::from_elem(n, f64::NAN);
        for i in 0..n {
            if b[i].abs() > 1e-10 && !b[i].is_nan() {
                result[i] = a[i] / b[i] - 1.0;
            }
        }
        result
    }

    /// Add scaled array.
    fn add_scaled(&self, a: &Array1<f64>, b: &Array1<f64>, scale: f64) -> Array1<f64> {
        let n = a.len();
        let mut result = Array1::from_elem(n, f64::NAN);
        for i in 0..n {
            if !a[i].is_nan() && !b[i].is_nan() {
                result[i] = a[i] + scale * b[i];
            }
        }
        result
    }

    /// Calculate Bollinger Band position.
    fn bollinger_position(&self, close: &Array1<f64>, lower: &Array1<f64>, upper: &Array1<f64>) -> Array1<f64> {
        let n = close.len();
        let mut result = Array1::from_elem(n, f64::NAN);
        for i in 0..n {
            if !lower[i].is_nan() && !upper[i].is_nan() {
                let range = upper[i] - lower[i];
                if range.abs() > 1e-10 {
                    result[i] = (close[i] - lower[i]) / range;
                }
            }
        }
        result
    }

    /// Calculate Bollinger Band width.
    fn bollinger_width(&self, upper: &Array1<f64>, lower: &Array1<f64>, middle: &Array1<f64>) -> Array1<f64> {
        let n = upper.len();
        let mut result = Array1::from_elem(n, f64::NAN);
        for i in 0..n {
            if !upper[i].is_nan() && !lower[i].is_nan() && !middle[i].is_nan() && middle[i].abs() > 1e-10 {
                result[i] = (upper[i] - lower[i]) / middle[i];
            }
        }
        result
    }

    /// Calculate RSI (Relative Strength Index).
    fn calculate_rsi(&self, close: &Array1<f64>, period: usize) -> Array1<f64> {
        let n = close.len();
        let mut result = Array1::from_elem(n, f64::NAN);

        if n < period + 1 {
            return result;
        }

        // Calculate price changes
        let mut gains = Array1::from_elem(n, 0.0);
        let mut losses = Array1::from_elem(n, 0.0);

        for i in 1..n {
            let change = close[i] - close[i - 1];
            if change > 0.0 {
                gains[i] = change;
            } else {
                losses[i] = -change;
            }
        }

        // Calculate average gains and losses
        let avg_gains = DataProcessor::rolling_mean(&gains, period);
        let avg_losses = DataProcessor::rolling_mean(&losses, period);

        // Calculate RSI
        for i in period..n {
            if avg_losses[i].abs() < 1e-10 {
                result[i] = 100.0;
            } else {
                let rs = avg_gains[i] / avg_losses[i];
                result[i] = 100.0 - (100.0 / (1.0 + rs));
            }
        }

        // Normalize to 0-1 range
        for i in 0..n {
            if !result[i].is_nan() {
                result[i] /= 100.0;
            }
        }

        result
    }

    /// Calculate MACD histogram.
    fn calculate_macd_hist(&self, close: &Array1<f64>) -> Array1<f64> {
        let ema_12 = DataProcessor::ema(close, 12);
        let ema_26 = DataProcessor::ema(close, 26);

        let n = close.len();
        let mut macd = Array1::from_elem(n, f64::NAN);
        for i in 0..n {
            if !ema_12[i].is_nan() && !ema_26[i].is_nan() {
                macd[i] = ema_12[i] - ema_26[i];
            }
        }

        let signal = DataProcessor::ema(&macd, 9);

        let mut hist = Array1::from_elem(n, f64::NAN);
        for i in 0..n {
            if !macd[i].is_nan() && !signal[i].is_nan() {
                hist[i] = macd[i] - signal[i];
            }
        }

        // Normalize by close price
        for i in 0..n {
            if !hist[i].is_nan() && close[i].abs() > 1e-10 {
                hist[i] /= close[i];
            }
        }

        hist
    }

    /// Calculate ATR ratio (ATR / close price).
    fn calculate_atr_ratio(
        &self,
        high: &Array1<f64>,
        low: &Array1<f64>,
        close: &Array1<f64>,
        period: usize,
    ) -> Array1<f64> {
        let n = high.len();
        let mut tr = Array1::from_elem(n, f64::NAN);

        // Calculate True Range
        tr[0] = high[0] - low[0];
        for i in 1..n {
            let hl = high[i] - low[i];
            let hc = (high[i] - close[i - 1]).abs();
            let lc = (low[i] - close[i - 1]).abs();
            tr[i] = hl.max(hc).max(lc);
        }

        // Calculate ATR
        let atr = DataProcessor::rolling_mean(&tr, period);

        // Calculate ratio
        let mut result = Array1::from_elem(n, f64::NAN);
        for i in 0..n {
            if !atr[i].is_nan() && close[i].abs() > 1e-10 {
                result[i] = atr[i] / close[i];
            }
        }

        result
    }

    /// Calculate volatility (annualized standard deviation of returns).
    fn calculate_volatility(&self, returns: &Array1<f64>, window: usize) -> Array1<f64> {
        let std = DataProcessor::rolling_std(returns, window);
        let n = std.len();
        let mut result = Array1::from_elem(n, f64::NAN);

        // Annualize (assuming hourly data for crypto, 24*365 periods per year)
        let annualization_factor = (24.0 * 365.0_f64).sqrt();
        for i in 0..n {
            if !std[i].is_nan() {
                result[i] = std[i] * annualization_factor;
            }
        }

        result
    }

    /// Get the lookback window size.
    pub fn lookback(&self) -> usize {
        self.lookback
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_klines(n: usize) -> Vec<Kline> {
        (0..n)
            .map(|i| Kline {
                timestamp: i as i64 * 3600000,
                open: 100.0 + (i as f64) * 0.1,
                high: 101.0 + (i as f64) * 0.1,
                low: 99.0 + (i as f64) * 0.1,
                close: 100.5 + (i as f64) * 0.1,
                volume: 1000.0 + (i as f64) * 10.0,
                turnover: 100000.0,
            })
            .collect()
    }

    #[test]
    fn test_feature_computation() {
        let klines = create_test_klines(100);
        let fe = FeatureEngineering::new(60);
        let features = fe.compute_features(&klines).unwrap();

        assert_eq!(features.dim().0, 100);
        assert_eq!(features.dim().1, 13); // Number of features
    }

    #[test]
    fn test_insufficient_data() {
        let klines = create_test_klines(30);
        let fe = FeatureEngineering::new(60);
        let result = fe.compute_features(&klines);

        assert!(result.is_err());
    }
}
