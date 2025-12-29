//! Feature engineering for cryptocurrency trading
//!
//! Provides technical indicators and feature generation from OHLCV data.

use crate::api::bybit::Kline;
use ndarray::{Array1, Array2};

/// Feature engineering utilities
pub struct FeatureEngineering;

impl FeatureEngineering {
    /// Generate all features from klines
    ///
    /// Returns a 2D array of features and a vector of feature names
    pub fn generate_features(klines: &[Kline]) -> (Array2<f64>, Vec<String>) {
        if klines.is_empty() {
            return (Array2::zeros((0, 0)), vec![]);
        }

        let n = klines.len();
        let closes: Vec<f64> = klines.iter().map(|k| k.close).collect();
        let highs: Vec<f64> = klines.iter().map(|k| k.high).collect();
        let lows: Vec<f64> = klines.iter().map(|k| k.low).collect();
        let volumes: Vec<f64> = klines.iter().map(|k| k.volume).collect();

        // Calculate all features
        let returns_1 = Self::returns(&closes, 1);
        let returns_5 = Self::returns(&closes, 5);
        let returns_10 = Self::returns(&closes, 10);

        let sma_5 = Self::sma(&closes, 5);
        let sma_10 = Self::sma(&closes, 10);
        let sma_20 = Self::sma(&closes, 20);

        let ema_5 = Self::ema(&closes, 5);
        let ema_10 = Self::ema(&closes, 10);
        let ema_20 = Self::ema(&closes, 20);

        let rsi_14 = Self::rsi(&closes, 14);
        let (macd, signal, _) = Self::macd(&closes, 12, 26, 9);

        let volatility_10 = Self::rolling_volatility(&closes, 10);
        let volatility_20 = Self::rolling_volatility(&closes, 20);

        let atr_14 = Self::atr(&highs, &lows, &closes, 14);
        let volume_ratio = Self::volume_ratio(&volumes, 20);

        let (bb_upper, bb_lower) = Self::bollinger_bands(&closes, 20, 2.0);
        let bb_position: Vec<f64> = closes
            .iter()
            .zip(bb_upper.iter().zip(bb_lower.iter()))
            .map(|(&c, (&u, &l))| {
                if (u - l).abs() < 1e-10 {
                    0.5
                } else {
                    (c - l) / (u - l)
                }
            })
            .collect();

        // Price position relative to SMAs
        let price_sma5_ratio: Vec<f64> = closes
            .iter()
            .zip(sma_5.iter())
            .map(|(&c, &s)| if s.abs() < 1e-10 { 1.0 } else { c / s })
            .collect();

        let price_sma20_ratio: Vec<f64> = closes
            .iter()
            .zip(sma_20.iter())
            .map(|(&c, &s)| if s.abs() < 1e-10 { 1.0 } else { c / s })
            .collect();

        // Collect all features into a 2D array
        let feature_names = vec![
            "returns_1".to_string(),
            "returns_5".to_string(),
            "returns_10".to_string(),
            "sma_5".to_string(),
            "sma_10".to_string(),
            "sma_20".to_string(),
            "ema_5".to_string(),
            "ema_10".to_string(),
            "ema_20".to_string(),
            "rsi_14".to_string(),
            "macd".to_string(),
            "macd_signal".to_string(),
            "volatility_10".to_string(),
            "volatility_20".to_string(),
            "atr_14".to_string(),
            "volume_ratio".to_string(),
            "bb_position".to_string(),
            "price_sma5_ratio".to_string(),
            "price_sma20_ratio".to_string(),
        ];

        let n_features = feature_names.len();
        let mut features = Array2::zeros((n, n_features));

        let all_features: Vec<&Vec<f64>> = vec![
            &returns_1,
            &returns_5,
            &returns_10,
            &sma_5,
            &sma_10,
            &sma_20,
            &ema_5,
            &ema_10,
            &ema_20,
            &rsi_14,
            &macd,
            &signal,
            &volatility_10,
            &volatility_20,
            &atr_14,
            &volume_ratio,
            &bb_position,
            &price_sma5_ratio,
            &price_sma20_ratio,
        ];

        for (j, feat) in all_features.iter().enumerate() {
            for (i, &val) in feat.iter().enumerate() {
                features[[i, j]] = val;
            }
        }

        (features, feature_names)
    }

    /// Create forward returns as target variable
    pub fn create_returns(klines: &[Kline], periods: usize) -> Vec<f64> {
        if klines.len() <= periods {
            return vec![];
        }

        let closes: Vec<f64> = klines.iter().map(|k| k.close).collect();
        Self::forward_returns(&closes, periods)
    }

    /// Calculate simple returns
    pub fn returns(prices: &[f64], periods: usize) -> Vec<f64> {
        let n = prices.len();
        let mut returns = vec![0.0; n];

        for i in periods..n {
            if prices[i - periods].abs() > 1e-10 {
                returns[i] = (prices[i] / prices[i - periods]) - 1.0;
            }
        }

        returns
    }

    /// Calculate forward returns (for target creation)
    pub fn forward_returns(prices: &[f64], periods: usize) -> Vec<f64> {
        let n = prices.len();
        let mut returns = vec![0.0; n];

        for i in 0..n.saturating_sub(periods) {
            if prices[i].abs() > 1e-10 {
                returns[i] = (prices[i + periods] / prices[i]) - 1.0;
            }
        }

        returns
    }

    /// Simple Moving Average
    pub fn sma(prices: &[f64], period: usize) -> Vec<f64> {
        let n = prices.len();
        let mut result = vec![0.0; n];

        for i in (period - 1)..n {
            let sum: f64 = prices[(i + 1 - period)..=i].iter().sum();
            result[i] = sum / period as f64;
        }

        // Fill early values with first available SMA
        if n >= period {
            let first_sma = result[period - 1];
            for i in 0..(period - 1) {
                result[i] = first_sma;
            }
        }

        result
    }

    /// Exponential Moving Average
    pub fn ema(prices: &[f64], period: usize) -> Vec<f64> {
        let n = prices.len();
        if n == 0 {
            return vec![];
        }

        let mut result = vec![0.0; n];
        let multiplier = 2.0 / (period as f64 + 1.0);

        // Initialize with SMA
        if n >= period {
            let sma: f64 = prices[..period].iter().sum::<f64>() / period as f64;
            result[period - 1] = sma;

            for i in period..n {
                result[i] = (prices[i] - result[i - 1]) * multiplier + result[i - 1];
            }

            // Fill early values
            for i in 0..(period - 1) {
                result[i] = result[period - 1];
            }
        } else {
            let avg = prices.iter().sum::<f64>() / n as f64;
            result.fill(avg);
        }

        result
    }

    /// Relative Strength Index
    pub fn rsi(prices: &[f64], period: usize) -> Vec<f64> {
        let n = prices.len();
        if n <= period {
            return vec![50.0; n];
        }

        let mut result = vec![50.0; n];
        let mut gains = vec![0.0; n];
        let mut losses = vec![0.0; n];

        // Calculate price changes
        for i in 1..n {
            let change = prices[i] - prices[i - 1];
            if change > 0.0 {
                gains[i] = change;
            } else {
                losses[i] = -change;
            }
        }

        // Calculate initial averages
        let mut avg_gain: f64 = gains[1..=period].iter().sum::<f64>() / period as f64;
        let mut avg_loss: f64 = losses[1..=period].iter().sum::<f64>() / period as f64;

        if avg_loss.abs() < 1e-10 {
            result[period] = 100.0;
        } else {
            let rs = avg_gain / avg_loss;
            result[period] = 100.0 - (100.0 / (1.0 + rs));
        }

        // Calculate RSI using smoothed averages
        for i in (period + 1)..n {
            avg_gain = (avg_gain * (period - 1) as f64 + gains[i]) / period as f64;
            avg_loss = (avg_loss * (period - 1) as f64 + losses[i]) / period as f64;

            if avg_loss.abs() < 1e-10 {
                result[i] = 100.0;
            } else {
                let rs = avg_gain / avg_loss;
                result[i] = 100.0 - (100.0 / (1.0 + rs));
            }
        }

        result
    }

    /// MACD (Moving Average Convergence Divergence)
    pub fn macd(prices: &[f64], fast: usize, slow: usize, signal_period: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let ema_fast = Self::ema(prices, fast);
        let ema_slow = Self::ema(prices, slow);

        let macd_line: Vec<f64> = ema_fast
            .iter()
            .zip(ema_slow.iter())
            .map(|(&f, &s)| f - s)
            .collect();

        let signal_line = Self::ema(&macd_line, signal_period);

        let histogram: Vec<f64> = macd_line
            .iter()
            .zip(signal_line.iter())
            .map(|(&m, &s)| m - s)
            .collect();

        (macd_line, signal_line, histogram)
    }

    /// Rolling volatility (standard deviation of returns)
    pub fn rolling_volatility(prices: &[f64], period: usize) -> Vec<f64> {
        let returns = Self::returns(prices, 1);
        Self::rolling_std(&returns, period)
    }

    /// Rolling standard deviation
    pub fn rolling_std(values: &[f64], period: usize) -> Vec<f64> {
        let n = values.len();
        let mut result = vec![0.0; n];

        for i in (period - 1)..n {
            let window = &values[(i + 1 - period)..=i];
            let mean = window.iter().sum::<f64>() / period as f64;
            let variance = window.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / period as f64;
            result[i] = variance.sqrt();
        }

        // Fill early values
        if n >= period {
            let first_std = result[period - 1];
            for i in 0..(period - 1) {
                result[i] = first_std;
            }
        }

        result
    }

    /// Average True Range
    pub fn atr(highs: &[f64], lows: &[f64], closes: &[f64], period: usize) -> Vec<f64> {
        let n = highs.len();
        if n == 0 {
            return vec![];
        }

        let mut tr = vec![highs[0] - lows[0]; n];

        for i in 1..n {
            let hl = highs[i] - lows[i];
            let hc = (highs[i] - closes[i - 1]).abs();
            let lc = (lows[i] - closes[i - 1]).abs();
            tr[i] = hl.max(hc).max(lc);
        }

        Self::ema(&tr, period)
    }

    /// Volume ratio (current volume / average volume)
    pub fn volume_ratio(volumes: &[f64], period: usize) -> Vec<f64> {
        let avg_volume = Self::sma(volumes, period);

        volumes
            .iter()
            .zip(avg_volume.iter())
            .map(|(&v, &avg)| {
                if avg.abs() < 1e-10 {
                    1.0
                } else {
                    v / avg
                }
            })
            .collect()
    }

    /// Bollinger Bands
    pub fn bollinger_bands(prices: &[f64], period: usize, std_dev: f64) -> (Vec<f64>, Vec<f64>) {
        let sma = Self::sma(prices, period);
        let rolling_std = Self::rolling_std(prices, period);

        let upper: Vec<f64> = sma
            .iter()
            .zip(rolling_std.iter())
            .map(|(&m, &s)| m + std_dev * s)
            .collect();

        let lower: Vec<f64> = sma
            .iter()
            .zip(rolling_std.iter())
            .map(|(&m, &s)| m - std_dev * s)
            .collect();

        (upper, lower)
    }

    /// Momentum indicator
    pub fn momentum(prices: &[f64], period: usize) -> Vec<f64> {
        let n = prices.len();
        let mut result = vec![0.0; n];

        for i in period..n {
            result[i] = prices[i] - prices[i - period];
        }

        result
    }

    /// Rate of Change
    pub fn roc(prices: &[f64], period: usize) -> Vec<f64> {
        let n = prices.len();
        let mut result = vec![0.0; n];

        for i in period..n {
            if prices[i - period].abs() > 1e-10 {
                result[i] = ((prices[i] - prices[i - period]) / prices[i - period]) * 100.0;
            }
        }

        result
    }

    /// Convert features to ndarray format
    pub fn to_ndarray(features: &[Vec<f64>]) -> Array2<f64> {
        if features.is_empty() || features[0].is_empty() {
            return Array2::zeros((0, 0));
        }

        let n_samples = features[0].len();
        let n_features = features.len();

        let mut array = Array2::zeros((n_samples, n_features));

        for (j, feature) in features.iter().enumerate() {
            for (i, &val) in feature.iter().enumerate() {
                array[[i, j]] = val;
            }
        }

        array
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_klines() -> Vec<Kline> {
        vec![
            Kline { timestamp: 0, open: 100.0, high: 105.0, low: 98.0, close: 102.0, volume: 1000.0, turnover: 100000.0 },
            Kline { timestamp: 1, open: 102.0, high: 108.0, low: 101.0, close: 107.0, volume: 1200.0, turnover: 120000.0 },
            Kline { timestamp: 2, open: 107.0, high: 110.0, low: 105.0, close: 106.0, volume: 800.0, turnover: 85000.0 },
            Kline { timestamp: 3, open: 106.0, high: 109.0, low: 103.0, close: 104.0, volume: 900.0, turnover: 93000.0 },
            Kline { timestamp: 4, open: 104.0, high: 107.0, low: 102.0, close: 105.0, volume: 1100.0, turnover: 115000.0 },
        ]
    }

    #[test]
    fn test_sma() {
        let prices = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sma = FeatureEngineering::sma(&prices, 3);

        assert!((sma[2] - 2.0).abs() < 1e-10); // (1+2+3)/3 = 2
        assert!((sma[3] - 3.0).abs() < 1e-10); // (2+3+4)/3 = 3
        assert!((sma[4] - 4.0).abs() < 1e-10); // (3+4+5)/3 = 4
    }

    #[test]
    fn test_returns() {
        let prices = vec![100.0, 110.0, 105.0, 115.0];
        let returns = FeatureEngineering::returns(&prices, 1);

        assert!((returns[1] - 0.1).abs() < 1e-10);      // 110/100 - 1 = 0.1
        assert!((returns[2] - (-0.0454545)).abs() < 0.001); // 105/110 - 1
    }

    #[test]
    fn test_generate_features() {
        let klines = create_test_klines();
        let (features, names) = FeatureEngineering::generate_features(&klines);

        assert_eq!(features.nrows(), klines.len());
        assert_eq!(features.ncols(), names.len());
        assert!(names.contains(&"rsi_14".to_string()));
        assert!(names.contains(&"volatility_10".to_string()));
    }
}
