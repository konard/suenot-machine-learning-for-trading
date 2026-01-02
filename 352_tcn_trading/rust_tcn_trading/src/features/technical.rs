//! Technical Indicators Implementation

use crate::api::Candle;
use ndarray::Array2;

/// Feature matrix containing all calculated technical indicators
#[derive(Debug, Clone)]
pub struct FeatureMatrix {
    /// Feature names
    pub feature_names: Vec<String>,
    /// Feature values [features x time]
    pub data: Array2<f64>,
    /// Number of features
    pub num_features: usize,
    /// Sequence length
    pub seq_len: usize,
}

impl FeatureMatrix {
    /// Create a new feature matrix
    pub fn new(features: Vec<(String, Vec<f64>)>) -> Self {
        if features.is_empty() {
            return Self {
                feature_names: vec![],
                data: Array2::zeros((0, 0)),
                num_features: 0,
                seq_len: 0,
            };
        }

        let num_features = features.len();
        let seq_len = features[0].1.len();

        let mut data = Array2::zeros((num_features, seq_len));
        let mut feature_names = Vec::with_capacity(num_features);

        for (i, (name, values)) in features.into_iter().enumerate() {
            feature_names.push(name);
            for (j, &val) in values.iter().enumerate() {
                data[[i, j]] = val;
            }
        }

        Self {
            feature_names,
            data,
            num_features,
            seq_len,
        }
    }

    /// Get feature by name
    pub fn get_feature(&self, name: &str) -> Option<Vec<f64>> {
        self.feature_names
            .iter()
            .position(|n| n == name)
            .map(|idx| self.data.row(idx).to_vec())
    }

    /// Get a window of features for model input
    pub fn window(&self, start: usize, end: usize) -> Array2<f64> {
        self.data.slice(ndarray::s![.., start..end]).to_owned()
    }
}

/// Technical indicators calculator
#[derive(Debug, Default)]
pub struct TechnicalIndicators;

impl TechnicalIndicators {
    /// Calculate all standard technical indicators
    pub fn calculate_all(candles: &[Candle]) -> FeatureMatrix {
        let closes: Vec<f64> = candles.iter().map(|c| c.close).collect();
        let highs: Vec<f64> = candles.iter().map(|c| c.high).collect();
        let lows: Vec<f64> = candles.iter().map(|c| c.low).collect();
        let volumes: Vec<f64> = candles.iter().map(|c| c.volume).collect();

        let mut features = Vec::new();

        // Returns
        features.push(("returns".to_string(), Self::returns(&closes)));
        features.push(("log_returns".to_string(), Self::log_returns(&closes)));

        // Moving averages
        features.push(("sma_10".to_string(), Self::sma(&closes, 10)));
        features.push(("sma_20".to_string(), Self::sma(&closes, 20)));
        features.push(("sma_50".to_string(), Self::sma(&closes, 50)));
        features.push(("ema_12".to_string(), Self::ema(&closes, 12)));
        features.push(("ema_26".to_string(), Self::ema(&closes, 26)));

        // RSI
        features.push(("rsi_14".to_string(), Self::rsi(&closes, 14)));
        features.push(("rsi_7".to_string(), Self::rsi(&closes, 7)));

        // MACD
        let (macd, signal, histogram) = Self::macd(&closes, 12, 26, 9);
        features.push(("macd".to_string(), macd));
        features.push(("macd_signal".to_string(), signal));
        features.push(("macd_histogram".to_string(), histogram));

        // Bollinger Bands
        let (upper, middle, lower) = Self::bollinger_bands(&closes, 20, 2.0);
        features.push(("bb_upper".to_string(), upper));
        features.push(("bb_middle".to_string(), middle));
        features.push(("bb_lower".to_string(), lower));
        features.push(("bb_width".to_string(), Self::bollinger_width(&closes, 20, 2.0)));

        // ATR
        features.push(("atr_14".to_string(), Self::atr(&highs, &lows, &closes, 14)));

        // Momentum
        features.push(("momentum_10".to_string(), Self::momentum(&closes, 10)));
        features.push(("roc_10".to_string(), Self::rate_of_change(&closes, 10)));

        // Volume indicators
        features.push(("volume_sma".to_string(), Self::sma(&volumes, 20)));
        features.push(("volume_ratio".to_string(), Self::volume_ratio(&volumes, 20)));
        features.push(("obv".to_string(), Self::obv(&closes, &volumes)));

        // Volatility
        features.push(("volatility_10".to_string(), Self::volatility(&closes, 10)));
        features.push(("volatility_20".to_string(), Self::volatility(&closes, 20)));

        // Price position
        features.push(("close".to_string(), closes.clone()));
        features.push(("high_low_ratio".to_string(), Self::high_low_ratio(&highs, &lows)));

        FeatureMatrix::new(features)
    }

    /// Calculate simple returns
    pub fn returns(prices: &[f64]) -> Vec<f64> {
        let mut result = vec![0.0];
        for i in 1..prices.len() {
            result.push((prices[i] - prices[i - 1]) / prices[i - 1]);
        }
        result
    }

    /// Calculate log returns
    pub fn log_returns(prices: &[f64]) -> Vec<f64> {
        let mut result = vec![0.0];
        for i in 1..prices.len() {
            result.push((prices[i] / prices[i - 1]).ln());
        }
        result
    }

    /// Calculate Simple Moving Average
    pub fn sma(data: &[f64], period: usize) -> Vec<f64> {
        let mut result = vec![f64::NAN; data.len()];
        if data.len() < period {
            return result;
        }

        for i in (period - 1)..data.len() {
            let sum: f64 = data[i + 1 - period..=i].iter().sum();
            result[i] = sum / period as f64;
        }
        result
    }

    /// Calculate Exponential Moving Average
    pub fn ema(data: &[f64], period: usize) -> Vec<f64> {
        let mut result = vec![f64::NAN; data.len()];
        if data.len() < period {
            return result;
        }

        let multiplier = 2.0 / (period as f64 + 1.0);

        // Initialize with SMA
        let sum: f64 = data[..period].iter().sum();
        result[period - 1] = sum / period as f64;

        // Calculate EMA
        for i in period..data.len() {
            result[i] = (data[i] - result[i - 1]) * multiplier + result[i - 1];
        }
        result
    }

    /// Calculate Relative Strength Index
    pub fn rsi(prices: &[f64], period: usize) -> Vec<f64> {
        let mut result = vec![f64::NAN; prices.len()];
        if prices.len() < period + 1 {
            return result;
        }

        let mut gains = vec![0.0; prices.len()];
        let mut losses = vec![0.0; prices.len()];

        for i in 1..prices.len() {
            let change = prices[i] - prices[i - 1];
            if change > 0.0 {
                gains[i] = change;
            } else {
                losses[i] = -change;
            }
        }

        // First average
        let avg_gain: f64 = gains[1..=period].iter().sum::<f64>() / period as f64;
        let avg_loss: f64 = losses[1..=period].iter().sum::<f64>() / period as f64;

        let mut prev_avg_gain = avg_gain;
        let mut prev_avg_loss = avg_loss;

        for i in period..prices.len() {
            let curr_avg_gain = (prev_avg_gain * (period - 1) as f64 + gains[i]) / period as f64;
            let curr_avg_loss = (prev_avg_loss * (period - 1) as f64 + losses[i]) / period as f64;

            if curr_avg_loss == 0.0 {
                result[i] = 100.0;
            } else {
                let rs = curr_avg_gain / curr_avg_loss;
                result[i] = 100.0 - 100.0 / (1.0 + rs);
            }

            prev_avg_gain = curr_avg_gain;
            prev_avg_loss = curr_avg_loss;
        }
        result
    }

    /// Calculate MACD
    pub fn macd(prices: &[f64], fast: usize, slow: usize, signal: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let ema_fast = Self::ema(prices, fast);
        let ema_slow = Self::ema(prices, slow);

        let mut macd_line = vec![f64::NAN; prices.len()];
        for i in 0..prices.len() {
            if !ema_fast[i].is_nan() && !ema_slow[i].is_nan() {
                macd_line[i] = ema_fast[i] - ema_slow[i];
            }
        }

        // Filter out NaN for signal calculation
        let valid_macd: Vec<f64> = macd_line.iter().filter(|x| !x.is_nan()).cloned().collect();
        let signal_line_valid = Self::ema(&valid_macd, signal);

        let mut signal_line = vec![f64::NAN; prices.len()];
        let mut valid_idx = 0;
        for i in 0..prices.len() {
            if !macd_line[i].is_nan() {
                if valid_idx < signal_line_valid.len() {
                    signal_line[i] = signal_line_valid[valid_idx];
                }
                valid_idx += 1;
            }
        }

        let mut histogram = vec![f64::NAN; prices.len()];
        for i in 0..prices.len() {
            if !macd_line[i].is_nan() && !signal_line[i].is_nan() {
                histogram[i] = macd_line[i] - signal_line[i];
            }
        }

        (macd_line, signal_line, histogram)
    }

    /// Calculate Bollinger Bands
    pub fn bollinger_bands(prices: &[f64], period: usize, std_dev: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let middle = Self::sma(prices, period);
        let mut upper = vec![f64::NAN; prices.len()];
        let mut lower = vec![f64::NAN; prices.len()];

        for i in (period - 1)..prices.len() {
            let slice = &prices[i + 1 - period..=i];
            let mean = middle[i];
            let variance: f64 = slice.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / period as f64;
            let std = variance.sqrt();

            upper[i] = mean + std_dev * std;
            lower[i] = mean - std_dev * std;
        }

        (upper, middle, lower)
    }

    /// Calculate Bollinger Band width
    pub fn bollinger_width(prices: &[f64], period: usize, std_dev: f64) -> Vec<f64> {
        let (upper, middle, lower) = Self::bollinger_bands(prices, period, std_dev);
        let mut width = vec![f64::NAN; prices.len()];
        for i in 0..prices.len() {
            if !upper[i].is_nan() && !lower[i].is_nan() && !middle[i].is_nan() && middle[i] != 0.0 {
                width[i] = (upper[i] - lower[i]) / middle[i];
            }
        }
        width
    }

    /// Calculate Average True Range
    pub fn atr(highs: &[f64], lows: &[f64], closes: &[f64], period: usize) -> Vec<f64> {
        let mut tr = vec![0.0; highs.len()];
        tr[0] = highs[0] - lows[0];

        for i in 1..highs.len() {
            let hl = highs[i] - lows[i];
            let hc = (highs[i] - closes[i - 1]).abs();
            let lc = (lows[i] - closes[i - 1]).abs();
            tr[i] = hl.max(hc).max(lc);
        }

        Self::sma(&tr, period)
    }

    /// Calculate momentum
    pub fn momentum(prices: &[f64], period: usize) -> Vec<f64> {
        let mut result = vec![f64::NAN; prices.len()];
        for i in period..prices.len() {
            result[i] = prices[i] - prices[i - period];
        }
        result
    }

    /// Calculate rate of change
    pub fn rate_of_change(prices: &[f64], period: usize) -> Vec<f64> {
        let mut result = vec![f64::NAN; prices.len()];
        for i in period..prices.len() {
            result[i] = (prices[i] - prices[i - period]) / prices[i - period] * 100.0;
        }
        result
    }

    /// Calculate volume ratio
    pub fn volume_ratio(volumes: &[f64], period: usize) -> Vec<f64> {
        let sma = Self::sma(volumes, period);
        let mut result = vec![f64::NAN; volumes.len()];
        for i in 0..volumes.len() {
            if !sma[i].is_nan() && sma[i] != 0.0 {
                result[i] = volumes[i] / sma[i];
            }
        }
        result
    }

    /// Calculate On-Balance Volume
    pub fn obv(closes: &[f64], volumes: &[f64]) -> Vec<f64> {
        let mut result = vec![0.0; closes.len()];
        result[0] = volumes[0];

        for i in 1..closes.len() {
            if closes[i] > closes[i - 1] {
                result[i] = result[i - 1] + volumes[i];
            } else if closes[i] < closes[i - 1] {
                result[i] = result[i - 1] - volumes[i];
            } else {
                result[i] = result[i - 1];
            }
        }
        result
    }

    /// Calculate rolling volatility (standard deviation of returns)
    pub fn volatility(prices: &[f64], period: usize) -> Vec<f64> {
        let returns = Self::log_returns(prices);
        let mut result = vec![f64::NAN; prices.len()];

        for i in period..prices.len() {
            let slice = &returns[i + 1 - period..=i];
            let mean: f64 = slice.iter().sum::<f64>() / period as f64;
            let variance: f64 = slice.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / period as f64;
            result[i] = variance.sqrt();
        }
        result
    }

    /// Calculate high/low ratio
    pub fn high_low_ratio(highs: &[f64], lows: &[f64]) -> Vec<f64> {
        highs
            .iter()
            .zip(lows.iter())
            .map(|(h, l)| if *l != 0.0 { h / l } else { 1.0 })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn create_test_candles() -> Vec<Candle> {
        (0..100)
            .map(|i| {
                let price = 100.0 + (i as f64 * 0.1).sin() * 10.0;
                Candle::new(
                    Utc::now(),
                    price - 0.5,
                    price + 1.0,
                    price - 1.0,
                    price,
                    1000.0 + i as f64 * 10.0,
                    price * 1000.0,
                )
            })
            .collect()
    }

    #[test]
    fn test_sma() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sma = TechnicalIndicators::sma(&data, 3);
        assert!((sma[2] - 2.0).abs() < 0.001);
        assert!((sma[3] - 3.0).abs() < 0.001);
        assert!((sma[4] - 4.0).abs() < 0.001);
    }

    #[test]
    fn test_rsi() {
        let prices: Vec<f64> = (0..50).map(|i| 100.0 + i as f64).collect();
        let rsi = TechnicalIndicators::rsi(&prices, 14);
        // Monotonically increasing prices should have RSI close to 100
        assert!(rsi[49] > 90.0);
    }

    #[test]
    fn test_calculate_all() {
        let candles = create_test_candles();
        let features = TechnicalIndicators::calculate_all(&candles);

        assert!(features.num_features > 0);
        assert_eq!(features.seq_len, 100);
        assert!(features.get_feature("rsi_14").is_some());
        assert!(features.get_feature("macd").is_some());
    }
}
