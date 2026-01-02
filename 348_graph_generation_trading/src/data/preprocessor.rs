//! Data preprocessing utilities for market data.
//!
//! Provides functions for calculating returns, normalizing data,
//! and preparing data for graph construction.

use super::OHLCV;

/// Calculate log returns from OHLCV data
///
/// # Arguments
///
/// * `candles` - Slice of OHLCV candles
///
/// # Returns
///
/// Vector of log returns (length = candles.len() - 1)
pub fn calculate_returns(candles: &[OHLCV]) -> Vec<f64> {
    if candles.len() < 2 {
        return Vec::new();
    }

    candles
        .windows(2)
        .map(|w| (w[1].close / w[0].close).ln())
        .collect()
}

/// Calculate simple returns from OHLCV data
pub fn calculate_simple_returns(candles: &[OHLCV]) -> Vec<f64> {
    if candles.len() < 2 {
        return Vec::new();
    }

    candles
        .windows(2)
        .map(|w| (w[1].close - w[0].close) / w[0].close)
        .collect()
}

/// Normalize returns to have zero mean and unit variance
///
/// # Arguments
///
/// * `returns` - Slice of return values
///
/// # Returns
///
/// Tuple of (normalized_returns, mean, std_dev)
pub fn normalize_returns(returns: &[f64]) -> (Vec<f64>, f64, f64) {
    if returns.is_empty() {
        return (Vec::new(), 0.0, 1.0);
    }

    let n = returns.len() as f64;
    let mean: f64 = returns.iter().sum::<f64>() / n;

    let variance: f64 = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;
    let std_dev = variance.sqrt();

    let normalized = if std_dev > 1e-10 {
        returns.iter().map(|r| (r - mean) / std_dev).collect()
    } else {
        returns.iter().map(|r| r - mean).collect()
    };

    (normalized, mean, std_dev)
}

/// Data preprocessor with configurable options
#[derive(Debug, Clone)]
pub struct DataPreprocessor {
    /// Whether to use log returns (vs simple returns)
    pub use_log_returns: bool,
    /// Whether to normalize the data
    pub normalize: bool,
    /// Rolling window for statistics (None = use all data)
    pub rolling_window: Option<usize>,
    /// Minimum number of valid observations required
    pub min_observations: usize,
}

impl Default for DataPreprocessor {
    fn default() -> Self {
        Self {
            use_log_returns: true,
            normalize: true,
            rolling_window: None,
            min_observations: 20,
        }
    }
}

impl DataPreprocessor {
    /// Create a new DataPreprocessor with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Set whether to use log returns
    pub fn with_log_returns(mut self, use_log: bool) -> Self {
        self.use_log_returns = use_log;
        self
    }

    /// Set whether to normalize the data
    pub fn with_normalization(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    /// Set rolling window size
    pub fn with_rolling_window(mut self, window: usize) -> Self {
        self.rolling_window = Some(window);
        self
    }

    /// Set minimum observations
    pub fn with_min_observations(mut self, min_obs: usize) -> Self {
        self.min_observations = min_obs;
        self
    }

    /// Process OHLCV candles into returns
    pub fn process(&self, candles: &[OHLCV]) -> Vec<f64> {
        let returns = if self.use_log_returns {
            calculate_returns(candles)
        } else {
            calculate_simple_returns(candles)
        };

        if self.normalize {
            let (normalized, _, _) = normalize_returns(&returns);
            normalized
        } else {
            returns
        }
    }

    /// Process multiple symbol data
    pub fn process_multi(&self, data: &[Vec<OHLCV>]) -> Vec<Vec<f64>> {
        data.iter().map(|candles| self.process(candles)).collect()
    }

    /// Calculate rolling returns with specified window
    pub fn rolling_returns(&self, candles: &[OHLCV], window: usize) -> Vec<f64> {
        if candles.len() < window + 1 {
            return Vec::new();
        }

        let all_returns = if self.use_log_returns {
            calculate_returns(candles)
        } else {
            calculate_simple_returns(candles)
        };

        // Calculate cumulative return over window
        all_returns
            .windows(window)
            .map(|w| w.iter().sum::<f64>())
            .collect()
    }
}

/// Calculate volatility (standard deviation of returns)
pub fn calculate_volatility(candles: &[OHLCV]) -> f64 {
    let returns = calculate_returns(candles);
    if returns.is_empty() {
        return 0.0;
    }

    let n = returns.len() as f64;
    let mean: f64 = returns.iter().sum::<f64>() / n;
    let variance: f64 = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n - 1.0);

    variance.sqrt()
}

/// Calculate annualized volatility
pub fn calculate_annualized_volatility(candles: &[OHLCV], periods_per_year: f64) -> f64 {
    calculate_volatility(candles) * periods_per_year.sqrt()
}

/// Calculate maximum drawdown
pub fn calculate_max_drawdown(candles: &[OHLCV]) -> f64 {
    if candles.is_empty() {
        return 0.0;
    }

    let mut max_price = candles[0].close;
    let mut max_drawdown = 0.0;

    for candle in candles.iter() {
        if candle.close > max_price {
            max_price = candle.close;
        }
        let drawdown = (max_price - candle.close) / max_price;
        if drawdown > max_drawdown {
            max_drawdown = drawdown;
        }
    }

    max_drawdown
}

/// Align multiple time series to common timestamps
pub fn align_data(data: &[Vec<OHLCV>]) -> Vec<Vec<OHLCV>> {
    if data.is_empty() {
        return Vec::new();
    }

    // Find minimum length
    let min_len = data.iter().map(|d| d.len()).min().unwrap_or(0);

    // Take last min_len candles from each (assuming they're aligned at the end)
    data.iter()
        .map(|candles| {
            let start = candles.len().saturating_sub(min_len);
            candles[start..].to_vec()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn create_test_candles() -> Vec<OHLCV> {
        vec![
            OHLCV::new(Utc::now(), 100.0, 105.0, 98.0, 102.0, 1000.0),
            OHLCV::new(Utc::now(), 102.0, 108.0, 100.0, 106.0, 1100.0),
            OHLCV::new(Utc::now(), 106.0, 110.0, 104.0, 104.0, 900.0),
            OHLCV::new(Utc::now(), 104.0, 107.0, 102.0, 105.0, 950.0),
        ]
    }

    #[test]
    fn test_calculate_returns() {
        let candles = create_test_candles();
        let returns = calculate_returns(&candles);

        assert_eq!(returns.len(), 3);

        // First return: ln(106/102)
        let expected_first = (106.0_f64 / 102.0).ln();
        assert!((returns[0] - expected_first).abs() < 1e-10);
    }

    #[test]
    fn test_calculate_simple_returns() {
        let candles = create_test_candles();
        let returns = calculate_simple_returns(&candles);

        assert_eq!(returns.len(), 3);

        // First return: (106 - 102) / 102
        let expected_first = (106.0 - 102.0) / 102.0;
        assert!((returns[0] - expected_first).abs() < 1e-10);
    }

    #[test]
    fn test_normalize_returns() {
        let returns = vec![0.01, 0.02, -0.01, 0.015, -0.005];
        let (normalized, mean, std) = normalize_returns(&returns);

        assert_eq!(normalized.len(), returns.len());

        // Check that normalized data has ~zero mean
        let norm_mean: f64 = normalized.iter().sum::<f64>() / normalized.len() as f64;
        assert!(norm_mean.abs() < 1e-10);
    }

    #[test]
    fn test_calculate_volatility() {
        let candles = create_test_candles();
        let vol = calculate_volatility(&candles);

        assert!(vol > 0.0);
        assert!(vol < 1.0); // Reasonable for typical price movements
    }

    #[test]
    fn test_max_drawdown() {
        let candles = vec![
            OHLCV::new(Utc::now(), 100.0, 105.0, 98.0, 100.0, 1000.0),
            OHLCV::new(Utc::now(), 100.0, 110.0, 98.0, 110.0, 1000.0), // New high
            OHLCV::new(Utc::now(), 110.0, 112.0, 88.0, 88.0, 1000.0),  // Drawdown: (110-88)/110 = 20%
            OHLCV::new(Utc::now(), 88.0, 95.0, 85.0, 95.0, 1000.0),
        ];

        let mdd = calculate_max_drawdown(&candles);
        assert!((mdd - 0.2).abs() < 1e-10); // 20% drawdown
    }

    #[test]
    fn test_preprocessor() {
        let candles = create_test_candles();
        let preprocessor = DataPreprocessor::new()
            .with_log_returns(true)
            .with_normalization(true);

        let processed = preprocessor.process(&candles);
        assert_eq!(processed.len(), 3);
    }
}
