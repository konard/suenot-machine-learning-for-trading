//! Pattern encoding for market data
//!
//! This module provides utilities for encoding market data (candles, indicators)
//! into pattern vectors suitable for Hopfield Network processing.

use crate::data::Candle;
use super::{ema, rsi, sma, std_dev, volatility, MarketRegime};

/// Pattern encoder configuration
#[derive(Debug, Clone)]
pub struct PatternEncoderConfig {
    /// Window size for pattern encoding
    pub window_size: usize,
    /// Whether to include volume in patterns
    pub include_volume: bool,
    /// Whether to include technical indicators
    pub include_indicators: bool,
    /// RSI period
    pub rsi_period: usize,
    /// SMA short period
    pub sma_short: usize,
    /// SMA long period
    pub sma_long: usize,
    /// Normalization method
    pub normalization: NormalizationMethod,
}

impl Default for PatternEncoderConfig {
    fn default() -> Self {
        Self {
            window_size: 24,
            include_volume: true,
            include_indicators: true,
            rsi_period: 14,
            sma_short: 7,
            sma_long: 25,
            normalization: NormalizationMethod::ZScore,
        }
    }
}

/// Normalization methods
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormalizationMethod {
    /// Min-max normalization to [0, 1]
    MinMax,
    /// Z-score normalization (mean=0, std=1)
    ZScore,
    /// Percentage change from first value
    PercentChange,
    /// No normalization
    None,
}

/// Pattern encoder for converting market data to pattern vectors
///
/// # Example
///
/// ```rust
/// use hopfield_trading::trading::patterns::{PatternEncoder, PatternEncoderConfig};
/// use hopfield_trading::data::Candle;
/// use chrono::Utc;
///
/// let config = PatternEncoderConfig::default();
/// let encoder = PatternEncoder::new(config);
///
/// // Create some sample candles
/// let candles: Vec<Candle> = (0..30).map(|i| {
///     Candle::new(
///         Utc::now(),
///         100.0 + i as f64,
///         102.0 + i as f64,
///         98.0 + i as f64,
///         101.0 + i as f64,
///         1000.0,
///         100000.0,
///     )
/// }).collect();
///
/// let patterns = encoder.encode_candles(&candles);
/// ```
#[derive(Debug, Clone)]
pub struct PatternEncoder {
    config: PatternEncoderConfig,
}

impl Default for PatternEncoder {
    fn default() -> Self {
        Self::new(PatternEncoderConfig::default())
    }
}

impl PatternEncoder {
    /// Create a new pattern encoder with custom configuration
    pub fn new(config: PatternEncoderConfig) -> Self {
        Self { config }
    }

    /// Get the configuration
    pub fn config(&self) -> &PatternEncoderConfig {
        &self.config
    }

    /// Encode multiple candles into a single pattern
    pub fn encode_window(&self, candles: &[Candle]) -> Vec<f64> {
        if candles.is_empty() {
            return Vec::new();
        }

        let mut pattern = Vec::new();

        // Encode OHLC for each candle
        for candle in candles {
            pattern.push(candle.open);
            pattern.push(candle.high);
            pattern.push(candle.low);
            pattern.push(candle.close);

            if self.config.include_volume {
                pattern.push(candle.volume);
            }
        }

        // Add technical indicators if enabled
        if self.config.include_indicators && candles.len() >= self.config.rsi_period + 1 {
            let closes: Vec<f64> = candles.iter().map(|c| c.close).collect();

            // RSI
            if let Some(rsi_val) = rsi(&closes, self.config.rsi_period) {
                pattern.push(rsi_val / 100.0); // Normalize to [0, 1]
            }

            // SMA ratio
            if let (Some(short), Some(long)) = (
                sma(&closes, self.config.sma_short),
                sma(&closes, self.config.sma_long),
            ) {
                pattern.push(short / long - 1.0); // Relative difference
            }

            // Volatility
            let vol = volatility(&closes);
            pattern.push(vol);
        }

        // Apply normalization
        self.normalize(&pattern)
    }

    /// Encode a series of candles into multiple patterns using sliding window
    pub fn encode_candles(&self, candles: &[Candle]) -> Vec<Vec<f64>> {
        if candles.len() < self.config.window_size {
            return Vec::new();
        }

        candles
            .windows(self.config.window_size)
            .map(|window| self.encode_window(window))
            .collect()
    }

    /// Encode a single candle into a pattern
    pub fn encode_candle(&self, candle: &Candle) -> Vec<f64> {
        self.encode_window(&[candle.clone()])
    }

    /// Normalize a pattern according to configuration
    pub fn normalize(&self, pattern: &[f64]) -> Vec<f64> {
        if pattern.is_empty() {
            return Vec::new();
        }

        match self.config.normalization {
            NormalizationMethod::MinMax => {
                let min = pattern.iter().cloned().fold(f64::INFINITY, f64::min);
                let max = pattern.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let range = (max - min).max(1e-10);

                pattern.iter().map(|&x| (x - min) / range).collect()
            }
            NormalizationMethod::ZScore => {
                let mean: f64 = pattern.iter().sum::<f64>() / pattern.len() as f64;
                let std = std_dev(pattern).max(1e-10);

                pattern.iter().map(|&x| (x - mean) / std).collect()
            }
            NormalizationMethod::PercentChange => {
                let first = pattern[0].abs().max(1e-10);
                pattern.iter().map(|&x| (x - pattern[0]) / first).collect()
            }
            NormalizationMethod::None => pattern.to_vec(),
        }
    }

    /// Calculate pattern dimension for given configuration
    pub fn pattern_dimension(&self) -> usize {
        let base_features = if self.config.include_volume { 5 } else { 4 };
        let indicator_features = if self.config.include_indicators { 3 } else { 0 };

        self.config.window_size * base_features + indicator_features
    }

    /// Encode close prices only (simplified pattern)
    pub fn encode_closes(&self, candles: &[Candle]) -> Vec<f64> {
        let closes: Vec<f64> = candles.iter().map(|c| c.close).collect();
        self.normalize(&closes)
    }

    /// Encode returns (percentage changes)
    pub fn encode_returns(&self, candles: &[Candle]) -> Vec<f64> {
        if candles.len() < 2 {
            return Vec::new();
        }

        let returns: Vec<f64> = candles
            .windows(2)
            .map(|w| (w[1].close / w[0].close) - 1.0)
            .collect();

        returns
    }

    /// Detect market regime from candles
    pub fn detect_regime(&self, candles: &[Candle]) -> MarketRegime {
        if candles.len() < self.config.sma_long {
            return MarketRegime::Unknown;
        }

        let closes: Vec<f64> = candles.iter().map(|c| c.close).collect();

        // Calculate indicators
        let sma_short = sma(&closes, self.config.sma_short).unwrap_or(0.0);
        let sma_long = sma(&closes, self.config.sma_long).unwrap_or(0.0);
        let vol = volatility(&closes);
        let rsi_val = rsi(&closes, self.config.rsi_period).unwrap_or(50.0);

        // Determine volatility regime
        let vol_threshold = 0.02; // 2% daily volatility threshold

        if vol > vol_threshold * 2.0 {
            return MarketRegime::HighVolatility;
        }

        if vol < vol_threshold * 0.5 {
            return MarketRegime::LowVolatility;
        }

        // Determine trend regime
        let trend_strength = (sma_short - sma_long) / sma_long;

        if trend_strength > 0.02 && rsi_val > 50.0 {
            return MarketRegime::BullTrend;
        }

        if trend_strength < -0.02 && rsi_val < 50.0 {
            return MarketRegime::BearTrend;
        }

        if trend_strength.abs() < 0.01 && (40.0..60.0).contains(&rsi_val) {
            return MarketRegime::Ranging;
        }

        MarketRegime::Unknown
    }
}

/// Labeled pattern for supervised learning
#[derive(Debug, Clone)]
pub struct LabeledPattern {
    /// The pattern vector
    pub pattern: Vec<f64>,
    /// Label (e.g., future return direction)
    pub label: PatternLabel,
    /// Confidence in the label
    pub confidence: f64,
    /// Original timestamp
    pub timestamp: Option<chrono::DateTime<chrono::Utc>>,
}

/// Pattern labels for classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PatternLabel {
    /// Strong bullish signal
    StrongBuy,
    /// Weak bullish signal
    Buy,
    /// Neutral signal
    Hold,
    /// Weak bearish signal
    Sell,
    /// Strong bearish signal
    StrongSell,
}

impl PatternLabel {
    /// Convert to numeric value
    pub fn to_value(&self) -> f64 {
        match self {
            PatternLabel::StrongBuy => 1.0,
            PatternLabel::Buy => 0.5,
            PatternLabel::Hold => 0.0,
            PatternLabel::Sell => -0.5,
            PatternLabel::StrongSell => -1.0,
        }
    }

    /// Create from return value
    pub fn from_return(ret: f64, threshold: f64) -> Self {
        if ret > threshold * 2.0 {
            PatternLabel::StrongBuy
        } else if ret > threshold {
            PatternLabel::Buy
        } else if ret < -threshold * 2.0 {
            PatternLabel::StrongSell
        } else if ret < -threshold {
            PatternLabel::Sell
        } else {
            PatternLabel::Hold
        }
    }
}

/// Create labeled patterns from candles with lookahead
pub fn create_labeled_patterns(
    candles: &[Candle],
    encoder: &PatternEncoder,
    lookahead: usize,
    threshold: f64,
) -> Vec<LabeledPattern> {
    if candles.len() < encoder.config().window_size + lookahead {
        return Vec::new();
    }

    let mut labeled = Vec::new();

    for i in 0..candles.len() - encoder.config().window_size - lookahead {
        let window_start = i;
        let window_end = i + encoder.config().window_size;
        let future_idx = window_end + lookahead - 1;

        let window = &candles[window_start..window_end];
        let pattern = encoder.encode_window(window);

        // Calculate future return
        let current_close = candles[window_end - 1].close;
        let future_close = candles[future_idx].close;
        let ret = (future_close - current_close) / current_close;

        let label = PatternLabel::from_return(ret, threshold);

        labeled.push(LabeledPattern {
            pattern,
            label,
            confidence: 1.0 - (ret.abs() / threshold).min(1.0) * 0.5,
            timestamp: Some(candles[window_end - 1].timestamp),
        });
    }

    labeled
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn make_candles(n: usize) -> Vec<Candle> {
        (0..n)
            .map(|i| {
                Candle::new(
                    Utc::now(),
                    100.0 + i as f64,
                    102.0 + i as f64,
                    98.0 + i as f64,
                    101.0 + i as f64,
                    1000.0 + i as f64 * 10.0,
                    100000.0,
                )
            })
            .collect()
    }

    #[test]
    fn test_encode_window() {
        let config = PatternEncoderConfig {
            window_size: 5,
            include_volume: true,
            include_indicators: false,
            ..Default::default()
        };
        let encoder = PatternEncoder::new(config);
        let candles = make_candles(5);

        let pattern = encoder.encode_window(&candles);
        assert!(!pattern.is_empty());
    }

    #[test]
    fn test_encode_candles() {
        let config = PatternEncoderConfig {
            window_size: 5,
            include_volume: false,
            include_indicators: false,
            ..Default::default()
        };
        let encoder = PatternEncoder::new(config);
        let candles = make_candles(10);

        let patterns = encoder.encode_candles(&candles);
        assert_eq!(patterns.len(), 6); // 10 - 5 + 1 = 6 windows
    }

    #[test]
    fn test_normalization() {
        let encoder = PatternEncoder::default();

        let pattern = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let normalized = encoder.normalize(&pattern);

        // Check that z-score normalized has mean ~0
        let mean: f64 = normalized.iter().sum::<f64>() / normalized.len() as f64;
        assert!(mean.abs() < 1e-10);
    }

    #[test]
    fn test_pattern_label() {
        assert_eq!(PatternLabel::from_return(0.05, 0.01), PatternLabel::StrongBuy);
        assert_eq!(PatternLabel::from_return(0.015, 0.01), PatternLabel::Buy);
        assert_eq!(PatternLabel::from_return(0.005, 0.01), PatternLabel::Hold);
        assert_eq!(PatternLabel::from_return(-0.015, 0.01), PatternLabel::Sell);
        assert_eq!(PatternLabel::from_return(-0.05, 0.01), PatternLabel::StrongSell);
    }
}
