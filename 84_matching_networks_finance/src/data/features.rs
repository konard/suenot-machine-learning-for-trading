//! Feature extraction for market data
//!
//! Converts raw OHLCV data into features suitable for the Matching Network.

use ndarray::{Array1, Array2};

/// OHLCV bar data
#[derive(Debug, Clone)]
pub struct OHLCVBar {
    /// Timestamp in milliseconds
    pub timestamp: i64,
    /// Opening price
    pub open: f64,
    /// Highest price
    pub high: f64,
    /// Lowest price
    pub low: f64,
    /// Closing price
    pub close: f64,
    /// Trading volume
    pub volume: f64,
}

impl OHLCVBar {
    /// Create a new OHLCV bar
    pub fn new(timestamp: i64, open: f64, high: f64, low: f64, close: f64, volume: f64) -> Self {
        Self {
            timestamp,
            open,
            high,
            low,
            close,
            volume,
        }
    }

    /// Calculate the bar's return
    pub fn return_pct(&self) -> f64 {
        if self.open > 0.0 {
            (self.close - self.open) / self.open
        } else {
            0.0
        }
    }

    /// Calculate the bar's range
    pub fn range(&self) -> f64 {
        self.high - self.low
    }

    /// Calculate the body size (absolute difference between open and close)
    pub fn body(&self) -> f64 {
        (self.close - self.open).abs()
    }

    /// Check if the bar is bullish (close > open)
    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }
}

/// Market features extracted from OHLCV data
#[derive(Debug, Clone)]
pub struct MarketFeatures {
    /// Raw feature vector
    pub features: Array1<f64>,
    /// Feature names
    pub feature_names: Vec<String>,
}

impl MarketFeatures {
    /// Get the number of features
    pub fn dim(&self) -> usize {
        self.features.len()
    }

    /// Convert to raw vector
    pub fn to_vec(&self) -> Vec<f64> {
        self.features.to_vec()
    }
}

/// Feature extractor for market data
#[derive(Debug, Clone)]
pub struct FeatureExtractor {
    /// Lookback period for calculations
    lookback: usize,
    /// Short-term moving average period
    ma_short: usize,
    /// Long-term moving average period
    ma_long: usize,
    /// RSI period
    rsi_period: usize,
    /// ATR period
    atr_period: usize,
}

impl FeatureExtractor {
    /// Create a new feature extractor with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with custom settings
    pub fn with_settings(
        lookback: usize,
        ma_short: usize,
        ma_long: usize,
        rsi_period: usize,
        atr_period: usize,
    ) -> Self {
        Self {
            lookback,
            ma_short,
            ma_long,
            rsi_period,
            atr_period,
        }
    }

    /// Extract features from OHLCV bars
    ///
    /// Returns a feature vector with the following components:
    /// - Returns (multiple timeframes)
    /// - Volatility measures
    /// - Trend indicators (MA crossovers)
    /// - Momentum indicators (RSI)
    /// - Volume features
    /// - Price position relative to range
    pub fn extract(&self, bars: &[OHLCVBar]) -> Option<MarketFeatures> {
        if bars.len() < self.lookback.max(self.ma_long) {
            return None;
        }

        let mut features = Vec::new();
        let mut feature_names = Vec::new();

        // Recent returns at different timeframes
        let returns = self.compute_returns(bars);
        features.extend(&returns);
        feature_names.extend(["return_1", "return_5", "return_10", "return_20"].map(String::from));

        // Volatility measures
        let volatility = self.compute_volatility(bars);
        features.extend(&volatility);
        feature_names.extend(["volatility_10", "volatility_20", "atr_ratio"].map(String::from));

        // Trend indicators
        let trend = self.compute_trend_indicators(bars);
        features.extend(&trend);
        feature_names.extend(["ma_cross", "trend_strength", "price_vs_ma"].map(String::from));

        // Momentum indicators
        let momentum = self.compute_momentum(bars);
        features.extend(&momentum);
        feature_names.extend(["rsi", "rsi_change", "momentum"].map(String::from));

        // Volume features
        let volume = self.compute_volume_features(bars);
        features.extend(&volume);
        feature_names.extend(["volume_ratio", "volume_trend"].map(String::from));

        // Price position
        let position = self.compute_price_position(bars);
        features.extend(&position);
        feature_names.extend(["price_position", "range_position"].map(String::from));

        // Candle patterns
        let candle = self.compute_candle_features(bars);
        features.extend(&candle);
        feature_names.extend(["body_ratio", "upper_shadow", "lower_shadow", "direction"].map(String::from));

        Some(MarketFeatures {
            features: Array1::from_vec(features),
            feature_names,
        })
    }

    /// Extract features for multiple windows
    pub fn extract_windows(&self, bars: &[OHLCVBar], window_size: usize, step: usize) -> Array2<f64> {
        let mut windows = Vec::new();

        let mut start = 0;
        while start + window_size <= bars.len() {
            let window = &bars[start..start + window_size];
            if let Some(features) = self.extract(window) {
                windows.push(features.features.to_vec());
            }
            start += step;
        }

        if windows.is_empty() {
            return Array2::zeros((0, 0));
        }

        let num_features = windows[0].len();
        let num_windows = windows.len();
        let flat: Vec<f64> = windows.into_iter().flatten().collect();

        Array2::from_shape_vec((num_windows, num_features), flat).unwrap()
    }

    /// Compute returns at different timeframes
    fn compute_returns(&self, bars: &[OHLCVBar]) -> Vec<f64> {
        let n = bars.len();
        let close = bars[n - 1].close;

        let return_1 = if n > 1 && bars[n - 2].close > 0.0 {
            (close - bars[n - 2].close) / bars[n - 2].close
        } else {
            0.0
        };

        let return_5 = if n > 5 && bars[n - 6].close > 0.0 {
            (close - bars[n - 6].close) / bars[n - 6].close
        } else {
            0.0
        };

        let return_10 = if n > 10 && bars[n - 11].close > 0.0 {
            (close - bars[n - 11].close) / bars[n - 11].close
        } else {
            0.0
        };

        let return_20 = if n > 20 && bars[n - 21].close > 0.0 {
            (close - bars[n - 21].close) / bars[n - 21].close
        } else {
            0.0
        };

        vec![return_1, return_5, return_10, return_20]
    }

    /// Compute volatility measures
    fn compute_volatility(&self, bars: &[OHLCVBar]) -> Vec<f64> {
        let n = bars.len();

        // 10-period volatility
        let vol_10 = if n >= 10 {
            self.compute_std_returns(&bars[n - 10..])
        } else {
            0.0
        };

        // 20-period volatility
        let vol_20 = if n >= 20 {
            self.compute_std_returns(&bars[n - 20..])
        } else {
            vol_10
        };

        // ATR ratio
        let atr = self.compute_atr(&bars[n.saturating_sub(self.atr_period)..]);
        let atr_ratio = if bars[n - 1].close > 0.0 {
            atr / bars[n - 1].close
        } else {
            0.0
        };

        vec![vol_10, vol_20, atr_ratio]
    }

    /// Compute standard deviation of returns
    fn compute_std_returns(&self, bars: &[OHLCVBar]) -> f64 {
        if bars.len() < 2 {
            return 0.0;
        }

        let returns: Vec<f64> = bars
            .windows(2)
            .map(|w| {
                if w[0].close > 0.0 {
                    (w[1].close - w[0].close) / w[0].close
                } else {
                    0.0
                }
            })
            .collect();

        let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance: f64 =
            returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;

        variance.sqrt()
    }

    /// Compute Average True Range
    fn compute_atr(&self, bars: &[OHLCVBar]) -> f64 {
        if bars.len() < 2 {
            return 0.0;
        }

        let true_ranges: Vec<f64> = bars
            .windows(2)
            .map(|w| {
                let high_low = w[1].high - w[1].low;
                let high_close = (w[1].high - w[0].close).abs();
                let low_close = (w[1].low - w[0].close).abs();
                high_low.max(high_close).max(low_close)
            })
            .collect();

        true_ranges.iter().sum::<f64>() / true_ranges.len() as f64
    }

    /// Compute trend indicators
    fn compute_trend_indicators(&self, bars: &[OHLCVBar]) -> Vec<f64> {
        let n = bars.len();
        let closes: Vec<f64> = bars.iter().map(|b| b.close).collect();

        // Moving averages
        let ma_short = self.compute_ma(&closes, self.ma_short);
        let ma_long = self.compute_ma(&closes, self.ma_long);

        // MA crossover signal (-1 to 1)
        let ma_cross = if ma_long > 0.0 {
            (ma_short - ma_long) / ma_long
        } else {
            0.0
        };

        // Trend strength (slope of MA)
        let ma_recent: Vec<f64> = if closes.len() >= 5 {
            closes[closes.len() - 5..]
                .windows(self.ma_short.min(5))
                .map(|w| w.iter().sum::<f64>() / w.len() as f64)
                .collect()
        } else {
            vec![closes[closes.len() - 1]]
        };

        let trend_strength = if ma_recent.len() >= 2 {
            let first = ma_recent[0];
            let last = ma_recent[ma_recent.len() - 1];
            if first > 0.0 {
                (last - first) / first
            } else {
                0.0
            }
        } else {
            0.0
        };

        // Price vs MA
        let price_vs_ma = if ma_long > 0.0 {
            (closes[n - 1] - ma_long) / ma_long
        } else {
            0.0
        };

        vec![ma_cross, trend_strength, price_vs_ma]
    }

    /// Compute simple moving average
    fn compute_ma(&self, values: &[f64], period: usize) -> f64 {
        if values.len() < period {
            return values.iter().sum::<f64>() / values.len() as f64;
        }
        values[values.len() - period..].iter().sum::<f64>() / period as f64
    }

    /// Compute momentum indicators
    fn compute_momentum(&self, bars: &[OHLCVBar]) -> Vec<f64> {
        let closes: Vec<f64> = bars.iter().map(|b| b.close).collect();

        // RSI
        let rsi = self.compute_rsi(&closes);

        // RSI change
        let rsi_prev = if closes.len() > 5 {
            self.compute_rsi(&closes[..closes.len() - 5])
        } else {
            rsi
        };
        let rsi_change = rsi - rsi_prev;

        // Simple momentum
        let momentum = if closes.len() >= 10 && closes[closes.len() - 10] > 0.0 {
            (closes[closes.len() - 1] - closes[closes.len() - 10]) / closes[closes.len() - 10]
        } else {
            0.0
        };

        // Normalize RSI to [-1, 1]
        let rsi_normalized = (rsi - 50.0) / 50.0;

        vec![rsi_normalized, rsi_change / 100.0, momentum]
    }

    /// Compute RSI
    fn compute_rsi(&self, closes: &[f64]) -> f64 {
        if closes.len() < 2 {
            return 50.0;
        }

        let period = self.rsi_period.min(closes.len() - 1);
        let changes: Vec<f64> = closes
            .windows(2)
            .map(|w| w[1] - w[0])
            .rev()
            .take(period)
            .collect();

        let gains: f64 = changes.iter().filter(|&&c| c > 0.0).sum();
        let losses: f64 = changes.iter().filter(|&&c| c < 0.0).map(|c| c.abs()).sum();

        if losses == 0.0 {
            return 100.0;
        }
        if gains == 0.0 {
            return 0.0;
        }

        let rs = gains / losses;
        100.0 - (100.0 / (1.0 + rs))
    }

    /// Compute volume features
    fn compute_volume_features(&self, bars: &[OHLCVBar]) -> Vec<f64> {
        let volumes: Vec<f64> = bars.iter().map(|b| b.volume).collect();
        let n = volumes.len();

        // Volume ratio (current vs average)
        let avg_volume = if n > 1 {
            volumes[..n - 1].iter().sum::<f64>() / (n - 1) as f64
        } else {
            volumes[n - 1]
        };

        let volume_ratio = if avg_volume > 0.0 {
            volumes[n - 1] / avg_volume
        } else {
            1.0
        };

        // Volume trend
        let recent_avg = if n >= 5 {
            volumes[n - 5..].iter().sum::<f64>() / 5.0
        } else {
            volumes.iter().sum::<f64>() / n as f64
        };

        let older_avg = if n >= 10 {
            volumes[n - 10..n - 5].iter().sum::<f64>() / 5.0
        } else {
            recent_avg
        };

        let volume_trend = if older_avg > 0.0 {
            (recent_avg - older_avg) / older_avg
        } else {
            0.0
        };

        // Normalize
        vec![
            (volume_ratio - 1.0).max(-2.0).min(2.0) / 2.0,
            volume_trend.max(-1.0).min(1.0),
        ]
    }

    /// Compute price position features
    fn compute_price_position(&self, bars: &[OHLCVBar]) -> Vec<f64> {
        let n = bars.len();
        let current_close = bars[n - 1].close;

        // Price position in recent range
        let lookback = 20.min(n);
        let recent = &bars[n - lookback..];
        let high = recent.iter().map(|b| b.high).fold(f64::NEG_INFINITY, f64::max);
        let low = recent.iter().map(|b| b.low).fold(f64::INFINITY, f64::min);

        let price_position = if high > low {
            2.0 * (current_close - low) / (high - low) - 1.0
        } else {
            0.0
        };

        // Range position (current bar)
        let bar = &bars[n - 1];
        let range_position = if bar.high > bar.low {
            2.0 * (bar.close - bar.low) / (bar.high - bar.low) - 1.0
        } else {
            0.0
        };

        vec![price_position, range_position]
    }

    /// Compute candle pattern features
    fn compute_candle_features(&self, bars: &[OHLCVBar]) -> Vec<f64> {
        let bar = &bars[bars.len() - 1];
        let range = bar.range();

        if range == 0.0 {
            return vec![0.0, 0.0, 0.0, 0.0];
        }

        let body = bar.body();
        let body_ratio = body / range;

        let (upper_shadow, lower_shadow) = if bar.is_bullish() {
            ((bar.high - bar.close) / range, (bar.open - bar.low) / range)
        } else {
            ((bar.high - bar.open) / range, (bar.close - bar.low) / range)
        };

        let direction = if bar.is_bullish() { 1.0 } else { -1.0 };

        vec![body_ratio, upper_shadow, lower_shadow, direction]
    }

    /// Get the expected feature dimension
    pub fn feature_dim(&self) -> usize {
        21 // Total: 4 returns + 3 volatility + 3 trend + 3 momentum + 2 volume + 2 position + 4 candle
    }
}

impl Default for FeatureExtractor {
    fn default() -> Self {
        Self {
            lookback: 50,
            ma_short: 10,
            ma_long: 20,
            rsi_period: 14,
            atr_period: 14,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_bars(n: usize) -> Vec<OHLCVBar> {
        (0..n)
            .map(|i| {
                let base = 100.0 + (i as f64) * 0.1;
                OHLCVBar::new(
                    i as i64 * 60000,
                    base,
                    base + 1.0,
                    base - 0.5,
                    base + 0.5,
                    1000.0 + (i as f64) * 10.0,
                )
            })
            .collect()
    }

    #[test]
    fn test_ohlcv_bar() {
        let bar = OHLCVBar::new(0, 100.0, 110.0, 95.0, 105.0, 1000.0);
        assert_eq!(bar.range(), 15.0);
        assert_eq!(bar.body(), 5.0);
        assert!(bar.is_bullish());
        assert!((bar.return_pct() - 0.05).abs() < 1e-10);
    }

    #[test]
    fn test_feature_extraction() {
        let bars = create_test_bars(50);
        let extractor = FeatureExtractor::new();

        let features = extractor.extract(&bars).unwrap();
        assert_eq!(features.dim(), extractor.feature_dim());
    }

    #[test]
    fn test_insufficient_data() {
        let bars = create_test_bars(5);
        let extractor = FeatureExtractor::new();

        let features = extractor.extract(&bars);
        assert!(features.is_none());
    }

    #[test]
    fn test_extract_windows() {
        let bars = create_test_bars(100);
        let extractor = FeatureExtractor::new();

        let windows = extractor.extract_windows(&bars, 50, 10);
        assert!(windows.nrows() > 0);
        assert_eq!(windows.ncols(), extractor.feature_dim());
    }
}
