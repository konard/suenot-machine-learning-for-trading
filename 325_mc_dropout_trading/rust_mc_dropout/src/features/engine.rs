//! Feature engineering engine

use super::indicators::TechnicalIndicators;
use crate::api::types::Kline;
use ndarray::Array1;

/// Feature engineering engine
pub struct FeatureEngine {
    /// Lookback periods for various features
    pub periods: Vec<usize>,
}

impl FeatureEngine {
    /// Create a new feature engine
    pub fn new() -> Self {
        Self {
            periods: vec![1, 5, 10, 20],
        }
    }

    /// Create features from kline data
    pub fn compute(&self, klines: &[Kline]) -> Option<Array1<f64>> {
        if klines.len() < 50 {
            return None;
        }

        let closes: Vec<f64> = klines.iter().map(|k| k.close).collect();
        let volumes: Vec<f64> = klines.iter().map(|k| k.volume).collect();
        let highs: Vec<f64> = klines.iter().map(|k| k.high).collect();
        let lows: Vec<f64> = klines.iter().map(|k| k.low).collect();

        let mut features = Vec::new();

        // Returns at different lookbacks
        for &period in &self.periods {
            let returns = TechnicalIndicators::returns(&closes, period);
            if let Some(&ret) = returns.last() {
                features.push(if ret.is_nan() { 0.0 } else { ret });
            }
        }

        // Volatility
        let volatility_10 = TechnicalIndicators::rolling_std(
            &TechnicalIndicators::returns(&closes, 1),
            10,
        );
        if let Some(&vol) = volatility_10.last() {
            features.push(if vol.is_nan() { 0.0 } else { vol });
        }

        let volatility_20 = TechnicalIndicators::rolling_std(
            &TechnicalIndicators::returns(&closes, 1),
            20,
        );
        if let Some(&vol) = volatility_20.last() {
            features.push(if vol.is_nan() { 0.0 } else { vol });
        }

        // Volume ratio
        let volume_sma = TechnicalIndicators::sma(&volumes, 20);
        if let (Some(&current_vol), Some(&avg_vol)) = (volumes.last(), volume_sma.last()) {
            let ratio = if avg_vol > 0.0 { current_vol / avg_vol } else { 1.0 };
            features.push(ratio);
        }

        // Price relative to MAs
        for &period in &[10, 20, 50] {
            let ma = TechnicalIndicators::sma(&closes, period);
            if let (Some(&price), Some(&ma_val)) = (closes.last(), ma.last()) {
                if !ma_val.is_nan() && ma_val > 0.0 {
                    features.push(price / ma_val - 1.0);
                } else {
                    features.push(0.0);
                }
            }
        }

        // Range (high - low) / close
        if let (Some(&high), Some(&low), Some(&close)) = (highs.last(), lows.last(), closes.last()) {
            if close > 0.0 {
                features.push((high - low) / close);
            } else {
                features.push(0.0);
            }
        }

        // Price position in range
        if let (Some(&high), Some(&low), Some(&close)) = (highs.last(), lows.last(), closes.last()) {
            let range = high - low;
            if range > 0.0 {
                features.push((close - low) / range);
            } else {
                features.push(0.5);
            }
        }

        // RSI
        let rsi = TechnicalIndicators::rsi(&closes, 14);
        if let Some(&rsi_val) = rsi.last() {
            // Normalize to [-1, 1]
            let normalized = if rsi_val.is_nan() { 0.0 } else { (rsi_val - 50.0) / 50.0 };
            features.push(normalized);
        }

        // MACD
        let (macd_line, signal_line, histogram) = TechnicalIndicators::macd(&closes, 12, 26, 9);
        if let Some(&hist) = histogram.last() {
            features.push(if hist.is_nan() { 0.0 } else { hist / closes.last().unwrap_or(&1.0) * 100.0 });
        }

        // Bollinger Band position
        let (upper, middle, lower) = TechnicalIndicators::bollinger_bands(&closes, 20, 2.0);
        if let (Some(&u), Some(&m), Some(&l), Some(&price)) = (
            upper.last(), middle.last(), lower.last(), closes.last()
        ) {
            let band_width = u - l;
            if band_width > 0.0 && !u.is_nan() && !l.is_nan() {
                features.push((price - l) / band_width);
            } else {
                features.push(0.5);
            }
        }

        // Momentum
        let momentum_10 = if closes.len() > 10 {
            closes.last().unwrap() / closes[closes.len() - 11] - 1.0
        } else {
            0.0
        };
        features.push(momentum_10);

        let momentum_20 = if closes.len() > 20 {
            closes.last().unwrap() / closes[closes.len() - 21] - 1.0
        } else {
            0.0
        };
        features.push(momentum_20);

        // Pad to ensure consistent feature count
        while features.len() < 20 {
            features.push(0.0);
        }

        // Truncate if too many
        features.truncate(20);

        Some(Array1::from_vec(features))
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

    fn create_test_klines(n: usize) -> Vec<Kline> {
        (0..n)
            .map(|i| Kline {
                start_time: i as i64 * 60000,
                open: 100.0 + (i as f64).sin(),
                high: 101.0 + (i as f64).sin(),
                low: 99.0 + (i as f64).sin(),
                close: 100.5 + (i as f64).sin(),
                volume: 1000.0 + (i as f64) * 10.0,
            })
            .collect()
    }

    #[test]
    fn test_feature_computation() {
        let klines = create_test_klines(100);
        let engine = FeatureEngine::new();
        let features = engine.compute(&klines);

        assert!(features.is_some());
        let features = features.unwrap();
        assert_eq!(features.len(), 20);
    }

    #[test]
    fn test_insufficient_data() {
        let klines = create_test_klines(30);
        let engine = FeatureEngine::new();
        let features = engine.compute(&klines);

        assert!(features.is_none());
    }
}
