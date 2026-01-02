//! Feature engineering for trading data

use crate::api::Candle;
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

/// Feature generator for OHLCV data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Features {
    /// Window size for moving averages
    pub ma_window: usize,
    /// RSI period
    pub rsi_period: usize,
    /// ATR period
    pub atr_period: usize,
}

impl Default for Features {
    fn default() -> Self {
        Self {
            ma_window: 20,
            rsi_period: 14,
            atr_period: 14,
        }
    }
}

impl Features {
    /// Create a new Features generator
    pub fn new(ma_window: usize, rsi_period: usize, atr_period: usize) -> Self {
        Self {
            ma_window,
            rsi_period,
            atr_period,
        }
    }

    /// Generate all features from candles
    ///
    /// Returns a 2D array with shape [num_features, num_candles]
    pub fn generate(&self, candles: &[Candle]) -> Array2<f32> {
        let n = candles.len();
        let num_features = 15;

        let mut features = Array2::zeros((num_features, n));

        // Extract raw OHLCV
        let opens: Vec<f32> = candles.iter().map(|c| c.open as f32).collect();
        let highs: Vec<f32> = candles.iter().map(|c| c.high as f32).collect();
        let lows: Vec<f32> = candles.iter().map(|c| c.low as f32).collect();
        let closes: Vec<f32> = candles.iter().map(|c| c.close as f32).collect();
        let volumes: Vec<f32> = candles.iter().map(|c| c.volume as f32).collect();

        // Feature 0-4: Normalized OHLCV
        let close_mean = closes.iter().sum::<f32>() / n as f32;
        for i in 0..n {
            features[[0, i]] = opens[i] / close_mean;
            features[[1, i]] = highs[i] / close_mean;
            features[[2, i]] = lows[i] / close_mean;
            features[[3, i]] = closes[i] / close_mean;
        }

        // Normalize volume
        let vol_mean = volumes.iter().sum::<f32>() / n as f32;
        let vol_std = (volumes.iter().map(|v| (v - vol_mean).powi(2)).sum::<f32>() / n as f32).sqrt();
        for i in 0..n {
            features[[4, i]] = if vol_std > 0.0 {
                (volumes[i] - vol_mean) / vol_std
            } else {
                0.0
            };
        }

        // Feature 5: Returns
        let returns = self.calculate_returns(&closes);
        for i in 0..n {
            features[[5, i]] = returns[i];
        }

        // Feature 6: Log returns
        let log_returns = self.calculate_log_returns(&closes);
        for i in 0..n {
            features[[6, i]] = log_returns[i];
        }

        // Feature 7: High-Low range (normalized)
        for i in 0..n {
            features[[7, i]] = (highs[i] - lows[i]) / closes[i];
        }

        // Feature 8: Body ratio
        for i in 0..n {
            let range = highs[i] - lows[i];
            features[[8, i]] = if range > 0.0 {
                (closes[i] - opens[i]).abs() / range
            } else {
                0.0
            };
        }

        // Feature 9: RSI
        let rsi = self.calculate_rsi(&closes, self.rsi_period);
        for i in 0..n {
            features[[9, i]] = rsi[i] / 100.0; // Normalize to 0-1
        }

        // Feature 10: SMA ratio
        let sma = self.calculate_sma(&closes, self.ma_window);
        for i in 0..n {
            features[[10, i]] = if sma[i] > 0.0 {
                closes[i] / sma[i] - 1.0
            } else {
                0.0
            };
        }

        // Feature 11: EMA ratio
        let ema = self.calculate_ema(&closes, self.ma_window);
        for i in 0..n {
            features[[11, i]] = if ema[i] > 0.0 {
                closes[i] / ema[i] - 1.0
            } else {
                0.0
            };
        }

        // Feature 12: ATR (normalized)
        let atr = self.calculate_atr(&highs, &lows, &closes, self.atr_period);
        for i in 0..n {
            features[[12, i]] = atr[i] / closes[i];
        }

        // Feature 13: Volume change
        let vol_sma = self.calculate_sma(&volumes, self.ma_window);
        for i in 0..n {
            features[[13, i]] = if vol_sma[i] > 0.0 {
                volumes[i] / vol_sma[i] - 1.0
            } else {
                0.0
            };
        }

        // Feature 14: Momentum (rate of change)
        let momentum = self.calculate_momentum(&closes, 10);
        for i in 0..n {
            features[[14, i]] = momentum[i];
        }

        features
    }

    /// Calculate simple returns
    fn calculate_returns(&self, prices: &[f32]) -> Vec<f32> {
        let n = prices.len();
        let mut returns = vec![0.0; n];

        for i in 1..n {
            if prices[i - 1] > 0.0 {
                returns[i] = (prices[i] - prices[i - 1]) / prices[i - 1];
            }
        }

        returns
    }

    /// Calculate log returns
    fn calculate_log_returns(&self, prices: &[f32]) -> Vec<f32> {
        let n = prices.len();
        let mut log_returns = vec![0.0; n];

        for i in 1..n {
            if prices[i - 1] > 0.0 && prices[i] > 0.0 {
                log_returns[i] = (prices[i] / prices[i - 1]).ln();
            }
        }

        log_returns
    }

    /// Calculate RSI (Relative Strength Index)
    fn calculate_rsi(&self, prices: &[f32], period: usize) -> Vec<f32> {
        let n = prices.len();
        let mut rsi = vec![50.0; n]; // Default to neutral

        if n < period + 1 {
            return rsi;
        }

        // Calculate gains and losses
        let mut gains = vec![0.0; n];
        let mut losses = vec![0.0; n];

        for i in 1..n {
            let change = prices[i] - prices[i - 1];
            if change > 0.0 {
                gains[i] = change;
            } else {
                losses[i] = -change;
            }
        }

        // Calculate average gains and losses
        let mut avg_gain = gains[1..=period].iter().sum::<f32>() / period as f32;
        let mut avg_loss = losses[1..=period].iter().sum::<f32>() / period as f32;

        for i in period..n {
            if i > period {
                avg_gain = (avg_gain * (period - 1) as f32 + gains[i]) / period as f32;
                avg_loss = (avg_loss * (period - 1) as f32 + losses[i]) / period as f32;
            }

            if avg_loss > 0.0 {
                let rs = avg_gain / avg_loss;
                rsi[i] = 100.0 - (100.0 / (1.0 + rs));
            } else if avg_gain > 0.0 {
                rsi[i] = 100.0;
            } else {
                rsi[i] = 50.0;
            }
        }

        rsi
    }

    /// Calculate Simple Moving Average
    fn calculate_sma(&self, values: &[f32], window: usize) -> Vec<f32> {
        let n = values.len();
        let mut sma = vec![0.0; n];

        if n < window {
            return sma;
        }

        let mut sum: f32 = values[..window].iter().sum();
        sma[window - 1] = sum / window as f32;

        for i in window..n {
            sum = sum - values[i - window] + values[i];
            sma[i] = sum / window as f32;
        }

        // Fill initial values
        for i in 0..window - 1 {
            sma[i] = sma[window - 1];
        }

        sma
    }

    /// Calculate Exponential Moving Average
    fn calculate_ema(&self, values: &[f32], window: usize) -> Vec<f32> {
        let n = values.len();
        let mut ema = vec![0.0; n];

        if n == 0 {
            return ema;
        }

        let alpha = 2.0 / (window + 1) as f32;
        ema[0] = values[0];

        for i in 1..n {
            ema[i] = alpha * values[i] + (1.0 - alpha) * ema[i - 1];
        }

        ema
    }

    /// Calculate Average True Range
    fn calculate_atr(&self, highs: &[f32], lows: &[f32], closes: &[f32], period: usize) -> Vec<f32> {
        let n = closes.len();
        let mut atr = vec![0.0; n];

        if n < 2 {
            return atr;
        }

        // Calculate True Range
        let mut tr = vec![0.0; n];
        tr[0] = highs[0] - lows[0];

        for i in 1..n {
            let hl = highs[i] - lows[i];
            let hc = (highs[i] - closes[i - 1]).abs();
            let lc = (lows[i] - closes[i - 1]).abs();
            tr[i] = hl.max(hc).max(lc);
        }

        // Calculate ATR using EMA of TR
        let alpha = 2.0 / (period + 1) as f32;
        atr[0] = tr[0];

        for i in 1..n {
            atr[i] = alpha * tr[i] + (1.0 - alpha) * atr[i - 1];
        }

        atr
    }

    /// Calculate momentum (rate of change)
    fn calculate_momentum(&self, prices: &[f32], period: usize) -> Vec<f32> {
        let n = prices.len();
        let mut momentum = vec![0.0; n];

        for i in period..n {
            if prices[i - period] > 0.0 {
                momentum[i] = (prices[i] - prices[i - period]) / prices[i - period];
            }
        }

        momentum
    }

    /// Generate labels for classification
    ///
    /// - 0: Down (return < -threshold)
    /// - 1: Neutral
    /// - 2: Up (return > threshold)
    pub fn generate_labels(
        &self,
        closes: &[f32],
        forward_window: usize,
        threshold: f32,
    ) -> Vec<u8> {
        let n = closes.len();
        let mut labels = vec![1u8; n]; // Default to neutral

        for i in 0..n.saturating_sub(forward_window) {
            let future_return = (closes[i + forward_window] - closes[i]) / closes[i];

            labels[i] = if future_return > threshold {
                2 // Up
            } else if future_return < -threshold {
                0 // Down
            } else {
                1 // Neutral
            };
        }

        labels
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_candles(n: usize) -> Vec<Candle> {
        (0..n)
            .map(|i| {
                let base = 50000.0 + (i as f64) * 10.0;
                Candle::new(
                    (i * 60000) as i64,
                    base,
                    base + 50.0,
                    base - 30.0,
                    base + 20.0,
                    1000.0 + (i as f64) * 10.0,
                    50000000.0,
                )
            })
            .collect()
    }

    #[test]
    fn test_feature_generation() {
        let features = Features::default();
        let candles = create_test_candles(100);
        let result = features.generate(&candles);

        assert_eq!(result.shape(), &[15, 100]);
    }

    #[test]
    fn test_label_generation() {
        let features = Features::default();
        let closes: Vec<f32> = (0..100).map(|i| 50000.0 + (i as f32) * 10.0).collect();
        let labels = features.generate_labels(&closes, 10, 0.001);

        assert_eq!(labels.len(), 100);
        // Since price is always increasing, most labels should be Up (2)
        assert!(labels[..90].iter().filter(|&&l| l == 2).count() > 80);
    }

    #[test]
    fn test_rsi_calculation() {
        let features = Features::default();
        let prices: Vec<f32> = (0..50).map(|i| 100.0 + (i as f32) * 0.5).collect();
        let rsi = features.calculate_rsi(&prices, 14);

        // Uptrending prices should have RSI > 50
        assert!(rsi[30] > 50.0);
    }

    #[test]
    fn test_sma_calculation() {
        let features = Features::default();
        let values: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let sma = features.calculate_sma(&values, 3);

        // SMA of [3,4,5] = 4.0
        assert!((sma[4] - 4.0).abs() < 0.001);
    }
}
