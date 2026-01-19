//! Feature Engineering
//!
//! Technical indicators and feature calculations for trading data.

use crate::api::KlineData;

/// Feature engineering for trading data
pub struct FeatureEngine {
    /// Lookback periods for RSI
    pub rsi_period: usize,
    /// Short period for MACD
    pub macd_short: usize,
    /// Long period for MACD
    pub macd_long: usize,
    /// Signal period for MACD
    pub macd_signal: usize,
    /// Period for Bollinger Bands
    pub bb_period: usize,
    /// Standard deviations for Bollinger Bands
    pub bb_std: f64,
}

impl Default for FeatureEngine {
    fn default() -> Self {
        Self {
            rsi_period: 14,
            macd_short: 12,
            macd_long: 26,
            macd_signal: 9,
            bb_period: 20,
            bb_std: 2.0,
        }
    }
}

impl FeatureEngine {
    /// Calculate all features from kline data
    ///
    /// Returns a 2D vector where each inner vector contains:
    /// [returns, log_returns, rsi, macd, macd_signal, bb_position, volume_ma_ratio]
    pub fn calculate_features(&self, klines: &[KlineData]) -> Vec<Vec<f64>> {
        let n = klines.len();
        if n < self.macd_long + self.macd_signal {
            return vec![];
        }

        let closes: Vec<f64> = klines.iter().map(|k| k.close).collect();
        let volumes: Vec<f64> = klines.iter().map(|k| k.volume).collect();
        let highs: Vec<f64> = klines.iter().map(|k| k.high).collect();
        let lows: Vec<f64> = klines.iter().map(|k| k.low).collect();

        // Calculate indicators
        let returns = self.calculate_returns(&closes);
        let log_returns = self.calculate_log_returns(&closes);
        let rsi = self.calculate_rsi(&closes, self.rsi_period);
        let (macd, macd_sig) = self.calculate_macd(&closes);
        let bb_position = self.calculate_bb_position(&closes);
        let volume_ratio = self.calculate_volume_ma_ratio(&volumes, 20);
        let atr = self.calculate_atr(&highs, &lows, &closes, 14);

        // Combine features (skip initial rows that don't have all indicators)
        let start_idx = self.macd_long + self.macd_signal;
        let mut features = Vec::with_capacity(n - start_idx);

        for i in start_idx..n {
            features.push(vec![
                returns[i - 1],              // Returns (shifted by 1)
                log_returns[i - 1],          // Log returns
                (rsi[i] - 50.0) / 50.0,      // RSI normalized to [-1, 1]
                macd[i] / closes[i] * 100.0, // MACD normalized
                macd_sig[i] / closes[i] * 100.0,
                bb_position[i],              // BB position already [-1, 1]
                (volume_ratio[i] - 1.0).tanh(), // Volume ratio
            ]);
        }

        features
    }

    /// Calculate simple returns
    pub fn calculate_returns(&self, prices: &[f64]) -> Vec<f64> {
        let mut returns = vec![0.0];
        for i in 1..prices.len() {
            if prices[i - 1] != 0.0 {
                returns.push((prices[i] - prices[i - 1]) / prices[i - 1]);
            } else {
                returns.push(0.0);
            }
        }
        returns
    }

    /// Calculate log returns
    pub fn calculate_log_returns(&self, prices: &[f64]) -> Vec<f64> {
        let mut returns = vec![0.0];
        for i in 1..prices.len() {
            if prices[i - 1] > 0.0 && prices[i] > 0.0 {
                returns.push((prices[i] / prices[i - 1]).ln());
            } else {
                returns.push(0.0);
            }
        }
        returns
    }

    /// Calculate RSI (Relative Strength Index)
    pub fn calculate_rsi(&self, prices: &[f64], period: usize) -> Vec<f64> {
        let n = prices.len();
        let mut rsi = vec![50.0; n]; // Default to neutral

        if n < period + 1 {
            return rsi;
        }

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

        // Calculate first average
        let mut avg_gain: f64 = gains[1..=period].iter().sum::<f64>() / period as f64;
        let mut avg_loss: f64 = losses[1..=period].iter().sum::<f64>() / period as f64;

        for i in period..n {
            if i > period {
                avg_gain = (avg_gain * (period - 1) as f64 + gains[i]) / period as f64;
                avg_loss = (avg_loss * (period - 1) as f64 + losses[i]) / period as f64;
            }

            if avg_loss == 0.0 {
                rsi[i] = 100.0;
            } else {
                let rs = avg_gain / avg_loss;
                rsi[i] = 100.0 - (100.0 / (1.0 + rs));
            }
        }

        rsi
    }

    /// Calculate MACD (Moving Average Convergence Divergence)
    pub fn calculate_macd(&self, prices: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = prices.len();
        let mut macd = vec![0.0; n];
        let mut signal = vec![0.0; n];

        let ema_short = self.calculate_ema(prices, self.macd_short);
        let ema_long = self.calculate_ema(prices, self.macd_long);

        for i in 0..n {
            macd[i] = ema_short[i] - ema_long[i];
        }

        let signal_line = self.calculate_ema(&macd, self.macd_signal);
        signal.copy_from_slice(&signal_line);

        (macd, signal)
    }

    /// Calculate EMA (Exponential Moving Average)
    pub fn calculate_ema(&self, values: &[f64], period: usize) -> Vec<f64> {
        let n = values.len();
        let mut ema = vec![0.0; n];

        if n == 0 || period == 0 {
            return ema;
        }

        let multiplier = 2.0 / (period + 1) as f64;

        // First value is SMA
        if n >= period {
            ema[period - 1] = values[0..period].iter().sum::<f64>() / period as f64;

            for i in period..n {
                ema[i] = (values[i] - ema[i - 1]) * multiplier + ema[i - 1];
            }
        }

        ema
    }

    /// Calculate SMA (Simple Moving Average)
    pub fn calculate_sma(&self, values: &[f64], period: usize) -> Vec<f64> {
        let n = values.len();
        let mut sma = vec![0.0; n];

        if n < period {
            return sma;
        }

        let mut sum: f64 = values[0..period].iter().sum();
        sma[period - 1] = sum / period as f64;

        for i in period..n {
            sum = sum - values[i - period] + values[i];
            sma[i] = sum / period as f64;
        }

        sma
    }

    /// Calculate Bollinger Bands position (where price is relative to bands)
    pub fn calculate_bb_position(&self, prices: &[f64]) -> Vec<f64> {
        let n = prices.len();
        let mut position = vec![0.0; n];

        let sma = self.calculate_sma(prices, self.bb_period);

        for i in self.bb_period - 1..n {
            // Calculate standard deviation
            let start = i + 1 - self.bb_period;
            let mean = sma[i];
            let variance: f64 = prices[start..=i]
                .iter()
                .map(|p| (p - mean).powi(2))
                .sum::<f64>()
                / self.bb_period as f64;
            let std_dev = variance.sqrt();

            let upper = mean + self.bb_std * std_dev;
            let lower = mean - self.bb_std * std_dev;

            // Position in range [-1, 1]
            if upper != lower {
                position[i] = (2.0 * (prices[i] - lower) / (upper - lower) - 1.0).clamp(-1.0, 1.0);
            }
        }

        position
    }

    /// Calculate volume moving average ratio
    pub fn calculate_volume_ma_ratio(&self, volumes: &[f64], period: usize) -> Vec<f64> {
        let n = volumes.len();
        let mut ratio = vec![1.0; n];

        let sma = self.calculate_sma(volumes, period);

        for i in period - 1..n {
            if sma[i] > 0.0 {
                ratio[i] = volumes[i] / sma[i];
            }
        }

        ratio
    }

    /// Calculate ATR (Average True Range)
    pub fn calculate_atr(&self, highs: &[f64], lows: &[f64], closes: &[f64], period: usize) -> Vec<f64> {
        let n = highs.len();
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

        // Calculate ATR as EMA of TR
        atr = self.calculate_ema(&tr, period);

        atr
    }

    /// Generate target variable (next period return)
    pub fn generate_targets(&self, prices: &[f64], offset: usize) -> Vec<f64> {
        let n = prices.len();
        let mut targets = vec![0.0; n];

        for i in 0..n.saturating_sub(offset) {
            if prices[i] != 0.0 {
                targets[i] = (prices[i + offset] - prices[i]) / prices[i];
            }
        }

        targets
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_klines(n: usize) -> Vec<KlineData> {
        (0..n)
            .map(|i| {
                let price = 100.0 + (i as f64 * 0.1).sin() * 10.0;
                KlineData {
                    timestamp: i as i64 * 60000,
                    open: price - 0.5,
                    high: price + 1.0,
                    low: price - 1.0,
                    close: price,
                    volume: 1000.0 + (i as f64 * 0.05).cos() * 500.0,
                    turnover: 100000.0,
                }
            })
            .collect()
    }

    #[test]
    fn test_rsi() {
        let engine = FeatureEngine::default();
        let prices: Vec<f64> = (0..50).map(|i| 100.0 + i as f64).collect();
        let rsi = engine.calculate_rsi(&prices, 14);

        // RSI should be high for consistently rising prices
        assert!(rsi.last().unwrap() > &70.0);
    }

    #[test]
    fn test_macd() {
        let engine = FeatureEngine::default();
        let prices: Vec<f64> = (0..50).map(|i| 100.0 + i as f64 * 0.5).collect();
        let (macd, signal) = engine.calculate_macd(&prices);

        // MACD should be positive for rising prices
        assert!(macd.last().unwrap() > &0.0);
        assert_eq!(macd.len(), prices.len());
        assert_eq!(signal.len(), prices.len());
    }

    #[test]
    fn test_calculate_features() {
        let engine = FeatureEngine::default();
        let klines = create_test_klines(100);
        let features = engine.calculate_features(&klines);

        // Should have fewer rows due to lookback requirements
        assert!(features.len() < klines.len());
        assert!(!features.is_empty());

        // Each feature vector should have 7 elements
        for f in &features {
            assert_eq!(f.len(), 7);
        }
    }

    #[test]
    fn test_targets() {
        let engine = FeatureEngine::default();
        let prices = vec![100.0, 101.0, 102.0, 103.0, 104.0];
        let targets = engine.generate_targets(&prices, 1);

        assert!((targets[0] - 0.01).abs() < 0.0001);
        assert!((targets[1] - 0.0099).abs() < 0.001);
    }
}
