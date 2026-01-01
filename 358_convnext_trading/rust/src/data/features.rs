//! Feature engineering for trading data

use anyhow::Result;
use ndarray::{Array1, Array2};

use super::Candle;

/// Feature builder for constructing input features from candle data
pub struct FeatureBuilder {
    /// Window size for moving averages
    pub sma_periods: Vec<usize>,
    /// Window size for EMA
    pub ema_periods: Vec<usize>,
    /// RSI period
    pub rsi_period: usize,
    /// ATR period
    pub atr_period: usize,
    /// Bollinger Bands period
    pub bb_period: usize,
    /// Bollinger Bands std dev multiplier
    pub bb_std: f64,
    /// MACD periods (fast, slow, signal)
    pub macd_periods: (usize, usize, usize),
    /// Stochastic periods (k, d)
    pub stoch_periods: (usize, usize),
}

impl Default for FeatureBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl FeatureBuilder {
    /// Create a new feature builder with default parameters
    pub fn new() -> Self {
        Self {
            sma_periods: vec![10, 20, 50],
            ema_periods: vec![9, 21, 50, 200],
            rsi_period: 14,
            atr_period: 14,
            bb_period: 20,
            bb_std: 2.0,
            macd_periods: (12, 26, 9),
            stoch_periods: (14, 3),
        }
    }

    /// Build features from candle data
    ///
    /// Returns array of shape [num_candles, num_features]
    pub fn build(&self, candles: &[Candle]) -> Result<Array2<f64>> {
        let n = candles.len();

        // Extract base data
        let opens: Vec<f64> = candles.iter().map(|c| c.open).collect();
        let highs: Vec<f64> = candles.iter().map(|c| c.high).collect();
        let lows: Vec<f64> = candles.iter().map(|c| c.low).collect();
        let closes: Vec<f64> = candles.iter().map(|c| c.close).collect();
        let volumes: Vec<f64> = candles.iter().map(|c| c.volume).collect();

        // Normalize OHLCV
        let price_mean = closes.iter().sum::<f64>() / n as f64;
        let price_std = (closes.iter().map(|x| (x - price_mean).powi(2)).sum::<f64>() / n as f64).sqrt();

        let norm_opens: Vec<f64> = opens.iter().map(|x| (x - price_mean) / price_std).collect();
        let norm_highs: Vec<f64> = highs.iter().map(|x| (x - price_mean) / price_std).collect();
        let norm_lows: Vec<f64> = lows.iter().map(|x| (x - price_mean) / price_std).collect();
        let norm_closes: Vec<f64> = closes.iter().map(|x| (x - price_mean) / price_std).collect();

        let vol_mean = volumes.iter().sum::<f64>() / n as f64;
        let vol_std = (volumes.iter().map(|x| (x - vol_mean).powi(2)).sum::<f64>() / n as f64).sqrt().max(1.0);
        let norm_volumes: Vec<f64> = volumes.iter().map(|x| (x - vol_mean) / vol_std).collect();

        // Calculate indicators
        let rsi = self.calculate_rsi(&closes, self.rsi_period);
        let atr = self.calculate_atr(&highs, &lows, &closes, self.atr_period);
        let (bb_upper, bb_middle, bb_lower) = self.calculate_bollinger(&closes, self.bb_period, self.bb_std);
        let (macd, macd_signal, macd_hist) = self.calculate_macd(&closes);
        let (stoch_k, stoch_d) = self.calculate_stochastic(&highs, &lows, &closes);

        // EMA features
        let mut ema_features: Vec<Vec<f64>> = Vec::new();
        for period in &self.ema_periods {
            let ema = self.calculate_ema(&closes, *period);
            let norm_ema: Vec<f64> = ema.iter().map(|x| (x - price_mean) / price_std).collect();
            ema_features.push(norm_ema);
        }

        // Price returns
        let returns = self.calculate_returns(&closes);

        // VWAP
        let vwap = self.calculate_vwap(&highs, &lows, &closes, &volumes);
        let norm_vwap: Vec<f64> = vwap.iter().map(|x| (x - price_mean) / price_std).collect();

        // Assemble feature matrix
        // Features: open, high, low, close, volume, rsi, atr, bb_upper, bb_lower, macd, macd_signal,
        //           stoch_k, stoch_d, ema9, ema21, ema50, ema200, returns, vwap, macd_hist
        let num_features = 20;
        let mut features = Array2::zeros((n, num_features));

        for i in 0..n {
            features[[i, 0]] = norm_opens[i];
            features[[i, 1]] = norm_highs[i];
            features[[i, 2]] = norm_lows[i];
            features[[i, 3]] = norm_closes[i];
            features[[i, 4]] = norm_volumes[i];
            features[[i, 5]] = (rsi[i] - 50.0) / 50.0; // Normalize RSI to [-1, 1]
            features[[i, 6]] = atr[i] / price_std;
            features[[i, 7]] = (bb_upper[i] - price_mean) / price_std;
            features[[i, 8]] = (bb_lower[i] - price_mean) / price_std;
            features[[i, 9]] = macd[i] / price_std;
            features[[i, 10]] = macd_signal[i] / price_std;
            features[[i, 11]] = (stoch_k[i] - 50.0) / 50.0;
            features[[i, 12]] = (stoch_d[i] - 50.0) / 50.0;
            features[[i, 13]] = ema_features[0][i];
            features[[i, 14]] = ema_features[1][i];
            features[[i, 15]] = ema_features[2][i];
            features[[i, 16]] = ema_features[3][i];
            features[[i, 17]] = returns[i] * 100.0; // Scale returns
            features[[i, 18]] = norm_vwap[i];
            features[[i, 19]] = macd_hist[i] / price_std;
        }

        Ok(features)
    }

    /// Calculate RSI
    fn calculate_rsi(&self, closes: &[f64], period: usize) -> Vec<f64> {
        let n = closes.len();
        let mut rsi = vec![50.0; n];

        if n < period + 1 {
            return rsi;
        }

        let mut gains = vec![0.0; n];
        let mut losses = vec![0.0; n];

        for i in 1..n {
            let change = closes[i] - closes[i - 1];
            if change > 0.0 {
                gains[i] = change;
            } else {
                losses[i] = -change;
            }
        }

        // Initial average
        let mut avg_gain: f64 = gains[1..=period].iter().sum::<f64>() / period as f64;
        let mut avg_loss: f64 = losses[1..=period].iter().sum::<f64>() / period as f64;

        for i in period..n {
            avg_gain = (avg_gain * (period - 1) as f64 + gains[i]) / period as f64;
            avg_loss = (avg_loss * (period - 1) as f64 + losses[i]) / period as f64;

            if avg_loss == 0.0 {
                rsi[i] = 100.0;
            } else {
                let rs = avg_gain / avg_loss;
                rsi[i] = 100.0 - (100.0 / (1.0 + rs));
            }
        }

        rsi
    }

    /// Calculate ATR
    fn calculate_atr(&self, highs: &[f64], lows: &[f64], closes: &[f64], period: usize) -> Vec<f64> {
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

        // Calculate ATR as EMA of TR
        atr[0] = tr[0];
        let alpha = 2.0 / (period + 1) as f64;

        for i in 1..n {
            atr[i] = alpha * tr[i] + (1.0 - alpha) * atr[i - 1];
        }

        atr
    }

    /// Calculate Bollinger Bands
    fn calculate_bollinger(&self, closes: &[f64], period: usize, std_mult: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = closes.len();
        let mut upper = vec![0.0; n];
        let mut middle = vec![0.0; n];
        let mut lower = vec![0.0; n];

        for i in 0..n {
            let start = if i >= period - 1 { i - period + 1 } else { 0 };
            let window: Vec<f64> = closes[start..=i].to_vec();
            let len = window.len() as f64;

            let mean = window.iter().sum::<f64>() / len;
            let variance = window.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / len;
            let std = variance.sqrt();

            middle[i] = mean;
            upper[i] = mean + std_mult * std;
            lower[i] = mean - std_mult * std;
        }

        (upper, middle, lower)
    }

    /// Calculate MACD
    fn calculate_macd(&self, closes: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let (fast, slow, signal) = self.macd_periods;

        let ema_fast = self.calculate_ema(closes, fast);
        let ema_slow = self.calculate_ema(closes, slow);

        let macd: Vec<f64> = ema_fast
            .iter()
            .zip(ema_slow.iter())
            .map(|(f, s)| f - s)
            .collect();

        let macd_signal = self.calculate_ema(&macd, signal);

        let macd_hist: Vec<f64> = macd
            .iter()
            .zip(macd_signal.iter())
            .map(|(m, s)| m - s)
            .collect();

        (macd, macd_signal, macd_hist)
    }

    /// Calculate Stochastic Oscillator
    fn calculate_stochastic(&self, highs: &[f64], lows: &[f64], closes: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let (k_period, d_period) = self.stoch_periods;
        let n = closes.len();

        let mut stoch_k = vec![50.0; n];

        for i in (k_period - 1)..n {
            let start = i - k_period + 1;
            let highest = highs[start..=i].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let lowest = lows[start..=i].iter().cloned().fold(f64::INFINITY, f64::min);

            if highest != lowest {
                stoch_k[i] = ((closes[i] - lowest) / (highest - lowest)) * 100.0;
            }
        }

        // Calculate %D as SMA of %K
        let stoch_d = self.calculate_sma(&stoch_k, d_period);

        (stoch_k, stoch_d)
    }

    /// Calculate EMA
    fn calculate_ema(&self, data: &[f64], period: usize) -> Vec<f64> {
        let n = data.len();
        let mut ema = vec![0.0; n];

        if n == 0 {
            return ema;
        }

        ema[0] = data[0];
        let alpha = 2.0 / (period + 1) as f64;

        for i in 1..n {
            ema[i] = alpha * data[i] + (1.0 - alpha) * ema[i - 1];
        }

        ema
    }

    /// Calculate SMA
    fn calculate_sma(&self, data: &[f64], period: usize) -> Vec<f64> {
        let n = data.len();
        let mut sma = vec![0.0; n];

        for i in 0..n {
            let start = if i >= period - 1 { i - period + 1 } else { 0 };
            let window = &data[start..=i];
            sma[i] = window.iter().sum::<f64>() / window.len() as f64;
        }

        sma
    }

    /// Calculate returns
    fn calculate_returns(&self, closes: &[f64]) -> Vec<f64> {
        let n = closes.len();
        let mut returns = vec![0.0; n];

        for i in 1..n {
            if closes[i - 1] != 0.0 {
                returns[i] = (closes[i] - closes[i - 1]) / closes[i - 1];
            }
        }

        returns
    }

    /// Calculate VWAP
    fn calculate_vwap(&self, highs: &[f64], lows: &[f64], closes: &[f64], volumes: &[f64]) -> Vec<f64> {
        let n = closes.len();
        let mut vwap = vec![0.0; n];

        let mut cumulative_tp_vol = 0.0;
        let mut cumulative_vol = 0.0;

        for i in 0..n {
            let typical_price = (highs[i] + lows[i] + closes[i]) / 3.0;
            cumulative_tp_vol += typical_price * volumes[i];
            cumulative_vol += volumes[i];

            if cumulative_vol > 0.0 {
                vwap[i] = cumulative_tp_vol / cumulative_vol;
            }
        }

        vwap
    }

    /// Get feature names
    pub fn feature_names(&self) -> Vec<&'static str> {
        vec![
            "open", "high", "low", "close", "volume",
            "rsi", "atr", "bb_upper", "bb_lower",
            "macd", "macd_signal", "stoch_k", "stoch_d",
            "ema9", "ema21", "ema50", "ema200",
            "returns", "vwap", "macd_hist",
        ]
    }

    /// Get number of features
    pub fn num_features(&self) -> usize {
        20
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_candles(n: usize) -> Vec<Candle> {
        (0..n)
            .map(|i| Candle {
                timestamp: i as i64 * 3600000,
                open: 100.0 + (i as f64).sin() * 5.0,
                high: 102.0 + (i as f64).sin() * 5.0,
                low: 98.0 + (i as f64).sin() * 5.0,
                close: 101.0 + (i as f64).cos() * 5.0,
                volume: 1000.0 + (i as f64 * 0.1).sin() * 100.0,
                turnover: 100000.0,
            })
            .collect()
    }

    #[test]
    fn test_feature_builder() {
        let candles = create_test_candles(100);
        let builder = FeatureBuilder::new();
        let features = builder.build(&candles).unwrap();

        assert_eq!(features.dim(), (100, 20));
    }

    #[test]
    fn test_rsi() {
        let closes = vec![44.0, 44.25, 44.5, 43.75, 44.5, 44.25, 44.0, 43.5, 44.0, 44.25,
                         43.75, 44.0, 44.5, 45.0, 45.25, 45.5, 45.0, 45.5, 46.0, 45.5];
        let builder = FeatureBuilder::new();
        let rsi = builder.calculate_rsi(&closes, 14);

        // RSI should be between 0 and 100
        for val in &rsi {
            assert!(*val >= 0.0 && *val <= 100.0);
        }
    }

    #[test]
    fn test_feature_names() {
        let builder = FeatureBuilder::new();
        assert_eq!(builder.feature_names().len(), builder.num_features());
    }
}
