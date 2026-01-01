//! Feature Engineering for Trading
//!
//! This module computes technical indicators and features from raw OHLCV data
//! to be used as input for the SE trading model.

use ndarray::{Array1, Array2};
use super::bybit::Kline;

/// Feature names for interpretability
pub const FEATURE_NAMES: &[&str] = &[
    "returns",
    "log_returns",
    "volatility",
    "rsi",
    "macd",
    "macd_signal",
    "macd_hist",
    "atr",
    "bollinger_pct",
    "volume_ma_ratio",
    "obv_normalized",
    "momentum",
];

/// Feature engineering engine
#[derive(Debug, Clone)]
pub struct FeatureEngine {
    /// RSI period
    rsi_period: usize,
    /// MACD fast period
    macd_fast: usize,
    /// MACD slow period
    macd_slow: usize,
    /// MACD signal period
    macd_signal_period: usize,
    /// ATR period
    atr_period: usize,
    /// Bollinger Bands period
    bb_period: usize,
    /// Bollinger Bands standard deviations
    bb_std: f64,
    /// Volume MA period
    volume_ma_period: usize,
    /// Momentum period
    momentum_period: usize,
}

impl Default for FeatureEngine {
    fn default() -> Self {
        Self {
            rsi_period: 14,
            macd_fast: 12,
            macd_slow: 26,
            macd_signal_period: 9,
            atr_period: 14,
            bb_period: 20,
            bb_std: 2.0,
            volume_ma_period: 20,
            momentum_period: 10,
        }
    }
}

impl FeatureEngine {
    /// Create a new feature engine with custom parameters
    pub fn new(
        rsi_period: usize,
        macd_fast: usize,
        macd_slow: usize,
        atr_period: usize,
    ) -> Self {
        Self {
            rsi_period,
            macd_fast,
            macd_slow,
            atr_period,
            ..Default::default()
        }
    }

    /// Get the number of features this engine produces
    pub fn num_features(&self) -> usize {
        FEATURE_NAMES.len()
    }

    /// Get feature names
    pub fn feature_names(&self) -> &[&str] {
        FEATURE_NAMES
    }

    /// Compute all features from kline data
    ///
    /// # Arguments
    ///
    /// * `klines` - Vector of Kline data
    ///
    /// # Returns
    ///
    /// Feature matrix of shape (time_steps, num_features)
    pub fn compute_features(&self, klines: &[Kline]) -> Array2<f64> {
        let n = klines.len();
        let num_features = self.num_features();
        let mut features = Array2::zeros((n, num_features));

        // Extract basic price/volume arrays
        let closes: Vec<f64> = klines.iter().map(|k| k.close).collect();
        let highs: Vec<f64> = klines.iter().map(|k| k.high).collect();
        let lows: Vec<f64> = klines.iter().map(|k| k.low).collect();
        let volumes: Vec<f64> = klines.iter().map(|k| k.volume).collect();

        // Compute returns
        let returns = self.compute_returns(&closes);
        let log_returns = self.compute_log_returns(&closes);

        // Compute volatility (rolling std of returns)
        let volatility = self.rolling_std(&returns, 20);

        // Compute RSI
        let rsi = self.compute_rsi(&closes, self.rsi_period);

        // Compute MACD
        let (macd, macd_signal, macd_hist) = self.compute_macd(
            &closes,
            self.macd_fast,
            self.macd_slow,
            self.macd_signal_period,
        );

        // Compute ATR
        let atr = self.compute_atr(klines, self.atr_period);

        // Compute Bollinger Band percentage
        let bb_pct = self.compute_bollinger_pct(&closes, self.bb_period, self.bb_std);

        // Compute volume MA ratio
        let volume_ma_ratio = self.compute_volume_ma_ratio(&volumes, self.volume_ma_period);

        // Compute OBV (normalized)
        let obv = self.compute_obv_normalized(&closes, &volumes);

        // Compute momentum
        let momentum = self.compute_momentum(&closes, self.momentum_period);

        // Fill feature matrix
        for i in 0..n {
            features[[i, 0]] = returns[i];
            features[[i, 1]] = log_returns[i];
            features[[i, 2]] = volatility[i];
            features[[i, 3]] = rsi[i];
            features[[i, 4]] = macd[i];
            features[[i, 5]] = macd_signal[i];
            features[[i, 6]] = macd_hist[i];
            features[[i, 7]] = atr[i];
            features[[i, 8]] = bb_pct[i];
            features[[i, 9]] = volume_ma_ratio[i];
            features[[i, 10]] = obv[i];
            features[[i, 11]] = momentum[i];
        }

        features
    }

    /// Compute simple returns
    fn compute_returns(&self, prices: &[f64]) -> Vec<f64> {
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

    /// Compute log returns
    fn compute_log_returns(&self, prices: &[f64]) -> Vec<f64> {
        let mut log_returns = vec![0.0];
        for i in 1..prices.len() {
            if prices[i] > 0.0 && prices[i - 1] > 0.0 {
                log_returns.push((prices[i] / prices[i - 1]).ln());
            } else {
                log_returns.push(0.0);
            }
        }
        log_returns
    }

    /// Compute rolling standard deviation
    fn rolling_std(&self, data: &[f64], period: usize) -> Vec<f64> {
        let mut result = vec![0.0; data.len()];

        for i in period..data.len() {
            let window: Vec<f64> = data[(i - period)..i].to_vec();
            let mean = window.iter().sum::<f64>() / period as f64;
            let variance = window.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / period as f64;
            result[i] = variance.sqrt();
        }

        result
    }

    /// Compute RSI (Relative Strength Index)
    fn compute_rsi(&self, prices: &[f64], period: usize) -> Vec<f64> {
        let n = prices.len();
        let mut rsi = vec![50.0; n]; // Default to neutral

        if n < period + 1 {
            return rsi;
        }

        // Compute price changes
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

        // Initial average gain/loss
        let mut avg_gain: f64 = gains[1..=period].iter().sum::<f64>() / period as f64;
        let mut avg_loss: f64 = losses[1..=period].iter().sum::<f64>() / period as f64;

        // Calculate RSI using smoothed averages
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

        // Normalize to [-1, 1] range
        rsi.iter_mut().for_each(|x| *x = (*x - 50.0) / 50.0);

        rsi
    }

    /// Compute EMA (Exponential Moving Average)
    fn compute_ema(&self, data: &[f64], period: usize) -> Vec<f64> {
        let mut ema = vec![0.0; data.len()];
        let multiplier = 2.0 / (period + 1) as f64;

        // Initial SMA for first EMA value
        if data.len() >= period {
            ema[period - 1] = data[..period].iter().sum::<f64>() / period as f64;

            for i in period..data.len() {
                ema[i] = (data[i] - ema[i - 1]) * multiplier + ema[i - 1];
            }
        }

        ema
    }

    /// Compute MACD (Moving Average Convergence Divergence)
    fn compute_macd(
        &self,
        prices: &[f64],
        fast_period: usize,
        slow_period: usize,
        signal_period: usize,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let ema_fast = self.compute_ema(prices, fast_period);
        let ema_slow = self.compute_ema(prices, slow_period);

        let n = prices.len();
        let mut macd = vec![0.0; n];

        for i in 0..n {
            macd[i] = ema_fast[i] - ema_slow[i];
        }

        let signal = self.compute_ema(&macd, signal_period);

        let mut histogram = vec![0.0; n];
        for i in 0..n {
            histogram[i] = macd[i] - signal[i];
        }

        // Normalize by price level
        let avg_price = prices.iter().sum::<f64>() / prices.len() as f64;
        if avg_price > 0.0 {
            macd.iter_mut().for_each(|x| *x /= avg_price * 0.01);
            histogram.iter_mut().for_each(|x| *x /= avg_price * 0.01);
        }

        (macd, signal, histogram)
    }

    /// Compute ATR (Average True Range)
    fn compute_atr(&self, klines: &[Kline], period: usize) -> Vec<f64> {
        let n = klines.len();
        let mut tr = vec![0.0; n];
        let mut atr = vec![0.0; n];

        // Compute True Range
        for i in 0..n {
            let prev_close = if i > 0 { klines[i - 1].close } else { klines[i].open };
            tr[i] = klines[i].true_range(Some(prev_close));
        }

        // Compute ATR as EMA of TR
        if n >= period {
            atr[period - 1] = tr[..period].iter().sum::<f64>() / period as f64;
            let multiplier = 2.0 / (period + 1) as f64;

            for i in period..n {
                atr[i] = (tr[i] - atr[i - 1]) * multiplier + atr[i - 1];
            }
        }

        // Normalize by price level
        for i in 0..n {
            if klines[i].close > 0.0 {
                atr[i] /= klines[i].close;
            }
        }

        atr
    }

    /// Compute Bollinger Band percentage
    fn compute_bollinger_pct(&self, prices: &[f64], period: usize, num_std: f64) -> Vec<f64> {
        let n = prices.len();
        let mut bb_pct = vec![0.0; n];

        for i in period..n {
            let window: Vec<f64> = prices[(i - period)..i].to_vec();
            let sma = window.iter().sum::<f64>() / period as f64;
            let variance = window.iter().map(|x| (x - sma).powi(2)).sum::<f64>() / period as f64;
            let std = variance.sqrt();

            let upper = sma + num_std * std;
            let lower = sma - num_std * std;

            if upper != lower {
                bb_pct[i] = (prices[i] - lower) / (upper - lower) * 2.0 - 1.0;
            }
        }

        bb_pct
    }

    /// Compute volume MA ratio
    fn compute_volume_ma_ratio(&self, volumes: &[f64], period: usize) -> Vec<f64> {
        let n = volumes.len();
        let mut ratio = vec![0.0; n];

        for i in period..n {
            let ma = volumes[(i - period)..i].iter().sum::<f64>() / period as f64;
            if ma > 0.0 {
                ratio[i] = (volumes[i] / ma) - 1.0;
            }
        }

        // Clip to reasonable range
        ratio.iter_mut().for_each(|x| *x = x.clamp(-5.0, 5.0) / 5.0);

        ratio
    }

    /// Compute normalized OBV (On-Balance Volume)
    fn compute_obv_normalized(&self, prices: &[f64], volumes: &[f64]) -> Vec<f64> {
        let n = prices.len();
        let mut obv = vec![0.0; n];

        for i in 1..n {
            let change = prices[i] - prices[i - 1];
            if change > 0.0 {
                obv[i] = obv[i - 1] + volumes[i];
            } else if change < 0.0 {
                obv[i] = obv[i - 1] - volumes[i];
            } else {
                obv[i] = obv[i - 1];
            }
        }

        // Normalize using z-score over rolling window
        let lookback = 50;
        let mut normalized = vec![0.0; n];

        for i in lookback..n {
            let window: Vec<f64> = obv[(i - lookback)..i].to_vec();
            let mean = window.iter().sum::<f64>() / lookback as f64;
            let variance = window.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / lookback as f64;
            let std = variance.sqrt();

            if std > 0.0 {
                normalized[i] = ((obv[i] - mean) / std).clamp(-3.0, 3.0) / 3.0;
            }
        }

        normalized
    }

    /// Compute momentum
    fn compute_momentum(&self, prices: &[f64], period: usize) -> Vec<f64> {
        let n = prices.len();
        let mut momentum = vec![0.0; n];

        for i in period..n {
            if prices[i - period] != 0.0 {
                momentum[i] = (prices[i] / prices[i - period]) - 1.0;
            }
        }

        // Clip to reasonable range
        momentum.iter_mut().for_each(|x| *x = x.clamp(-0.5, 0.5) * 2.0);

        momentum
    }
}

/// Compute features for a sliding window
pub fn compute_windowed_features(
    klines: &[Kline],
    window_size: usize,
    engine: &FeatureEngine,
) -> Vec<Array2<f64>> {
    let n = klines.len();
    let mut windows = Vec::new();

    if n < window_size {
        return windows;
    }

    for i in 0..=(n - window_size) {
        let window_klines = &klines[i..(i + window_size)];
        let features = engine.compute_features(window_klines);
        windows.push(features);
    }

    windows
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::bybit::generate_sample_data;

    #[test]
    fn test_feature_computation() {
        let engine = FeatureEngine::default();
        let klines = generate_sample_data(100, 50000.0);
        let features = engine.compute_features(&klines);

        assert_eq!(features.nrows(), 100);
        assert_eq!(features.ncols(), engine.num_features());
    }

    #[test]
    fn test_rsi_range() {
        let engine = FeatureEngine::default();
        let klines = generate_sample_data(50, 100.0);
        let features = engine.compute_features(&klines);

        // RSI is normalized to [-1, 1]
        for i in 0..features.nrows() {
            assert!(features[[i, 3]] >= -1.0 && features[[i, 3]] <= 1.0);
        }
    }

    #[test]
    fn test_windowed_features() {
        let engine = FeatureEngine::default();
        let klines = generate_sample_data(100, 50000.0);
        let windows = compute_windowed_features(&klines, 50, &engine);

        assert_eq!(windows.len(), 51); // 100 - 50 + 1
        assert_eq!(windows[0].nrows(), 50);
    }
}
