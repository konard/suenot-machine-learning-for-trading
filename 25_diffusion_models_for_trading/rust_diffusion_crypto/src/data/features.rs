//! Technical indicator computation.

use ndarray::{Array1, Array2, s};

use super::OHLCV;

/// Feature engineering for financial time series.
pub struct FeatureEngineer {
    /// RSI period
    pub rsi_period: usize,
    /// MACD fast period
    pub macd_fast: usize,
    /// MACD slow period
    pub macd_slow: usize,
    /// MACD signal period
    pub macd_signal: usize,
    /// Bollinger Bands period
    pub bb_period: usize,
    /// Bollinger Bands standard deviations
    pub bb_std: f64,
    /// ATR period
    pub atr_period: usize,
}

impl Default for FeatureEngineer {
    fn default() -> Self {
        Self {
            rsi_period: 14,
            macd_fast: 12,
            macd_slow: 26,
            macd_signal: 9,
            bb_period: 20,
            bb_std: 2.0,
            atr_period: 14,
        }
    }
}

impl FeatureEngineer {
    /// Create a new feature engineer with default parameters.
    pub fn new() -> Self {
        Self::default()
    }

    /// Compute RSI (Relative Strength Index).
    pub fn compute_rsi(&self, prices: &[f64]) -> Vec<f64> {
        let n = prices.len();
        let mut rsi = vec![50.0; n]; // Default to neutral

        if n < self.rsi_period + 1 {
            return rsi;
        }

        // Calculate price changes
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

        // Calculate initial average gain/loss
        let mut avg_gain: f64 = gains[1..=self.rsi_period].iter().sum::<f64>() / self.rsi_period as f64;
        let mut avg_loss: f64 = losses[1..=self.rsi_period].iter().sum::<f64>() / self.rsi_period as f64;

        // Calculate RSI
        for i in self.rsi_period..n {
            if i > self.rsi_period {
                avg_gain = (avg_gain * (self.rsi_period - 1) as f64 + gains[i]) / self.rsi_period as f64;
                avg_loss = (avg_loss * (self.rsi_period - 1) as f64 + losses[i]) / self.rsi_period as f64;
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

    /// Compute EMA (Exponential Moving Average).
    pub fn compute_ema(&self, prices: &[f64], period: usize) -> Vec<f64> {
        let n = prices.len();
        let mut ema = vec![0.0; n];

        if n == 0 || period == 0 {
            return ema;
        }

        let multiplier = 2.0 / (period + 1) as f64;

        // Initialize with SMA
        let sma: f64 = prices[..period.min(n)].iter().sum::<f64>() / period as f64;
        ema[period.min(n) - 1] = sma;

        // Calculate EMA
        for i in period..n {
            ema[i] = (prices[i] - ema[i - 1]) * multiplier + ema[i - 1];
        }

        ema
    }

    /// Compute MACD (Moving Average Convergence Divergence).
    pub fn compute_macd(&self, prices: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let ema_fast = self.compute_ema(prices, self.macd_fast);
        let ema_slow = self.compute_ema(prices, self.macd_slow);

        let n = prices.len();
        let mut macd_line = vec![0.0; n];

        for i in 0..n {
            macd_line[i] = ema_fast[i] - ema_slow[i];
        }

        let signal_line = self.compute_ema(&macd_line, self.macd_signal);

        let mut histogram = vec![0.0; n];
        for i in 0..n {
            histogram[i] = macd_line[i] - signal_line[i];
        }

        (macd_line, signal_line, histogram)
    }

    /// Compute Bollinger Bands %B.
    pub fn compute_bollinger_pct_b(&self, prices: &[f64]) -> Vec<f64> {
        let n = prices.len();
        let mut pct_b = vec![0.5; n]; // Default to middle

        for i in self.bb_period..n {
            let window = &prices[i - self.bb_period..i];
            let mean: f64 = window.iter().sum::<f64>() / self.bb_period as f64;
            let variance: f64 = window.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / self.bb_period as f64;
            let std = variance.sqrt();

            let upper = mean + self.bb_std * std;
            let lower = mean - self.bb_std * std;

            if upper != lower {
                pct_b[i] = (prices[i] - lower) / (upper - lower);
            }
        }

        pct_b
    }

    /// Compute rolling volatility.
    pub fn compute_volatility(&self, returns: &[f64], window: usize) -> Vec<f64> {
        let n = returns.len();
        let mut volatility = vec![0.0; n];

        for i in window..n {
            let window_returns = &returns[i - window..i];
            let mean: f64 = window_returns.iter().sum::<f64>() / window as f64;
            let variance: f64 = window_returns.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / window as f64;
            volatility[i] = variance.sqrt();
        }

        volatility
    }

    /// Compute momentum.
    pub fn compute_momentum(&self, prices: &[f64], period: usize) -> Vec<f64> {
        let n = prices.len();
        let mut momentum = vec![0.0; n];

        for i in period..n {
            momentum[i] = prices[i] / prices[i - period] - 1.0;
        }

        momentum
    }

    /// Compute all features from OHLCV data.
    pub fn compute_all(&self, data: &[OHLCV]) -> Array2<f64> {
        let n = data.len();
        let closes: Vec<f64> = data.iter().map(|x| x.close).collect();
        let volumes: Vec<f64> = data.iter().map(|x| x.volume).collect();

        // Compute returns
        let mut returns = vec![0.0; n];
        for i in 1..n {
            returns[i] = (closes[i] - closes[i - 1]) / closes[i - 1];
        }

        // Compute features
        let volatility_24h = self.compute_volatility(&returns, 24);
        let rsi = self.compute_rsi(&closes);
        let (macd, _, _) = self.compute_macd(&closes);
        let bb_pct = self.compute_bollinger_pct_b(&closes);
        let momentum_24h = self.compute_momentum(&closes, 24);

        // Compute volume ratio
        let mut volume_sma = vec![0.0; n];
        for i in 24..n {
            volume_sma[i] = volumes[i - 24..i].iter().sum::<f64>() / 24.0;
        }
        let mut volume_ratio = vec![1.0; n];
        for i in 24..n {
            if volume_sma[i] > 0.0 {
                volume_ratio[i] = volumes[i] / volume_sma[i];
            }
        }

        // Create feature matrix
        // Features: returns, volatility_24h, rsi, macd, bb_pct, volume_ratio, momentum_24h
        let num_features = 7;
        let mut features = Array2::zeros((n, num_features));

        for i in 0..n {
            features[[i, 0]] = returns[i];
            features[[i, 1]] = volatility_24h[i];
            features[[i, 2]] = rsi[i] / 100.0; // Normalize to 0-1
            features[[i, 3]] = macd[i];
            features[[i, 4]] = bb_pct[i];
            features[[i, 5]] = volume_ratio[i].min(5.0) / 5.0; // Normalize
            features[[i, 6]] = momentum_24h[i];
        }

        features
    }

    /// Get feature names.
    pub fn feature_names(&self) -> Vec<&'static str> {
        vec![
            "returns",
            "volatility_24h",
            "rsi_14",
            "macd",
            "bb_pct",
            "volume_ratio",
            "momentum_24h",
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rsi() {
        let engineer = FeatureEngineer::new();
        let prices = vec![44.0, 44.5, 43.5, 44.5, 44.0, 43.5, 44.0, 44.5, 45.0, 45.5,
                         46.0, 46.5, 46.0, 46.5, 47.0, 47.5, 47.0, 47.5, 48.0, 48.5];
        let rsi = engineer.compute_rsi(&prices);

        assert_eq!(rsi.len(), prices.len());
        // RSI should be between 0 and 100
        for val in &rsi {
            assert!(*val >= 0.0 && *val <= 100.0);
        }
    }

    #[test]
    fn test_ema() {
        let engineer = FeatureEngineer::new();
        let prices = vec![10.0, 11.0, 12.0, 11.0, 12.0, 13.0, 12.0, 13.0, 14.0, 15.0];
        let ema = engineer.compute_ema(&prices, 3);

        assert_eq!(ema.len(), prices.len());
    }
}
