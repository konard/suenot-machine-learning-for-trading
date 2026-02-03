//! Feature engineering for TimeParticle models.

use crate::data::loader::OhlcvData;
use ndarray::{Array1, Array2};

/// Feature engineering engine for creating multiscale features.
pub struct FeatureEngine {
    return_periods: Vec<usize>,
    sma_windows: Vec<usize>,
    volatility_windows: Vec<usize>,
}

impl FeatureEngine {
    /// Create a new feature engine with default parameters.
    pub fn new() -> Self {
        Self {
            return_periods: vec![1, 5, 10, 20, 60],
            sma_windows: vec![5, 10, 20, 50, 100],
            volatility_windows: vec![5, 10, 20, 50],
        }
    }

    /// Create a new feature engine with custom parameters.
    pub fn with_params(
        return_periods: Vec<usize>,
        sma_windows: Vec<usize>,
        volatility_windows: Vec<usize>,
    ) -> Self {
        Self {
            return_periods,
            sma_windows,
            volatility_windows,
        }
    }

    /// Compute all features from OHLCV data.
    pub fn compute_features(&self, data: &OhlcvData) -> Array2<f64> {
        let n = data.len();
        if n == 0 {
            return Array2::zeros((0, 0));
        }

        let mut features: Vec<Vec<f64>> = Vec::new();

        // Returns at different scales
        for &period in &self.return_periods {
            features.push(self.compute_returns(&data.close, period));
        }

        // SMA ratios at different scales
        for &window in &self.sma_windows {
            let sma = self.compute_sma(&data.close, window);
            let ratio: Vec<f64> = data
                .close
                .iter()
                .zip(sma.iter())
                .map(|(c, s)| if *s != 0.0 { c / s } else { 1.0 })
                .collect();
            features.push(ratio);
        }

        // Volatility at different scales
        for &window in &self.volatility_windows {
            features.push(self.compute_volatility(&data.close, window));
        }

        // Volume ratios
        for &window in &[5, 10, 20] {
            let vol_sma = self.compute_sma(&data.volume, window);
            let ratio: Vec<f64> = data
                .volume
                .iter()
                .zip(vol_sma.iter())
                .map(|(v, s)| if *s != 0.0 { v / s } else { 1.0 })
                .collect();
            features.push(ratio);
        }

        // RSI
        features.push(self.compute_rsi(&data.close, 14));

        // MACD
        let (macd, signal, hist) = self.compute_macd(&data.close);
        features.push(macd);
        features.push(signal);
        features.push(hist);

        // Convert to matrix
        let n_features = features.len();
        let mut matrix = Array2::zeros((n, n_features));

        for (j, feature) in features.iter().enumerate() {
            for (i, &value) in feature.iter().enumerate() {
                matrix[[i, j]] = if value.is_finite() { value } else { 0.0 };
            }
        }

        matrix
    }

    /// Compute returns for a given period.
    fn compute_returns(&self, prices: &[f64], period: usize) -> Vec<f64> {
        let n = prices.len();
        let mut returns = vec![0.0; n];

        for i in period..n {
            if prices[i - period] != 0.0 {
                returns[i] = prices[i] / prices[i - period] - 1.0;
            }
        }

        returns
    }

    /// Compute simple moving average.
    fn compute_sma(&self, values: &[f64], window: usize) -> Vec<f64> {
        let n = values.len();
        let mut sma = vec![0.0; n];

        for i in 0..n {
            if i >= window - 1 {
                let sum: f64 = values[i + 1 - window..=i].iter().sum();
                sma[i] = sum / window as f64;
            } else {
                let sum: f64 = values[..=i].iter().sum();
                sma[i] = sum / (i + 1) as f64;
            }
        }

        sma
    }

    /// Compute exponential moving average.
    fn compute_ema(&self, values: &[f64], span: usize) -> Vec<f64> {
        let n = values.len();
        if n == 0 {
            return Vec::new();
        }

        let alpha = 2.0 / (span as f64 + 1.0);
        let mut ema = vec![values[0]; n];

        for i in 1..n {
            ema[i] = alpha * values[i] + (1.0 - alpha) * ema[i - 1];
        }

        ema
    }

    /// Compute volatility (rolling standard deviation of returns).
    fn compute_volatility(&self, prices: &[f64], window: usize) -> Vec<f64> {
        let returns = self.compute_returns(prices, 1);
        let n = returns.len();
        let mut vol = vec![0.0; n];

        for i in window..n {
            let slice = &returns[i + 1 - window..=i];
            let mean: f64 = slice.iter().sum::<f64>() / window as f64;
            let variance: f64 =
                slice.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / window as f64;
            vol[i] = variance.sqrt();
        }

        vol
    }

    /// Compute RSI (Relative Strength Index).
    fn compute_rsi(&self, prices: &[f64], period: usize) -> Vec<f64> {
        let n = prices.len();
        let mut rsi = vec![50.0; n]; // Default to neutral

        if n <= period {
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

        // Smoothed averages
        let mut avg_gain = gains[1..=period].iter().sum::<f64>() / period as f64;
        let mut avg_loss = losses[1..=period].iter().sum::<f64>() / period as f64;

        for i in period..n {
            if i > period {
                avg_gain = (avg_gain * (period - 1) as f64 + gains[i]) / period as f64;
                avg_loss = (avg_loss * (period - 1) as f64 + losses[i]) / period as f64;
            }

            if avg_loss == 0.0 {
                rsi[i] = 100.0;
            } else {
                let rs = avg_gain / avg_loss;
                rsi[i] = 100.0 - 100.0 / (1.0 + rs);
            }
        }

        rsi
    }

    /// Compute MACD (Moving Average Convergence Divergence).
    fn compute_macd(&self, prices: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let ema12 = self.compute_ema(prices, 12);
        let ema26 = self.compute_ema(prices, 26);

        let macd: Vec<f64> = ema12
            .iter()
            .zip(ema26.iter())
            .map(|(a, b)| a - b)
            .collect();

        let signal = self.compute_ema(&macd, 9);

        let hist: Vec<f64> = macd
            .iter()
            .zip(signal.iter())
            .map(|(m, s)| m - s)
            .collect();

        (macd, signal, hist)
    }

    /// Get number of features produced.
    pub fn num_features(&self) -> usize {
        self.return_periods.len()
            + self.sma_windows.len()
            + self.volatility_windows.len()
            + 3  // Volume ratios
            + 1  // RSI
            + 3  // MACD, signal, histogram
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
    use crate::data::loader::generate_simulated_data;

    #[test]
    fn test_feature_engine() {
        let engine = FeatureEngine::new();
        let data = generate_simulated_data(200, 42);

        let features = engine.compute_features(&data);
        assert_eq!(features.nrows(), 200);
        assert_eq!(features.ncols(), engine.num_features());
    }

    #[test]
    fn test_returns() {
        let engine = FeatureEngine::new();
        let prices = vec![100.0, 105.0, 102.0, 108.0, 110.0];
        let returns = engine.compute_returns(&prices, 1);

        assert_eq!(returns[0], 0.0);
        assert!((returns[1] - 0.05).abs() < 1e-10);
    }

    #[test]
    fn test_sma() {
        let engine = FeatureEngine::new();
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sma = engine.compute_sma(&values, 3);

        assert_eq!(sma[2], 2.0);
        assert_eq!(sma[3], 3.0);
        assert_eq!(sma[4], 4.0);
    }

    #[test]
    fn test_rsi() {
        let engine = FeatureEngine::new();
        let mut prices = vec![100.0];
        for i in 1..50 {
            prices.push(prices[i - 1] + 1.0); // Uptrend
        }
        let rsi = engine.compute_rsi(&prices, 14);

        // In a consistent uptrend, RSI should be high
        assert!(rsi[49] > 70.0);
    }
}
