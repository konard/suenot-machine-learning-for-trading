//! Feature engineering for cryptocurrency data.

use anyhow::Result;
use ndarray::{Array1, Array2};

use super::OHLCV;

/// Feature engineer for computing technical indicators and node features.
pub struct FeatureEngineer {
    /// Window size for rolling calculations
    window: usize,
}

impl FeatureEngineer {
    /// Create a new feature engineer with specified window size.
    pub fn new(window: usize) -> Self {
        Self { window }
    }

    /// Compute all features for a single symbol.
    pub fn compute_features(&self, data: &[OHLCV]) -> Result<Vec<f64>> {
        if data.len() < self.window {
            anyhow::bail!("Insufficient data for feature computation");
        }

        let closes: Vec<f64> = data.iter().map(|o| o.close).collect();
        let highs: Vec<f64> = data.iter().map(|o| o.high).collect();
        let lows: Vec<f64> = data.iter().map(|o| o.low).collect();
        let volumes: Vec<f64> = data.iter().map(|o| o.volume).collect();

        let mut features = Vec::new();

        // Momentum features
        features.push(self.momentum(&closes, 1));
        features.push(self.momentum(&closes, 7));
        features.push(self.momentum(&closes, 14));
        features.push(self.momentum(&closes, 30));

        // Volatility
        features.push(self.volatility(&closes, self.window));

        // RSI
        features.push(self.rsi(&closes, 14));

        // MACD
        let (macd, signal) = self.macd(&closes);
        features.push(macd);
        features.push(signal);

        // Volume ratio
        features.push(self.volume_ratio(&volumes, self.window));

        // Price position
        features.push(self.price_position(&closes, self.window));

        // ATR (Average True Range)
        features.push(self.atr(&highs, &lows, &closes, 14));

        Ok(features)
    }

    /// Compute features for multiple symbols as a 2D array.
    pub fn compute_features_matrix(&self, data: &[Vec<OHLCV>]) -> Result<Array2<f64>> {
        let num_symbols = data.len();
        let features: Vec<Vec<f64>> = data
            .iter()
            .map(|ohlcv| self.compute_features(ohlcv))
            .collect::<Result<Vec<_>>>()?;

        if features.is_empty() {
            anyhow::bail!("No features computed");
        }

        let num_features = features[0].len();
        let mut matrix = Array2::zeros((num_symbols, num_features));

        for (i, feat) in features.iter().enumerate() {
            for (j, &val) in feat.iter().enumerate() {
                matrix[[i, j]] = val;
            }
        }

        Ok(matrix)
    }

    /// Calculate momentum (return over n periods).
    fn momentum(&self, prices: &[f64], periods: usize) -> f64 {
        if prices.len() <= periods {
            return 0.0;
        }
        let current = prices[prices.len() - 1];
        let past = prices[prices.len() - 1 - periods];
        if past == 0.0 {
            0.0
        } else {
            (current - past) / past
        }
    }

    /// Calculate realized volatility.
    fn volatility(&self, prices: &[f64], window: usize) -> f64 {
        if prices.len() < window + 1 {
            return 0.0;
        }

        let returns: Vec<f64> = prices
            .windows(2)
            .map(|w| {
                if w[0] == 0.0 {
                    0.0
                } else {
                    (w[1] - w[0]) / w[0]
                }
            })
            .collect();

        let recent_returns = &returns[returns.len().saturating_sub(window)..];
        self.std_dev(recent_returns)
    }

    /// Calculate RSI (Relative Strength Index).
    fn rsi(&self, prices: &[f64], period: usize) -> f64 {
        if prices.len() < period + 1 {
            return 50.0; // Neutral
        }

        let changes: Vec<f64> = prices.windows(2).map(|w| w[1] - w[0]).collect();
        let recent_changes = &changes[changes.len().saturating_sub(period)..];

        let gains: f64 = recent_changes.iter().filter(|&&x| x > 0.0).sum();
        let losses: f64 = recent_changes.iter().filter(|&&x| x < 0.0).map(|x| -x).sum();

        let avg_gain = gains / period as f64;
        let avg_loss = losses / period as f64;

        if avg_loss == 0.0 {
            100.0
        } else {
            let rs = avg_gain / avg_loss;
            100.0 - (100.0 / (1.0 + rs))
        }
    }

    /// Calculate MACD (Moving Average Convergence Divergence).
    fn macd(&self, prices: &[f64]) -> (f64, f64) {
        let ema12 = self.ema(prices, 12);
        let ema26 = self.ema(prices, 26);
        let macd_line = ema12 - ema26;

        // Signal line (9-period EMA of MACD)
        // Simplified: just use the current MACD as approximation
        let signal = macd_line * 0.9; // Approximation

        (macd_line, signal)
    }

    /// Calculate EMA (Exponential Moving Average).
    fn ema(&self, prices: &[f64], period: usize) -> f64 {
        if prices.is_empty() {
            return 0.0;
        }
        if prices.len() < period {
            return prices.iter().sum::<f64>() / prices.len() as f64;
        }

        let multiplier = 2.0 / (period as f64 + 1.0);
        let mut ema = prices[0..period].iter().sum::<f64>() / period as f64;

        for &price in &prices[period..] {
            ema = (price - ema) * multiplier + ema;
        }

        ema
    }

    /// Calculate volume ratio (current vs average).
    fn volume_ratio(&self, volumes: &[f64], window: usize) -> f64 {
        if volumes.len() < window {
            return 1.0;
        }

        let current = volumes[volumes.len() - 1];
        let avg: f64 =
            volumes[volumes.len() - window..].iter().sum::<f64>() / window as f64;

        if avg == 0.0 {
            1.0
        } else {
            current / avg
        }
    }

    /// Calculate price position (relative to window high/low).
    fn price_position(&self, prices: &[f64], window: usize) -> f64 {
        if prices.len() < window {
            return 0.5;
        }

        let window_prices = &prices[prices.len() - window..];
        let high = window_prices.iter().cloned().fold(f64::MIN, f64::max);
        let low = window_prices.iter().cloned().fold(f64::MAX, f64::min);
        let current = prices[prices.len() - 1];

        if high == low {
            0.5
        } else {
            (current - low) / (high - low)
        }
    }

    /// Calculate ATR (Average True Range).
    fn atr(&self, highs: &[f64], lows: &[f64], closes: &[f64], period: usize) -> f64 {
        if closes.len() < period + 1 {
            return 0.0;
        }

        let mut true_ranges = Vec::new();
        for i in 1..closes.len() {
            let hl = highs[i] - lows[i];
            let hc = (highs[i] - closes[i - 1]).abs();
            let lc = (lows[i] - closes[i - 1]).abs();
            true_ranges.push(hl.max(hc).max(lc));
        }

        let recent_tr = &true_ranges[true_ranges.len().saturating_sub(period)..];
        recent_tr.iter().sum::<f64>() / recent_tr.len() as f64
    }

    /// Calculate standard deviation.
    fn std_dev(&self, values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance =
            values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
        variance.sqrt()
    }

    /// Normalize features using z-score.
    pub fn normalize_features(&self, features: &Array2<f64>) -> Array2<f64> {
        let mut normalized = features.clone();

        for j in 0..features.ncols() {
            let col = features.column(j);
            let mean = col.mean().unwrap_or(0.0);
            let std = self.std_dev(col.as_slice().unwrap());

            if std > 0.0 {
                for i in 0..features.nrows() {
                    normalized[[i, j]] = (features[[i, j]] - mean) / std;
                }
            }
        }

        normalized
    }
}

impl Default for FeatureEngineer {
    fn default() -> Self {
        Self::new(20)
    }
}

/// Compute returns from price series.
pub fn compute_returns(prices: &[f64]) -> Vec<f64> {
    prices
        .windows(2)
        .map(|w| {
            if w[0] == 0.0 {
                0.0
            } else {
                (w[1] - w[0]) / w[0]
            }
        })
        .collect()
}

/// Compute correlation matrix from returns.
pub fn compute_correlation_matrix(returns: &[Vec<f64>]) -> Array2<f64> {
    let n = returns.len();
    let mut corr_matrix = Array2::eye(n);

    for i in 0..n {
        for j in (i + 1)..n {
            let corr = pearson_correlation(&returns[i], &returns[j]);
            corr_matrix[[i, j]] = corr;
            corr_matrix[[j, i]] = corr;
        }
    }

    corr_matrix
}

/// Compute Pearson correlation coefficient.
pub fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.is_empty() {
        return 0.0;
    }

    let n = x.len() as f64;
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for (xi, yi) in x.iter().zip(y.iter()) {
        let dx = xi - mean_x;
        let dy = yi - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    let denom = (var_x * var_y).sqrt();
    if denom == 0.0 {
        0.0
    } else {
        cov / denom
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_momentum() {
        let engineer = FeatureEngineer::new(20);
        let prices = vec![100.0, 102.0, 105.0, 103.0, 108.0];
        let mom = engineer.momentum(&prices, 1);
        assert!((mom - (108.0 - 103.0) / 103.0).abs() < 0.001);
    }

    #[test]
    fn test_rsi() {
        let engineer = FeatureEngineer::new(20);
        // Steadily increasing prices should give high RSI
        let prices: Vec<f64> = (0..20).map(|i| 100.0 + i as f64).collect();
        let rsi = engineer.rsi(&prices, 14);
        assert!(rsi > 70.0);
    }

    #[test]
    fn test_pearson_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let corr = pearson_correlation(&x, &y);
        assert!((corr - 1.0).abs() < 0.001);

        let z = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let corr_neg = pearson_correlation(&x, &z);
        assert!((corr_neg + 1.0).abs() < 0.001);
    }
}
