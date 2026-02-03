//! Feature engineering for trading models.

use super::{loader::Dataset, OhlcvBar};

/// Feature engineer for computing trading indicators.
pub struct FeatureEngineer {
    /// Whether to include volume-based features
    include_volume: bool,
}

impl Default for FeatureEngineer {
    fn default() -> Self {
        Self::new(true)
    }
}

impl FeatureEngineer {
    /// Create a new feature engineer.
    pub fn new(include_volume: bool) -> Self {
        Self { include_volume }
    }

    /// Compute all features from OHLCV bars.
    pub fn compute_features(&self, bars: &[OhlcvBar]) -> Vec<Vec<f64>> {
        let n = bars.len();
        if n == 0 {
            return Vec::new();
        }

        let mut features: Vec<Vec<f64>> = Vec::with_capacity(n);

        // Extract raw prices
        let closes: Vec<f64> = bars.iter().map(|b| b.close).collect();
        let highs: Vec<f64> = bars.iter().map(|b| b.high).collect();
        let lows: Vec<f64> = bars.iter().map(|b| b.low).collect();
        let volumes: Vec<f64> = bars.iter().map(|b| b.volume).collect();

        // Compute derived features
        let returns = self.compute_returns(&closes);
        let returns_5 = self.compute_returns_n(&closes, 5);
        let returns_10 = self.compute_returns_n(&closes, 10);
        let sma_5 = self.sma(&closes, 5);
        let sma_10 = self.sma(&closes, 10);
        let sma_20 = self.sma(&closes, 20);
        let ema_12 = self.ema(&closes, 12);
        let ema_26 = self.ema(&closes, 26);
        let macd = self.compute_macd(&ema_12, &ema_26);
        let macd_signal = self.ema(&macd, 9);
        let rsi = self.compute_rsi(&closes, 14);
        let volatility = self.compute_volatility(&returns, 10);
        let atr = self.compute_atr(bars, 14);
        let (bb_upper, bb_lower) = self.compute_bollinger_bands(&closes, 20, 2.0);

        // Volume features
        let volume_sma = if self.include_volume {
            self.sma(&volumes, 10)
        } else {
            vec![0.0; n]
        };

        // Assemble features for each bar
        for i in 0..n {
            let mut row = vec![
                returns[i],
                returns_5[i],
                returns_10[i],
                if sma_5[i] > 0.0 {
                    closes[i] / sma_5[i]
                } else {
                    1.0
                },
                if sma_10[i] > 0.0 {
                    closes[i] / sma_10[i]
                } else {
                    1.0
                },
                if sma_20[i] > 0.0 {
                    closes[i] / sma_20[i]
                } else {
                    1.0
                },
                macd[i],
                macd_signal[i],
                macd[i] - macd_signal[i], // MACD histogram
                rsi[i],
                volatility[i],
                atr[i],
                self.bb_position(closes[i], bb_upper[i], bb_lower[i]),
            ];

            if self.include_volume {
                row.push(if volume_sma[i] > 0.0 {
                    volumes[i] / volume_sma[i]
                } else {
                    1.0
                });
            }

            features.push(row);
        }

        features
    }

    /// Get feature names.
    pub fn feature_names(&self) -> Vec<String> {
        let mut names = vec![
            "returns".to_string(),
            "returns_5".to_string(),
            "returns_10".to_string(),
            "close_sma_5_ratio".to_string(),
            "close_sma_10_ratio".to_string(),
            "close_sma_20_ratio".to_string(),
            "macd".to_string(),
            "macd_signal".to_string(),
            "macd_hist".to_string(),
            "rsi_14".to_string(),
            "volatility_10".to_string(),
            "atr_14".to_string(),
            "bb_position".to_string(),
        ];

        if self.include_volume {
            names.push("volume_ratio".to_string());
        }

        names
    }

    /// Prepare dataset with sequences for model input.
    pub fn prepare_dataset(
        &self,
        bars: &[OhlcvBar],
        lookback: usize,
        label_threshold: f64,
        forward_periods: usize,
    ) -> Dataset {
        let features = self.compute_features(bars);
        let feature_names = self.feature_names();

        let mut dataset = Dataset::new(feature_names);

        // Create sequences with labels
        for i in lookback..bars.len().saturating_sub(forward_periods) {
            let sequence: Vec<Vec<f64>> = features[i - lookback..i].to_vec();

            // Compute label based on future returns
            let current_close = bars[i - 1].close;
            let future_close = bars[i + forward_periods - 1].close;
            let future_return = (future_close - current_close) / current_close;

            let label = if future_return > label_threshold {
                2 // Buy
            } else if future_return < -label_threshold {
                0 // Sell
            } else {
                1 // Hold
            };

            dataset.add_sample(sequence, Some(label));
        }

        dataset
    }

    /// Compute simple returns.
    fn compute_returns(&self, closes: &[f64]) -> Vec<f64> {
        let mut returns = vec![0.0];
        for i in 1..closes.len() {
            if closes[i - 1] > 0.0 {
                returns.push((closes[i] - closes[i - 1]) / closes[i - 1]);
            } else {
                returns.push(0.0);
            }
        }
        returns
    }

    /// Compute n-period returns.
    fn compute_returns_n(&self, closes: &[f64], n: usize) -> Vec<f64> {
        let mut returns = vec![0.0; closes.len()];
        for i in n..closes.len() {
            if closes[i - n] > 0.0 {
                returns[i] = (closes[i] - closes[i - n]) / closes[i - n];
            }
        }
        returns
    }

    /// Compute Simple Moving Average.
    fn sma(&self, values: &[f64], period: usize) -> Vec<f64> {
        let mut sma = vec![0.0; values.len()];
        for i in period - 1..values.len() {
            let sum: f64 = values[i + 1 - period..=i].iter().sum();
            sma[i] = sum / period as f64;
        }
        sma
    }

    /// Compute Exponential Moving Average.
    fn ema(&self, values: &[f64], period: usize) -> Vec<f64> {
        let mut ema = vec![0.0; values.len()];
        if values.is_empty() {
            return ema;
        }

        let multiplier = 2.0 / (period as f64 + 1.0);

        // Initialize with SMA
        if values.len() >= period {
            let sum: f64 = values[..period].iter().sum();
            ema[period - 1] = sum / period as f64;

            for i in period..values.len() {
                ema[i] = (values[i] - ema[i - 1]) * multiplier + ema[i - 1];
            }
        }

        ema
    }

    /// Compute MACD line.
    fn compute_macd(&self, ema_fast: &[f64], ema_slow: &[f64]) -> Vec<f64> {
        ema_fast
            .iter()
            .zip(ema_slow.iter())
            .map(|(f, s)| f - s)
            .collect()
    }

    /// Compute RSI.
    fn compute_rsi(&self, closes: &[f64], period: usize) -> Vec<f64> {
        let mut rsi = vec![50.0; closes.len()];
        if closes.len() < period + 1 {
            return rsi;
        }

        let mut gains = Vec::with_capacity(closes.len());
        let mut losses = Vec::with_capacity(closes.len());

        gains.push(0.0);
        losses.push(0.0);

        for i in 1..closes.len() {
            let change = closes[i] - closes[i - 1];
            if change > 0.0 {
                gains.push(change);
                losses.push(0.0);
            } else {
                gains.push(0.0);
                losses.push(-change);
            }
        }

        // Calculate average gains and losses
        let mut avg_gain: f64 = gains[1..=period].iter().sum::<f64>() / period as f64;
        let mut avg_loss: f64 = losses[1..=period].iter().sum::<f64>() / period as f64;

        for i in period..closes.len() {
            if i > period {
                avg_gain = (avg_gain * (period as f64 - 1.0) + gains[i]) / period as f64;
                avg_loss = (avg_loss * (period as f64 - 1.0) + losses[i]) / period as f64;
            }

            if avg_loss > 0.0 {
                let rs = avg_gain / avg_loss;
                rsi[i] = 100.0 - (100.0 / (1.0 + rs));
            } else {
                rsi[i] = 100.0;
            }
        }

        rsi
    }

    /// Compute rolling volatility.
    fn compute_volatility(&self, returns: &[f64], period: usize) -> Vec<f64> {
        let mut volatility = vec![0.0; returns.len()];

        for i in period - 1..returns.len() {
            let window = &returns[i + 1 - period..=i];
            let mean: f64 = window.iter().sum::<f64>() / period as f64;
            let variance: f64 =
                window.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / period as f64;
            volatility[i] = variance.sqrt();
        }

        volatility
    }

    /// Compute Average True Range.
    fn compute_atr(&self, bars: &[OhlcvBar], period: usize) -> Vec<f64> {
        let mut atr = vec![0.0; bars.len()];
        if bars.len() < 2 {
            return atr;
        }

        let mut tr_values = Vec::with_capacity(bars.len());
        tr_values.push(bars[0].high - bars[0].low);

        for i in 1..bars.len() {
            tr_values.push(bars[i].true_range(bars[i - 1].close));
        }

        // Compute SMA of TR
        atr = self.sma(&tr_values, period);
        atr
    }

    /// Compute Bollinger Bands.
    fn compute_bollinger_bands(
        &self,
        closes: &[f64],
        period: usize,
        num_std: f64,
    ) -> (Vec<f64>, Vec<f64>) {
        let sma = self.sma(closes, period);
        let mut upper = vec![0.0; closes.len()];
        let mut lower = vec![0.0; closes.len()];

        for i in period - 1..closes.len() {
            let window = &closes[i + 1 - period..=i];
            let std_dev = self.std_dev(window);
            upper[i] = sma[i] + num_std * std_dev;
            lower[i] = sma[i] - num_std * std_dev;
        }

        (upper, lower)
    }

    /// Compute standard deviation.
    fn std_dev(&self, values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
        let variance: f64 =
            values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
        variance.sqrt()
    }

    /// Compute Bollinger Band position (0 to 1).
    fn bb_position(&self, close: f64, upper: f64, lower: f64) -> f64 {
        if (upper - lower).abs() < 1e-10 {
            return 0.5;
        }
        ((close - lower) / (upper - lower)).clamp(0.0, 1.0)
    }
}

/// Normalize features using z-score normalization.
pub fn normalize_features(features: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<f64>, Vec<f64>) {
    if features.is_empty() || features[0].is_empty() {
        return (Vec::new(), Vec::new(), Vec::new());
    }

    let n_features = features[0].len();
    let n_samples = features.len();

    // Compute mean and std for each feature
    let mut means = vec![0.0; n_features];
    let mut stds = vec![0.0; n_features];

    for f in 0..n_features {
        let values: Vec<f64> = features.iter().map(|row| row[f]).collect();
        means[f] = values.iter().sum::<f64>() / n_samples as f64;
        let variance: f64 =
            values.iter().map(|v| (v - means[f]).powi(2)).sum::<f64>() / n_samples as f64;
        stds[f] = variance.sqrt().max(1e-10); // Avoid division by zero
    }

    // Normalize
    let normalized: Vec<Vec<f64>> = features
        .iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .map(|(f, v)| (v - means[f]) / stds[f])
                .collect()
        })
        .collect();

    (normalized, means, stds)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_bars() -> Vec<OhlcvBar> {
        (0..100)
            .map(|i| {
                let base = 100.0 + (i as f64 * 0.1);
                OhlcvBar::new(
                    i * 1000,
                    base,
                    base + 1.0,
                    base - 0.5,
                    base + 0.5,
                    1000.0 + i as f64 * 10.0,
                )
            })
            .collect()
    }

    #[test]
    fn test_compute_features() {
        let bars = create_test_bars();
        let fe = FeatureEngineer::new(true);
        let features = fe.compute_features(&bars);

        assert_eq!(features.len(), bars.len());
        assert_eq!(features[0].len(), fe.feature_names().len());
    }

    #[test]
    fn test_sma() {
        let fe = FeatureEngineer::new(true);
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sma = fe.sma(&values, 3);

        assert!((sma[2] - 2.0).abs() < 0.01);
        assert!((sma[3] - 3.0).abs() < 0.01);
        assert!((sma[4] - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_normalize() {
        let features = vec![vec![1.0, 10.0], vec![2.0, 20.0], vec![3.0, 30.0]];

        let (normalized, means, stds) = normalize_features(&features);

        assert_eq!(normalized.len(), 3);
        assert!((means[0] - 2.0).abs() < 0.01);
        assert!((means[1] - 20.0).abs() < 0.01);
    }
}
