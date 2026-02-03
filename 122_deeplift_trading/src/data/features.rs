//! Feature engineering for trading data.

use super::bybit::Kline;

/// Feature generator for technical indicators.
#[derive(Debug, Clone)]
pub struct FeatureGenerator {
    /// Lookback window for indicators
    pub window: usize,
}

impl FeatureGenerator {
    /// Create a new feature generator.
    pub fn new(window: usize) -> Self {
        Self { window }
    }

    /// Compute all features from kline data.
    ///
    /// Returns 11 features:
    /// - returns_1d, returns_5d, returns_10d
    /// - sma_ratio, ema_ratio
    /// - volatility, momentum
    /// - rsi, macd
    /// - bb_position, volume_sma_ratio
    pub fn compute_features(&self, klines: &[Kline]) -> Vec<Vec<f64>> {
        if klines.len() < self.window + 26 {
            return Vec::new();
        }

        let closes: Vec<f64> = klines.iter().map(|k| k.close).collect();
        let volumes: Vec<f64> = klines.iter().map(|k| k.volume).collect();

        let returns_1 = self.compute_returns(&closes, 1);
        let returns_5 = self.compute_returns(&closes, 5);
        let returns_10 = self.compute_returns(&closes, 10);
        let sma_ratio = self.compute_sma_ratio(&closes);
        let ema_ratio = self.compute_ema_ratio(&closes);
        let volatility = self.compute_volatility(&closes);
        let momentum = self.compute_momentum(&closes);
        let rsi = self.compute_rsi(&closes, 14);
        let macd = self.compute_macd(&closes);
        let bb_position = self.compute_bollinger_position(&closes);
        let volume_ratio = self.compute_volume_sma_ratio(&volumes);

        // Find minimum length
        let min_len = [
            returns_1.len(),
            returns_5.len(),
            returns_10.len(),
            sma_ratio.len(),
            ema_ratio.len(),
            volatility.len(),
            momentum.len(),
            rsi.len(),
            macd.len(),
            bb_position.len(),
            volume_ratio.len(),
        ]
        .into_iter()
        .min()
        .unwrap_or(0);

        if min_len == 0 {
            return Vec::new();
        }

        // Combine features
        let mut features = Vec::with_capacity(min_len);
        for i in 0..min_len {
            features.push(vec![
                returns_1[returns_1.len() - min_len + i],
                returns_5[returns_5.len() - min_len + i],
                returns_10[returns_10.len() - min_len + i],
                sma_ratio[sma_ratio.len() - min_len + i],
                ema_ratio[ema_ratio.len() - min_len + i],
                volatility[volatility.len() - min_len + i],
                momentum[momentum.len() - min_len + i],
                rsi[rsi.len() - min_len + i],
                macd[macd.len() - min_len + i],
                bb_position[bb_position.len() - min_len + i],
                volume_ratio[volume_ratio.len() - min_len + i],
            ]);
        }

        features
    }

    fn compute_returns(&self, closes: &[f64], period: usize) -> Vec<f64> {
        if closes.len() <= period {
            return Vec::new();
        }
        closes
            .iter()
            .skip(period)
            .zip(closes.iter())
            .map(|(current, past)| current / past - 1.0)
            .collect()
    }

    fn compute_sma_ratio(&self, closes: &[f64]) -> Vec<f64> {
        if closes.len() < self.window {
            return Vec::new();
        }

        let mut result = Vec::with_capacity(closes.len() - self.window + 1);
        for i in (self.window - 1)..closes.len() {
            let sma: f64 = closes[(i + 1 - self.window)..=i].iter().sum::<f64>() / self.window as f64;
            result.push(closes[i] / sma - 1.0);
        }
        result
    }

    fn compute_ema_ratio(&self, closes: &[f64]) -> Vec<f64> {
        if closes.len() < self.window {
            return Vec::new();
        }

        let alpha = 2.0 / (self.window as f64 + 1.0);
        let mut ema = closes[0];
        let mut ema_values = vec![ema];

        for &close in closes.iter().skip(1) {
            ema = alpha * close + (1.0 - alpha) * ema;
            ema_values.push(ema);
        }

        closes
            .iter()
            .skip(self.window - 1)
            .zip(ema_values.iter().skip(self.window - 1))
            .map(|(close, ema)| close / ema - 1.0)
            .collect()
    }

    fn compute_volatility(&self, closes: &[f64]) -> Vec<f64> {
        if closes.len() < self.window + 1 {
            return Vec::new();
        }

        let log_returns: Vec<f64> = closes
            .windows(2)
            .map(|w| (w[1] / w[0]).ln())
            .collect();

        let mut result = Vec::with_capacity(log_returns.len() - self.window + 1);
        for i in (self.window - 1)..log_returns.len() {
            let window_returns = &log_returns[(i + 1 - self.window)..=i];
            let mean: f64 = window_returns.iter().sum::<f64>() / self.window as f64;
            let variance: f64 = window_returns
                .iter()
                .map(|r| (r - mean).powi(2))
                .sum::<f64>()
                / self.window as f64;
            result.push(variance.sqrt());
        }
        result
    }

    fn compute_momentum(&self, closes: &[f64]) -> Vec<f64> {
        if closes.len() < self.window {
            return Vec::new();
        }
        closes
            .iter()
            .skip(self.window - 1)
            .zip(closes.iter())
            .map(|(current, past)| current / past - 1.0)
            .collect()
    }

    fn compute_rsi(&self, closes: &[f64], period: usize) -> Vec<f64> {
        if closes.len() < period + 1 {
            return Vec::new();
        }

        let changes: Vec<f64> = closes.windows(2).map(|w| w[1] - w[0]).collect();
        let gains: Vec<f64> = changes.iter().map(|&c| c.max(0.0)).collect();
        let losses: Vec<f64> = changes.iter().map(|&c| (-c).max(0.0)).collect();

        let mut result = Vec::with_capacity(changes.len() - period + 1);
        for i in (period - 1)..changes.len() {
            let avg_gain: f64 = gains[(i + 1 - period)..=i].iter().sum::<f64>() / period as f64;
            let avg_loss: f64 = losses[(i + 1 - period)..=i].iter().sum::<f64>() / period as f64;

            let rsi = if avg_loss > 0.0 {
                100.0 - (100.0 / (1.0 + avg_gain / avg_loss))
            } else {
                100.0
            };
            result.push(rsi / 100.0); // Normalize to [0, 1]
        }
        result
    }

    fn compute_macd(&self, closes: &[f64]) -> Vec<f64> {
        if closes.len() < 26 {
            return Vec::new();
        }

        // EMA 12
        let alpha12 = 2.0 / 13.0;
        let mut ema12 = closes[0];
        let mut ema12_values = vec![ema12];
        for &close in closes.iter().skip(1) {
            ema12 = alpha12 * close + (1.0 - alpha12) * ema12;
            ema12_values.push(ema12);
        }

        // EMA 26
        let alpha26 = 2.0 / 27.0;
        let mut ema26 = closes[0];
        let mut ema26_values = vec![ema26];
        for &close in closes.iter().skip(1) {
            ema26 = alpha26 * close + (1.0 - alpha26) * ema26;
            ema26_values.push(ema26);
        }

        ema12_values
            .iter()
            .skip(25)
            .zip(ema26_values.iter().skip(25))
            .zip(closes.iter().skip(25))
            .map(|((e12, e26), close)| (e12 - e26) / close)
            .collect()
    }

    fn compute_bollinger_position(&self, closes: &[f64]) -> Vec<f64> {
        if closes.len() < self.window {
            return Vec::new();
        }

        let mut result = Vec::with_capacity(closes.len() - self.window + 1);
        for i in (self.window - 1)..closes.len() {
            let window_data = &closes[(i + 1 - self.window)..=i];
            let mean: f64 = window_data.iter().sum::<f64>() / self.window as f64;
            let variance: f64 = window_data.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                / self.window as f64;
            let std = variance.sqrt();

            let position = if std > 0.0 {
                (closes[i] - mean) / (2.0 * std)
            } else {
                0.0
            };
            result.push(position);
        }
        result
    }

    fn compute_volume_sma_ratio(&self, volumes: &[f64]) -> Vec<f64> {
        if volumes.len() < self.window {
            return Vec::new();
        }

        let mut result = Vec::with_capacity(volumes.len() - self.window + 1);
        for i in (self.window - 1)..volumes.len() {
            let sma: f64 =
                volumes[(i + 1 - self.window)..=i].iter().sum::<f64>() / self.window as f64;
            let ratio = if sma > 0.0 {
                volumes[i] / sma - 1.0
            } else {
                0.0
            };
            result.push(ratio);
        }
        result
    }
}

/// Feature names for the 11 standard features.
pub fn feature_names() -> Vec<String> {
    vec![
        "returns_1d".to_string(),
        "returns_5d".to_string(),
        "returns_10d".to_string(),
        "sma_ratio".to_string(),
        "ema_ratio".to_string(),
        "volatility".to_string(),
        "momentum".to_string(),
        "rsi".to_string(),
        "macd".to_string(),
        "bb_position".to_string(),
        "volume_sma_ratio".to_string(),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_klines(n: usize) -> Vec<Kline> {
        let mut klines = Vec::with_capacity(n);
        let mut price = 50000.0;

        for i in 0..n {
            price *= 1.0 + (i as f64 * 0.001).sin() * 0.01;
            klines.push(Kline {
                timestamp: i as i64 * 3600000,
                open: price,
                high: price * 1.005,
                low: price * 0.995,
                close: price,
                volume: 1000000.0,
                turnover: price * 1000000.0,
            });
        }
        klines
    }

    #[test]
    fn test_feature_computation() {
        let klines = create_test_klines(100);
        let generator = FeatureGenerator::new(20);
        let features = generator.compute_features(&klines);

        assert!(!features.is_empty());
        assert_eq!(features[0].len(), 11);
    }
}
