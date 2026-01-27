//! Technical feature computation for trading data.

use crate::data::bybit::Kline;

/// Feature generator for computing technical indicators
#[derive(Debug, Clone)]
pub struct FeatureGenerator {
    window: usize,
}

impl FeatureGenerator {
    /// Create with specified window
    pub fn new(window: usize) -> Self {
        Self { window }
    }

    /// Create with default window of 20
    pub fn default_window() -> Self {
        Self { window: 20 }
    }

    /// Compute all features from kline data
    ///
    /// Returns Vec of feature vectors with 11 features each
    pub fn compute_features(&self, klines: &[Kline]) -> Vec<Vec<f64>> {
        if klines.len() < self.window + 10 {
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
        let vol_sma_ratio = self.compute_volume_sma_ratio(&volumes);

        let min_len = [
            returns_1.len(), returns_5.len(), returns_10.len(),
            sma_ratio.len(), ema_ratio.len(), volatility.len(),
            momentum.len(), rsi.len(), macd.len(), bb_position.len(),
            vol_sma_ratio.len(),
        ].iter().cloned().min().unwrap_or(0);

        if min_len == 0 {
            return Vec::new();
        }

        (0..min_len).map(|i| {
            vec![
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
                vol_sma_ratio[vol_sma_ratio.len() - min_len + i],
            ]
        }).collect()
    }

    fn compute_returns(&self, closes: &[f64], period: usize) -> Vec<f64> {
        if closes.len() <= period { return Vec::new(); }
        (period..closes.len()).map(|i| closes[i] / closes[i - period] - 1.0).collect()
    }

    fn compute_sma_ratio(&self, closes: &[f64]) -> Vec<f64> {
        if closes.len() < self.window { return Vec::new(); }
        let mut result = Vec::new();
        for i in (self.window - 1)..closes.len() {
            let sma: f64 = closes[i + 1 - self.window..=i].iter().sum::<f64>() / self.window as f64;
            result.push(closes[i] / sma - 1.0);
        }
        result
    }

    fn compute_ema_ratio(&self, closes: &[f64]) -> Vec<f64> {
        if closes.len() < self.window { return Vec::new(); }
        let alpha = 2.0 / (self.window as f64 + 1.0);
        let mut ema = vec![0.0; closes.len()];
        ema[0] = closes[0];
        for i in 1..closes.len() {
            ema[i] = alpha * closes[i] + (1.0 - alpha) * ema[i - 1];
        }
        (self.window - 1..closes.len()).map(|i| closes[i] / ema[i] - 1.0).collect()
    }

    fn compute_volatility(&self, closes: &[f64]) -> Vec<f64> {
        if closes.len() < self.window + 1 { return Vec::new(); }
        let log_returns: Vec<f64> = closes.windows(2)
            .map(|w| (w[1] / w[0]).ln())
            .collect();
        (self.window - 1..log_returns.len()).map(|i| {
            let slice = &log_returns[i + 1 - self.window..=i];
            let mean = slice.iter().sum::<f64>() / slice.len() as f64;
            (slice.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / slice.len() as f64).sqrt()
        }).collect()
    }

    fn compute_momentum(&self, closes: &[f64]) -> Vec<f64> {
        if closes.len() < self.window + 1 { return Vec::new(); }
        (self.window..closes.len()).map(|i| closes[i] / closes[i - self.window] - 1.0).collect()
    }

    fn compute_rsi(&self, closes: &[f64], period: usize) -> Vec<f64> {
        if closes.len() < period + 1 { return Vec::new(); }
        let deltas: Vec<f64> = closes.windows(2).map(|w| w[1] - w[0]).collect();
        let gains: Vec<f64> = deltas.iter().map(|d| d.max(0.0)).collect();
        let losses: Vec<f64> = deltas.iter().map(|d| (-d).max(0.0)).collect();

        let mut result = Vec::new();
        for i in period..=deltas.len() {
            let avg_gain: f64 = gains[i - period..i].iter().sum::<f64>() / period as f64;
            let avg_loss: f64 = losses[i - period..i].iter().sum::<f64>() / period as f64;
            let rs = if avg_loss != 0.0 { avg_gain / avg_loss } else { 100.0 };
            result.push((100.0 - 100.0 / (1.0 + rs)) / 100.0);
        }
        result
    }

    fn compute_macd(&self, closes: &[f64]) -> Vec<f64> {
        if closes.len() < 26 { return Vec::new(); }
        let mut ema12 = vec![0.0; closes.len()];
        let mut ema26 = vec![0.0; closes.len()];
        ema12[0] = closes[0];
        ema26[0] = closes[0];
        for i in 1..closes.len() {
            ema12[i] = 2.0 / 13.0 * closes[i] + 11.0 / 13.0 * ema12[i - 1];
            ema26[i] = 2.0 / 27.0 * closes[i] + 25.0 / 27.0 * ema26[i - 1];
        }
        (25..closes.len()).map(|i| (ema12[i] - ema26[i]) / closes[i]).collect()
    }

    fn compute_bollinger_position(&self, closes: &[f64]) -> Vec<f64> {
        if closes.len() < self.window { return Vec::new(); }
        (self.window - 1..closes.len()).map(|i| {
            let window = &closes[i + 1 - self.window..=i];
            let mean = window.iter().sum::<f64>() / window.len() as f64;
            let std = (window.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / window.len() as f64).sqrt();
            if std > 0.0 { (closes[i] - mean) / (2.0 * std) } else { 0.0 }
        }).collect()
    }

    fn compute_volume_sma_ratio(&self, volumes: &[f64]) -> Vec<f64> {
        if volumes.len() < self.window { return Vec::new(); }
        let mut result = Vec::new();
        for i in (self.window - 1)..volumes.len() {
            let sma: f64 = volumes[i + 1 - self.window..=i].iter().sum::<f64>() / self.window as f64;
            result.push(if sma != 0.0 { volumes[i] / sma - 1.0 } else { 0.0 });
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::bybit::SimulatedDataGenerator;

    #[test]
    fn test_feature_computation() {
        let klines = SimulatedDataGenerator::generate_klines(200, 50000.0, 0.02);
        let gen = FeatureGenerator::default_window();
        let features = gen.compute_features(&klines);
        assert!(!features.is_empty());
        assert_eq!(features[0].len(), 11);
    }
}
