//! Feature Extractor for Market Data

use crate::data::Candle;

/// Feature extractor for market data
pub struct FeatureExtractor {
    lookback: usize,
}

impl FeatureExtractor {
    /// Create a new feature extractor
    pub fn new(lookback: usize) -> Self {
        Self { lookback }
    }

    /// Extract features from candle data
    pub fn extract(&self, candles: &[Candle]) -> Vec<f64> {
        if candles.is_empty() { return vec![0.0; 10]; }

        let returns: Vec<f64> = candles.windows(2)
            .map(|w| (w[1].close - w[0].close) / w[0].close).collect();
        let last = candles.last().unwrap();

        vec![
            *returns.last().unwrap_or(&0.0),
            returns.iter().rev().take(24).sum(),
            self.std_dev(&returns) * (24.0 * 365.0_f64).sqrt(),
            self.skewness(&returns),
            self.kurtosis(&returns),
            self.ema(&returns, 12),
            self.rsi(&returns, 14),
            last.close_position(),
            last.range_pct(),
            (last.volume / candles.iter().map(|c| c.volume).sum::<f64>()) * candles.len() as f64,
        ]
    }

    fn std_dev(&self, v: &[f64]) -> f64 {
        if v.len() < 2 { return 0.0; }
        let m = v.iter().sum::<f64>() / v.len() as f64;
        (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (v.len() - 1) as f64).sqrt()
    }

    fn skewness(&self, v: &[f64]) -> f64 {
        if v.len() < 3 { return 0.0; }
        let m = v.iter().sum::<f64>() / v.len() as f64;
        let s = self.std_dev(v);
        if s < 1e-10 { return 0.0; }
        v.iter().map(|x| ((x - m) / s).powi(3)).sum::<f64>() / v.len() as f64
    }

    fn kurtosis(&self, v: &[f64]) -> f64 {
        if v.len() < 4 { return 0.0; }
        let m = v.iter().sum::<f64>() / v.len() as f64;
        let s = self.std_dev(v);
        if s < 1e-10 { return 0.0; }
        v.iter().map(|x| ((x - m) / s).powi(4)).sum::<f64>() / v.len() as f64 - 3.0
    }

    fn ema(&self, v: &[f64], span: usize) -> f64 {
        if v.is_empty() { return 0.0; }
        let alpha = 2.0 / (span as f64 + 1.0);
        v.iter().fold(v[0], |acc, &x| alpha * x + (1.0 - alpha) * acc)
    }

    fn rsi(&self, returns: &[f64], period: usize) -> f64 {
        if returns.len() < period { return 50.0; }
        let recent: Vec<f64> = returns.iter().rev().take(period).cloned().collect();
        let gains: f64 = recent.iter().filter(|&&r| r > 0.0).sum();
        let losses: f64 = recent.iter().filter(|&&r| r < 0.0).map(|r| r.abs()).sum();
        if losses < 1e-10 { 100.0 } else { 100.0 - 100.0 / (1.0 + gains / losses) }
    }
}

impl Default for FeatureExtractor {
    fn default() -> Self { Self::new(168) }
}
