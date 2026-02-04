//! Feature engineering engine

use crate::api::types::Kline;
use super::indicators::TechnicalIndicators;
use ndarray::Array1;

/// Feature engineering engine
///
/// Computes technical indicators and prepares features for VAE input.
pub struct FeatureEngine {
    /// RSI period
    rsi_period: usize,
    /// MACD parameters (fast, slow, signal)
    macd_params: (usize, usize, usize),
    /// Bollinger Bands period
    bb_period: usize,
    /// Volatility lookback
    volatility_period: usize,
}

impl Default for FeatureEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl FeatureEngine {
    /// Create a new feature engine with default parameters
    pub fn new() -> Self {
        Self {
            rsi_period: 14,
            macd_params: (12, 26, 9),
            bb_period: 20,
            volatility_period: 20,
        }
    }

    /// Configure RSI period
    pub fn with_rsi_period(mut self, period: usize) -> Self {
        self.rsi_period = period;
        self
    }

    /// Configure Bollinger Bands period
    pub fn with_bb_period(mut self, period: usize) -> Self {
        self.bb_period = period;
        self
    }

    /// Compute all features from klines
    ///
    /// Returns a feature matrix where each row is a timestep and columns are features.
    pub fn compute(&self, klines: &[Kline]) -> Vec<Vec<f64>> {
        if klines.is_empty() {
            return vec![];
        }

        // Extract price data
        let closes: Vec<f64> = klines.iter().map(|k| k.close).collect();
        let volumes: Vec<f64> = klines.iter().map(|k| k.volume).collect();

        // Calculate indicators
        let returns = TechnicalIndicators::returns(&closes);
        let log_returns = TechnicalIndicators::log_returns(&closes);
        let volatility = TechnicalIndicators::volatility(&closes, self.volatility_period);
        let atr = TechnicalIndicators::atr(klines, 14);
        let rsi = TechnicalIndicators::rsi(&closes, self.rsi_period);
        let (macd, macd_signal, macd_hist) = TechnicalIndicators::macd(
            &closes,
            self.macd_params.0,
            self.macd_params.1,
            self.macd_params.2,
        );
        let (bb_upper, bb_middle, bb_lower) = TechnicalIndicators::bollinger_bands(
            &closes,
            self.bb_period,
            2.0,
        );

        // Volume features
        let volume_sma = TechnicalIndicators::sma(&volumes, 20);
        let volume_ratio: Vec<f64> = volumes.iter()
            .zip(volume_sma.iter())
            .map(|(&v, &sma)| if sma > 0.0 { v / sma } else { 1.0 })
            .collect();

        // Bollinger band position
        let bb_position: Vec<f64> = closes.iter()
            .zip(bb_upper.iter().zip(bb_lower.iter()))
            .map(|(&c, (&u, &l))| {
                if u != l && !u.is_nan() && !l.is_nan() {
                    (c - l) / (u - l)
                } else {
                    0.5
                }
            })
            .collect();

        // Bollinger band width
        let bb_width: Vec<f64> = bb_upper.iter()
            .zip(bb_lower.iter().zip(bb_middle.iter()))
            .map(|(&u, (&l, &m))| {
                if m > 0.0 && !u.is_nan() && !l.is_nan() {
                    (u - l) / m
                } else {
                    0.0
                }
            })
            .collect();

        // Normalize prices
        let close_sma50 = TechnicalIndicators::sma(&closes, 50);
        let close_norm: Vec<f64> = closes.iter()
            .zip(close_sma50.iter())
            .enumerate()
            .map(|(i, (&c, &sma))| {
                if sma > 0.0 && !sma.is_nan() {
                    // Calculate rolling std
                    let start = if i >= 50 { i - 49 } else { 0 };
                    let slice = &closes[start..=i];
                    let mean: f64 = slice.iter().sum::<f64>() / slice.len() as f64;
                    let std: f64 = (slice.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                        / slice.len() as f64).sqrt();
                    if std > 0.0 { (c - sma) / std } else { 0.0 }
                } else {
                    0.0
                }
            })
            .collect();

        // Volume normalization
        let volume_sma50 = TechnicalIndicators::sma(&volumes, 50);
        let volume_norm: Vec<f64> = volumes.iter()
            .zip(volume_sma50.iter())
            .enumerate()
            .map(|(i, (&v, &sma))| {
                if sma > 0.0 && !sma.is_nan() {
                    let start = if i >= 50 { i - 49 } else { 0 };
                    let slice = &volumes[start..=i];
                    let mean: f64 = slice.iter().sum::<f64>() / slice.len() as f64;
                    let std: f64 = (slice.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                        / slice.len() as f64).sqrt();
                    if std > 0.0 { (v - sma) / std } else { 0.0 }
                } else {
                    0.0
                }
            })
            .collect();

        // Build feature matrix
        let n = klines.len();
        let mut features = Vec::with_capacity(n);

        for i in 0..n {
            let row = vec![
                self.safe_value(returns[i]),
                self.safe_value(log_returns[i]),
                self.safe_value(volatility[i]),
                self.safe_value(atr[i]),
                self.safe_value(rsi[i]) / 100.0,  // Normalize to [0, 1]
                self.safe_value(macd_hist[i]),
                self.safe_value(bb_position[i]),
                self.safe_value(bb_width[i]),
                self.safe_value(volume_ratio[i]),
                self.safe_value(close_norm[i]),
                self.safe_value(volume_norm[i]),
                // Additional derived features
                if klines[i].is_bullish() { 1.0 } else { 0.0 },
                self.safe_value(klines[i].body() / klines[i].range().max(0.0001)),
            ];
            features.push(row);
        }

        features
    }

    /// Convert feature matrix to flattened sequence for VAE input
    pub fn to_sequence(&self, features: &[Vec<f64>], seq_len: usize) -> Vec<Array1<f64>> {
        if features.len() < seq_len {
            return vec![];
        }

        let mut sequences = Vec::new();
        let n_features = features[0].len();

        for i in seq_len..=features.len() {
            let mut seq = Vec::with_capacity(seq_len * n_features);
            for j in (i - seq_len)..i {
                seq.extend_from_slice(&features[j]);
            }
            sequences.push(Array1::from_vec(seq));
        }

        sequences
    }

    /// Get number of features
    pub fn n_features(&self) -> usize {
        13  // Based on compute() output
    }

    /// Safely convert value (replace NaN/Inf with 0)
    fn safe_value(&self, v: f64) -> f64 {
        if v.is_finite() { v } else { 0.0 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_klines() -> Vec<Kline> {
        (0..100).map(|i| {
            let base = 100.0 + (i as f64 * 0.1).sin() * 5.0;
            Kline {
                start_time: i * 60000,
                open: base,
                high: base + 1.0,
                low: base - 1.0,
                close: base + 0.5,
                volume: 1000.0 + i as f64 * 10.0,
                turnover: 0.0,
            }
        }).collect()
    }

    #[test]
    fn test_feature_computation() {
        let klines = create_test_klines();
        let engine = FeatureEngine::new();
        let features = engine.compute(&klines);

        assert_eq!(features.len(), klines.len());
        assert_eq!(features[0].len(), engine.n_features());
    }

    #[test]
    fn test_sequence_creation() {
        let klines = create_test_klines();
        let engine = FeatureEngine::new();
        let features = engine.compute(&klines);
        let sequences = engine.to_sequence(&features, 60);

        assert_eq!(sequences.len(), features.len() - 60 + 1);
        assert_eq!(sequences[0].len(), 60 * engine.n_features());
    }
}
