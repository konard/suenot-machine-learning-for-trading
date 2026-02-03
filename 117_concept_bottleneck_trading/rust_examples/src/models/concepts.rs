//! Trading concepts for the Concept Bottleneck Model.

use crate::api::bybit::Kline;
use serde::{Deserialize, Serialize};

/// Direction of the market trend.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum TrendDirection {
    Bullish,
    Bearish,
    Neutral,
}

/// Volatility regime classification.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum VolatilityRegime {
    High,
    Normal,
    Low,
}

/// Momentum signal classification.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum MomentumSignal {
    Positive,
    Negative,
    Neutral,
}

/// Volume profile relative to average.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum VolumeProfile {
    AboveAverage,
    Average,
    BelowAverage,
}

/// Container for all extracted trading concepts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingConcepts {
    pub trend_direction: TrendDirection,
    pub trend_strength: f64,
    pub volatility_regime: VolatilityRegime,
    pub volatility_value: f64,
    pub momentum_signal: MomentumSignal,
    pub momentum_value: f64,
    pub volume_profile: VolumeProfile,
    pub volume_ratio: f64,
    pub rsi: f64,
    pub support_distance: f64,
    pub resistance_distance: f64,
}

impl TradingConcepts {
    /// Convert concepts to a numerical vector for model input.
    pub fn to_vector(&self) -> Vec<f64> {
        vec![
            match self.trend_direction {
                TrendDirection::Bullish => 1.0,
                TrendDirection::Neutral => 0.0,
                TrendDirection::Bearish => -1.0,
            },
            self.trend_strength,
            match self.volatility_regime {
                VolatilityRegime::High => 1.0,
                VolatilityRegime::Normal => 0.0,
                VolatilityRegime::Low => -1.0,
            },
            self.volatility_value,
            match self.momentum_signal {
                MomentumSignal::Positive => 1.0,
                MomentumSignal::Neutral => 0.0,
                MomentumSignal::Negative => -1.0,
            },
            self.momentum_value,
            match self.volume_profile {
                VolumeProfile::AboveAverage => 1.0,
                VolumeProfile::Average => 0.0,
                VolumeProfile::BelowAverage => -1.0,
            },
            self.volume_ratio,
            self.rsi / 100.0,
            self.support_distance,
            self.resistance_distance,
        ]
    }
}

/// Extracts trading concepts from market data.
pub struct ConceptExtractor {
    pub trend_window: usize,
    pub volatility_window: usize,
    pub volume_window: usize,
    pub rsi_window: usize,
}

impl ConceptExtractor {
    /// Create a new concept extractor with default parameters.
    pub fn new() -> Self {
        Self {
            trend_window: 20,
            volatility_window: 20,
            volume_window: 20,
            rsi_window: 14,
        }
    }

    /// Create a concept extractor with custom parameters.
    pub fn with_params(
        trend_window: usize,
        volatility_window: usize,
        volume_window: usize,
        rsi_window: usize,
    ) -> Self {
        Self {
            trend_window,
            volatility_window,
            volume_window,
            rsi_window,
        }
    }

    /// Extract trading concepts from kline data.
    pub fn extract(&self, klines: &[Kline]) -> Option<TradingConcepts> {
        let min_required = self.trend_window.max(self.volatility_window).max(self.rsi_window);
        if klines.len() < min_required {
            return None;
        }

        let closes: Vec<f64> = klines.iter().map(|k| k.close).collect();
        let highs: Vec<f64> = klines.iter().map(|k| k.high).collect();
        let lows: Vec<f64> = klines.iter().map(|k| k.low).collect();
        let volumes: Vec<f64> = klines.iter().map(|k| k.volume).collect();

        let (trend_dir, trend_str) = self.compute_trend(&closes);
        let (vol_regime, vol_value) = self.compute_volatility(&closes);
        let (mom_signal, mom_value) = self.compute_momentum(&closes);
        let (vol_profile, vol_ratio) = self.compute_volume_profile(&volumes);
        let rsi = self.compute_rsi(&closes);
        let (support_dist, resist_dist) = self.compute_support_resistance(&closes, &highs, &lows);

        Some(TradingConcepts {
            trend_direction: trend_dir,
            trend_strength: trend_str,
            volatility_regime: vol_regime,
            volatility_value: vol_value,
            momentum_signal: mom_signal,
            momentum_value: mom_value,
            volume_profile: vol_profile,
            volume_ratio: vol_ratio,
            rsi,
            support_distance: support_dist,
            resistance_distance: resist_dist,
        })
    }

    fn compute_trend(&self, closes: &[f64]) -> (TrendDirection, f64) {
        let n = closes.len();
        let window = &closes[n.saturating_sub(self.trend_window)..];

        // Linear regression slope
        let n_w = window.len() as f64;
        let x_mean = (n_w - 1.0) / 2.0;
        let y_mean: f64 = window.iter().sum::<f64>() / n_w;

        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for (i, &y) in window.iter().enumerate() {
            let x = i as f64;
            numerator += (x - x_mean) * (y - y_mean);
            denominator += (x - x_mean).powi(2);
        }

        let slope = if denominator != 0.0 {
            numerator / denominator
        } else {
            0.0
        };

        let normalized_slope = slope / y_mean;
        let strength = (normalized_slope.abs() * 100.0).min(1.0);

        let direction = if normalized_slope > 0.001 {
            TrendDirection::Bullish
        } else if normalized_slope < -0.001 {
            TrendDirection::Bearish
        } else {
            TrendDirection::Neutral
        };

        (direction, strength)
    }

    fn compute_volatility(&self, closes: &[f64]) -> (VolatilityRegime, f64) {
        if closes.len() < 2 {
            return (VolatilityRegime::Normal, 1.0);
        }

        // Calculate returns
        let returns: Vec<f64> = closes
            .windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect();

        let n = returns.len();
        let window_start = n.saturating_sub(self.volatility_window);

        // Current volatility
        let current_window = &returns[window_start..];
        let current_vol = std_dev(current_window);

        // Historical volatility
        let historical_vol = std_dev(&returns);

        let vol_ratio = if historical_vol > 0.0 {
            current_vol / historical_vol
        } else {
            1.0
        };

        let regime = if vol_ratio > 1.5 {
            VolatilityRegime::High
        } else if vol_ratio < 0.7 {
            VolatilityRegime::Low
        } else {
            VolatilityRegime::Normal
        };

        (regime, vol_ratio)
    }

    fn compute_momentum(&self, closes: &[f64]) -> (MomentumSignal, f64) {
        let n = closes.len();
        let lookback = self.trend_window.min(n - 1);

        let roc = (closes[n - 1] - closes[n - 1 - lookback]) / closes[n - 1 - lookback];

        let signal = if roc > 0.02 {
            MomentumSignal::Positive
        } else if roc < -0.02 {
            MomentumSignal::Negative
        } else {
            MomentumSignal::Neutral
        };

        (signal, roc)
    }

    fn compute_volume_profile(&self, volumes: &[f64]) -> (VolumeProfile, f64) {
        let n = volumes.len();
        let window_start = n.saturating_sub(self.volume_window);

        let current_vol = volumes[n - 1];
        let avg_vol: f64 = volumes[window_start..].iter().sum::<f64>()
            / (n - window_start) as f64;

        let vol_ratio = if avg_vol > 0.0 {
            current_vol / avg_vol
        } else {
            1.0
        };

        let profile = if vol_ratio > 1.5 {
            VolumeProfile::AboveAverage
        } else if vol_ratio < 0.7 {
            VolumeProfile::BelowAverage
        } else {
            VolumeProfile::Average
        };

        (profile, vol_ratio)
    }

    fn compute_rsi(&self, closes: &[f64]) -> f64 {
        if closes.len() < self.rsi_window + 1 {
            return 50.0;
        }

        let changes: Vec<f64> = closes.windows(2).map(|w| w[1] - w[0]).collect();
        let n = changes.len();
        let window = &changes[n.saturating_sub(self.rsi_window)..];

        let gains: f64 = window.iter().filter(|&&x| x > 0.0).sum();
        let losses: f64 = window.iter().filter(|&&x| x < 0.0).map(|x| -x).sum();

        let avg_gain = gains / self.rsi_window as f64;
        let avg_loss = losses / self.rsi_window as f64;

        if avg_loss == 0.0 {
            return 100.0;
        }

        let rs = avg_gain / avg_loss;
        100.0 - (100.0 / (1.0 + rs))
    }

    fn compute_support_resistance(
        &self,
        closes: &[f64],
        highs: &[f64],
        lows: &[f64],
    ) -> (f64, f64) {
        let n = closes.len();
        let window_start = n.saturating_sub(self.trend_window);

        let recent_high = highs[window_start..]
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let recent_low = lows[window_start..]
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);

        let current_price = closes[n - 1];
        let price_range = recent_high - recent_low;

        if price_range == 0.0 {
            return (0.5, 0.5);
        }

        let support_dist = (current_price - recent_low) / price_range;
        let resist_dist = (recent_high - current_price) / price_range;

        (support_dist, resist_dist)
    }
}

impl Default for ConceptExtractor {
    fn default() -> Self {
        Self::new()
    }
}

/// Calculate standard deviation.
fn std_dev(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let mean: f64 = data.iter().sum::<f64>() / data.len() as f64;
    let variance: f64 = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
    variance.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::bybit::generate_sample_klines;

    #[test]
    fn test_concept_extraction() {
        let klines = generate_sample_klines(100);
        let extractor = ConceptExtractor::new();
        let concepts = extractor.extract(&klines);

        assert!(concepts.is_some());
        let c = concepts.unwrap();
        assert!(c.trend_strength >= 0.0 && c.trend_strength <= 1.0);
        assert!(c.rsi >= 0.0 && c.rsi <= 100.0);
    }

    #[test]
    fn test_concepts_to_vector() {
        let concepts = TradingConcepts {
            trend_direction: TrendDirection::Bullish,
            trend_strength: 0.5,
            volatility_regime: VolatilityRegime::Normal,
            volatility_value: 1.0,
            momentum_signal: MomentumSignal::Positive,
            momentum_value: 0.03,
            volume_profile: VolumeProfile::Average,
            volume_ratio: 1.0,
            rsi: 55.0,
            support_distance: 0.6,
            resistance_distance: 0.4,
        };

        let vector = concepts.to_vector();
        assert_eq!(vector.len(), 11);
        assert_eq!(vector[0], 1.0); // Bullish
    }
}
