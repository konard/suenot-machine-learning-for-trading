//! Market regime detection and prediction.

use crate::data::features::FeatureGenerator;
use crate::model::network::ZeroShotModel;
use crate::{MarketRegime, Result};

/// Regime detector using statistical methods.
#[derive(Debug, Clone)]
pub struct RegimeDetector {
    /// Lookback period for trend detection
    pub trend_lookback: usize,
    /// Threshold for strong trend classification
    pub strong_threshold: f64,
    /// Threshold for weak trend classification
    pub weak_threshold: f64,
}

impl Default for RegimeDetector {
    fn default() -> Self {
        Self {
            trend_lookback: 20,
            strong_threshold: 0.02, // 2% return threshold
            weak_threshold: 0.005,  // 0.5% return threshold
        }
    }
}

impl RegimeDetector {
    /// Detect market regime from price data.
    pub fn detect(&self, prices: &[f64]) -> MarketRegime {
        if prices.len() < self.trend_lookback {
            return MarketRegime::Sideways;
        }

        // Calculate cumulative return over lookback period
        let recent_prices = &prices[prices.len() - self.trend_lookback..];
        let start_price = recent_prices[0];
        let end_price = recent_prices[recent_prices.len() - 1];

        if start_price <= 0.0 {
            return MarketRegime::Sideways;
        }

        let cumulative_return = (end_price - start_price) / start_price;

        // Calculate volatility
        let returns: Vec<f64> = recent_prices
            .windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect();

        let volatility = std_dev(&returns);

        // Adjust thresholds based on volatility
        let vol_adjusted_strong = self.strong_threshold * (1.0 + volatility * 10.0);
        let vol_adjusted_weak = self.weak_threshold * (1.0 + volatility * 10.0);

        // Classify regime
        if cumulative_return > vol_adjusted_strong {
            MarketRegime::StrongUptrend
        } else if cumulative_return > vol_adjusted_weak {
            MarketRegime::WeakUptrend
        } else if cumulative_return < -vol_adjusted_strong {
            MarketRegime::StrongDowntrend
        } else if cumulative_return < -vol_adjusted_weak {
            MarketRegime::WeakDowntrend
        } else {
            MarketRegime::Sideways
        }
    }

    /// Detect regime with confidence score.
    pub fn detect_with_confidence(&self, prices: &[f64]) -> (MarketRegime, f64) {
        if prices.len() < self.trend_lookback {
            return (MarketRegime::Sideways, 0.0);
        }

        let regime = self.detect(prices);

        // Calculate confidence based on consistency of trend
        let recent_prices = &prices[prices.len() - self.trend_lookback..];
        let returns: Vec<f64> = recent_prices
            .windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect();

        // Count consistent directional moves
        let (positive_count, negative_count) = returns.iter().fold((0, 0), |(p, n), &r| {
            if r > 0.0 {
                (p + 1, n)
            } else if r < 0.0 {
                (p, n + 1)
            } else {
                (p, n)
            }
        });

        let total = returns.len() as f64;
        let confidence = match regime {
            MarketRegime::StrongUptrend | MarketRegime::WeakUptrend => {
                positive_count as f64 / total
            }
            MarketRegime::StrongDowntrend | MarketRegime::WeakDowntrend => {
                negative_count as f64 / total
            }
            MarketRegime::Sideways => {
                // For sideways, confidence is based on balance
                1.0 - (positive_count as f64 - negative_count as f64).abs() / total
            }
        };

        (regime, confidence.clamp(0.0, 1.0))
    }
}

/// Regime predictor using the zero-shot model.
pub struct ZeroShotRegimePredictor {
    /// The underlying model
    model: ZeroShotModel,
    /// Feature generator
    feature_generator: FeatureGenerator,
    /// Fallback statistical detector
    fallback_detector: RegimeDetector,
}

impl ZeroShotRegimePredictor {
    /// Create a new predictor.
    pub fn new(model: ZeroShotModel) -> Self {
        Self {
            model,
            feature_generator: FeatureGenerator::default(),
            fallback_detector: RegimeDetector::default(),
        }
    }

    /// Predict regime from kline data.
    pub fn predict(&self, klines: &[crate::data::bybit::Kline]) -> Result<MarketRegime> {
        let features = self.feature_generator.generate(klines)?;
        self.model.predict_regime(&features)
    }

    /// Predict with confidence score.
    pub fn predict_with_confidence(
        &self,
        klines: &[crate::data::bybit::Kline],
    ) -> Result<(MarketRegime, f64)> {
        let features = self.feature_generator.generate(klines)?;
        let (regime, scores) = self.model.predict_regime_with_scores(&features)?;

        // Confidence is the max score (softmax probability)
        let max_score = scores
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let sum_exp: f64 = scores.iter().map(|s| s.exp()).sum();
        let confidence = max_score.exp() / sum_exp;

        Ok((regime, confidence))
    }

    /// Get statistical fallback prediction from prices.
    pub fn predict_statistical(&self, prices: &[f64]) -> (MarketRegime, f64) {
        self.fallback_detector.detect_with_confidence(prices)
    }
}

/// Calculate standard deviation.
fn std_dev(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
    let variance: f64 = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
    variance.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_regime_detection() {
        let detector = RegimeDetector::default();

        // Strong uptrend
        let prices: Vec<f64> = (0..30).map(|i| 100.0 * (1.002_f64).powi(i)).collect();
        let regime = detector.detect(&prices);
        assert!(matches!(
            regime,
            MarketRegime::StrongUptrend | MarketRegime::WeakUptrend
        ));

        // Strong downtrend
        let prices: Vec<f64> = (0..30).map(|i| 100.0 * (0.998_f64).powi(i)).collect();
        let regime = detector.detect(&prices);
        assert!(matches!(
            regime,
            MarketRegime::StrongDowntrend | MarketRegime::WeakDowntrend
        ));

        // Sideways
        let prices: Vec<f64> = (0..30)
            .map(|i| 100.0 + (i as f64 * 0.1).sin() * 0.5)
            .collect();
        let regime = detector.detect(&prices);
        assert!(matches!(regime, MarketRegime::Sideways));
    }

    #[test]
    fn test_regime_with_confidence() {
        let detector = RegimeDetector::default();

        // Consistent uptrend should have high confidence
        let prices: Vec<f64> = (0..30).map(|i| 100.0 + i as f64 * 0.5).collect();
        let (regime, confidence) = detector.detect_with_confidence(&prices);
        assert!(matches!(
            regime,
            MarketRegime::StrongUptrend | MarketRegime::WeakUptrend
        ));
        assert!(confidence > 0.5);
    }
}
