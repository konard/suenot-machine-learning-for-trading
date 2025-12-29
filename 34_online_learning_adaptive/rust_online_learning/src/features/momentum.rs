//! Momentum Features
//!
//! Calculates various momentum indicators from price data.

use crate::api::Candle;

/// Momentum feature generator
///
/// Computes momentum signals over multiple lookback periods.
#[derive(Debug, Clone)]
pub struct MomentumFeatures {
    /// Lookback periods for momentum calculation
    periods: Vec<usize>,
}

impl MomentumFeatures {
    /// Create a new momentum feature generator
    ///
    /// # Arguments
    ///
    /// * `periods` - Lookback periods (e.g., [12, 24, 48, 96] for hourly data)
    pub fn new(periods: Vec<usize>) -> Self {
        Self { periods }
    }

    /// Create with default periods (12h, 24h, 48h, 96h for hourly data)
    pub fn default_hourly() -> Self {
        Self::new(vec![12, 24, 48, 96])
    }

    /// Create with daily periods (1d, 5d, 21d, 63d for daily data)
    pub fn default_daily() -> Self {
        Self::new(vec![1, 5, 21, 63])
    }

    /// Compute momentum features from candles
    ///
    /// Returns None if not enough data for the longest lookback period.
    pub fn compute(&self, candles: &[Candle]) -> Option<Vec<f64>> {
        let max_period = *self.periods.iter().max()?;

        if candles.len() <= max_period {
            return None;
        }

        let current_price = candles.last()?.close;

        let features: Vec<f64> = self
            .periods
            .iter()
            .map(|&period| {
                let past_idx = candles.len() - 1 - period;
                let past_price = candles[past_idx].close;

                // Momentum as percentage return
                (current_price - past_price) / past_price
            })
            .collect();

        Some(features)
    }

    /// Compute momentum with volume weighting
    pub fn compute_volume_weighted(&self, candles: &[Candle]) -> Option<Vec<f64>> {
        let max_period = *self.periods.iter().max()?;

        if candles.len() <= max_period {
            return None;
        }

        let features: Vec<f64> = self
            .periods
            .iter()
            .map(|&period| {
                let start_idx = candles.len() - 1 - period;

                // Volume-weighted return
                let mut weighted_sum = 0.0;
                let mut volume_sum = 0.0;

                for i in start_idx..candles.len() - 1 {
                    let ret = (candles[i + 1].close - candles[i].close) / candles[i].close;
                    let vol = candles[i + 1].volume;
                    weighted_sum += ret * vol;
                    volume_sum += vol;
                }

                if volume_sum > 0.0 {
                    weighted_sum / volume_sum
                } else {
                    0.0
                }
            })
            .collect();

        Some(features)
    }

    /// Get feature names
    pub fn feature_names(&self) -> Vec<String> {
        self.periods
            .iter()
            .map(|p| format!("mom_{}", p))
            .collect()
    }

    /// Compute additional features (volatility, volume momentum)
    pub fn compute_extended(&self, candles: &[Candle]) -> Option<ExtendedFeatures> {
        let max_period = *self.periods.iter().max()?;

        if candles.len() <= max_period {
            return None;
        }

        // Price momentum
        let price_momentum = self.compute(candles)?;

        // Volume momentum
        let volume_momentum = self.compute_volume_momentum(candles, 24)?;

        // Volatility (realized)
        let volatility = self.compute_volatility(candles, 24)?;

        // RSI
        let rsi = self.compute_rsi(candles, 14)?;

        // Price position in range
        let price_position = self.compute_price_position(candles, 24)?;

        Some(ExtendedFeatures {
            price_momentum,
            volume_momentum,
            volatility,
            rsi,
            price_position,
        })
    }

    /// Compute volume momentum
    fn compute_volume_momentum(&self, candles: &[Candle], period: usize) -> Option<f64> {
        if candles.len() <= period * 2 {
            return None;
        }

        let n = candles.len();

        // Recent average volume
        let recent_vol: f64 = candles[n - period..].iter().map(|c| c.volume).sum::<f64>() / period as f64;

        // Past average volume
        let past_vol: f64 = candles[n - 2 * period..n - period]
            .iter()
            .map(|c| c.volume)
            .sum::<f64>()
            / period as f64;

        if past_vol > 0.0 {
            Some((recent_vol - past_vol) / past_vol)
        } else {
            Some(0.0)
        }
    }

    /// Compute realized volatility
    fn compute_volatility(&self, candles: &[Candle], period: usize) -> Option<f64> {
        if candles.len() <= period {
            return None;
        }

        let n = candles.len();
        let returns: Vec<f64> = candles[n - period..]
            .windows(2)
            .map(|w| (w[1].close - w[0].close) / w[0].close)
            .collect();

        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;

        Some(variance.sqrt())
    }

    /// Compute RSI (Relative Strength Index)
    fn compute_rsi(&self, candles: &[Candle], period: usize) -> Option<f64> {
        if candles.len() <= period {
            return None;
        }

        let n = candles.len();
        let mut gains = 0.0;
        let mut losses = 0.0;

        for w in candles[n - period..].windows(2) {
            let change = w[1].close - w[0].close;
            if change > 0.0 {
                gains += change;
            } else {
                losses -= change;
            }
        }

        let avg_gain = gains / period as f64;
        let avg_loss = losses / period as f64;

        if avg_loss == 0.0 {
            return Some(100.0);
        }

        let rs = avg_gain / avg_loss;
        Some(100.0 - (100.0 / (1.0 + rs)))
    }

    /// Compute price position in recent range (0 = low, 1 = high)
    fn compute_price_position(&self, candles: &[Candle], period: usize) -> Option<f64> {
        if candles.len() <= period {
            return None;
        }

        let n = candles.len();
        let recent = &candles[n - period..];

        let high = recent.iter().map(|c| c.high).fold(f64::NEG_INFINITY, f64::max);
        let low = recent.iter().map(|c| c.low).fold(f64::INFINITY, f64::min);
        let current = candles.last()?.close;

        if high > low {
            Some((current - low) / (high - low))
        } else {
            Some(0.5)
        }
    }
}

/// Extended features including momentum, volatility, and other indicators
#[derive(Debug, Clone)]
pub struct ExtendedFeatures {
    /// Price momentum over multiple periods
    pub price_momentum: Vec<f64>,
    /// Volume momentum
    pub volume_momentum: f64,
    /// Realized volatility
    pub volatility: f64,
    /// RSI indicator
    pub rsi: f64,
    /// Price position in recent range (0-1)
    pub price_position: f64,
}

impl ExtendedFeatures {
    /// Convert to flat feature vector
    pub fn to_vec(&self) -> Vec<f64> {
        let mut features = self.price_momentum.clone();
        features.push(self.volume_momentum);
        features.push(self.volatility);
        features.push((self.rsi - 50.0) / 50.0); // Normalize to -1..1
        features.push(self.price_position * 2.0 - 1.0); // Normalize to -1..1
        features
    }

    /// Get feature names
    pub fn feature_names(periods: &[usize]) -> Vec<String> {
        let mut names: Vec<String> = periods.iter().map(|p| format!("mom_{}", p)).collect();
        names.push("vol_mom".to_string());
        names.push("volatility".to_string());
        names.push("rsi".to_string());
        names.push("price_pos".to_string());
        names
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_candles(n: usize) -> Vec<Candle> {
        (0..n)
            .map(|i| Candle {
                timestamp: (i * 3600000) as u64,
                open: 100.0 + i as f64 * 0.1,
                high: 101.0 + i as f64 * 0.1,
                low: 99.0 + i as f64 * 0.1,
                close: 100.5 + i as f64 * 0.1,
                volume: 1000.0 + (i % 10) as f64 * 100.0,
                turnover: 100000.0,
            })
            .collect()
    }

    #[test]
    fn test_momentum_features() {
        let features = MomentumFeatures::new(vec![5, 10, 20]);
        let candles = create_test_candles(50);

        let result = features.compute(&candles);
        assert!(result.is_some());

        let mom = result.unwrap();
        assert_eq!(mom.len(), 3);

        // Momentum should be positive (prices trending up)
        assert!(mom.iter().all(|&m| m > 0.0));
    }

    #[test]
    fn test_insufficient_data() {
        let features = MomentumFeatures::new(vec![50, 100]);
        let candles = create_test_candles(20);

        let result = features.compute(&candles);
        assert!(result.is_none());
    }

    #[test]
    fn test_extended_features() {
        let features = MomentumFeatures::new(vec![5, 10]);
        let candles = create_test_candles(50);

        let extended = features.compute_extended(&candles);
        assert!(extended.is_some());

        let ext = extended.unwrap();
        assert_eq!(ext.price_momentum.len(), 2);
        assert!(ext.rsi >= 0.0 && ext.rsi <= 100.0);
        assert!(ext.price_position >= 0.0 && ext.price_position <= 1.0);
    }

    #[test]
    fn test_feature_names() {
        let features = MomentumFeatures::new(vec![12, 24, 48]);
        let names = features.feature_names();

        assert_eq!(names, vec!["mom_12", "mom_24", "mom_48"]);
    }
}
