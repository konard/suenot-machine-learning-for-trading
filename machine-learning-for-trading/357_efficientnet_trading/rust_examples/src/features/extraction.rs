//! Feature extraction from candle data

use crate::data::Candle;

/// Feature set extracted from candle data
#[derive(Debug, Clone)]
pub struct FeatureSet {
    pub returns: Vec<f64>,
    pub volatility: f64,
    pub momentum: f64,
    pub rsi: f64,
    pub macd: f64,
    pub macd_signal: f64,
    pub bollinger_position: f64,
    pub volume_ratio: f64,
    pub trend_strength: f64,
}

/// Feature extractor
pub struct FeatureExtractor {
    rsi_period: usize,
    macd_fast: usize,
    macd_slow: usize,
    macd_signal: usize,
    bb_period: usize,
}

impl FeatureExtractor {
    pub fn new() -> Self {
        Self {
            rsi_period: 14,
            macd_fast: 12,
            macd_slow: 26,
            macd_signal: 9,
            bb_period: 20,
        }
    }

    /// Extract features from candles
    pub fn extract(&self, candles: &[Candle]) -> Option<FeatureSet> {
        if candles.len() < self.macd_slow + self.macd_signal {
            return None;
        }

        let closes: Vec<f64> = candles.iter().map(|c| c.close).collect();
        let volumes: Vec<f64> = candles.iter().map(|c| c.volume).collect();

        // Calculate returns
        let returns: Vec<f64> = closes
            .windows(2)
            .map(|w| (w[1] / w[0]).ln())
            .collect();

        // Calculate volatility (std of returns)
        let volatility = self.std(&returns);

        // Calculate momentum (simple return over period)
        let momentum = if closes.len() >= 10 {
            (closes.last()? / closes[closes.len() - 10] - 1.0) * 100.0
        } else {
            0.0
        };

        // Calculate RSI
        let rsi = self.calculate_rsi(&closes);

        // Calculate MACD
        let (macd, macd_signal) = self.calculate_macd(&closes);

        // Calculate Bollinger Band position
        let bollinger_position = self.calculate_bollinger_position(&closes);

        // Calculate volume ratio
        let volume_ratio = self.calculate_volume_ratio(&volumes);

        // Calculate trend strength (ADX-like)
        let trend_strength = self.calculate_trend_strength(candles);

        Some(FeatureSet {
            returns,
            volatility,
            momentum,
            rsi,
            macd,
            macd_signal,
            bollinger_position,
            volume_ratio,
            trend_strength,
        })
    }

    fn calculate_rsi(&self, closes: &[f64]) -> f64 {
        if closes.len() < self.rsi_period + 1 {
            return 50.0;
        }

        let changes: Vec<f64> = closes.windows(2).map(|w| w[1] - w[0]).collect();
        let recent_changes = &changes[changes.len() - self.rsi_period..];

        let gains: f64 = recent_changes.iter().filter(|&&c| c > 0.0).sum();
        let losses: f64 = recent_changes.iter().filter(|&&c| c < 0.0).map(|c| c.abs()).sum();

        if losses == 0.0 {
            return 100.0;
        }

        let avg_gain = gains / self.rsi_period as f64;
        let avg_loss = losses / self.rsi_period as f64;

        if avg_loss == 0.0 {
            return 100.0;
        }

        let rs = avg_gain / avg_loss;
        100.0 - (100.0 / (1.0 + rs))
    }

    fn calculate_macd(&self, closes: &[f64]) -> (f64, f64) {
        if closes.len() < self.macd_slow + self.macd_signal {
            return (0.0, 0.0);
        }

        let ema_fast = self.ema(closes, self.macd_fast);
        let ema_slow = self.ema(closes, self.macd_slow);

        let macd_line: Vec<f64> = ema_fast
            .iter()
            .zip(ema_slow.iter())
            .map(|(f, s)| f - s)
            .collect();

        let signal_line = self.ema(&macd_line, self.macd_signal);

        let macd = *macd_line.last().unwrap_or(&0.0);
        let signal = *signal_line.last().unwrap_or(&0.0);

        (macd, signal)
    }

    fn calculate_bollinger_position(&self, closes: &[f64]) -> f64 {
        if closes.len() < self.bb_period {
            return 0.0;
        }

        let recent = &closes[closes.len() - self.bb_period..];
        let mean = self.mean(recent);
        let std = self.std(recent);

        if std == 0.0 {
            return 0.0;
        }

        let current = *closes.last().unwrap_or(&mean);
        (current - mean) / (2.0 * std)
    }

    fn calculate_volume_ratio(&self, volumes: &[f64]) -> f64 {
        if volumes.len() < 20 {
            return 1.0;
        }

        let avg_volume = self.mean(&volumes[volumes.len() - 20..]);
        let current_volume = *volumes.last().unwrap_or(&avg_volume);

        if avg_volume == 0.0 {
            return 1.0;
        }

        current_volume / avg_volume
    }

    fn calculate_trend_strength(&self, candles: &[Candle]) -> f64 {
        if candles.len() < 14 {
            return 0.0;
        }

        let recent = &candles[candles.len() - 14..];

        let mut plus_dm_sum = 0.0;
        let mut minus_dm_sum = 0.0;
        let mut tr_sum = 0.0;

        for window in recent.windows(2) {
            let prev = &window[0];
            let curr = &window[1];

            let high_diff = curr.high - prev.high;
            let low_diff = prev.low - curr.low;

            let plus_dm = if high_diff > low_diff && high_diff > 0.0 {
                high_diff
            } else {
                0.0
            };
            let minus_dm = if low_diff > high_diff && low_diff > 0.0 {
                low_diff
            } else {
                0.0
            };

            let tr = (curr.high - curr.low)
                .max((curr.high - prev.close).abs())
                .max((curr.low - prev.close).abs());

            plus_dm_sum += plus_dm;
            minus_dm_sum += minus_dm;
            tr_sum += tr;
        }

        if tr_sum == 0.0 {
            return 0.0;
        }

        let plus_di = plus_dm_sum / tr_sum * 100.0;
        let minus_di = minus_dm_sum / tr_sum * 100.0;
        let di_sum = plus_di + minus_di;

        if di_sum == 0.0 {
            return 0.0;
        }

        ((plus_di - minus_di).abs() / di_sum) * 100.0
    }

    fn ema(&self, data: &[f64], period: usize) -> Vec<f64> {
        if data.is_empty() || period == 0 {
            return Vec::new();
        }

        let multiplier = 2.0 / (period as f64 + 1.0);
        let mut ema = Vec::with_capacity(data.len());

        // Start with SMA
        if data.len() >= period {
            let sma: f64 = data[..period].iter().sum::<f64>() / period as f64;
            ema.push(sma);

            for &value in &data[period..] {
                let prev = *ema.last().unwrap();
                ema.push((value - prev) * multiplier + prev);
            }
        }

        ema
    }

    fn mean(&self, data: &[f64]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }
        data.iter().sum::<f64>() / data.len() as f64
    }

    fn std(&self, data: &[f64]) -> f64 {
        if data.len() < 2 {
            return 0.0;
        }

        let mean = self.mean(data);
        let variance: f64 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (data.len() - 1) as f64;
        variance.sqrt()
    }
}

impl Default for FeatureExtractor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_candles(n: usize) -> Vec<Candle> {
        (0..n)
            .map(|i| {
                let base = 100.0 + (i as f64 * 0.1).sin() * 5.0;
                Candle::new(
                    i as u64 * 60000,
                    base,
                    base + 2.0,
                    base - 1.0,
                    base + 1.0,
                    1000.0 + i as f64 * 10.0,
                )
            })
            .collect()
    }

    #[test]
    fn test_feature_extraction() {
        let extractor = FeatureExtractor::new();
        let candles = sample_candles(50);
        let features = extractor.extract(&candles);

        assert!(features.is_some());
        let f = features.unwrap();

        assert!(f.rsi >= 0.0 && f.rsi <= 100.0);
        assert!(f.volatility >= 0.0);
    }

    #[test]
    fn test_insufficient_data() {
        let extractor = FeatureExtractor::new();
        let candles = sample_candles(10);
        let features = extractor.extract(&candles);

        assert!(features.is_none());
    }
}
