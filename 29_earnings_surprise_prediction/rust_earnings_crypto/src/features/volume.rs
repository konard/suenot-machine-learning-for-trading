//! Volume analysis tools

use crate::data::types::Candle;

/// Volume analyzer for detecting anomalies and patterns
pub struct VolumeAnalyzer {
    /// Lookback period for calculations
    lookback: usize,
}

impl Default for VolumeAnalyzer {
    fn default() -> Self {
        Self { lookback: 20 }
    }
}

impl VolumeAnalyzer {
    /// Create a new volume analyzer
    pub fn new(lookback: usize) -> Self {
        Self { lookback }
    }

    /// Calculate volume moving average
    pub fn volume_ma(&self, candles: &[Candle]) -> Vec<f64> {
        let volumes: Vec<f64> = candles.iter().map(|c| c.volume).collect();
        let n = volumes.len();
        let mut result = vec![f64::NAN; n];

        for i in (self.lookback - 1)..n {
            let sum: f64 = volumes[(i + 1 - self.lookback)..=i].iter().sum();
            result[i] = sum / self.lookback as f64;
        }

        result
    }

    /// Calculate relative volume (current / average)
    pub fn relative_volume(&self, candles: &[Candle]) -> Vec<f64> {
        let ma = self.volume_ma(candles);
        let n = candles.len();
        let mut result = vec![f64::NAN; n];

        for i in 0..n {
            if !ma[i].is_nan() && ma[i] > 0.0 {
                result[i] = candles[i].volume / ma[i];
            }
        }

        result
    }

    /// Calculate volume z-score
    pub fn volume_zscore(&self, candles: &[Candle]) -> Vec<f64> {
        let volumes: Vec<f64> = candles.iter().map(|c| c.volume).collect();
        let n = volumes.len();
        let mut result = vec![f64::NAN; n];

        for i in (self.lookback - 1)..n {
            let slice = &volumes[(i + 1 - self.lookback)..=i];
            let mean: f64 = slice.iter().sum::<f64>() / self.lookback as f64;
            let variance: f64 = slice.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                / (self.lookback - 1) as f64;
            let std = variance.sqrt();

            if std > 1e-10 {
                result[i] = (volumes[i] - mean) / std;
            }
        }

        result
    }

    /// Calculate On-Balance Volume (OBV)
    pub fn obv(&self, candles: &[Candle]) -> Vec<f64> {
        if candles.is_empty() {
            return vec![];
        }

        let mut result = vec![0.0];

        for i in 1..candles.len() {
            let prev_obv = result[i - 1];
            let volume = candles[i].volume;

            if candles[i].close > candles[i - 1].close {
                result.push(prev_obv + volume);
            } else if candles[i].close < candles[i - 1].close {
                result.push(prev_obv - volume);
            } else {
                result.push(prev_obv);
            }
        }

        result
    }

    /// Calculate Volume-Weighted Average Price (VWAP)
    pub fn vwap(&self, candles: &[Candle]) -> Vec<f64> {
        let n = candles.len();
        let mut result = vec![f64::NAN; n];

        for i in (self.lookback - 1)..n {
            let slice = &candles[(i + 1 - self.lookback)..=i];
            let total_volume: f64 = slice.iter().map(|c| c.volume).sum();
            let total_turnover: f64 = slice.iter().map(|c| c.turnover).sum();

            if total_volume > 0.0 {
                result[i] = total_turnover / total_volume;
            }
        }

        result
    }

    /// Calculate Money Flow Index (MFI)
    pub fn mfi(&self, candles: &[Candle], period: usize) -> Vec<f64> {
        if candles.len() < period + 1 {
            return vec![f64::NAN; candles.len()];
        }

        let n = candles.len();
        let mut result = vec![f64::NAN; n];

        // Calculate typical price and money flow
        let mut typical_prices = Vec::with_capacity(n);
        let mut money_flows = Vec::with_capacity(n);

        for c in candles {
            let tp = (c.high + c.low + c.close) / 3.0;
            typical_prices.push(tp);
            money_flows.push(tp * c.volume);
        }

        // Calculate positive and negative money flow
        for i in period..n {
            let mut positive_flow = 0.0;
            let mut negative_flow = 0.0;

            for j in (i - period + 1)..=i {
                if typical_prices[j] > typical_prices[j - 1] {
                    positive_flow += money_flows[j];
                } else if typical_prices[j] < typical_prices[j - 1] {
                    negative_flow += money_flows[j];
                }
            }

            if negative_flow > 0.0 {
                let money_ratio = positive_flow / negative_flow;
                result[i] = 100.0 - (100.0 / (1.0 + money_ratio));
            } else if positive_flow > 0.0 {
                result[i] = 100.0;
            }
        }

        result
    }

    /// Detect volume anomalies (> threshold std deviations)
    pub fn detect_anomalies(&self, candles: &[Candle], threshold: f64) -> Vec<bool> {
        let zscores = self.volume_zscore(candles);
        zscores
            .iter()
            .map(|z| z.is_finite() && z.abs() > threshold)
            .collect()
    }

    /// Calculate volume profile (volume at price levels)
    pub fn volume_profile(&self, candles: &[Candle], num_bins: usize) -> Vec<(f64, f64)> {
        if candles.is_empty() || num_bins == 0 {
            return vec![];
        }

        let min_price = candles
            .iter()
            .map(|c| c.low)
            .fold(f64::INFINITY, f64::min);
        let max_price = candles
            .iter()
            .map(|c| c.high)
            .fold(f64::NEG_INFINITY, f64::max);

        if (max_price - min_price).abs() < 1e-10 {
            return vec![(min_price, candles.iter().map(|c| c.volume).sum())];
        }

        let bin_size = (max_price - min_price) / num_bins as f64;
        let mut bins = vec![0.0; num_bins];

        for candle in candles {
            // Simple approximation: assign volume to typical price bin
            let typical_price = (candle.high + candle.low + candle.close) / 3.0;
            let bin_idx = ((typical_price - min_price) / bin_size).floor() as usize;
            let bin_idx = bin_idx.min(num_bins - 1);
            bins[bin_idx] += candle.volume;
        }

        bins.iter()
            .enumerate()
            .map(|(i, &vol)| (min_price + (i as f64 + 0.5) * bin_size, vol))
            .collect()
    }

    /// Find Point of Control (price level with highest volume)
    pub fn point_of_control(&self, candles: &[Candle], num_bins: usize) -> Option<f64> {
        let profile = self.volume_profile(candles, num_bins);
        profile
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|&(price, _)| price)
    }
}

/// Volume statistics for a period
#[derive(Debug, Clone)]
pub struct VolumeStats {
    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub max: f64,
    pub median: f64,
    pub total: f64,
}

impl VolumeStats {
    /// Calculate volume statistics
    pub fn calculate(volumes: &[f64]) -> Self {
        if volumes.is_empty() {
            return Self {
                mean: 0.0,
                std: 0.0,
                min: 0.0,
                max: 0.0,
                median: 0.0,
                total: 0.0,
            };
        }

        let total: f64 = volumes.iter().sum();
        let n = volumes.len() as f64;
        let mean = total / n;

        let variance: f64 = volumes.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
        let std = variance.sqrt();

        let min = volumes.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = volumes.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        let mut sorted = volumes.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = if sorted.len() % 2 == 0 {
            (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
        } else {
            sorted[sorted.len() / 2]
        };

        Self {
            mean,
            std,
            min,
            max,
            median,
            total,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_candles() -> Vec<Candle> {
        vec![
            Candle { timestamp: 0, open: 100.0, high: 105.0, low: 98.0, close: 102.0, volume: 1000.0, turnover: 100000.0 },
            Candle { timestamp: 1000, open: 102.0, high: 108.0, low: 101.0, close: 107.0, volume: 1500.0, turnover: 157500.0 },
            Candle { timestamp: 2000, open: 107.0, high: 110.0, low: 104.0, close: 105.0, volume: 1200.0, turnover: 126000.0 },
            Candle { timestamp: 3000, open: 105.0, high: 106.0, low: 100.0, close: 101.0, volume: 5000.0, turnover: 505000.0 },
            Candle { timestamp: 4000, open: 101.0, high: 103.0, low: 99.0, close: 102.0, volume: 900.0, turnover: 91800.0 },
        ]
    }

    #[test]
    fn test_relative_volume() {
        let analyzer = VolumeAnalyzer::new(3);
        let candles = make_candles();
        let rel_vol = analyzer.relative_volume(&candles);

        // First two values should be NaN (not enough lookback)
        assert!(rel_vol[0].is_nan());
        assert!(rel_vol[1].is_nan());
        // Third value and beyond should be valid
        assert!(!rel_vol[2].is_nan());
    }

    #[test]
    fn test_obv() {
        let analyzer = VolumeAnalyzer::new(3);
        let candles = make_candles();
        let obv = analyzer.obv(&candles);

        assert_eq!(obv.len(), candles.len());
        assert_eq!(obv[0], 0.0);
    }

    #[test]
    fn test_volume_stats() {
        let volumes = vec![100.0, 200.0, 300.0, 400.0, 500.0];
        let stats = VolumeStats::calculate(&volumes);

        assert_eq!(stats.mean, 300.0);
        assert_eq!(stats.min, 100.0);
        assert_eq!(stats.max, 500.0);
        assert_eq!(stats.median, 300.0);
        assert_eq!(stats.total, 1500.0);
    }
}
