//! Surprise calculation module
//!
//! Calculates "earnings-like" surprise metrics for crypto:
//! - Price surprise (deviation from expected return)
//! - Volume surprise (deviation from expected volume)
//! - Composite surprise scores

use crate::data::types::Candle;
use serde::{Deserialize, Serialize};

/// Surprise metrics for a single candle
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurpriseMetrics {
    /// Timestamp
    pub timestamp: u64,
    /// Actual return
    pub actual_return: f64,
    /// Expected return (based on history)
    pub expected_return: f64,
    /// Price surprise (z-score)
    pub price_surprise: f64,
    /// Actual volume
    pub actual_volume: f64,
    /// Expected volume
    pub expected_volume: f64,
    /// Volume surprise (z-score)
    pub volume_surprise: f64,
    /// Composite surprise score
    pub composite_score: f64,
}

impl SurpriseMetrics {
    /// Check if this is a significant surprise
    pub fn is_significant(&self, threshold: f64) -> bool {
        self.price_surprise.abs() > threshold || self.volume_surprise.abs() > threshold
    }

    /// Get direction of surprise
    pub fn direction(&self) -> SurpriseDirection {
        if self.composite_score > 1.0 {
            SurpriseDirection::PositiveSurprise
        } else if self.composite_score < -1.0 {
            SurpriseDirection::NegativeSurprise
        } else {
            SurpriseDirection::AsExpected
        }
    }
}

/// Direction of surprise
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SurpriseDirection {
    PositiveSurprise,
    NegativeSurprise,
    AsExpected,
}

/// Calculator for surprise metrics
pub struct SurpriseCalculator {
    /// Lookback period for calculating expectations
    lookback: usize,
    /// Weight for volume in composite score
    volume_weight: f64,
}

impl Default for SurpriseCalculator {
    fn default() -> Self {
        Self {
            lookback: 20,
            volume_weight: 0.3,
        }
    }
}

impl SurpriseCalculator {
    /// Create a new surprise calculator
    pub fn new(lookback: usize) -> Self {
        Self {
            lookback,
            ..Default::default()
        }
    }

    /// Set volume weight for composite score
    pub fn with_volume_weight(mut self, weight: f64) -> Self {
        self.volume_weight = weight.clamp(0.0, 1.0);
        self
    }

    /// Calculate surprise metrics for all candles
    pub fn calculate(&self, candles: &[Candle]) -> Vec<SurpriseMetrics> {
        if candles.len() < self.lookback + 1 {
            return vec![];
        }

        let n = candles.len();

        // Calculate returns
        let returns: Vec<f64> = candles.iter().map(|c| c.return_pct()).collect();
        let volumes: Vec<f64> = candles.iter().map(|c| c.volume).collect();

        let mut results = Vec::with_capacity(n - self.lookback);

        for i in self.lookback..n {
            let return_slice = &returns[(i - self.lookback)..i];
            let volume_slice = &volumes[(i - self.lookback)..i];

            // Calculate return expectations
            let return_mean = mean(return_slice);
            let return_std = std(return_slice);

            // Calculate volume expectations
            let volume_mean = mean(volume_slice);
            let volume_std = std(volume_slice);

            // Calculate surprises
            let price_surprise = if return_std > 1e-10 {
                (returns[i] - return_mean) / return_std
            } else {
                0.0
            };

            let volume_surprise = if volume_std > 1e-10 {
                (volumes[i] - volume_mean) / volume_std
            } else {
                0.0
            };

            // Composite score (weighted combination)
            let composite = (1.0 - self.volume_weight) * price_surprise
                + self.volume_weight * volume_surprise * price_surprise.signum();

            results.push(SurpriseMetrics {
                timestamp: candles[i].timestamp,
                actual_return: returns[i],
                expected_return: return_mean,
                price_surprise,
                actual_volume: volumes[i],
                expected_volume: volume_mean,
                volume_surprise,
                composite_score: composite,
            });
        }

        results
    }

    /// Calculate rolling surprise statistics
    pub fn rolling_stats(&self, surprises: &[SurpriseMetrics]) -> SurpriseStats {
        if surprises.is_empty() {
            return SurpriseStats::default();
        }

        let price_surprises: Vec<f64> = surprises.iter().map(|s| s.price_surprise).collect();
        let vol_surprises: Vec<f64> = surprises.iter().map(|s| s.volume_surprise).collect();
        let composites: Vec<f64> = surprises.iter().map(|s| s.composite_score).collect();

        let positive_count = surprises
            .iter()
            .filter(|s| s.direction() == SurpriseDirection::PositiveSurprise)
            .count();
        let negative_count = surprises
            .iter()
            .filter(|s| s.direction() == SurpriseDirection::NegativeSurprise)
            .count();

        SurpriseStats {
            count: surprises.len(),
            positive_surprises: positive_count,
            negative_surprises: negative_count,
            avg_price_surprise: mean(&price_surprises),
            std_price_surprise: std(&price_surprises),
            avg_volume_surprise: mean(&vol_surprises),
            std_volume_surprise: std(&vol_surprises),
            avg_composite: mean(&composites),
            max_composite: composites.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)),
            min_composite: composites.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
        }
    }

    /// Find extreme surprises (above threshold)
    pub fn find_extremes(&self, surprises: &[SurpriseMetrics], threshold: f64) -> Vec<&SurpriseMetrics> {
        surprises
            .iter()
            .filter(|s| s.composite_score.abs() > threshold)
            .collect()
    }

    /// Calculate surprise persistence (autocorrelation)
    pub fn surprise_persistence(&self, surprises: &[SurpriseMetrics], lag: usize) -> f64 {
        if surprises.len() <= lag {
            return 0.0;
        }

        let scores: Vec<f64> = surprises.iter().map(|s| s.composite_score).collect();
        let n = scores.len() - lag;

        let x: Vec<f64> = scores[..n].to_vec();
        let y: Vec<f64> = scores[lag..].to_vec();

        correlation(&x, &y)
    }
}

/// Statistics about surprises
#[derive(Debug, Clone, Default)]
pub struct SurpriseStats {
    pub count: usize,
    pub positive_surprises: usize,
    pub negative_surprises: usize,
    pub avg_price_surprise: f64,
    pub std_price_surprise: f64,
    pub avg_volume_surprise: f64,
    pub std_volume_surprise: f64,
    pub avg_composite: f64,
    pub max_composite: f64,
    pub min_composite: f64,
}

impl SurpriseStats {
    /// Get hit rate for positive surprises
    pub fn positive_rate(&self) -> f64 {
        if self.count > 0 {
            self.positive_surprises as f64 / self.count as f64
        } else {
            0.0
        }
    }

    /// Get hit rate for negative surprises
    pub fn negative_rate(&self) -> f64 {
        if self.count > 0 {
            self.negative_surprises as f64 / self.count as f64
        } else {
            0.0
        }
    }
}

// Helper functions

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

fn std(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let m = mean(values);
    let variance: f64 = values.iter().map(|x| (x - m).powi(2)).sum::<f64>()
        / (values.len() - 1) as f64;
    variance.sqrt()
}

fn correlation(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.is_empty() {
        return 0.0;
    }

    let n = x.len() as f64;
    let mean_x = mean(x);
    let mean_y = mean(y);

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for i in 0..x.len() {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    if var_x > 1e-10 && var_y > 1e-10 {
        cov / (var_x.sqrt() * var_y.sqrt())
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_candles() -> Vec<Candle> {
        (0..50)
            .map(|i| {
                let noise = (i as f64 * 0.5).sin() * 2.0;
                Candle {
                    timestamp: i * 1000,
                    open: 100.0 + noise,
                    high: 102.0 + noise,
                    low: 98.0 + noise,
                    close: 101.0 + noise,
                    volume: 1000.0 + (i as f64 * 10.0).sin() * 200.0,
                    turnover: 100000.0,
                }
            })
            .collect()
    }

    #[test]
    fn test_surprise_calculation() {
        let calculator = SurpriseCalculator::new(10);
        let candles = make_candles();
        let surprises = calculator.calculate(&candles);

        assert_eq!(surprises.len(), candles.len() - 10);
    }

    #[test]
    fn test_surprise_stats() {
        let calculator = SurpriseCalculator::new(10);
        let candles = make_candles();
        let surprises = calculator.calculate(&candles);
        let stats = calculator.rolling_stats(&surprises);

        assert_eq!(stats.count, surprises.len());
    }

    #[test]
    fn test_surprise_direction() {
        let metrics = SurpriseMetrics {
            timestamp: 0,
            actual_return: 0.05,
            expected_return: 0.01,
            price_surprise: 2.5,
            actual_volume: 1500.0,
            expected_volume: 1000.0,
            volume_surprise: 1.5,
            composite_score: 2.0,
        };

        assert_eq!(metrics.direction(), SurpriseDirection::PositiveSurprise);
        assert!(metrics.is_significant(1.5));
    }
}
