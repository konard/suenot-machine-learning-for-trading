//! # VPIN (Volume-Synchronized Probability of Informed Trading)
//!
//! Implementation based on Easley, López de Prado, and O'Hara (2012):
//! "Flow Toxicity and Liquidity in a High-frequency World"
//!
//! VPIN measures the probability that informed traders are present in the market.

use crate::data::trade::{Trade, TradeClassifier, TradeSide, VolumeBucket};
use chrono::{DateTime, Utc};
use std::collections::VecDeque;

/// VPIN Calculator
#[derive(Debug)]
pub struct VpinCalculator {
    /// Target volume per bucket
    bucket_volume: f64,
    /// Number of buckets for VPIN calculation
    num_buckets: usize,
    /// Current bucket being filled
    current_bucket: VolumeBucket,
    /// Completed buckets
    completed_buckets: VecDeque<CompletedBucket>,
    /// Trade classifier for Lee-Ready
    classifier: TradeClassifier,
    /// Current mid price for classification
    current_mid: f64,
    /// VPIN history
    vpin_history: VecDeque<VpinPoint>,
    /// Maximum history size
    max_history: usize,
}

/// A completed volume bucket
#[derive(Debug, Clone)]
pub struct CompletedBucket {
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub volume: f64,
    pub buy_volume: f64,
    pub sell_volume: f64,
    pub imbalance: f64,
}

/// VPIN data point
#[derive(Debug, Clone)]
pub struct VpinPoint {
    pub timestamp: DateTime<Utc>,
    pub vpin: f64,
    pub num_buckets: usize,
}

impl VpinCalculator {
    /// Create a new VPIN calculator
    ///
    /// # Arguments
    /// * `bucket_volume` - Target volume for each bucket (e.g., 50 BTC)
    /// * `num_buckets` - Number of buckets for rolling VPIN (typically 50)
    pub fn new(bucket_volume: f64, num_buckets: usize) -> Self {
        Self {
            bucket_volume,
            num_buckets,
            current_bucket: VolumeBucket::new(bucket_volume),
            completed_buckets: VecDeque::with_capacity(num_buckets + 10),
            classifier: TradeClassifier::new(),
            current_mid: 0.0,
            vpin_history: VecDeque::with_capacity(1000),
            max_history: 1000,
        }
    }

    /// Create with default parameters
    pub fn default_crypto() -> Self {
        // Typical for BTC: 50 BTC per bucket, 50 buckets for VPIN
        Self::new(50.0, 50)
    }

    /// Update mid price for trade classification
    pub fn set_mid_price(&mut self, mid: f64) {
        self.current_mid = mid;
    }

    /// Process a new trade
    ///
    /// Returns Some(vpin) when a new bucket is completed
    pub fn add_trade(&mut self, trade: &Trade) -> Option<f64> {
        // Classify trade if needed
        let side = if trade.is_buyer_maker {
            TradeSide::Sell
        } else {
            TradeSide::Buy
        };

        // Add to current bucket
        let mut remaining = trade.size;
        let mut new_vpin = None;

        while remaining > 0.0 {
            let bucket_remaining = self.current_bucket.remaining();
            let fill = remaining.min(bucket_remaining);

            // Update bucket
            if self.current_bucket.start_time.is_none() {
                self.current_bucket.start_time = Some(trade.timestamp);
            }
            self.current_bucket.end_time = Some(trade.timestamp);
            self.current_bucket.filled_volume += fill;

            match side {
                TradeSide::Buy => self.current_bucket.buy_volume += fill,
                TradeSide::Sell => self.current_bucket.sell_volume += fill,
            }

            remaining -= fill;

            // Check if bucket is complete
            if self.current_bucket.is_complete() {
                let completed = CompletedBucket {
                    start_time: self.current_bucket.start_time.unwrap_or_else(Utc::now),
                    end_time: self.current_bucket.end_time.unwrap_or_else(Utc::now),
                    volume: self.current_bucket.filled_volume,
                    buy_volume: self.current_bucket.buy_volume,
                    sell_volume: self.current_bucket.sell_volume,
                    imbalance: self.current_bucket.imbalance(),
                };

                self.completed_buckets.push_back(completed);

                // Keep only required number of buckets
                while self.completed_buckets.len() > self.num_buckets + 10 {
                    self.completed_buckets.pop_front();
                }

                // Calculate VPIN if we have enough buckets
                if self.completed_buckets.len() >= self.num_buckets {
                    let vpin = self.calculate_vpin();
                    new_vpin = Some(vpin);

                    // Store in history
                    let point = VpinPoint {
                        timestamp: trade.timestamp,
                        vpin,
                        num_buckets: self.num_buckets,
                    };
                    self.vpin_history.push_back(point);
                    if self.vpin_history.len() > self.max_history {
                        self.vpin_history.pop_front();
                    }
                }

                // Start new bucket
                self.current_bucket = VolumeBucket::new(self.bucket_volume);
            }
        }

        new_vpin
    }

    /// Process multiple trades
    pub fn add_trades(&mut self, trades: &[Trade]) -> Vec<f64> {
        trades.iter().filter_map(|t| self.add_trade(t)).collect()
    }

    /// Calculate VPIN from completed buckets
    fn calculate_vpin(&self) -> f64 {
        let buckets: Vec<_> = self
            .completed_buckets
            .iter()
            .rev()
            .take(self.num_buckets)
            .collect();

        if buckets.is_empty() {
            return 0.0;
        }

        let total_imbalance: f64 = buckets.iter().map(|b| b.imbalance).sum();
        total_imbalance / buckets.len() as f64
    }

    /// Get current VPIN value
    pub fn current_vpin(&self) -> Option<f64> {
        if self.completed_buckets.len() >= self.num_buckets {
            Some(self.calculate_vpin())
        } else {
            None
        }
    }

    /// Get the number of completed buckets
    pub fn bucket_count(&self) -> usize {
        self.completed_buckets.len()
    }

    /// Get progress of current bucket (0.0 to 1.0)
    pub fn current_bucket_progress(&self) -> f64 {
        self.current_bucket.filled_volume / self.bucket_volume
    }

    /// Get recent VPIN values
    pub fn recent_vpin(&self, n: usize) -> Vec<f64> {
        self.vpin_history
            .iter()
            .rev()
            .take(n)
            .map(|p| p.vpin)
            .collect()
    }

    /// Calculate VPIN Z-score
    pub fn z_score(&self, window: usize) -> Option<f64> {
        if self.vpin_history.len() < window {
            return None;
        }

        let values: Vec<f64> = self
            .vpin_history
            .iter()
            .rev()
            .take(window)
            .map(|p| p.vpin)
            .collect();

        let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
        let variance: f64 =
            values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
        let std = variance.sqrt();

        if std > 0.0 {
            let current = values.first()?;
            Some((current - mean) / std)
        } else {
            Some(0.0)
        }
    }

    /// Get VPIN statistics
    pub fn statistics(&self) -> VpinStatistics {
        if self.vpin_history.is_empty() {
            return VpinStatistics::default();
        }

        let values: Vec<f64> = self.vpin_history.iter().map(|p| p.vpin).collect();
        let n = values.len() as f64;

        let sum: f64 = values.iter().sum();
        let mean = sum / n;

        let variance: f64 = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        let std = variance.sqrt();

        let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let current = self.current_vpin().unwrap_or(0.0);

        // Percentile calculation
        let mut sorted = values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let percentile = |p: f64| -> f64 {
            let idx = ((p / 100.0) * (sorted.len() - 1) as f64) as usize;
            sorted[idx]
        };

        VpinStatistics {
            count: values.len(),
            current,
            mean,
            std,
            min,
            max,
            percentile_25: percentile(25.0),
            percentile_50: percentile(50.0),
            percentile_75: percentile(75.0),
            percentile_95: percentile(95.0),
        }
    }

    /// Reset the calculator
    pub fn reset(&mut self) {
        self.current_bucket = VolumeBucket::new(self.bucket_volume);
        self.completed_buckets.clear();
        self.classifier = TradeClassifier::new();
        self.vpin_history.clear();
    }

    /// Check if VPIN indicates high toxicity
    pub fn is_toxic(&self, threshold: f64) -> bool {
        self.current_vpin().map(|v| v > threshold).unwrap_or(false)
    }
}

/// VPIN statistics
#[derive(Debug, Clone, Default)]
pub struct VpinStatistics {
    pub count: usize,
    pub current: f64,
    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub max: f64,
    pub percentile_25: f64,
    pub percentile_50: f64,
    pub percentile_75: f64,
    pub percentile_95: f64,
}

impl VpinStatistics {
    /// Get toxicity level interpretation
    pub fn toxicity_level(&self) -> ToxicityLevel {
        if self.current > 0.7 {
            ToxicityLevel::VeryHigh
        } else if self.current > 0.5 {
            ToxicityLevel::High
        } else if self.current > 0.3 {
            ToxicityLevel::Medium
        } else {
            ToxicityLevel::Low
        }
    }
}

/// Toxicity level interpretation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToxicityLevel {
    Low,
    Medium,
    High,
    VeryHigh,
}

impl std::fmt::Display for ToxicityLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ToxicityLevel::Low => write!(f, "Low"),
            ToxicityLevel::Medium => write!(f, "Medium"),
            ToxicityLevel::High => write!(f, "High"),
            ToxicityLevel::VeryHigh => write!(f, "Very High"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_trade(size: f64, is_buy: bool) -> Trade {
        Trade::new(
            "BTCUSDT".to_string(),
            Utc::now(),
            50000.0,
            size,
            !is_buy, // is_buyer_maker is opposite of is_buy
            uuid::Uuid::new_v4().to_string(),
        )
    }

    #[test]
    fn test_vpin_bucket_completion() {
        let mut calc = VpinCalculator::new(10.0, 5);

        // Add trades totaling one bucket
        for _ in 0..10 {
            let trade = create_trade(1.0, true);
            calc.add_trade(&trade);
        }

        assert_eq!(calc.bucket_count(), 1);
    }

    #[test]
    fn test_vpin_calculation() {
        let mut calc = VpinCalculator::new(10.0, 2);

        // Fill first bucket with all buys
        for _ in 0..10 {
            calc.add_trade(&create_trade(1.0, true));
        }

        // Fill second bucket with all sells
        for _ in 0..10 {
            calc.add_trade(&create_trade(1.0, false));
        }

        // VPIN should be high (imbalance in each bucket)
        let vpin = calc.current_vpin().unwrap();
        assert!(vpin > 0.9); // Both buckets have 100% imbalance
    }

    #[test]
    fn test_vpin_balanced() {
        let mut calc = VpinCalculator::new(10.0, 2);

        // Fill buckets with balanced trading
        for _ in 0..20 {
            calc.add_trade(&create_trade(0.5, true));
            calc.add_trade(&create_trade(0.5, false));
        }

        // VPIN should be low (balanced)
        let vpin = calc.current_vpin().unwrap();
        assert!(vpin < 0.1);
    }

    #[test]
    fn test_bucket_progress() {
        let mut calc = VpinCalculator::new(10.0, 5);

        calc.add_trade(&create_trade(3.0, true));
        assert!((calc.current_bucket_progress() - 0.3).abs() < 0.001);

        calc.add_trade(&create_trade(2.0, true));
        assert!((calc.current_bucket_progress() - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_toxicity_levels() {
        let stats = VpinStatistics {
            current: 0.75,
            ..Default::default()
        };
        assert_eq!(stats.toxicity_level(), ToxicityLevel::VeryHigh);

        let stats = VpinStatistics {
            current: 0.55,
            ..Default::default()
        };
        assert_eq!(stats.toxicity_level(), ToxicityLevel::High);

        let stats = VpinStatistics {
            current: 0.2,
            ..Default::default()
        };
        assert_eq!(stats.toxicity_level(), ToxicityLevel::Low);
    }
}
