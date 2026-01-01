//! Spike Encoding Module
//!
//! Converts market data into spike trains for processing by the SNN.

pub mod rate;
pub mod temporal;
pub mod delta;

use crate::neuron::SpikeEvent;
use chrono::{DateTime, Utc};

/// Market data structure
#[derive(Debug, Clone)]
pub struct MarketData {
    /// Bid prices (best to worst)
    pub bid_prices: Vec<f64>,
    /// Ask prices (best to worst)
    pub ask_prices: Vec<f64>,
    /// Bid volumes
    pub bid_volumes: Vec<f64>,
    /// Ask volumes
    pub ask_volumes: Vec<f64>,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

impl MarketData {
    /// Get the mid price
    pub fn mid_price(&self) -> Option<f64> {
        match (self.bid_prices.first(), self.ask_prices.first()) {
            (Some(&bid), Some(&ask)) => Some((bid + ask) / 2.0),
            _ => None,
        }
    }

    /// Get the spread
    pub fn spread(&self) -> Option<f64> {
        match (self.bid_prices.first(), self.ask_prices.first()) {
            (Some(&bid), Some(&ask)) => Some(ask - bid),
            _ => None,
        }
    }

    /// Get order imbalance
    pub fn order_imbalance(&self) -> f64 {
        let bid_vol: f64 = self.bid_volumes.iter().sum();
        let ask_vol: f64 = self.ask_volumes.iter().sum();
        let total = bid_vol + ask_vol;
        if total > 0.0 {
            (bid_vol - ask_vol) / total
        } else {
            0.0
        }
    }
}

/// Encoder configuration
#[derive(Debug, Clone)]
pub struct EncoderConfig {
    /// Maximum spike rate (Hz)
    pub max_rate: f64,
    /// Number of neurons per feature
    pub neurons_per_feature: usize,
    /// Price normalization range
    pub price_range: (f64, f64),
    /// Volume normalization range
    pub volume_range: (f64, f64),
    /// Time window for encoding (ms)
    pub time_window: f64,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            max_rate: 100.0,
            neurons_per_feature: 4,
            price_range: (0.0, 100000.0),
            volume_range: (0.0, 1000.0),
            time_window: 10.0,
        }
    }
}

/// Trait for spike encoders
pub trait Encoder: Send + Sync {
    /// Encode market data into spikes
    fn encode(&self, data: &MarketData) -> Vec<SpikeEvent>;

    /// Get the number of output neurons
    fn output_size(&self) -> usize;

    /// Reset encoder state
    fn reset(&mut self);
}

/// Normalize a value to [0, 1] range
pub fn normalize(value: f64, min: f64, max: f64) -> f64 {
    ((value - min) / (max - min)).clamp(0.0, 1.0)
}

/// Denormalize a value from [0, 1] range
pub fn denormalize(normalized: f64, min: f64, max: f64) -> f64 {
    normalized * (max - min) + min
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_market_data_mid_price() {
        let data = MarketData {
            bid_prices: vec![100.0],
            ask_prices: vec![101.0],
            bid_volumes: vec![1.0],
            ask_volumes: vec![1.0],
            timestamp: Utc::now(),
        };

        assert_eq!(data.mid_price(), Some(100.5));
    }

    #[test]
    fn test_market_data_spread() {
        let data = MarketData {
            bid_prices: vec![100.0],
            ask_prices: vec![101.0],
            bid_volumes: vec![1.0],
            ask_volumes: vec![1.0],
            timestamp: Utc::now(),
        };

        assert_eq!(data.spread(), Some(1.0));
    }

    #[test]
    fn test_order_imbalance() {
        let data = MarketData {
            bid_prices: vec![100.0],
            ask_prices: vec![101.0],
            bid_volumes: vec![10.0],
            ask_volumes: vec![5.0],
            timestamp: Utc::now(),
        };

        let imbalance = data.order_imbalance();
        assert!((imbalance - 0.333).abs() < 0.01);
    }

    #[test]
    fn test_normalize() {
        assert_eq!(normalize(50.0, 0.0, 100.0), 0.5);
        assert_eq!(normalize(0.0, 0.0, 100.0), 0.0);
        assert_eq!(normalize(100.0, 0.0, 100.0), 1.0);
    }
}
