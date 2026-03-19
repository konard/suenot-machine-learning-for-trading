//! Data processing utilities for ELECTRA trading
//!
//! This module provides functions for processing market data
//! and preparing it for model consumption.

use crate::api::bybit::Kline;
use serde::{Deserialize, Serialize};

/// Processed market data point with computed returns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessedData {
    pub timestamp: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub returns: f64,
    pub log_returns: f64,
    pub volatility: f64,
    pub direction: i32,
}

/// Data processing pipeline
pub struct DataProcessor {
    lookback_window: usize,
}

impl DataProcessor {
    /// Create a new data processor
    pub fn new(lookback_window: usize) -> Self {
        Self { lookback_window }
    }

    /// Process raw kline data into structured format
    pub fn process_klines(&self, klines: &[Kline]) -> Vec<ProcessedData> {
        if klines.len() < 2 {
            return Vec::new();
        }

        let mut processed = Vec::with_capacity(klines.len());

        for i in 1..klines.len() {
            let prev = &klines[i - 1];
            let curr = &klines[i];

            let returns = if prev.close != 0.0 {
                (curr.close - prev.close) / prev.close
            } else {
                0.0
            };

            let log_returns = if prev.close > 0.0 && curr.close > 0.0 {
                (curr.close / prev.close).ln()
            } else {
                0.0
            };

            let volatility = self.compute_volatility(klines, i);

            let direction = if returns > 0.005 {
                1
            } else if returns < -0.005 {
                -1
            } else {
                0
            };

            processed.push(ProcessedData {
                timestamp: curr.timestamp,
                open: curr.open,
                high: curr.high,
                low: curr.low,
                close: curr.close,
                volume: curr.volume,
                returns,
                log_returns,
                volatility,
                direction,
            });
        }

        processed
    }

    /// Compute rolling volatility at a given index
    fn compute_volatility(&self, klines: &[Kline], idx: usize) -> f64 {
        let start = if idx >= self.lookback_window {
            idx - self.lookback_window
        } else {
            0
        };

        let window = &klines[start..=idx];
        if window.len() < 2 {
            return 0.0;
        }

        let returns: Vec<f64> = window
            .windows(2)
            .map(|w| {
                if w[0].close > 0.0 {
                    (w[1].close / w[0].close).ln()
                } else {
                    0.0
                }
            })
            .collect();

        let n = returns.len() as f64;
        let mean = returns.iter().sum::<f64>() / n;
        let variance = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;
        variance.sqrt()
    }

    /// Generate synthetic news labels based on price movement direction
    pub fn generate_labels(&self, data: &[ProcessedData]) -> Vec<i32> {
        data.iter().map(|d| d.direction).collect()
    }

    /// Split data into train and test sets
    pub fn train_test_split(
        &self,
        data: &[ProcessedData],
        test_ratio: f64,
    ) -> (Vec<ProcessedData>, Vec<ProcessedData>) {
        let split_idx = ((1.0 - test_ratio) * data.len() as f64) as usize;
        let train = data[..split_idx].to_vec();
        let test = data[split_idx..].to_vec();
        (train, test)
    }
}

impl Default for DataProcessor {
    fn default() -> Self {
        Self::new(20)
    }
}
