//! Data buffer for streaming market data
//!
//! Provides a ring buffer implementation for efficient storage of
//! streaming market data with fixed memory usage.

use super::types::OhlcvBar;
use std::collections::VecDeque;

/// Ring buffer for OHLCV data
#[derive(Debug)]
pub struct DataBuffer {
    /// Maximum capacity
    capacity: usize,
    /// Internal storage
    data: VecDeque<OhlcvBar>,
}

impl DataBuffer {
    /// Create a new buffer with specified capacity
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            data: VecDeque::with_capacity(capacity),
        }
    }

    /// Add a new bar to the buffer
    pub fn push(&mut self, bar: OhlcvBar) {
        if self.data.len() >= self.capacity {
            self.data.pop_front();
        }
        self.data.push_back(bar);
    }

    /// Get the latest n bars
    pub fn latest(&self, n: usize) -> Vec<&OhlcvBar> {
        let start = self.data.len().saturating_sub(n);
        self.data.iter().skip(start).collect()
    }

    /// Get all bars
    pub fn all(&self) -> Vec<&OhlcvBar> {
        self.data.iter().collect()
    }

    /// Get closing prices
    pub fn closes(&self) -> Vec<f64> {
        self.data.iter().map(|b| b.close).collect()
    }

    /// Get volumes
    pub fn volumes(&self) -> Vec<f64> {
        self.data.iter().map(|b| b.volume).collect()
    }

    /// Get the number of bars in buffer
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Check if buffer is full
    pub fn is_full(&self) -> bool {
        self.data.len() >= self.capacity
    }

    /// Clear the buffer
    pub fn clear(&mut self) {
        self.data.clear();
    }

    /// Get capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get the latest bar
    pub fn last(&self) -> Option<&OhlcvBar> {
        self.data.back()
    }

    /// Calculate simple returns over a period
    pub fn returns(&self, period: usize) -> Vec<f64> {
        let closes = self.closes();
        if closes.len() < period + 1 {
            return vec![];
        }

        closes
            .windows(period + 1)
            .map(|w| (w[period] - w[0]) / w[0])
            .collect()
    }
}

impl Default for DataBuffer {
    fn default() -> Self {
        Self::new(1000)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn create_test_bar(close: f64) -> OhlcvBar {
        OhlcvBar {
            timestamp: Utc::now(),
            open: close,
            high: close + 1.0,
            low: close - 1.0,
            close,
            volume: 1000.0,
            turnover: close * 1000.0,
        }
    }

    #[test]
    fn test_buffer_push() {
        let mut buffer = DataBuffer::new(3);
        buffer.push(create_test_bar(100.0));
        buffer.push(create_test_bar(101.0));
        buffer.push(create_test_bar(102.0));

        assert_eq!(buffer.len(), 3);
        assert!(buffer.is_full());
    }

    #[test]
    fn test_buffer_overflow() {
        let mut buffer = DataBuffer::new(3);
        buffer.push(create_test_bar(100.0));
        buffer.push(create_test_bar(101.0));
        buffer.push(create_test_bar(102.0));
        buffer.push(create_test_bar(103.0));

        assert_eq!(buffer.len(), 3);
        let closes = buffer.closes();
        assert!((closes[0] - 101.0).abs() < 0.001);
        assert!((closes[2] - 103.0).abs() < 0.001);
    }

    #[test]
    fn test_buffer_latest() {
        let mut buffer = DataBuffer::new(10);
        for i in 0..5 {
            buffer.push(create_test_bar(100.0 + i as f64));
        }

        let latest = buffer.latest(3);
        assert_eq!(latest.len(), 3);
        assert!((latest[0].close - 102.0).abs() < 0.001);
        assert!((latest[2].close - 104.0).abs() < 0.001);
    }

    #[test]
    fn test_buffer_returns() {
        let mut buffer = DataBuffer::new(10);
        buffer.push(create_test_bar(100.0));
        buffer.push(create_test_bar(110.0));
        buffer.push(create_test_bar(121.0));

        let returns = buffer.returns(1);
        assert_eq!(returns.len(), 2);
        assert!((returns[0] - 0.1).abs() < 0.001);
        assert!((returns[1] - 0.1).abs() < 0.001);
    }
}
