//! Stream Simulator
//!
//! Simulates streaming data from historical candles for online learning.

use crate::api::Candle;
use crate::features::MomentumFeatures;
use std::collections::VecDeque;

/// Stream observation for online learning
#[derive(Debug, Clone)]
pub struct StreamObservation {
    /// Timestamp
    pub timestamp: u64,
    /// Feature vector
    pub features: Vec<f64>,
    /// Target value (next period return)
    pub target: f64,
    /// Current price
    pub price: f64,
}

/// Stream simulator for historical data
///
/// Converts historical candle data into a stream of observations
/// suitable for online learning.
#[derive(Debug)]
pub struct StreamSimulator {
    /// Historical candles
    candles: Vec<Candle>,
    /// Current position in stream
    position: usize,
    /// Warmup period (for feature calculation)
    warmup: usize,
    /// Feature generator
    feature_gen: MomentumFeatures,
    /// Buffer for recent candles
    buffer: VecDeque<Candle>,
    /// Maximum buffer size
    max_buffer_size: usize,
}

impl StreamSimulator {
    /// Create a new stream simulator
    ///
    /// # Arguments
    ///
    /// * `candles` - Historical candle data
    /// * `periods` - Momentum periods for feature calculation
    pub fn new(candles: Vec<Candle>, periods: Vec<usize>) -> Self {
        let warmup = *periods.iter().max().unwrap_or(&100) + 1;
        let max_buffer_size = warmup * 2;

        Self {
            candles,
            position: 0,
            warmup,
            feature_gen: MomentumFeatures::new(periods),
            buffer: VecDeque::with_capacity(max_buffer_size),
            max_buffer_size,
        }
    }

    /// Create with default hourly periods
    pub fn new_hourly(candles: Vec<Candle>) -> Self {
        Self::new(candles, vec![12, 24, 48, 96])
    }

    /// Create with default daily periods
    pub fn new_daily(candles: Vec<Candle>) -> Self {
        Self::new(candles, vec![1, 5, 21, 63])
    }

    /// Get total number of candles
    pub fn total_candles(&self) -> usize {
        self.candles.len()
    }

    /// Get current position
    pub fn current_position(&self) -> usize {
        self.position
    }

    /// Get remaining observations
    pub fn remaining(&self) -> usize {
        if self.position + 1 >= self.candles.len() {
            0
        } else {
            self.candles.len() - self.position - 1
        }
    }

    /// Check if stream has more data
    pub fn has_next(&self) -> bool {
        self.position + 1 < self.candles.len()
    }

    /// Get next observation from stream
    ///
    /// Returns None if no more data available.
    pub fn next(&mut self) -> Option<StreamObservation> {
        if !self.has_next() {
            return None;
        }

        // Add current candle to buffer
        self.buffer.push_back(self.candles[self.position].clone());

        // Limit buffer size
        while self.buffer.len() > self.max_buffer_size {
            self.buffer.pop_front();
        }

        self.position += 1;

        // Need warmup period for features
        if self.buffer.len() <= self.warmup || !self.has_next() {
            return self.next();
        }

        // Compute features from buffer
        let buffer_vec: Vec<Candle> = self.buffer.iter().cloned().collect();
        let features = self.feature_gen.compute(&buffer_vec)?;

        // Target is next period return
        let current_price = self.candles[self.position].close;
        let next_price = self.candles[self.position + 1].close;
        let target = (next_price - current_price) / current_price;

        Some(StreamObservation {
            timestamp: self.candles[self.position].timestamp,
            features,
            target,
            price: current_price,
        })
    }

    /// Reset stream to beginning
    pub fn reset(&mut self) {
        self.position = 0;
        self.buffer.clear();
    }

    /// Skip to position
    pub fn skip_to(&mut self, position: usize) {
        self.position = position.min(self.candles.len() - 1);
        self.buffer.clear();

        // Fill buffer with candles before position
        let start = if position > self.warmup {
            position - self.warmup
        } else {
            0
        };

        for i in start..position {
            self.buffer.push_back(self.candles[i].clone());
        }
    }

    /// Get feature names
    pub fn feature_names(&self) -> Vec<String> {
        self.feature_gen.feature_names()
    }

    /// Iterate through all observations
    pub fn iter(&mut self) -> StreamIterator<'_> {
        self.reset();
        StreamIterator { simulator: self }
    }
}

/// Iterator over stream observations
pub struct StreamIterator<'a> {
    simulator: &'a mut StreamSimulator,
}

impl<'a> Iterator for StreamIterator<'a> {
    type Item = StreamObservation;

    fn next(&mut self) -> Option<Self::Item> {
        self.simulator.next()
    }
}

/// Batch stream simulator
///
/// Groups observations into batches for mini-batch learning.
#[derive(Debug)]
pub struct BatchStreamSimulator {
    /// Base simulator
    simulator: StreamSimulator,
    /// Batch size
    batch_size: usize,
}

impl BatchStreamSimulator {
    /// Create batch simulator
    pub fn new(candles: Vec<Candle>, periods: Vec<usize>, batch_size: usize) -> Self {
        Self {
            simulator: StreamSimulator::new(candles, periods),
            batch_size,
        }
    }

    /// Get next batch
    pub fn next_batch(&mut self) -> Option<Vec<StreamObservation>> {
        let mut batch = Vec::with_capacity(self.batch_size);

        for _ in 0..self.batch_size {
            if let Some(obs) = self.simulator.next() {
                batch.push(obs);
            } else {
                break;
            }
        }

        if batch.is_empty() {
            None
        } else {
            Some(batch)
        }
    }

    /// Reset to beginning
    pub fn reset(&mut self) {
        self.simulator.reset();
    }

    /// Check if has more batches
    pub fn has_next(&self) -> bool {
        self.simulator.has_next()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_candles(n: usize) -> Vec<Candle> {
        (0..n)
            .map(|i| Candle {
                timestamp: (i * 3600000) as u64,
                open: 100.0 + (i as f64).sin() * 5.0,
                high: 102.0 + (i as f64).sin() * 5.0,
                low: 98.0 + (i as f64).sin() * 5.0,
                close: 100.0 + ((i + 1) as f64).sin() * 5.0,
                volume: 1000.0,
                turnover: 100000.0,
            })
            .collect()
    }

    #[test]
    fn test_stream_simulator_creation() {
        let candles = create_test_candles(200);
        let simulator = StreamSimulator::new_hourly(candles);

        assert_eq!(simulator.total_candles(), 200);
        assert!(simulator.has_next());
    }

    #[test]
    fn test_stream_iteration() {
        let candles = create_test_candles(200);
        let mut simulator = StreamSimulator::new(candles, vec![10, 20]);

        let mut count = 0;
        while let Some(obs) = simulator.next() {
            assert_eq!(obs.features.len(), 2);
            count += 1;
        }

        // Should have observations after warmup period
        assert!(count > 100);
    }

    #[test]
    fn test_stream_reset() {
        let candles = create_test_candles(200);
        let mut simulator = StreamSimulator::new(candles, vec![10, 20]);

        // Consume some observations
        for _ in 0..50 {
            simulator.next();
        }

        let pos_before = simulator.current_position();
        simulator.reset();

        assert_eq!(simulator.current_position(), 0);
        assert!(pos_before > 0);
    }

    #[test]
    fn test_batch_simulator() {
        let candles = create_test_candles(200);
        let mut batch_sim = BatchStreamSimulator::new(candles, vec![10, 20], 32);

        let batch = batch_sim.next_batch();
        assert!(batch.is_some());
        assert_eq!(batch.unwrap().len(), 32);
    }

    #[test]
    fn test_skip_to() {
        let candles = create_test_candles(200);
        let mut simulator = StreamSimulator::new(candles, vec![10, 20]);

        simulator.skip_to(100);
        assert_eq!(simulator.current_position(), 100);
    }
}
