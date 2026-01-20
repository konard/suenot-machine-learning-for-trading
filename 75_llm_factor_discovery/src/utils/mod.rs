//! Utilities module
//!
//! Configuration loading and metrics recording.

mod config;

pub use config::{load_config, AppConfig, ConfigError};

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

/// Simple metrics recorder
pub struct MetricsRecorder {
    counters: Mutex<HashMap<String, AtomicU64>>,
    timings: Mutex<HashMap<String, Vec<f64>>>,
}

impl MetricsRecorder {
    /// Create a new metrics recorder
    pub fn new() -> Self {
        Self {
            counters: Mutex::new(HashMap::new()),
            timings: Mutex::new(HashMap::new()),
        }
    }

    /// Increment a counter
    pub fn increment(&self, name: &str) {
        let mut counters = self.counters.lock().unwrap();
        counters
            .entry(name.to_string())
            .or_insert_with(|| AtomicU64::new(0))
            .fetch_add(1, Ordering::SeqCst);
    }

    /// Record a timing value
    pub fn record_timing(&self, name: &str, duration_ms: f64) {
        let mut timings = self.timings.lock().unwrap();
        timings
            .entry(name.to_string())
            .or_insert_with(Vec::new)
            .push(duration_ms);
    }

    /// Get counter value
    pub fn get_counter(&self, name: &str) -> u64 {
        let counters = self.counters.lock().unwrap();
        counters
            .get(name)
            .map(|c| c.load(Ordering::SeqCst))
            .unwrap_or(0)
    }

    /// Get timing statistics
    pub fn get_timing_stats(&self, name: &str) -> Option<TimingStats> {
        let timings = self.timings.lock().unwrap();
        let values = timings.get(name)?;

        if values.is_empty() {
            return None;
        }

        let count = values.len();
        let sum: f64 = values.iter().sum();
        let mean = sum / count as f64;
        let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let variance: f64 = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / count as f64;
        let std = variance.sqrt();

        Some(TimingStats {
            count,
            mean,
            std,
            min,
            max,
        })
    }
}

impl Default for MetricsRecorder {
    fn default() -> Self {
        Self::new()
    }
}

/// Timing statistics
#[derive(Debug, Clone)]
pub struct TimingStats {
    pub count: usize,
    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub max: f64,
}

/// Global metrics instance
pub struct Metrics;

impl Metrics {
    /// Get the global metrics recorder
    pub fn global() -> &'static MetricsRecorder {
        use std::sync::OnceLock;
        static METRICS: OnceLock<MetricsRecorder> = OnceLock::new();
        METRICS.get_or_init(MetricsRecorder::new)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_counter() {
        let recorder = MetricsRecorder::new();
        recorder.increment("test_counter");
        recorder.increment("test_counter");

        assert_eq!(recorder.get_counter("test_counter"), 2);
    }

    #[test]
    fn test_timing() {
        let recorder = MetricsRecorder::new();
        recorder.record_timing("test_timing", 10.0);
        recorder.record_timing("test_timing", 20.0);
        recorder.record_timing("test_timing", 30.0);

        let stats = recorder.get_timing_stats("test_timing").unwrap();
        assert_eq!(stats.count, 3);
        assert!((stats.mean - 20.0).abs() < 1e-10);
        assert!((stats.min - 10.0).abs() < 1e-10);
        assert!((stats.max - 30.0).abs() < 1e-10);
    }
}
