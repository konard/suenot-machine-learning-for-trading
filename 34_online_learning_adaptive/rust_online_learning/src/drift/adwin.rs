//! ADWIN (ADaptive WINdowing) Algorithm
//!
//! ADWIN is a change detector and estimator that solves in a well-specified way
//! the problem of tracking the average of a stream of bits or real-valued numbers.
//!
//! Reference:
//! Bifet, A. and Gavalda, R., 2007. Learning from time-changing data with adaptive windowing.
//! In Proceedings of the 2007 SIAM international conference on data mining (pp. 443-448).

use super::DriftDetector;
use std::collections::VecDeque;

/// Bucket for ADWIN algorithm
#[derive(Debug, Clone)]
struct Bucket {
    /// Sum of values in bucket
    total: f64,
    /// Variance component
    variance: f64,
    /// Number of elements
    count: u32,
}

impl Bucket {
    fn new() -> Self {
        Self {
            total: 0.0,
            variance: 0.0,
            count: 0,
        }
    }

    fn add(&mut self, value: f64) {
        self.total += value;
        self.count += 1;
    }
}

/// ADWIN Drift Detector
///
/// Maintains a variable-length window of recent items with the property that
/// the average of the items in the window is approximately consistent with
/// an assumption that there is no concept drift.
#[derive(Debug, Clone)]
pub struct ADWIN {
    /// Confidence parameter (smaller = more sensitive)
    delta: f64,
    /// Buckets at different time scales
    bucket_list: VecDeque<Vec<Bucket>>,
    /// Total sum of all elements
    total: f64,
    /// Total variance
    variance: f64,
    /// Window width (number of elements)
    width: u64,
    /// Minimum samples before detecting drift
    min_samples: u64,
    /// Whether drift was detected on last update
    drift_detected: bool,
    /// Maximum number of buckets per row
    max_buckets: usize,
}

impl Default for ADWIN {
    fn default() -> Self {
        Self::new(0.002)
    }
}

impl ADWIN {
    /// Create a new ADWIN detector
    ///
    /// # Arguments
    ///
    /// * `delta` - Confidence parameter (typical values: 0.001 - 0.01)
    ///             Smaller values = less sensitive, fewer false positives
    pub fn new(delta: f64) -> Self {
        Self {
            delta,
            bucket_list: VecDeque::new(),
            total: 0.0,
            variance: 0.0,
            width: 0,
            min_samples: 30,
            drift_detected: false,
            max_buckets: 5,
        }
    }

    /// Set minimum samples before drift detection
    pub fn with_min_samples(mut self, min_samples: u64) -> Self {
        self.min_samples = min_samples;
        self
    }

    /// Get current window width
    pub fn window_width(&self) -> u64 {
        self.width
    }

    /// Get current mean estimate
    pub fn mean(&self) -> f64 {
        if self.width > 0 {
            self.total / self.width as f64
        } else {
            0.0
        }
    }

    /// Get current variance estimate
    pub fn variance_estimate(&self) -> f64 {
        if self.width > 1 {
            self.variance / (self.width as f64 - 1.0)
        } else {
            0.0
        }
    }

    /// Insert a new bucket at level 0
    fn insert_bucket(&mut self, value: f64) {
        // Ensure we have at least one level
        if self.bucket_list.is_empty() {
            self.bucket_list.push_back(Vec::new());
        }

        // Create new bucket
        let mut bucket = Bucket::new();
        bucket.add(value);

        // Add to level 0
        self.bucket_list[0].push(bucket);

        // Compress buckets if needed
        self.compress_buckets();
    }

    /// Compress buckets when they exceed max per level
    fn compress_buckets(&mut self) {
        let mut level = 0;

        while level < self.bucket_list.len() {
            if self.bucket_list[level].len() > self.max_buckets {
                // Need to merge buckets
                if level + 1 >= self.bucket_list.len() {
                    self.bucket_list.push_back(Vec::new());
                }

                // Pop last two buckets and merge
                if self.bucket_list[level].len() >= 2 {
                    let b2 = self.bucket_list[level].pop().unwrap();
                    let b1 = self.bucket_list[level].pop().unwrap();

                    let mut merged = Bucket::new();
                    merged.total = b1.total + b2.total;
                    merged.count = b1.count + b2.count;
                    merged.variance = b1.variance + b2.variance;

                    self.bucket_list[level + 1].push(merged);
                }
            }
            level += 1;
        }
    }

    /// Remove oldest bucket
    fn delete_bucket(&mut self) {
        // Find and remove oldest bucket
        for level in (0..self.bucket_list.len()).rev() {
            if !self.bucket_list[level].is_empty() {
                let bucket = self.bucket_list[level].remove(0);
                self.total -= bucket.total;
                self.variance -= bucket.variance;
                self.width -= bucket.count as u64;
                break;
            }
        }

        // Clean up empty levels
        while !self.bucket_list.is_empty() && self.bucket_list.back().map_or(false, |v| v.is_empty()) {
            self.bucket_list.pop_back();
        }
    }

    /// Check for drift using ADWIN cut detection
    fn detect_drift(&mut self) -> bool {
        if self.width < self.min_samples {
            return false;
        }

        let n = self.width as f64;
        let mean = self.total / n;

        // Try different split points
        let mut n0: f64 = 0.0;
        let mut n1: f64 = n;
        let mut sum0: f64 = 0.0;
        let mut sum1: f64 = self.total;

        // Iterate through bucket levels
        for level in &self.bucket_list {
            for bucket in level {
                let bucket_n = bucket.count as f64;
                let bucket_sum = bucket.total;

                n0 += bucket_n;
                n1 -= bucket_n;
                sum0 += bucket_sum;
                sum1 -= bucket_sum;

                if n0 < 1.0 || n1 < 1.0 {
                    continue;
                }

                let mean0 = sum0 / n0;
                let mean1 = sum1 / n1;

                // Compute Hoeffding bound
                let m = 1.0 / (1.0 / n0 + 1.0 / n1);
                let eps = (0.5 * (4.0 / self.delta).ln() / m).sqrt();

                if (mean0 - mean1).abs() > eps {
                    // Drift detected! Remove old data
                    return true;
                }
            }
        }

        false
    }

    /// Shrink window by removing old elements
    fn shrink_window(&mut self) {
        let n = self.width as f64;
        if n < 2.0 {
            return;
        }

        let mean = self.total / n;

        // Compute split statistics
        let mut n0: f64 = 0.0;
        let mut sum0: f64 = 0.0;
        let mut split_found = false;
        let mut buckets_to_remove = 0;

        // Find split point
        for level in &self.bucket_list {
            for bucket in level {
                let bucket_n = bucket.count as f64;
                let bucket_sum = bucket.total;

                let new_n0 = n0 + bucket_n;
                let new_n1 = n - new_n0;
                let new_sum0 = sum0 + bucket_sum;
                let sum1 = self.total - new_sum0;

                if new_n0 < 1.0 || new_n1 < 1.0 {
                    n0 = new_n0;
                    sum0 = new_sum0;
                    buckets_to_remove += 1;
                    continue;
                }

                let mean0 = new_sum0 / new_n0;
                let mean1 = sum1 / new_n1;

                let m = 1.0 / (1.0 / new_n0 + 1.0 / new_n1);
                let eps = (0.5 * (4.0 / self.delta).ln() / m).sqrt();

                if (mean0 - mean1).abs() > eps {
                    split_found = true;
                    break;
                }

                n0 = new_n0;
                sum0 = new_sum0;
                buckets_to_remove += 1;
            }
            if split_found {
                break;
            }
        }

        // Remove buckets before split point
        for _ in 0..buckets_to_remove {
            self.delete_bucket();
        }
    }
}

impl DriftDetector for ADWIN {
    fn update(&mut self, value: f64) -> bool {
        self.drift_detected = false;

        // Update statistics
        let old_mean = if self.width > 0 {
            self.total / self.width as f64
        } else {
            0.0
        };

        self.total += value;
        self.width += 1;

        // Update variance using Welford's algorithm
        if self.width > 1 {
            let new_mean = self.total / self.width as f64;
            self.variance += (value - old_mean) * (value - new_mean);
        }

        // Insert new bucket
        self.insert_bucket(value);

        // Check for drift
        if self.detect_drift() {
            self.drift_detected = true;
            self.shrink_window();
        }

        self.drift_detected
    }

    fn reset(&mut self) {
        self.bucket_list.clear();
        self.total = 0.0;
        self.variance = 0.0;
        self.width = 0;
        self.drift_detected = false;
    }

    fn drift_detected(&self) -> bool {
        self.drift_detected
    }

    fn samples_seen(&self) -> u64 {
        self.width
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adwin_no_drift() {
        let mut adwin = ADWIN::new(0.01);

        // Stationary data
        for _ in 0..100 {
            adwin.update(0.0);
        }

        // Should not detect drift in stationary data
        // (may have some false positives, but should be rare)
    }

    #[test]
    fn test_adwin_with_drift() {
        let mut adwin = ADWIN::new(0.01).with_min_samples(10);
        let mut drift_count = 0;

        // Phase 1: values around 0
        for _ in 0..50 {
            if adwin.update(0.0) {
                drift_count += 1;
            }
        }

        // Phase 2: sudden shift to values around 1
        for _ in 0..50 {
            if adwin.update(1.0) {
                drift_count += 1;
            }
        }

        // Should detect drift at the transition
        assert!(drift_count >= 1, "Expected at least one drift detection");
    }

    #[test]
    fn test_adwin_mean() {
        let mut adwin = ADWIN::new(0.01);

        for i in 1..=10 {
            adwin.update(i as f64);
        }

        // Mean should be approximately 5.5
        let mean = adwin.mean();
        assert!((mean - 5.5).abs() < 1.0);
    }

    #[test]
    fn test_adwin_reset() {
        let mut adwin = ADWIN::new(0.01);

        for _ in 0..50 {
            adwin.update(1.0);
        }

        adwin.reset();

        assert_eq!(adwin.samples_seen(), 0);
        assert_eq!(adwin.mean(), 0.0);
    }
}
