//! Concept Drift Detection Module
//!
//! This module provides implementations of drift detection algorithms
//! for detecting changes in data distributions.

mod adwin;
mod ddm;

pub use adwin::ADWIN;
pub use ddm::DDM;

/// Trait for drift detectors
pub trait DriftDetector {
    /// Update detector with new value, returns true if drift detected
    fn update(&mut self, value: f64) -> bool;

    /// Reset detector state
    fn reset(&mut self);

    /// Check if drift was detected on last update
    fn drift_detected(&self) -> bool;

    /// Get number of samples processed
    fn samples_seen(&self) -> u64;
}
