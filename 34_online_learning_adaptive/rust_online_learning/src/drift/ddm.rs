//! DDM (Drift Detection Method)
//!
//! DDM monitors the error rate of a classifier and signals drift when
//! the error rate significantly increases.
//!
//! Reference:
//! Gama, J., Medas, P., Castillo, G. and Rodrigues, P., 2004.
//! Learning with drift detection. In Brazilian symposium on artificial intelligence (pp. 286-295).

use super::DriftDetector;

/// Drift detection state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DriftState {
    /// No drift detected
    NoDrift,
    /// Warning: possible drift
    Warning,
    /// Drift detected
    Drift,
}

/// DDM Drift Detector
///
/// Detects drift based on the error rate of predictions.
/// Assumes binary errors (0 or 1) but can work with continuous values
/// by treating them as error magnitudes.
#[derive(Debug, Clone)]
pub struct DDM {
    /// Minimum number of samples before detection
    min_samples: u64,
    /// Number of samples seen
    n_samples: u64,
    /// Sum of errors
    sum_errors: f64,
    /// Sum of squared errors
    sum_errors_sq: f64,
    /// Minimum error rate + standard deviation seen
    min_p_s: f64,
    /// Error rate at minimum
    min_p: f64,
    /// Standard deviation at minimum
    min_s: f64,
    /// Sample count at minimum
    min_sample: u64,
    /// Warning level multiplier
    warning_level: f64,
    /// Drift level multiplier
    drift_level: f64,
    /// Current state
    state: DriftState,
    /// Whether drift was detected on last update
    drift_detected: bool,
    /// Whether we're in warning state
    in_warning: bool,
    /// Number of drifts detected
    drift_count: u64,
}

impl Default for DDM {
    fn default() -> Self {
        Self::new(30)
    }
}

impl DDM {
    /// Create a new DDM detector
    ///
    /// # Arguments
    ///
    /// * `min_samples` - Minimum samples before detection starts
    pub fn new(min_samples: u64) -> Self {
        Self {
            min_samples,
            n_samples: 0,
            sum_errors: 0.0,
            sum_errors_sq: 0.0,
            min_p_s: f64::MAX,
            min_p: f64::MAX,
            min_s: f64::MAX,
            min_sample: 0,
            warning_level: 2.0,
            drift_level: 3.0,
            state: DriftState::NoDrift,
            drift_detected: false,
            in_warning: false,
            drift_count: 0,
        }
    }

    /// Set warning level (default: 2.0)
    pub fn with_warning_level(mut self, level: f64) -> Self {
        self.warning_level = level;
        self
    }

    /// Set drift level (default: 3.0)
    pub fn with_drift_level(mut self, level: f64) -> Self {
        self.drift_level = level;
        self
    }

    /// Get current state
    pub fn state(&self) -> DriftState {
        self.state
    }

    /// Check if in warning state
    pub fn in_warning(&self) -> bool {
        self.in_warning
    }

    /// Get current error rate estimate
    pub fn error_rate(&self) -> f64 {
        if self.n_samples > 0 {
            self.sum_errors / self.n_samples as f64
        } else {
            0.0
        }
    }

    /// Get number of drifts detected
    pub fn drift_count(&self) -> u64 {
        self.drift_count
    }

    /// Update with prediction error
    ///
    /// # Arguments
    ///
    /// * `error` - Error value (0 or 1 for binary, or continuous error magnitude)
    pub fn add_element(&mut self, error: f64) -> DriftState {
        self.drift_detected = false;
        self.n_samples += 1;
        self.sum_errors += error;
        self.sum_errors_sq += error * error;

        if self.n_samples < self.min_samples {
            return DriftState::NoDrift;
        }

        // Compute error rate and standard deviation
        let p = self.sum_errors / self.n_samples as f64;
        let s = ((p * (1.0 - p)) / self.n_samples as f64).sqrt();
        let p_s = p + s;

        // Update minimum
        if p_s < self.min_p_s {
            self.min_p_s = p_s;
            self.min_p = p;
            self.min_s = s;
            self.min_sample = self.n_samples;
        }

        // Check for drift
        if p + s >= self.min_p + self.drift_level * self.min_s {
            self.state = DriftState::Drift;
            self.drift_detected = true;
            self.drift_count += 1;
            self.in_warning = false;

            // Reset after drift
            self.partial_reset();
        } else if p + s >= self.min_p + self.warning_level * self.min_s {
            self.state = DriftState::Warning;
            self.in_warning = true;
        } else {
            self.state = DriftState::NoDrift;
            self.in_warning = false;
        }

        self.state
    }

    /// Partial reset after drift (keep some state)
    fn partial_reset(&mut self) {
        self.n_samples = 0;
        self.sum_errors = 0.0;
        self.sum_errors_sq = 0.0;
        self.min_p_s = f64::MAX;
        self.min_p = f64::MAX;
        self.min_s = f64::MAX;
        self.min_sample = 0;
    }
}

impl DriftDetector for DDM {
    fn update(&mut self, value: f64) -> bool {
        // Treat value as error (should be positive)
        let error = value.abs();
        self.add_element(error);
        self.drift_detected
    }

    fn reset(&mut self) {
        self.n_samples = 0;
        self.sum_errors = 0.0;
        self.sum_errors_sq = 0.0;
        self.min_p_s = f64::MAX;
        self.min_p = f64::MAX;
        self.min_s = f64::MAX;
        self.min_sample = 0;
        self.state = DriftState::NoDrift;
        self.drift_detected = false;
        self.in_warning = false;
        self.drift_count = 0;
    }

    fn drift_detected(&self) -> bool {
        self.drift_detected
    }

    fn samples_seen(&self) -> u64 {
        self.n_samples
    }
}

/// EDDM (Early Drift Detection Method)
///
/// Similar to DDM but uses distance between errors instead of error rate.
/// More sensitive to gradual concept drifts.
#[derive(Debug, Clone)]
pub struct EDDM {
    /// Minimum samples before detection
    min_samples: u64,
    /// Number of samples seen
    n_samples: u64,
    /// Number of errors seen
    n_errors: u64,
    /// Last error position
    last_error_pos: u64,
    /// Sum of distances between errors
    sum_distances: f64,
    /// Sum of squared distances
    sum_distances_sq: f64,
    /// Maximum p' + s' seen
    max_p_s: f64,
    /// Warning threshold
    warning_threshold: f64,
    /// Drift threshold
    drift_threshold: f64,
    /// Current state
    state: DriftState,
    /// Drift detected flag
    drift_detected: bool,
}

impl Default for EDDM {
    fn default() -> Self {
        Self::new(30)
    }
}

impl EDDM {
    /// Create new EDDM detector
    pub fn new(min_samples: u64) -> Self {
        Self {
            min_samples,
            n_samples: 0,
            n_errors: 0,
            last_error_pos: 0,
            sum_distances: 0.0,
            sum_distances_sq: 0.0,
            max_p_s: 0.0,
            warning_threshold: 0.95,
            drift_threshold: 0.9,
            state: DriftState::NoDrift,
            drift_detected: false,
        }
    }

    /// Set thresholds
    pub fn with_thresholds(mut self, warning: f64, drift: f64) -> Self {
        self.warning_threshold = warning;
        self.drift_threshold = drift;
        self
    }

    /// Get current state
    pub fn state(&self) -> DriftState {
        self.state
    }

    /// Update with error (1 = error, 0 = correct)
    pub fn add_element(&mut self, is_error: bool) -> DriftState {
        self.drift_detected = false;
        self.n_samples += 1;

        if is_error {
            if self.n_errors > 0 {
                let distance = (self.n_samples - self.last_error_pos) as f64;
                self.sum_distances += distance;
                self.sum_distances_sq += distance * distance;
            }

            self.n_errors += 1;
            self.last_error_pos = self.n_samples;

            if self.n_errors < self.min_samples as u64 {
                return DriftState::NoDrift;
            }

            // Compute mean and std of distances
            let n = self.n_errors as f64;
            let p = self.sum_distances / n;
            let variance = (self.sum_distances_sq / n) - (p * p);
            let s = if variance > 0.0 { variance.sqrt() } else { 0.0 };
            let p_s = p + 2.0 * s;

            // Update maximum
            if p_s > self.max_p_s {
                self.max_p_s = p_s;
            }

            // Check for drift
            let ratio = p_s / self.max_p_s;

            if ratio < self.drift_threshold {
                self.state = DriftState::Drift;
                self.drift_detected = true;
                self.reset();
            } else if ratio < self.warning_threshold {
                self.state = DriftState::Warning;
            } else {
                self.state = DriftState::NoDrift;
            }
        }

        self.state
    }
}

impl DriftDetector for EDDM {
    fn update(&mut self, value: f64) -> bool {
        // Treat value > 0.5 as error
        self.add_element(value > 0.5);
        self.drift_detected
    }

    fn reset(&mut self) {
        self.n_samples = 0;
        self.n_errors = 0;
        self.last_error_pos = 0;
        self.sum_distances = 0.0;
        self.sum_distances_sq = 0.0;
        self.max_p_s = 0.0;
        self.state = DriftState::NoDrift;
        self.drift_detected = false;
    }

    fn drift_detected(&self) -> bool {
        self.drift_detected
    }

    fn samples_seen(&self) -> u64 {
        self.n_samples
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ddm_no_drift() {
        let mut ddm = DDM::new(30);

        // Low error rate
        for _ in 0..100 {
            ddm.add_element(0.0); // No error
        }

        assert_eq!(ddm.state(), DriftState::NoDrift);
    }

    #[test]
    fn test_ddm_with_drift() {
        let mut ddm = DDM::new(10);

        // Low error phase
        for _ in 0..50 {
            ddm.add_element(0.1);
        }

        // High error phase - should trigger drift
        let mut drift_detected = false;
        for _ in 0..50 {
            if ddm.add_element(0.9) == DriftState::Drift {
                drift_detected = true;
            }
        }

        assert!(drift_detected || ddm.drift_count() > 0);
    }

    #[test]
    fn test_eddm_creation() {
        let eddm = EDDM::new(30);
        assert_eq!(eddm.state(), DriftState::NoDrift);
        assert_eq!(eddm.samples_seen(), 0);
    }

    #[test]
    fn test_ddm_reset() {
        let mut ddm = DDM::new(10);

        for _ in 0..50 {
            ddm.add_element(0.5);
        }

        ddm.reset();

        assert_eq!(ddm.samples_seen(), 0);
        assert_eq!(ddm.error_rate(), 0.0);
    }
}
