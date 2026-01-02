//! Online Energy Estimation for real-time anomaly detection
//!
//! This module provides streaming energy estimation that can be
//! updated incrementally without retraining on full dataset.

use ndarray::{Array1, Array2};
use std::collections::VecDeque;

/// Online energy estimator using running statistics
///
/// Maintains a sliding window of recent observations and
/// computes energy based on deviation from running statistics.
#[derive(Debug, Clone)]
pub struct OnlineEnergyEstimator {
    /// Sliding window of recent observations
    buffer: VecDeque<Array1<f64>>,
    /// Maximum buffer size
    max_size: usize,
    /// Running mean
    mean: Option<Array1<f64>>,
    /// Running variance
    variance: Option<Array1<f64>>,
    /// Number of observations processed
    count: usize,
    /// Decay factor for exponential moving average (0 < alpha <= 1)
    alpha: f64,
}

impl OnlineEnergyEstimator {
    /// Create a new online estimator
    ///
    /// # Arguments
    /// * `window_size` - Size of sliding window
    /// * `alpha` - Exponential decay factor (higher = more weight to recent data)
    pub fn new(window_size: usize, alpha: f64) -> Self {
        Self {
            buffer: VecDeque::with_capacity(window_size),
            max_size: window_size,
            mean: None,
            variance: None,
            count: 0,
            alpha: alpha.clamp(0.01, 1.0),
        }
    }

    /// Create with default parameters
    pub fn default_params() -> Self {
        Self::new(100, 0.1)
    }

    /// Update the estimator with a new observation
    pub fn update(&mut self, x: &Array1<f64>) -> EnergyResult {
        let dim = x.len();

        // Initialize statistics if needed
        if self.mean.is_none() {
            self.mean = Some(x.clone());
            self.variance = Some(Array1::ones(dim));
        }

        let mean = self.mean.as_mut().unwrap();
        let variance = self.variance.as_mut().unwrap();

        // Compute energy before update
        let energy = self.compute_energy(x);
        let normalized_energy = self.normalize_energy(energy);

        // Update running statistics (exponential moving average)
        for i in 0..dim {
            let delta = x[i] - mean[i];
            mean[i] += self.alpha * delta;

            // Update variance with Welford's online algorithm
            let new_delta = x[i] - mean[i];
            variance[i] = (1.0 - self.alpha) * variance[i] + self.alpha * delta * new_delta;
        }

        // Update buffer
        if self.buffer.len() >= self.max_size {
            self.buffer.pop_front();
        }
        self.buffer.push_back(x.clone());
        self.count += 1;

        EnergyResult {
            energy,
            normalized_energy,
            is_anomaly: normalized_energy > 2.0, // 2 standard deviations
            regime: self.classify_regime(normalized_energy),
        }
    }

    /// Compute energy for a point using current statistics
    fn compute_energy(&self, x: &Array1<f64>) -> f64 {
        match (&self.mean, &self.variance) {
            (Some(mean), Some(variance)) => {
                // Mahalanobis-like distance (assuming diagonal covariance)
                let mut energy = 0.0;
                for i in 0..x.len() {
                    let std = variance[i].sqrt().max(1e-10);
                    let z = (x[i] - mean[i]) / std;
                    energy += z * z;
                }
                energy / x.len() as f64
            }
            _ => 0.0,
        }
    }

    /// Normalize energy to standard deviation units
    fn normalize_energy(&self, energy: f64) -> f64 {
        if self.count < 10 {
            return 0.0;
        }

        // Compute mean and std of energies in buffer
        let energies: Vec<f64> = self
            .buffer
            .iter()
            .map(|x| self.compute_energy(x))
            .collect();

        let mean_energy: f64 = energies.iter().sum::<f64>() / energies.len() as f64;
        let var_energy: f64 = energies
            .iter()
            .map(|e| (e - mean_energy).powi(2))
            .sum::<f64>()
            / energies.len() as f64;
        let std_energy = var_energy.sqrt().max(1e-10);

        (energy - mean_energy) / std_energy
    }

    /// Classify market regime based on normalized energy
    fn classify_regime(&self, normalized_energy: f64) -> MarketRegime {
        if normalized_energy < -1.0 {
            MarketRegime::Calm
        } else if normalized_energy < 1.0 {
            MarketRegime::Normal
        } else if normalized_energy < 2.0 {
            MarketRegime::Elevated
        } else {
            MarketRegime::Crisis
        }
    }

    /// Get current energy statistics
    pub fn get_stats(&self) -> OnlineStats {
        let energies: Vec<f64> = self
            .buffer
            .iter()
            .map(|x| self.compute_energy(x))
            .collect();

        if energies.is_empty() {
            return OnlineStats::default();
        }

        let mean = energies.iter().sum::<f64>() / energies.len() as f64;
        let variance =
            energies.iter().map(|e| (e - mean).powi(2)).sum::<f64>() / energies.len() as f64;
        let std = variance.sqrt();

        let min = energies.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = energies.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        OnlineStats {
            mean,
            std,
            min,
            max,
            count: self.count,
            buffer_size: self.buffer.len(),
        }
    }

    /// Reset the estimator
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.mean = None;
        self.variance = None;
        self.count = 0;
    }

    /// Get recommended position scale (0-1)
    pub fn position_scale(&self, energy: f64) -> f64 {
        let normalized = self.normalize_energy(energy);

        // Reduce position as energy increases
        if normalized < 0.0 {
            1.0 // Low energy = full position
        } else if normalized < 1.0 {
            1.0 - 0.2 * normalized // Slight reduction
        } else if normalized < 2.0 {
            0.8 - 0.3 * (normalized - 1.0) // More reduction
        } else {
            (0.5 - 0.2 * (normalized - 2.0)).max(0.0) // Heavy reduction
        }
    }
}

/// Result of energy estimation
#[derive(Debug, Clone)]
pub struct EnergyResult {
    /// Raw energy value
    pub energy: f64,
    /// Normalized energy (in standard deviations)
    pub normalized_energy: f64,
    /// Whether this is an anomaly
    pub is_anomaly: bool,
    /// Detected market regime
    pub regime: MarketRegime,
}

/// Market regime classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarketRegime {
    /// Very calm market (low energy)
    Calm,
    /// Normal market conditions
    Normal,
    /// Elevated volatility/unusual activity
    Elevated,
    /// Crisis mode (very high energy)
    Crisis,
}

impl MarketRegime {
    /// Get string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            MarketRegime::Calm => "Calm",
            MarketRegime::Normal => "Normal",
            MarketRegime::Elevated => "Elevated",
            MarketRegime::Crisis => "Crisis",
        }
    }

    /// Get recommended risk multiplier
    pub fn risk_multiplier(&self) -> f64 {
        match self {
            MarketRegime::Calm => 1.2,    // Can take more risk
            MarketRegime::Normal => 1.0,  // Normal risk
            MarketRegime::Elevated => 0.5, // Reduce risk
            MarketRegime::Crisis => 0.1,  // Minimal risk
        }
    }
}

/// Online statistics
#[derive(Debug, Clone, Default)]
pub struct OnlineStats {
    /// Mean energy
    pub mean: f64,
    /// Standard deviation of energy
    pub std: f64,
    /// Minimum energy
    pub min: f64,
    /// Maximum energy
    pub max: f64,
    /// Total observations processed
    pub count: usize,
    /// Current buffer size
    pub buffer_size: usize,
}

/// Adaptive energy threshold based on recent history
#[derive(Debug, Clone)]
pub struct AdaptiveThreshold {
    /// Base threshold in standard deviations
    base_threshold: f64,
    /// Recent energy values
    energy_history: VecDeque<f64>,
    /// History size
    history_size: usize,
    /// Adaptation rate
    adaptation_rate: f64,
}

impl AdaptiveThreshold {
    /// Create a new adaptive threshold
    pub fn new(base_threshold: f64, history_size: usize, adaptation_rate: f64) -> Self {
        Self {
            base_threshold,
            energy_history: VecDeque::with_capacity(history_size),
            history_size,
            adaptation_rate,
        }
    }

    /// Update threshold with new energy observation
    pub fn update(&mut self, energy: f64) {
        if self.energy_history.len() >= self.history_size {
            self.energy_history.pop_front();
        }
        self.energy_history.push_back(energy);
    }

    /// Get current adaptive threshold
    pub fn threshold(&self) -> f64 {
        if self.energy_history.len() < 10 {
            return self.base_threshold;
        }

        // Compute recent volatility of energy
        let energies: Vec<f64> = self.energy_history.iter().cloned().collect();
        let mean = energies.iter().sum::<f64>() / energies.len() as f64;
        let variance =
            energies.iter().map(|e| (e - mean).powi(2)).sum::<f64>() / energies.len() as f64;
        let energy_volatility = variance.sqrt();

        // Adapt threshold based on energy volatility
        // Higher volatility → higher threshold (fewer false alarms)
        self.base_threshold * (1.0 + self.adaptation_rate * energy_volatility)
    }

    /// Check if energy exceeds adaptive threshold
    pub fn is_anomaly(&self, normalized_energy: f64) -> bool {
        normalized_energy > self.threshold()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_online_estimator() {
        let mut estimator = OnlineEnergyEstimator::new(50, 0.1);

        // Feed some normal data
        for i in 0..100 {
            let x = array![i as f64 * 0.01, (i as f64 * 0.02).sin()];
            let result = estimator.update(&x);
            assert!(result.energy.is_finite());
        }

        let stats = estimator.get_stats();
        assert_eq!(stats.buffer_size, 50);
        assert_eq!(stats.count, 100);
    }

    #[test]
    fn test_market_regime() {
        assert_eq!(MarketRegime::Calm.risk_multiplier(), 1.2);
        assert_eq!(MarketRegime::Crisis.risk_multiplier(), 0.1);
    }

    #[test]
    fn test_adaptive_threshold() {
        let mut threshold = AdaptiveThreshold::new(2.0, 100, 0.5);

        // Feed low-volatility energies
        for _ in 0..50 {
            threshold.update(1.0);
        }

        let t1 = threshold.threshold();

        // Feed high-volatility energies
        for i in 0..50 {
            threshold.update(1.0 + (i as f64 % 3.0));
        }

        let t2 = threshold.threshold();

        // Threshold should increase with higher volatility
        assert!(t2 > t1);
    }
}
