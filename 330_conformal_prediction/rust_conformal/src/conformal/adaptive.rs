//! Adaptive Conformal Inference for time series

use super::predictor::{SplitConformalPredictor, PredictionInterval};

/// Adaptive Conformal Inference (ACI)
///
/// Adjusts the miscoverage level alpha online to maintain coverage
/// under distribution shift in time series data.
#[derive(Debug, Clone)]
pub struct AdaptiveConformalPredictor {
    /// Base conformal predictor
    base_predictor: SplitConformalPredictor,
    /// Target miscoverage rate
    target_alpha: f64,
    /// Current adaptive alpha
    alpha_t: f64,
    /// Learning rate for alpha updates
    gamma: f64,
    /// Window size for rolling calibration
    window_size: usize,
    /// Coverage history
    coverage_history: Vec<bool>,
    /// Alpha history
    alpha_history: Vec<f64>,
}

impl AdaptiveConformalPredictor {
    /// Create new adaptive conformal predictor
    pub fn new(target_alpha: f64, gamma: f64, window_size: usize) -> Self {
        Self {
            base_predictor: SplitConformalPredictor::new(target_alpha),
            target_alpha,
            alpha_t: target_alpha,
            gamma: gamma.clamp(0.001, 0.1),
            window_size,
            coverage_history: Vec::new(),
            alpha_history: vec![target_alpha],
        }
    }

    /// Get current adaptive alpha
    pub fn current_alpha(&self) -> f64 {
        self.alpha_t
    }

    /// Get target alpha
    pub fn target_alpha(&self) -> f64 {
        self.target_alpha
    }

    /// Get coverage history
    pub fn coverage_history(&self) -> &[bool] {
        &self.coverage_history
    }

    /// Get alpha history
    pub fn alpha_history(&self) -> &[f64] {
        &self.alpha_history
    }

    /// Update alpha based on observed coverage
    ///
    /// If the true value was not covered, increase alpha (wider intervals)
    /// If the true value was covered, decrease alpha (narrower intervals)
    pub fn update(&mut self, y_true: f64, interval: &PredictionInterval) {
        let covered = interval.covers(y_true);
        self.coverage_history.push(covered);

        // Update alpha: error = (1 - covered) - target_alpha
        // If not covered (error positive), increase alpha
        // If covered (error negative), decrease alpha
        let error = if covered { 0.0 } else { 1.0 } - self.target_alpha;
        self.alpha_t = (self.alpha_t + self.gamma * error).clamp(0.01, 0.5);

        self.alpha_history.push(self.alpha_t);
    }

    /// Get rolling coverage rate over recent window
    pub fn rolling_coverage(&self, window: usize) -> f64 {
        if self.coverage_history.is_empty() {
            return 0.5;
        }

        let start = if self.coverage_history.len() > window {
            self.coverage_history.len() - window
        } else {
            0
        };

        let recent = &self.coverage_history[start..];
        let covered_count = recent.iter().filter(|&&c| c).count();
        covered_count as f64 / recent.len() as f64
    }

    /// Calibrate base predictor with calibration scores
    pub fn calibrate(&mut self, scores: Vec<f64>) {
        self.base_predictor.calibrate_from_scores(scores);
    }

    /// Get adjusted quantile for current alpha_t
    pub fn adjusted_quantile(&self) -> Option<f64> {
        // Scale the base quantile by alpha ratio
        let base_quantile = self.base_predictor.quantile()?;
        let alpha_ratio = self.alpha_t / self.target_alpha;

        // Adjust quantile: smaller alpha_t -> larger quantile (wider intervals)
        Some(base_quantile * (2.0 - alpha_ratio).max(0.5))
    }

    /// Generate prediction interval with adaptive alpha
    pub fn predict_interval(&self, point_pred: f64) -> Option<PredictionInterval> {
        let q = self.adjusted_quantile()?;

        Some(PredictionInterval::new(
            point_pred,
            point_pred - q,
            point_pred + q,
            self.alpha_t,
        ))
    }

    /// Rolling recalibration using recent history
    pub fn rolling_calibrate(&mut self, recent_scores: &[f64]) {
        if recent_scores.len() < self.window_size / 2 {
            return;
        }

        // Use most recent scores for calibration
        let start = if recent_scores.len() > self.window_size {
            recent_scores.len() - self.window_size
        } else {
            0
        };

        let calibration_scores = recent_scores[start..].to_vec();
        self.calibrate(calibration_scores);
    }

    /// Reset coverage history
    pub fn reset_history(&mut self) {
        self.coverage_history.clear();
        self.alpha_history.clear();
        self.alpha_history.push(self.target_alpha);
        self.alpha_t = self.target_alpha;
    }
}

/// Statistics about adaptive behavior
#[derive(Debug, Clone)]
pub struct AdaptiveStats {
    /// Mean alpha over time
    pub mean_alpha: f64,
    /// Std of alpha
    pub std_alpha: f64,
    /// Overall coverage
    pub overall_coverage: f64,
    /// Number of updates
    pub num_updates: usize,
}

impl AdaptiveConformalPredictor {
    /// Compute statistics about adaptive behavior
    pub fn compute_stats(&self) -> AdaptiveStats {
        let alphas = &self.alpha_history;
        let n = alphas.len() as f64;

        let mean_alpha = alphas.iter().sum::<f64>() / n;
        let variance = alphas.iter().map(|a| (a - mean_alpha).powi(2)).sum::<f64>() / n;
        let std_alpha = variance.sqrt();

        let overall_coverage = if self.coverage_history.is_empty() {
            0.0
        } else {
            let covered = self.coverage_history.iter().filter(|&&c| c).count();
            covered as f64 / self.coverage_history.len() as f64
        };

        AdaptiveStats {
            mean_alpha,
            std_alpha,
            overall_coverage,
            num_updates: self.coverage_history.len(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_update() {
        let mut predictor = AdaptiveConformalPredictor::new(0.1, 0.01, 100);
        predictor.calibrate(vec![0.1, 0.2, 0.3, 0.4, 0.5]);

        let initial_alpha = predictor.current_alpha();

        // Simulate not covered
        let interval = PredictionInterval::new(5.0, 4.5, 5.5, 0.1);
        predictor.update(6.0, &interval); // Not covered

        assert!(predictor.current_alpha() > initial_alpha);
    }

    #[test]
    fn test_rolling_coverage() {
        let mut predictor = AdaptiveConformalPredictor::new(0.1, 0.01, 100);

        // Add some coverage history
        for _ in 0..9 {
            predictor.coverage_history.push(true);
        }
        predictor.coverage_history.push(false);

        let coverage = predictor.rolling_coverage(10);
        assert!((coverage - 0.9).abs() < 1e-10);
    }
}
