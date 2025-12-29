//! Adaptive Conformal Inference for Time Series
//!
//! Standard conformal prediction assumes exchangeability, which is violated
//! in time series. Adaptive Conformal Inference (ACI) addresses this by
//! dynamically adjusting the coverage level based on recent performance.

use super::model::Model;
use super::PredictionInterval;
use crate::data::processor::DataProcessor;
use ndarray::Array2;

/// Adaptive Conformal Predictor for Time Series
///
/// Dynamically adjusts coverage level based on recent errors,
/// maintaining approximate coverage under distribution shift.
///
/// # Reference
/// Gibbs & Candès (2021) "Adaptive Conformal Inference Under Distribution Shift"
#[derive(Debug)]
pub struct AdaptiveConformalPredictor<M: Model> {
    /// Underlying prediction model
    model: M,
    /// Target coverage level
    target_coverage: f64,
    /// Learning rate for adaptation
    gamma: f64,
    /// Current miscoverage rate (adjusted over time)
    alpha_t: f64,
    /// History of coverage outcomes (true = covered)
    history: Vec<bool>,
    /// Rolling window of calibration scores
    calibration_scores: Vec<f64>,
    /// Maximum number of scores to keep
    max_scores: usize,
}

impl<M: Model> AdaptiveConformalPredictor<M> {
    /// Create a new Adaptive Conformal Predictor
    ///
    /// # Arguments
    /// * `model` - The underlying prediction model
    /// * `target_coverage` - Target coverage level (e.g., 0.9 for 90%)
    /// * `gamma` - Learning rate for adaptation (e.g., 0.05)
    pub fn new(model: M, target_coverage: f64, gamma: f64) -> Self {
        let target_coverage = target_coverage.max(0.5).min(0.99);
        Self {
            model,
            target_coverage,
            gamma: gamma.max(0.001).min(0.5),
            alpha_t: 1.0 - target_coverage,
            history: Vec::new(),
            calibration_scores: Vec::new(),
            max_scores: 500,
        }
    }

    /// Initial training and calibration
    pub fn fit(
        &mut self,
        x_train: &Array2<f64>,
        y_train: &[f64],
        x_calib: &Array2<f64>,
        y_calib: &[f64],
    ) {
        // Train underlying model
        self.model.fit(x_train, y_train);

        // Compute calibration scores
        let y_pred_calib = self.model.predict(x_calib);
        self.calibration_scores = y_calib
            .iter()
            .zip(y_pred_calib.iter())
            .map(|(&y_true, &y_pred)| (y_true - y_pred).abs())
            .collect();

        // Reset history
        self.history.clear();
        self.alpha_t = 1.0 - self.target_coverage;
    }

    /// Update the predictor with new observation
    ///
    /// This should be called after each prediction when the true value is known.
    pub fn update(&mut self, y_true: f64, prediction: &PredictionInterval) {
        let covered = prediction.covers(y_true);
        self.history.push(covered);

        // Update alpha using gradient-like update
        // If covered more than target: decrease alpha (narrower intervals)
        // If covered less than target: increase alpha (wider intervals)
        if covered {
            // Covered -> can afford narrower intervals
            self.alpha_t += self.gamma * (self.alpha_t - 0.0);
        } else {
            // Not covered -> need wider intervals
            self.alpha_t += self.gamma * (self.alpha_t - 1.0);
        }

        // Clip to valid range
        self.alpha_t = self.alpha_t.max(0.001).min(0.5);

        // Add new calibration score
        let score = (y_true - prediction.prediction).abs();
        self.calibration_scores.push(score);

        // Limit size of calibration scores
        if self.calibration_scores.len() > self.max_scores {
            self.calibration_scores.remove(0);
        }
    }

    /// Update with new data point (adds to calibration)
    pub fn update_calibration(&mut self, x: &[f64], y_true: f64) {
        let y_pred = self.model.predict_one(x);
        let score = (y_true - y_pred).abs();

        self.calibration_scores.push(score);
        if self.calibration_scores.len() > self.max_scores {
            self.calibration_scores.remove(0);
        }
    }

    /// Predict with adaptive interval
    pub fn predict_one(&self, x: &[f64]) -> PredictionInterval {
        let y_pred = self.model.predict_one(x);

        // Compute interval width based on current alpha
        let q_level = (1.0 - self.alpha_t).min(1.0);
        let q_hat = DataProcessor::quantile(&self.calibration_scores, q_level);

        PredictionInterval::new(y_pred, y_pred - q_hat, y_pred + q_hat)
    }

    /// Predict multiple samples
    pub fn predict(&self, x: &Array2<f64>) -> Vec<PredictionInterval> {
        let q_level = (1.0 - self.alpha_t).min(1.0);
        let q_hat = DataProcessor::quantile(&self.calibration_scores, q_level);

        let predictions = self.model.predict(x);

        predictions
            .into_iter()
            .map(|pred| PredictionInterval::new(pred, pred - q_hat, pred + q_hat))
            .collect()
    }

    /// Get current alpha (miscoverage rate)
    pub fn current_alpha(&self) -> f64 {
        self.alpha_t
    }

    /// Get current coverage level
    pub fn current_coverage_level(&self) -> f64 {
        1.0 - self.alpha_t
    }

    /// Get recent empirical coverage
    pub fn recent_coverage(&self, window: usize) -> Option<f64> {
        if self.history.is_empty() {
            return None;
        }

        let start = self.history.len().saturating_sub(window);
        let recent = &self.history[start..];

        let n_covered = recent.iter().filter(|&&c| c).count();
        Some(n_covered as f64 / recent.len() as f64)
    }

    /// Get target coverage
    pub fn target_coverage(&self) -> f64 {
        self.target_coverage
    }

    /// Get number of observations in history
    pub fn n_observations(&self) -> usize {
        self.history.len()
    }

    /// Get current interval width
    pub fn interval_width(&self) -> f64 {
        let q_level = (1.0 - self.alpha_t).min(1.0);
        DataProcessor::quantile(&self.calibration_scores, q_level) * 2.0
    }

    /// Retrain the underlying model with new data
    pub fn retrain(&mut self, x_train: &Array2<f64>, y_train: &[f64]) {
        self.model.fit(x_train, y_train);
    }
}

/// Rolling Conformal Predictor
///
/// Uses a rolling window of calibration data instead of adaptive alpha.
/// Simpler than ACI but less adaptive to distribution shift.
#[derive(Debug)]
pub struct RollingConformalPredictor<M: Model> {
    model: M,
    alpha: f64,
    calibration_window: usize,
    calibration_scores: Vec<f64>,
}

impl<M: Model> RollingConformalPredictor<M> {
    pub fn new(model: M, alpha: f64, calibration_window: usize) -> Self {
        Self {
            model,
            alpha: alpha.max(0.01).min(0.5),
            calibration_window,
            calibration_scores: Vec::new(),
        }
    }

    /// Initial fit
    pub fn fit(&mut self, x_train: &Array2<f64>, y_train: &[f64]) {
        self.model.fit(x_train, y_train);
        self.calibration_scores.clear();
    }

    /// Add calibration point
    pub fn add_calibration(&mut self, x: &[f64], y_true: f64) {
        let y_pred = self.model.predict_one(x);
        let score = (y_true - y_pred).abs();

        self.calibration_scores.push(score);

        // Keep only recent scores
        if self.calibration_scores.len() > self.calibration_window {
            self.calibration_scores.remove(0);
        }
    }

    /// Predict with interval
    pub fn predict_one(&self, x: &[f64]) -> PredictionInterval {
        let y_pred = self.model.predict_one(x);

        if self.calibration_scores.is_empty() {
            return PredictionInterval::new(y_pred, y_pred, y_pred);
        }

        let n = self.calibration_scores.len() as f64;
        let q_level = ((n + 1.0) * (1.0 - self.alpha)).ceil() / n;
        let q_level = q_level.min(1.0);
        let q_hat = DataProcessor::quantile(&self.calibration_scores, q_level);

        PredictionInterval::new(y_pred, y_pred - q_hat, y_pred + q_hat)
    }

    /// Get current interval width
    pub fn interval_width(&self) -> f64 {
        if self.calibration_scores.is_empty() {
            return 0.0;
        }

        let n = self.calibration_scores.len() as f64;
        let q_level = ((n + 1.0) * (1.0 - self.alpha)).ceil() / n;
        let q_level = q_level.min(1.0);
        DataProcessor::quantile(&self.calibration_scores, q_level) * 2.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conformal::model::LinearModel;
    use ndarray::array;

    #[test]
    fn test_adaptive_conformal_basic() {
        let x_train = array![
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
            [6.0],
            [7.0],
            [8.0],
            [9.0],
            [10.0]
        ];
        let y_train: Vec<f64> = (1..=10).map(|x| 2.0 * x as f64).collect();

        let x_calib = array![[11.0], [12.0], [13.0], [14.0], [15.0]];
        let y_calib: Vec<f64> = (11..=15).map(|x| 2.0 * x as f64).collect();

        let model = LinearModel::new(true);
        let mut acp = AdaptiveConformalPredictor::new(model, 0.9, 0.05);
        acp.fit(&x_train, &y_train, &x_calib, &y_calib);

        // Initial alpha should be close to 0.1
        assert!((acp.current_alpha() - 0.1).abs() < 0.01);

        // Make predictions
        let interval = acp.predict_one(&[16.0]);
        assert!(interval.width > 0.0);
    }

    #[test]
    fn test_adaptive_update() {
        let x_train = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let y_train: Vec<f64> = (1..=5).map(|x| 2.0 * x as f64).collect();

        let x_calib = array![[6.0], [7.0], [8.0]];
        let y_calib: Vec<f64> = (6..=8).map(|x| 2.0 * x as f64).collect();

        let model = LinearModel::new(true);
        let mut acp = AdaptiveConformalPredictor::new(model, 0.9, 0.1);
        acp.fit(&x_train, &y_train, &x_calib, &y_calib);

        let initial_alpha = acp.current_alpha();

        // Simulate covering (alpha should decrease)
        let interval = acp.predict_one(&[10.0]);
        acp.update(20.0, &interval); // This should be covered

        let after_cover = acp.current_alpha();

        // After covering, alpha should increase (move toward 0)
        // which means intervals get narrower
        assert!(after_cover >= initial_alpha * 0.8);
    }

    #[test]
    fn test_rolling_conformal() {
        let model = LinearModel::new(true);
        let mut rcp = RollingConformalPredictor::new(model, 0.1, 20);

        let x_train = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let y_train: Vec<f64> = (1..=5).map(|x| 2.0 * x as f64).collect();
        rcp.fit(&x_train, &y_train);

        // Add calibration points
        for i in 6..=15 {
            rcp.add_calibration(&[i as f64], 2.0 * i as f64);
        }

        // Make prediction
        let interval = rcp.predict_one(&[20.0]);
        assert!(interval.width > 0.0);
    }
}
