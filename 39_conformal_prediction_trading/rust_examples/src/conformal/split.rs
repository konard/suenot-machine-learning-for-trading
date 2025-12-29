//! Split Conformal Prediction
//!
//! The simplest and most practical conformal prediction method.
//! Provides prediction intervals with guaranteed coverage.

use super::model::Model;
use super::PredictionInterval;
use crate::data::processor::DataProcessor;
use ndarray::Array2;

/// Split Conformal Predictor
///
/// Provides calibrated prediction intervals with guaranteed coverage.
///
/// # Coverage Guarantee
///
/// For exchangeable data, if we target coverage level 1-α, then:
/// P(Y ∈ [lower, upper]) ≥ 1 - α
///
/// This guarantee holds in finite samples, not just asymptotically.
#[derive(Debug)]
pub struct SplitConformalPredictor<M: Model> {
    /// Underlying prediction model
    model: M,
    /// Target miscoverage rate (1 - coverage)
    alpha: f64,
    /// Calibrated quantile of nonconformity scores
    q_hat: Option<f64>,
    /// Stored calibration scores
    calibration_scores: Vec<f64>,
}

impl<M: Model> SplitConformalPredictor<M> {
    /// Create a new Split Conformal Predictor
    ///
    /// # Arguments
    /// * `model` - The underlying prediction model
    /// * `alpha` - Miscoverage rate (e.g., 0.1 for 90% coverage)
    pub fn new(model: M, alpha: f64) -> Self {
        Self {
            model,
            alpha: alpha.max(0.01).min(0.5), // Clip to reasonable range
            q_hat: None,
            calibration_scores: Vec::new(),
        }
    }

    /// Fit the predictor on training data and calibrate on calibration data
    ///
    /// # Arguments
    /// * `x_train` - Training features
    /// * `y_train` - Training targets
    /// * `x_calib` - Calibration features (held out from training)
    /// * `y_calib` - Calibration targets
    pub fn fit(
        &mut self,
        x_train: &Array2<f64>,
        y_train: &[f64],
        x_calib: &Array2<f64>,
        y_calib: &[f64],
    ) {
        // Step 1: Train underlying model
        self.model.fit(x_train, y_train);

        // Step 2: Get predictions on calibration set
        let y_pred_calib = self.model.predict(x_calib);

        // Step 3: Compute nonconformity scores (absolute residuals)
        self.calibration_scores = y_calib
            .iter()
            .zip(y_pred_calib.iter())
            .map(|(&y_true, &y_pred)| (y_true - y_pred).abs())
            .collect();

        // Step 4: Compute quantile for prediction intervals
        // The ceil((n+1)(1-α))/n quantile ensures finite-sample coverage
        let n = self.calibration_scores.len() as f64;
        let q_level = ((n + 1.0) * (1.0 - self.alpha)).ceil() / n;
        let q_level = q_level.min(1.0);

        self.q_hat = Some(DataProcessor::quantile(&self.calibration_scores, q_level));
    }

    /// Predict with prediction intervals
    ///
    /// Returns a vector of PredictionInterval for each input sample
    pub fn predict(&self, x: &Array2<f64>) -> Vec<PredictionInterval> {
        let q_hat = self.q_hat.unwrap_or(0.0);
        let predictions = self.model.predict(x);

        predictions
            .into_iter()
            .map(|pred| PredictionInterval::new(pred, pred - q_hat, pred + q_hat))
            .collect()
    }

    /// Predict for a single sample
    pub fn predict_one(&self, x: &[f64]) -> PredictionInterval {
        let q_hat = self.q_hat.unwrap_or(0.0);
        let pred = self.model.predict_one(x);

        PredictionInterval::new(pred, pred - q_hat, pred + q_hat)
    }

    /// Get the current interval width (2 * q_hat)
    pub fn interval_width(&self) -> f64 {
        self.q_hat.unwrap_or(0.0) * 2.0
    }

    /// Compute empirical coverage on test data
    pub fn coverage(&self, x_test: &Array2<f64>, y_test: &[f64]) -> f64 {
        let intervals = self.predict(x_test);

        let n_covered = intervals
            .iter()
            .zip(y_test.iter())
            .filter(|(interval, &y)| interval.covers(y))
            .count();

        n_covered as f64 / y_test.len() as f64
    }

    /// Get the calibration quantile
    pub fn quantile(&self) -> Option<f64> {
        self.q_hat
    }

    /// Get reference to calibration scores
    pub fn calibration_scores(&self) -> &[f64] {
        &self.calibration_scores
    }

    /// Get the alpha (miscoverage rate)
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Get the target coverage (1 - alpha)
    pub fn target_coverage(&self) -> f64 {
        1.0 - self.alpha
    }
}

/// Conformalized Quantile Regression (simplified version)
///
/// Produces adaptive intervals that vary in width based on input features.
/// More informative than split conformal for heteroscedastic data.
#[derive(Debug)]
pub struct ConformizedQuantilePredictor<M: Model> {
    /// Model for lower quantile
    lower_model: M,
    /// Model for upper quantile
    upper_model: M,
    /// Model for point prediction
    point_model: M,
    /// Target miscoverage rate
    alpha: f64,
    /// Calibration adjustment
    q_hat: Option<f64>,
}

impl<M: Model + Clone> ConformizedQuantilePredictor<M> {
    /// Create a new CQR predictor
    pub fn new(model_template: M, alpha: f64) -> Self {
        Self {
            lower_model: model_template.clone(),
            upper_model: model_template.clone(),
            point_model: model_template,
            alpha: alpha.max(0.01).min(0.5),
            q_hat: None,
        }
    }

    /// Fit the CQR predictor
    ///
    /// Note: This is a simplified implementation that approximates CQR
    /// by fitting separate models with adjusted targets.
    pub fn fit(
        &mut self,
        x_train: &Array2<f64>,
        y_train: &[f64],
        x_calib: &Array2<f64>,
        y_calib: &[f64],
    ) {
        // Fit point prediction model
        self.point_model.fit(x_train, y_train);

        // Get residuals on training data for quantile estimation
        let train_preds = self.point_model.predict(x_train);
        let train_residuals: Vec<f64> = y_train
            .iter()
            .zip(train_preds.iter())
            .map(|(&y, &pred)| y - pred)
            .collect();

        // Compute quantiles of residuals
        let lower_quantile = DataProcessor::quantile(&train_residuals, self.alpha / 2.0);
        let upper_quantile = DataProcessor::quantile(&train_residuals, 1.0 - self.alpha / 2.0);

        // Create adjusted targets for quantile models
        let y_lower: Vec<f64> = y_train.iter().map(|&y| y - lower_quantile).collect();
        let y_upper: Vec<f64> = y_train.iter().map(|&y| y - upper_quantile).collect();

        // Fit quantile models
        self.lower_model.fit(x_train, &y_lower);
        self.upper_model.fit(x_train, &y_upper);

        // Calibrate on calibration set
        let lower_calib = self.lower_model.predict(x_calib);
        let upper_calib = self.upper_model.predict(x_calib);

        // Compute conformity scores
        let scores: Vec<f64> = y_calib
            .iter()
            .zip(lower_calib.iter().zip(upper_calib.iter()))
            .map(|(&y, (&lower, &upper))| {
                (lower + lower_quantile - y).max(y - upper - upper_quantile)
            })
            .collect();

        // Compute calibration quantile
        let n = scores.len() as f64;
        let q_level = ((n + 1.0) * (1.0 - self.alpha)).ceil() / n;
        let q_level = q_level.min(1.0);

        self.q_hat = Some(DataProcessor::quantile(&scores, q_level));
    }

    /// Predict with adaptive intervals
    pub fn predict(&self, x: &Array2<f64>) -> Vec<PredictionInterval> {
        let q_hat = self.q_hat.unwrap_or(0.0);
        let point_preds = self.point_model.predict(x);
        let lower_preds = self.lower_model.predict(x);
        let upper_preds = self.upper_model.predict(x);

        point_preds
            .into_iter()
            .zip(lower_preds.into_iter().zip(upper_preds.into_iter()))
            .map(|(pred, (lower, upper))| {
                PredictionInterval::new(pred, lower - q_hat, upper + q_hat)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conformal::model::LinearModel;
    use ndarray::array;

    #[test]
    fn test_split_conformal_basic() {
        // Create synthetic data: y = 2*x + noise
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
        let y_train: Vec<f64> = (1..=10).map(|x| 2.0 * x as f64 + 0.1).collect();

        let x_calib = array![[11.0], [12.0], [13.0], [14.0], [15.0]];
        let y_calib: Vec<f64> = (11..=15).map(|x| 2.0 * x as f64 + 0.1).collect();

        let model = LinearModel::new(true);
        let mut cp = SplitConformalPredictor::new(model, 0.1);
        cp.fit(&x_train, &y_train, &x_calib, &y_calib);

        // Check that q_hat is set
        assert!(cp.quantile().is_some());

        // Check predictions
        let x_test = array![[16.0], [17.0]];
        let intervals = cp.predict(&x_test);

        assert_eq!(intervals.len(), 2);

        // Each interval should have positive width
        for interval in &intervals {
            assert!(interval.width > 0.0);
        }
    }

    #[test]
    fn test_split_conformal_coverage() {
        // Create data where we know the coverage should be high
        let n_train = 100;
        let n_calib = 50;
        let n_test = 50;

        let mut x_train = Array2::zeros((n_train, 1));
        let mut y_train = Vec::with_capacity(n_train);

        for i in 0..n_train {
            let x = i as f64 / 10.0;
            x_train[[i, 0]] = x;
            y_train.push(x * 2.0);
        }

        let mut x_calib = Array2::zeros((n_calib, 1));
        let mut y_calib = Vec::with_capacity(n_calib);

        for i in 0..n_calib {
            let x = (n_train + i) as f64 / 10.0;
            x_calib[[i, 0]] = x;
            y_calib.push(x * 2.0);
        }

        let mut x_test = Array2::zeros((n_test, 1));
        let mut y_test = Vec::with_capacity(n_test);

        for i in 0..n_test {
            let x = (n_train + n_calib + i) as f64 / 10.0;
            x_test[[i, 0]] = x;
            y_test.push(x * 2.0);
        }

        let model = LinearModel::new(true);
        let mut cp = SplitConformalPredictor::new(model, 0.1);
        cp.fit(&x_train, &y_train, &x_calib, &y_calib);

        let coverage = cp.coverage(&x_test, &y_test);

        // Coverage should be close to target (90%)
        // Allow some tolerance due to finite sample effects
        assert!(coverage >= 0.7, "Coverage {} should be >= 0.7", coverage);
    }

    #[test]
    fn test_prediction_interval() {
        let interval = PredictionInterval::new(0.0, -1.0, 1.0);

        assert!(interval.covers(0.0));
        assert!(interval.covers(0.5));
        assert!(interval.covers(-0.5));
        assert!(!interval.covers(1.5));
        assert!(!interval.covers(-1.5));

        assert!(interval.crosses_zero());
        assert!(!interval.is_positive());
        assert!(!interval.is_negative());
    }
}
