//! Split Conformal Predictor implementation

use ndarray::Array1;
use super::scores::{NonConformityScore, compute_quantile};

/// Prediction interval result
#[derive(Debug, Clone)]
pub struct PredictionInterval {
    /// Point prediction
    pub point: f64,
    /// Lower bound of interval
    pub lower: f64,
    /// Upper bound of interval
    pub upper: f64,
    /// Interval width
    pub width: f64,
    /// Confidence level (1 - alpha)
    pub confidence: f64,
}

impl PredictionInterval {
    /// Create new prediction interval
    pub fn new(point: f64, lower: f64, upper: f64, alpha: f64) -> Self {
        Self {
            point,
            lower,
            upper,
            width: upper - lower,
            confidence: 1.0 - alpha,
        }
    }

    /// Check if a value is covered by this interval
    pub fn covers(&self, value: f64) -> bool {
        value >= self.lower && value <= self.upper
    }
}

/// Split (Inductive) Conformal Predictor
///
/// Uses a held-out calibration set to compute non-conformity scores
/// and construct prediction intervals with guaranteed coverage.
#[derive(Debug, Clone)]
pub struct SplitConformalPredictor {
    /// Miscoverage rate (e.g., 0.1 for 90% coverage)
    alpha: f64,
    /// Non-conformity score function
    score_type: NonConformityScore,
    /// Calibration scores
    calibration_scores: Option<Vec<f64>>,
    /// Computed quantile for prediction
    quantile: Option<f64>,
}

impl SplitConformalPredictor {
    /// Create new split conformal predictor
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha: alpha.clamp(0.01, 0.5),
            score_type: NonConformityScore::Absolute,
            calibration_scores: None,
            quantile: None,
        }
    }

    /// Set the non-conformity score type
    pub fn with_score_type(mut self, score_type: NonConformityScore) -> Self {
        self.score_type = score_type;
        self
    }

    /// Get the alpha value
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Get the computed quantile
    pub fn quantile(&self) -> Option<f64> {
        self.quantile
    }

    /// Calibrate the predictor using residuals from calibration set
    ///
    /// # Arguments
    /// * `y_cal` - True values on calibration set
    /// * `y_pred_cal` - Predicted values on calibration set
    pub fn calibrate(&mut self, y_cal: &Array1<f64>, y_pred_cal: &Array1<f64>) {
        // Compute non-conformity scores
        let scores = self.score_type.compute(y_cal, y_pred_cal);
        self.calibrate_from_scores(scores.to_vec());
    }

    /// Calibrate directly from pre-computed scores
    pub fn calibrate_from_scores(&mut self, scores: Vec<f64>) {
        let n = scores.len();
        if n == 0 {
            return;
        }

        // Compute quantile level for finite-sample coverage guarantee
        // q = ceil((n+1)(1-alpha)) / n
        let quantile_level = ((1.0 - self.alpha) * (n + 1) as f64).ceil() / n as f64;
        let quantile_level = quantile_level.min(1.0);

        // Compute quantile
        self.quantile = Some(compute_quantile(&scores, quantile_level));
        self.calibration_scores = Some(scores);
    }

    /// Check if predictor is calibrated
    pub fn is_calibrated(&self) -> bool {
        self.quantile.is_some()
    }

    /// Generate prediction interval for a single point prediction
    pub fn predict_interval(&self, point_pred: f64) -> Option<PredictionInterval> {
        let q = self.quantile?;

        Some(PredictionInterval::new(
            point_pred,
            point_pred - q,
            point_pred + q,
            self.alpha,
        ))
    }

    /// Generate prediction intervals for multiple predictions
    pub fn predict_intervals(&self, point_preds: &Array1<f64>) -> Option<Vec<PredictionInterval>> {
        let q = self.quantile?;

        let intervals: Vec<PredictionInterval> = point_preds
            .iter()
            .map(|&pred| PredictionInterval::new(pred, pred - q, pred + q, self.alpha))
            .collect();

        Some(intervals)
    }

    /// Evaluate coverage on test data
    pub fn evaluate_coverage(
        &self,
        y_test: &Array1<f64>,
        y_pred_test: &Array1<f64>,
    ) -> Option<CoverageMetrics> {
        let intervals = self.predict_intervals(y_pred_test)?;

        let n = y_test.len();
        let mut covered_count = 0;
        let mut total_width = 0.0;

        for (i, interval) in intervals.iter().enumerate() {
            if interval.covers(y_test[i]) {
                covered_count += 1;
            }
            total_width += interval.width;
        }

        let coverage = covered_count as f64 / n as f64;
        let avg_width = total_width / n as f64;

        Some(CoverageMetrics {
            coverage,
            target_coverage: 1.0 - self.alpha,
            coverage_gap: (coverage - (1.0 - self.alpha)).abs(),
            mean_width: avg_width,
            samples: n,
        })
    }
}

/// Coverage evaluation metrics
#[derive(Debug, Clone)]
pub struct CoverageMetrics {
    /// Empirical coverage rate
    pub coverage: f64,
    /// Target coverage (1 - alpha)
    pub target_coverage: f64,
    /// Absolute gap between empirical and target coverage
    pub coverage_gap: f64,
    /// Mean interval width
    pub mean_width: f64,
    /// Number of samples
    pub samples: usize,
}

impl std::fmt::Display for CoverageMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Coverage: {:.2}% (target: {:.2}%), Gap: {:.2}%, Width: {:.6}",
            self.coverage * 100.0,
            self.target_coverage * 100.0,
            self.coverage_gap * 100.0,
            self.mean_width
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calibration() {
        let mut predictor = SplitConformalPredictor::new(0.1);

        let y_cal = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let y_pred_cal = Array1::from_vec(vec![1.1, 2.2, 2.9, 3.8, 5.1]);

        predictor.calibrate(&y_cal, &y_pred_cal);

        assert!(predictor.is_calibrated());
        assert!(predictor.quantile().is_some());
    }

    #[test]
    fn test_prediction_interval() {
        let mut predictor = SplitConformalPredictor::new(0.1);

        // Calibrate with known scores
        predictor.calibrate_from_scores(vec![0.1, 0.2, 0.3, 0.4, 0.5]);

        let interval = predictor.predict_interval(10.0).unwrap();

        assert_eq!(interval.point, 10.0);
        assert!(interval.lower < 10.0);
        assert!(interval.upper > 10.0);
        assert!(interval.width > 0.0);
    }

    #[test]
    fn test_coverage() {
        let mut predictor = SplitConformalPredictor::new(0.1);
        predictor.calibrate_from_scores(vec![0.1, 0.2, 0.3, 0.4, 0.5]);

        let interval = predictor.predict_interval(5.0).unwrap();

        assert!(interval.covers(5.0));
        assert!(interval.covers(5.0 + interval.width / 3.0));
        assert!(!interval.covers(5.0 + interval.width * 2.0));
    }
}
