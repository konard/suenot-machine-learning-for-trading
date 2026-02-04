//! Non-conformity score functions for conformal prediction

use ndarray::Array1;

/// Types of non-conformity scores
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NonConformityScore {
    /// Absolute residual: |y - y_hat|
    Absolute,
    /// Signed residual: y - y_hat
    Signed,
    /// Normalized residual: |y - y_hat| / sigma
    Normalized,
}

impl NonConformityScore {
    /// Compute non-conformity scores
    pub fn compute(&self, y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Array1<f64> {
        match self {
            NonConformityScore::Absolute => {
                (y_true - y_pred).mapv(|x| x.abs())
            }
            NonConformityScore::Signed => {
                y_true - y_pred
            }
            NonConformityScore::Normalized => {
                let residuals = y_true - y_pred;
                let mad = median_absolute_deviation(&residuals);
                let scale = if mad > 1e-10 { mad } else { 1.0 };
                residuals.mapv(|x| x.abs() / scale)
            }
        }
    }

    /// Compute single score
    pub fn compute_single(&self, y_true: f64, y_pred: f64, scale: Option<f64>) -> f64 {
        match self {
            NonConformityScore::Absolute => (y_true - y_pred).abs(),
            NonConformityScore::Signed => y_true - y_pred,
            NonConformityScore::Normalized => {
                let s = scale.unwrap_or(1.0);
                (y_true - y_pred).abs() / s.max(1e-10)
            }
        }
    }
}

/// Compute CQR (Conformalized Quantile Regression) score
/// score = max(q_low - y, y - q_high)
pub fn cqr_score(y_true: &Array1<f64>, q_low: &Array1<f64>, q_high: &Array1<f64>) -> Array1<f64> {
    let n = y_true.len();
    let mut scores = Array1::zeros(n);

    for i in 0..n {
        let score_low = q_low[i] - y_true[i];
        let score_high = y_true[i] - q_high[i];
        scores[i] = score_low.max(score_high);
    }

    scores
}

/// Compute median absolute deviation
pub fn median_absolute_deviation(data: &Array1<f64>) -> f64 {
    let median = compute_median(data.as_slice().unwrap());
    let deviations: Vec<f64> = data.iter().map(|x| (x - median).abs()).collect();
    compute_median(&deviations) * 1.4826 // Scale factor for Gaussian
}

/// Compute median of a slice
pub fn compute_median(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }

    let mut sorted: Vec<f64> = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = sorted.len();
    if n % 2 == 0 {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    }
}

/// Compute quantile of a slice
pub fn compute_quantile(data: &[f64], q: f64) -> f64 {
    if data.is_empty() {
        return 0.0;
    }

    let mut sorted: Vec<f64> = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = sorted.len();
    let idx = (q * (n - 1) as f64).floor() as usize;
    let frac = q * (n - 1) as f64 - idx as f64;

    if idx + 1 < n {
        sorted[idx] * (1.0 - frac) + sorted[idx + 1] * frac
    } else {
        sorted[idx]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_absolute_score() {
        let y_true = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let y_pred = Array1::from_vec(vec![1.5, 2.0, 2.5]);

        let scores = NonConformityScore::Absolute.compute(&y_true, &y_pred);

        assert!((scores[0] - 0.5).abs() < 1e-10);
        assert!((scores[1] - 0.0).abs() < 1e-10);
        assert!((scores[2] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_median() {
        let data = vec![1.0, 3.0, 5.0, 7.0, 9.0];
        assert!((compute_median(&data) - 5.0).abs() < 1e-10);

        let data2 = vec![1.0, 2.0, 3.0, 4.0];
        assert!((compute_median(&data2) - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_quantile() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        assert!((compute_quantile(&data, 0.5) - 3.0).abs() < 1e-10);
        assert!((compute_quantile(&data, 0.0) - 1.0).abs() < 1e-10);
        assert!((compute_quantile(&data, 1.0) - 5.0).abs() < 1e-10);
    }
}
