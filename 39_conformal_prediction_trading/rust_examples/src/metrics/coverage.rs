//! Coverage and interval quality metrics
//!
//! Metrics for evaluating conformal prediction performance.

use crate::conformal::PredictionInterval;

/// Coverage and interval quality metrics
#[derive(Debug, Clone)]
pub struct CoverageMetrics {
    /// Empirical coverage (fraction of true values in intervals)
    pub coverage: f64,
    /// Average interval width
    pub avg_width: f64,
    /// Standard deviation of interval widths
    pub width_std: f64,
    /// Sharpness (inverse of average width)
    pub sharpness: f64,
    /// Winkler score (combined coverage and sharpness)
    pub winkler_score: f64,
    /// Number of samples
    pub n_samples: usize,
}

impl CoverageMetrics {
    /// Calculate metrics from prediction intervals and actual values
    pub fn calculate(intervals: &[PredictionInterval], actuals: &[f64], alpha: f64) -> Self {
        let n = intervals.len().min(actuals.len());
        if n == 0 {
            return Self::empty();
        }

        // Coverage
        let n_covered = intervals
            .iter()
            .zip(actuals.iter())
            .take(n)
            .filter(|(interval, &actual)| interval.covers(actual))
            .count();
        let coverage = n_covered as f64 / n as f64;

        // Width statistics
        let widths: Vec<f64> = intervals.iter().take(n).map(|i| i.width).collect();
        let avg_width = widths.iter().sum::<f64>() / n as f64;
        let width_std = Self::std_dev(&widths);
        let sharpness = if avg_width > 0.0 {
            1.0 / avg_width
        } else {
            f64::INFINITY
        };

        // Winkler score (lower is better)
        let winkler_score = Self::calculate_winkler(intervals, actuals, alpha);

        Self {
            coverage,
            avg_width,
            width_std,
            sharpness,
            winkler_score,
            n_samples: n,
        }
    }

    /// Calculate Winkler score
    ///
    /// Winkler score combines interval width with coverage penalty.
    /// Lower is better.
    fn calculate_winkler(intervals: &[PredictionInterval], actuals: &[f64], alpha: f64) -> f64 {
        let n = intervals.len().min(actuals.len());
        if n == 0 {
            return 0.0;
        }

        let mut total_score = 0.0;

        for (interval, &actual) in intervals.iter().zip(actuals.iter()).take(n) {
            let width = interval.width;

            if actual < interval.lower {
                // Below lower bound: width + penalty
                total_score += width + (2.0 / alpha) * (interval.lower - actual);
            } else if actual > interval.upper {
                // Above upper bound: width + penalty
                total_score += width + (2.0 / alpha) * (actual - interval.upper);
            } else {
                // Covered: just width
                total_score += width;
            }
        }

        total_score / n as f64
    }

    /// Calculate conditional coverage (coverage in different regimes)
    pub fn conditional_coverage(
        intervals: &[PredictionInterval],
        actuals: &[f64],
        condition: &[bool],
    ) -> (f64, f64) {
        let n = intervals.len().min(actuals.len()).min(condition.len());
        if n == 0 {
            return (0.0, 0.0);
        }

        let mut n_true = 0;
        let mut n_true_covered = 0;
        let mut n_false = 0;
        let mut n_false_covered = 0;

        for ((interval, &actual), &cond) in
            intervals.iter().zip(actuals.iter()).zip(condition.iter()).take(n)
        {
            if cond {
                n_true += 1;
                if interval.covers(actual) {
                    n_true_covered += 1;
                }
            } else {
                n_false += 1;
                if interval.covers(actual) {
                    n_false_covered += 1;
                }
            }
        }

        let coverage_true = if n_true > 0 {
            n_true_covered as f64 / n_true as f64
        } else {
            0.0
        };

        let coverage_false = if n_false > 0 {
            n_false_covered as f64 / n_false as f64
        } else {
            0.0
        };

        (coverage_true, coverage_false)
    }

    /// Calculate coverage by interval width quantiles
    pub fn coverage_by_width_quantile(
        intervals: &[PredictionInterval],
        actuals: &[f64],
        n_quantiles: usize,
    ) -> Vec<(f64, f64, usize)> {
        let n = intervals.len().min(actuals.len());
        if n == 0 {
            return vec![];
        }

        // Sort by width
        let mut indexed: Vec<(usize, f64)> = intervals
            .iter()
            .enumerate()
            .take(n)
            .map(|(i, interval)| (i, interval.width))
            .collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let bucket_size = (n + n_quantiles - 1) / n_quantiles;
        let mut results = Vec::new();

        for q in 0..n_quantiles {
            let start = q * bucket_size;
            let end = ((q + 1) * bucket_size).min(n);
            if start >= n {
                break;
            }

            let bucket_indices: Vec<usize> = indexed[start..end].iter().map(|(i, _)| *i).collect();
            let n_bucket = bucket_indices.len();

            let n_covered = bucket_indices
                .iter()
                .filter(|&&i| intervals[i].covers(actuals[i]))
                .count();

            let avg_width: f64 =
                bucket_indices.iter().map(|&i| intervals[i].width).sum::<f64>() / n_bucket as f64;

            let coverage = n_covered as f64 / n_bucket as f64;

            results.push((avg_width, coverage, n_bucket));
        }

        results
    }

    /// Empty metrics
    fn empty() -> Self {
        Self {
            coverage: 0.0,
            avg_width: 0.0,
            width_std: 0.0,
            sharpness: 0.0,
            winkler_score: 0.0,
            n_samples: 0,
        }
    }

    /// Standard deviation helper
    fn std_dev(values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
        variance.sqrt()
    }

    /// Generate a summary report
    pub fn report(&self) -> String {
        format!(
            "Coverage Metrics:\n\
             ================\n\
             Coverage:      {:.2}%\n\
             Avg Width:     {:.4}\n\
             Width Std:     {:.4}\n\
             Sharpness:     {:.2}\n\
             Winkler Score: {:.4}\n\
             N Samples:     {}",
            self.coverage * 100.0,
            self.avg_width,
            self.width_std,
            self.sharpness,
            self.winkler_score,
            self.n_samples
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coverage_metrics() {
        let intervals = vec![
            PredictionInterval::new(0.0, -1.0, 1.0),
            PredictionInterval::new(0.0, -1.0, 1.0),
            PredictionInterval::new(0.0, -1.0, 1.0),
            PredictionInterval::new(0.0, -1.0, 1.0),
        ];
        let actuals = vec![0.5, -0.5, 2.0, 0.0]; // 3 covered, 1 not

        let metrics = CoverageMetrics::calculate(&intervals, &actuals, 0.1);

        assert!((metrics.coverage - 0.75).abs() < 1e-10);
        assert!((metrics.avg_width - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_conditional_coverage() {
        let intervals = vec![
            PredictionInterval::new(0.0, -1.0, 1.0),
            PredictionInterval::new(0.0, -1.0, 1.0),
            PredictionInterval::new(0.0, -1.0, 1.0),
            PredictionInterval::new(0.0, -1.0, 1.0),
        ];
        let actuals = vec![0.5, 2.0, 0.0, 3.0];
        let condition = vec![true, true, false, false]; // 2 true (1 covered), 2 false (1 covered)

        let (cov_true, cov_false) = CoverageMetrics::conditional_coverage(&intervals, &actuals, &condition);

        assert!((cov_true - 0.5).abs() < 1e-10); // 1/2 covered when true
        assert!((cov_false - 0.5).abs() < 1e-10); // 1/2 covered when false
    }
}
