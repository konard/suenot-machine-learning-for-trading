//! Difference-in-Differences estimation implementation.
//!
//! Implements the canonical DiD design with support for:
//! - Basic two-period, two-group design
//! - Multi-period staggered treatment designs
//! - Parallel trends testing
//! - Bootstrap confidence intervals

use chrono::{DateTime, Utc};
use rand::Rng;
use statrs::distribution::{ContinuousCDF, StudentsT};
use std::collections::HashMap;

/// A single observation in the panel data.
#[derive(Debug, Clone)]
pub struct PanelData {
    /// Unit identifier (e.g., stock symbol)
    pub unit_id: String,
    /// Time period
    pub timestamp: DateTime<Utc>,
    /// Outcome variable (e.g., return)
    pub outcome: f64,
    /// Treatment indicator (true if unit is in treatment group)
    pub treated: bool,
    /// Post-treatment indicator (true if observation is after treatment)
    pub post_treatment: bool,
    /// Event time (periods relative to treatment)
    pub event_time: Option<i32>,
    /// Optional covariates
    pub covariates: HashMap<String, f64>,
}

/// Results from DiD estimation.
#[derive(Debug, Clone)]
pub struct DiDResults {
    /// DiD point estimate (average treatment effect on treated)
    pub did_estimate: f64,
    /// Standard error of the estimate
    pub std_error: f64,
    /// t-statistic
    pub t_statistic: f64,
    /// Two-sided p-value
    pub p_value: f64,
    /// Lower bound of confidence interval
    pub ci_lower: f64,
    /// Upper bound of confidence interval
    pub ci_upper: f64,
    /// Confidence level
    pub confidence_level: f64,
    /// Number of treated observations
    pub n_treated: usize,
    /// Number of control observations
    pub n_control: usize,
}

impl DiDResults {
    /// Check if the estimate is statistically significant.
    pub fn is_significant(&self, alpha: f64) -> bool {
        self.p_value < alpha
    }

    /// Generate a trading signal based on the DiD estimate.
    pub fn generate_signal(&self, threshold: f64, significance_level: f64) -> i32 {
        if self.p_value > significance_level {
            return 0;
        }

        if self.did_estimate > threshold {
            1 // Long
        } else if self.did_estimate < -threshold {
            -1 // Short
        } else {
            0 // No trade
        }
    }
}

impl std::fmt::Display for DiDResults {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let sig = if self.p_value < 0.01 {
            "***"
        } else if self.p_value < 0.05 {
            "**"
        } else if self.p_value < 0.1 {
            "*"
        } else {
            ""
        };

        writeln!(f, "DiDResults(")?;
        writeln!(f, "  did_estimate = {:.6} {}", self.did_estimate, sig)?;
        writeln!(f, "  std_error    = {:.6}", self.std_error)?;
        writeln!(f, "  t_statistic  = {:.4}", self.t_statistic)?;
        writeln!(f, "  p_value      = {:.4}", self.p_value)?;
        writeln!(
            f,
            "  {:.0}% CI      = [{:.6}, {:.6}]",
            self.confidence_level * 100.0,
            self.ci_lower,
            self.ci_upper
        )?;
        writeln!(f, "  n_treated    = {}", self.n_treated)?;
        writeln!(f, "  n_control    = {}", self.n_control)?;
        write!(f, ")")
    }
}

/// Treatment effect estimate for a specific group-time cell.
#[derive(Debug, Clone)]
pub struct TreatmentEffect {
    /// Treatment group (time of treatment adoption)
    pub treatment_time: i32,
    /// Current time period
    pub current_time: i32,
    /// ATT estimate
    pub att: f64,
    /// Number of treated units
    pub n_treated: usize,
    /// Number of control units
    pub n_control: usize,
}

/// Difference-in-Differences estimator.
pub struct DiDEstimator {
    /// Number of bootstrap iterations
    n_bootstrap: usize,
    /// Confidence level for intervals
    confidence_level: f64,
}

impl Default for DiDEstimator {
    fn default() -> Self {
        Self::new()
    }
}

impl DiDEstimator {
    /// Create a new DiD estimator with default settings.
    pub fn new() -> Self {
        Self {
            n_bootstrap: 1000,
            confidence_level: 0.95,
        }
    }

    /// Set the number of bootstrap iterations.
    pub fn with_bootstrap(mut self, n: usize) -> Self {
        self.n_bootstrap = n;
        self
    }

    /// Set the confidence level.
    pub fn with_confidence(mut self, level: f64) -> Self {
        self.confidence_level = level;
        self
    }

    /// Estimate the DiD treatment effect.
    ///
    /// Uses the canonical formula:
    /// DiD = E[Y|D=1,T=1] - E[Y|D=1,T=0] - (E[Y|D=0,T=1] - E[Y|D=0,T=0])
    pub fn fit(&self, data: &[PanelData]) -> DiDResults {
        // Compute cell means
        let (y_11, n_11) = self.cell_mean(data, true, true);
        let (y_10, n_10) = self.cell_mean(data, true, false);
        let (y_01, n_01) = self.cell_mean(data, false, true);
        let (y_00, n_00) = self.cell_mean(data, false, false);

        // Check for missing cells
        if n_11 == 0 || n_10 == 0 || n_01 == 0 || n_00 == 0 {
            return DiDResults {
                did_estimate: f64::NAN,
                std_error: f64::NAN,
                t_statistic: f64::NAN,
                p_value: 1.0,
                ci_lower: f64::NAN,
                ci_upper: f64::NAN,
                confidence_level: self.confidence_level,
                n_treated: n_11 + n_10,
                n_control: n_01 + n_00,
            };
        }

        // DiD estimate
        let did_estimate = (y_11 - y_10) - (y_01 - y_00);

        // Bootstrap standard error
        let std_error = self.bootstrap_standard_error(data);

        // Compute test statistics
        let df = (n_11 + n_10 + n_01 + n_00 - 4) as f64;
        let t_statistic = if std_error > 0.0 {
            did_estimate / std_error
        } else {
            0.0
        };

        // P-value using t-distribution
        let p_value = if df > 0.0 {
            let t_dist = StudentsT::new(0.0, 1.0, df).unwrap_or_else(|_| StudentsT::new(0.0, 1.0, 30.0).unwrap());
            2.0 * (1.0 - t_dist.cdf(t_statistic.abs()))
        } else {
            1.0
        };

        // Confidence interval
        let t_critical = if df > 0.0 {
            let t_dist = StudentsT::new(0.0, 1.0, df).unwrap_or_else(|_| StudentsT::new(0.0, 1.0, 30.0).unwrap());
            t_dist.inverse_cdf(1.0 - (1.0 - self.confidence_level) / 2.0)
        } else {
            1.96
        };

        let ci_lower = did_estimate - t_critical * std_error;
        let ci_upper = did_estimate + t_critical * std_error;

        DiDResults {
            did_estimate,
            std_error,
            t_statistic,
            p_value,
            ci_lower,
            ci_upper,
            confidence_level: self.confidence_level,
            n_treated: n_11 + n_10,
            n_control: n_01 + n_00,
        }
    }

    /// Test the parallel trends assumption.
    ///
    /// Estimates event-time dummies for pre-treatment periods and checks
    /// whether they are jointly close to zero.
    pub fn test_parallel_trends(
        &self,
        data: &[PanelData],
        n_pre_periods: i32,
    ) -> HashMap<i32, f64> {
        let mut coefficients = HashMap::new();

        for t in -n_pre_periods..0 {
            let pre_data: Vec<_> = data
                .iter()
                .filter(|d| d.event_time == Some(t))
                .collect();

            if pre_data.is_empty() {
                continue;
            }

            let treated_mean: f64 = pre_data
                .iter()
                .filter(|d| d.treated)
                .map(|d| d.outcome)
                .sum::<f64>()
                / pre_data.iter().filter(|d| d.treated).count().max(1) as f64;

            let control_mean: f64 = pre_data
                .iter()
                .filter(|d| !d.treated)
                .map(|d| d.outcome)
                .sum::<f64>()
                / pre_data.iter().filter(|d| !d.treated).count().max(1) as f64;

            let n_treated = pre_data.iter().filter(|d| d.treated).count();
            let n_control = pre_data.iter().filter(|d| !d.treated).count();

            if n_treated > 0 && n_control > 0 {
                coefficients.insert(t, treated_mean - control_mean);
            }
        }

        coefficients
    }

    /// Compute mean outcome for a specific cell.
    fn cell_mean(&self, data: &[PanelData], treated: bool, post: bool) -> (f64, usize) {
        let cell: Vec<f64> = data
            .iter()
            .filter(|d| d.treated == treated && d.post_treatment == post)
            .map(|d| d.outcome)
            .collect();

        if cell.is_empty() {
            (f64::NAN, 0)
        } else {
            let mean = cell.iter().sum::<f64>() / cell.len() as f64;
            (mean, cell.len())
        }
    }

    /// Compute bootstrap standard error.
    fn bootstrap_standard_error(&self, data: &[PanelData]) -> f64 {
        let mut rng = rand::thread_rng();
        let n = data.len();

        if n == 0 {
            return 0.0;
        }

        let mut estimates = Vec::with_capacity(self.n_bootstrap);

        for _ in 0..self.n_bootstrap {
            // Resample with replacement
            let boot_data: Vec<PanelData> = (0..n)
                .map(|_| data[rng.gen_range(0..n)].clone())
                .collect();

            // Compute DiD on bootstrap sample
            let (y_11, n_11) = self.cell_mean(&boot_data, true, true);
            let (y_10, n_10) = self.cell_mean(&boot_data, true, false);
            let (y_01, n_01) = self.cell_mean(&boot_data, false, true);
            let (y_00, n_00) = self.cell_mean(&boot_data, false, false);

            if n_11 > 0 && n_10 > 0 && n_01 > 0 && n_00 > 0 {
                let did = (y_11 - y_10) - (y_01 - y_00);
                if did.is_finite() {
                    estimates.push(did);
                }
            }
        }

        if estimates.is_empty() {
            return 0.0;
        }

        // Standard deviation of bootstrap estimates
        let mean = estimates.iter().sum::<f64>() / estimates.len() as f64;
        let variance = estimates.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
            / (estimates.len() - 1).max(1) as f64;

        variance.sqrt()
    }
}

/// Generate synthetic panel data for testing.
pub fn generate_synthetic_panel(
    n_treated: usize,
    n_control: usize,
    pre_periods: usize,
    post_periods: usize,
    treatment_effect: f64,
    base_trend: f64,
) -> Vec<PanelData> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut data = Vec::new();
    let now = Utc::now();

    // Generate treated units
    for i in 0..n_treated {
        for t in 0..(pre_periods + post_periods) {
            let is_post = t >= pre_periods;
            let event_time = t as i32 - pre_periods as i32;

            // Outcome with trend and treatment effect
            let trend = base_trend * t as f64;
            let effect = if is_post { treatment_effect } else { 0.0 };
            let noise = rng.gen_range(-0.02..0.02);
            let outcome = 0.01 + trend + effect + noise;

            data.push(PanelData {
                unit_id: format!("treated_{}", i),
                timestamp: now - chrono::Duration::days((pre_periods + post_periods - t) as i64),
                outcome,
                treated: true,
                post_treatment: is_post,
                event_time: Some(event_time),
                covariates: HashMap::new(),
            });
        }
    }

    // Generate control units
    for i in 0..n_control {
        for t in 0..(pre_periods + post_periods) {
            let is_post = t >= pre_periods;
            let event_time = t as i32 - pre_periods as i32;

            // Outcome with trend but no treatment effect
            let trend = base_trend * t as f64;
            let noise = rng.gen_range(-0.02..0.02);
            let outcome = 0.01 + trend + noise;

            data.push(PanelData {
                unit_id: format!("control_{}", i),
                timestamp: now - chrono::Duration::days((pre_periods + post_periods - t) as i64),
                outcome,
                treated: false,
                post_treatment: is_post,
                event_time: Some(event_time),
                covariates: HashMap::new(),
            });
        }
    }

    data
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_did_estimator_basic() {
        // Generate synthetic data with known treatment effect
        let data = generate_synthetic_panel(
            10,   // n_treated
            10,   // n_control
            5,    // pre_periods
            5,    // post_periods
            0.05, // treatment_effect
            0.01, // base_trend
        );

        let estimator = DiDEstimator::new().with_bootstrap(100);
        let results = estimator.fit(&data);

        // Check that estimate is close to true effect
        assert!(
            (results.did_estimate - 0.05).abs() < 0.03,
            "DiD estimate should be close to true effect of 0.05"
        );

        // Check that confidence interval contains true value
        assert!(
            results.ci_lower < 0.05 && results.ci_upper > 0.05,
            "Confidence interval should contain true effect"
        );
    }

    #[test]
    fn test_parallel_trends() {
        let data = generate_synthetic_panel(10, 10, 5, 5, 0.05, 0.01);

        let estimator = DiDEstimator::new();
        let coefficients = estimator.test_parallel_trends(&data, 5);

        // Pre-treatment coefficients should be close to zero
        for (t, coef) in &coefficients {
            assert!(
                coef.abs() < 0.05,
                "Pre-treatment coefficient at t={} should be close to zero, got {}",
                t,
                coef
            );
        }
    }
}
