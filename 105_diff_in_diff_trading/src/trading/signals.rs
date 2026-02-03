//! Signal generation from DiD estimates.

use crate::model::DiDResults;

/// Trading signal direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Signal {
    /// Buy / Long signal
    Long,
    /// Sell / Short signal
    Short,
    /// No action
    Flat,
}

impl Signal {
    /// Convert to numeric value (1 for Long, -1 for Short, 0 for Flat).
    pub fn as_i32(&self) -> i32 {
        match self {
            Signal::Long => 1,
            Signal::Short => -1,
            Signal::Flat => 0,
        }
    }
}

/// Signal generator based on DiD estimates.
pub struct SignalGenerator {
    /// Minimum absolute effect size to generate a signal
    threshold: f64,
    /// Maximum p-value for statistical significance
    significance_level: f64,
    /// Confidence level for position sizing
    confidence_level: f64,
}

impl Default for SignalGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl SignalGenerator {
    /// Create a new signal generator with default settings.
    pub fn new() -> Self {
        Self {
            threshold: 0.02,
            significance_level: 0.05,
            confidence_level: 0.95,
        }
    }

    /// Set the effect size threshold.
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold;
        self
    }

    /// Set the significance level.
    pub fn with_significance(mut self, level: f64) -> Self {
        self.significance_level = level;
        self
    }

    /// Set the confidence level.
    pub fn with_confidence(mut self, level: f64) -> Self {
        self.confidence_level = level;
        self
    }

    /// Generate a trading signal from DiD results.
    pub fn generate(&self, results: &DiDResults) -> Signal {
        // Check statistical significance
        if results.p_value > self.significance_level {
            return Signal::Flat;
        }

        // Check effect size
        if results.did_estimate > self.threshold {
            Signal::Long
        } else if results.did_estimate < -self.threshold {
            Signal::Short
        } else {
            Signal::Flat
        }
    }

    /// Compute position size based on DiD results.
    ///
    /// Uses a Kelly-criterion inspired approach based on t-statistic.
    pub fn compute_position_size(
        &self,
        results: &DiDResults,
        max_position: f64,
        portfolio_value: f64,
    ) -> f64 {
        if results.std_error <= 0.0 {
            return 0.0;
        }

        // Confidence based on t-statistic
        let t_stat = results.t_statistic.abs();
        let confidence = (t_stat / 2.0).min(1.0);

        // Size is proportional to confidence
        let size_fraction = confidence * max_position;

        portfolio_value * size_fraction
    }

    /// Compute stop-loss level based on control group volatility.
    pub fn compute_stop_loss(&self, results: &DiDResults, multiplier: f64) -> f64 {
        results.std_error * multiplier
    }

    /// Compute take-profit level based on DiD effect estimate.
    pub fn compute_take_profit(&self, results: &DiDResults, multiplier: f64) -> f64 {
        results.did_estimate.abs() * multiplier
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_results(estimate: f64, std_error: f64, p_value: f64) -> DiDResults {
        let t_stat = if std_error > 0.0 {
            estimate / std_error
        } else {
            0.0
        };
        DiDResults {
            did_estimate: estimate,
            std_error,
            t_statistic: t_stat,
            p_value,
            ci_lower: estimate - 1.96 * std_error,
            ci_upper: estimate + 1.96 * std_error,
            confidence_level: 0.95,
            n_treated: 100,
            n_control: 100,
        }
    }

    #[test]
    fn test_signal_generation() {
        let generator = SignalGenerator::new();

        // Significant positive effect -> Long
        let results = make_test_results(0.05, 0.01, 0.001);
        assert_eq!(generator.generate(&results), Signal::Long);

        // Significant negative effect -> Short
        let results = make_test_results(-0.05, 0.01, 0.001);
        assert_eq!(generator.generate(&results), Signal::Short);

        // Non-significant effect -> Flat
        let results = make_test_results(0.05, 0.01, 0.10);
        assert_eq!(generator.generate(&results), Signal::Flat);

        // Small effect -> Flat
        let results = make_test_results(0.01, 0.01, 0.001);
        assert_eq!(generator.generate(&results), Signal::Flat);
    }

    #[test]
    fn test_position_sizing() {
        let generator = SignalGenerator::new();
        let results = make_test_results(0.05, 0.01, 0.001);

        let size = generator.compute_position_size(&results, 0.15, 100000.0);
        assert!(size > 0.0);
        assert!(size <= 15000.0); // Max 15% of portfolio
    }
}
