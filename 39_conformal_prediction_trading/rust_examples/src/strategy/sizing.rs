//! Position sizing based on prediction intervals
//!
//! Maps interval width and edge to position size using various schemes.

/// Position sizing method
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SizingMethod {
    /// Size inversely proportional to interval width
    Inverse,
    /// Linear decrease based on width
    Linear,
    /// Exponential decay based on width
    Exponential,
    /// Fixed size regardless of width
    Fixed,
    /// Kelly-like sizing using edge/width ratio
    Kelly,
}

/// Position sizer based on prediction interval confidence
#[derive(Debug, Clone)]
pub struct PositionSizer {
    method: SizingMethod,
    max_size: f64,
    min_size: f64,
    /// Baseline width for normalization
    baseline_width: f64,
    /// Kelly fraction multiplier (for Kelly method)
    kelly_fraction: f64,
}

impl Default for PositionSizer {
    fn default() -> Self {
        Self::inverse()
    }
}

impl PositionSizer {
    /// Create inverse sizer (narrower intervals = larger positions)
    pub fn inverse() -> Self {
        Self {
            method: SizingMethod::Inverse,
            max_size: 1.0,
            min_size: 0.0,
            baseline_width: 0.02,
            kelly_fraction: 0.5,
        }
    }

    /// Create linear sizer
    pub fn linear() -> Self {
        Self {
            method: SizingMethod::Linear,
            max_size: 1.0,
            min_size: 0.0,
            baseline_width: 0.02,
            kelly_fraction: 0.5,
        }
    }

    /// Create exponential sizer
    pub fn exponential() -> Self {
        Self {
            method: SizingMethod::Exponential,
            max_size: 1.0,
            min_size: 0.0,
            baseline_width: 0.02,
            kelly_fraction: 0.5,
        }
    }

    /// Create fixed size sizer
    pub fn fixed(size: f64) -> Self {
        Self {
            method: SizingMethod::Fixed,
            max_size: size,
            min_size: size,
            baseline_width: 0.02,
            kelly_fraction: 0.5,
        }
    }

    /// Create Kelly-based sizer
    pub fn kelly(fraction: f64) -> Self {
        Self {
            method: SizingMethod::Kelly,
            max_size: 1.0,
            min_size: 0.0,
            baseline_width: 0.02,
            kelly_fraction: fraction.max(0.1).min(1.0),
        }
    }

    /// Set maximum size
    pub fn with_max_size(mut self, max_size: f64) -> Self {
        self.max_size = max_size.max(0.0);
        self
    }

    /// Set minimum size
    pub fn with_min_size(mut self, min_size: f64) -> Self {
        self.min_size = min_size.max(0.0);
        self
    }

    /// Set baseline width
    pub fn with_baseline_width(mut self, width: f64) -> Self {
        self.baseline_width = width.max(0.001);
        self
    }

    /// Compute position size based on interval width and edge
    ///
    /// # Arguments
    /// * `interval_width` - Width of the prediction interval
    /// * `edge` - Expected edge (worst-case expected return)
    pub fn compute_size(&self, interval_width: f64, edge: f64) -> f64 {
        if interval_width <= 0.0 {
            return 0.0;
        }

        let raw_size = match self.method {
            SizingMethod::Inverse => {
                // Size = baseline / current (narrower = larger)
                self.baseline_width / interval_width
            }
            SizingMethod::Linear => {
                // Size = 1 - width/baseline (capped at 0)
                (1.0 - interval_width / self.baseline_width).max(0.0)
            }
            SizingMethod::Exponential => {
                // Size = exp(-width/baseline)
                (-interval_width / self.baseline_width).exp()
            }
            SizingMethod::Fixed => self.max_size,
            SizingMethod::Kelly => {
                // Kelly: edge / variance, using width as volatility proxy
                let implied_std = interval_width / 2.0;
                let implied_variance = implied_std * implied_std;
                if implied_variance < 1e-10 {
                    0.0
                } else {
                    self.kelly_fraction * edge / implied_variance
                }
            }
        };

        // Apply bounds
        raw_size.max(self.min_size).min(self.max_size)
    }

    /// Compute size with only interval width (using default edge assumption)
    pub fn compute_size_from_width(&self, interval_width: f64) -> f64 {
        // Assume edge = 25% of interval width for sizing
        let assumed_edge = interval_width * 0.25;
        self.compute_size(interval_width, assumed_edge)
    }
}

/// Kelly criterion calculator
pub struct KellyCriterion;

impl KellyCriterion {
    /// Calculate Kelly fraction from prediction interval
    ///
    /// # Arguments
    /// * `prediction` - Point prediction (expected return)
    /// * `lower` - Lower bound of interval
    /// * `upper` - Upper bound of interval
    /// * `risk_free_rate` - Risk-free rate (default 0)
    pub fn calculate(prediction: f64, lower: f64, upper: f64, risk_free_rate: f64) -> f64 {
        let interval_width = upper - lower;
        let expected_excess = prediction - risk_free_rate;

        // No edge -> no position
        if expected_excess <= 0.0 {
            return 0.0;
        }

        // Degenerate interval
        if interval_width <= 0.0 {
            return 0.0;
        }

        // Use interval width as volatility proxy
        // Kelly: f* = mu / sigma^2
        let implied_std = interval_width / 2.0;
        let implied_variance = implied_std * implied_std;

        let kelly = expected_excess / implied_variance;

        // Apply half-Kelly for safety
        kelly * 0.5
    }

    /// Calculate with constraints
    pub fn calculate_constrained(
        prediction: f64,
        lower: f64,
        upper: f64,
        risk_free_rate: f64,
        max_leverage: f64,
    ) -> f64 {
        let kelly = Self::calculate(prediction, lower, upper, risk_free_rate);
        kelly.max(-max_leverage).min(max_leverage)
    }
}

/// Volatility-based position sizer
pub struct VolatilityTargetSizer {
    target_volatility: f64,
    max_size: f64,
}

impl VolatilityTargetSizer {
    /// Create a new volatility target sizer
    ///
    /// # Arguments
    /// * `target_volatility` - Target annualized volatility (e.g., 0.15 for 15%)
    pub fn new(target_volatility: f64) -> Self {
        Self {
            target_volatility,
            max_size: 1.0,
        }
    }

    /// Set maximum size
    pub fn with_max_size(mut self, max_size: f64) -> Self {
        self.max_size = max_size;
        self
    }

    /// Compute position size based on interval width
    ///
    /// Assumes interval width proxies for daily volatility
    pub fn compute_size(&self, interval_width: f64) -> f64 {
        if interval_width <= 0.0 {
            return 0.0;
        }

        // Interval width represents ~1.6 sigma for 90% coverage
        // Daily vol ~= width / 3.2
        let daily_vol = interval_width / 3.2;

        // Annualize
        let annual_vol = daily_vol * (252.0_f64).sqrt();

        if annual_vol <= 0.0 {
            return 0.0;
        }

        // Size = target_vol / actual_vol
        let size = self.target_volatility / annual_vol;

        size.min(self.max_size).max(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inverse_sizing() {
        let sizer = PositionSizer::inverse().with_baseline_width(0.02);

        // At baseline width, size should be 1.0
        let size = sizer.compute_size(0.02, 0.01);
        assert!((size - 1.0).abs() < 0.01);

        // Narrower interval -> larger size
        let size_narrow = sizer.compute_size(0.01, 0.005);
        assert!(size_narrow > size);

        // Wider interval -> smaller size
        let size_wide = sizer.compute_size(0.04, 0.02);
        assert!(size_wide < size);
    }

    #[test]
    fn test_kelly_sizing() {
        let sizer = PositionSizer::kelly(0.5);

        let size = sizer.compute_size(0.02, 0.01);
        assert!(size > 0.0);
        assert!(size <= 1.0);
    }

    #[test]
    fn test_kelly_criterion() {
        // Expected return 1%, interval width 2%
        let kelly = KellyCriterion::calculate(0.01, 0.0, 0.02, 0.0);

        // Should be positive
        assert!(kelly > 0.0);

        // Negative expected return -> zero
        let kelly_neg = KellyCriterion::calculate(-0.01, -0.02, 0.0, 0.0);
        assert!((kelly_neg - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_fixed_sizing() {
        let sizer = PositionSizer::fixed(0.5);

        // Size should always be 0.5
        assert!((sizer.compute_size(0.01, 0.005) - 0.5).abs() < 1e-10);
        assert!((sizer.compute_size(0.05, 0.025) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_volatility_target_sizer() {
        let sizer = VolatilityTargetSizer::new(0.15);

        // Narrow interval (low vol) -> larger size
        let size_low_vol = sizer.compute_size(0.01);
        // Wide interval (high vol) -> smaller size
        let size_high_vol = sizer.compute_size(0.05);

        assert!(size_low_vol > size_high_vol);
    }
}
