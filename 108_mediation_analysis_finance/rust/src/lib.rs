//! # Mediation Analysis for Financial Trading
//!
//! High-performance Rust implementation of causal mediation analysis
//! for algorithmic trading systems.
//!
//! ## Features
//!
//! - Classical Baron-Kenny mediation analysis
//! - Real-time streaming analysis with rolling windows
//! - Trading signal generation based on mediation dynamics
//! - High-performance numerical computations
//!
//! ## Example
//!
//! ```rust
//! use mediation_analysis::MediationAnalyzer;
//!
//! let mut analyzer = MediationAnalyzer::new(60);
//!
//! // Feed data points
//! for i in 0..100 {
//!     let x = (i as f64 - 50.0) / 50.0;
//!     let m = 0.5 * x + rand::random::<f64>() * 0.1;
//!     let y = 0.3 * x + 0.4 * m + rand::random::<f64>() * 0.1;
//!     analyzer.update(x, m, y);
//! }
//!
//! // Get analysis results
//! if let Some(results) = analyzer.analyze() {
//!     println!("Total effect: {:.4}", results.total_effect);
//!     println!("Direct effect: {:.4}", results.direct_effect);
//!     println!("Indirect effect: {:.4}", results.indirect_effect);
//! }
//! ```

use std::f64::consts::SQRT_2;

/// Results from mediation analysis
#[derive(Debug, Clone)]
pub struct MediationResults {
    /// Total effect of X on Y
    pub total_effect: f64,
    pub total_effect_se: f64,

    /// Direct effect (X → Y controlling for M)
    pub direct_effect: f64,
    pub direct_effect_se: f64,

    /// Indirect effect (X → M → Y)
    pub indirect_effect: f64,
    pub indirect_effect_se: f64,

    /// Path coefficients
    pub a_path: f64,  // X → M
    pub a_path_se: f64,
    pub b_path: f64,  // M → Y | X
    pub b_path_se: f64,

    /// Statistical tests
    pub sobel_z: f64,
    pub sobel_pvalue: f64,

    /// Proportion of effect mediated
    pub proportion_mediated: f64,
}

impl MediationResults {
    /// Check if mediation is statistically significant
    pub fn is_significant(&self, alpha: f64) -> bool {
        self.sobel_pvalue < alpha
    }

    /// Get mediation type
    pub fn mediation_type(&self, alpha: f64) -> MediationType {
        let direct_sig = self.direct_effect_se > 0.0 &&
            (self.direct_effect / self.direct_effect_se).abs() > 1.96;
        let indirect_sig = self.sobel_pvalue < alpha;

        match (direct_sig, indirect_sig) {
            (false, true) => MediationType::Full,
            (true, true) => MediationType::Partial,
            (true, false) => MediationType::DirectOnly,
            (false, false) => MediationType::None,
        }
    }
}

/// Type of mediation detected
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MediationType {
    Full,
    Partial,
    DirectOnly,
    None,
}

/// Trading signal based on mediation analysis
#[derive(Debug, Clone)]
pub struct TradingSignal {
    /// Signal direction: -1 (short), 0 (neutral), 1 (long)
    pub signal: i32,
    /// Signal confidence (0 to 1)
    pub confidence: f64,
    /// Is mediation channel active?
    pub mediator_activated: bool,
    /// Expected effect size
    pub expected_effect: f64,
}

/// Mediation analyzer for streaming financial data
pub struct MediationAnalyzer {
    /// Lookback window for rolling estimation
    lookback: usize,
    /// Minimum observations before analysis
    min_observations: usize,

    /// History buffers
    x_history: Vec<f64>,
    m_history: Vec<f64>,
    y_history: Vec<f64>,
}

impl MediationAnalyzer {
    /// Create new analyzer with specified lookback window
    ///
    /// # Arguments
    ///
    /// * `lookback` - Number of observations to use for estimation
    ///
    /// # Example
    ///
    /// ```rust
    /// use mediation_analysis::MediationAnalyzer;
    /// let analyzer = MediationAnalyzer::new(60);
    /// ```
    pub fn new(lookback: usize) -> Self {
        Self {
            lookback,
            min_observations: 30.min(lookback),
            x_history: Vec::with_capacity(lookback),
            m_history: Vec::with_capacity(lookback),
            y_history: Vec::with_capacity(lookback),
        }
    }

    /// Create analyzer with custom minimum observations
    pub fn with_min_observations(lookback: usize, min_observations: usize) -> Self {
        Self {
            lookback,
            min_observations,
            x_history: Vec::with_capacity(lookback),
            m_history: Vec::with_capacity(lookback),
            y_history: Vec::with_capacity(lookback),
        }
    }

    /// Update with new observation
    ///
    /// # Arguments
    ///
    /// * `x` - Treatment value
    /// * `m` - Mediator value
    /// * `y` - Outcome value
    pub fn update(&mut self, x: f64, m: f64, y: f64) {
        self.x_history.push(x);
        self.m_history.push(m);
        self.y_history.push(y);

        // Keep only lookback window
        if self.x_history.len() > self.lookback {
            self.x_history.remove(0);
            self.m_history.remove(0);
            self.y_history.remove(0);
        }
    }

    /// Get number of observations
    pub fn len(&self) -> usize {
        self.x_history.len()
    }

    /// Check if analyzer has data
    pub fn is_empty(&self) -> bool {
        self.x_history.is_empty()
    }

    /// Check if we have enough data for analysis
    pub fn can_analyze(&self) -> bool {
        self.x_history.len() >= self.min_observations
    }

    /// Run mediation analysis on current data
    ///
    /// Returns `None` if there isn't enough data or analysis fails.
    pub fn analyze(&self) -> Option<MediationResults> {
        if !self.can_analyze() {
            return None;
        }

        let n = self.x_history.len();

        // Standardize data
        let x_std = standardize(&self.x_history);
        let m_std = standardize(&self.m_history);
        let y_std = standardize(&self.y_history);

        // Step 1: Total effect (Y ~ X)
        let (c, c_se) = simple_regression(&x_std, &y_std)?;

        // Step 2: a path (M ~ X)
        let (a, a_se) = simple_regression(&x_std, &m_std)?;

        // Step 3: Direct effect and b path (Y ~ X + M)
        let (c_prime, b, c_prime_se, b_se) = multiple_regression(&x_std, &m_std, &y_std)?;

        // Indirect effect
        let indirect = a * b;

        // Sobel test
        let sobel_se = (b * b * a_se * a_se + a * a * b_se * b_se).sqrt();
        let sobel_z = if sobel_se > 1e-10 {
            indirect / sobel_se
        } else {
            0.0
        };
        let sobel_pvalue = 2.0 * (1.0 - normal_cdf(sobel_z.abs()));

        // Proportion mediated
        let prop_mediated = if c.abs() > 1e-10 {
            indirect / c
        } else {
            0.0
        };

        Some(MediationResults {
            total_effect: c,
            total_effect_se: c_se,
            direct_effect: c_prime,
            direct_effect_se: c_prime_se,
            indirect_effect: indirect,
            indirect_effect_se: sobel_se,
            a_path: a,
            a_path_se: a_se,
            b_path: b,
            b_path_se: b_se,
            sobel_z,
            sobel_pvalue,
            proportion_mediated: prop_mediated,
        })
    }

    /// Generate trading signal based on mediation analysis
    ///
    /// # Arguments
    ///
    /// * `current_x` - Current treatment value
    /// * `current_m` - Current mediator value
    /// * `mediation_threshold` - Minimum proportion mediated to consider channel active
    /// * `signal_threshold` - Minimum expected effect to generate signal
    pub fn generate_signal(
        &self,
        current_x: f64,
        current_m: f64,
        mediation_threshold: f64,
        signal_threshold: f64,
    ) -> Option<TradingSignal> {
        let results = self.analyze()?;

        // Standardize current values
        let x_mean = mean(&self.x_history);
        let x_std_dev = std_dev(&self.x_history);
        let x_norm = (current_x - x_mean) / (x_std_dev + 1e-10);

        let m_mean = mean(&self.m_history);
        let m_std_dev = std_dev(&self.m_history);
        let m_norm = (current_m - m_mean) / (m_std_dev + 1e-10);

        // Expected mediator value given treatment
        let m_expected = results.a_path * x_norm;

        // Is mediator activated (stronger than expected)?
        let mediator_activated = m_norm.abs() > m_expected.abs() * 1.2;

        // Calculate expected effect
        let expected_effect = if mediator_activated && results.proportion_mediated.abs() > mediation_threshold {
            // Full effect expected
            (results.direct_effect + results.indirect_effect) * x_norm
        } else {
            // Only direct effect expected
            results.direct_effect * x_norm
        };

        // Confidence based on mediation strength
        let confidence = if mediator_activated {
            0.8 * expected_effect.abs().min(1.0)
        } else {
            0.5 * expected_effect.abs().min(1.0)
        };

        // Generate signal
        let signal = if expected_effect.abs() < signal_threshold {
            0
        } else if expected_effect > 0.0 {
            1
        } else {
            -1
        };

        Some(TradingSignal {
            signal,
            confidence,
            mediator_activated,
            expected_effect,
        })
    }

    /// Reset the analyzer, clearing all history
    pub fn reset(&mut self) {
        self.x_history.clear();
        self.m_history.clear();
        self.y_history.clear();
    }
}

/// Standardize a vector to zero mean and unit variance
fn standardize(data: &[f64]) -> Vec<f64> {
    let m = mean(data);
    let s = std_dev(data);

    if s < 1e-10 {
        return data.to_vec();
    }

    data.iter().map(|x| (x - m) / s).collect()
}

/// Calculate mean of a slice
fn mean(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    data.iter().sum::<f64>() / data.len() as f64
}

/// Calculate standard deviation of a slice
fn std_dev(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return 0.0;
    }
    let m = mean(data);
    let variance: f64 = data.iter().map(|x| (x - m).powi(2)).sum::<f64>() / data.len() as f64;
    variance.sqrt()
}

/// Simple linear regression: y = b0 + b1*x
/// Returns (b1, se_b1)
fn simple_regression(x: &[f64], y: &[f64]) -> Option<(f64, f64)> {
    let n = x.len();
    if n < 3 {
        return None;
    }

    let x_mean = mean(x);
    let y_mean = mean(y);

    let mut ss_xy = 0.0;
    let mut ss_xx = 0.0;

    for i in 0..n {
        let x_diff = x[i] - x_mean;
        let y_diff = y[i] - y_mean;
        ss_xy += x_diff * y_diff;
        ss_xx += x_diff * x_diff;
    }

    if ss_xx < 1e-10 {
        return None;
    }

    let b1 = ss_xy / ss_xx;
    let b0 = y_mean - b1 * x_mean;

    // Residual standard error
    let mut ss_res = 0.0;
    for i in 0..n {
        let pred = b0 + b1 * x[i];
        let resid = y[i] - pred;
        ss_res += resid * resid;
    }

    let mse = ss_res / (n as f64 - 2.0);
    let se_b1 = (mse / ss_xx).sqrt();

    Some((b1, se_b1))
}

/// Multiple regression: y = b0 + b1*x + b2*m
/// Returns (b1, b2, se_b1, se_b2)
fn multiple_regression(x: &[f64], m: &[f64], y: &[f64]) -> Option<(f64, f64, f64, f64)> {
    let n = x.len();
    if n < 4 {
        return None;
    }

    // Build design matrix [1, x, m] and solve via normal equations
    // Using simplified 3x3 matrix operations

    // Calculate sums
    let sum_x: f64 = x.iter().sum();
    let sum_m: f64 = m.iter().sum();
    let sum_y: f64 = y.iter().sum();
    let sum_xx: f64 = x.iter().map(|xi| xi * xi).sum();
    let sum_mm: f64 = m.iter().map(|mi| mi * mi).sum();
    let sum_xm: f64 = x.iter().zip(m.iter()).map(|(xi, mi)| xi * mi).sum();
    let sum_xy: f64 = x.iter().zip(y.iter()).map(|(xi, yi)| xi * yi).sum();
    let sum_my: f64 = m.iter().zip(y.iter()).map(|(mi, yi)| mi * yi).sum();

    let nf = n as f64;

    // X'X matrix
    let a = [[nf, sum_x, sum_m],
             [sum_x, sum_xx, sum_xm],
             [sum_m, sum_xm, sum_mm]];

    // X'y vector
    let b = [sum_y, sum_xy, sum_my];

    // Solve using Cramer's rule
    let det = determinant_3x3(&a);
    if det.abs() < 1e-10 {
        return None;
    }

    // Solve for beta
    let beta0 = {
        let mut a_i = a;
        for j in 0..3 {
            a_i[j][0] = b[j];
        }
        determinant_3x3(&a_i) / det
    };

    let beta1 = {
        let mut a_i = a;
        for j in 0..3 {
            a_i[j][1] = b[j];
        }
        determinant_3x3(&a_i) / det
    };

    let beta2 = {
        let mut a_i = a;
        for j in 0..3 {
            a_i[j][2] = b[j];
        }
        determinant_3x3(&a_i) / det
    };

    // Calculate residuals and MSE
    let mut ss_res = 0.0;
    for i in 0..n {
        let pred = beta0 + beta1 * x[i] + beta2 * m[i];
        let resid = y[i] - pred;
        ss_res += resid * resid;
    }

    let mse = ss_res / (n as f64 - 3.0);

    // Inverse of X'X for variance-covariance matrix (approximate)
    let inv = invert_3x3(&a)?;

    let se_b1 = (mse * inv[1][1]).sqrt();
    let se_b2 = (mse * inv[2][2]).sqrt();

    Some((beta1, beta2, se_b1, se_b2))
}

/// 3x3 matrix determinant
fn determinant_3x3(a: &[[f64; 3]; 3]) -> f64 {
    a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
}

/// 3x3 matrix inverse
fn invert_3x3(a: &[[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let det = determinant_3x3(a);
    if det.abs() < 1e-10 {
        return None;
    }

    let mut inv = [[0.0; 3]; 3];

    // Cofactors / det
    inv[0][0] = (a[1][1] * a[2][2] - a[1][2] * a[2][1]) / det;
    inv[0][1] = (a[0][2] * a[2][1] - a[0][1] * a[2][2]) / det;
    inv[0][2] = (a[0][1] * a[1][2] - a[0][2] * a[1][1]) / det;
    inv[1][0] = (a[1][2] * a[2][0] - a[1][0] * a[2][2]) / det;
    inv[1][1] = (a[0][0] * a[2][2] - a[0][2] * a[2][0]) / det;
    inv[1][2] = (a[0][2] * a[1][0] - a[0][0] * a[1][2]) / det;
    inv[2][0] = (a[1][0] * a[2][1] - a[1][1] * a[2][0]) / det;
    inv[2][1] = (a[0][1] * a[2][0] - a[0][0] * a[2][1]) / det;
    inv[2][2] = (a[0][0] * a[1][1] - a[0][1] * a[1][0]) / det;

    Some(inv)
}

/// Standard normal CDF approximation
fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / SQRT_2))
}

/// Error function approximation
fn erf(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mediation_analysis_basic() {
        let mut analyzer = MediationAnalyzer::new(100);

        // Generate synthetic data with known mediation structure
        let a_true = 0.5;
        let b_true = 0.4;
        let c_prime_true = 0.3;

        for i in 0..100 {
            let x = (i as f64 - 50.0) / 50.0;  // Uniform(-1, 1)
            let noise_m = (i as f64 * 0.1).sin() * 0.1;  // Deterministic "noise"
            let noise_y = (i as f64 * 0.2).cos() * 0.1;

            let m = a_true * x + noise_m;
            let y = c_prime_true * x + b_true * m + noise_y;

            analyzer.update(x, m, y);
        }

        let results = analyzer.analyze().expect("Should have results");

        // Check that effects are in reasonable range
        assert!(results.total_effect.abs() < 2.0);
        assert!(results.direct_effect.abs() < 2.0);
        assert!(results.indirect_effect.abs() < 2.0);

        // Total should approximately equal Direct + Indirect
        let total_check = (results.total_effect - results.direct_effect - results.indirect_effect).abs();
        assert!(total_check < 0.3, "Total effect decomposition failed: {}", total_check);
    }

    #[test]
    fn test_signal_generation() {
        let mut analyzer = MediationAnalyzer::new(60);

        // Generate data
        for i in 0..60 {
            let x = (i as f64 - 30.0) / 30.0;
            let m = 0.5 * x + (i as f64 * 0.1).sin() * 0.1;
            let y = 0.3 * x + 0.4 * m + (i as f64 * 0.2).cos() * 0.1;

            analyzer.update(x, m, y);
        }

        // Test signal generation
        let signal = analyzer.generate_signal(0.5, 0.3, 0.3, 0.1);
        assert!(signal.is_some());

        let s = signal.unwrap();
        assert!(s.signal >= -1 && s.signal <= 1);
        assert!(s.confidence >= 0.0 && s.confidence <= 1.0);
    }

    #[test]
    fn test_empty_analyzer() {
        let analyzer = MediationAnalyzer::new(60);
        assert!(analyzer.is_empty());
        assert!(!analyzer.can_analyze());
        assert!(analyzer.analyze().is_none());
    }

    #[test]
    fn test_insufficient_data() {
        let mut analyzer = MediationAnalyzer::new(60);

        for i in 0..10 {
            analyzer.update(i as f64, i as f64, i as f64);
        }

        assert_eq!(analyzer.len(), 10);
        assert!(!analyzer.can_analyze());
        assert!(analyzer.analyze().is_none());
    }

    #[test]
    fn test_reset() {
        let mut analyzer = MediationAnalyzer::new(60);

        for i in 0..50 {
            analyzer.update(i as f64, i as f64, i as f64);
        }

        assert_eq!(analyzer.len(), 50);
        analyzer.reset();
        assert!(analyzer.is_empty());
    }

    #[test]
    fn test_mediation_type() {
        let results = MediationResults {
            total_effect: 0.5,
            total_effect_se: 0.1,
            direct_effect: 0.2,
            direct_effect_se: 0.05,  // t = 4, significant
            indirect_effect: 0.3,
            indirect_effect_se: 0.05,
            a_path: 0.5,
            a_path_se: 0.1,
            b_path: 0.6,
            b_path_se: 0.1,
            sobel_z: 3.0,
            sobel_pvalue: 0.003,  // significant
            proportion_mediated: 0.6,
        };

        assert_eq!(results.mediation_type(0.05), MediationType::Partial);
        assert!(results.is_significant(0.05));
    }
}
