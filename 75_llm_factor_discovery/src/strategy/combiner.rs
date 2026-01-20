//! Factor combination strategies
//!
//! Methods for combining multiple factors into composite signals.

use super::signals::FactorSignal;
use serde::{Deserialize, Serialize};

/// Method for combining factors
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum CombineMethod {
    /// Simple equal weighting
    EqualWeight,
    /// IC-weighted combination
    IcWeighted,
    /// IC_IR-weighted combination
    IcIrWeighted,
    /// Risk parity weighting
    RiskParity,
    /// Custom weights
    Custom,
}

/// Factor combiner
pub struct FactorCombiner {
    method: CombineMethod,
    custom_weights: Option<Vec<f64>>,
}

impl FactorCombiner {
    /// Create a new combiner with equal weights
    pub fn equal_weight() -> Self {
        Self {
            method: CombineMethod::EqualWeight,
            custom_weights: None,
        }
    }

    /// Create IC-weighted combiner
    pub fn ic_weighted() -> Self {
        Self {
            method: CombineMethod::IcWeighted,
            custom_weights: None,
        }
    }

    /// Create IC_IR-weighted combiner
    pub fn ic_ir_weighted() -> Self {
        Self {
            method: CombineMethod::IcIrWeighted,
            custom_weights: None,
        }
    }

    /// Create with custom weights
    pub fn with_weights(weights: Vec<f64>) -> Self {
        Self {
            method: CombineMethod::Custom,
            custom_weights: Some(weights),
        }
    }

    /// Combine factor values into a single signal
    pub fn combine(&self, factor_values: &[f64], ic_values: Option<&[f64]>) -> f64 {
        if factor_values.is_empty() {
            return 0.0;
        }

        match self.method {
            CombineMethod::EqualWeight => {
                factor_values.iter().sum::<f64>() / factor_values.len() as f64
            }
            CombineMethod::IcWeighted => {
                if let Some(ics) = ic_values {
                    self.weighted_combine(factor_values, ics)
                } else {
                    // Fall back to equal weight
                    factor_values.iter().sum::<f64>() / factor_values.len() as f64
                }
            }
            CombineMethod::IcIrWeighted => {
                if let Some(ics) = ic_values {
                    // Use absolute IC as weights
                    let abs_weights: Vec<f64> = ics.iter().map(|ic| ic.abs()).collect();
                    self.weighted_combine(factor_values, &abs_weights)
                } else {
                    factor_values.iter().sum::<f64>() / factor_values.len() as f64
                }
            }
            CombineMethod::RiskParity => {
                // Simple risk parity: equal volatility contribution
                // Would need volatility estimates in practice
                factor_values.iter().sum::<f64>() / factor_values.len() as f64
            }
            CombineMethod::Custom => {
                if let Some(ref weights) = self.custom_weights {
                    self.weighted_combine(factor_values, weights)
                } else {
                    factor_values.iter().sum::<f64>() / factor_values.len() as f64
                }
            }
        }
    }

    /// Weighted combination helper
    fn weighted_combine(&self, values: &[f64], weights: &[f64]) -> f64 {
        if values.len() != weights.len() {
            return 0.0;
        }

        let total_weight: f64 = weights.iter().map(|w| w.abs()).sum();
        if total_weight == 0.0 {
            return 0.0;
        }

        values
            .iter()
            .zip(weights.iter())
            .map(|(v, w)| v * w)
            .sum::<f64>()
            / total_weight
    }

    /// Combine factor signals
    pub fn combine_signals(&self, signals: &[FactorSignal], ic_values: Option<&[f64]>) -> f64 {
        let values: Vec<f64> = signals.iter().map(|s| s.z_score).collect();
        self.combine(&values, ic_values)
    }

    /// Calculate optimal weights using historical data
    pub fn optimize_weights(
        factor_values: &[Vec<f64>],
        forward_returns: &[f64],
    ) -> Vec<f64> {
        if factor_values.is_empty() || forward_returns.is_empty() {
            return vec![];
        }

        // Simple optimization: use correlation with returns as weights
        let weights: Vec<f64> = factor_values
            .iter()
            .map(|fv| {
                // Calculate correlation with forward returns
                if fv.len() != forward_returns.len() {
                    return 0.0;
                }

                let n = fv.len() as f64;
                let mean_f: f64 = fv.iter().sum::<f64>() / n;
                let mean_r: f64 = forward_returns.iter().sum::<f64>() / n;

                let mut cov = 0.0;
                let mut var_f = 0.0;
                let mut var_r = 0.0;

                for (f, r) in fv.iter().zip(forward_returns.iter()) {
                    let df = f - mean_f;
                    let dr = r - mean_r;
                    cov += df * dr;
                    var_f += df * df;
                    var_r += dr * dr;
                }

                if var_f > 0.0 && var_r > 0.0 {
                    cov / (var_f.sqrt() * var_r.sqrt())
                } else {
                    0.0
                }
            })
            .collect();

        weights
    }
}

impl Default for FactorCombiner {
    fn default() -> Self {
        Self::equal_weight()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_equal_weight_combine() {
        let combiner = FactorCombiner::equal_weight();
        let values = vec![1.0, 2.0, 3.0];

        let result = combiner.combine(&values, None);
        assert!((result - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_custom_weight_combine() {
        let combiner = FactorCombiner::with_weights(vec![2.0, 1.0]);
        let values = vec![3.0, 6.0];

        // (3*2 + 6*1) / 3 = 12/3 = 4
        let result = combiner.combine(&values, None);
        assert!((result - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_ic_weighted_combine() {
        let combiner = FactorCombiner::ic_weighted();
        let values = vec![1.0, 2.0];
        let ics = vec![0.05, 0.01];

        // Higher IC factor should have more weight
        let result = combiner.combine(&values, Some(&ics));
        // (1*0.05 + 2*0.01) / 0.06 = 0.07/0.06 = 1.167
        assert!(result < 1.5);
        assert!(result > 1.0);
    }
}
