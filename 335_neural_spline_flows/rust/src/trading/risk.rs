//! Risk management using Neural Spline Flows
//!
//! This module provides risk metrics computation from learned distributions:
//! - Value at Risk (VaR)
//! - Conditional Value at Risk (CVaR)
//! - Position sizing based on risk

use crate::flow::NeuralSplineFlow;
use ndarray::Array1;
use serde::{Deserialize, Serialize};

/// Risk metrics computed from the distribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskMetrics {
    /// Value at Risk at specified confidence level
    pub var: f64,
    /// Conditional Value at Risk (Expected Shortfall)
    pub cvar: f64,
    /// Expected return
    pub expected_return: f64,
    /// Return standard deviation
    pub return_std: f64,
    /// Skewness of return distribution
    pub skewness: f64,
    /// Kurtosis of return distribution
    pub kurtosis: f64,
    /// Confidence level used
    pub confidence_level: f64,
}

/// Risk manager configuration
#[derive(Debug, Clone)]
pub struct RiskManagerConfig {
    /// Index of return feature in feature vector
    pub return_feature_idx: usize,
    /// Confidence level for VaR/CVaR (e.g., 0.95 for 95%)
    pub confidence_level: f64,
    /// Number of samples for risk estimation
    pub num_samples: usize,
    /// Maximum allowed position size
    pub max_position: f64,
    /// Risk scaling factor
    pub risk_scale: f64,
}

impl Default for RiskManagerConfig {
    fn default() -> Self {
        Self {
            return_feature_idx: 0,
            confidence_level: 0.95,
            num_samples: 10000,
            max_position: 1.0,
            risk_scale: 1.0,
        }
    }
}

/// Risk manager using Neural Spline Flow
pub struct RiskManager {
    /// NSF model
    model: NeuralSplineFlow,
    /// Configuration
    config: RiskManagerConfig,
}

impl RiskManager {
    /// Create a new risk manager
    pub fn new(model: NeuralSplineFlow, config: RiskManagerConfig) -> Self {
        Self { model, config }
    }

    /// Create with default configuration
    pub fn with_defaults(model: NeuralSplineFlow) -> Self {
        Self::new(model, RiskManagerConfig::default())
    }

    /// Compute risk metrics from the learned distribution
    pub fn compute_risk_metrics(&self) -> RiskMetrics {
        // Sample from the distribution
        let samples = self.model.sample(self.config.num_samples);
        let returns: Vec<f64> = samples
            .column(self.config.return_feature_idx)
            .iter()
            .cloned()
            .collect();

        // Sort returns for quantile computation
        let mut sorted_returns = returns.clone();
        sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Compute VaR (negative quantile)
        let var_idx = ((1.0 - self.config.confidence_level) * self.config.num_samples as f64) as usize;
        let var = sorted_returns[var_idx.min(sorted_returns.len() - 1)];

        // Compute CVaR (average of returns below VaR)
        let tail_returns: Vec<f64> = sorted_returns[..=var_idx].to_vec();
        let cvar = if tail_returns.is_empty() {
            var
        } else {
            tail_returns.iter().sum::<f64>() / tail_returns.len() as f64
        };

        // Compute moments
        let n = returns.len() as f64;
        let expected_return = returns.iter().sum::<f64>() / n;

        let variance = returns.iter().map(|r| (r - expected_return).powi(2)).sum::<f64>() / n;
        let return_std = variance.sqrt();

        // Skewness
        let skewness = if return_std > 0.0 {
            let m3 = returns
                .iter()
                .map(|r| ((r - expected_return) / return_std).powi(3))
                .sum::<f64>()
                / n;
            m3
        } else {
            0.0
        };

        // Kurtosis (excess kurtosis)
        let kurtosis = if return_std > 0.0 {
            let m4 = returns
                .iter()
                .map(|r| ((r - expected_return) / return_std).powi(4))
                .sum::<f64>()
                / n;
            m4 - 3.0
        } else {
            0.0
        };

        RiskMetrics {
            var,
            cvar,
            expected_return,
            return_std,
            skewness,
            kurtosis,
            confidence_level: self.config.confidence_level,
        }
    }

    /// Compute position size based on risk
    ///
    /// Uses inverse of CVaR to scale position size
    pub fn compute_position_size(&self, base_signal: f64) -> f64 {
        let metrics = self.compute_risk_metrics();

        // Scale by inverse of tail risk
        let risk_adjustment = 1.0 / (metrics.cvar.abs() + 0.01);
        let position = base_signal * risk_adjustment * self.config.risk_scale;

        // Clip to maximum position
        position.clamp(-self.config.max_position, self.config.max_position)
    }

    /// Check if current risk is acceptable
    pub fn is_risk_acceptable(&self, max_var: f64) -> bool {
        let metrics = self.compute_risk_metrics();
        metrics.var.abs() <= max_var
    }

    /// Get quantile of the return distribution
    pub fn get_return_quantile(&self, quantile: f64) -> f64 {
        assert!(quantile >= 0.0 && quantile <= 1.0);

        let samples = self.model.sample(self.config.num_samples);
        let mut returns: Vec<f64> = samples
            .column(self.config.return_feature_idx)
            .iter()
            .cloned()
            .collect();

        returns.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let idx = (quantile * (returns.len() - 1) as f64) as usize;
        returns[idx]
    }

    /// Get probability of positive return
    pub fn prob_positive_return(&self) -> f64 {
        let samples = self.model.sample(self.config.num_samples);
        let returns: Vec<f64> = samples
            .column(self.config.return_feature_idx)
            .iter()
            .cloned()
            .collect();

        let positive_count = returns.iter().filter(|&&r| r > 0.0).count();
        positive_count as f64 / returns.len() as f64
    }

    /// Get probability of return exceeding threshold
    pub fn prob_return_above(&self, threshold: f64) -> f64 {
        let samples = self.model.sample(self.config.num_samples);
        let returns: Vec<f64> = samples
            .column(self.config.return_feature_idx)
            .iter()
            .cloned()
            .collect();

        let count = returns.iter().filter(|&&r| r > threshold).count();
        count as f64 / returns.len() as f64
    }

    /// Get reference to the underlying model
    pub fn model(&self) -> &NeuralSplineFlow {
        &self.model
    }

    /// Get configuration
    pub fn config(&self) -> &RiskManagerConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flow::NSFConfig;

    #[test]
    fn test_risk_metrics() {
        let config = NSFConfig::new(8);
        let model = NeuralSplineFlow::new(config);
        let risk_manager = RiskManager::with_defaults(model);

        let metrics = risk_manager.compute_risk_metrics();

        assert!(metrics.var.is_finite());
        assert!(metrics.cvar.is_finite());
        assert!(metrics.expected_return.is_finite());
        assert!(metrics.return_std >= 0.0);
    }

    #[test]
    fn test_position_sizing() {
        let config = NSFConfig::new(8);
        let model = NeuralSplineFlow::new(config);
        let risk_manager = RiskManager::with_defaults(model);

        let position = risk_manager.compute_position_size(0.5);

        assert!(position.abs() <= 1.0);
    }

    #[test]
    fn test_quantiles() {
        let config = NSFConfig::new(8);
        let model = NeuralSplineFlow::new(config);
        let risk_manager = RiskManager::with_defaults(model);

        let q25 = risk_manager.get_return_quantile(0.25);
        let q50 = risk_manager.get_return_quantile(0.50);
        let q75 = risk_manager.get_return_quantile(0.75);

        assert!(q25 <= q50);
        assert!(q50 <= q75);
    }
}
