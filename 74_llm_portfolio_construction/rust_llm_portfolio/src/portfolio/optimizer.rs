//! Portfolio optimization algorithms

use crate::portfolio::types::{Portfolio, PortfolioConstraints};
use ndarray::{Array1, Array2};
use std::collections::HashMap;

/// Optimization result with metrics
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    pub portfolio: Portfolio,
    pub expected_return: f64,
    pub volatility: f64,
    pub sharpe_ratio: f64,
    pub success: bool,
}

/// Mean-variance portfolio optimizer
pub struct MeanVarianceOptimizer {
    risk_free_rate: f64,
    max_iterations: usize,
}

impl Default for MeanVarianceOptimizer {
    fn default() -> Self {
        Self::new(0.04)
    }
}

impl MeanVarianceOptimizer {
    /// Create a new optimizer
    pub fn new(risk_free_rate: f64) -> Self {
        Self {
            risk_free_rate,
            max_iterations: 1000,
        }
    }

    /// Optimize portfolio using gradient descent
    pub fn optimize(
        &self,
        expected_returns: &Array1<f64>,
        covariance_matrix: &Array2<f64>,
        symbols: &[String],
        llm_scores: Option<&Array1<f64>>,
        constraints: &PortfolioConstraints,
    ) -> OptimizationResult {
        let n_assets = expected_returns.len();

        // Blend LLM scores with expected returns if provided
        let adjusted_returns = if let Some(scores) = llm_scores {
            let mean_score = scores.mean().unwrap_or(5.0);
            let std_score = scores.std(0.0);

            if std_score > 0.0 {
                let normalized_scores = (scores - mean_score) / std_score;
                let blend_weight = 0.3;
                expected_returns * (1.0 - blend_weight) + &normalized_scores * blend_weight * 0.01
            } else {
                expected_returns.clone()
            }
        } else {
            expected_returns.clone()
        };

        // Initialize with equal weights
        let mut weights = Array1::from_elem(n_assets, 1.0 / n_assets as f64);

        // Simple gradient descent for Sharpe ratio maximization
        let learning_rate = 0.01;

        for _ in 0..self.max_iterations {
            // Calculate portfolio metrics
            let port_return = weights.dot(&adjusted_returns);
            let port_variance = weights.dot(&covariance_matrix.dot(&weights));
            let port_vol = port_variance.sqrt();

            if port_vol < 1e-10 {
                break;
            }

            // Calculate gradient of Sharpe ratio
            let excess_return = port_return - self.risk_free_rate;
            let grad_return = &adjusted_returns;
            let grad_variance = &(2.0 * covariance_matrix.dot(&weights));

            // Gradient of Sharpe = (grad_return * vol - excess_return * grad_vol) / vol^2
            let grad_vol = grad_variance / (2.0 * port_vol);
            let grad_sharpe = (grad_return * port_vol - excess_return * &grad_vol) / port_variance;

            // Update weights
            weights = &weights + learning_rate * &grad_sharpe;

            // Apply constraints
            weights = self.apply_constraints(&weights, constraints);
        }

        // Calculate final metrics
        let port_return = weights.dot(&adjusted_returns);
        let port_variance = weights.dot(&covariance_matrix.dot(&weights));
        let port_vol = port_variance.sqrt();
        let sharpe = if port_vol > 0.0 {
            (port_return - self.risk_free_rate) / port_vol
        } else {
            0.0
        };

        // Create portfolio
        let weight_map: HashMap<String, f64> = symbols
            .iter()
            .zip(weights.iter())
            .map(|(s, w)| (s.clone(), w.max(0.0)))
            .collect();

        let mut portfolio = Portfolio::new(weight_map);
        portfolio.metadata.insert("method".to_string(), "mean_variance".to_string());

        OptimizationResult {
            portfolio,
            expected_return: port_return,
            volatility: port_vol,
            sharpe_ratio: sharpe,
            success: true,
        }
    }

    /// Apply constraints to weights
    fn apply_constraints(
        &self,
        weights: &Array1<f64>,
        constraints: &PortfolioConstraints,
    ) -> Array1<f64> {
        let mut constrained = weights.clone();

        // Apply bounds
        for w in constrained.iter_mut() {
            if constraints.long_only && *w < constraints.min_weight {
                *w = constraints.min_weight;
            }
            if *w > constraints.max_weight {
                *w = constraints.max_weight;
            }
        }

        // Normalize to sum to 1
        let sum: f64 = constrained.iter().sum();
        if sum > 0.0 {
            constrained /= sum;
        }

        constrained
    }

    /// Risk parity portfolio allocation
    pub fn risk_parity(
        &self,
        covariance_matrix: &Array2<f64>,
        symbols: &[String],
    ) -> OptimizationResult {
        let n_assets = covariance_matrix.nrows();

        // Initialize with equal weights
        let mut weights = Array1::from_elem(n_assets, 1.0 / n_assets as f64);
        let target_risk_contrib = 1.0 / n_assets as f64;

        let learning_rate = 0.01;

        for _ in 0..self.max_iterations {
            let port_variance = weights.dot(&covariance_matrix.dot(&weights));
            let port_vol = port_variance.sqrt();

            if port_vol < 1e-10 {
                break;
            }

            // Calculate marginal risk contribution
            let marginal_contrib = covariance_matrix.dot(&weights);
            let risk_contrib = &weights * &marginal_contrib / port_vol;

            // Calculate gradient towards equal risk contribution
            let mut gradient = Array1::zeros(n_assets);
            for i in 0..n_assets {
                let actual_contrib = risk_contrib[i] / port_vol;
                gradient[i] = actual_contrib - target_risk_contrib;
            }

            // Update weights
            weights = &weights - learning_rate * &gradient;

            // Ensure positive weights
            for w in weights.iter_mut() {
                if *w < 0.01 {
                    *w = 0.01;
                }
            }

            // Normalize
            let sum: f64 = weights.iter().sum();
            weights /= sum;
        }

        // Calculate final metrics
        let port_variance = weights.dot(&covariance_matrix.dot(&weights));
        let port_vol = port_variance.sqrt();

        // Create portfolio
        let weight_map: HashMap<String, f64> = symbols
            .iter()
            .zip(weights.iter())
            .map(|(s, w)| (s.clone(), w.max(0.0)))
            .collect();

        let mut portfolio = Portfolio::new(weight_map);
        portfolio.metadata.insert("method".to_string(), "risk_parity".to_string());

        OptimizationResult {
            portfolio,
            expected_return: 0.0, // Not calculated for risk parity
            volatility: port_vol,
            sharpe_ratio: 0.0,
            success: true,
        }
    }
}

/// Generate portfolio from LLM scores
pub fn score_weighted_portfolio(
    scores: &[(String, f64)], // (symbol, composite_score)
    constraints: &PortfolioConstraints,
) -> Portfolio {
    // Filter by minimum score
    let mut valid_scores: Vec<_> = scores
        .iter()
        .filter(|(_, score)| *score >= constraints.min_score)
        .cloned()
        .collect();

    if valid_scores.is_empty() {
        valid_scores = scores.to_vec();
    }

    // Sort by score (descending)
    valid_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Take top N assets
    let selected: Vec<_> = valid_scores
        .into_iter()
        .take(constraints.max_assets)
        .collect();

    // Calculate weights proportional to scores
    let total_score: f64 = selected.iter().map(|(_, s)| *s).sum();

    let weights: HashMap<String, f64> = if total_score > 0.0 {
        selected
            .into_iter()
            .map(|(symbol, score)| {
                let raw_weight = score / total_score;
                let weight = raw_weight
                    .max(constraints.min_weight)
                    .min(constraints.max_weight);
                (symbol, weight)
            })
            .collect()
    } else {
        // Equal weight if all scores are 0
        let weight = 1.0 / selected.len() as f64;
        selected
            .into_iter()
            .map(|(symbol, _)| (symbol, weight))
            .collect()
    };

    let mut portfolio = Portfolio::new(weights);
    portfolio.metadata.insert("method".to_string(), "score_weighted".to_string());
    portfolio
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_score_weighted_portfolio() {
        let scores = vec![
            ("BTC".to_string(), 8.0),
            ("ETH".to_string(), 7.0),
            ("SOL".to_string(), 6.0),
            ("BNB".to_string(), 5.0),
        ];

        let constraints = PortfolioConstraints {
            max_weight: 0.40,
            min_weight: 0.10,
            max_assets: 4,
            min_score: 4.0,
            long_only: true,
        };

        let portfolio = score_weighted_portfolio(&scores, &constraints);

        // Check total sums to 1
        let total: f64 = portfolio.weights.values().sum();
        assert!((total - 1.0).abs() < 0.0001);

        // Check BTC has highest weight
        assert!(portfolio.get_weight("BTC") >= portfolio.get_weight("ETH"));
    }

    #[test]
    fn test_mean_variance_optimizer() {
        let optimizer = MeanVarianceOptimizer::new(0.04);

        let expected_returns = Array1::from_vec(vec![0.20, 0.15, 0.10]);
        let covariance_matrix = Array2::from_shape_vec(
            (3, 3),
            vec![
                0.04, 0.01, 0.005,
                0.01, 0.03, 0.008,
                0.005, 0.008, 0.02,
            ],
        )
        .unwrap();

        let symbols = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let constraints = PortfolioConstraints::default();

        let result = optimizer.optimize(
            &expected_returns,
            &covariance_matrix,
            &symbols,
            None,
            &constraints,
        );

        assert!(result.success);
        let total: f64 = result.portfolio.weights.values().sum();
        assert!((total - 1.0).abs() < 0.01);
    }
}
