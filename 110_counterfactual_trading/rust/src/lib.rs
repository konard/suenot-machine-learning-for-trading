//! Counterfactual Trading Library
//!
//! This library implements counterfactual estimation methods for trading decision analysis.
//! It supports outcome regression, propensity score estimation, and doubly robust estimation.
//!
//! # Example
//!
//! ```rust
//! use counterfactual_trading::{DoublyRobustEstimator, CounterfactualResult};
//! use ndarray::{Array1, Array2};
//!
//! let mut estimator = DoublyRobustEstimator::new();
//! // Fit with training data and estimate counterfactuals
//! ```

use ndarray::{Array1, Array2, Axis};
use std::collections::VecDeque;

/// Result of counterfactual estimation
#[derive(Debug, Clone)]
pub struct CounterfactualResult {
    pub observed_outcome: f64,
    pub counterfactual_outcome: f64,
    pub treatment_effect: f64,
}

/// Result of Average Treatment Effect estimation
#[derive(Debug, Clone)]
pub struct ATEResult {
    pub ate: f64,
    pub se: f64,
    pub ci_low: f64,
    pub ci_high: f64,
}

/// Outcome regression model for counterfactual estimation
pub struct OutcomeRegression {
    coef_treated: Array1<f64>,
    coef_control: Array1<f64>,
    intercept_treated: f64,
    intercept_control: f64,
    fitted: bool,
}

impl OutcomeRegression {
    /// Create a new outcome regression model
    pub fn new() -> Self {
        OutcomeRegression {
            coef_treated: Array1::zeros(0),
            coef_control: Array1::zeros(0),
            intercept_treated: 0.0,
            intercept_control: 0.0,
            fitted: false,
        }
    }

    /// Fit outcome models for treated and control groups
    pub fn fit(&mut self, x: &Array2<f64>, treatment: &Array1<f64>, outcome: &Array1<f64>) {
        let n = x.nrows();

        // Separate treated and control
        let mut x_treated = Vec::new();
        let mut y_treated = Vec::new();
        let mut x_control = Vec::new();
        let mut y_control = Vec::new();

        for i in 0..n {
            if treatment[i] > 0.5 {
                x_treated.push(x.row(i).to_owned());
                y_treated.push(outcome[i]);
            } else {
                x_control.push(x.row(i).to_owned());
                y_control.push(outcome[i]);
            }
        }

        // Fit treated model
        if !x_treated.is_empty() {
            let x_t = stack_rows(&x_treated);
            let y_t = Array1::from_vec(y_treated);
            let (coef, intercept) = ols_regression(&x_t, &y_t);
            self.coef_treated = coef;
            self.intercept_treated = intercept;
        }

        // Fit control model
        if !x_control.is_empty() {
            let x_c = stack_rows(&x_control);
            let y_c = Array1::from_vec(y_control);
            let (coef, intercept) = ols_regression(&x_c, &y_c);
            self.coef_control = coef;
            self.intercept_control = intercept;
        }

        self.fitted = true;
    }

    /// Predict outcome under treatment
    pub fn predict_treated(&self, x: &Array1<f64>) -> f64 {
        if self.coef_treated.len() == 0 {
            return 0.0;
        }
        x.dot(&self.coef_treated) + self.intercept_treated
    }

    /// Predict outcome under control
    pub fn predict_control(&self, x: &Array1<f64>) -> f64 {
        if self.coef_control.len() == 0 {
            return 0.0;
        }
        x.dot(&self.coef_control) + self.intercept_control
    }

    /// Estimate counterfactual outcome
    pub fn estimate_counterfactual(
        &self,
        x: &Array1<f64>,
        treatment: f64,
        observed_outcome: f64,
    ) -> CounterfactualResult {
        let cf_outcome = if treatment > 0.5 {
            self.predict_control(x)
        } else {
            self.predict_treated(x)
        };

        let treatment_effect = if treatment > 0.5 {
            observed_outcome - cf_outcome
        } else {
            cf_outcome - observed_outcome
        };

        CounterfactualResult {
            observed_outcome,
            counterfactual_outcome: cf_outcome,
            treatment_effect,
        }
    }
}

impl Default for OutcomeRegression {
    fn default() -> Self {
        Self::new()
    }
}

/// Propensity score model using logistic regression
pub struct PropensityModel {
    coef: Array1<f64>,
    intercept: f64,
    fitted: bool,
}

impl PropensityModel {
    /// Create a new propensity model
    pub fn new() -> Self {
        PropensityModel {
            coef: Array1::zeros(0),
            intercept: 0.0,
            fitted: false,
        }
    }

    /// Fit logistic regression for propensity scores
    pub fn fit(&mut self, x: &Array2<f64>, treatment: &Array1<f64>) {
        let n = x.nrows();
        let p = x.ncols();

        // Initialize coefficients
        let mut beta = Array1::zeros(p + 1);
        let learning_rate = 0.01;
        let max_iter = 1000;

        // Gradient descent for logistic regression
        for _ in 0..max_iter {
            let mut gradient = Array1::zeros(p + 1);

            for i in 0..n {
                let xi = x.row(i);
                let yi = treatment[i];

                // Linear combination
                let mut z = beta[0];
                for j in 0..p {
                    z += beta[j + 1] * xi[j];
                }

                // Sigmoid
                let prob = 1.0 / (1.0 + (-z).exp());

                // Gradient
                let error = prob - yi;
                gradient[0] += error;
                for j in 0..p {
                    gradient[j + 1] += error * xi[j];
                }
            }

            // Update
            for j in 0..=p {
                beta[j] -= learning_rate * gradient[j] / n as f64;
            }
        }

        self.intercept = beta[0];
        self.coef = beta.slice(ndarray::s![1..]).to_owned();
        self.fitted = true;
    }

    /// Predict propensity score P(T=1|X)
    pub fn predict(&self, x: &Array1<f64>) -> f64 {
        if self.coef.len() == 0 {
            return 0.5;
        }
        let z = self.intercept + x.dot(&self.coef);
        1.0 / (1.0 + (-z).exp())
    }
}

impl Default for PropensityModel {
    fn default() -> Self {
        Self::new()
    }
}

/// Doubly robust estimator for counterfactual inference
pub struct DoublyRobustEstimator {
    outcome_model: OutcomeRegression,
    propensity_model: PropensityModel,
    fitted: bool,
}

impl DoublyRobustEstimator {
    /// Create a new doubly robust estimator
    pub fn new() -> Self {
        DoublyRobustEstimator {
            outcome_model: OutcomeRegression::new(),
            propensity_model: PropensityModel::new(),
            fitted: false,
        }
    }

    /// Fit both outcome and propensity models
    pub fn fit(&mut self, x: &Array2<f64>, treatment: &Array1<f64>, outcome: &Array1<f64>) {
        self.outcome_model.fit(x, treatment, outcome);
        self.propensity_model.fit(x, treatment);
        self.fitted = true;
    }

    /// Estimate Average Treatment Effect using doubly robust estimator
    pub fn estimate_ate(
        &self,
        x: &Array2<f64>,
        treatment: &Array1<f64>,
        outcome: &Array1<f64>,
    ) -> ATEResult {
        let n = x.nrows();

        // Get propensity scores
        let propensity: Vec<f64> = (0..n)
            .map(|i| {
                let p = self.propensity_model.predict(&x.row(i).to_owned());
                p.clamp(0.01, 0.99)
            })
            .collect();

        // Get outcome predictions
        let mu1: Vec<f64> = (0..n)
            .map(|i| self.outcome_model.predict_treated(&x.row(i).to_owned()))
            .collect();
        let mu0: Vec<f64> = (0..n)
            .map(|i| self.outcome_model.predict_control(&x.row(i).to_owned()))
            .collect();

        // Doubly robust estimator
        let mut treated_sum = 0.0;
        let mut control_sum = 0.0;

        for i in 0..n {
            let t = treatment[i];
            let y = outcome[i];
            let e = propensity[i];

            treated_sum += t * y / e - (t - e) / e * mu1[i];
            control_sum += (1.0 - t) * y / (1.0 - e) + (t - e) / (1.0 - e) * mu0[i];
        }

        let ate = treated_sum / n as f64 - control_sum / n as f64;

        // Standard error via influence function
        let mut influence = Vec::with_capacity(n);
        for i in 0..n {
            let t = treatment[i];
            let y = outcome[i];
            let e = propensity[i];

            let treated_term = t * y / e - (t - e) / e * mu1[i];
            let control_term = (1.0 - t) * y / (1.0 - e) + (t - e) / (1.0 - e) * mu0[i];

            influence.push(treated_term - control_term - ate);
        }

        let variance: f64 = influence.iter().map(|x| x * x).sum::<f64>() / n as f64;
        let se = (variance / n as f64).sqrt();

        ATEResult {
            ate,
            se,
            ci_low: ate - 1.96 * se,
            ci_high: ate + 1.96 * se,
        }
    }

    /// Estimate individual counterfactual outcome
    pub fn estimate_counterfactual(
        &self,
        x: &Array1<f64>,
        treatment: f64,
        observed_outcome: f64,
    ) -> CounterfactualResult {
        self.outcome_model
            .estimate_counterfactual(x, treatment, observed_outcome)
    }
}

impl Default for DoublyRobustEstimator {
    fn default() -> Self {
        Self::new()
    }
}

/// Trading decision with counterfactual analysis
#[derive(Debug, Clone)]
pub struct TradeDecision {
    pub timestamp: i64,
    pub action: i32,
    pub observed_return: f64,
    pub counterfactual_return: f64,
    pub treatment_effect: f64,
}

/// Strategy attribution results
#[derive(Debug, Clone)]
pub struct StrategyAttribution {
    pub total_return: f64,
    pub market_component: f64,
    pub strategy_alpha: f64,
    pub alpha_contribution_pct: f64,
}

/// Regret metrics
#[derive(Debug, Clone)]
pub struct RegretMetrics {
    pub total_regret: f64,
    pub mean_regret: f64,
    pub max_regret: f64,
    pub regret_frequency: f64,
}

/// Counterfactual trading strategy
pub struct CounterfactualTradingStrategy {
    cf_estimator: DoublyRobustEstimator,
    decision_history: VecDeque<TradeDecision>,
    max_history: usize,
}

impl CounterfactualTradingStrategy {
    /// Create a new counterfactual trading strategy
    pub fn new(cf_estimator: DoublyRobustEstimator, max_history: usize) -> Self {
        CounterfactualTradingStrategy {
            cf_estimator,
            decision_history: VecDeque::with_capacity(max_history),
            max_history,
        }
    }

    /// Evaluate a trading decision using counterfactual analysis
    pub fn evaluate_decision(
        &mut self,
        features: &Array1<f64>,
        action: i32,
        observed_return: f64,
        timestamp: i64,
    ) -> TradeDecision {
        let treatment = if action != 0 { 1.0 } else { 0.0 };

        let cf_result = self
            .cf_estimator
            .estimate_counterfactual(features, treatment, observed_return);

        let decision = TradeDecision {
            timestamp,
            action,
            observed_return,
            counterfactual_return: cf_result.counterfactual_outcome,
            treatment_effect: cf_result.treatment_effect,
        };

        if self.decision_history.len() >= self.max_history {
            self.decision_history.pop_front();
        }
        self.decision_history.push_back(decision.clone());

        decision
    }

    /// Compute strategy attribution
    pub fn compute_attribution(&self) -> StrategyAttribution {
        let total_return: f64 = self.decision_history.iter().map(|d| d.observed_return).sum();

        let cf_return: f64 = self
            .decision_history
            .iter()
            .map(|d| d.counterfactual_return)
            .sum();

        let strategy_alpha = total_return - cf_return;

        let alpha_contribution = if total_return.abs() > 1e-10 {
            strategy_alpha / total_return.abs() * 100.0
        } else {
            0.0
        };

        StrategyAttribution {
            total_return,
            market_component: cf_return,
            strategy_alpha,
            alpha_contribution_pct: alpha_contribution,
        }
    }

    /// Compute counterfactual regret
    pub fn compute_regret(&self) -> RegretMetrics {
        let regrets: Vec<f64> = self
            .decision_history
            .iter()
            .map(|d| (d.counterfactual_return - d.observed_return).max(0.0))
            .collect();

        if regrets.is_empty() {
            return RegretMetrics {
                total_regret: 0.0,
                mean_regret: 0.0,
                max_regret: 0.0,
                regret_frequency: 0.0,
            };
        }

        let total_regret: f64 = regrets.iter().sum();
        let mean_regret = total_regret / regrets.len() as f64;
        let max_regret = regrets.iter().cloned().fold(0.0, f64::max);
        let regret_frequency =
            regrets.iter().filter(|&&r| r > 0.0).count() as f64 / regrets.len() as f64;

        RegretMetrics {
            total_regret,
            mean_regret,
            max_regret,
            regret_frequency,
        }
    }
}

/// Backtest result metrics
#[derive(Debug, Clone)]
pub struct BacktestResult {
    pub total_return: f64,
    pub counterfactual_return: f64,
    pub strategy_alpha: f64,
    pub sharpe_ratio: f64,
    pub counterfactual_sharpe: f64,
    pub regret: f64,
    pub max_drawdown: f64,
    pub win_rate: f64,
}

/// Compute Sharpe ratio
pub fn compute_sharpe_ratio(returns: &[f64], periods_per_year: f64) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }

    let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
    let variance: f64 = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;
    let std_dev = variance.sqrt();

    if std_dev < 1e-10 {
        return 0.0;
    }

    mean / std_dev * periods_per_year.sqrt()
}

/// Compute maximum drawdown
pub fn compute_max_drawdown(returns: &[f64]) -> f64 {
    let mut cumulative = 1.0;
    let mut running_max = 1.0;
    let mut max_dd = 0.0;

    for &ret in returns {
        cumulative *= 1.0 + ret;
        running_max = running_max.max(cumulative);
        let dd = (cumulative - running_max) / running_max;
        max_dd = max_dd.min(dd);
    }

    max_dd
}

/// Compute win rate
pub fn compute_win_rate(returns: &[f64]) -> f64 {
    let wins = returns.iter().filter(|&&r| r > 0.0).count();
    let total = returns.iter().filter(|&&r| r != 0.0).count();

    if total == 0 {
        return 0.0;
    }

    wins as f64 / total as f64
}

/// Compute total regret
pub fn compute_regret(observed: &[f64], counterfactual: &[f64]) -> f64 {
    observed
        .iter()
        .zip(counterfactual.iter())
        .map(|(o, c)| (c - o).max(0.0))
        .sum()
}

// Helper functions

fn ols_regression(x: &Array2<f64>, y: &Array1<f64>) -> (Array1<f64>, f64) {
    let n = x.nrows();
    let p = x.ncols();

    if n == 0 || p == 0 {
        return (Array1::zeros(p), 0.0);
    }

    // Add intercept column
    let mut x_aug = Array2::ones((n, p + 1));
    for i in 0..n {
        for j in 0..p {
            x_aug[[i, j + 1]] = x[[i, j]];
        }
    }

    // Normal equations: (X'X)^-1 X'y
    let xt = x_aug.t();
    let xtx = xt.dot(&x_aug);
    let xty = xt.dot(y);

    let beta = solve_linear_system(&xtx, &xty);

    let intercept = beta[0];
    let coef = beta.slice(ndarray::s![1..]).to_owned();

    (coef, intercept)
}

fn solve_linear_system(a: &Array2<f64>, b: &Array1<f64>) -> Array1<f64> {
    let n = a.nrows();

    if n == 0 {
        return Array1::zeros(0);
    }

    let mut aug = Array2::zeros((n, n + 1));

    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = a[[i, j]];
        }
        aug[[i, n]] = b[i];
    }

    // Gauss-Jordan elimination
    for i in 0..n {
        // Find pivot
        let mut max_row = i;
        for k in (i + 1)..n {
            if aug[[k, i]].abs() > aug[[max_row, i]].abs() {
                max_row = k;
            }
        }

        // Swap rows
        for j in 0..=n {
            let temp = aug[[i, j]];
            aug[[i, j]] = aug[[max_row, j]];
            aug[[max_row, j]] = temp;
        }

        // Eliminate
        if aug[[i, i]].abs() > 1e-10 {
            for k in (i + 1)..n {
                let factor = aug[[k, i]] / aug[[i, i]];
                for j in i..=n {
                    aug[[k, j]] -= factor * aug[[i, j]];
                }
            }
        }
    }

    // Back substitution
    let mut x = Array1::zeros(n);
    for i in (0..n).rev() {
        x[i] = aug[[i, n]];
        for j in (i + 1)..n {
            x[i] -= aug[[i, j]] * x[j];
        }
        if aug[[i, i]].abs() > 1e-10 {
            x[i] /= aug[[i, i]];
        }
    }

    x
}

fn stack_rows(rows: &[Array1<f64>]) -> Array2<f64> {
    let n = rows.len();
    if n == 0 {
        return Array2::zeros((0, 0));
    }
    let p = rows[0].len();
    let mut result = Array2::zeros((n, p));
    for (i, row) in rows.iter().enumerate() {
        for j in 0..p {
            result[[i, j]] = row[j];
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_outcome_regression() {
        let x = Array2::from_shape_vec((10, 2), vec![
            1.0, 0.5, 2.0, 0.3, 1.5, 0.4, 2.5, 0.6, 1.2, 0.35,
            1.8, 0.45, 2.2, 0.55, 1.3, 0.32, 1.9, 0.48, 2.1, 0.52
        ]).unwrap();
        let treatment = Array1::from_vec(vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]);
        let outcome = Array1::from_vec(vec![0.05, 0.01, 0.04, 0.02, 0.03, 0.015, 0.06, 0.01, 0.04, 0.02]);

        let mut model = OutcomeRegression::new();
        model.fit(&x, &treatment, &outcome);

        assert!(model.fitted);
    }

    #[test]
    fn test_doubly_robust_estimator() {
        let x = Array2::from_shape_vec((20, 2), (0..40).map(|i| i as f64 * 0.1).collect()).unwrap();
        let treatment = Array1::from_vec((0..20).map(|i| if i % 2 == 0 { 1.0 } else { 0.0 }).collect());
        let outcome = Array1::from_vec((0..20).map(|i| (i as f64) * 0.01).collect());

        let mut estimator = DoublyRobustEstimator::new();
        estimator.fit(&x, &treatment, &outcome);

        let ate = estimator.estimate_ate(&x, &treatment, &outcome);

        // Just check it runs without error
        assert!(!ate.ate.is_nan());
    }

    #[test]
    fn test_sharpe_ratio() {
        let returns = vec![0.01, -0.005, 0.02, -0.01, 0.015];
        let sharpe = compute_sharpe_ratio(&returns, 252.0);
        assert!(!sharpe.is_nan());
    }

    #[test]
    fn test_max_drawdown() {
        let returns = vec![0.1, -0.2, 0.05, -0.1, 0.15];
        let dd = compute_max_drawdown(&returns);
        assert!(dd <= 0.0);
    }
}
