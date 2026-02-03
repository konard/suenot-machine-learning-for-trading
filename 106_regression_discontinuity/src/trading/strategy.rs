//! RDD-based trading strategy implementation.

use crate::data::indicators::compute_rsi;
use crate::model::rdd::{Kernel, RDDModel, RDDResults};
use crate::trading::signals::{SignalStrength, TradingSignal};
use crate::{RDDError, Result};

/// RDD-based trading strategy.
#[derive(Debug, Clone)]
pub struct RDDStrategy {
    /// Name of the strategy
    pub name: String,
    /// Cutoff/threshold value
    pub cutoff: f64,
    /// Bandwidth around cutoff
    pub bandwidth: f64,
    /// Kernel function
    pub kernel: Kernel,
    /// Minimum observations required
    pub min_observations: usize,
    /// Minimum effect size to trade
    pub min_effect: f64,
    /// Maximum p-value to trade
    pub max_p_value: f64,
    /// Holding period in bars
    pub holding_period: usize,
    /// Fitted RDD results (if any)
    rdd_results: Option<RDDResults>,
}

impl RDDStrategy {
    /// Create a new RDD strategy.
    pub fn new(name: &str, cutoff: f64, bandwidth: f64) -> Self {
        Self {
            name: name.to_string(),
            cutoff,
            bandwidth,
            kernel: Kernel::Triangular,
            min_observations: 100,
            min_effect: 0.001,
            max_p_value: 0.1,
            holding_period: 24,
            rdd_results: None,
        }
    }

    /// Create an RSI oversold strategy.
    pub fn rsi_oversold() -> Self {
        Self {
            name: "RSI Oversold".to_string(),
            cutoff: 30.0,
            bandwidth: 5.0,
            kernel: Kernel::Triangular,
            min_observations: 100,
            min_effect: 0.001,
            max_p_value: 0.1,
            holding_period: 24,
            rdd_results: None,
        }
    }

    /// Create an RSI overbought strategy.
    pub fn rsi_overbought() -> Self {
        Self {
            name: "RSI Overbought".to_string(),
            cutoff: 70.0,
            bandwidth: 5.0,
            kernel: Kernel::Triangular,
            min_observations: 100,
            min_effect: 0.001,
            max_p_value: 0.1,
            holding_period: 24,
            rdd_results: None,
        }
    }

    /// Create a funding rate strategy.
    pub fn funding_rate(threshold: f64) -> Self {
        Self {
            name: format!("Funding Rate {:.0}%", threshold),
            cutoff: threshold,
            bandwidth: 10.0,
            kernel: Kernel::Triangular,
            min_observations: 50,
            min_effect: 0.0005,
            max_p_value: 0.1,
            holding_period: 8,
            rdd_results: None,
        }
    }

    /// Set kernel function.
    pub fn with_kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
        self
    }

    /// Set minimum observations.
    pub fn with_min_observations(mut self, n: usize) -> Self {
        self.min_observations = n;
        self
    }

    /// Set minimum effect size.
    pub fn with_min_effect(mut self, effect: f64) -> Self {
        self.min_effect = effect;
        self
    }

    /// Set maximum p-value.
    pub fn with_max_p_value(mut self, p: f64) -> Self {
        self.max_p_value = p;
        self
    }

    /// Set holding period.
    pub fn with_holding_period(mut self, periods: usize) -> Self {
        self.holding_period = periods;
        self
    }

    /// Fit the RDD model on historical data.
    pub fn fit(&mut self, running_var: &[f64], outcome: &[f64]) -> Result<&RDDResults> {
        if running_var.len() < self.min_observations {
            return Err(RDDError::InsufficientData {
                needed: self.min_observations,
                got: running_var.len(),
            });
        }

        let mut model = RDDModel::new(self.cutoff, self.bandwidth, self.kernel);
        let results = model.fit(running_var, outcome)?;

        self.rdd_results = Some(results);
        Ok(self.rdd_results.as_ref().unwrap())
    }

    /// Fit the strategy using price data and RSI.
    pub fn fit_rsi(&mut self, prices: &[f64], rsi_period: usize, forward_periods: usize) -> Result<&RDDResults> {
        let rsi = compute_rsi(prices, rsi_period);

        // Calculate forward returns
        let mut forward_returns = vec![f64::NAN; prices.len()];
        for i in 0..(prices.len().saturating_sub(forward_periods)) {
            if prices[i] > 0.0 {
                forward_returns[i] = (prices[i + forward_periods] - prices[i]) / prices[i];
            }
        }

        // Filter valid observations
        let valid: Vec<(f64, f64)> = rsi
            .iter()
            .zip(forward_returns.iter())
            .filter(|(r, f)| !r.is_nan() && !f.is_nan())
            .map(|(&r, &f)| (r, f))
            .collect();

        let running: Vec<f64> = valid.iter().map(|(r, _)| *r).collect();
        let outcomes: Vec<f64> = valid.iter().map(|(_, f)| *f).collect();

        self.fit(&running, &outcomes)
    }

    /// Get the fitted RDD results.
    pub fn results(&self) -> Option<&RDDResults> {
        self.rdd_results.as_ref()
    }

    /// Check if the strategy has been fitted.
    pub fn is_fitted(&self) -> bool {
        self.rdd_results.is_some()
    }

    /// Generate a trading signal for the current indicator value.
    pub fn generate_signal(&self, current_value: f64) -> TradingSignal {
        let results = match &self.rdd_results {
            Some(r) => r,
            None => return TradingSignal::neutral(),
        };

        // Check if within bandwidth
        let distance = current_value - self.cutoff;
        if distance.abs() > self.bandwidth {
            return TradingSignal::neutral();
        }

        // Check statistical significance
        if results.p_value > self.max_p_value {
            return TradingSignal::neutral();
        }

        // Check effect size
        if results.tau.abs() < self.min_effect {
            return TradingSignal::neutral();
        }

        // Determine signal
        let strength = SignalStrength::from_effect(results.tau, results.se, self.min_effect);
        let is_treated = distance >= 0.0;

        // If above cutoff and positive effect, expect continuation
        // If below cutoff and positive effect, position for crossing
        let direction = if is_treated {
            results.tau
        } else {
            // Below threshold - anticipate crossing
            results.tau * (1.0 - distance.abs() / self.bandwidth)
        };

        let distance_factor = 1.0 - distance.abs() / self.bandwidth;
        let suggested_size = strength.as_f64() * distance_factor;

        TradingSignal {
            direction,
            strength,
            effect_size: results.tau,
            confidence_interval: (results.ci_lower, results.ci_upper),
            distance_to_threshold: distance,
            suggested_size,
        }
    }

    /// Generate signals for a series of indicator values.
    pub fn generate_signals(&self, values: &[f64]) -> Vec<TradingSignal> {
        values.iter().map(|&v| self.generate_signal(v)).collect()
    }

    /// Get strategy summary.
    pub fn summary(&self) -> String {
        let mut s = format!(
            "Strategy: {}\n\
             Cutoff: {:.2}\n\
             Bandwidth: {:.2}\n\
             Holding Period: {} bars\n",
            self.name, self.cutoff, self.bandwidth, self.holding_period
        );

        if let Some(results) = &self.rdd_results {
            s.push_str(&format!(
                "\nRDD Results:\n\
                 Treatment Effect: {:.4} ({:.4})\n\
                 95% CI: [{:.4}, {:.4}]\n\
                 p-value: {:.4}\n\
                 N left: {}, N right: {}",
                results.tau,
                results.se,
                results.ci_lower,
                results.ci_upper,
                results.p_value,
                results.n_left,
                results.n_right
            ));
        } else {
            s.push_str("\n(Not fitted)");
        }

        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::bybit::generate_synthetic_data;

    #[test]
    fn test_rsi_strategy() {
        // Generate synthetic data
        let data = generate_synthetic_data("TEST", 500, 100.0, 0.02, 42);
        let prices = data.close_prices();

        let mut strategy = RDDStrategy::rsi_oversold()
            .with_min_observations(50);

        // This may or may not find a significant effect on random data
        match strategy.fit_rsi(&prices, 14, 24) {
            Ok(results) => {
                println!("RSI Strategy fitted: tau={:.4}", results.tau);
                assert!(strategy.is_fitted());
            }
            Err(e) => {
                println!("Strategy fitting error (expected on random data): {}", e);
            }
        }
    }

    #[test]
    fn test_signal_generation() {
        let mut strategy = RDDStrategy::rsi_oversold();

        // Manually set results for testing
        strategy.rdd_results = Some(RDDResults {
            tau: 0.02,
            se: 0.005,
            ci_lower: 0.01,
            ci_upper: 0.03,
            t_stat: 4.0,
            p_value: 0.001,
            bandwidth: 5.0,
            n_left: 50,
            n_right: 50,
            alpha_left: 0.01,
            alpha_right: 0.03,
            beta_left: 0.0,
            beta_right: 0.0,
        });

        // Test signal at various RSI values
        let signal_at_25 = strategy.generate_signal(25.0);
        assert!(signal_at_25.is_actionable());

        let signal_at_30 = strategy.generate_signal(30.0);
        assert!(signal_at_30.is_actionable());

        let signal_at_50 = strategy.generate_signal(50.0);
        assert!(!signal_at_50.is_actionable()); // Outside bandwidth
    }
}
