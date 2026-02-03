//! Synthetic Control Method implementation.
//!
//! Provides the core SCM estimator with:
//! - Weight optimization
//! - Treatment effect estimation
//! - Placebo test inference

use anyhow::{anyhow, Result};
use std::collections::HashMap;

/// Results from Synthetic Control Method analysis.
#[derive(Debug, Clone)]
pub struct SCMResults {
    /// Optimal donor weights
    pub weights: Vec<f64>,
    /// Donor labels
    pub donor_labels: Vec<String>,
    /// Pre-treatment RMSE
    pub pre_treatment_rmse: f64,
    /// Treatment effects over time
    pub treatment_effects: Vec<f64>,
    /// Cumulative treatment effect
    pub cumulative_effect: f64,
    /// Cumulative abnormal return
    pub car: f64,
    /// Pre-treatment fitted values
    pub pre_treatment_fit: Vec<f64>,
    /// Post-treatment synthetic values
    pub post_treatment_synthetic: Vec<f64>,
}

/// Results from placebo tests.
#[derive(Debug, Clone)]
pub struct PlaceboResults {
    /// Placebo effects by donor
    pub placebo_effects: HashMap<String, Vec<f64>>,
    /// P-value from permutation test
    pub p_value: f64,
    /// Rank of actual effect among placebos
    pub rank: usize,
    /// Number of placebos run
    pub n_placebos: usize,
}

/// Synthetic Control Method model for causal inference in trading.
///
/// This model constructs a synthetic version of a treated unit using
/// a weighted combination of donor units.
///
/// # Example
///
/// ```rust,no_run
/// use synthetic_control_trading::SyntheticControlModel;
///
/// let mut model = SyntheticControlModel::new();
/// let treated_pre = vec![0.01, 0.02, -0.01, 0.015];
/// let donors_pre = vec![
///     vec![0.008, 0.018, -0.012, 0.014],
///     vec![0.012, 0.022, -0.008, 0.016],
/// ];
/// model.fit(&treated_pre, &donors_pre).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct SyntheticControlModel {
    /// Regularization penalty
    v_penalty: f64,
    /// Maximum weight for any donor
    max_weight: f64,
    /// Minimum weight for any donor
    min_weight: f64,
    /// Fitted weights
    weights: Option<Vec<f64>>,
    /// Donor labels
    donor_labels: Option<Vec<String>>,
    /// Pre-treatment RMSE
    pre_treatment_rmse: Option<f64>,
    /// Stored pre-treatment data
    treated_pre: Option<Vec<f64>>,
    donors_pre: Option<Vec<Vec<f64>>>,
}

impl Default for SyntheticControlModel {
    fn default() -> Self {
        Self::new()
    }
}

impl SyntheticControlModel {
    /// Create a new Synthetic Control Model.
    pub fn new() -> Self {
        Self {
            v_penalty: 0.0,
            max_weight: 1.0,
            min_weight: 0.0,
            weights: None,
            donor_labels: None,
            pre_treatment_rmse: None,
            treated_pre: None,
            donors_pre: None,
        }
    }

    /// Set regularization penalty.
    pub fn with_penalty(mut self, penalty: f64) -> Self {
        self.v_penalty = penalty;
        self
    }

    /// Set weight bounds.
    pub fn with_bounds(mut self, min_weight: f64, max_weight: f64) -> Self {
        self.min_weight = min_weight;
        self.max_weight = max_weight;
        self
    }

    /// Fit the synthetic control model to pre-treatment data.
    ///
    /// Finds optimal weights W* such that: Σⱼ wⱼ * donor_j ≈ treated
    ///
    /// # Arguments
    ///
    /// * `treated` - Pre-treatment outcomes for treated unit
    /// * `donors` - Pre-treatment outcomes for each donor (outer vec = donors, inner vec = time)
    ///
    /// # Returns
    ///
    /// Result indicating success or failure
    pub fn fit(&mut self, treated: &[f64], donors: &[Vec<f64>]) -> Result<()> {
        if donors.is_empty() {
            return Err(anyhow!("At least one donor is required"));
        }

        let n_donors = donors.len();
        let n_periods = treated.len();

        // Validate donor lengths
        for (i, donor) in donors.iter().enumerate() {
            if donor.len() != n_periods {
                return Err(anyhow!(
                    "Donor {} has {} periods, expected {}",
                    i,
                    donor.len(),
                    n_periods
                ));
            }
        }

        // Store data
        self.treated_pre = Some(treated.to_vec());
        self.donors_pre = Some(donors.to_vec());
        self.donor_labels = Some((0..n_donors).map(|i| format!("Donor_{}", i)).collect());

        // Optimize weights
        self.weights = Some(self.optimize_weights(treated, donors)?);

        // Calculate pre-treatment fit
        let synthetic_pre = self.compute_synthetic(treated.len(), donors)?;
        let mse: f64 = treated
            .iter()
            .zip(synthetic_pre.iter())
            .map(|(t, s)| (t - s).powi(2))
            .sum::<f64>()
            / n_periods as f64;
        self.pre_treatment_rmse = Some(mse.sqrt());

        Ok(())
    }

    /// Fit with custom donor labels.
    pub fn fit_with_labels(
        &mut self,
        treated: &[f64],
        donors: &[Vec<f64>],
        labels: Vec<String>,
    ) -> Result<()> {
        self.fit(treated, donors)?;
        self.donor_labels = Some(labels);
        Ok(())
    }

    /// Optimize donor weights using constrained least squares.
    fn optimize_weights(&self, treated: &[f64], donors: &[Vec<f64>]) -> Result<Vec<f64>> {
        let n_donors = donors.len();
        let n_periods = treated.len();

        // Simple iterative optimization (gradient descent with projection)
        let mut weights = vec![1.0 / n_donors as f64; n_donors];
        let learning_rate = 0.1;
        let max_iterations = 1000;
        let tolerance = 1e-8;

        for _ in 0..max_iterations {
            // Compute synthetic
            let synthetic: Vec<f64> = (0..n_periods)
                .map(|t| {
                    donors
                        .iter()
                        .zip(weights.iter())
                        .map(|(d, w)| d[t] * w)
                        .sum()
                })
                .collect();

            // Compute gradients
            let mut gradients = vec![0.0; n_donors];
            for (j, donor) in donors.iter().enumerate() {
                for t in 0..n_periods {
                    gradients[j] += 2.0 * (synthetic[t] - treated[t]) * donor[t];
                }
                gradients[j] /= n_periods as f64;
                gradients[j] += 2.0 * self.v_penalty * weights[j];
            }

            // Update weights
            let mut new_weights = weights.clone();
            for (j, grad) in gradients.iter().enumerate() {
                new_weights[j] -= learning_rate * grad;
            }

            // Project to simplex
            new_weights = self.project_to_simplex(&new_weights);

            // Check convergence
            let diff: f64 = new_weights
                .iter()
                .zip(weights.iter())
                .map(|(n, o)| (n - o).abs())
                .sum();

            weights = new_weights;

            if diff < tolerance {
                break;
            }
        }

        Ok(weights)
    }

    /// Project vector onto probability simplex.
    fn project_to_simplex(&self, v: &[f64]) -> Vec<f64> {
        let mut u: Vec<f64> = v.to_vec();
        u.sort_by(|a, b| b.partial_cmp(a).unwrap());

        let mut cssv = 0.0;
        let mut rho = 0;

        for (i, &ui) in u.iter().enumerate() {
            cssv += ui;
            if ui - (cssv - 1.0) / (i + 1) as f64 > 0.0 {
                rho = i + 1;
            }
        }

        let theta = (u[..rho].iter().sum::<f64>() - 1.0) / rho as f64;

        v.iter().map(|&vi| (vi - theta).max(0.0)).collect()
    }

    /// Compute synthetic control values.
    fn compute_synthetic(&self, n_periods: usize, donors: &[Vec<f64>]) -> Result<Vec<f64>> {
        let weights = self
            .weights
            .as_ref()
            .ok_or_else(|| anyhow!("Model not fitted"))?;

        let synthetic: Vec<f64> = (0..n_periods)
            .map(|t| {
                donors
                    .iter()
                    .zip(weights.iter())
                    .map(|(d, w)| {
                        if t < d.len() {
                            d[t] * w
                        } else {
                            0.0
                        }
                    })
                    .sum()
            })
            .collect();

        Ok(synthetic)
    }

    /// Estimate treatment effects using fitted weights.
    ///
    /// # Arguments
    ///
    /// * `treated_post` - Post-treatment outcomes for treated unit
    /// * `donors_post` - Post-treatment outcomes for donors
    ///
    /// # Returns
    ///
    /// Dictionary-like structure with effects, CAR, etc.
    pub fn estimate_effects(
        &self,
        treated_post: &[f64],
        donors_post: &[Vec<f64>],
    ) -> Result<SCMEffects> {
        let weights = self
            .weights
            .as_ref()
            .ok_or_else(|| anyhow!("Model not fitted"))?;

        let n_post = treated_post.len();

        // Compute synthetic post-treatment
        let synthetic_post: Vec<f64> = (0..n_post)
            .map(|t| {
                donors_post
                    .iter()
                    .zip(weights.iter())
                    .map(|(d, w)| {
                        if t < d.len() {
                            d[t] * w
                        } else {
                            0.0
                        }
                    })
                    .sum()
            })
            .collect();

        // Treatment effects (gap)
        let effects: Vec<f64> = treated_post
            .iter()
            .zip(synthetic_post.iter())
            .map(|(t, s)| t - s)
            .collect();

        let cumulative: f64 = effects.iter().sum();

        Ok(SCMEffects {
            effects,
            cumulative,
            car: cumulative,
            synthetic_post,
            treated_post: treated_post.to_vec(),
        })
    }

    /// Get the fitted weights.
    pub fn weights(&self) -> Option<&Vec<f64>> {
        self.weights.as_ref()
    }

    /// Get weights as a hashmap with labels.
    pub fn weights_map(&self) -> Option<HashMap<String, f64>> {
        match (&self.weights, &self.donor_labels) {
            (Some(w), Some(l)) => Some(l.iter().cloned().zip(w.iter().cloned()).collect()),
            _ => None,
        }
    }

    /// Get pre-treatment RMSE.
    pub fn pre_treatment_rmse(&self) -> Option<f64> {
        self.pre_treatment_rmse
    }

    /// Get comprehensive results.
    pub fn get_results(
        &self,
        treated_post: &[f64],
        donors_post: &[Vec<f64>],
    ) -> Result<SCMResults> {
        let effects = self.estimate_effects(treated_post, donors_post)?;

        let pre_fit = self.compute_synthetic(
            self.treated_pre.as_ref().map_or(0, |v| v.len()),
            self.donors_pre.as_ref().unwrap_or(&vec![]),
        )?;

        Ok(SCMResults {
            weights: self.weights.clone().unwrap_or_default(),
            donor_labels: self.donor_labels.clone().unwrap_or_default(),
            pre_treatment_rmse: self.pre_treatment_rmse.unwrap_or(0.0),
            treatment_effects: effects.effects,
            cumulative_effect: effects.cumulative,
            car: effects.car,
            pre_treatment_fit: pre_fit,
            post_treatment_synthetic: effects.synthetic_post,
        })
    }

    /// Run in-space placebo tests for inference.
    pub fn run_placebo_tests(
        &self,
        treated_post: &[f64],
        donors_post: &[Vec<f64>],
    ) -> Result<PlaceboResults> {
        let donors_pre = self
            .donors_pre
            .as_ref()
            .ok_or_else(|| anyhow!("Model not fitted"))?;

        // Get actual treatment effect
        let actual_effects = self.estimate_effects(treated_post, donors_post)?;
        let actual_car = actual_effects.car.abs();

        let n_donors = donors_pre.len();
        let mut placebo_effects: HashMap<String, Vec<f64>> = HashMap::new();
        let mut placebo_cars: Vec<f64> = Vec::new();

        for j in 0..n_donors {
            // Donor j as treated, others as donors
            let placebo_treated_pre = &donors_pre[j];
            let placebo_donors_pre: Vec<Vec<f64>> = donors_pre
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != j)
                .map(|(_, d)| d.clone())
                .collect();

            let placebo_treated_post = &donors_post[j];
            let placebo_donors_post: Vec<Vec<f64>> = donors_post
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != j)
                .map(|(_, d)| d.clone())
                .collect();

            // Fit placebo model
            let mut placebo_model = SyntheticControlModel::new()
                .with_penalty(self.v_penalty)
                .with_bounds(self.min_weight, self.max_weight);

            if placebo_model
                .fit(placebo_treated_pre, &placebo_donors_pre)
                .is_ok()
            {
                // Only include if pre-treatment fit is reasonable
                if let Some(rmse) = placebo_model.pre_treatment_rmse() {
                    if rmse < 2.0 * self.pre_treatment_rmse.unwrap_or(f64::MAX) {
                        if let Ok(effects) =
                            placebo_model.estimate_effects(placebo_treated_post, &placebo_donors_post)
                        {
                            let label = format!("Donor_{}", j);
                            placebo_effects.insert(label, effects.effects);
                            placebo_cars.push(effects.car.abs());
                        }
                    }
                }
            }
        }

        // Calculate p-value (rank-based)
        let rank = placebo_cars.iter().filter(|&&c| c >= actual_car).count() + 1;
        let p_value = rank as f64 / (placebo_cars.len() + 1) as f64;

        Ok(PlaceboResults {
            placebo_effects,
            p_value,
            rank,
            n_placebos: placebo_cars.len(),
        })
    }
}

/// Effects from SCM analysis.
#[derive(Debug, Clone)]
pub struct SCMEffects {
    /// Point-by-point treatment effects
    pub effects: Vec<f64>,
    /// Cumulative treatment effect
    pub cumulative: f64,
    /// Cumulative abnormal return
    pub car: f64,
    /// Synthetic post-treatment values
    pub synthetic_post: Vec<f64>,
    /// Actual post-treatment values
    pub treated_post: Vec<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fit_basic() {
        let mut model = SyntheticControlModel::new();

        let treated = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let donors = vec![
            vec![1.1, 2.1, 3.1, 4.1, 5.1],
            vec![0.9, 1.9, 2.9, 3.9, 4.9],
        ];

        model.fit(&treated, &donors).unwrap();

        assert!(model.weights().is_some());
        assert!(model.pre_treatment_rmse().is_some());
    }

    #[test]
    fn test_estimate_effects() {
        let mut model = SyntheticControlModel::new();

        let treated_pre = vec![1.0, 2.0, 3.0, 4.0];
        let donors_pre = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![1.0, 2.0, 3.0, 4.0],
        ];

        model.fit(&treated_pre, &donors_pre).unwrap();

        let treated_post = vec![5.5, 6.5]; // Higher than expected
        let donors_post = vec![vec![5.0, 6.0], vec![5.0, 6.0]];

        let effects = model.estimate_effects(&treated_post, &donors_post).unwrap();

        assert!(effects.car > 0.0); // Positive treatment effect
    }

    #[test]
    fn test_project_to_simplex() {
        let model = SyntheticControlModel::new();

        let v = vec![0.5, 0.3, 0.3];
        let projected = model.project_to_simplex(&v);

        let sum: f64 = projected.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(projected.iter().all(|&x| x >= 0.0));
    }
}
