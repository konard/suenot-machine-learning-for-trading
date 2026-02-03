//! Gradient-based counterfactual optimizer.

use std::fmt;

use crate::model::TradingClassifier;
use super::metrics::{count_changes, find_changed_features, ProximityMetric};

/// Result of counterfactual generation.
#[derive(Debug, Clone)]
pub struct CounterfactualResult {
    /// Original input
    pub original: Vec<f64>,
    /// Generated counterfactual
    pub counterfactual: Vec<f64>,
    /// Original predicted class
    pub original_class: usize,
    /// Target class
    pub target_class: usize,
    /// Original prediction probability
    pub original_prob: f64,
    /// Counterfactual prediction probability
    pub counterfactual_prob: f64,
    /// Whether counterfactual is valid
    pub is_valid: bool,
    /// Number of features changed
    pub num_features_changed: usize,
    /// L1 distance
    pub l1_distance: f64,
    /// L2 distance
    pub l2_distance: f64,
    /// Changed features: (index, original, counterfactual)
    pub changed_features: Vec<(usize, f64, f64)>,
}

impl fmt::Display for CounterfactualResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let class_names = ["SELL", "HOLD", "BUY"];

        writeln!(f, "Counterfactual Explanation")?;
        writeln!(f, "==========================")?;
        writeln!(
            f,
            "Original prediction: {} ({:.1}% confidence)",
            class_names[self.original_class],
            self.original_prob * 100.0
        )?;
        writeln!(f, "Target prediction: {}", class_names[self.target_class])?;
        writeln!(f, "Counterfactual valid: {}", self.is_valid)?;
        writeln!(f)?;
        writeln!(
            f,
            "To change from {} to {}:",
            class_names[self.original_class], class_names[self.target_class]
        )?;

        if self.changed_features.is_empty() {
            writeln!(f, "  No valid counterfactual found")?;
        } else {
            for (idx, orig, cf) in &self.changed_features {
                let direction = if cf > orig { "increase" } else { "decrease" };
                let diff = (cf - orig).abs();
                writeln!(
                    f,
                    "  - Feature {}: {:.3} -> {:.3} ({} by {:.3})",
                    idx, orig, cf, direction, diff
                )?;
            }
        }

        writeln!(f)?;
        writeln!(f, "Number of features changed: {}", self.num_features_changed)?;
        writeln!(f, "L1 distance: {:.4}", self.l1_distance)?;
        writeln!(f, "L2 distance: {:.4}", self.l2_distance)
    }
}

/// Counterfactual optimizer using numerical gradient descent.
pub struct CounterfactualOptimizer<'a> {
    model: &'a TradingClassifier,
    lambda_proximity: f64,
    lambda_validity: f64,
    lambda_sparsity: f64,
    actionable_mask: Option<Vec<bool>>,
}

impl<'a> CounterfactualOptimizer<'a> {
    /// Create a new optimizer.
    pub fn new(model: &'a TradingClassifier) -> Self {
        Self {
            model,
            lambda_proximity: 1.0,
            lambda_validity: 1.0,
            lambda_sparsity: 0.1,
            actionable_mask: None,
        }
    }

    /// Set proximity weight.
    pub fn with_proximity_weight(mut self, weight: f64) -> Self {
        self.lambda_proximity = weight;
        self
    }

    /// Set validity weight.
    pub fn with_validity_weight(mut self, weight: f64) -> Self {
        self.lambda_validity = weight;
        self
    }

    /// Set sparsity weight.
    pub fn with_sparsity_weight(mut self, weight: f64) -> Self {
        self.lambda_sparsity = weight;
        self
    }

    /// Set actionable mask.
    pub fn with_actionable_mask(mut self, mask: Vec<bool>) -> Self {
        self.actionable_mask = Some(mask);
        self
    }

    /// Generate counterfactual explanation.
    pub fn generate(
        &self,
        x: &[f64],
        target_class: usize,
        num_steps: usize,
        learning_rate: f64,
    ) -> CounterfactualResult {
        // Get original prediction
        let orig_probs = self.model.predict_proba(x);
        let orig_class = self.model.predict(x);
        let orig_prob = orig_probs[orig_class];

        // Initialize counterfactual
        let mut x_cf = x.to_vec();
        let mut best_cf: Option<Vec<f64>> = None;
        let mut best_loss = f64::INFINITY;

        // Optimization loop
        for _ in 0..num_steps {
            // Compute numerical gradients
            let gradients = self.compute_gradients(&x_cf, x, target_class);

            // Apply gradient descent
            for (i, grad) in gradients.iter().enumerate() {
                // Check actionability
                if let Some(ref mask) = self.actionable_mask {
                    if !mask[i] {
                        continue;
                    }
                }
                x_cf[i] -= learning_rate * grad;
            }

            // Check if valid
            let cf_pred = self.model.predict(&x_cf);
            let loss = self.compute_loss(&x_cf, x, target_class);

            if cf_pred == target_class && loss < best_loss {
                best_loss = loss;
                best_cf = Some(x_cf.clone());
            }
        }

        // Use best or final
        let final_cf = best_cf.unwrap_or(x_cf);

        // Compute metrics
        let cf_probs = self.model.predict_proba(&final_cf);
        let cf_pred = self.model.predict(&final_cf);

        let is_valid = cf_pred == target_class;
        let l1 = ProximityMetric::L1.distance(x, &final_cf);
        let l2 = ProximityMetric::L2.distance(x, &final_cf);
        let num_changed = count_changes(x, &final_cf, 0.01);
        let changed = find_changed_features(x, &final_cf, 0.01);

        CounterfactualResult {
            original: x.to_vec(),
            counterfactual: final_cf,
            original_class: orig_class,
            target_class,
            original_prob: orig_prob,
            counterfactual_prob: cf_probs[target_class],
            is_valid,
            num_features_changed: num_changed,
            l1_distance: l1,
            l2_distance: l2,
            changed_features: changed,
        }
    }

    fn compute_gradients(&self, x_cf: &[f64], x_orig: &[f64], target_class: usize) -> Vec<f64> {
        let eps = 1e-5;
        let mut gradients = vec![0.0; x_cf.len()];

        for i in 0..x_cf.len() {
            let mut x_plus = x_cf.to_vec();
            let mut x_minus = x_cf.to_vec();
            x_plus[i] += eps;
            x_minus[i] -= eps;

            let loss_plus = self.compute_loss(&x_plus, x_orig, target_class);
            let loss_minus = self.compute_loss(&x_minus, x_orig, target_class);

            gradients[i] = (loss_plus - loss_minus) / (2.0 * eps);
        }

        gradients
    }

    fn compute_loss(&self, x_cf: &[f64], x_orig: &[f64], target_class: usize) -> f64 {
        // Proximity loss (L1)
        let proximity: f64 = x_cf
            .iter()
            .zip(x_orig.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();

        // Validity loss (negative log probability)
        let probs = self.model.predict_proba(x_cf);
        let validity = -probs[target_class].ln().max(-10.0);

        // Sparsity loss (approximate L0)
        let sparsity: f64 = x_cf
            .iter()
            .zip(x_orig.iter())
            .map(|(a, b)| 1.0 - (-(a - b).abs() / 0.1).exp())
            .sum();

        self.lambda_proximity * proximity
            + self.lambda_validity * validity
            + self.lambda_sparsity * sparsity
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimizer_creation() {
        let model = TradingClassifier::new(6, 64, 3);
        let optimizer = CounterfactualOptimizer::new(&model)
            .with_proximity_weight(1.0)
            .with_validity_weight(1.0)
            .with_sparsity_weight(0.1);

        assert_eq!(optimizer.lambda_proximity, 1.0);
    }

    #[test]
    fn test_generate_counterfactual() {
        let model = TradingClassifier::new(6, 64, 3);
        let optimizer = CounterfactualOptimizer::new(&model);

        let x = vec![0.5, 0.01, -0.2, 0.02, 0.3, 0.01];
        let result = optimizer.generate(&x, 2, 50, 0.05);

        assert_eq!(result.original.len(), 6);
        assert_eq!(result.counterfactual.len(), 6);
    }
}
