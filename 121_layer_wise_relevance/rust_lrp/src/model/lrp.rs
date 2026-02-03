//! LRP propagation rules

use ndarray::{Array1, Array2, Axis};

/// Available LRP rules
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LRPRule {
    /// Basic rule, no stabilization
    Zero,
    /// Stabilized with epsilon term
    Epsilon,
    /// Emphasizes positive contributions
    Gamma,
    /// Separates positive and negative contributions
    AlphaBeta,
}

/// LRP rule implementations
pub struct LRPRules;

impl LRPRules {
    /// LRP-0: Basic rule
    ///
    /// R_i = sum_j (a_i * w_ij / sum_k a_k * w_kj) * R_j
    pub fn lrp_zero(
        activations: &Array1<f64>,
        weights: &Array2<f64>,
        relevance: &Array1<f64>,
    ) -> Array1<f64> {
        let in_features = activations.len();
        let out_features = relevance.len();

        let mut r_prev = Array1::zeros(in_features);

        for j in 0..out_features {
            // Compute z = a * w for this output neuron
            let mut z_sum = 0.0;
            for i in 0..in_features {
                z_sum += activations[i] * weights[[i, j]];
            }

            // Avoid division by zero
            let z_sum = if z_sum.abs() < 1e-9 {
                1e-9 * z_sum.signum().max(1.0)
            } else {
                z_sum
            };

            // Distribute relevance
            for i in 0..in_features {
                let z_ij = activations[i] * weights[[i, j]];
                r_prev[i] += (z_ij / z_sum) * relevance[j];
            }
        }

        r_prev
    }

    /// LRP-epsilon: Stabilized rule
    ///
    /// Adds epsilon to denominator for numerical stability
    pub fn lrp_epsilon(
        activations: &Array1<f64>,
        weights: &Array2<f64>,
        relevance: &Array1<f64>,
        epsilon: f64,
    ) -> Array1<f64> {
        let in_features = activations.len();
        let out_features = relevance.len();

        let mut r_prev = Array1::zeros(in_features);

        for j in 0..out_features {
            let mut z_sum = 0.0;
            for i in 0..in_features {
                z_sum += activations[i] * weights[[i, j]];
            }

            // Add epsilon with sign preservation
            let z_sum_stabilized = z_sum + epsilon * z_sum.signum().max(1.0);

            for i in 0..in_features {
                let z_ij = activations[i] * weights[[i, j]];
                r_prev[i] += (z_ij / z_sum_stabilized) * relevance[j];
            }
        }

        r_prev
    }

    /// LRP-gamma: Emphasizes positive contributions
    ///
    /// Uses modified weights: w' = w + gamma * max(w, 0)
    pub fn lrp_gamma(
        activations: &Array1<f64>,
        weights: &Array2<f64>,
        relevance: &Array1<f64>,
        gamma: f64,
    ) -> Array1<f64> {
        let in_features = activations.len();
        let out_features = relevance.len();

        let mut r_prev = Array1::zeros(in_features);

        for j in 0..out_features {
            let mut z_sum = 0.0;
            for i in 0..in_features {
                let w_modified = weights[[i, j]] + gamma * weights[[i, j]].max(0.0);
                z_sum += activations[i] * w_modified;
            }

            let z_sum = if z_sum.abs() < 1e-9 { 1e-9 } else { z_sum };

            for i in 0..in_features {
                let w_modified = weights[[i, j]] + gamma * weights[[i, j]].max(0.0);
                let z_ij = activations[i] * w_modified;
                r_prev[i] += (z_ij / z_sum) * relevance[j];
            }
        }

        r_prev
    }

    /// LRP-alpha-beta: Separates positive and negative contributions
    ///
    /// R_i = alpha * (a_i * w_ij)+ / sum_k (a_k * w_kj)+ * R_j
    ///     - beta * (a_i * w_ij)- / sum_k (a_k * w_kj)- * R_j
    pub fn lrp_alpha_beta(
        activations: &Array1<f64>,
        weights: &Array2<f64>,
        relevance: &Array1<f64>,
        alpha: f64,
        beta: f64,
    ) -> Array1<f64> {
        let in_features = activations.len();
        let out_features = relevance.len();

        let mut r_prev = Array1::zeros(in_features);

        for j in 0..out_features {
            let mut z_pos_sum = 0.0;
            let mut z_neg_sum = 0.0;

            for i in 0..in_features {
                let z_ij = activations[i] * weights[[i, j]];
                if z_ij > 0.0 {
                    z_pos_sum += z_ij;
                } else {
                    z_neg_sum += z_ij;
                }
            }

            let z_pos_sum = if z_pos_sum.abs() < 1e-9 { 1e-9 } else { z_pos_sum };
            let z_neg_sum = if z_neg_sum.abs() < 1e-9 { -1e-9 } else { z_neg_sum };

            for i in 0..in_features {
                let z_ij = activations[i] * weights[[i, j]];
                if z_ij > 0.0 {
                    r_prev[i] += alpha * (z_ij / z_pos_sum) * relevance[j];
                } else {
                    r_prev[i] += beta * (z_ij / z_neg_sum) * relevance[j];
                }
            }
        }

        r_prev
    }

    /// Apply LRP rule based on rule type
    pub fn apply(
        rule: LRPRule,
        activations: &Array1<f64>,
        weights: &Array2<f64>,
        relevance: &Array1<f64>,
        epsilon: f64,
        gamma: f64,
        alpha: f64,
        beta: f64,
    ) -> Array1<f64> {
        match rule {
            LRPRule::Zero => Self::lrp_zero(activations, weights, relevance),
            LRPRule::Epsilon => Self::lrp_epsilon(activations, weights, relevance, epsilon),
            LRPRule::Gamma => Self::lrp_gamma(activations, weights, relevance, gamma),
            LRPRule::AlphaBeta => Self::lrp_alpha_beta(activations, weights, relevance, alpha, beta),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_lrp_epsilon_conservation() {
        let activations = array![1.0, 2.0, 3.0];
        let weights = array![[0.5, 0.3], [0.4, 0.2], [0.1, 0.5]];
        let relevance = array![1.0, 0.5];

        let r_prev = LRPRules::lrp_epsilon(&activations, &weights, &relevance, 0.01);

        // Check conservation (approximately)
        let sum: f64 = r_prev.sum();
        let expected: f64 = relevance.sum();
        assert!((sum - expected).abs() < 0.1);
    }
}
