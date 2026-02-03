//! LRP configuration

use super::lrp::LRPRule;

/// Configuration for LRP analysis
#[derive(Debug, Clone)]
pub struct LRPConfig {
    /// Epsilon for LRP-epsilon rule
    pub epsilon: f64,
    /// Gamma for LRP-gamma rule
    pub gamma: f64,
    /// Alpha for LRP-alpha-beta rule
    pub alpha: f64,
    /// Beta for LRP-alpha-beta rule
    pub beta: f64,
    /// Default rule to use
    pub default_rule: LRPRule,
}

impl Default for LRPConfig {
    fn default() -> Self {
        Self {
            epsilon: 0.01,
            gamma: 0.25,
            alpha: 2.0,
            beta: 1.0,
            default_rule: LRPRule::Epsilon,
        }
    }
}

impl LRPConfig {
    /// Create new config
    pub fn new() -> Self {
        Self::default()
    }

    /// Set epsilon
    pub fn with_epsilon(mut self, epsilon: f64) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Set gamma
    pub fn with_gamma(mut self, gamma: f64) -> Self {
        self.gamma = gamma;
        self
    }

    /// Set alpha
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set beta
    pub fn with_beta(mut self, beta: f64) -> Self {
        self.beta = beta;
        self
    }

    /// Set default rule
    pub fn with_default_rule(mut self, rule: LRPRule) -> Self {
        self.default_rule = rule;
        self
    }
}
