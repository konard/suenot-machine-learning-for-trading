//! Rule structures and matching logic for trading decisions.
//!
//! This module provides the core data structures for representing
//! interpretable if-then rules extracted from machine learning models.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Comparison operators for rule conditions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Operator {
    /// Less than or equal (<=)
    LessOrEqual,
    /// Greater than (>)
    GreaterThan,
}

impl fmt::Display for Operator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Operator::LessOrEqual => write!(f, "<="),
            Operator::GreaterThan => write!(f, ">"),
        }
    }
}

/// A single condition in a rule.
///
/// Represents a comparison of the form: `feature[feature_idx] operator threshold`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Condition {
    /// Index of the feature to compare
    pub feature_idx: usize,
    /// Comparison operator
    pub operator: Operator,
    /// Threshold value for comparison
    pub threshold: f64,
}

impl Condition {
    /// Create a new condition.
    pub fn new(feature_idx: usize, operator: Operator, threshold: f64) -> Self {
        Self {
            feature_idx,
            operator,
            threshold,
        }
    }

    /// Check if this condition is satisfied by the given features.
    pub fn matches(&self, features: &[f64]) -> bool {
        if self.feature_idx >= features.len() {
            return false;
        }

        let value = features[self.feature_idx];

        match self.operator {
            Operator::LessOrEqual => value <= self.threshold,
            Operator::GreaterThan => value > self.threshold,
        }
    }

    /// Convert condition to string with feature name.
    pub fn to_string_with_name(&self, feature_names: &[String]) -> String {
        let feature_name = feature_names
            .get(self.feature_idx)
            .map(|s| s.as_str())
            .unwrap_or("unknown");

        format!("{} {} {:.4}", feature_name, self.operator, self.threshold)
    }
}

/// A trading rule consisting of conditions and a prediction.
///
/// Rules have the form:
/// `IF condition1 AND condition2 AND ... THEN prediction (confidence)`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Rule {
    /// Conditions that must all be satisfied
    pub conditions: Vec<Condition>,
    /// Prediction: 1 for BUY, -1 for SELL, 0 for HOLD
    pub prediction: i32,
    /// Confidence level (0.0 to 1.0)
    pub confidence: f64,
    /// Optional weight for ensemble voting
    pub weight: f64,
    /// Source tree index (for ensemble rules)
    pub source_tree: Option<usize>,
}

impl Rule {
    /// Create a new rule.
    pub fn new(conditions: Vec<Condition>, prediction: i32, confidence: f64) -> Self {
        Self {
            conditions,
            prediction,
            confidence,
            weight: 1.0,
            source_tree: None,
        }
    }

    /// Create a rule with weight.
    pub fn with_weight(mut self, weight: f64) -> Self {
        self.weight = weight;
        self
    }

    /// Set the source tree index.
    pub fn with_source_tree(mut self, tree_idx: usize) -> Self {
        self.source_tree = Some(tree_idx);
        self
    }

    /// Check if all conditions in this rule are satisfied.
    pub fn matches(&self, features: &[f64]) -> bool {
        self.conditions.iter().all(|cond| cond.matches(features))
    }

    /// Convert rule to human-readable string.
    pub fn to_string(&self, feature_names: &[String]) -> String {
        if self.conditions.is_empty() {
            return format!(
                "DEFAULT: {} (confidence: {:.2})",
                self.prediction_str(),
                self.confidence
            );
        }

        let conditions_str: Vec<String> = self
            .conditions
            .iter()
            .map(|c| c.to_string_with_name(feature_names))
            .collect();

        format!(
            "IF {} THEN {} (confidence: {:.2})",
            conditions_str.join(" AND "),
            self.prediction_str(),
            self.confidence
        )
    }

    /// Get prediction as string.
    pub fn prediction_str(&self) -> &'static str {
        match self.prediction {
            1 => "BUY",
            -1 => "SELL",
            _ => "HOLD",
        }
    }

    /// Calculate the complexity of this rule (number of conditions).
    pub fn complexity(&self) -> usize {
        self.conditions.len()
    }
}

/// A collection of rules with evaluation methods.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuleSet {
    /// The rules in this set
    pub rules: Vec<Rule>,
    /// Feature names for interpretation
    pub feature_names: Vec<String>,
}

impl RuleSet {
    /// Create a new rule set.
    pub fn new(rules: Vec<Rule>, feature_names: Vec<String>) -> Self {
        Self {
            rules,
            feature_names,
        }
    }

    /// Get all matching rules for given features.
    pub fn get_matching_rules(&self, features: &[f64]) -> Vec<&Rule> {
        self.rules.iter().filter(|r| r.matches(features)).collect()
    }

    /// Calculate fidelity against a model's predictions.
    ///
    /// Fidelity measures how well the rules match the original model's behavior.
    pub fn calculate_fidelity(&self, features: &[Vec<f64>], model_predictions: &[i32]) -> f64 {
        if features.is_empty() {
            return 0.0;
        }

        let mut matches = 0;
        for (feat, &model_pred) in features.iter().zip(model_predictions.iter()) {
            let rule_pred = self.predict(feat);
            if rule_pred == model_pred {
                matches += 1;
            }
        }

        matches as f64 / features.len() as f64
    }

    /// Calculate coverage: fraction of samples covered by at least one rule.
    pub fn calculate_coverage(&self, features: &[Vec<f64>]) -> f64 {
        if features.is_empty() {
            return 0.0;
        }

        let covered = features
            .iter()
            .filter(|feat| self.rules.iter().any(|r| r.matches(feat)))
            .count();

        covered as f64 / features.len() as f64
    }

    /// Calculate average rule complexity.
    pub fn average_complexity(&self) -> f64 {
        if self.rules.is_empty() {
            return 0.0;
        }

        let total: usize = self.rules.iter().map(|r| r.complexity()).sum();
        total as f64 / self.rules.len() as f64
    }

    /// Make a prediction using weighted voting.
    pub fn predict(&self, features: &[f64]) -> i32 {
        let mut buy_score = 0.0;
        let mut sell_score = 0.0;

        for rule in &self.rules {
            if rule.matches(features) {
                let vote = rule.weight * rule.confidence;
                match rule.prediction {
                    1 => buy_score += vote,
                    -1 => sell_score += vote,
                    _ => {}
                }
            }
        }

        if buy_score > sell_score && buy_score > 0.5 {
            1
        } else if sell_score > buy_score && sell_score > 0.5 {
            -1
        } else {
            0
        }
    }

    /// Get explanation for a prediction.
    pub fn explain(&self, features: &[f64]) -> Vec<String> {
        self.get_matching_rules(features)
            .iter()
            .map(|r| r.to_string(&self.feature_names))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_condition_matches() {
        let cond = Condition::new(0, Operator::GreaterThan, 50.0);
        assert!(cond.matches(&[60.0, 30.0]));
        assert!(!cond.matches(&[40.0, 30.0]));
    }

    #[test]
    fn test_rule_matches() {
        let rule = Rule::new(
            vec![
                Condition::new(0, Operator::GreaterThan, 70.0),
                Condition::new(1, Operator::LessOrEqual, 0.0),
            ],
            -1,
            0.75,
        );

        assert!(rule.matches(&[75.0, -0.5]));
        assert!(!rule.matches(&[60.0, -0.5])); // First condition fails
        assert!(!rule.matches(&[75.0, 0.5])); // Second condition fails
    }

    #[test]
    fn test_rule_to_string() {
        let rule = Rule::new(
            vec![Condition::new(0, Operator::GreaterThan, 70.0)],
            -1,
            0.75,
        );

        let feature_names = vec!["RSI".to_string()];
        let s = rule.to_string(&feature_names);
        assert!(s.contains("RSI > 70.0000"));
        assert!(s.contains("SELL"));
    }

    #[test]
    fn test_ruleset_predict() {
        let rules = vec![
            Rule::new(
                vec![Condition::new(0, Operator::GreaterThan, 70.0)],
                -1,
                0.8,
            ),
            Rule::new(
                vec![Condition::new(0, Operator::LessOrEqual, 30.0)],
                1,
                0.8,
            ),
        ];

        let feature_names = vec!["RSI".to_string()];
        let ruleset = RuleSet::new(rules, feature_names);

        assert_eq!(ruleset.predict(&[75.0]), -1); // SELL
        assert_eq!(ruleset.predict(&[25.0]), 1); // BUY
        assert_eq!(ruleset.predict(&[50.0]), 0); // HOLD (no rules match)
    }
}
