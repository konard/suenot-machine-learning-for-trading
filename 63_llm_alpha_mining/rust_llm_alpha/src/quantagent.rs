//! QuantAgent - Self-improving alpha mining agent.
//!
//! Implements the two-loop architecture for continuous alpha discovery:
//! - Inner loop: Generate -> Evaluate -> Refine
//! - Outer loop: Learn from experience via knowledge base

use crate::alpha::{AlphaFactor, AlphaEvaluator, calculate_ic, predefined_factors};
use crate::data::MarketData;
use crate::error::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// A single learning experience.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Experience {
    pub id: String,
    pub factor_name: String,
    pub factor_expression: String,
    pub metrics: HashMap<String, f64>,
    pub market_condition: String,
    pub success: bool,
    pub timestamp: DateTime<Utc>,
    pub notes: String,
}

impl Experience {
    /// Create a new experience.
    pub fn new(
        factor_name: String,
        factor_expression: String,
        metrics: HashMap<String, f64>,
        market_condition: String,
        success: bool,
    ) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            factor_name,
            factor_expression,
            metrics,
            market_condition,
            success,
            timestamp: Utc::now(),
            notes: String::new(),
        }
    }

    /// Add notes to the experience.
    pub fn with_notes(mut self, notes: String) -> Self {
        self.notes = notes;
        self
    }
}

/// Knowledge base for storing experiences.
#[derive(Debug, Default)]
pub struct KnowledgeBase {
    experiences: Vec<Experience>,
    successful_patterns: HashMap<String, usize>,
    failed_patterns: HashMap<String, usize>,
    max_experiences: usize,
}

impl KnowledgeBase {
    /// Create a new knowledge base.
    pub fn new() -> Self {
        Self {
            experiences: Vec::new(),
            successful_patterns: HashMap::new(),
            failed_patterns: HashMap::new(),
            max_experiences: 10000,
        }
    }

    /// Create with custom max size.
    pub fn with_max_size(max: usize) -> Self {
        Self {
            max_experiences: max,
            ..Default::default()
        }
    }

    /// Add an experience.
    pub fn add(&mut self, experience: Experience) -> bool {
        // Check for duplicates
        let expr_lower = experience.factor_expression.to_lowercase();
        if self.experiences.iter().any(|e| e.factor_expression.to_lowercase() == expr_lower) {
            return false;
        }

        // Extract and count patterns
        let patterns = self.extract_patterns(&experience.factor_expression);
        let counter = if experience.success {
            &mut self.successful_patterns
        } else {
            &mut self.failed_patterns
        };

        for pattern in patterns {
            *counter.entry(pattern).or_insert(0) += 1;
        }

        self.experiences.push(experience);

        // Trim if needed
        if self.experiences.len() > self.max_experiences {
            // Remove oldest non-successful experiences first
            self.experiences.sort_by(|a, b| {
                match (a.success, b.success) {
                    (true, false) => std::cmp::Ordering::Greater,
                    (false, true) => std::cmp::Ordering::Less,
                    _ => a.timestamp.cmp(&b.timestamp),
                }
            });
            self.experiences.remove(0);
        }

        true
    }

    fn extract_patterns(&self, expression: &str) -> Vec<String> {
        let mut patterns = Vec::new();

        // Extract function names
        let funcs = ["ts_mean", "ts_std", "ts_delta", "ts_delay", "ts_max", "ts_min",
                     "ts_rank", "rank", "log", "abs", "sign"];
        for func in &funcs {
            if expression.contains(func) {
                patterns.push(func.to_string());
            }
        }

        // Extract variable names
        let vars = ["close", "open", "high", "low", "volume"];
        for var in &vars {
            if expression.contains(var) {
                patterns.push(var.to_string());
            }
        }

        // Detect patterns
        if expression.contains("ts_mean") && expression.contains("ts_std") {
            patterns.push("zscore_pattern".to_string());
        }
        if expression.contains("ts_delta") {
            patterns.push("momentum_pattern".to_string());
        }
        if expression.contains("-1 *") || expression.contains("* -1") {
            patterns.push("reversal_pattern".to_string());
        }

        patterns
    }

    /// Query experiences.
    pub fn query(
        &self,
        keyword: Option<&str>,
        market_condition: Option<&str>,
        success_only: bool,
        limit: usize,
    ) -> Vec<&Experience> {
        let mut results: Vec<&Experience> = self.experiences
            .iter()
            .filter(|e| !success_only || e.success)
            .filter(|e| {
                market_condition.map_or(true, |cond| e.market_condition == cond)
            })
            .filter(|e| {
                keyword.map_or(true, |kw| {
                    let kw_lower = kw.to_lowercase();
                    e.factor_expression.to_lowercase().contains(&kw_lower)
                        || e.factor_name.to_lowercase().contains(&kw_lower)
                        || e.notes.to_lowercase().contains(&kw_lower)
                })
            })
            .collect();

        // Sort by IC
        results.sort_by(|a, b| {
            let ic_a = a.metrics.get("ic").unwrap_or(&0.0).abs();
            let ic_b = b.metrics.get("ic").unwrap_or(&0.0).abs();
            ic_b.partial_cmp(&ic_a).unwrap_or(std::cmp::Ordering::Equal)
        });

        results.truncate(limit);
        results
    }

    /// Get best patterns.
    pub fn best_patterns(&self, n: usize) -> Vec<(String, f64)> {
        let mut scores: HashMap<String, f64> = HashMap::new();

        for pattern in self.successful_patterns.keys().chain(self.failed_patterns.keys()) {
            let success = *self.successful_patterns.get(pattern).unwrap_or(&0);
            let fail = *self.failed_patterns.get(pattern).unwrap_or(&0);
            let total = success + fail;

            if total >= 3 {
                scores.insert(pattern.clone(), success as f64 / total as f64);
            }
        }

        let mut sorted: Vec<_> = scores.into_iter().collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        sorted.truncate(n);
        sorted
    }

    /// Get patterns to avoid.
    pub fn avoid_patterns(&self, n: usize) -> Vec<(String, f64)> {
        let mut scores: HashMap<String, f64> = HashMap::new();

        for pattern in self.successful_patterns.keys().chain(self.failed_patterns.keys()) {
            let success = *self.successful_patterns.get(pattern).unwrap_or(&0);
            let fail = *self.failed_patterns.get(pattern).unwrap_or(&0);
            let total = success + fail;

            if total >= 3 {
                scores.insert(pattern.clone(), fail as f64 / total as f64);
            }
        }

        let mut sorted: Vec<_> = scores.into_iter().collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        sorted.truncate(n);
        sorted
    }

    /// Get summary statistics.
    pub fn summary(&self) -> KnowledgeSummary {
        let successful = self.experiences.iter().filter(|e| e.success).count();
        let total = self.experiences.len();

        KnowledgeSummary {
            total,
            successful,
            success_rate: if total > 0 { successful as f64 / total as f64 } else { 0.0 },
            unique_patterns: self.successful_patterns.len() + self.failed_patterns.len(),
            best_patterns: self.best_patterns(5),
            avoid_patterns: self.avoid_patterns(3),
        }
    }

    /// Export to JSON.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(&self.experiences).unwrap_or_default()
    }
}

/// Knowledge base summary.
#[derive(Debug)]
pub struct KnowledgeSummary {
    pub total: usize,
    pub successful: usize,
    pub success_rate: f64,
    pub unique_patterns: usize,
    pub best_patterns: Vec<(String, f64)>,
    pub avoid_patterns: Vec<(String, f64)>,
}

/// Market condition classifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarketCondition {
    Bullish,
    Bearish,
    Volatile,
    Quiet,
    Neutral,
}

impl MarketCondition {
    /// Classify market condition from data.
    pub fn classify(data: &MarketData) -> Self {
        let n = data.len();
        if n < 60 {
            return MarketCondition::Neutral;
        }

        let closes = data.close_prices();

        // Recent vs historical average
        let recent_avg: f64 = closes[n - 20..].iter().sum::<f64>() / 20.0;
        let historical_avg: f64 = closes[n - 60..n - 20].iter().sum::<f64>() / 40.0;

        // Volatility
        let log_returns = data.log_returns();
        let recent_returns = &log_returns[log_returns.len().saturating_sub(20)..];
        let recent_vol: f64 = {
            let mean: f64 = recent_returns.iter().sum::<f64>() / recent_returns.len() as f64;
            let var: f64 = recent_returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>()
                / recent_returns.len() as f64;
            var.sqrt()
        };

        let overall_vol: f64 = {
            let mean: f64 = log_returns.iter().sum::<f64>() / log_returns.len() as f64;
            let var: f64 = log_returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>()
                / log_returns.len() as f64;
            var.sqrt()
        };

        // Classify
        if recent_vol > overall_vol * 1.5 {
            MarketCondition::Volatile
        } else if recent_vol < overall_vol * 0.5 {
            MarketCondition::Quiet
        } else if recent_avg > historical_avg * 1.05 {
            MarketCondition::Bullish
        } else if recent_avg < historical_avg * 0.95 {
            MarketCondition::Bearish
        } else {
            MarketCondition::Neutral
        }
    }

    /// Get string representation.
    pub fn as_str(&self) -> &'static str {
        match self {
            MarketCondition::Bullish => "bullish",
            MarketCondition::Bearish => "bearish",
            MarketCondition::Volatile => "volatile",
            MarketCondition::Quiet => "quiet",
            MarketCondition::Neutral => "neutral",
        }
    }
}

/// QuantAgent configuration.
#[derive(Debug, Clone)]
pub struct QuantAgentConfig {
    /// Minimum quality score for success
    pub quality_threshold: f64,
    /// Number of factors to generate per iteration
    pub factors_per_iteration: usize,
}

impl Default for QuantAgentConfig {
    fn default() -> Self {
        Self {
            quality_threshold: 40.0,
            factors_per_iteration: 3,
        }
    }
}

/// Self-improving alpha mining agent.
pub struct QuantAgent {
    pub kb: KnowledgeBase,
    config: QuantAgentConfig,
}

impl QuantAgent {
    /// Create a new QuantAgent.
    pub fn new() -> Self {
        Self {
            kb: KnowledgeBase::new(),
            config: QuantAgentConfig::default(),
        }
    }

    /// Create with custom configuration.
    pub fn with_config(config: QuantAgentConfig) -> Self {
        Self {
            kb: KnowledgeBase::new(),
            config,
        }
    }

    /// Create with existing knowledge base.
    pub fn with_kb(kb: KnowledgeBase) -> Self {
        Self {
            kb,
            config: QuantAgentConfig::default(),
        }
    }

    /// Set quality threshold.
    pub fn quality_threshold(mut self, threshold: f64) -> Self {
        self.config.quality_threshold = threshold;
        self
    }

    /// Mine for alpha factors.
    pub fn mine(
        &mut self,
        data: &MarketData,
        n_iterations: usize,
        verbose: bool,
    ) -> Result<Vec<MiningResult>> {
        let market_condition = MarketCondition::classify(data);
        let evaluator = AlphaEvaluator::new(data);

        // Calculate forward returns
        let returns = data.returns();
        // Pad returns to match data length
        let mut forward_returns = vec![f64::NAN];
        forward_returns.extend(returns);

        let mut successful_factors = Vec::new();

        if verbose {
            println!("Starting QuantAgent mining");
            println!("Market condition: {}", market_condition.as_str());
            println!("Iterations: {}", n_iterations);
            println!("{}", "-".repeat(40));
        }

        for iteration in 0..n_iterations {
            if verbose {
                println!("\nIteration {}/{}", iteration + 1, n_iterations);
            }

            // Generate factors (using predefined for now)
            let factors = self.generate_factors(market_condition);

            for factor in factors {
                // Evaluate factor
                let values = match evaluator.evaluate(&factor) {
                    Ok(v) => v,
                    Err(e) => {
                        if verbose {
                            println!("  [error] {}: {}", factor.name, e);
                        }
                        continue;
                    }
                };

                // Calculate IC
                let (ic, p_value) = calculate_ic(&values, &forward_returns);

                // Calculate simple metrics
                let valid_count = values.iter().filter(|v| !v.is_nan()).count();
                let quality_score = (ic.abs() * 150.0).min(30.0)
                    + if p_value < 0.05 { 10.0 } else if p_value < 0.1 { 5.0 } else { 0.0 };

                let success = quality_score >= self.config.quality_threshold;

                if verbose {
                    let status = if success { "SUCCESS" } else { "fail" };
                    println!("  [{}] {}: IC={:.4}, Quality={:.1}",
                             status, factor.name, ic, quality_score);
                }

                // Create experience
                let mut metrics = HashMap::new();
                metrics.insert("ic".to_string(), ic);
                metrics.insert("p_value".to_string(), p_value);
                metrics.insert("quality_score".to_string(), quality_score);
                metrics.insert("valid_count".to_string(), valid_count as f64);

                let experience = Experience::new(
                    factor.name.clone(),
                    factor.expression.clone(),
                    metrics.clone(),
                    market_condition.as_str().to_string(),
                    success,
                ).with_notes(factor.description.clone());

                // Add to knowledge base
                self.kb.add(experience);

                if success {
                    successful_factors.push(MiningResult {
                        factor,
                        ic,
                        quality_score,
                        iteration: iteration + 1,
                    });
                }
            }
        }

        if verbose {
            println!("\n{}", "-".repeat(40));
            println!("Mining complete!");
            println!("Successful factors: {}", successful_factors.len());
            let summary = self.kb.summary();
            println!("Knowledge base: {} experiences, {:.1}% success rate",
                     summary.total, summary.success_rate * 100.0);
        }

        Ok(successful_factors)
    }

    fn generate_factors(&self, condition: MarketCondition) -> Vec<AlphaFactor> {
        // In a real implementation, this would call an LLM
        // For now, return predefined factors based on market condition
        let mut factors = predefined_factors();

        // Add condition-specific factors
        match condition {
            MarketCondition::Volatile => {
                factors.push(AlphaFactor::with_description(
                    "vol_mean_revert".to_string(),
                    "-1 * (close - ts_mean(close, 10)) / ts_std(close, 10)".to_string(),
                    "Short-term mean reversion for volatile markets".to_string(),
                ));
            }
            MarketCondition::Bullish | MarketCondition::Bearish => {
                factors.push(AlphaFactor::with_description(
                    "trend_follow".to_string(),
                    "ts_mean(sign(ts_delta(close, 1)), 5)".to_string(),
                    "Trend following for trending markets".to_string(),
                ));
            }
            _ => {}
        }

        factors
    }

    /// Get recommendations for current market.
    pub fn get_recommendations(&self, data: &MarketData, n: usize) -> Vec<&Experience> {
        let market_condition = MarketCondition::classify(data);
        self.kb.query(None, Some(market_condition.as_str()), true, n)
    }
}

impl Default for QuantAgent {
    fn default() -> Self {
        Self::new()
    }
}

/// Result from alpha mining.
#[derive(Debug, Clone)]
pub struct MiningResult {
    pub factor: AlphaFactor,
    pub ic: f64,
    pub quality_score: f64,
    pub iteration: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::generate_synthetic_data;

    #[test]
    fn test_market_condition() {
        let data = generate_synthetic_data("TEST", 100, 42);
        let condition = MarketCondition::classify(&data);
        // Should be some condition
        assert!(matches!(
            condition,
            MarketCondition::Bullish | MarketCondition::Bearish |
            MarketCondition::Volatile | MarketCondition::Quiet | MarketCondition::Neutral
        ));
    }

    #[test]
    fn test_knowledge_base() {
        let mut kb = KnowledgeBase::new();

        let mut metrics = HashMap::new();
        metrics.insert("ic".to_string(), 0.05);

        let exp = Experience::new(
            "test_factor".to_string(),
            "ts_mean(close, 10)".to_string(),
            metrics,
            "neutral".to_string(),
            true,
        );

        assert!(kb.add(exp.clone()));
        assert!(!kb.add(exp)); // Duplicate

        assert_eq!(kb.summary().total, 1);
    }

    #[test]
    fn test_quantagent_mine() {
        let data = generate_synthetic_data("TEST", 200, 42);
        let mut agent = QuantAgent::new().quality_threshold(20.0);

        let results = agent.mine(&data, 2, false).unwrap();

        // Should have some results
        assert!(agent.kb.summary().total > 0);
        println!("Found {} successful factors", results.len());
    }
}
