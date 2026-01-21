//! Compliance checking logic using LLM and rule-based systems

use crate::data::{
    ComplianceResult, ComplianceStatus, Jurisdiction, TradingActivity, Violation,
    ViolationSeverity,
};
use crate::error::{Error, Result};
use crate::regulations::RegulationDatabase;
use async_trait::async_trait;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;

/// Configuration for compliance checking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceConfig {
    /// Jurisdictions to check against
    pub jurisdictions: Vec<Jurisdiction>,

    /// Confidence threshold for auto-approval
    pub auto_approve_threshold: f64,

    /// Enable real-time checking
    pub real_time_enabled: bool,

    /// Position limits by symbol
    pub position_limits: HashMap<String, f64>,

    /// Leverage limits
    pub max_leverage: f64,

    /// Notional value limits
    pub notional_limit: f64,
}

impl Default for ComplianceConfig {
    fn default() -> Self {
        Self {
            jurisdictions: vec![Jurisdiction::UnitedStates],
            auto_approve_threshold: 0.95,
            real_time_enabled: true,
            position_limits: HashMap::new(),
            max_leverage: 100.0,
            notional_limit: 1_000_000.0,
        }
    }
}

/// Trait for compliance checkers
#[async_trait]
pub trait ComplianceCheck {
    /// Check a trading activity for compliance
    async fn check(&self, activity: &TradingActivity) -> Result<ComplianceResult>;
}

/// Rule-based pre-checker for fast compliance checks
#[derive(Debug, Clone)]
pub struct RuleBasedPreChecker {
    config: ComplianceConfig,
}

impl RuleBasedPreChecker {
    /// Create a new rule-based pre-checker
    pub fn new(config: ComplianceConfig) -> Self {
        Self { config }
    }

    /// Perform fast rule-based pre-checks
    pub fn check(&self, activity: &TradingActivity) -> Option<ComplianceResult> {
        let mut violations = Vec::new();

        // Check leverage limits (for crypto)
        if activity.is_crypto() && activity.leverage > self.config.max_leverage {
            violations.push(
                Violation::new(
                    "LEVERAGE_LIMIT",
                    format!(
                        "Leverage {}x exceeds maximum allowed {}x",
                        activity.leverage, self.config.max_leverage
                    ),
                    ViolationSeverity::High,
                )
                .with_evidence(format!("Requested leverage: {}x", activity.leverage)),
            );
        }

        // Check for negative quantities
        if activity.quantity <= 0.0 {
            violations.push(
                Violation::new(
                    "INVALID_QUANTITY",
                    "Quantity must be positive",
                    ViolationSeverity::Critical,
                )
                .with_evidence(format!("Quantity: {}", activity.quantity)),
            );
        }

        // Check notional value limits
        if let Some(notional) = activity.notional_value() {
            if notional > self.config.notional_limit {
                violations.push(
                    Violation::new(
                        "NOTIONAL_LIMIT",
                        format!(
                            "Notional value ${:.2} exceeds limit ${:.2}",
                            notional, self.config.notional_limit
                        ),
                        ViolationSeverity::Medium,
                    )
                    .with_evidence(format!(
                        "Notional: ${:.2}, Limit: ${:.2}",
                        notional, self.config.notional_limit
                    )),
                );
            }
        }

        // Check position limits
        if let Some(limit) = self.config.position_limits.get(&activity.symbol) {
            if activity.quantity > *limit {
                violations.push(
                    Violation::new(
                        "POSITION_LIMIT",
                        format!(
                            "Position {} exceeds limit {} for {}",
                            activity.quantity, limit, activity.symbol
                        ),
                        ViolationSeverity::High,
                    )
                    .with_evidence(format!(
                        "Requested: {}, Limit: {}",
                        activity.quantity, limit
                    )),
                );
            }
        }

        // If violations found, return early result
        if !violations.is_empty() {
            let has_critical = violations
                .iter()
                .any(|v| v.severity == ViolationSeverity::Critical);

            let status = if has_critical {
                ComplianceStatus::Rejected
            } else {
                ComplianceStatus::ReviewRequired
            };

            return Some(ComplianceResult {
                activity_id: activity.id.clone(),
                status,
                confidence: 1.0,
                violations,
                explanation: "Pre-check violations detected".to_string(),
                recommendations: vec!["Review and correct the flagged issues".to_string()],
                regulations_checked: vec!["INTERNAL_RULES".to_string()],
                checked_at: Utc::now(),
                audit_id: generate_audit_id(&activity.id),
            });
        }

        None
    }
}

/// LLM-based compliance checker
#[derive(Debug, Clone)]
pub struct LlmComplianceChecker {
    api_key: String,
    model: String,
    regulation_db: RegulationDatabase,
    config: ComplianceConfig,
}

impl LlmComplianceChecker {
    /// Create a new LLM compliance checker
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            model: "gpt-4".to_string(),
            regulation_db: RegulationDatabase::new(),
            config: ComplianceConfig::default(),
        }
    }

    /// Set the model to use
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Set the configuration
    pub fn with_config(mut self, config: ComplianceConfig) -> Self {
        self.config = config;
        self
    }

    /// Build the compliance prompt for the LLM
    fn build_prompt(&self, activity: &TradingActivity, regulations: &str) -> String {
        format!(
            r#"You are a regulatory compliance expert. Analyze the following trading activity for compliance.

TRADING ACTIVITY:
- ID: {}
- Type: {:?}
- Symbol: {}
- Side: {:?}
- Quantity: {}
- Price: {:?}
- Order Type: {:?}
- Timestamp: {}
- Account Type: {}
- Jurisdiction: {:?}
- Leverage: {}x
- Strategy: {}

APPLICABLE REGULATIONS:
{}

Analyze this activity and provide your assessment in the following JSON format:
{{
    "status": "APPROVED" | "REJECTED" | "REVIEW_REQUIRED",
    "confidence": 0.0 to 1.0,
    "violations": [
        {{
            "rule": "regulation identifier",
            "description": "what was violated",
            "severity": "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
        }}
    ],
    "explanation": "detailed explanation of your assessment",
    "recommendations": ["list of recommended actions"]
}}

Consider:
1. Pre-trade risk controls and position limits
2. Order size relative to typical market volume
3. Leverage and margin requirements (for crypto)
4. Potential market manipulation indicators
5. KYC/AML concerns for large transactions

Respond ONLY with the JSON object."#,
            activity.id,
            activity.activity_type,
            activity.symbol,
            activity.side,
            activity.quantity,
            activity.price,
            activity.order_type,
            activity.timestamp,
            activity.account.account_type,
            activity.account.jurisdiction,
            activity.leverage,
            activity.strategy_id.as_deref().unwrap_or("N/A"),
            regulations
        )
    }

    /// Parse LLM response into ComplianceResult
    fn parse_response(
        &self,
        response: &str,
        activity: &TradingActivity,
        regulations_checked: Vec<String>,
    ) -> Result<ComplianceResult> {
        // Try to parse JSON from response
        let data: serde_json::Value =
            serde_json::from_str(response).map_err(|e| Error::Json(e))?;

        // Parse status
        let status = match data["status"].as_str().unwrap_or("REVIEW_REQUIRED") {
            "APPROVED" => ComplianceStatus::Approved,
            "REJECTED" => ComplianceStatus::Rejected,
            _ => ComplianceStatus::ReviewRequired,
        };

        // Parse confidence
        let confidence = data["confidence"].as_f64().unwrap_or(0.5);

        // Parse violations
        let violations: Vec<Violation> = data["violations"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| {
                        let severity = match v["severity"].as_str().unwrap_or("MEDIUM") {
                            "LOW" => ViolationSeverity::Low,
                            "HIGH" => ViolationSeverity::High,
                            "CRITICAL" => ViolationSeverity::Critical,
                            _ => ViolationSeverity::Medium,
                        };

                        Some(Violation::new(
                            v["rule"].as_str().unwrap_or("UNKNOWN"),
                            v["description"].as_str().unwrap_or(""),
                            severity,
                        ))
                    })
                    .collect()
            })
            .unwrap_or_default();

        // Parse explanation
        let explanation = data["explanation"]
            .as_str()
            .unwrap_or("No explanation provided")
            .to_string();

        // Parse recommendations
        let recommendations: Vec<String> = data["recommendations"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|r| r.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        // Apply auto-approve threshold
        let final_status = if status == ComplianceStatus::Approved
            && confidence < self.config.auto_approve_threshold
        {
            ComplianceStatus::ReviewRequired
        } else {
            status
        };

        Ok(ComplianceResult {
            activity_id: activity.id.clone(),
            status: final_status,
            confidence,
            violations,
            explanation,
            recommendations,
            regulations_checked,
            checked_at: Utc::now(),
            audit_id: generate_audit_id(&activity.id),
        })
    }
}

#[async_trait]
impl ComplianceCheck for LlmComplianceChecker {
    async fn check(&self, activity: &TradingActivity) -> Result<ComplianceResult> {
        // Get applicable regulations
        let jurisdiction = &self.config.jurisdictions.first().unwrap_or(&Jurisdiction::Global);
        let regulations = self.regulation_db.get_applicable(activity, jurisdiction);
        let regulations_text = self.regulation_db.get_regulation_text(&regulations);
        let regulations_checked: Vec<String> = regulations.iter().map(|r| r.id.clone()).collect();

        // Build prompt
        let _prompt = self.build_prompt(activity, &regulations_text);

        // In a real implementation, this would call the LLM API
        // For now, return a mock result based on simple heuristics

        // Simulate LLM analysis with basic heuristics
        let mut violations = Vec::new();
        let mut confidence = 0.95;

        // Check for large orders
        if let Some(notional) = activity.notional_value() {
            if notional > 100_000.0 {
                confidence = 0.85;
                if notional > 1_000_000.0 {
                    violations.push(Violation::new(
                        "SEC_15C3_5",
                        "Large order may require additional pre-trade risk checks",
                        ViolationSeverity::Medium,
                    ));
                }
            }
        }

        // Check leverage for crypto
        if activity.is_crypto() && activity.leverage > 25.0 {
            confidence = 0.80;
            violations.push(Violation::new(
                "BYBIT_LEVERAGE",
                format!("High leverage ({}x) increases liquidation risk", activity.leverage),
                ViolationSeverity::Low,
            ));
        }

        let status = if violations.is_empty() {
            ComplianceStatus::Approved
        } else if violations.iter().any(|v| v.severity == ViolationSeverity::High) {
            ComplianceStatus::ReviewRequired
        } else {
            ComplianceStatus::Approved
        };

        let explanation = if violations.is_empty() {
            format!(
                "Trading activity for {} {} shares of {} complies with applicable regulations.",
                activity.side, activity.quantity, activity.symbol
            )
        } else {
            format!(
                "Trading activity for {} has potential compliance concerns that should be reviewed.",
                activity.symbol
            )
        };

        Ok(ComplianceResult {
            activity_id: activity.id.clone(),
            status,
            confidence,
            violations,
            explanation,
            recommendations: Vec::new(),
            regulations_checked,
            checked_at: Utc::now(),
            audit_id: generate_audit_id(&activity.id),
        })
    }
}

/// Complete compliance checking pipeline
pub struct ComplianceChecker {
    pre_checker: RuleBasedPreChecker,
    llm_checker: LlmComplianceChecker,
}

impl ComplianceChecker {
    /// Create a new compliance checker
    pub fn new(api_key: String) -> Self {
        Self {
            pre_checker: RuleBasedPreChecker::new(ComplianceConfig::default()),
            llm_checker: LlmComplianceChecker::new(api_key),
        }
    }

    /// Set configuration
    pub fn with_config(mut self, config: ComplianceConfig) -> Self {
        self.pre_checker = RuleBasedPreChecker::new(config.clone());
        self.llm_checker = self.llm_checker.with_config(config);
        self
    }

    /// Check a trading activity through the full pipeline
    pub async fn check(&self, activity: &TradingActivity) -> Result<ComplianceResult> {
        // Step 1: Fast pre-checks
        if let Some(mut pre_result) = self.pre_checker.check(activity) {
            // If critical violations, return early
            if pre_result.status == ComplianceStatus::Rejected {
                return Ok(pre_result);
            }

            // Otherwise, continue to LLM check and merge results
            let llm_result = self.llm_checker.check(activity).await?;

            // Merge violations
            pre_result.violations.extend(llm_result.violations);
            pre_result.explanation = format!(
                "{}. LLM Analysis: {}",
                pre_result.explanation, llm_result.explanation
            );

            return Ok(pre_result);
        }

        // Step 2: LLM-based analysis
        self.llm_checker.check(activity).await
    }
}

/// Generate a unique audit ID
fn generate_audit_id(activity_id: &str) -> String {
    let data = format!("{}{}", activity_id, Utc::now().timestamp_nanos_opt().unwrap_or(0));
    let hash = Sha256::digest(data.as_bytes());
    hex::encode(&hash[..8])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::TradeSide;

    #[test]
    fn test_pre_checker_negative_quantity() {
        let config = ComplianceConfig::default();
        let checker = RuleBasedPreChecker::new(config);

        let activity = TradingActivity::new_order("AAPL", -100.0, TradeSide::Buy, Some(150.0));
        let result = checker.check(&activity);

        assert!(result.is_some());
        let result = result.unwrap();
        assert_eq!(result.status, ComplianceStatus::Rejected);
    }

    #[test]
    fn test_pre_checker_high_leverage() {
        let mut config = ComplianceConfig::default();
        config.max_leverage = 50.0;
        let checker = RuleBasedPreChecker::new(config);

        let activity =
            TradingActivity::new_crypto_order("BTCUSDT", 1.0, TradeSide::Buy, None, 100.0);
        let result = checker.check(&activity);

        assert!(result.is_some());
        let result = result.unwrap();
        assert!(result
            .violations
            .iter()
            .any(|v| v.rule == "LEVERAGE_LIMIT"));
    }

    #[test]
    fn test_pre_checker_normal_activity() {
        let config = ComplianceConfig::default();
        let checker = RuleBasedPreChecker::new(config);

        let activity = TradingActivity::new_order("AAPL", 100.0, TradeSide::Buy, Some(150.0));
        let result = checker.check(&activity);

        // Should pass pre-checks
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_llm_checker() {
        let checker = LlmComplianceChecker::new("test-key".to_string());

        let activity = TradingActivity::new_order("AAPL", 100.0, TradeSide::Buy, Some(150.0));
        let result = checker.check(&activity).await.unwrap();

        // Should be approved for normal activity
        assert_eq!(result.status, ComplianceStatus::Approved);
        assert!(result.confidence > 0.0);
    }
}
