//! Data structures for trading activities and compliance results

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Trading activity types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActivityType {
    /// Order submission
    OrderSubmission,
    /// Order modification
    OrderModification,
    /// Order cancellation
    OrderCancellation,
    /// Trade execution
    TradeExecution,
    /// Position change
    PositionChange,
    /// Funds transfer
    FundsTransfer,
    /// Strategy deployment
    StrategyDeployment,
}

/// Trade side (buy or sell)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TradeSide {
    Buy,
    Sell,
}

impl std::fmt::Display for TradeSide {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TradeSide::Buy => write!(f, "buy"),
            TradeSide::Sell => write!(f, "sell"),
        }
    }
}

/// Order types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OrderType {
    Market,
    Limit,
    StopLoss,
    StopLimit,
    TakeProfit,
}

/// Jurisdiction for compliance checking
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Jurisdiction {
    UnitedStates,
    EuropeanUnion,
    UnitedKingdom,
    Singapore,
    HongKong,
    Crypto,
    Global,
}

impl Default for Jurisdiction {
    fn default() -> Self {
        Self::UnitedStates
    }
}

/// Account information for compliance context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccountInfo {
    /// Account identifier
    pub account_id: String,
    /// Account type (retail, institutional, proprietary)
    pub account_type: String,
    /// Jurisdiction
    pub jurisdiction: Jurisdiction,
    /// Account restrictions
    pub restrictions: Vec<String>,
    /// KYC level (for crypto)
    pub kyc_level: Option<String>,
}

impl Default for AccountInfo {
    fn default() -> Self {
        Self {
            account_id: Uuid::new_v4().to_string(),
            account_type: "retail".to_string(),
            jurisdiction: Jurisdiction::UnitedStates,
            restrictions: Vec::new(),
            kyc_level: None,
        }
    }
}

/// Trading activity to be checked for compliance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingActivity {
    /// Unique identifier
    pub id: String,

    /// Type of activity
    pub activity_type: ActivityType,

    /// Trading symbol
    pub symbol: String,

    /// Quantity
    pub quantity: f64,

    /// Price (if applicable)
    pub price: Option<f64>,

    /// Order type
    pub order_type: Option<OrderType>,

    /// Trade side
    pub side: TradeSide,

    /// Timestamp
    pub timestamp: DateTime<Utc>,

    /// Account information
    pub account: AccountInfo,

    /// Strategy identifier
    pub strategy_id: Option<String>,

    /// Leverage (for crypto/margin)
    pub leverage: f64,

    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl TradingActivity {
    /// Create a new order submission activity
    pub fn new_order(
        symbol: impl Into<String>,
        quantity: f64,
        side: TradeSide,
        price: Option<f64>,
    ) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            activity_type: ActivityType::OrderSubmission,
            symbol: symbol.into(),
            quantity,
            price,
            order_type: if price.is_some() {
                Some(OrderType::Limit)
            } else {
                Some(OrderType::Market)
            },
            side,
            timestamp: Utc::now(),
            account: AccountInfo::default(),
            strategy_id: None,
            leverage: 1.0,
            metadata: HashMap::new(),
        }
    }

    /// Create a new crypto order with leverage
    pub fn new_crypto_order(
        symbol: impl Into<String>,
        quantity: f64,
        side: TradeSide,
        price: Option<f64>,
        leverage: f64,
    ) -> Self {
        let mut activity = Self::new_order(symbol, quantity, side, price);
        activity.leverage = leverage;
        activity.account.jurisdiction = Jurisdiction::Crypto;
        activity
    }

    /// Set the account information
    pub fn with_account(mut self, account: AccountInfo) -> Self {
        self.account = account;
        self
    }

    /// Set the strategy ID
    pub fn with_strategy(mut self, strategy_id: impl Into<String>) -> Self {
        self.strategy_id = Some(strategy_id.into());
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Calculate notional value
    pub fn notional_value(&self) -> Option<f64> {
        self.price.map(|p| self.quantity * p)
    }

    /// Check if this is a crypto symbol
    pub fn is_crypto(&self) -> bool {
        let crypto_suffixes = ["USDT", "USD", "BTC", "ETH", "USDC", "PERP"];
        crypto_suffixes
            .iter()
            .any(|s| self.symbol.to_uppercase().ends_with(s))
    }
}

/// Compliance check status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComplianceStatus {
    /// Activity approved
    Approved,
    /// Activity rejected
    Rejected,
    /// Manual review required
    ReviewRequired,
    /// Check pending
    Pending,
}

/// Violation severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ViolationSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// A compliance violation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Violation {
    /// Rule or regulation violated
    pub rule: String,

    /// Description of violation
    pub description: String,

    /// Severity level
    pub severity: ViolationSeverity,

    /// Supporting evidence
    pub evidence: Vec<String>,
}

impl Violation {
    /// Create a new violation
    pub fn new(
        rule: impl Into<String>,
        description: impl Into<String>,
        severity: ViolationSeverity,
    ) -> Self {
        Self {
            rule: rule.into(),
            description: description.into(),
            severity,
            evidence: Vec::new(),
        }
    }

    /// Add evidence to the violation
    pub fn with_evidence(mut self, evidence: impl Into<String>) -> Self {
        self.evidence.push(evidence.into());
        self
    }
}

/// Result of a compliance check
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceResult {
    /// Activity that was checked
    pub activity_id: String,

    /// Overall status
    pub status: ComplianceStatus,

    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,

    /// List of potential violations
    pub violations: Vec<Violation>,

    /// Natural language explanation
    pub explanation: String,

    /// Recommended actions
    pub recommendations: Vec<String>,

    /// Regulations checked
    pub regulations_checked: Vec<String>,

    /// Timestamp
    pub checked_at: DateTime<Utc>,

    /// Audit trail identifier
    pub audit_id: String,
}

impl ComplianceResult {
    /// Create a new approved result
    pub fn approved(activity_id: impl Into<String>, explanation: impl Into<String>) -> Self {
        Self {
            activity_id: activity_id.into(),
            status: ComplianceStatus::Approved,
            confidence: 1.0,
            violations: Vec::new(),
            explanation: explanation.into(),
            recommendations: Vec::new(),
            regulations_checked: Vec::new(),
            checked_at: Utc::now(),
            audit_id: generate_audit_id(),
        }
    }

    /// Create a new rejected result
    pub fn rejected(
        activity_id: impl Into<String>,
        explanation: impl Into<String>,
        violations: Vec<Violation>,
    ) -> Self {
        Self {
            activity_id: activity_id.into(),
            status: ComplianceStatus::Rejected,
            confidence: 1.0,
            violations,
            explanation: explanation.into(),
            recommendations: Vec::new(),
            regulations_checked: Vec::new(),
            checked_at: Utc::now(),
            audit_id: generate_audit_id(),
        }
    }

    /// Create a result requiring review
    pub fn review_required(
        activity_id: impl Into<String>,
        explanation: impl Into<String>,
        violations: Vec<Violation>,
    ) -> Self {
        Self {
            activity_id: activity_id.into(),
            status: ComplianceStatus::ReviewRequired,
            confidence: 0.5,
            violations,
            explanation: explanation.into(),
            recommendations: Vec::new(),
            regulations_checked: Vec::new(),
            checked_at: Utc::now(),
            audit_id: generate_audit_id(),
        }
    }

    /// Set confidence score
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Add recommendations
    pub fn with_recommendations(mut self, recommendations: Vec<String>) -> Self {
        self.recommendations = recommendations;
        self
    }

    /// Add regulations checked
    pub fn with_regulations(mut self, regulations: Vec<String>) -> Self {
        self.regulations_checked = regulations;
        self
    }

    /// Check if result has critical violations
    pub fn has_critical_violations(&self) -> bool {
        self.violations
            .iter()
            .any(|v| v.severity == ViolationSeverity::Critical)
    }
}

/// Generate a unique audit ID
fn generate_audit_id() -> String {
    use sha2::{Digest, Sha256};

    let data = format!("{}{}", Uuid::new_v4(), Utc::now().timestamp_nanos_opt().unwrap_or(0));
    let hash = Sha256::digest(data.as_bytes());
    hex::encode(&hash[..8])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trading_activity_creation() {
        let activity = TradingActivity::new_order("AAPL", 100.0, TradeSide::Buy, Some(150.0));

        assert_eq!(activity.symbol, "AAPL");
        assert_eq!(activity.quantity, 100.0);
        assert_eq!(activity.side, TradeSide::Buy);
        assert_eq!(activity.price, Some(150.0));
        assert_eq!(activity.notional_value(), Some(15000.0));
    }

    #[test]
    fn test_crypto_detection() {
        let btc = TradingActivity::new_crypto_order("BTCUSDT", 1.0, TradeSide::Buy, None, 10.0);
        let aapl = TradingActivity::new_order("AAPL", 100.0, TradeSide::Buy, None);

        assert!(btc.is_crypto());
        assert!(!aapl.is_crypto());
    }

    #[test]
    fn test_compliance_result() {
        let result = ComplianceResult::approved("order_001", "Trade approved");

        assert_eq!(result.status, ComplianceStatus::Approved);
        assert_eq!(result.confidence, 1.0);
        assert!(result.violations.is_empty());
    }

    #[test]
    fn test_violation() {
        let violation = Violation::new(
            "SEC_15C3_5",
            "Order exceeds position limit",
            ViolationSeverity::High,
        )
        .with_evidence("Position: 150,000, Limit: 100,000");

        assert_eq!(violation.severity, ViolationSeverity::High);
        assert_eq!(violation.evidence.len(), 1);
    }
}
