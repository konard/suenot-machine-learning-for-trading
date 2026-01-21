//! Regulation database and retrieval for compliance checking

use crate::data::{Jurisdiction, TradingActivity};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A regulation entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Regulation {
    /// Regulation identifier
    pub id: String,

    /// Name of the regulation
    pub name: String,

    /// Applicable jurisdiction
    pub jurisdiction: Jurisdiction,

    /// Description
    pub description: String,

    /// Compliance checks required
    pub checks: Vec<String>,

    /// Thresholds for this regulation
    pub thresholds: HashMap<String, f64>,
}

impl Regulation {
    /// Create a new regulation
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        jurisdiction: Jurisdiction,
        description: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            jurisdiction,
            description: description.into(),
            checks: Vec::new(),
            thresholds: HashMap::new(),
        }
    }

    /// Add checks to the regulation
    pub fn with_checks(mut self, checks: Vec<&str>) -> Self {
        self.checks = checks.into_iter().map(String::from).collect();
        self
    }

    /// Add a threshold
    pub fn with_threshold(mut self, key: impl Into<String>, value: f64) -> Self {
        self.thresholds.insert(key.into(), value);
        self
    }
}

/// Database of regulations for compliance checking
#[derive(Debug, Clone)]
pub struct RegulationDatabase {
    regulations: HashMap<String, Regulation>,
}

impl Default for RegulationDatabase {
    fn default() -> Self {
        Self::new()
    }
}

impl RegulationDatabase {
    /// Create a new regulation database with default regulations
    pub fn new() -> Self {
        let mut db = Self {
            regulations: HashMap::new(),
        };
        db.load_default_regulations();
        db
    }

    /// Load default US, EU, and Crypto regulations
    fn load_default_regulations(&mut self) {
        // SEC Rule 15c3-5 (Market Access Rule)
        self.add(
            Regulation::new(
                "SEC_15C3_5",
                "SEC Rule 15c3-5 (Market Access Rule)",
                Jurisdiction::UnitedStates,
                "Requires broker-dealers with market access to implement risk controls",
            )
            .with_checks(vec![
                "pre_trade_risk_controls",
                "position_limits",
                "order_size_limits",
                "credit_limits",
            ])
            .with_threshold("single_order_limit_pct", 0.05)
            .with_threshold("position_limit_pct", 0.10),
        );

        // Regulation SHO
        self.add(
            Regulation::new(
                "REG_SHO",
                "Regulation SHO",
                Jurisdiction::UnitedStates,
                "Short selling regulations",
            )
            .with_checks(vec![
                "locate_requirement",
                "close_out_requirement",
                "threshold_securities",
            ])
            .with_threshold("settlement_days", 3.0),
        );

        // FINRA Best Execution
        self.add(
            Regulation::new(
                "FINRA_BEST_EXEC",
                "FINRA Best Execution",
                Jurisdiction::UnitedStates,
                "Best execution requirements for customer orders",
            )
            .with_checks(vec![
                "price_improvement",
                "execution_quality",
                "routing_analysis",
            ]),
        );

        // MiFID II Algorithmic Trading
        self.add(
            Regulation::new(
                "MIFID_II_ALGO",
                "MiFID II Algorithmic Trading",
                Jurisdiction::EuropeanUnion,
                "Requirements for algorithmic trading systems",
            )
            .with_checks(vec![
                "kill_switch",
                "throttling",
                "system_resilience",
                "pre_trade_controls",
            ])
            .with_threshold("order_to_trade_ratio_max", 100.0),
        );

        // MAR (Market Abuse Regulation)
        self.add(
            Regulation::new(
                "MAR",
                "Market Abuse Regulation",
                Jurisdiction::EuropeanUnion,
                "Prohibits insider dealing and market manipulation",
            )
            .with_checks(vec![
                "insider_trading_check",
                "market_manipulation_check",
                "suspicious_transaction_report",
            ]),
        );

        // Bybit Leverage Limits
        self.add(
            Regulation::new(
                "BYBIT_LEVERAGE",
                "Bybit Leverage Limits",
                Jurisdiction::Crypto,
                "Bybit exchange leverage and position limits",
            )
            .with_checks(vec![
                "leverage_limit",
                "position_limit",
                "margin_requirement",
            ])
            .with_threshold("max_leverage", 100.0)
            .with_threshold("default_max_leverage", 25.0),
        );

        // Bybit Position Limits
        self.add(
            Regulation::new(
                "BYBIT_POSITION",
                "Bybit Position Limits",
                Jurisdiction::Crypto,
                "Position size limits for Bybit contracts",
            )
            .with_checks(vec!["position_size", "notional_limit", "tier_validation"]),
        );

        // AML/KYC Requirements
        self.add(
            Regulation::new(
                "AML_KYC",
                "AML/KYC Requirements",
                Jurisdiction::Global,
                "Anti-money laundering and know your customer requirements",
            )
            .with_checks(vec![
                "kyc_verification",
                "transaction_monitoring",
                "suspicious_activity",
            ])
            .with_threshold("large_transaction_usd", 10000.0),
        );

        // Wash Trading Prevention
        self.add(
            Regulation::new(
                "WASH_TRADING",
                "Wash Trading Prevention",
                Jurisdiction::Global,
                "Prohibition of wash trading and self-dealing",
            )
            .with_checks(vec![
                "self_matching_check",
                "circular_trading_check",
                "artificial_volume_check",
            ]),
        );
    }

    /// Add a regulation to the database
    pub fn add(&mut self, regulation: Regulation) {
        self.regulations.insert(regulation.id.clone(), regulation);
    }

    /// Get a regulation by ID
    pub fn get(&self, id: &str) -> Option<&Regulation> {
        self.regulations.get(id)
    }

    /// Get all regulations
    pub fn all(&self) -> impl Iterator<Item = &Regulation> {
        self.regulations.values()
    }

    /// Get regulations applicable to a trading activity
    pub fn get_applicable(
        &self,
        activity: &TradingActivity,
        jurisdiction: &Jurisdiction,
    ) -> Vec<&Regulation> {
        let is_crypto = activity.is_crypto();

        self.regulations
            .values()
            .filter(|reg| {
                // Match jurisdiction
                reg.jurisdiction == *jurisdiction
                    || reg.jurisdiction == Jurisdiction::Global
                    // Include crypto regulations for crypto activities
                    || (is_crypto && reg.jurisdiction == Jurisdiction::Crypto)
            })
            .collect()
    }

    /// Get regulation text for LLM prompt
    pub fn get_regulation_text(&self, regulations: &[&Regulation]) -> String {
        regulations
            .iter()
            .map(|reg| {
                format!(
                    "- {}: {}\n  Description: {}\n  Checks: {}",
                    reg.id,
                    reg.name,
                    reg.description,
                    reg.checks.join(", ")
                )
            })
            .collect::<Vec<_>>()
            .join("\n\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::TradeSide;

    #[test]
    fn test_regulation_database() {
        let db = RegulationDatabase::new();

        // Check that default regulations are loaded
        assert!(db.get("SEC_15C3_5").is_some());
        assert!(db.get("BYBIT_LEVERAGE").is_some());
        assert!(db.get("AML_KYC").is_some());
    }

    #[test]
    fn test_applicable_regulations() {
        let db = RegulationDatabase::new();

        // US stock order
        let stock_activity =
            TradingActivity::new_order("AAPL", 100.0, TradeSide::Buy, Some(150.0));
        let us_regs = db.get_applicable(&stock_activity, &Jurisdiction::UnitedStates);
        assert!(us_regs.iter().any(|r| r.id == "SEC_15C3_5"));
        assert!(us_regs.iter().any(|r| r.id == "AML_KYC")); // Global

        // Crypto order
        let crypto_activity =
            TradingActivity::new_crypto_order("BTCUSDT", 1.0, TradeSide::Buy, None, 10.0);
        let crypto_regs = db.get_applicable(&crypto_activity, &Jurisdiction::Crypto);
        assert!(crypto_regs.iter().any(|r| r.id == "BYBIT_LEVERAGE"));
        assert!(crypto_regs.iter().any(|r| r.id == "WASH_TRADING")); // Global
    }
}
