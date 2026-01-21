//! Cryptocurrency compliance checking example (Bybit)
//!
//! This example demonstrates compliance checking for cryptocurrency
//! trading activities on platforms like Bybit.

use llm_compliance_check::prelude::*;
use std::collections::HashMap;

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("LLM Compliance Check - Cryptocurrency Example (Bybit)");
    println!("======================================================\n");

    // Configure compliance checker for crypto
    let mut config = ComplianceConfig::default();
    config.jurisdictions = vec![Jurisdiction::Crypto];
    config.max_leverage = 100.0;
    config.notional_limit = 100_000.0;

    // Set position limits for specific pairs
    let mut position_limits = HashMap::new();
    position_limits.insert("BTCUSDT".to_string(), 10.0); // Max 10 BTC
    position_limits.insert("ETHUSDT".to_string(), 100.0); // Max 100 ETH
    config.position_limits = position_limits;

    let pre_checker = RuleBasedPreChecker::new(config);
    let report_gen = ReportGenerator::new();

    // Create sample crypto trading activities
    let activities = vec![
        // Normal BTC order with moderate leverage
        TradingActivity::new_crypto_order("BTCUSDT", 0.5, TradeSide::Buy, Some(45000.0), 10.0)
            .with_metadata("order_type", "limit")
            .with_metadata("position_mode", "one_way"),

        // ETH order with high leverage
        TradingActivity::new_crypto_order("ETHUSDT", 5.0, TradeSide::Buy, Some(2500.0), 50.0)
            .with_metadata("margin_mode", "isolated"),

        // Large BTC position (may exceed limits)
        TradingActivity::new_crypto_order("BTCUSDT", 15.0, TradeSide::Buy, Some(45000.0), 5.0),

        // Extreme leverage (should trigger warning)
        TradingActivity::new_crypto_order("BTCUSDT", 0.1, TradeSide::Buy, Some(45000.0), 125.0),
    ];

    println!("Checking {} crypto activities...\n", activities.len());

    for (i, activity) in activities.iter().enumerate() {
        println!("--- Activity {} ---", i + 1);
        println!("Symbol: {}", activity.symbol);
        println!("Side: {:?}", activity.side);
        println!("Quantity: {}", activity.quantity);
        println!("Price: {:?}", activity.price);
        println!("Leverage: {}x", activity.leverage);

        if let Some(notional) = activity.notional_value() {
            println!("Notional Value: ${:.2}", notional);
        }

        // Run compliance check
        match pre_checker.check(activity) {
            Some(result) => {
                println!("\nResult: {:?}", result.status);
                println!("Confidence: {:.1}%", result.confidence * 100.0);

                if !result.violations.is_empty() {
                    println!("\nViolations:");
                    for v in &result.violations {
                        let severity_icon = match v.severity {
                            ViolationSeverity::Low => "🟡",
                            ViolationSeverity::Medium => "🟠",
                            ViolationSeverity::High => "🔴",
                            ViolationSeverity::Critical => "⛔",
                        };
                        println!("  {} [{}] {}", severity_icon, v.rule, v.description);
                        for evidence in &v.evidence {
                            println!("      Evidence: {}", evidence);
                        }
                    }
                }

                if !result.recommendations.is_empty() {
                    println!("\nRecommendations:");
                    for rec in &result.recommendations {
                        println!("  • {}", rec);
                    }
                }
            }
            None => {
                // No pre-check violations, create mock approved result
                let result = ComplianceResult::approved(
                    &activity.id,
                    format!(
                        "Crypto order for {} {} at {}x leverage complies with exchange rules.",
                        activity.quantity,
                        activity.symbol,
                        activity.leverage
                    ),
                )
                .with_confidence(0.92)
                .with_regulations(vec![
                    "BYBIT_LEVERAGE".to_string(),
                    "BYBIT_POSITION".to_string(),
                    "AML_KYC".to_string(),
                    "WASH_TRADING".to_string(),
                ]);

                println!("\nResult: ✅ {:?}", result.status);
                println!("Confidence: {:.1}%", result.confidence * 100.0);
                println!("Explanation: {}", result.explanation);
            }
        }

        println!("\n");
    }

    // Display regulation database for crypto
    println!("=== Applicable Crypto Regulations ===\n");

    let reg_db = RegulationDatabase::new();
    let sample_activity = TradingActivity::new_crypto_order(
        "BTCUSDT", 1.0, TradeSide::Buy, None, 10.0
    );
    let regulations = reg_db.get_applicable(&sample_activity, &Jurisdiction::Crypto);

    for reg in regulations {
        println!("📋 {} ({})", reg.name, reg.id);
        println!("   {}", reg.description);
        println!("   Checks: {}", reg.checks.join(", "));
        if !reg.thresholds.is_empty() {
            println!("   Thresholds:");
            for (key, value) in &reg.thresholds {
                println!("     - {}: {}", key, value);
            }
        }
        println!();
    }

    // Generate metrics summary
    println!("=== Compliance Metrics Summary ===\n");

    let mut metrics = ComplianceMetrics::new(
        chrono::Utc::now() - chrono::Duration::hours(1),
        chrono::Utc::now(),
    );

    // Record some sample results
    for activity in &activities {
        let result = match pre_checker.check(activity) {
            Some(r) => r,
            None => ComplianceResult::approved(&activity.id, "Approved"),
        };
        metrics.record(&result, 50.0);
    }

    println!("{}", metrics.summary());

    Ok(())
}
