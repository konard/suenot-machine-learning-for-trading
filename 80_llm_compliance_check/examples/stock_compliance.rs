//! Stock market compliance checking example
//!
//! This example demonstrates compliance checking for traditional
//! stock market trading activities under US regulations.

use llm_compliance_check::prelude::*;
use std::collections::HashMap;

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("LLM Compliance Check - Stock Market Example");
    println!("============================================\n");

    // Configure compliance checker for US stock market
    let mut config = ComplianceConfig::default();
    config.jurisdictions = vec![Jurisdiction::UnitedStates];
    config.auto_approve_threshold = 0.90;
    config.notional_limit = 500_000.0; // $500K per order

    // Set position limits for specific stocks
    let mut position_limits = HashMap::new();
    position_limits.insert("AAPL".to_string(), 10000.0); // Max 10,000 shares
    position_limits.insert("TSLA".to_string(), 5000.0);  // Max 5,000 shares
    position_limits.insert("GME".to_string(), 1000.0);   // More restricted
    config.position_limits = position_limits;

    let pre_checker = RuleBasedPreChecker::new(config);
    let report_gen = ReportGenerator::new();

    // Create sample stock trading activities
    let activities = vec![
        // Normal retail order
        TradingActivity::new_order("AAPL", 100.0, TradeSide::Buy, Some(175.0))
            .with_account(AccountInfo {
                account_id: "RETAIL_001".to_string(),
                account_type: "retail".to_string(),
                jurisdiction: Jurisdiction::UnitedStates,
                restrictions: vec![],
                kyc_level: None,
            }),

        // Institutional order
        TradingActivity::new_order("GOOGL", 500.0, TradeSide::Sell, Some(140.0))
            .with_account(AccountInfo {
                account_id: "INST_001".to_string(),
                account_type: "institutional".to_string(),
                jurisdiction: Jurisdiction::UnitedStates,
                restrictions: vec![],
                kyc_level: None,
            })
            .with_strategy("VWAP_ALGO"),

        // Large order (may need review)
        TradingActivity::new_order("TSLA", 2000.0, TradeSide::Buy, Some(250.0))
            .with_account(AccountInfo {
                account_id: "PROP_001".to_string(),
                account_type: "proprietary".to_string(),
                jurisdiction: Jurisdiction::UnitedStates,
                restrictions: vec![],
                kyc_level: None,
            }),

        // Order exceeding position limit
        TradingActivity::new_order("GME", 2000.0, TradeSide::Buy, Some(20.0)),

        // Short sale (additional checks required)
        TradingActivity::new_order("NVDA", 500.0, TradeSide::Sell, Some(450.0))
            .with_metadata("order_type", "short_sale")
            .with_metadata("locate_id", "LOC_12345"),
    ];

    println!("Checking {} stock trading activities...\n", activities.len());

    // Track metrics
    let mut metrics = ComplianceMetrics::new(
        chrono::Utc::now() - chrono::Duration::hours(1),
        chrono::Utc::now(),
    );

    for (i, activity) in activities.iter().enumerate() {
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("Activity #{}: {} {} {} shares @ ${:.2}",
            i + 1,
            match activity.side {
                TradeSide::Buy => "BUY",
                TradeSide::Sell => "SELL",
            },
            activity.symbol,
            activity.quantity,
            activity.price.unwrap_or(0.0)
        );
        println!("Account: {} ({})",
            activity.account.account_id,
            activity.account.account_type
        );

        if let Some(strategy) = &activity.strategy_id {
            println!("Strategy: {}", strategy);
        }

        if let Some(notional) = activity.notional_value() {
            println!("Notional Value: ${:.2}", notional);
        }

        let start_time = std::time::Instant::now();

        // Run pre-checks
        let result = match pre_checker.check(activity) {
            Some(r) => r,
            None => {
                // Simulate LLM check for activities that pass pre-checks
                let mut result = ComplianceResult::approved(
                    &activity.id,
                    format!(
                        "Order to {} {} shares of {} has been analyzed and complies with SEC Rule 15c3-5 and FINRA best execution requirements.",
                        match activity.side {
                            TradeSide::Buy => "buy",
                            TradeSide::Sell => "sell",
                        },
                        activity.quantity,
                        activity.symbol
                    ),
                );

                result.confidence = 0.95;
                result.regulations_checked = vec![
                    "SEC_15C3_5".to_string(),
                    "FINRA_BEST_EXEC".to_string(),
                    "REG_SHO".to_string(),
                ];

                // Add warnings for large orders
                if let Some(notional) = activity.notional_value() {
                    if notional > 100_000.0 {
                        result.recommendations.push(
                            "Consider using algorithmic execution for large orders to minimize market impact.".to_string()
                        );
                    }
                }

                // Special handling for short sales
                if activity.metadata.get("order_type") == Some(&"short_sale".to_string()) {
                    result.recommendations.push(
                        "Short sale locate requirement verified.".to_string()
                    );
                    result.regulations_checked.push("REG_SHO_LOCATE".to_string());
                }

                result
            }
        };

        let latency = start_time.elapsed().as_millis() as f64;
        metrics.record(&result, latency);

        // Display result
        let status_icon = match result.status {
            ComplianceStatus::Approved => "✅",
            ComplianceStatus::Rejected => "🚫",
            ComplianceStatus::ReviewRequired => "⚠️",
            ComplianceStatus::Pending => "⏳",
        };

        println!("\nResult: {} {:?}", status_icon, result.status);
        println!("Confidence: {:.1}%", result.confidence * 100.0);
        println!("Audit ID: {}", result.audit_id);

        if !result.violations.is_empty() {
            println!("\n⚠️  Violations:");
            for v in &result.violations {
                println!("   [{:?}] {}: {}", v.severity, v.rule, v.description);
            }
        }

        if !result.recommendations.is_empty() {
            println!("\n💡 Recommendations:");
            for rec in &result.recommendations {
                println!("   • {}", rec);
            }
        }

        println!("\nRegulations Checked: {}", result.regulations_checked.join(", "));
        println!("Check Latency: {:.2}ms", latency);

        println!();
    }

    // Display US stock market regulations
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Applicable US Stock Market Regulations");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let reg_db = RegulationDatabase::new();
    let sample_activity = TradingActivity::new_order("AAPL", 100.0, TradeSide::Buy, Some(150.0));
    let regulations = reg_db.get_applicable(&sample_activity, &Jurisdiction::UnitedStates);

    for reg in regulations {
        println!("📋 {} ({})", reg.name, reg.id);
        println!("   {}", reg.description);
        if !reg.checks.is_empty() {
            println!("   Required Checks: {}", reg.checks.join(", "));
        }
        println!();
    }

    // Display compliance metrics summary
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Compliance Metrics Summary");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("{}", metrics.summary());

    // Generate detailed report for one activity
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Sample Detailed Report");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let sample_result = ComplianceResult::approved(
        "sample_order",
        "This is a sample detailed compliance report demonstrating the report format.",
    )
    .with_confidence(0.95)
    .with_regulations(vec!["SEC_15C3_5".to_string(), "FINRA_BEST_EXEC".to_string()])
    .with_recommendations(vec![
        "Consider using algorithmic execution for large orders.".to_string(),
        "Monitor position concentration.".to_string(),
    ]);

    println!("{}", report_gen.generate_detail_report(&sample_result));

    Ok(())
}
