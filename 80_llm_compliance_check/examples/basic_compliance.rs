//! Basic compliance checking example
//!
//! This example demonstrates how to check trading activities for compliance
//! using the LLM compliance checker.

use llm_compliance_check::prelude::*;

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing for logging
    tracing_subscriber::fmt::init();

    println!("LLM Compliance Check - Basic Example");
    println!("=====================================\n");

    // Create sample trading activities
    let activities = vec![
        // Normal stock order
        TradingActivity::new_order("AAPL", 100.0, TradeSide::Buy, Some(150.0)),
        // Larger order
        TradingActivity::new_order("GOOGL", 5000.0, TradeSide::Sell, Some(140.0)),
        // Order without price (market order)
        TradingActivity::new_order("MSFT", 200.0, TradeSide::Buy, None),
    ];

    // Create pre-checker for fast rule-based checks
    let config = ComplianceConfig::default();
    let pre_checker = RuleBasedPreChecker::new(config);

    // Create report generator
    let report_gen = ReportGenerator::new();

    println!("Checking {} activities...\n", activities.len());

    for activity in &activities {
        println!("Checking: {} {} {} shares",
            match activity.side {
                TradeSide::Buy => "BUY",
                TradeSide::Sell => "SELL",
            },
            activity.symbol,
            activity.quantity
        );

        // First, run pre-checks
        match pre_checker.check(activity) {
            Some(result) => {
                println!("Pre-check Result: {:?}", result.status);
                if !result.violations.is_empty() {
                    println!("  Violations found:");
                    for v in &result.violations {
                        println!("    - {}: {}", v.rule, v.description);
                    }
                }
            }
            None => {
                println!("Pre-check: PASSED");

                // In a real scenario, would proceed to LLM analysis
                // For this demo, create a mock approved result
                let result = ComplianceResult::approved(
                    &activity.id,
                    format!(
                        "Order to {} {} shares of {} complies with all applicable regulations.",
                        match activity.side {
                            TradeSide::Buy => "buy",
                            TradeSide::Sell => "sell",
                        },
                        activity.quantity,
                        activity.symbol
                    ),
                )
                .with_confidence(0.95)
                .with_regulations(vec![
                    "SEC_15C3_5".to_string(),
                    "FINRA_BEST_EXEC".to_string(),
                ]);

                // Generate report
                let report = report_gen.generate_detail_report(&result);
                println!("{}", report);
            }
        }
        println!();
    }

    // Test with an invalid activity (negative quantity)
    println!("\nTesting with invalid activity (negative quantity):");
    let invalid_activity = TradingActivity::new_order("AAPL", -100.0, TradeSide::Buy, Some(150.0));

    if let Some(result) = pre_checker.check(&invalid_activity) {
        println!("Status: {:?}", result.status);
        println!("Explanation: {}", result.explanation);
        for v in &result.violations {
            println!("  - {}: {} (Severity: {:?})", v.rule, v.description, v.severity);
        }
    }

    println!("\nDone!");

    Ok(())
}
