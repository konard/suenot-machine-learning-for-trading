//! Parser testing experiments
//!
//! Test various factor expression parsing scenarios.
//!
//! Run with: cargo run --example test_parser

use llm_factor_discovery::parser::{FactorExpr, FactorValidator, ValidatorConfig};
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

fn main() -> anyhow::Result<()> {
    // Initialize logging
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .with_target(false)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    info!("=== Factor Expression Parser Experiments ===\n");

    // Test cases: valid expressions
    let valid_expressions = vec![
        // Simple
        "close",
        "42",
        "rank(close)",

        // Nested
        "rank(ts_mean(close, 20))",
        "zscore(correlation(returns(close, 1), volume, 60))",

        // Complex
        "rank(ts_mean(returns(close, 5), 20))",
        "div(sub(high, low), close)",
        "ts_rank(ts_delta(close, 5), 20)",

        // Multi-level nesting
        "rank(zscore(ts_mean(returns(close, 1), 10)))",
        "add(rank(close), rank(volume))",

        // With negative numbers
        "sub(0, rank(close))",

        // All function types
        "abs(returns(close, 1))",
        "sign(ts_delta(close, 1))",
        "log(volume)",
        "sqrt(volatility(close, 20))",
        "power(returns(close, 1), 2)",
        "max(high, ts_max(close, 10))",
        "min(low, ts_min(close, 10))",
    ];

    info!("Testing valid expressions:\n");
    for expr_str in &valid_expressions {
        match FactorExpr::parse(expr_str) {
            Ok(expr) => {
                info!("✓ {}", expr_str);
                info!("  Parsed: {}", expr);
                info!("  Depth: {}", expr.depth());
                info!("  Variables: {:?}\n", expr.variables());
            }
            Err(e) => {
                info!("✗ {} - UNEXPECTED ERROR: {}\n", expr_str, e);
            }
        }
    }

    // Test cases: invalid expressions
    let invalid_expressions = vec![
        ("", "Empty expression"),
        ("unknown_func(close)", "Unknown function"),
        ("rank(close", "Unclosed parenthesis"),
        ("rank()", "Wrong argument count"),
        ("ts_mean(close)", "Missing window argument"),
        ("correlation(close, volume)", "Missing window for correlation"),
    ];

    info!("Testing invalid expressions:\n");
    for (expr_str, description) in &invalid_expressions {
        match FactorExpr::parse(expr_str) {
            Ok(expr) => {
                info!("✗ '{}' ({}) - UNEXPECTED SUCCESS: {}\n", expr_str, description, expr);
            }
            Err(e) => {
                info!("✓ '{}' ({})", expr_str, description);
                info!("  Error: {}\n", e);
            }
        }
    }

    // Test validator
    info!("=== Factor Validator Experiments ===\n");

    let validator = FactorValidator::new();

    let expressions_to_validate = vec![
        "rank(close)",
        "rank(unknown_variable)",
        "ts_mean(close, -5)",
        "ts_mean(close, 500)",
        "div(close, 0)",
        "correlation(returns(close, 1), volume, 60)",
    ];

    for expr_str in &expressions_to_validate {
        if let Ok(expr) = FactorExpr::parse(expr_str) {
            let result = validator.validate(&expr);

            info!("Expression: {}", expr_str);
            info!("  Valid: {}", result.is_valid);
            info!("  Complexity: {:.2}", result.complexity_score);

            if !result.errors.is_empty() {
                info!("  Errors:");
                for err in &result.errors {
                    info!("    - {}", err);
                }
            }

            if !result.warnings.is_empty() {
                info!("  Warnings:");
                for warn in &result.warnings {
                    info!("    - {}", warn);
                }
            }

            info!("");
        }
    }

    // Test custom validator config
    info!("=== Custom Validator Config ===\n");

    let custom_config = ValidatorConfig {
        known_variables: vec![
            "open".to_string(),
            "high".to_string(),
            "low".to_string(),
            "close".to_string(),
            "volume".to_string(),
            "funding_rate".to_string(), // Custom crypto variable
            "open_interest".to_string(), // Custom crypto variable
        ],
        max_depth: 5,
        max_window_size: 168, // 1 week for hourly data
        allow_unknown_variables: false,
    };

    let custom_validator = FactorValidator::with_config(custom_config);

    let crypto_expressions = vec![
        "rank(funding_rate)",
        "correlation(returns(close, 1), open_interest, 24)",
        "unknown_var",
    ];

    info!("Crypto-specific validation:\n");
    for expr_str in &crypto_expressions {
        if let Ok(expr) = FactorExpr::parse(expr_str) {
            let result = custom_validator.validate(&expr);
            info!("{}: {}", expr_str, if result.is_valid { "VALID" } else { "INVALID" });
            for err in &result.errors {
                info!("  Error: {}", err);
            }
        }
    }

    info!("\n=== Parser Experiments Complete ===");

    Ok(())
}
