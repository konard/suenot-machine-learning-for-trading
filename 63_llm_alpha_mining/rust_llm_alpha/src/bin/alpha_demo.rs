//! Alpha Factor Generation and Evaluation Demo
//!
//! Demonstrates how to create, evaluate, and compare alpha factors.

use llm_alpha_mining::data::generate_synthetic_data;
use llm_alpha_mining::alpha::{AlphaFactor, AlphaEvaluator, calculate_ic, predefined_factors};

fn main() -> anyhow::Result<()> {
    println!("====================================================");
    println!("LLM Alpha Mining - Alpha Factor Demo (Rust)");
    println!("====================================================");

    // 1. Load data
    println!("\n1. LOADING DATA");
    println!("{}", "-".repeat(40));

    let data = generate_synthetic_data("BTCUSDT", 200, 42);
    println!("Loaded {} records for BTCUSDT", data.len());

    // 2. Create evaluator
    println!("\n2. INITIALIZING EVALUATOR");
    println!("{}", "-".repeat(40));

    let evaluator = AlphaEvaluator::new(&data);
    println!("Evaluator ready with {} data points", data.len());

    // 3. Evaluate predefined factors
    println!("\n3. EVALUATING PREDEFINED FACTORS");
    println!("{}", "-".repeat(40));

    let factors = predefined_factors();
    let returns = data.returns();
    // Pad returns for alignment
    let mut forward_returns = vec![f64::NAN];
    forward_returns.extend(returns);

    for factor in &factors {
        print!("\n{}: ", factor.name);

        match evaluator.evaluate(factor) {
            Ok(values) => {
                let valid_count = values.iter().filter(|v| !v.is_nan()).count();
                let (ic, p_value) = calculate_ic(&values, &forward_returns);

                println!("");
                println!("  Expression: {}", factor.expression);
                println!("  Valid values: {}/{}", valid_count, values.len());
                println!("  IC: {:.4}", ic);
                println!("  P-value: {:.4}", p_value);
                println!("  Significant: {}", if p_value < 0.05 { "Yes" } else { "No" });

                // Calculate basic stats
                let valid_vals: Vec<f64> = values.iter().filter(|v| !v.is_nan()).cloned().collect();
                if !valid_vals.is_empty() {
                    let mean: f64 = valid_vals.iter().sum::<f64>() / valid_vals.len() as f64;
                    let variance: f64 = valid_vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                        / valid_vals.len() as f64;
                    let std = variance.sqrt();
                    println!("  Mean: {:.6}", mean);
                    println!("  Std: {:.6}", std);
                }
            }
            Err(e) => {
                println!("Error - {}", e);
            }
        }
    }

    // 4. Validate expressions
    println!("\n4. EXPRESSION VALIDATION");
    println!("{}", "-".repeat(40));

    let test_expressions = [
        ("ts_mean(close, 20)", true),
        ("ts_delta(close, 5) / ts_delay(close, 5)", true),
        ("rank(close) * volume", true),
        ("import os", false),
        ("eval(x)", false),
        ("unknown_func(close)", false),
    ];

    for (expr, expected) in &test_expressions {
        let is_valid = evaluator.validate(expr);
        let status = if is_valid { "VALID" } else { "INVALID" };
        let check = if is_valid == *expected { "OK" } else { "UNEXPECTED" };
        println!("  [{}] {} ({})", status, &expr[..expr.len().min(40)], check);
    }

    // 5. Create custom factor
    println!("\n5. CUSTOM FACTOR");
    println!("{}", "-".repeat(40));

    let custom = AlphaFactor::with_description(
        "custom_momentum".to_string(),
        "ts_delta(close, 10) / ts_delay(close, 10)".to_string(),
        "10-day momentum factor".to_string(),
    );

    println!("Created: {}", custom.name);
    println!("Expression: {}", custom.expression);

    match evaluator.evaluate(&custom) {
        Ok(values) => {
            let (ic, p_value) = calculate_ic(&values, &forward_returns);
            println!("IC: {:.4}, P-value: {:.4}", ic, p_value);

            // Quality score calculation
            let quality = (ic.abs() * 150.0).min(30.0)
                + if p_value < 0.05 { 10.0 } else if p_value < 0.1 { 5.0 } else { 0.0 };
            println!("Quality Score: {:.1}/100", quality);
        }
        Err(e) => {
            println!("Error: {}", e);
        }
    }

    // 6. Factor comparison
    println!("\n6. FACTOR COMPARISON");
    println!("{}", "-".repeat(40));

    println!("\n{:<25} {:>10} {:>10} {:>12}",
             "Factor", "IC", "P-value", "Significant");
    println!("{}", "-".repeat(60));

    for factor in &factors {
        if let Ok(values) = evaluator.evaluate(factor) {
            let (ic, p_value) = calculate_ic(&values, &forward_returns);
            let sig = if p_value < 0.05 { "Yes" } else { "No" };
            println!("{:<25} {:>10.4} {:>10.4} {:>12}",
                     factor.name, ic, p_value, sig);
        }
    }

    // 7. Complex expression
    println!("\n7. COMPLEX EXPRESSION");
    println!("{}", "-".repeat(40));

    let complex = AlphaFactor::with_description(
        "zscore_momentum".to_string(),
        "(ts_delta(close, 5) - ts_mean(ts_delta(close, 5), 20)) / ts_std(ts_delta(close, 5), 20)".to_string(),
        "Z-score of 5-day momentum".to_string(),
    );

    println!("Expression: {}", complex.expression);

    // Note: This complex expression may not parse with our simple parser
    match evaluator.evaluate(&complex) {
        Ok(values) => {
            let (ic, _) = calculate_ic(&values, &forward_returns);
            println!("Successfully evaluated. IC: {:.4}", ic);
        }
        Err(e) => {
            println!("Note: Complex expression parsing limited: {}", e);
            println!("(Full expression support would require a more sophisticated parser)");
        }
    }

    println!("\n====================================================");
    println!("Demo complete!");

    Ok(())
}
