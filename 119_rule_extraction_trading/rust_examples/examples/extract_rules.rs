//! Example: Extracting rules from sample data
//!
//! This example demonstrates how to use the rule extraction algorithms
//! to derive interpretable trading rules from model predictions.
//!
//! Run with: cargo run --example extract_rules

use anyhow::Result;
use rand::Rng;
use rule_extraction_trading::{
    extraction::{DecisionTree, DecisionTreeExtractor, TreeNode, TrepanExtractor},
    rules::{Condition, Operator, Rule, RuleSet},
};

fn main() -> Result<()> {
    println!("=== Rule Extraction Examples ===\n");

    // Define feature names
    let feature_names = vec![
        "RSI".to_string(),
        "MACD".to_string(),
        "SMA_Ratio".to_string(),
        "Volatility".to_string(),
    ];

    // Example 1: Extract rules from a manually constructed decision tree
    println!("=== Example 1: Rules from Decision Tree ===\n");
    extract_from_tree(&feature_names)?;

    // Example 2: Extract rules from simulated model predictions
    println!("\n=== Example 2: TREPAN-like Rule Extraction ===\n");
    extract_with_trepan(&feature_names)?;

    // Example 3: Evaluate rule quality
    println!("\n=== Example 3: Rule Quality Evaluation ===\n");
    evaluate_rules(&feature_names)?;

    Ok(())
}

fn extract_from_tree(feature_names: &[String]) -> Result<()> {
    // Manually construct a decision tree:
    //
    //                 [RSI > 50]
    //                /          \
    //        [MACD <= 0]       [SMA_Ratio > 1.01]
    //          /    \              /        \
    //       SELL   HOLD         BUY       HOLD
    //

    let nodes = vec![
        // Node 0: Root - RSI > 50?
        TreeNode {
            feature: 0, // RSI
            threshold: 50.0,
            left_child: 1,
            right_child: 2,
            value: vec![],
            n_samples: 1000,
        },
        // Node 1: RSI <= 50, check MACD
        TreeNode {
            feature: 1, // MACD
            threshold: 0.0,
            left_child: 3,
            right_child: 4,
            value: vec![],
            n_samples: 450,
        },
        // Node 2: RSI > 50, check SMA_Ratio
        TreeNode {
            feature: 2, // SMA_Ratio
            threshold: 1.01,
            left_child: 5,
            right_child: 6,
            value: vec![],
            n_samples: 550,
        },
        // Node 3: Leaf - SELL (RSI <= 50 AND MACD <= 0)
        TreeNode {
            feature: -1,
            threshold: 0.0,
            left_child: -1,
            right_child: -1,
            value: vec![180.0, 20.0], // 90% SELL
            n_samples: 200,
        },
        // Node 4: Leaf - HOLD (RSI <= 50 AND MACD > 0)
        TreeNode {
            feature: -1,
            threshold: 0.0,
            left_child: -1,
            right_child: -1,
            value: vec![125.0, 125.0], // 50/50
            n_samples: 250,
        },
        // Node 5: Leaf - HOLD (RSI > 50 AND SMA_Ratio <= 1.01)
        TreeNode {
            feature: -1,
            threshold: 0.0,
            left_child: -1,
            right_child: -1,
            value: vec![150.0, 150.0], // 50/50
            n_samples: 300,
        },
        // Node 6: Leaf - BUY (RSI > 50 AND SMA_Ratio > 1.01)
        TreeNode {
            feature: -1,
            threshold: 0.0,
            left_child: -1,
            right_child: -1,
            value: vec![25.0, 225.0], // 90% BUY
            n_samples: 250,
        },
    ];

    let tree = DecisionTree::new(nodes, feature_names.to_vec());

    let extractor = DecisionTreeExtractor::new(feature_names.to_vec())
        .with_min_samples(10)
        .with_max_depth(5);

    let rules = extractor.extract_rules(&tree);

    println!("Extracted {} rules from decision tree:\n", rules.len());

    for (i, rule) in rules.iter().enumerate() {
        println!(
            "Rule {}: {}",
            i + 1,
            rule.to_string(feature_names)
        );
    }

    // Test rule matching
    println!("\n--- Testing Rule Matching ---");

    let test_cases = vec![
        (vec![75.0, -0.5, 1.05, 0.02], "High RSI, positive SMA ratio"),
        (vec![30.0, -0.5, 0.98, 0.02], "Low RSI, negative MACD"),
        (vec![55.0, 0.5, 0.99, 0.02], "Moderate RSI, positive MACD"),
    ];

    let ruleset = RuleSet::new(rules, feature_names.to_vec());

    for (features, description) in test_cases {
        let signal = ruleset.predict(&features);
        let signal_str = match signal {
            1 => "BUY",
            -1 => "SELL",
            _ => "HOLD",
        };
        println!(
            "\n{}: {} -> Signal: {}",
            description,
            format!("[RSI={:.0}, MACD={:.1}, SMA_R={:.2}, Vol={:.2}]",
                    features[0], features[1], features[2], features[3]),
            signal_str
        );

        let explanations = ruleset.explain(&features);
        for exp in explanations {
            println!("  Matched: {}", exp);
        }
    }

    Ok(())
}

fn extract_with_trepan(feature_names: &[String]) -> Result<()> {
    // Generate synthetic data that follows certain patterns
    let mut rng = rand::thread_rng();
    let n_samples = 500;

    let mut features = Vec::new();
    let mut labels = Vec::new();

    for _ in 0..n_samples {
        // Generate random features
        let rsi: f64 = rng.gen_range(10.0..90.0);
        let macd: f64 = rng.gen_range(-2.0..2.0);
        let sma_ratio: f64 = rng.gen_range(0.95..1.05);
        let volatility: f64 = rng.gen_range(0.005..0.05);

        features.push(vec![rsi, macd, sma_ratio, volatility]);

        // Define ground truth rules (what we want to discover)
        let label = if rsi < 30.0 && macd > 0.0 {
            1 // BUY: Oversold with positive momentum
        } else if rsi > 70.0 && macd < 0.0 {
            -1 // SELL: Overbought with negative momentum
        } else if sma_ratio > 1.02 && volatility < 0.03 {
            1 // BUY: Strong uptrend, low volatility
        } else if sma_ratio < 0.98 && volatility > 0.02 {
            -1 // SELL: Downtrend, high volatility
        } else {
            // Add some randomness for ambiguous cases
            if rng.gen::<f64>() < 0.5 {
                1
            } else {
                -1
            }
        };

        labels.push(label);
    }

    println!("Generated {} samples with patterns to discover", n_samples);
    println!("Label distribution: BUY={}, SELL={}",
             labels.iter().filter(|&&l| l == 1).count(),
             labels.iter().filter(|&&l| l == -1).count());

    // Extract rules
    let extractor = TrepanExtractor::new(feature_names.to_vec())
        .with_max_depth(4)
        .with_min_samples_leaf(30);

    let ruleset = extractor.extract_from_predictions(&features, &labels);

    println!("\nExtracted {} rules:\n", ruleset.rules.len());

    for (i, rule) in ruleset.rules.iter().enumerate() {
        println!("Rule {}: {}", i + 1, rule.to_string(feature_names));
    }

    // Calculate metrics
    let fidelity = ruleset.calculate_fidelity(&features, &labels);
    let coverage = ruleset.calculate_coverage(&features);
    let avg_complexity = ruleset.average_complexity();

    println!("\n--- Extraction Metrics ---");
    println!("Fidelity: {:.1}%", fidelity * 100.0);
    println!("Coverage: {:.1}%", coverage * 100.0);
    println!("Average complexity: {:.2} conditions per rule", avg_complexity);

    Ok(())
}

fn evaluate_rules(feature_names: &[String]) -> Result<()> {
    // Create a set of manually defined rules
    let rules = vec![
        Rule::new(
            vec![
                Condition::new(0, Operator::LessOrEqual, 30.0), // RSI <= 30
                Condition::new(1, Operator::GreaterThan, 0.0),  // MACD > 0
            ],
            1, // BUY
            0.85,
        ),
        Rule::new(
            vec![
                Condition::new(0, Operator::GreaterThan, 70.0), // RSI > 70
                Condition::new(1, Operator::LessOrEqual, 0.0),  // MACD <= 0
            ],
            -1, // SELL
            0.82,
        ),
        Rule::new(
            vec![
                Condition::new(2, Operator::GreaterThan, 1.02), // SMA_Ratio > 1.02
            ],
            1, // BUY
            0.65,
        ),
        Rule::new(
            vec![
                Condition::new(2, Operator::LessOrEqual, 0.98), // SMA_Ratio <= 0.98
                Condition::new(3, Operator::GreaterThan, 0.02), // Volatility > 0.02
            ],
            -1, // SELL
            0.70,
        ),
    ];

    println!("Evaluating manually defined rules:\n");
    for (i, rule) in rules.iter().enumerate() {
        println!("Rule {}: {}", i + 1, rule.to_string(feature_names));
    }

    let ruleset = RuleSet::new(rules, feature_names.to_vec());

    // Generate test data
    let mut rng = rand::thread_rng();
    let mut test_features = Vec::new();
    let mut model_predictions = Vec::new();

    for _ in 0..200 {
        let rsi: f64 = rng.gen_range(10.0..90.0);
        let macd: f64 = rng.gen_range(-2.0..2.0);
        let sma_ratio: f64 = rng.gen_range(0.95..1.05);
        let volatility: f64 = rng.gen_range(0.005..0.05);

        test_features.push(vec![rsi, macd, sma_ratio, volatility]);

        // Simulate model prediction (with some noise)
        let pred = if rsi < 35.0 && macd > -0.5 {
            1
        } else if rsi > 65.0 && macd < 0.5 {
            -1
        } else if sma_ratio > 1.01 {
            1
        } else if sma_ratio < 0.99 {
            -1
        } else {
            if rng.gen::<bool>() { 1 } else { -1 }
        };
        model_predictions.push(pred);
    }

    // Evaluate
    let fidelity = ruleset.calculate_fidelity(&test_features, &model_predictions);
    let coverage = ruleset.calculate_coverage(&test_features);
    let avg_complexity = ruleset.average_complexity();

    println!("\n--- Evaluation Results ---");
    println!("Fidelity (match with model): {:.1}%", fidelity * 100.0);
    println!("Coverage (samples covered): {:.1}%", coverage * 100.0);
    println!("Average complexity: {:.2} conditions/rule", avg_complexity);
    println!("Total rules: {}", ruleset.rules.len());

    // Show some example predictions
    println!("\n--- Sample Predictions ---");
    println!("{:<50} {:>12} {:>12}", "Features", "Rule Pred", "Model Pred");
    println!("{:-<76}", "");

    for i in 0..5 {
        let rule_pred = ruleset.predict(&test_features[i]);
        let model_pred = model_predictions[i];

        let rule_str = match rule_pred { 1 => "BUY", -1 => "SELL", _ => "HOLD" };
        let model_str = match model_pred { 1 => "BUY", -1 => "SELL", _ => "HOLD" };

        println!(
            "[RSI={:4.1}, MACD={:5.2}, SMA_R={:.3}, Vol={:.3}] {:>12} {:>12}",
            test_features[i][0], test_features[i][1],
            test_features[i][2], test_features[i][3],
            rule_str, model_str
        );
    }

    Ok(())
}
