//! QuantAgent Demo
//!
//! Demonstrates the self-improving alpha mining agent.

use llm_alpha_mining::data::generate_synthetic_data;
use llm_alpha_mining::quantagent::{QuantAgent, KnowledgeBase, MarketCondition, Experience};
use std::collections::HashMap;

fn main() -> anyhow::Result<()> {
    println!("====================================================");
    println!("LLM Alpha Mining - QuantAgent Demo (Rust)");
    println!("====================================================");

    // 1. Load data
    println!("\n1. LOADING DATA");
    println!("{}", "-".repeat(40));

    let data = generate_synthetic_data("BTCUSDT", 300, 42);
    println!("Loaded {} records for BTCUSDT", data.len());

    // 2. Initialize QuantAgent
    println!("\n2. INITIALIZING QUANTAGENT");
    println!("{}", "-".repeat(40));

    let mut agent = QuantAgent::new().quality_threshold(30.0);

    // Classify market
    let market_condition = MarketCondition::classify(&data);
    println!("Current market condition: {}", market_condition.as_str());

    // 3. Run mining
    println!("\n3. RUNNING ALPHA MINING");
    println!("{}", "-".repeat(40));

    let results = agent.mine(&data, 5, true)?;

    // 4. Display results
    println!("\n4. MINING RESULTS");
    println!("{}", "-".repeat(40));

    println!("\nTotal successful factors: {}", results.len());

    if !results.is_empty() {
        println!("\nTop factors found:");

        // Sort by quality score
        let mut sorted_results = results.clone();
        sorted_results.sort_by(|a, b| b.quality_score.partial_cmp(&a.quality_score).unwrap());

        for (i, result) in sorted_results.iter().take(5).enumerate() {
            println!("\n  {}. {}", i + 1, result.factor.name);
            println!("     Expression: {}", result.factor.expression);
            println!("     IC: {:.4}", result.ic);
            println!("     Quality Score: {:.1}/100", result.quality_score);
            println!("     Iteration found: {}", result.iteration);
        }
    }

    // 5. Knowledge base analysis
    println!("\n5. KNOWLEDGE BASE ANALYSIS");
    println!("{}", "-".repeat(40));

    let summary = agent.kb.summary();
    println!("\nTotal experiences: {}", summary.total);
    println!("Successful: {}", summary.successful);
    println!("Success rate: {:.1}%", summary.success_rate * 100.0);
    println!("Unique patterns: {}", summary.unique_patterns);

    if !summary.best_patterns.is_empty() {
        println!("\nBest performing patterns:");
        for (pattern, score) in &summary.best_patterns {
            println!("  - {}: {:.1}% success rate", pattern, score * 100.0);
        }
    }

    if !summary.avoid_patterns.is_empty() {
        println!("\nPatterns to avoid:");
        for (pattern, score) in &summary.avoid_patterns {
            println!("  - {}: {:.1}% failure rate", pattern, score * 100.0);
        }
    }

    // 6. Recommendations
    println!("\n6. FACTOR RECOMMENDATIONS");
    println!("{}", "-".repeat(40));

    let recommendations = agent.get_recommendations(&data, 5);

    if !recommendations.is_empty() {
        println!("\nRecommended factors for current market:");
        for (i, rec) in recommendations.iter().enumerate() {
            println!("\n  {}. {}", i + 1, rec.factor_name);
            println!("     Expression: {}", rec.factor_expression);
            println!("     Historical IC: {:.4}", rec.metrics.get("ic").unwrap_or(&0.0));
            println!("     Market: {}", rec.market_condition);
        }
    } else {
        println!("\n  No recommendations available (need more experience)");
    }

    // 7. Manual knowledge base usage
    println!("\n7. MANUAL KNOWLEDGE BASE USAGE");
    println!("{}", "-".repeat(40));

    // Create and add a manual experience
    let mut metrics = HashMap::new();
    metrics.insert("ic".to_string(), 0.08);
    metrics.insert("sharpe".to_string(), 1.5);

    let manual_exp = Experience::new(
        "golden_cross_lite".to_string(),
        "ts_mean(close, 10) / ts_mean(close, 50) - 1".to_string(),
        metrics,
        "bullish".to_string(),
        true,
    ).with_notes("Manual addition: works well in trending markets".to_string());

    let added = agent.kb.add(manual_exp);
    println!("\nManually added experience: {}", added);

    // Query knowledge base
    println!("\nQuerying for 'mean' factors:");
    let mean_factors = agent.kb.query(Some("mean"), None, false, 3);
    for exp in mean_factors {
        println!("  - {}: IC={:.4}", exp.factor_name, exp.metrics.get("ic").unwrap_or(&0.0));
    }

    // 8. Learning simulation
    println!("\n8. LEARNING SIMULATION");
    println!("{}", "-".repeat(40));

    println!("\nSimulating multiple mining sessions...");

    let mut sim_agent = QuantAgent::new().quality_threshold(30.0);
    let mut success_rates = Vec::new();

    for session in 0..3 {
        let results = sim_agent.mine(&data, 3, false)?;
        let success_rate = results.len() as f64 / 18.0; // 3 iterations * ~6 factors
        success_rates.push(success_rate);

        let summary = sim_agent.kb.summary();
        println!("  Session {}: Success rate={:.0}%, KB size={}",
                 session + 1,
                 success_rate * 100.0,
                 summary.total);
    }

    println!("\nLearning trend: {}",
             success_rates.iter()
                 .map(|r| format!("{:.0}%", r * 100.0))
                 .collect::<Vec<_>>()
                 .join(" -> "));

    // 9. Export knowledge base
    println!("\n9. KNOWLEDGE BASE EXPORT");
    println!("{}", "-".repeat(40));

    let kb_json = agent.kb.to_json();
    println!("\nKnowledge base exported ({} bytes)", kb_json.len());
    println!("Can be saved to file and reloaded later:");
    println!("  std::fs::write(\"kb.json\", kb_json)?;");
    println!("  let kb_data = std::fs::read_to_string(\"kb.json\")?;");

    // 10. Market conditions
    println!("\n10. MARKET CONDITION CLASSIFICATION");
    println!("{}", "-".repeat(40));

    // Test different seeds for different market conditions
    println!("\nClassifying different market scenarios:");
    for seed in [42, 123, 456, 789, 1000] {
        let test_data = generate_synthetic_data("TEST", 100, seed);
        let condition = MarketCondition::classify(&test_data);
        println!("  Seed {}: {}", seed, condition.as_str());
    }

    println!("\n====================================================");
    println!("Demo complete!");

    Ok(())
}
