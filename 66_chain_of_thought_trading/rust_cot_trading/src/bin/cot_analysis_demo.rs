//! Chain-of-Thought Analysis Demo
//!
//! Demonstrates basic CoT analysis for trading.

use cot_trading::{CoTAnalyzer, SignalGenerator};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("====================================================");
    println!("Chain-of-Thought Trading Analysis Demo");
    println!("====================================================\n");

    // Create mock analyzer (no API key needed)
    let analyzer = CoTAnalyzer::new_mock();

    // Example analysis query
    let query = r#"
    Analyze AAPL stock for trading:
    - Current price: $185.00
    - 1-day change: +1.2%
    - 5-day change: +3.5%
    - RSI (14): 58
    - MACD: 0.85 (above signal line)
    - Price above SMA20 ($182) and SMA50 ($178)
    - Volume: 1.3x average

    Should I buy, sell, or hold?
    "#;

    println!("Query:\n{}\n", query.trim());
    println!("Running Chain-of-Thought analysis...\n");

    let analysis = analyzer.analyze(query).await?;

    println!("Reasoning Chain:");
    println!("-----------------");
    for (i, step) in analysis.reasoning_steps.iter().enumerate() {
        println!("\nStep {}: {}", i + 1, step.step_name);
        println!("  Input: {}", step.input_data);
        println!("  Reasoning: {}", step.reasoning);
        println!("  Conclusion: {}", step.conclusion);
        println!("  Confidence: {:.0}%", step.confidence * 100.0);
    }

    println!("\n-----------------");
    println!("Final Answer: {}", analysis.final_answer);
    println!("Overall Confidence: {:.0}%", analysis.confidence * 100.0);
    println!("Processing Time: {}ms", analysis.processing_time_ms);

    // Self-consistency analysis
    println!("\n====================================================");
    println!("Self-Consistency Analysis (3 samples)");
    println!("====================================================\n");

    let consistent = analyzer.analyze_with_consistency(query, 3).await?;
    println!("Aggregated Confidence: {:.0}%", consistent.confidence * 100.0);

    // Signal generation
    println!("\n====================================================");
    println!("Signal Generation with Full Reasoning");
    println!("====================================================\n");

    let generator = SignalGenerator::new(CoTAnalyzer::new_mock());

    let signal = generator.generate(
        "AAPL",
        185.0,  // current price
        58.0,   // RSI
        0.85,   // MACD
        0.50,   // MACD signal
        182.0,  // SMA 20
        178.0,  // SMA 50
        1.3,    // volume ratio
        2.5,    // ATR
    ).await?;

    println!("Signal: {:?}", signal.signal_type);
    println!("Confidence: {:.0}%", signal.confidence * 100.0);
    println!("Stop Loss: ${:.2}", signal.stop_loss);
    println!("Take Profit: ${:.2}", signal.take_profit);

    println!("\nReasoning Chain:");
    for (i, reason) in signal.reasoning_chain.iter().enumerate() {
        println!("  {}. {}", i + 1, reason);
    }

    println!("\n====================================================");
    println!("Demo complete!");
    println!("====================================================");

    Ok(())
}
