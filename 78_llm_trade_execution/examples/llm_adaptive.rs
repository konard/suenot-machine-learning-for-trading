//! LLM-adaptive execution example.
//!
//! This example demonstrates how to use LLM-assisted execution
//! with the adaptive strategy as fallback.
//!
//! Note: Requires OpenAI or Anthropic API key for actual LLM usage.

use llm_trade_execution::{
    ExecutionConfig, ExecutionEngine, LlmAdapter, LlmConfig,
    ParentOrder, Side, AdaptiveStrategy, ImplementationShortfallStrategy,
    MarketImpactEstimator, AlmgrenChrissParams,
};
use llm_trade_execution::utils::MetricsRecorder;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("=== LLM-Adaptive Execution Example ===\n");

    // Check for API key
    let use_llm = std::env::var("OPENAI_API_KEY").is_ok()
        || std::env::var("ANTHROPIC_API_KEY").is_ok();

    if use_llm {
        println!("LLM API key found - will use LLM-assisted execution");
    } else {
        println!("No LLM API key found - using heuristic fallback");
        println!("Set OPENAI_API_KEY or ANTHROPIC_API_KEY to enable LLM execution");
    }
    println!();

    // Create configuration
    let config = ExecutionConfig {
        min_slice_interval_ms: 100,
        max_slice_interval_ms: 500,
        use_llm,
        llm_interval_ms: 1000,
        adaptive_sizing: true,
        verbose: true,
        ..Default::default()
    };

    // Create LLM adapter if available
    let llm_adapter = if let Ok(key) = std::env::var("OPENAI_API_KEY") {
        Some(LlmAdapter::openai(key)?)
    } else if let Ok(key) = std::env::var("ANTHROPIC_API_KEY") {
        Some(LlmAdapter::anthropic(key)?)
    } else {
        None
    };

    // Create impact estimator for crypto
    let impact_estimator = MarketImpactEstimator::with_params(
        AlmgrenChrissParams::crypto()
    );

    // Create execution engine
    let mut engine = ExecutionEngine::new(config.clone());

    if let Some(adapter) = llm_adapter {
        engine = engine.with_llm_adapter(adapter);
    }
    engine = engine.with_impact_estimator(impact_estimator);

    // Create metrics recorder
    let mut metrics = MetricsRecorder::new();

    // Example 1: Aggressive order execution
    println!("--- Example 1: Aggressive Buy Order ---");

    let order1 = ParentOrder::new(
        "BTCUSDT".to_string(),
        Side::Buy,
        1.0,    // 1 BTC
        10,     // 10 seconds (fast for demo)
    )
    .with_urgency(0.8)  // High urgency
    .with_max_participation(0.2);

    let strategy1 = Box::new(AdaptiveStrategy::new()
        .with_spread_threshold(15.0));

    let result1 = engine.execute(order1, strategy1).await?;

    println!("Result:");
    println!("  Filled: {:.4} / {:.4}", result1.filled_quantity, result1.total_quantity);
    println!("  Avg Price: {:.2}", result1.average_price);
    println!("  IS: {:.2} bps", result1.implementation_shortfall);
    println!("  Child Orders: {}", result1.child_order_count);
    println!("  LLM Decisions: {}", result1.llm_decisions.len());

    if !result1.llm_decisions.is_empty() {
        println!("\n  LLM Decision Sample:");
        for (i, decision) in result1.llm_decisions.iter().take(3).enumerate() {
            println!("    {}: {:?} (confidence: {:.2})",
                i + 1, decision.action, decision.confidence);
            println!("       {}", decision.reasoning);
        }
    }

    metrics.record(&result1);
    println!();

    // Example 2: Passive order execution
    println!("--- Example 2: Passive Sell Order ---");

    let mut engine2 = ExecutionEngine::new(config.clone());

    let order2 = ParentOrder::new(
        "ETHUSDT".to_string(),
        Side::Sell,
        5.0,    // 5 ETH
        15,     // 15 seconds
    )
    .with_urgency(0.2)  // Low urgency - be passive
    .with_max_participation(0.05);

    let strategy2 = Box::new(AdaptiveStrategy::new());

    let result2 = engine2.execute(order2, strategy2).await?;

    println!("Result:");
    println!("  Filled: {:.4} / {:.4}", result2.filled_quantity, result2.total_quantity);
    println!("  Avg Price: {:.2}", result2.average_price);
    println!("  IS: {:.2} bps", result2.implementation_shortfall);
    println!("  Child Orders: {}", result2.child_order_count);

    metrics.record(&result2);
    println!();

    // Example 3: Implementation Shortfall strategy
    println!("--- Example 3: Implementation Shortfall Strategy ---");

    let mut engine3 = ExecutionEngine::new(config);

    let order3 = ParentOrder::new(
        "BTCUSDT".to_string(),
        Side::Buy,
        0.5,
        10,
    ).with_urgency(0.5);

    let strategy3 = Box::new(ImplementationShortfallStrategy::with_params(
        AlmgrenChrissParams::crypto()
    ));

    let result3 = engine3.execute(order3, strategy3).await?;

    println!("Result:");
    println!("  Filled: {:.4} / {:.4}", result3.filled_quantity, result3.total_quantity);
    println!("  Avg Price: {:.2}", result3.average_price);
    println!("  IS: {:.2} bps", result3.implementation_shortfall);
    println!("  VWAP Slippage: {:.2} bps", result3.vwap_slippage);

    metrics.record(&result3);
    println!();

    // Print overall metrics
    println!("=== Overall Metrics ===");
    metrics.print_summary();

    // Example of optimal trajectory calculation
    println!("\n=== Optimal Execution Trajectory ===");
    let estimator = MarketImpactEstimator::crypto();
    let trajectory = estimator.optimal_trajectory(10.0, 10);

    println!("Trajectory for 10 BTC over 10 periods (Almgren-Chriss optimal):");
    for (i, qty) in trajectory.iter().enumerate() {
        let bar_len = (qty / 10.0 * 40.0) as usize;
        let bar: String = "█".repeat(bar_len);
        println!("  Period {:2}: {:6.3} BTC {}", i + 1, qty, bar);
    }

    println!("\nNote: Actual execution used simulated market data.");
    println!("For real trading, integrate with Bybit or other exchange APIs.");

    Ok(())
}
